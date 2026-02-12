"""
KV Cache Compression for Enigma AI Engine

Reduce memory usage during inference.

Features:
- Quantized KV cache
- Sliding window attention
- Token eviction strategies
- Memory-efficient storage
- Streaming support

Usage:
    from enigma_engine.core.kv_cache_compression import CompressedKVCache, CacheConfig
    
    cache = CompressedKVCache(
        max_length=4096,
        compression="quantize"
    )
    
    # Use during generation
    cache.update(key, value, layer_idx)
    k, v = cache.get(layer_idx)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """KV cache compression methods."""
    NONE = "none"
    QUANTIZE = "quantize"  # INT8/INT4 quantization
    SLIDING_WINDOW = "sliding_window"  # Fixed window
    EVICT_OLDEST = "evict_oldest"  # Remove oldest tokens
    EVICT_ATTENTION = "evict_attention"  # Remove low-attention tokens
    HYBRID = "hybrid"  # Combination of methods


@dataclass
class CacheConfig:
    """Configuration for KV cache compression."""
    max_length: int = 4096  # Maximum sequence length
    compression: CompressionType = CompressionType.NONE
    window_size: int = 1024  # For sliding window
    quantize_bits: int = 8  # 8 or 4
    eviction_ratio: float = 0.25  # Fraction to evict
    sink_tokens: int = 4  # Attention sinks to always keep
    local_tokens: int = 256  # Recent tokens to always keep


@dataclass
class CacheStats:
    """Statistics for cache usage."""
    memory_bytes: int = 0
    tokens_stored: int = 0
    tokens_evicted: int = 0
    compression_ratio: float = 1.0


class QuantizedTensor:
    """Quantized tensor storage."""
    
    def __init__(
        self,
        tensor: torch.Tensor,
        bits: int = 8,
        per_channel: bool = True
    ):
        """Quantize tensor to INT8 or INT4."""
        self.bits = bits
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device = tensor.device
        
        # Calculate scale and zero point per channel or per tensor
        if per_channel and len(tensor.shape) > 1:
            # Per-channel quantization (along last dim)
            dim = -1
            self.scale = tensor.abs().amax(dim=dim, keepdim=True) / (2 ** (bits - 1) - 1)
            self.scale = torch.where(self.scale == 0, torch.ones_like(self.scale), self.scale)
        else:
            # Per-tensor quantization
            self.scale = tensor.abs().max() / (2 ** (bits - 1) - 1)
            if self.scale == 0:
                self.scale = torch.tensor(1.0)
        
        # Quantize
        if bits == 8:
            self.data = (tensor / self.scale).round().clamp(-128, 127).to(torch.int8)
        elif bits == 4:
            # Pack 4-bit into uint8
            quantized = (tensor / self.scale).round().clamp(-8, 7)
            quantized = quantized.to(torch.int8) + 8  # Shift to 0-15
            # Pack two values per byte
            if tensor.numel() % 2 == 0:
                flat = quantized.view(-1)
                self.data = (flat[::2] << 4 | flat[1::2]).to(torch.uint8)
            else:
                # Pad for odd sizes
                flat = quantized.view(-1)
                padded = torch.zeros(flat.numel() + 1, dtype=torch.uint8, device=self.device)
                padded[:flat.numel()] = flat
                self.data = (padded[::2] << 4 | padded[1::2]).to(torch.uint8)
        else:
            raise ValueError(f"Unsupported quantization bits: {bits}")
    
    def dequantize(self) -> torch.Tensor:
        """Dequantize back to float."""
        if self.bits == 8:
            return self.data.to(self.dtype) * self.scale
        elif self.bits == 4:
            # Unpack 4-bit
            high = (self.data >> 4).to(torch.int8) - 8
            low = (self.data & 0x0F).to(torch.int8) - 8
            
            # Interleave
            numel = self.shape.numel()
            unpacked = torch.zeros(numel, dtype=torch.int8, device=self.device)
            unpacked[::2] = high[:numel // 2 + numel % 2]
            unpacked[1::2] = low[:numel // 2]
            
            return unpacked[:numel].view(self.shape).to(self.dtype) * self.scale
    
    def memory_size(self) -> int:
        """Get memory size in bytes."""
        if self.bits == 8:
            return self.data.numel()
        else:
            return self.data.numel()  # Already packed


class CompressedKVCache:
    """Memory-efficient KV cache with compression."""
    
    def __init__(
        self,
        num_layers: int = 12,
        config: Optional[CacheConfig] = None
    ):
        """
        Initialize compressed KV cache.
        
        Args:
            num_layers: Number of transformer layers
            config: Cache configuration
        """
        self.num_layers = num_layers
        self.config = config or CacheConfig()
        
        # Storage: layer_idx -> (keys, values)
        self._cache: Dict[int, Tuple[Any, Any]] = {}
        
        # Attention scores for eviction
        self._attention_scores: Dict[int, torch.Tensor] = {}
        
        # Statistics
        self._stats = CacheStats()
        
        logger.info(
            f"CompressedKVCache initialized: "
            f"compression={self.config.compression.value}, "
            f"max_length={self.config.max_length}"
        )
    
    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        attention_weights: Optional[torch.Tensor] = None
    ):
        """
        Update cache with new key-value pairs.
        
        Args:
            key: Key tensor [batch, heads, seq_len, head_dim]
            value: Value tensor [batch, heads, seq_len, head_dim]
            layer_idx: Layer index
            attention_weights: Optional attention weights for eviction
        """
        # Get existing cache
        if layer_idx in self._cache:
            old_k, old_v = self._get_raw(layer_idx)
            
            # Concatenate
            key = torch.cat([old_k, key], dim=2)
            value = torch.cat([old_v, value], dim=2)
        
        # Apply compression/eviction if exceeds max length
        if key.shape[2] > self.config.max_length:
            key, value = self._compress(key, value, layer_idx, attention_weights)
        
        # Store (optionally quantized)
        self._store(key, value, layer_idx)
        
        # Update attention scores if provided
        if attention_weights is not None:
            self._update_attention_scores(attention_weights, layer_idx)
        
        # Update stats
        self._update_stats()
    
    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get key-value pairs for a layer.
        
        Returns:
            Tuple of (keys, values)
        """
        return self._get_raw(layer_idx)
    
    def _get_raw(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get raw tensors (dequantize if needed)."""
        if layer_idx not in self._cache:
            raise KeyError(f"Layer {layer_idx} not in cache")
        
        key, value = self._cache[layer_idx]
        
        if isinstance(key, QuantizedTensor):
            key = key.dequantize()
        if isinstance(value, QuantizedTensor):
            value = value.dequantize()
        
        return key, value
    
    def _store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int
    ):
        """Store tensors (with optional quantization)."""
        if self.config.compression == CompressionType.QUANTIZE:
            key = QuantizedTensor(key, bits=self.config.quantize_bits)
            value = QuantizedTensor(value, bits=self.config.quantize_bits)
        
        self._cache[layer_idx] = (key, value)
    
    def _compress(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        attention_weights: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply compression/eviction strategy."""
        compression = self.config.compression
        
        if compression == CompressionType.SLIDING_WINDOW:
            return self._sliding_window(key, value)
        elif compression == CompressionType.EVICT_OLDEST:
            return self._evict_oldest(key, value)
        elif compression == CompressionType.EVICT_ATTENTION:
            return self._evict_attention(key, value, layer_idx)
        elif compression == CompressionType.HYBRID:
            return self._hybrid_compression(key, value, layer_idx, attention_weights)
        else:
            # NONE or QUANTIZE - just truncate
            return key[:, :, -self.config.max_length:, :], value[:, :, -self.config.max_length:, :]
    
    def _sliding_window(
        self,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply sliding window attention."""
        window = self.config.window_size
        sink = self.config.sink_tokens
        
        seq_len = key.shape[2]
        
        if seq_len <= window + sink:
            return key, value
        
        # Keep sink tokens + window
        keep_indices = list(range(sink)) + list(range(seq_len - window, seq_len))
        
        key = key[:, :, keep_indices, :]
        value = value[:, :, keep_indices, :]
        
        self._stats.tokens_evicted += seq_len - len(keep_indices)
        
        return key, value
    
    def _evict_oldest(
        self,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evict oldest tokens."""
        seq_len = key.shape[2]
        target_len = self.config.max_length
        
        sink = self.config.sink_tokens
        local = self.config.local_tokens
        
        if seq_len <= target_len:
            return key, value
        
        # Keep sink tokens + recent local tokens + some middle
        evict_count = seq_len - target_len
        
        # Keep first 'sink' and last 'local'
        middle_start = sink
        middle_end = seq_len - local
        
        if middle_end > middle_start + evict_count:
            # Evict from middle
            keep_middle = middle_end - middle_start - evict_count
            keep_indices = (
                list(range(sink)) +
                list(range(sink, sink + keep_middle)) +
                list(range(seq_len - local, seq_len))
            )
        else:
            # Not enough middle, keep what we can
            keep_indices = list(range(sink)) + list(range(seq_len - local, seq_len))
        
        key = key[:, :, keep_indices, :]
        value = value[:, :, keep_indices, :]
        
        self._stats.tokens_evicted += seq_len - len(keep_indices)
        
        return key, value
    
    def _evict_attention(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evict tokens with lowest attention scores."""
        seq_len = key.shape[2]
        target_len = self.config.max_length
        
        if seq_len <= target_len:
            return key, value
        
        sink = self.config.sink_tokens
        local = self.config.local_tokens
        
        # Get attention scores
        if layer_idx in self._attention_scores:
            scores = self._attention_scores[layer_idx]
            
            # Average across batch and heads
            if len(scores.shape) > 1:
                scores = scores.mean(dim=tuple(range(len(scores.shape) - 1)))
            
            # Protect sink and local tokens
            scores[:sink] = float('inf')
            scores[-local:] = float('inf')
            
            # Keep top-k by attention
            k = target_len
            _, keep_indices = scores.topk(k, largest=True)
            keep_indices = keep_indices.sort().values
        else:
            # Fallback to oldest eviction
            return self._evict_oldest(key, value)
        
        key = key[:, :, keep_indices, :]
        value = value[:, :, keep_indices, :]
        
        self._stats.tokens_evicted += seq_len - len(keep_indices)
        
        return key, value
    
    def _hybrid_compression(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        attention_weights: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply hybrid compression strategy."""
        # First apply sliding window
        key, value = self._sliding_window(key, value)
        
        # Then attention-based eviction if still too long
        if key.shape[2] > self.config.max_length:
            key, value = self._evict_attention(key, value, layer_idx)
        
        return key, value
    
    def _update_attention_scores(
        self,
        attention_weights: torch.Tensor,
        layer_idx: int
    ):
        """Update cumulative attention scores."""
        # Sum attention received by each token
        # attention_weights: [batch, heads, query_len, key_len]
        scores = attention_weights.sum(dim=2)  # Sum over queries
        scores = scores.mean(dim=(0, 1))  # Average over batch and heads
        
        if layer_idx in self._attention_scores:
            old_scores = self._attention_scores[layer_idx]
            # Pad to match new length
            if old_scores.shape[0] < scores.shape[0]:
                pad = torch.zeros(
                    scores.shape[0] - old_scores.shape[0],
                    device=old_scores.device
                )
                old_scores = torch.cat([old_scores, pad])
            scores = 0.9 * old_scores[:scores.shape[0]] + 0.1 * scores
        
        self._attention_scores[layer_idx] = scores
    
    def _update_stats(self) -> None:
        """Update cache statistics."""
        total_bytes = 0
        total_tokens = 0
        
        for layer_idx, (key, value) in self._cache.items():
            if isinstance(key, QuantizedTensor):
                total_bytes += key.memory_size() + value.memory_size()
                total_tokens += key.shape.numel() // key.shape[-1]
            else:
                total_bytes += key.numel() * key.element_size()
                total_bytes += value.numel() * value.element_size()
                total_tokens += key.shape[2]
        
        self._stats.memory_bytes = total_bytes
        self._stats.tokens_stored = total_tokens
        
        # Calculate compression ratio
        uncompressed = total_tokens * 2 * 4  # 2 for k,v, 4 bytes for float32
        if uncompressed > 0:
            self._stats.compression_ratio = total_bytes / uncompressed
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._attention_scores.clear()
        self._stats = CacheStats()
    
    def clear_layer(self, layer_idx: int) -> None:
        """Clear cache for a specific layer."""
        if layer_idx in self._cache:
            del self._cache[layer_idx]
        if layer_idx in self._attention_scores:
            del self._attention_scores[layer_idx]


class StreamingKVCache(CompressedKVCache):
    """KV cache optimized for streaming generation."""
    
    def __init__(
        self,
        num_layers: int = 12,
        config: Optional[CacheConfig] = None,
        chunk_size: int = 16
    ):
        """
        Initialize streaming cache.
        
        Args:
            chunk_size: Process tokens in chunks of this size
        """
        super().__init__(num_layers, config)
        self.chunk_size = chunk_size
        self._buffer: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
    
    def stream_update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int
    ):
        """
        Update cache in streaming fashion.
        
        Buffers updates and flushes when chunk is complete.
        """
        if layer_idx not in self._buffer:
            self._buffer[layer_idx] = []
        
        self._buffer[layer_idx].append((key, value))
        
        # Check if chunk is complete
        total_tokens = sum(k.shape[2] for k, _ in self._buffer[layer_idx])
        
        if total_tokens >= self.chunk_size:
            self._flush_buffer(layer_idx)
    
    def _flush_buffer(self, layer_idx: int) -> None:
        """Flush buffer to main cache."""
        if layer_idx not in self._buffer or not self._buffer[layer_idx]:
            return
        
        # Concatenate buffered tensors
        keys = torch.cat([k for k, _ in self._buffer[layer_idx]], dim=2)
        values = torch.cat([v for _, v in self._buffer[layer_idx]], dim=2)
        
        # Update main cache
        self.update(keys, values, layer_idx)
        
        # Clear buffer
        self._buffer[layer_idx] = []
    
    def flush_all(self) -> None:
        """Flush all buffers."""
        for layer_idx in list(self._buffer.keys()):
            self._flush_buffer(layer_idx)
