"""
KV Cache Compression for Enigma AI Engine

Reduce memory usage for long conversations.

Features:
- Eviction policies (LRU, attention-based)
- Quantization of cached values
- Sliding window cache
- Importance scoring
- Streaming cache management

Usage:
    from enigma_engine.core.kv_compression import KVCacheManager, get_cache_manager
    
    cache_manager = get_cache_manager()
    
    # Apply compression to model
    cache_manager.apply_to_model(model)
"""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    ATTENTION = "attention"  # Based on attention scores
    SLIDING = "sliding"  # Keep recent tokens only
    IMPORTANCE = "importance"  # Based on computed importance
    H2O = "h2o"  # Heavy Hitter Oracle


class CacheQuantization(Enum):
    """Quantization levels for cache."""
    NONE = "none"  # No quantization (FP32/FP16)
    INT8 = "int8"  # 8-bit quantization
    INT4 = "int4"  # 4-bit quantization


@dataclass
class CompressionConfig:
    """Configuration for KV cache compression."""
    # Cache limits
    max_cache_size: int = 4096  # Max tokens to cache
    
    # Eviction
    eviction_policy: EvictionPolicy = EvictionPolicy.H2O
    eviction_ratio: float = 0.2  # Fraction to evict when full
    
    # Quantization
    quantization: CacheQuantization = CacheQuantization.NONE
    
    # Sliding window
    window_size: int = 2048
    
    # H2O specific
    heavy_hitter_tokens: int = 256  # Most attended tokens to keep
    recent_tokens: int = 512  # Recent tokens to always keep
    
    # Memory
    offload_to_cpu: bool = False
    
    # Debugging
    track_statistics: bool = True


@dataclass
class CacheStatistics:
    """Statistics for cache compression."""
    total_tokens_seen: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    memory_saved_mb: float = 0.0
    
    compression_ratio: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens_seen,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "evictions": self.evictions,
            "memory_saved_mb": round(self.memory_saved_mb, 2),
            "compression_ratio": round(self.compression_ratio, 2)
        }


class CacheEntry:
    """Entry in the KV cache."""
    
    def __init__(
        self,
        position: int,
        key: Any,
        value: Any,
        attention_score: float = 0.0
    ):
        self.position = position
        self.key = key
        self.value = value
        self.attention_score = attention_score
        self.access_count = 0
        self.last_access = time.time()
        self.importance = 0.0
    
    def access(self, attention_score: Optional[float] = None):
        """Record cache access."""
        self.access_count += 1
        self.last_access = time.time()
        if attention_score is not None:
            # Exponential moving average
            self.attention_score = 0.9 * self.attention_score + 0.1 * attention_score


class CompressedKVCache:
    """Compressed KV cache with eviction."""
    
    def __init__(
        self,
        layer_id: int,
        num_heads: int,
        head_dim: int,
        config: CompressionConfig
    ):
        """
        Initialize compressed cache.
        
        Args:
            layer_id: Layer index
            num_heads: Number of attention heads
            head_dim: Head dimension
            config: Compression configuration
        """
        self._layer_id = layer_id
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._config = config
        
        # Cache storage
        self._entries: OrderedDict[int, CacheEntry] = OrderedDict()
        
        # Statistics
        self._stats = CacheStatistics()
        
        # For attention-based eviction
        self._attention_scores: Dict[int, float] = {}
    
    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._entries)
    
    @property
    def stats(self) -> CacheStatistics:
        """Get cache statistics."""
        return self._stats
    
    def _quantize_tensor(self, tensor: Any) -> Tuple[Any, Any]:
        """Quantize tensor to save memory."""
        import torch
        
        if self._config.quantization == CacheQuantization.NONE:
            return tensor, None
        
        elif self._config.quantization == CacheQuantization.INT8:
            # Scale to int8 range
            scale = tensor.abs().max() / 127
            quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
            return quantized, scale
        
        elif self._config.quantization == CacheQuantization.INT4:
            # Pack two int4 values into one int8
            scale = tensor.abs().max() / 7
            quantized = (tensor / scale).round().clamp(-8, 7).to(torch.int8)
            return quantized, scale
        
        return tensor, None
    
    def _dequantize_tensor(
        self,
        quantized: Any,
        scale: Any
    ) -> Any:
        """Dequantize tensor."""
        import torch
        
        if scale is None:
            return quantized
        
        return quantized.float() * scale
    
    def put(
        self,
        position: int,
        key: Any,
        value: Any,
        attention_score: float = 0.0
    ):
        """
        Add entry to cache.
        
        Args:
            position: Token position
            key: Key tensor
            value: Value tensor
            attention_score: Attention score for importance
        """
        self._stats.total_tokens_seen += 1
        
        # Quantize if configured
        if self._config.quantization != CacheQuantization.NONE:
            key, key_scale = self._quantize_tensor(key)
            value, value_scale = self._quantize_tensor(value)
            key = (key, key_scale)
            value = (value, value_scale)
        
        # Offload to CPU if configured
        if self._config.offload_to_cpu:
            import torch
            if isinstance(key, tuple):
                key = (key[0].cpu(), key[1].cpu() if key[1] is not None else None)
                value = (value[0].cpu(), value[1].cpu() if value[1] is not None else None)
            else:
                key = key.cpu()
                value = value.cpu()
        
        entry = CacheEntry(position, key, value, attention_score)
        
        # Check if eviction needed
        if self.size >= self._config.max_cache_size:
            self._evict()
        
        self._entries[position] = entry
        self._attention_scores[position] = attention_score
    
    def get(
        self,
        position: int,
        attention_score: Optional[float] = None
    ) -> Optional[Tuple[Any, Any]]:
        """
        Get entry from cache.
        
        Args:
            position: Token position
            attention_score: Current attention score
            
        Returns:
            Tuple of (key, value) or None
        """
        if position not in self._entries:
            self._stats.cache_misses += 1
            return None
        
        self._stats.cache_hits += 1
        entry = self._entries[position]
        entry.access(attention_score)
        
        # Move to end for LRU
        self._entries.move_to_end(position)
        
        # Dequantize if needed
        key, value = entry.key, entry.value
        
        if isinstance(key, tuple):
            key = self._dequantize_tensor(key[0], key[1])
            value = self._dequantize_tensor(value[0], value[1])
        
        return key, value
    
    def get_all(self) -> Tuple[Any, Any]:
        """
        Get all cached keys and values.
        
        Returns:
            Tuple of (keys tensor, values tensor)
        """
        import torch
        
        if not self._entries:
            return None, None
        
        keys = []
        values = []
        
        for entry in self._entries.values():
            key, value = entry.key, entry.value
            
            if isinstance(key, tuple):
                key = self._dequantize_tensor(key[0], key[1])
                value = self._dequantize_tensor(value[0], value[1])
            
            keys.append(key)
            values.append(value)
        
        # Stack into tensors
        keys = torch.stack(keys, dim=0) if keys else None
        values = torch.stack(values, dim=0) if values else None
        
        return keys, values
    
    def _evict(self):
        """Evict entries based on policy."""
        num_to_evict = int(self.size * self._config.eviction_ratio)
        num_to_evict = max(1, num_to_evict)
        
        policy = self._config.eviction_policy
        
        if policy == EvictionPolicy.LRU:
            self._evict_lru(num_to_evict)
        elif policy == EvictionPolicy.LFU:
            self._evict_lfu(num_to_evict)
        elif policy == EvictionPolicy.ATTENTION:
            self._evict_attention(num_to_evict)
        elif policy == EvictionPolicy.SLIDING:
            self._evict_sliding()
        elif policy == EvictionPolicy.IMPORTANCE:
            self._evict_importance(num_to_evict)
        elif policy == EvictionPolicy.H2O:
            self._evict_h2o()
        
        self._stats.evictions += num_to_evict
    
    def _evict_lru(self, count: int):
        """Evict least recently used."""
        for _ in range(min(count, len(self._entries))):
            # First item is least recently used
            oldest = next(iter(self._entries))
            del self._entries[oldest]
            self._attention_scores.pop(oldest, None)
    
    def _evict_lfu(self, count: int):
        """Evict least frequently used."""
        # Sort by access count
        sorted_entries = sorted(
            self._entries.items(),
            key=lambda x: x[1].access_count
        )
        
        for pos, _ in sorted_entries[:count]:
            del self._entries[pos]
            self._attention_scores.pop(pos, None)
    
    def _evict_attention(self, count: int):
        """Evict lowest attention score."""
        # Sort by attention score
        sorted_entries = sorted(
            self._entries.items(),
            key=lambda x: x[1].attention_score
        )
        
        for pos, _ in sorted_entries[:count]:
            del self._entries[pos]
            self._attention_scores.pop(pos, None)
    
    def _evict_sliding(self):
        """Keep only recent window."""
        window = self._config.window_size
        
        if self.size <= window:
            return
        
        # Sort by position
        sorted_pos = sorted(self._entries.keys())
        
        # Remove oldest beyond window
        for pos in sorted_pos[:-window]:
            del self._entries[pos]
            self._attention_scores.pop(pos, None)
    
    def _evict_importance(self, count: int):
        """Evict by computed importance score."""
        for entry in self._entries.values():
            # Combine factors for importance
            entry.importance = (
                entry.attention_score * 0.5 +
                (entry.access_count / max(1, self._stats.total_tokens_seen)) * 0.3 +
                (1.0 / (time.time() - entry.last_access + 1)) * 0.2
            )
        
        sorted_entries = sorted(
            self._entries.items(),
            key=lambda x: x[1].importance
        )
        
        for pos, _ in sorted_entries[:count]:
            del self._entries[pos]
            self._attention_scores.pop(pos, None)
    
    def _evict_h2o(self):
        """
        Heavy Hitter Oracle eviction.
        
        Keep heavy hitters (high attention) + recent tokens.
        """
        heavy_count = self._config.heavy_hitter_tokens
        recent_count = self._config.recent_tokens
        
        # Get sorted positions
        sorted_pos = sorted(self._entries.keys())
        
        # Always keep recent tokens
        recent_positions = set(sorted_pos[-recent_count:]) if sorted_pos else set()
        
        # Identify heavy hitters by attention
        sorted_by_attention = sorted(
            [(pos, self._attention_scores.get(pos, 0)) for pos in sorted_pos],
            key=lambda x: -x[1]
        )
        heavy_positions = set(pos for pos, _ in sorted_by_attention[:heavy_count])
        
        # Keep union of recent and heavy
        keep_positions = recent_positions | heavy_positions
        
        # Remove everything else
        to_remove = [pos for pos in self._entries if pos not in keep_positions]
        
        for pos in to_remove:
            del self._entries[pos]
            self._attention_scores.pop(pos, None)
    
    def clear(self):
        """Clear the cache."""
        self._entries.clear()
        self._attention_scores.clear()


class KVCacheManager:
    """Manager for compressed KV caches across layers."""
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """
        Initialize cache manager.
        
        Args:
            config: Compression configuration
        """
        self._config = config or CompressionConfig()
        self._caches: Dict[int, CompressedKVCache] = {}
        self._stats = CacheStatistics()
    
    def get_cache(
        self,
        layer_id: int,
        num_heads: int,
        head_dim: int
    ) -> CompressedKVCache:
        """
        Get or create cache for layer.
        
        Args:
            layer_id: Layer index
            num_heads: Number of attention heads
            head_dim: Head dimension
            
        Returns:
            CompressedKVCache for the layer
        """
        if layer_id not in self._caches:
            self._caches[layer_id] = CompressedKVCache(
                layer_id, num_heads, head_dim, self._config
            )
        return self._caches[layer_id]
    
    def apply_to_model(self, model: Any) -> bool:
        """
        Apply compression to model.
        
        Args:
            model: Model to modify
            
        Returns:
            True if successful
        """
        try:
            # Store reference to manager
            model._kv_cache_manager = self
            
            # Hook into attention layers
            for name, module in model.named_modules():
                if 'attention' in name.lower() or 'attn' in name.lower():
                    if hasattr(module, 'layer_idx'):
                        layer_id = module.layer_idx
                    else:
                        # Extract layer ID from name
                        import re
                        match = re.search(r'layers?\.?(\d+)', name)
                        layer_id = int(match.group(1)) if match else 0
                    
                    # Set up compressed cache
                    num_heads = getattr(module, 'num_heads', 8)
                    head_dim = getattr(module, 'head_dim', 64)
                    
                    cache = self.get_cache(layer_id, num_heads, head_dim)
                    module._compressed_kv_cache = cache
                    
                    logger.debug(f"Applied compression to {name}")
            
            logger.info(f"Applied KV cache compression to {len(self._caches)} layers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply compression: {e}")
            return False
    
    def clear_all(self):
        """Clear all caches."""
        for cache in self._caches.values():
            cache.clear()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        total_entries = sum(cache.size for cache in self._caches.values())
        
        return {
            "num_layers": len(self._caches),
            "total_entries": total_entries,
            "config": {
                "max_cache_size": self._config.max_cache_size,
                "eviction_policy": self._config.eviction_policy.value,
                "quantization": self._config.quantization.value
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        stats = {
            "layers": len(self._caches),
            "per_layer": {}
        }
        
        for layer_id, cache in self._caches.items():
            stats["per_layer"][layer_id] = cache.stats.to_dict()
        
        return stats


# Global instance
_cache_manager: Optional[KVCacheManager] = None


def get_cache_manager(
    config: Optional[CompressionConfig] = None
) -> KVCacheManager:
    """Get or create global cache manager."""
    global _cache_manager
    if _cache_manager is None or config is not None:
        _cache_manager = KVCacheManager(config)
    return _cache_manager
