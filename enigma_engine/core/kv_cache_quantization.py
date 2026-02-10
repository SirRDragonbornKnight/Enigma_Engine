"""
KV Cache Quantization for Enigma AI Engine.

Provides INT8 quantization for KV cache to reduce memory usage
during inference while maintaining quality.
"""
import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for KV cache quantization."""
    enabled: bool = True
    dtype: str = "int8"  # int8, int4, fp16
    symmetric: bool = True  # Symmetric vs asymmetric quantization
    per_channel: bool = True  # Per-channel vs per-tensor quantization
    group_size: int = 128  # Group size for grouped quantization
    calibration_samples: int = 100  # Samples for calibration
    dynamic: bool = True  # Dynamic vs static quantization


class QuantizedTensor:
    """
    A quantized tensor with scale and zero point.
    
    Stores the quantized values and parameters needed for dequantization.
    """
    
    def __init__(
        self,
        quantized_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.int8
    ):
        self.data = quantized_data
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype
        self.original_shape = quantized_data.shape
    
    def dequantize(self) -> torch.Tensor:
        """Convert back to floating point."""
        result = self.data.float() * self.scale
        if self.zero_point is not None:
            result = result - self.zero_point * self.scale
        return result
    
    def numel(self) -> int:
        """Number of elements."""
        return self.data.numel()
    
    def memory_savings(self, original_dtype: torch.dtype = torch.float32) -> float:
        """Calculate memory savings ratio."""
        original_bits = torch.finfo(original_dtype).bits if original_dtype.is_floating_point else 32
        quantized_bits = 8 if self.dtype == torch.int8 else 4
        return 1.0 - (quantized_bits / original_bits)


def quantize_tensor(
    tensor: torch.Tensor,
    config: QuantizationConfig
) -> QuantizedTensor:
    """
    Quantize a tensor to INT8.
    
    Args:
        tensor: Input tensor (float32 or float16)
        config: Quantization configuration
        
    Returns:
        QuantizedTensor with quantized data and scale
    """
    if not config.enabled:
        # Return a "quantized" tensor that's actually just the original
        return QuantizedTensor(
            tensor,
            torch.ones(1, device=tensor.device),
            None,
            tensor.dtype
        )
    
    # Select quantization dtype
    if config.dtype == "int8":
        qmin, qmax = -128, 127
        target_dtype = torch.int8
    elif config.dtype == "int4":
        qmin, qmax = -8, 7
        target_dtype = torch.int8  # Store as int8, use only 4 bits
    else:  # fp16
        qmin, qmax = None, None
        target_dtype = torch.float16
    
    if config.dtype == "fp16":
        # Simple dtype conversion for fp16
        return QuantizedTensor(
            tensor.half(),
            torch.ones(1, device=tensor.device),
            None,
            torch.float16
        )
    
    # Calculate scale and zero point
    if config.symmetric:
        # Symmetric quantization: scale = max(abs) / qmax
        if config.per_channel and len(tensor.shape) > 1:
            # Per-channel: compute scale along last dimension
            abs_max = tensor.abs().amax(dim=-1, keepdim=True)
            scale = abs_max / qmax
        else:
            # Per-tensor
            abs_max = tensor.abs().max()
            scale = abs_max / qmax
        
        # Avoid division by zero
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        
        # Quantize
        quantized = torch.clamp(
            torch.round(tensor / scale),
            qmin, qmax
        ).to(target_dtype)
        
        zero_point = None
    else:
        # Asymmetric quantization
        if config.per_channel and len(tensor.shape) > 1:
            min_val = tensor.amin(dim=-1, keepdim=True)
            max_val = tensor.amax(dim=-1, keepdim=True)
        else:
            min_val = tensor.min()
            max_val = tensor.max()
        
        scale = (max_val - min_val) / (qmax - qmin)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        
        zero_point = qmin - torch.round(min_val / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax)
        
        quantized = torch.clamp(
            torch.round(tensor / scale + zero_point),
            qmin, qmax
        ).to(target_dtype)
    
    return QuantizedTensor(quantized, scale, zero_point, target_dtype)


def dequantize_tensor(qtensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize a QuantizedTensor back to float."""
    return qtensor.dequantize()


class QuantizedKVCache:
    """
    Quantized Key-Value cache for transformer attention.
    
    Stores keys and values in INT8 format to reduce memory usage
    by ~4x compared to FP32 or ~2x compared to FP16.
    
    Usage:
        cache = QuantizedKVCache(config)
        
        # During generation
        cache.update(layer_idx, key, value)
        cached_k, cached_v = cache.get(layer_idx)
    """
    
    def __init__(
        self,
        config: QuantizationConfig = None,
        max_batch_size: int = 1,
        max_seq_len: int = 2048,
        num_layers: int = 12,
        num_heads: int = 8,
        head_dim: int = 64
    ):
        """
        Initialize quantized KV cache.
        
        Args:
            config: Quantization configuration
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
        """
        self.config = config or QuantizationConfig()
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Cache storage: layer_idx -> (quantized_k, quantized_v)
        self._cache: dict[int, tuple[Optional[QuantizedTensor], Optional[QuantizedTensor]]] = {}
        
        # Current sequence length
        self.seq_len = 0
        
        # Statistics
        self.total_quantized = 0
        self.total_dequantized = 0
    
    def reset(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self.seq_len = 0
    
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value pairs.
        
        Args:
            layer_idx: Transformer layer index
            key: New key tensor [batch, heads, seq, head_dim]
            value: New value tensor [batch, heads, seq, head_dim]
            
        Returns:
            Tuple of (full_key, full_value) including cached values
        """
        new_seq_len = key.shape[2]
        
        if layer_idx in self._cache:
            # Dequantize existing cache
            cached_k, cached_v = self._cache[layer_idx]
            
            if cached_k is not None:
                decoded_k = cached_k.dequantize()
                decoded_v = cached_v.dequantize()
                
                # Concatenate with new
                full_k = torch.cat([decoded_k, key], dim=2)
                full_v = torch.cat([decoded_v, value], dim=2)
                
                self.total_dequantized += 2
            else:
                full_k = key
                full_v = value
        else:
            full_k = key
            full_v = value
        
        # Quantize and store
        self._cache[layer_idx] = (
            quantize_tensor(full_k, self.config),
            quantize_tensor(full_v, self.config)
        )
        
        self.seq_len = full_k.shape[2]
        self.total_quantized += 2
        
        return full_k, full_v
    
    def get(
        self,
        layer_idx: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get cached key-value pairs.
        
        Args:
            layer_idx: Transformer layer index
            
        Returns:
            Tuple of (key, value) tensors or (None, None) if not cached
        """
        if layer_idx not in self._cache:
            return None, None
        
        cached_k, cached_v = self._cache[layer_idx]
        
        if cached_k is None:
            return None, None
        
        self.total_dequantized += 2
        return cached_k.dequantize(), cached_v.dequantize()
    
    def memory_usage(self) -> dict[str, Any]:
        """Calculate current memory usage."""
        total_bytes = 0
        total_elements = 0
        
        for layer_idx, (qk, qv) in self._cache.items():
            if qk is not None:
                total_elements += qk.numel() + qv.numel()
                
                # Calculate bytes based on dtype
                if qk.dtype == torch.int8:
                    bytes_per_elem = 1
                elif qk.dtype == torch.float16:
                    bytes_per_elem = 2
                else:
                    bytes_per_elem = 4
                
                total_bytes += (qk.numel() + qv.numel()) * bytes_per_elem
                
                # Add scale storage
                total_bytes += qk.scale.numel() * 4  # float32 scale
                total_bytes += qv.scale.numel() * 4
        
        # Calculate theoretical FP32 usage
        fp32_bytes = total_elements * 4
        
        return {
            "quantized_bytes": total_bytes,
            "fp32_equivalent_bytes": fp32_bytes,
            "savings_ratio": 1.0 - (total_bytes / fp32_bytes) if fp32_bytes > 0 else 0,
            "num_layers_cached": len(self._cache),
            "sequence_length": self.seq_len,
            "quantizations": self.total_quantized,
            "dequantizations": self.total_dequantized,
        }


class QuantizedAttention(nn.Module):
    """
    Attention layer with quantized KV cache support.
    
    Drop-in replacement for standard attention that uses
    quantized KV caching for memory efficiency.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        config: QuantizationConfig = None
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.dropout = dropout
        self.config = config or QuantizationConfig()
        
        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[QuantizedKVCache] = None,
        layer_idx: int = 0,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional quantized KV caching.
        
        Args:
            x: Input tensor [batch, seq, dim]
            kv_cache: Optional quantized KV cache
            layer_idx: Layer index for caching
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use KV cache if provided
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)
        
        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)
        
        return out


def create_quantized_kv_cache(
    model_config: dict[str, Any],
    quant_config: Optional[QuantizationConfig] = None
) -> QuantizedKVCache:
    """
    Create a quantized KV cache for a model.
    
    Args:
        model_config: Model configuration with num_layers, num_heads, head_dim
        quant_config: Optional quantization configuration
        
    Returns:
        QuantizedKVCache instance
    """
    return QuantizedKVCache(
        config=quant_config or QuantizationConfig(),
        num_layers=model_config.get("num_layers", 12),
        num_heads=model_config.get("num_heads", 8),
        head_dim=model_config.get("head_dim", 64),
        max_seq_len=model_config.get("max_seq_len", 2048),
        max_batch_size=model_config.get("max_batch_size", 1),
    )


# Utility functions
def estimate_kv_cache_size(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: str = "fp32"
) -> int:
    """
    Estimate KV cache memory usage in bytes.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dtype: Data type (fp32, fp16, int8)
        
    Returns:
        Estimated memory usage in bytes
    """
    elements = 2 * batch_size * seq_len * num_layers * num_heads * head_dim
    
    bytes_per_elem = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }.get(dtype, 4)
    
    return int(elements * bytes_per_elem)


def get_recommended_config(
    available_memory_gb: float,
    model_size: str = "small"
) -> QuantizationConfig:
    """
    Get recommended quantization config based on available memory.
    
    Args:
        available_memory_gb: Available GPU memory in GB
        model_size: Model size preset (tiny, small, medium, large)
        
    Returns:
        Recommended QuantizationConfig
    """
    if available_memory_gb < 4:
        # Very limited memory - aggressive quantization
        return QuantizationConfig(
            enabled=True,
            dtype="int8",
            symmetric=True,
            per_channel=False,  # Simpler, uses less metadata
            dynamic=True,
        )
    elif available_memory_gb < 8:
        # Moderate memory
        return QuantizationConfig(
            enabled=True,
            dtype="int8",
            symmetric=True,
            per_channel=True,
            dynamic=True,
        )
    elif available_memory_gb < 16:
        # Good memory - use fp16
        return QuantizationConfig(
            enabled=True,
            dtype="fp16",
            dynamic=True,
        )
    else:
        # Plenty of memory - optional quantization
        return QuantizationConfig(
            enabled=False,
        )
