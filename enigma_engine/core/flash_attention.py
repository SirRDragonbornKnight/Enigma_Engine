"""
Flash Attention Implementation

Optimized attention for faster training and inference.
Supports Flash Attention 1, 2, and 3 patterns.

FILE: enigma_engine/core/flash_attention.py
TYPE: Core/Attention
MAIN CLASSES: FlashAttention, FlashAttention2, FlashAttention3
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Check for flash-attn library
try:
    import flash_attn
    from flash_attn import flash_attn_func
    from flash_attn.flash_attn_interface import flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
    FLASH_ATTN_VERSION = getattr(flash_attn, '__version__', '2.0')
    logger.info(f"Flash Attention available (v{FLASH_ATTN_VERSION})")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    FLASH_ATTN_VERSION = None
    logger.warning("flash-attn not installed - using PyTorch attention")


class AttentionBackend(Enum):
    """Available attention backends."""
    PYTORCH_NATIVE = "pytorch_native"
    PYTORCH_SDPA = "pytorch_sdpa"  # Scaled Dot Product Attention
    FLASH_ATTN_1 = "flash_attn_1"
    FLASH_ATTN_2 = "flash_attn_2"
    FLASH_ATTN_3 = "flash_attn_3"
    MEMORY_EFFICIENT = "memory_efficient"


@dataclass
class AttentionConfig:
    """Configuration for attention computation."""
    head_dim: int = 64
    num_heads: int = 8
    num_kv_heads: int = 8  # For GQA
    dropout: float = 0.0
    causal: bool = True
    window_size: tuple[int, int] = (-1, -1)  # For sliding window attention
    softmax_scale: Optional[float] = None
    deterministic: bool = False


def get_best_backend() -> AttentionBackend:
    """Get the best available attention backend."""
    if FLASH_ATTN_AVAILABLE:
        if FLASH_ATTN_VERSION and FLASH_ATTN_VERSION.startswith('3'):
            return AttentionBackend.FLASH_ATTN_3
        elif FLASH_ATTN_VERSION and FLASH_ATTN_VERSION.startswith('2'):
            return AttentionBackend.FLASH_ATTN_2
        return AttentionBackend.FLASH_ATTN_1
    
    # Check for PyTorch 2.0+ SDPA
    if hasattr(F, 'scaled_dot_product_attention'):
        return AttentionBackend.PYTORCH_SDPA
    
    return AttentionBackend.PYTORCH_NATIVE


class AttentionBase(nn.Module):
    """Base class for attention implementations."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.scale = config.softmax_scale or (1.0 / math.sqrt(config.head_dim))
    
    def forward(
        self,
        q: torch.Tensor,  # [batch, seq_len, num_heads, head_dim]
        k: torch.Tensor,  # [batch, seq_len, num_kv_heads, head_dim]
        v: torch.Tensor,  # [batch, seq_len, num_kv_heads, head_dim]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class PyTorchNativeAttention(AttentionBase):
    """Standard PyTorch attention implementation."""
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Expand KV heads for GQA
        num_kv_heads = k.shape[2]
        if num_kv_heads != num_heads:
            kv_repeat = num_heads // num_kv_heads
            k = k.repeat_interleave(kv_repeat, dim=2)
            v = v.repeat_interleave(kv_repeat, dim=2)
        
        # Reshape for attention: [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Causal mask
        if self.config.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Dropout
        if self.training and self.config.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.config.dropout)
        
        # Apply to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape back: [batch, seq, heads, dim]
        output = output.transpose(1, 2)
        
        return output


class PyTorchSDPAttention(AttentionBase):
    """PyTorch 2.0+ Scaled Dot Product Attention."""
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Expand KV heads for GQA
        num_kv_heads = k.shape[2]
        if num_kv_heads != num_heads:
            kv_repeat = num_heads // num_kv_heads
            k = k.repeat_interleave(kv_repeat, dim=2)
            v = v.repeat_interleave(kv_repeat, dim=2)
        
        # Reshape for SDPA: [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Use SDPA
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=self.config.causal and attention_mask is None,
            scale=self.scale
        )
        
        # Reshape back: [batch, seq, heads, dim]
        output = output.transpose(1, 2)
        
        return output


class FlashAttention2(AttentionBase):
    """
    Flash Attention 2 implementation.
    
    Key features:
    - IO-aware algorithm reduces memory reads/writes
    - Supports GQA (Grouped Query Attention)
    - Supports causal and non-causal
    - Supports variable length sequences
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError("flash-attn library required for FlashAttention2")
    
    def forward(
        self,
        q: torch.Tensor,  # [batch, seq_len, num_heads, head_dim]
        k: torch.Tensor,  # [batch, seq_len, num_kv_heads, head_dim]
        v: torch.Tensor,  # [batch, seq_len, num_kv_heads, head_dim]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Flash attention expects [batch, seq, heads, dim]
        # which is what we have
        
        output = flash_attn_func(
            q, k, v,
            dropout_p=self.config.dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=self.config.causal,
            window_size=self.config.window_size,
            deterministic=self.config.deterministic
        )
        
        return output


class FlashAttention3(AttentionBase):
    """
    Flash Attention 3 implementation (when available).
    
    Additional features over FA2:
    - Better support for newer GPU architectures
    - Improved pipelining
    - FP8 support
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError("flash-attn library required")
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # FA3 has same API as FA2 for basic usage
        output = flash_attn_func(
            q, k, v,
            dropout_p=self.config.dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=self.config.causal,
            window_size=self.config.window_size,
            deterministic=self.config.deterministic
        )
        
        return output


class FlashAttentionWithKVCache(nn.Module):
    """
    Flash Attention with KV cache for efficient inference.
    
    Optimized for autoregressive generation.
    """
    
    def __init__(self, config: AttentionConfig, max_seq_len: int = 8192):
        super().__init__()
        self.config = config
        self.max_seq_len = max_seq_len
        self.scale = config.softmax_scale or (1.0 / math.sqrt(config.head_dim))
        
        # KV cache will be allocated on first use
        self._k_cache: Optional[torch.Tensor] = None
        self._v_cache: Optional[torch.Tensor] = None
        self._cache_seq_len = 0
    
    def reset_cache(self):
        """Reset the KV cache."""
        self._k_cache = None
        self._v_cache = None
        self._cache_seq_len = 0
    
    def _init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Initialize KV cache."""
        self._k_cache = torch.zeros(
            batch_size,
            self.max_seq_len,
            self.config.num_kv_heads,
            self.config.head_dim,
            device=device,
            dtype=dtype
        )
        self._v_cache = torch.zeros(
            batch_size,
            self.max_seq_len,
            self.config.num_kv_heads,
            self.config.head_dim,
            device=device,
            dtype=dtype
        )
    
    def forward(
        self,
        q: torch.Tensor,  # [batch, 1 or seq_len, num_heads, head_dim]
        k: torch.Tensor,  # [batch, 1 or seq_len, num_kv_heads, head_dim]
        v: torch.Tensor,  # [batch, 1 or seq_len, num_kv_heads, head_dim]
        use_cache: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        if use_cache:
            # Initialize cache if needed
            if self._k_cache is None:
                self._init_cache(batch_size, q.device, q.dtype)
            
            # Update cache
            new_seq_len = self._cache_seq_len + seq_len
            self._k_cache[:, self._cache_seq_len:new_seq_len] = k
            self._v_cache[:, self._cache_seq_len:new_seq_len] = v
            
            # Use cached KV
            k_full = self._k_cache[:, :new_seq_len]
            v_full = self._v_cache[:, :new_seq_len]
            
            self._cache_seq_len = new_seq_len
            
            if FLASH_ATTN_AVAILABLE:
                # Use flash attention with kv cache
                try:
                    output = flash_attn_with_kvcache(
                        q, self._k_cache, self._v_cache,
                        cache_seqlens=torch.tensor([new_seq_len], device=q.device),
                        softmax_scale=self.scale,
                        causal=self.config.causal
                    )
                    return output
                except Exception:
                    pass  # Fall back to standard
            
            # Fallback: standard attention with full KV
            return self._standard_attention(q, k_full, v_full)
        else:
            # No cache - standard attention
            return self._standard_attention(q, k, v)
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """Standard attention computation."""
        batch_size, q_len, num_heads, head_dim = q.shape
        kv_len = k.shape[1]
        
        # Expand KV heads for GQA
        num_kv_heads = k.shape[2]
        if num_kv_heads != num_heads:
            kv_repeat = num_heads // num_kv_heads
            k = k.repeat_interleave(kv_repeat, dim=2)
            v = v.repeat_interleave(kv_repeat, dim=2)
        
        # Reshape: [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask (only need to mask new tokens against future)
        if self.config.causal:
            # Create causal mask for the query positions
            causal_mask = torch.triu(
                torch.ones(q_len, kv_len, device=q.device, dtype=torch.bool),
                diagonal=kv_len - q_len + 1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output.transpose(1, 2)


class SlidingWindowAttention(AttentionBase):
    """
    Sliding window attention for long sequences.
    
    Each token only attends to a fixed window of previous tokens.
    """
    
    def __init__(self, config: AttentionConfig, window_size: int):
        config.window_size = (window_size, 0)  # Left window, no right
        super().__init__(config)
        self.window_size = window_size
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if FLASH_ATTN_AVAILABLE:
            # Flash attention handles sliding window natively
            return FlashAttention2.forward(self, q, k, v, attention_mask)
        
        # Fallback: standard attention with window mask
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Expand KV heads
        num_kv_heads = k.shape[2]
        if num_kv_heads != num_heads:
            kv_repeat = num_heads // num_kv_heads
            k = k.repeat_interleave(kv_repeat, dim=2)
            v = v.repeat_interleave(kv_repeat, dim=2)
        
        # Reshape
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Create sliding window mask
        positions = torch.arange(seq_len, device=q.device)
        mask = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs() > self.window_size
        
        # Apply causal constraint
        if self.config.causal:
            causal_mask = torch.triu(torch.ones_like(mask), diagonal=1)
            mask = mask | causal_mask.bool()
        
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output.transpose(1, 2)


def create_attention(config: AttentionConfig, backend: AttentionBackend = None) -> AttentionBase:
    """
    Create attention module with best available backend.
    
    Args:
        config: Attention configuration
        backend: Specific backend to use (auto-selects if None)
    
    Returns:
        Attention module
    """
    if backend is None:
        backend = get_best_backend()
    
    logger.info(f"Creating attention with backend: {backend.value}")
    
    if backend == AttentionBackend.FLASH_ATTN_3:
        return FlashAttention3(config)
    elif backend == AttentionBackend.FLASH_ATTN_2:
        return FlashAttention2(config)
    elif backend == AttentionBackend.FLASH_ATTN_1:
        return FlashAttention2(config)  # API compatible
    elif backend == AttentionBackend.PYTORCH_SDPA:
        return PyTorchSDPAttention(config)
    else:
        return PyTorchNativeAttention(config)


__all__ = [
    'FlashAttention2',
    'FlashAttention3',
    'FlashAttentionWithKVCache',
    'SlidingWindowAttention',
    'PyTorchSDPAttention',
    'PyTorchNativeAttention',
    'AttentionConfig',
    'AttentionBackend',
    'create_attention',
    'get_best_backend',
    'FLASH_ATTN_AVAILABLE',
    'FLASH_ATTN_VERSION'
]
