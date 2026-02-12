"""
Positional Embeddings for Forge

Contains:
- RotaryEmbedding: Rotary Position Embeddings (RoPE)
- SinusoidalEmbedding: Classic sinusoidal positional encoding
- LearnedEmbedding: Learned positional embeddings
"""
import math

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    
    Encodes position information by rotating query and key vectors.
    This allows the model to learn relative positions naturally.
    
    From the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    
    Args:
        dim: Head dimension (must be even)
        max_seq_len: Maximum sequence length to precompute
        base: Base for frequency computation
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int) -> None:
        """Build sin/cos cache for the given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin for the given sequence length.
        
        Args:
            x: Input tensor (unused, for device detection)
            seq_len: Sequence length
            
        Returns:
            Tuple of (cos, sin) tensors
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to queries and keys.
    
    Args:
        q: Query tensor (batch, heads, seq_len, head_dim)
        k: Key tensor (batch, heads, seq_len, head_dim)
        cos: Cosine tensor
        sin: Sine tensor
        
    Returns:
        Tuple of rotated (q, k)
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SinusoidalEmbedding(nn.Module):
    """
    Classic sinusoidal positional encoding from "Attention is All You Need".
    
    Uses fixed sine and cosine functions of different frequencies.
    
    Args:
        dim: Embedding dimension
        max_seq_len: Maximum sequence length
    """
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class LearnedEmbedding(nn.Module):
    """
    Learned positional embeddings.
    
    Simple embedding table that's trained along with the model.
    
    Args:
        max_seq_len: Maximum sequence length
        dim: Embedding dimension
    """
    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional embeddings.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            
        Returns:
            Tensor with positional embeddings added
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.embedding(positions)
