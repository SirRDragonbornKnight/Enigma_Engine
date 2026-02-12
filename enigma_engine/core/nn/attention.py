"""
Attention Mechanisms for Forge

Contains:
- MultiHeadAttention: Standard multi-head attention
- GroupedQueryAttention: GQA for memory efficiency  
- SlidingWindowAttention: For long sequences
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import RotaryEmbedding, apply_rotary_pos_emb


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention with RoPE.
    
    Features:
    - Rotary positional embeddings
    - Optional KV-cache for efficient generation
    - Causal masking
    
    Args:
        dim: Model dimension
        n_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        dropout: Attention dropout rate
    """
    def __init__(
        self, 
        dim: int, 
        n_heads: int, 
        max_seq_len: int = 2048, 
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            attention_mask: Optional attention mask
            kv_cache: Optional KV cache for generation
            use_cache: Whether to return updated cache
            
        Returns:
            Output tensor and optional new KV cache
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Handle KV cache
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        new_cache = (k, v) if use_cache else None
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(x, k.shape[2])
        q_pos = q.shape[2]
        q, k_rot = apply_rotary_pos_emb(
            q, k[:, :, -q_pos:, :] if kv_cache else k,
            cos[:, :, -q_pos:, :], sin[:, :, -q_pos:, :]
        )
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k_rot], dim=2)
        else:
            k = k_rot
        
        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, k.shape[2], dtype=torch.bool, device=x.device), 
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask[None, None, :, :], float('-inf'))
        else:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.o_proj(attn_output), new_cache


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    
    Uses fewer key-value heads than query heads for memory efficiency.
    Each group of query heads shares the same key-value head.
    
    From "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
    
    Args:
        dim: Model dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key-value heads (must divide n_heads)
        max_seq_len: Maximum sequence length
        dropout: Attention dropout rate
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        assert n_heads % n_kv_heads == 0
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Repeat k, v for each query head group
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        new_cache = (k, v) if use_cache else None
        
        cos, sin = self.rotary_emb(x, k.shape[2])
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, k.shape[2], dtype=torch.bool, device=x.device),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask[None, None, :, :], float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.o_proj(attn_output), new_cache


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention for long sequences.
    
    Each token only attends to tokens within a fixed window.
    More memory efficient for very long sequences.
    
    From Mistral/Longformer architecture.
    
    Args:
        dim: Model dimension
        n_heads: Number of heads
        window_size: Size of attention window
        max_seq_len: Maximum sequence length
        dropout: Attention dropout rate
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        window_size: int = 512,
        max_seq_len: int = 2048,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Only keep last window_size keys/values
        if k.shape[2] > self.window_size:
            k = k[:, :, -self.window_size:, :]
            v = v[:, :, -self.window_size:, :]
        
        new_cache = (k, v) if use_cache else None
        
        cos, sin = self.rotary_emb(x, k.shape[2])
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Create sliding window mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, k.shape[2], dtype=torch.bool, device=x.device),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask[None, None, :, :], float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.o_proj(attn_output), new_cache
