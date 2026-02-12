"""
Additional Neural Network Layers and Components for Forge

This module provides building blocks that can be used to extend or customize
the Enigma AI Engine model architecture. These are standalone components that can be
mixed and matched.

NOTE: These components are also available in the enigma_engine.core.nn subpackage
with more documentation and features. Consider importing from there instead:

    from enigma_engine.core.nn import (
        SwiGLU, GeGLU, FeedForward,  # activations
        MultiHeadAttention, GroupedQueryAttention,  # attention
        LoRALayer, MixtureOfExperts,  # advanced
    )
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Standard feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GLU(nn.Module):
    """Gated Linear Unit - gates the output with a learnable mechanism."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_ff * 2)
        self.out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, gate = self.proj(x).chunk(2, dim=-1)
        return self.dropout(self.out(hidden * torch.sigmoid(gate)))


class GeGLU(nn.Module):
    """GELU-Gated Linear Unit - more expressive than standard GLU."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.out(F.gelu(self.w1(x)) * self.w2(x)))


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention - uses single K,V heads with multiple Q heads.
    More memory efficient for generation while maintaining quality.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        # Single K,V head shared across all Q heads
        self.k_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)

        # Expand K,V to match number of Q heads
        k = k.expand(-1, self.n_heads, -1, -1)
        v = v.expand(-1, self.n_heads, -1, -1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.o_proj(attn_output)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) - balance between MHA and MQA.
    Uses fewer K,V heads than Q heads for efficiency.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.n_groups = n_heads // n_kv_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.head_dim * n_kv_heads, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim * n_kv_heads, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat K,V for each group
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.o_proj(attn_output)


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention - attends only to local context.
    Efficient for long sequences where full attention is too expensive.
    """

    def __init__(self, dim: int, n_heads: int, window_size: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Create sliding window mask
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = False
        mask = mask.triu(1)  # Also apply causal mask

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.masked_fill(mask[None, None, :, :], float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.o_proj(attn_output)


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer for efficient fine-tuning.
    Adds trainable low-rank matrices to frozen pretrained weights.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization - learns per-sample scale and shift."""

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.scale_proj = nn.Linear(cond_dim, dim)
        self.shift_proj = nn.Linear(cond_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        scale = self.scale_proj(cond)
        shift = self.shift_proj(cond)
        return x * (1 + scale) + shift


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) - routes inputs to specialized sub-networks.
    Efficient way to scale models with sparse computation.
    """

    def __init__(self, dim: int, n_experts: int = 4, top_k: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # Router
        self.router = nn.Linear(dim, n_experts)

        # Expert networks
        self.experts = nn.ModuleList([
            FeedForward(dim, dim * 4, dropout)
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        # Get routing weights
        router_logits = self.router(x_flat)
        router_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        router_weights = F.softmax(router_weights, dim=-1)

        # Compute expert outputs
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            mask = (selected_experts == i).any(dim=-1)
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = expert(expert_input)

                # Get weight for this expert
                expert_idx = (selected_experts[mask] == i)
                weights = (router_weights[mask] * expert_idx.float()).sum(dim=-1, keepdim=True)

                output[mask] += weights * expert_output

        return output.view(batch_size, seq_len, dim)
