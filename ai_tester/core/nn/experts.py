"""
Mixture of Experts and Advanced Components

Contains:
- MixtureOfExperts: Sparse routing to expert networks
- LoRALayer: Low-rank adaptation for efficient fine-tuning
- AdaptiveLayerNorm: Conditional normalization
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .activations import FeedForward


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer for efficient fine-tuning.
    
    Adds trainable low-rank matrices to frozen pretrained weights.
    Instead of fine-tuning all weights, LoRA trains small adapter matrices
    that can be merged back into the base weights.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        rank: Rank of the low-rank matrices (smaller = fewer params)
        alpha: Scaling factor for the LoRA update
    
    Usage:
        # Wrap existing linear layer
        base_layer = nn.Linear(512, 512)
        lora = LoRALayer(512, 512, rank=8)
        
        # During forward:
        output = base_layer(x) + lora(x)
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices: W = A @ B where A is (in, rank) and B is (rank, out)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A @ self.lora_B) * self.scaling
    
    def merge_weights(self, base_weight: torch.Tensor) -> torch.Tensor:
        """Merge LoRA weights into base weight matrix."""
        return base_weight + (self.lora_A @ self.lora_B).T * self.scaling


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization - learns per-sample scale and shift.
    
    Used in conditional generation where normalization parameters
    depend on a conditioning signal (like class embeddings, style codes, etc.)
    
    Args:
        dim: Feature dimension to normalize
        cond_dim: Conditioning signal dimension
    """
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.scale_proj = nn.Linear(cond_dim, dim)
        self.shift_proj = nn.Linear(cond_dim, dim)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            cond: Conditioning tensor (batch, cond_dim)
        """
        x = self.norm(x)
        scale = self.scale_proj(cond)
        shift = self.shift_proj(cond)
        # Expand conditioning for sequence dimension
        if scale.dim() == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return x * (1 + scale) + shift


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) - routes inputs to specialized sub-networks.
    
    Efficient way to scale models with sparse computation.
    Only top-k experts are activated per token, keeping compute constant
    while increasing model capacity.
    
    Args:
        dim: Input/output dimension
        n_experts: Total number of expert networks
        top_k: Number of experts to route each token to
        dropout: Dropout rate in expert FFN
    
    Usage:
        moe = MixtureOfExperts(dim=512, n_experts=8, top_k=2)
        output = moe(hidden_states)  # Routes to 2 of 8 experts per token
    """
    def __init__(self, dim: int, n_experts: int = 4, top_k: int = 2, dropout: float = 0.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        # Router network decides which experts to use
        self.router = nn.Linear(dim, n_experts)
        
        # Expert networks (each is a small FFN)
        self.experts = nn.ModuleList([
            FeedForward(dim, dim * 4, dropout)
            for _ in range(n_experts)
        ])
        
        # For load balancing loss
        self.register_buffer("expert_counts", torch.zeros(n_experts))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            
        Returns:
            Output tensor (batch, seq_len, dim)
        """
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
                
                # Get weight for this expert for each selected token
                expert_idx = (selected_experts[mask] == i)
                weights = (router_weights[mask] * expert_idx.float()).sum(dim=-1, keepdim=True)
                
                output[mask] += weights * expert_output
        
        return output.view(batch_size, seq_len, dim)
    
    def get_load_balance_loss(self) -> torch.Tensor:
        """
        Compute auxiliary loss for expert load balancing.
        
        Encourages even distribution of tokens across experts.
        """
        counts = self.expert_counts
        if counts.sum() == 0:
            return torch.tensor(0.0, device=counts.device)
        
        # Ideal: each expert gets 1/n_experts of tokens
        ideal = counts.sum() / self.n_experts
        balance_loss = ((counts - ideal) ** 2).mean()
        return balance_loss


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention - uses single K,V heads with multiple Q heads.
    
    More memory efficient for generation while maintaining quality.
    Good for inference on memory-constrained devices.
    
    Args:
        dim: Model dimension
        n_heads: Number of query heads
        dropout: Attention dropout rate
    """
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
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


__all__ = [
    "LoRALayer",
    "AdaptiveLayerNorm", 
    "MixtureOfExperts",
    "MultiQueryAttention",
]
