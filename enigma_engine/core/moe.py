"""
Mixture of Experts (MoE)

Sparse mixture of experts architecture for efficient scaling.
Supports top-k routing, load balancing, and expert parallelism.

FILE: enigma_engine/core/moe.py
TYPE: Core AI Architecture
MAIN CLASSES: MoELayer, Router, Expert, MoETransformer
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class RouterType(Enum):
    """Types of MoE routers."""
    TOP_K = "top_k"
    EXPERT_CHOICE = "expert_choice"
    SOFT = "soft"


@dataclass
class MoEConfig:
    """Mixture of Experts configuration."""
    num_experts: int = 8
    num_experts_per_tok: int = 2
    hidden_size: int = 768
    intermediate_size: int = 3072
    
    # Load balancing
    load_balancing_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    
    # Capacity
    capacity_factor: float = 1.25
    drop_tokens: bool = False
    
    # Expert parallelism
    expert_parallel: bool = False


class Router(nn.Module):
    """
    Route tokens to experts using learned routing.
    
    Implements top-k routing with auxiliary load balancing loss.
    """
    
    def __init__(self,
                 hidden_size: int,
                 num_experts: int,
                 top_k: int = 2,
                 router_type: RouterType = RouterType.TOP_K):
        """
        Initialize router.
        
        Args:
            hidden_size: Input hidden size
            num_experts: Number of experts
            top_k: Number of experts per token
            router_type: Type of routing
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_type = router_type
        
        # Router gate
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # For jitter noise during training
        self.jitter_noise = 0.01
    
    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Route tokens to experts.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            router_probs: Routing probabilities [batch, seq_len, num_experts]
            expert_indices: Selected expert indices [batch, seq_len, top_k]
            expert_weights: Weights for selected experts [batch, seq_len, top_k]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute routing logits
        router_logits = self.gate(hidden_states)  # [B, S, E]
        
        # Add noise during training for exploration
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise
        
        # Compute routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        expert_weights, expert_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        
        # Normalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        return router_probs, expert_indices, expert_weights
    
    def compute_load_balancing_loss(self,
                                    router_probs: Tensor,
                                    expert_indices: Tensor) -> Tensor:
        """
        Compute auxiliary load balancing loss.
        
        Encourages balanced routing across experts.
        """
        num_tokens = router_probs.shape[0] * router_probs.shape[1]
        
        # Fraction of tokens routed to each expert
        tokens_per_expert = torch.zeros(self.num_experts, device=router_probs.device)
        for i in range(self.num_experts):
            tokens_per_expert[i] = (expert_indices == i).sum().float()
        tokens_per_expert = tokens_per_expert / num_tokens
        
        # Mean routing probability per expert
        mean_probs = router_probs.mean(dim=(0, 1))
        
        # Load balancing loss (want uniform distribution)
        lb_loss = self.num_experts * (tokens_per_expert * mean_probs).sum()
        
        return lb_loss
    
    def compute_router_z_loss(self, router_logits: Tensor) -> Tensor:
        """
        Compute router z-loss for training stability.
        
        Penalizes large router logits.
        """
        z_loss = torch.logsumexp(router_logits, dim=-1).mean()
        return z_loss


class Expert(nn.Module):
    """
    Single expert MLP.
    
    Uses SwiGLU activation for better performance.
    """
    
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 dropout: float = 0.0):
        """
        Initialize expert.
        
        Args:
            hidden_size: Input/output dimension
            intermediate_size: FFN intermediate dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with SwiGLU."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        output = self.down_proj(hidden)
        return self.dropout(output)


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.
    
    Routes tokens to selected experts and combines outputs.
    """
    
    def __init__(self, config: MoEConfig):
        """
        Initialize MoE layer.
        
        Args:
            config: MoE configuration
        """
        super().__init__()
        
        self.config = config
        
        # Router
        self.router = Router(
            config.hidden_size,
            config.num_experts,
            config.num_experts_per_tok
        )
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_experts)
        ])
        
        # Auxiliary loss accumulators
        self._aux_loss = 0.0
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Forward pass through MoE layer.
        
        Args:
            hidden_states: Input [batch, seq_len, hidden_size]
            
        Returns:
            Output [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Route tokens
        router_probs, expert_indices, expert_weights = self.router(hidden_states)
        
        # Compute auxiliary loss
        if self.training:
            lb_loss = self.router.compute_load_balancing_loss(router_probs, expert_indices)
            self._aux_loss = self.config.load_balancing_loss_coef * lb_loss
        
        # Initialize output
        output = torch.zeros_like(hidden_states)
        
        # Process each expert
        for expert_idx in range(self.config.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)  # [B, S]
            
            if not expert_mask.any():
                continue
            
            # Get tokens for this expert
            tokens = hidden_states[expert_mask]  # [N, H]
            
            # Process through expert
            expert_output = self.experts[expert_idx](tokens)
            
            # Get weights for this expert's outputs
            # Find which top-k position this expert is in
            for k in range(self.config.num_experts_per_tok):
                k_mask = expert_mask & (expert_indices[:, :, k] == expert_idx)
                if k_mask.any():
                    weights = expert_weights[:, :, k][k_mask].unsqueeze(-1)
                    
                    # Map back to positions
                    positions = torch.where(k_mask)
                    output[positions[0], positions[1]] += weights * expert_output[:len(weights)]
        
        return output
    
    def get_aux_loss(self) -> float:
        """Get auxiliary loss for this forward pass."""
        loss = self._aux_loss
        self._aux_loss = 0.0
        # Convert tensor to float if needed
        if hasattr(loss, 'item'):
            return loss.item()
        return loss


class MoETransformerBlock(nn.Module):
    """
    Transformer block with MoE FFN.
    
    Replaces standard FFN with sparse MoE layer.
    """
    
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 moe_config: MoEConfig,
                 dropout: float = 0.1):
        """
        Initialize MoE transformer block.
        
        Args:
            hidden_size: Model dimension
            num_heads: Number of attention heads
            moe_config: MoE configuration
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # MoE FFN
        self.moe = MoELayer(moe_config)
        
        # Norms
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                x: Tensor,
                attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor
        """
        # Self-attention
        residual = x
        x = self.attn_norm(x)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attention_mask)
        x = residual + self.dropout(attn_output)
        
        # MoE FFN
        residual = x
        x = self.ffn_norm(x)
        moe_output = self.moe(x)
        x = residual + self.dropout(moe_output)
        
        return x
    
    def get_aux_loss(self) -> float:
        """Get auxiliary loss from MoE layer."""
        return self.moe.get_aux_loss()


class MoETransformer(nn.Module):
    """
    Complete MoE Transformer model.
    
    Combines embedding, MoE transformer blocks, and output head.
    """
    
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 max_seq_len: int = 2048,
                 moe_config: MoEConfig = None,
                 moe_every_n_layers: int = 2):
        """
        Initialize MoE Transformer.
        
        Args:
            vocab_size: Vocabulary size
            hidden_size: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            moe_config: MoE configuration
            moe_every_n_layers: Use MoE every N layers (1 = all layers)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Update MoE config with hidden size
        if moe_config is None:
            moe_config = MoEConfig(hidden_size=hidden_size)
        moe_config.hidden_size = hidden_size
        self.moe_config = moe_config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = nn.Embedding(max_seq_len, hidden_size)
        
        # Transformer layers (MoE or dense)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if (i + 1) % moe_every_n_layers == 0:
                # MoE layer
                self.layers.append(MoETransformerBlock(
                    hidden_size, num_heads, moe_config
                ))
            else:
                # Dense layer (standard transformer block)
                self.layers.append(self._create_dense_block(hidden_size, num_heads))
        
        # Output
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight
        
        self._init_weights()
    
    def _create_dense_block(self, hidden_size: int, num_heads: int) -> nn.Module:
        """Create a standard dense transformer block."""
        class DenseBlock(nn.Module):
            def __init__(self, hidden_size, num_heads):
                super().__init__()
                self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
                self.ffn = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
                self.norm1 = nn.LayerNorm(hidden_size)
                self.norm2 = nn.LayerNorm(hidden_size)
            
            def forward(self, x, attention_mask=None):
                residual = x
                x = self.norm1(x)
                x, _ = self.attn(x, x, x, attn_mask=attention_mask)
                x = residual + x
                
                residual = x
                x = self.norm2(x)
                x = self.ffn(x)
                x = residual + x
                return x
            
            def get_aux_loss(self):
                return 0.0
        
        return DenseBlock(hidden_size, num_heads)
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self,
                input_ids: Tensor,
                attention_mask: Optional[Tensor] = None) -> tuple[Tensor, float]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Optional attention mask
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            aux_loss: Total auxiliary loss
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(positions)
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
                diagonal=1
            )
        
        # Forward through layers
        aux_loss = 0.0
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            aux_loss += layer.get_aux_loss()
        
        # Output
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, aux_loss
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
            n_params -= self.embed_positions.weight.numel()
        return n_params
    
    def get_num_active_params(self) -> int:
        """Get number of active parameters per forward pass."""
        # Only top-k experts are active
        active = 0
        
        for layer in self.layers:
            if isinstance(layer, MoETransformerBlock):
                # Count attention + router + 2 experts
                active += self.hidden_size * self.hidden_size * 4  # Attention
                active += self.hidden_size * self.moe_config.num_experts  # Router
                active += self.moe_config.num_experts_per_tok * (
                    self.hidden_size * self.moe_config.intermediate_size * 3
                )  # Active experts
            else:
                # Dense layer fully active
                active += sum(p.numel() for p in layer.parameters())
        
        return active


def create_moe_model(vocab_size: int = 32000,
                     hidden_size: int = 768,
                     num_layers: int = 12,
                     num_experts: int = 8,
                     experts_per_token: int = 2) -> MoETransformer:
    """
    Create a Mixture of Experts model.
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Model hidden dimension
        num_layers: Number of layers
        num_experts: Total number of experts
        experts_per_token: Active experts per token
        
    Returns:
        MoE Transformer model
    """
    config = MoEConfig(
        num_experts=num_experts,
        num_experts_per_tok=experts_per_token,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4
    )
    
    return MoETransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        moe_config=config
    )


__all__ = [
    'MoEConfig',
    'MoELayer',
    'MoETransformerBlock',
    'MoETransformer',
    'Router',
    'RouterType',
    'Expert',
    'create_moe_model'
]
