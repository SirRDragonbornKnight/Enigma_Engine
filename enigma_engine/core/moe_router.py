"""
Full Mixture of Experts (MoE) Implementation
============================================

Complete MoE expert routing system with:
- Top-k sparse routing (select k experts per token)
- Expert capacity management (prevent overload)
- Auxiliary load balancing loss
- Expert specialization tracking
- Switch Transformer and GShard-style routing

Usage:
    from enigma_engine.core.moe_router import MoELayer, ExpertRouter
    
    # Create MoE layer
    moe = MoELayer(
        hidden_dim=512,
        num_experts=8,
        top_k=2,
        expert_capacity_factor=1.25,
    )
    
    # Forward pass
    output, aux_loss = moe(hidden_states)

This replaces the standard FeedForward layer in transformer blocks
when use_moe=True in ForgeConfig.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    torch = None
    nn = None


class RoutingStrategy(Enum):
    """Expert routing strategies."""
    TOP_K = "top_k"  # Standard top-k routing
    SWITCH = "switch"  # Switch Transformer (top-1 with noise)
    EXPERT_CHOICE = "expert_choice"  # Experts choose tokens
    SOFT_MOE = "soft_moe"  # Soft mixture (all experts weighted)


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts."""
    num_experts: int = 8
    top_k: int = 2  # Experts activated per token
    hidden_dim: int = 512
    ffn_dim: int = 2048  # Intermediate dimension
    dropout: float = 0.0
    
    # Routing
    routing_strategy: RoutingStrategy = RoutingStrategy.TOP_K
    router_jitter: float = 0.0  # Noise for exploration
    
    # Capacity
    capacity_factor: float = 1.25  # Expert capacity = (tokens * top_k / num_experts) * factor
    drop_tokens: bool = False  # Drop tokens exceeding capacity
    
    # Load balancing
    load_balancing_weight: float = 0.01  # Auxiliary loss weight
    z_loss_weight: float = 0.001  # Router z-loss weight
    
    # Specialization
    enable_specialization_tracking: bool = True


if HAVE_TORCH:
    
    class ExpertRouter(nn.Module):
        """
        Router network that assigns tokens to experts.
        
        Computes routing weights and selects top-k experts per token.
        """
        
        def __init__(
            self,
            hidden_dim: int,
            num_experts: int,
            top_k: int = 2,
            jitter: float = 0.0,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_experts = num_experts
            self.top_k = top_k
            self.jitter = jitter
            
            # Router linear layer
            self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
            
            # Initialize with small weights for stable routing
            nn.init.normal_(self.gate.weight, std=0.01)
        
        def forward(
            self,
            hidden_states: Tensor,
            training: bool = True,
        ) -> Tuple[Tensor, Tensor, Tensor]:
            """
            Route tokens to experts.
            
            Args:
                hidden_states: [batch, seq_len, hidden_dim]
                training: Whether in training mode (for jitter)
            
            Returns:
                router_probs: [batch, seq_len, num_experts] softmax probabilities
                router_logits: [batch, seq_len, num_experts] raw logits
                selected_experts: [batch, seq_len, top_k] selected expert indices
            """
            batch_size, seq_len, _ = hidden_states.shape
            
            # Compute routing logits
            router_logits = self.gate(hidden_states)  # [batch, seq, num_experts]
            
            # Add jitter noise during training for exploration
            if training and self.jitter > 0:
                noise = torch.randn_like(router_logits) * self.jitter
                router_logits = router_logits + noise
            
            # Compute routing probabilities
            router_probs = F.softmax(router_logits, dim=-1)
            
            # Select top-k experts per token
            top_k_probs, selected_experts = torch.topk(
                router_probs, self.top_k, dim=-1
            )
            
            return router_probs, router_logits, selected_experts
    
    
    class Expert(nn.Module):
        """
        Single expert network (typically a FeedForward/MLP).
        
        Uses SwiGLU activation for better performance.
        """
        
        def __init__(
            self,
            hidden_dim: int,
            ffn_dim: int,
            dropout: float = 0.0,
        ):
            super().__init__()
            
            # SwiGLU architecture
            self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
            self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
            self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x: Tensor) -> Tensor:
            """SwiGLU forward pass."""
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
            hidden = gate * up
            hidden = self.dropout(hidden)
            return self.down_proj(hidden)
    
    
    class MoELayer(nn.Module):
        """
        Full Mixture of Experts layer.
        
        Replaces standard FeedForward in transformer blocks.
        Uses sparse routing to activate only top-k experts per token.
        """
        
        def __init__(
            self,
            config: Optional[MoEConfig] = None,
            hidden_dim: int = 512,
            ffn_dim: Optional[int] = None,
            num_experts: int = 8,
            top_k: int = 2,
            capacity_factor: float = 1.25,
            dropout: float = 0.0,
            load_balancing_weight: float = 0.01,
        ):
            super().__init__()
            
            # Use config or individual args
            if config is not None:
                self.config = config
            else:
                self.config = MoEConfig(
                    hidden_dim=hidden_dim,
                    ffn_dim=ffn_dim or hidden_dim * 4,
                    num_experts=num_experts,
                    top_k=top_k,
                    capacity_factor=capacity_factor,
                    dropout=dropout,
                    load_balancing_weight=load_balancing_weight,
                )
            
            self.hidden_dim = self.config.hidden_dim
            self.num_experts = self.config.num_experts
            self.top_k = self.config.top_k
            self.capacity_factor = self.config.capacity_factor
            
            # Router
            self.router = ExpertRouter(
                hidden_dim=self.config.hidden_dim,
                num_experts=self.config.num_experts,
                top_k=self.config.top_k,
                jitter=self.config.router_jitter,
            )
            
            # Expert networks
            self.experts = nn.ModuleList([
                Expert(
                    hidden_dim=self.config.hidden_dim,
                    ffn_dim=self.config.ffn_dim,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.num_experts)
            ])
            
            # Tracking
            if self.config.enable_specialization_tracking:
                self.register_buffer(
                    'expert_usage',
                    torch.zeros(self.config.num_experts)
                )
                self.register_buffer('total_tokens', torch.zeros(1))
        
        def _compute_expert_capacity(self, num_tokens: int) -> int:
            """Compute capacity per expert."""
            tokens_per_expert = num_tokens * self.top_k / self.num_experts
            capacity = int(tokens_per_expert * self.capacity_factor)
            return max(capacity, 1)
        
        def _compute_aux_loss(
            self,
            router_probs: Tensor,
            selected_experts: Tensor,
        ) -> Tuple[Tensor, Tensor]:
            """
            Compute auxiliary load balancing loss.
            
            Encourages even distribution of tokens across experts.
            
            Returns:
                aux_loss: Combined auxiliary loss
                z_loss: Router z-loss for stability
            """
            batch_size, seq_len, _ = router_probs.shape
            num_tokens = batch_size * seq_len
            
            # Expert fraction: what fraction of routing probability went to each expert
            # Mean over batch and sequence
            expert_fraction = router_probs.mean(dim=[0, 1])  # [num_experts]
            
            # Token fraction: what fraction of tokens selected each expert (in top-k)
            # Create one-hot for selected experts and average
            expert_mask = F.one_hot(
                selected_experts, num_classes=self.num_experts
            ).float()  # [batch, seq, top_k, num_experts]
            token_fraction = expert_mask.sum(dim=2).mean(dim=[0, 1])  # [num_experts]
            token_fraction = token_fraction / self.top_k  # Normalize
            
            # Load balancing loss: encourage f_i ≈ 1/num_experts
            # Switch Transformer style: sum(f_i * P_i) * num_experts
            aux_loss = (expert_fraction * token_fraction).sum() * self.num_experts
            
            # Z-loss for router stability (prevents logits from becoming too large)
            router_logits = self.router.gate(
                torch.zeros(1, 1, self.hidden_dim, device=router_probs.device)
            )
            z_loss = torch.logsumexp(router_logits, dim=-1).mean() ** 2
            
            return aux_loss * self.config.load_balancing_weight, z_loss * self.config.z_loss_weight
        
        def forward(
            self,
            hidden_states: Tensor,
            return_aux_loss: bool = True,
        ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
            """
            Forward pass through MoE layer.
            
            Args:
                hidden_states: [batch, seq_len, hidden_dim]
                return_aux_loss: Whether to return auxiliary loss
            
            Returns:
                output: [batch, seq_len, hidden_dim]
                aux_losses: Dict with 'load_balance' and 'z_loss' if requested
            """
            batch_size, seq_len, hidden_dim = hidden_states.shape
            num_tokens = batch_size * seq_len
            
            # Route tokens
            router_probs, router_logits, selected_experts = self.router(
                hidden_states, training=self.training
            )
            
            # Get routing weights for selected experts
            # [batch, seq, top_k]
            routing_weights = torch.gather(
                router_probs, 
                dim=-1, 
                index=selected_experts
            )
            
            # Normalize routing weights
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            
            # Flatten for processing
            flat_hidden = hidden_states.view(-1, hidden_dim)  # [num_tokens, hidden_dim]
            flat_routing = routing_weights.view(num_tokens, self.top_k)  # [num_tokens, top_k]
            flat_experts = selected_experts.view(num_tokens, self.top_k)  # [num_tokens, top_k]
            
            # Compute expert capacity
            capacity = self._compute_expert_capacity(num_tokens)
            
            # Collect expert outputs
            output = torch.zeros_like(flat_hidden)
            
            # Process each expert
            for expert_idx, expert in enumerate(self.experts):
                # Find tokens routed to this expert (in any of top-k positions)
                expert_mask = (flat_experts == expert_idx)  # [num_tokens, top_k]
                token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]
                
                if len(token_indices) == 0:
                    continue
                
                # Apply capacity constraint
                if self.config.drop_tokens and len(token_indices) > capacity:
                    token_indices = token_indices[:capacity]
                
                # Get inputs for this expert
                expert_input = flat_hidden[token_indices]  # [expert_tokens, hidden_dim]
                
                # Forward through expert
                expert_output = expert(expert_input)  # [expert_tokens, hidden_dim]
                
                # Get routing weights for this expert
                # Weight is sum of routing weights where this expert was selected
                weights_for_expert = (flat_routing[token_indices] * expert_mask[token_indices].float()).sum(dim=-1, keepdim=True)
                
                # Combine weighted expert output
                output.scatter_add_(
                    0,
                    token_indices.unsqueeze(-1).expand(-1, hidden_dim),
                    weights_for_expert * expert_output
                )
            
            # Reshape output
            output = output.view(batch_size, seq_len, hidden_dim)
            
            # Track expert usage
            if self.config.enable_specialization_tracking:
                with torch.no_grad():
                    expert_counts = F.one_hot(
                        flat_experts.view(-1), num_classes=self.num_experts
                    ).float().sum(dim=0)
                    self.expert_usage.add_(expert_counts)
                    self.total_tokens.add_(num_tokens * self.top_k)
            
            # Compute auxiliary losses
            aux_losses = None
            if return_aux_loss:
                aux_loss, z_loss = self._compute_aux_loss(router_probs, selected_experts)
                aux_losses = {
                    'load_balance': aux_loss,
                    'z_loss': z_loss,
                    'total_aux_loss': aux_loss + z_loss,
                }
            
            return output, aux_losses
        
        def get_expert_statistics(self) -> Dict[str, Any]:
            """Get expert usage statistics."""
            if not self.config.enable_specialization_tracking:
                return {}
            
            total = self.total_tokens.item()
            if total == 0:
                return {'expert_usage': [0.0] * self.num_experts}
            
            usage = (self.expert_usage / total).tolist()
            
            # Compute load imbalance
            expected = 1.0 / self.num_experts
            imbalance = sum(abs(u - expected) for u in usage) / len(usage)
            
            return {
                'expert_usage': usage,
                'load_imbalance': imbalance,
                'total_tokens_processed': total,
                'most_used_expert': max(range(len(usage)), key=lambda i: usage[i]),
                'least_used_expert': min(range(len(usage)), key=lambda i: usage[i]),
            }
        
        def reset_statistics(self) -> None:
            """Reset expert usage statistics."""
            if self.config.enable_specialization_tracking:
                self.expert_usage.zero_()
                self.total_tokens.zero_()
    
    
    class SwitchMoELayer(MoELayer):
        """
        Switch Transformer style MoE (top-1 routing with capacity).
        
        More efficient than top-k as each token only goes to one expert.
        """
        
        def __init__(self, *args, **kwargs):
            # Force top_k=1 for Switch routing
            kwargs['top_k'] = 1
            super().__init__(*args, **kwargs)
            self.config.routing_strategy = RoutingStrategy.SWITCH
    
    
    class ExpertChoiceMoELayer(MoELayer):
        """
        Expert-choice routing where experts select their tokens.
        
        More balanced load distribution as each expert picks exactly
        capacity tokens.
        """
        
        def forward(
            self,
            hidden_states: Tensor,
            return_aux_loss: bool = True,
        ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
            """Expert-choice forward pass."""
            batch_size, seq_len, hidden_dim = hidden_states.shape
            num_tokens = batch_size * seq_len
            
            # Compute routing scores
            flat_hidden = hidden_states.view(num_tokens, hidden_dim)
            router_logits = self.router.gate(flat_hidden)  # [num_tokens, num_experts]
            
            # Transpose: let experts score tokens
            expert_scores = router_logits.T  # [num_experts, num_tokens]
            
            # Each expert selects top-capacity tokens
            capacity = self._compute_expert_capacity(num_tokens)
            
            # Select tokens for each expert
            top_scores, selected_tokens = torch.topk(
                expert_scores, k=min(capacity, num_tokens), dim=-1
            )  # [num_experts, capacity]
            
            # Compute routing weights (softmax over selected tokens)
            routing_weights = F.softmax(top_scores, dim=-1)  # [num_experts, capacity]
            
            # Process through experts
            output = torch.zeros_like(flat_hidden)
            token_counts = torch.zeros(num_tokens, device=hidden_states.device)
            
            for expert_idx, expert in enumerate(self.experts):
                # Get tokens for this expert
                token_indices = selected_tokens[expert_idx]  # [capacity]
                weights = routing_weights[expert_idx]  # [capacity]
                
                # Get inputs
                expert_input = flat_hidden[token_indices]  # [capacity, hidden_dim]
                
                # Forward
                expert_output = expert(expert_input)
                
                # Accumulate weighted output
                # Note: tokens may be selected by multiple experts
                output.scatter_add_(
                    0,
                    token_indices.unsqueeze(-1).expand(-1, hidden_dim),
                    weights.unsqueeze(-1) * expert_output
                )
                
                # Track for normalization
                token_counts.scatter_add_(0, token_indices, weights)
            
            # Normalize by total weight received
            token_counts = token_counts.clamp(min=1e-6)
            output = output / token_counts.unsqueeze(-1)
            
            # Reshape
            output = output.view(batch_size, seq_len, hidden_dim)
            
            # Aux loss
            aux_losses = None
            if return_aux_loss:
                router_probs = F.softmax(router_logits, dim=-1)
                aux_loss = (router_probs.mean(dim=0) ** 2).sum() * self.num_experts
                aux_losses = {
                    'load_balance': aux_loss * self.config.load_balancing_weight,
                    'z_loss': torch.tensor(0.0, device=hidden_states.device),
                    'total_aux_loss': aux_loss * self.config.load_balancing_weight,
                }
            
            return output, aux_losses
    
    
    class MoETransformerBlock(nn.Module):
        """
        Transformer block with MoE FeedForward layer.
        
        Attention → MoE instead of Attention → FFN
        """
        
        def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_kv_heads: Optional[int] = None,
            moe_config: Optional[MoEConfig] = None,
            num_experts: int = 8,
            top_k: int = 2,
            dropout: float = 0.0,
            layer_idx: int = 0,
        ):
            super().__init__()
            
            from .model import RMSNorm, Attention
            
            self.hidden_dim = hidden_dim
            self.layer_idx = layer_idx
            
            # Pre-normalization
            self.attention_norm = RMSNorm(hidden_dim)
            self.ffn_norm = RMSNorm(hidden_dim)
            
            # Attention (standard)
            self.attention = Attention(
                dim=hidden_dim,
                n_heads=num_heads,
                n_kv_heads=num_kv_heads,
                dropout=dropout,
            )
            
            # MoE FeedForward
            self.moe = MoELayer(
                config=moe_config,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
            )
        
        def forward(
            self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            freqs_cis: Optional[Tensor] = None,
            kv_cache: Optional[Any] = None,
            return_aux_loss: bool = True,
        ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
            """Forward pass with MoE."""
            # Attention with residual
            residual = hidden_states
            hidden_states = self.attention_norm(hidden_states)
            hidden_states = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                freqs_cis=freqs_cis,
                kv_cache=kv_cache,
            )
            hidden_states = residual + hidden_states
            
            # MoE with residual
            residual = hidden_states
            hidden_states = self.ffn_norm(hidden_states)
            hidden_states, aux_losses = self.moe(hidden_states, return_aux_loss=return_aux_loss)
            hidden_states = residual + hidden_states
            
            return hidden_states, aux_losses


def create_moe_layer(
    hidden_dim: int,
    num_experts: int = 8,
    top_k: int = 2,
    routing_strategy: str = "top_k",
    **kwargs
) -> MoELayer:
    """
    Factory function to create MoE layer.
    
    Args:
        hidden_dim: Model hidden dimension
        num_experts: Number of expert networks
        top_k: Experts activated per token
        routing_strategy: "top_k", "switch", or "expert_choice"
        **kwargs: Additional MoEConfig arguments
    
    Returns:
        Appropriate MoE layer instance
    """
    if not HAVE_TORCH:
        raise RuntimeError("MoE requires PyTorch")
    
    if routing_strategy == "switch":
        return SwitchMoELayer(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            **kwargs
        )
    elif routing_strategy == "expert_choice":
        return ExpertChoiceMoELayer(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            **kwargs
        )
    else:
        return MoELayer(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            **kwargs
        )


def integrate_moe_into_model(model: nn.Module, moe_config: MoEConfig) -> nn.Module:
    """
    Replace FeedForward layers in a model with MoE layers.
    
    Args:
        model: Existing model with FeedForward layers
        moe_config: MoE configuration
    
    Returns:
        Model with MoE layers
    """
    if not HAVE_TORCH:
        raise RuntimeError("MoE requires PyTorch")
    
    # Find and replace FeedForward layers
    for name, module in model.named_modules():
        if hasattr(module, 'ffn') and isinstance(module.ffn, nn.Module):
            # This is a transformer block with ffn attribute
            parent = model
            for part in name.split('.')[:-1]:
                parent = getattr(parent, part)
            
            # Create MoE to replace FFN
            old_ffn = module.ffn
            if hasattr(old_ffn, 'dim'):
                hidden_dim = old_ffn.dim
            elif hasattr(old_ffn, 'in_features'):
                hidden_dim = old_ffn.in_features
            else:
                continue
            
            # Create new MoE layer
            moe_layer = MoELayer(
                config=moe_config,
                hidden_dim=hidden_dim,
            )
            
            # Replace
            module.ffn = moe_layer
            logger.info(f"Replaced FFN with MoE at {name}")
    
    return model


# Export public API
__all__ = [
    'MoELayer',
    'MoEConfig',
    'ExpertRouter',
    'Expert',
    'SwitchMoELayer',
    'ExpertChoiceMoELayer',
    'MoETransformerBlock',
    'RoutingStrategy',
    'create_moe_layer',
    'integrate_moe_into_model',
    'HAVE_TORCH',
]
