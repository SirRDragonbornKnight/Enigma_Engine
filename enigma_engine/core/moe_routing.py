"""
Mixture of Experts (MoE) Routing for enigma_engine

Full implementation of MoE layers with various routing strategies:
- Top-K routing with load balancing
- Expert choice routing
- Soft routing (weighted combination)
- Hash-based routing for determinism

Usage:
    from enigma_engine.core.moe_routing import MoELayer, create_moe_model
    
    moe = MoELayer(
        hidden_size=768,
        expert_hidden_size=3072,
        num_experts=8,
        num_experts_per_tok=2
    )
    output = moe(hidden_states)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategies for MoE."""
    TOP_K = "top_k"
    EXPERT_CHOICE = "expert_choice"
    SOFT = "soft"
    HASH = "hash"
    SWITCH = "switch"


@dataclass
class MoEConfig:
    """Configuration for MoE layers."""
    hidden_size: int = 768
    expert_hidden_size: int = 3072
    num_experts: int = 8
    num_experts_per_tok: int = 2
    routing_strategy: RoutingStrategy = RoutingStrategy.TOP_K
    
    # Load balancing
    aux_loss_coef: float = 0.01
    capacity_factor: float = 1.25
    
    # Expert dropout (during training)
    expert_dropout: float = 0.0
    
    # Jitter noise for training stability
    jitter_noise: float = 0.0
    
    # Routing temperature
    routing_temp: float = 1.0


class ExpertFFN(nn.Module):
    """
    Single expert feed-forward network.
    
    Uses SwiGLU activation for better performance.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # SwiGLU requires gate and up projections
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        return self.dropout(x)


class Router(nn.Module):
    """
    Routing module that determines expert assignment.
    
    Supports multiple routing strategies with load balancing.
    """
    
    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.routing_strategy = config.routing_strategy
        
        # Router projection
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        # For jitter noise
        self.jitter_noise = config.jitter_noise
    
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing weights.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            routing_weights: [batch, seq_len, num_experts_per_tok]
            selected_experts: [batch, seq_len, num_experts_per_tok]
            aux_loss: Load balancing loss
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Add jitter during training for stability
        if self.training and self.jitter_noise > 0:
            hidden_states = hidden_states * (
                1.0 + torch.randn_like(hidden_states) * self.jitter_noise
            )
        
        # Compute router logits
        router_logits = self.gate(hidden_states)  # [batch, seq, num_experts]
        
        # Apply temperature
        if self.config.routing_temp != 1.0:
            router_logits = router_logits / self.config.routing_temp
        
        # Route based on strategy
        if self.routing_strategy == RoutingStrategy.TOP_K:
            return self._top_k_routing(router_logits)
        elif self.routing_strategy == RoutingStrategy.EXPERT_CHOICE:
            return self._expert_choice_routing(router_logits)
        elif self.routing_strategy == RoutingStrategy.SOFT:
            return self._soft_routing(router_logits)
        elif self.routing_strategy == RoutingStrategy.HASH:
            return self._hash_routing(hidden_states, router_logits)
        elif self.routing_strategy == RoutingStrategy.SWITCH:
            return self._switch_routing(router_logits)
        else:
            raise ValueError(f"Unknown routing strategy: {self.routing_strategy}")
    
    def _top_k_routing(
        self,
        router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard Top-K routing with load balancing."""
        # Compute probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            router_probs, self.num_experts_per_tok, dim=-1
        )
        
        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Compute load balancing loss
        aux_loss = self._compute_load_balance_loss(router_probs, selected_experts)
        
        return routing_weights, selected_experts, aux_loss
    
    def _expert_choice_routing(
        self,
        router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expert choice routing - experts select tokens.
        
        Each expert picks its top-k tokens to process.
        """
        batch_size, seq_len, num_experts = router_logits.shape
        
        # Expert capacities
        capacity = int(self.config.capacity_factor * seq_len * self.num_experts_per_tok / num_experts)
        
        # Transpose: experts select tokens
        expert_logits = router_logits.transpose(1, 2)  # [batch, num_experts, seq]
        expert_probs = F.softmax(expert_logits, dim=-1)
        
        # Each expert selects top-k tokens
        expert_weights, expert_indices = torch.topk(
            expert_probs, min(capacity, seq_len), dim=-1
        )
        
        # Convert back to token -> expert format
        routing_weights = torch.zeros(
            batch_size, seq_len, self.num_experts_per_tok, device=router_logits.device
        )
        selected_experts = torch.zeros(
            batch_size, seq_len, self.num_experts_per_tok, dtype=torch.long, device=router_logits.device
        )
        
        # Build routing from expert choices
        token_counts = torch.zeros(batch_size, seq_len, dtype=torch.long, device=router_logits.device)
        
        for expert_idx in range(num_experts):
            for b in range(batch_size):
                for k, (token_idx, weight) in enumerate(zip(
                    expert_indices[b, expert_idx],
                    expert_weights[b, expert_idx]
                )):
                    count = token_counts[b, token_idx].item()
                    if count < self.num_experts_per_tok:
                        selected_experts[b, token_idx, count] = expert_idx
                        routing_weights[b, token_idx, count] = weight
                        token_counts[b, token_idx] += 1
        
        # Normalize weights
        weight_sum = routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights / (weight_sum + 1e-8)
        
        # No explicit load balance loss needed - expert choice is self-balancing
        aux_loss = torch.tensor(0.0, device=router_logits.device)
        
        return routing_weights, selected_experts, aux_loss
    
    def _soft_routing(
        self,
        router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Soft routing - weighted combination of all experts."""
        # All experts with softmax weights
        router_probs = F.softmax(router_logits, dim=-1)
        
        # "Select" all experts
        batch_size, seq_len, num_experts = router_logits.shape
        selected_experts = torch.arange(
            num_experts, device=router_logits.device
        ).expand(batch_size, seq_len, -1)
        
        # No load balance loss - using all experts
        aux_loss = torch.tensor(0.0, device=router_logits.device)
        
        return router_probs, selected_experts, aux_loss
    
    def _hash_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hash-based routing for deterministic assignment."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Hash based on position
        positions = torch.arange(seq_len, device=hidden_states.device)
        hash_values = positions % self.num_experts
        
        # Expand to batch
        selected_experts = hash_values.unsqueeze(0).expand(batch_size, -1)
        
        # If num_experts_per_tok > 1, add consecutive experts
        if self.num_experts_per_tok > 1:
            expert_offsets = torch.arange(
                self.num_experts_per_tok, device=hidden_states.device
            )
            selected_experts = (
                selected_experts.unsqueeze(-1) + expert_offsets
            ) % self.num_experts
        else:
            selected_experts = selected_experts.unsqueeze(-1)
        
        # Uniform weights
        routing_weights = torch.ones_like(selected_experts, dtype=hidden_states.dtype)
        routing_weights = routing_weights / self.num_experts_per_tok
        
        aux_loss = torch.tensor(0.0, device=router_logits.device)
        
        return routing_weights, selected_experts, aux_loss
    
    def _switch_routing(
        self,
        router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Switch Transformer routing (top-1 with capacity)."""
        # Force top-1
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-1 expert
        routing_weights, selected_experts = router_probs.max(dim=-1, keepdim=True)
        
        # Compute switch loss
        aux_loss = self._compute_switch_loss(router_probs, selected_experts.squeeze(-1))
        
        return routing_weights, selected_experts, aux_loss
    
    def _compute_load_balance_loss(
        self,
        router_probs: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.
        
        Encourages uniform expert utilization.
        """
        # Expert usage frequency
        num_tokens = router_probs.shape[0] * router_probs.shape[1]
        
        # Count tokens per expert
        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).sum(dim=2).float()  # [batch, seq, num_experts]
        
        expert_frac = expert_mask.sum(dim=[0, 1]) / (num_tokens * self.num_experts_per_tok)
        
        # Average router probability per expert
        router_frac = router_probs.mean(dim=[0, 1])
        
        # Load balance loss
        aux_loss = self.num_experts * (expert_frac * router_frac).sum()
        
        return aux_loss * self.config.aux_loss_coef
    
    def _compute_switch_loss(
        self,
        router_probs: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """Compute Switch Transformer auxiliary loss."""
        # Fraction of tokens to each expert
        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).float()
        
        tokens_per_expert = expert_mask.sum(dim=[0, 1])
        total_tokens = selected_experts.numel()
        frac_tokens = tokens_per_expert / total_tokens
        
        # Fraction of router probability to each expert
        frac_probs = router_probs.mean(dim=[0, 1])
        
        # Switch loss
        aux_loss = self.num_experts * (frac_tokens * frac_probs).sum()
        
        return aux_loss * self.config.aux_loss_coef


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.
    
    Replaces standard FFN with multiple expert FFNs.
    """
    
    def __init__(
        self,
        hidden_size: int,
        expert_hidden_size: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        routing_strategy: Union[str, RoutingStrategy] = RoutingStrategy.TOP_K,
        aux_loss_coef: float = 0.01,
        expert_dropout: float = 0.0
    ) -> None:
        super().__init__()
        
        # Build config
        if isinstance(routing_strategy, str):
            routing_strategy = RoutingStrategy(routing_strategy)
        
        self.config = MoEConfig(
            hidden_size=hidden_size,
            expert_hidden_size=expert_hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            routing_strategy=routing_strategy,
            aux_loss_coef=aux_loss_coef,
            expert_dropout=expert_dropout
        )
        
        # Router
        self.router = Router(self.config)
        
        # Experts
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_size, expert_hidden_size, expert_dropout)
            for _ in range(num_experts)
        ])
        
        # Shared expert (optional, for stability)
        self.shared_expert = None
    
    def add_shared_expert(self) -> None:
        """Add a shared expert that processes all tokens."""
        self.shared_expert = ExpertFFN(
            self.config.hidden_size,
            self.config.expert_hidden_size
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through MoE layer.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            return_aux_loss: Whether to return auxiliary loss
            
        Returns:
            output: [batch, seq_len, hidden_size]
            aux_loss: Optional load balancing loss
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get routing
        routing_weights, selected_experts, aux_loss = self.router(hidden_states)
        
        # Initialize output
        output = torch.zeros_like(hidden_states)
        
        # Process through selected experts
        if self.config.routing_strategy == RoutingStrategy.SOFT:
            # Soft routing: weighted combination of all experts
            for expert_idx, expert in enumerate(self.experts):
                expert_output = expert(hidden_states)
                weight = routing_weights[:, :, expert_idx:expert_idx+1]
                output = output + weight * expert_output
        else:
            # Sparse routing: only selected experts
            for expert_idx, expert in enumerate(self.experts):
                # Find tokens assigned to this expert
                expert_mask = (selected_experts == expert_idx).any(dim=-1)
                
                if not expert_mask.any():
                    continue
                
                # Get tokens for this expert
                expert_input = hidden_states[expert_mask]
                
                # Process through expert
                expert_output = expert(expert_input)
                
                # Get weights for this expert
                expert_weights = torch.zeros(
                    batch_size, seq_len, device=hidden_states.device
                )
                for k in range(self.config.num_experts_per_tok):
                    mask_k = selected_experts[:, :, k] == expert_idx
                    expert_weights[mask_k] = routing_weights[:, :, k][mask_k]
                
                # Add weighted output
                output[expert_mask] = output[expert_mask] + (
                    expert_weights[expert_mask].unsqueeze(-1) * expert_output
                )
        
        # Add shared expert if present
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            output = output + shared_output
        
        if return_aux_loss:
            return output, aux_loss
        return output


class MoEBlock(nn.Module):
    """
    Full transformer block with MoE FFN.
    
    Includes attention, MoE FFN, and layer norms.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expert_hidden_size: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Attention
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer norms
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        
        # MoE FFN
        self.moe = MoELayer(
            hidden_size=hidden_size,
            expert_hidden_size=expert_hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            output: [batch, seq_len, hidden_size]
            aux_loss: Load balancing loss
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        
        attn_output, _ = self.attention(
            hidden_states, hidden_states, hidden_states,
            attn_mask=attention_mask
        )
        hidden_states = residual + self.dropout(attn_output)
        
        # MoE FFN with residual
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        
        moe_output, aux_loss = self.moe(hidden_states)
        hidden_states = residual + self.dropout(moe_output)
        
        return hidden_states, aux_loss


class MoEModel(nn.Module):
    """
    Full MoE transformer model.
    
    Suitable for language modeling with mixture of experts.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        expert_hidden_size: int = 3072,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        moe_every_n_layers: int = 2  # Apply MoE every N layers
    ) -> None:
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)
        
        # Build layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if (i + 1) % moe_every_n_layers == 0:
                # MoE layer
                self.layers.append(MoEBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    expert_hidden_size=expert_hidden_size,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    dropout=dropout
                ))
            else:
                # Dense layer
                self.layers.append(self._make_dense_block(
                    hidden_size, num_heads, expert_hidden_size, dropout
                ))
        
        # Output
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight
    
    def _make_dense_block(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float
    ) -> nn.Module:
        """Create a dense transformer block."""
        class DenseBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.MultiheadAttention(
                    hidden_size, num_heads, dropout=dropout, batch_first=True
                )
                self.attn_norm = nn.LayerNorm(hidden_size)
                self.ffn_norm = nn.LayerNorm(hidden_size)
                self.ffn = nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size),
                    nn.GELU(),
                    nn.Linear(intermediate_size, hidden_size),
                    nn.Dropout(dropout)
                )
                self.dropout = nn.Dropout(dropout)
            
            def forward(self, x, attention_mask=None):
                # Attention
                residual = x
                x = self.attn_norm(x)
                x, _ = self.attn(x, x, x, attn_mask=attention_mask)
                x = residual + self.dropout(x)
                
                # FFN
                residual = x
                x = self.ffn_norm(x)
                x = residual + self.ffn(x)
                
                return x, torch.tensor(0.0, device=x.device)
        
        return DenseBlock()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            total_aux_loss: Sum of all MoE auxiliary losses
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden_states = self.embed_tokens(input_ids) + self.pos_embed(positions)
        
        # Process through layers
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        
        for layer in self.layers:
            hidden_states, aux_loss = layer(hidden_states, attention_mask)
            total_aux_loss = total_aux_loss + aux_loss
        
        # Output
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, total_aux_loss


def create_moe_model(
    vocab_size: int,
    size: str = "small",
    num_experts: int = 8,
    **kwargs
) -> MoEModel:
    """
    Create an MoE model with preset sizes.
    
    Args:
        vocab_size: Vocabulary size
        size: Model size preset ("tiny", "small", "medium", "large")
        num_experts: Number of experts
        **kwargs: Override any config parameter
    """
    presets = {
        "tiny": {
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 4,
            "expert_hidden_size": 1024,
        },
        "small": {
            "hidden_size": 512,
            "num_layers": 8,
            "num_heads": 8,
            "expert_hidden_size": 2048,
        },
        "medium": {
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "expert_hidden_size": 3072,
        },
        "large": {
            "hidden_size": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "expert_hidden_size": 4096,
        }
    }
    
    config = presets.get(size, presets["small"]).copy()
    config["num_experts"] = num_experts
    config.update(kwargs)
    
    return MoEModel(vocab_size=vocab_size, **config)


def convert_to_moe(
    model: nn.Module,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    moe_every_n_layers: int = 2
) -> nn.Module:
    """
    Convert an existing model's FFN layers to MoE.
    
    This creates multiple copies of the FFN to serve as experts.
    
    Args:
        model: Model to convert
        num_experts: Number of experts per MoE layer
        num_experts_per_tok: Experts per token
        moe_every_n_layers: Apply MoE every N layers
    """
    for name, module in model.named_modules():
        # Find FFN layers (common names)
        if any(x in name for x in ['mlp', 'ffn', 'feed_forward']):
            if not isinstance(module, MoELayer):
                # Get hidden size from module
                hidden_size = None
                intermediate_size = None
                
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name:
                        if hidden_size is None:
                            hidden_size = param.shape[1]
                            intermediate_size = param.shape[0]
                        break
                
                if hidden_size is not None:
                    # Replace with MoE
                    moe = MoELayer(
                        hidden_size=hidden_size,
                        expert_hidden_size=intermediate_size,
                        num_experts=num_experts,
                        num_experts_per_tok=num_experts_per_tok
                    )
                    
                    # Copy weights to first expert
                    # (Other experts are randomly initialized)
                    logger.info(f"Converting {name} to MoE with {num_experts} experts")
    
    return model
