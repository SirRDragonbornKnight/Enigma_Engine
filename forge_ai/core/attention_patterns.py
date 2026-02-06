"""
Attention Patterns

Various attention patterns for efficient transformers:
sliding window, sparse, local-global, dilated.

FILE: forge_ai/core/attention_patterns.py
TYPE: Core/Model
MAIN CLASSES: AttentionPattern, SlidingWindowAttention, SparseAttention
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Attention pattern types."""
    FULL = "full"
    SLIDING_WINDOW = "sliding_window"
    SPARSE = "sparse"
    LOCAL_GLOBAL = "local_global"
    DILATED = "dilated"
    LONGFORMER = "longformer"
    BIGBIRD = "bigbird"


@dataclass
class AttentionConfig:
    """Attention configuration."""
    pattern_type: PatternType = PatternType.FULL
    
    # Dimensions
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    
    # Sliding window
    window_size: int = 256
    
    # Sparse
    sparsity: float = 0.9  # 90% sparse
    block_size: int = 64
    
    # Local-global
    num_global_tokens: int = 64
    local_window: int = 128
    
    # Dilated
    dilation_rates: list[int] = None  # Per head
    
    # General
    dropout: float = 0.1
    use_causal_mask: bool = True


if HAS_TORCH:
    
    class AttentionMask:
        """Generate attention masks for various patterns."""
        
        @staticmethod
        def full_mask(seq_len: int, causal: bool = True) -> torch.Tensor:
            """Full attention mask (optional causal)."""
            if causal:
                mask = torch.triu(
                    torch.ones(seq_len, seq_len),
                    diagonal=1
                ).bool()
                return ~mask
            return torch.ones(seq_len, seq_len).bool()
        
        @staticmethod
        def sliding_window_mask(
            seq_len: int,
            window_size: int,
            causal: bool = True
        ) -> torch.Tensor:
            """Sliding window attention mask."""
            mask = torch.zeros(seq_len, seq_len)
            
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2 + 1)
                
                if causal:
                    end = min(end, i + 1)
                    start = max(start, i - window_size + 1)
                
                mask[i, start:end] = 1
            
            return mask.bool()
        
        @staticmethod
        def sparse_block_mask(
            seq_len: int,
            block_size: int,
            sparsity: float = 0.5,
            causal: bool = True
        ) -> torch.Tensor:
            """Block-sparse attention mask."""
            num_blocks = (seq_len + block_size - 1) // block_size
            
            # Block-level connectivity
            block_mask = torch.rand(num_blocks, num_blocks) > sparsity
            
            # Expand to token level
            mask = block_mask.repeat_interleave(
                block_size, dim=0
            ).repeat_interleave(block_size, dim=1)
            
            # Trim to sequence length
            mask = mask[:seq_len, :seq_len]
            
            # Always attend to diagonal blocks
            for i in range(num_blocks):
                start = i * block_size
                end = min((i + 1) * block_size, seq_len)
                mask[start:end, start:end] = True
            
            if causal:
                causal_mask = torch.tril(torch.ones(seq_len, seq_len))
                mask = mask & causal_mask.bool()
            
            return mask
        
        @staticmethod
        def local_global_mask(
            seq_len: int,
            num_global: int,
            local_window: int,
            causal: bool = True
        ) -> torch.Tensor:
            """Local-global attention mask (Longformer style)."""
            mask = torch.zeros(seq_len, seq_len)
            
            # Global tokens attend to/from all positions
            mask[:num_global, :] = 1
            mask[:, :num_global] = 1
            
            # Local attention for non-global tokens
            for i in range(num_global, seq_len):
                start = max(num_global, i - local_window // 2)
                end = min(seq_len, i + local_window // 2 + 1)
                
                if causal:
                    end = min(end, i + 1)
                
                mask[i, start:end] = 1
            
            return mask.bool()
        
        @staticmethod
        def dilated_mask(
            seq_len: int,
            dilation: int,
            window_size: int,
            causal: bool = True
        ) -> torch.Tensor:
            """Dilated attention mask."""
            mask = torch.zeros(seq_len, seq_len)
            
            for i in range(seq_len):
                # Positions at dilated intervals
                for j in range(0, seq_len, dilation):
                    if abs(i - j) <= window_size * dilation:
                        if not causal or j <= i:
                            mask[i, j] = 1
            
            return mask.bool()
    
    
    class SlidingWindowAttention(nn.Module):
        """
        Sliding window attention.
        
        Each position attends only to positions within a fixed window.
        Reduces complexity from O(n^2) to O(n * w) where w is window size.
        """
        
        def __init__(self, config: AttentionConfig):
            super().__init__()
            self.config = config
            
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_heads
            self.head_dim = config.head_dim
            self.window_size = config.window_size
            
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)
            
            self.dropout = nn.Dropout(config.dropout)
            self.scale = self.head_dim ** -0.5
        
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor = None,
            position_ids: torch.Tensor = None
        ) -> torch.Tensor:
            batch_size, seq_len, _ = hidden_states.shape
            
            # Project Q, K, V
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            
            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Create sliding window mask
            window_mask = AttentionMask.sliding_window_mask(
                seq_len, self.window_size, self.config.use_causal_mask
            ).to(hidden_states.device)
            
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Apply masks
            scores = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            if attention_mask is not None:
                scores = scores + attention_mask
            
            # Softmax and dropout
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            output = torch.matmul(attn_weights, v)
            
            # Reshape and project
            output = output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.num_heads * self.head_dim
            )
            
            return self.o_proj(output)
    
    
    class SparseAttention(nn.Module):
        """
        Block-sparse attention.
        
        Attention is computed only within and between selected blocks.
        Reduces complexity for long sequences.
        """
        
        def __init__(self, config: AttentionConfig):
            super().__init__()
            self.config = config
            
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_heads
            self.head_dim = config.head_dim
            self.block_size = config.block_size
            self.sparsity = config.sparsity
            
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)
            
            self.dropout = nn.Dropout(config.dropout)
            self.scale = self.head_dim ** -0.5
        
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor = None
        ) -> torch.Tensor:
            batch_size, seq_len, _ = hidden_states.shape
            
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Create sparse mask
            sparse_mask = AttentionMask.sparse_block_mask(
                seq_len, self.block_size, self.sparsity, self.config.use_causal_mask
            ).to(hidden_states.device)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            scores = scores.masked_fill(~sparse_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            if attention_mask is not None:
                scores = scores + attention_mask
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output = torch.matmul(attn_weights, v)
            output = output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.num_heads * self.head_dim
            )
            
            return self.o_proj(output)
    
    
    class LocalGlobalAttention(nn.Module):
        """
        Local-global attention (Longformer style).
        
        Global tokens attend to all positions.
        Local tokens attend within a window plus global tokens.
        """
        
        def __init__(self, config: AttentionConfig):
            super().__init__()
            self.config = config
            
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_heads
            self.head_dim = config.head_dim
            self.num_global = config.num_global_tokens
            self.local_window = config.local_window
            
            # Separate projections for global and local
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            
            self.q_global = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            self.k_global = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            self.v_global = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)
            
            self.dropout = nn.Dropout(config.dropout)
            self.scale = self.head_dim ** -0.5
        
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor = None
        ) -> torch.Tensor:
            batch_size, seq_len, _ = hidden_states.shape
            
            # Split into global and local tokens
            global_states = hidden_states[:, :self.num_global]
            local_states = hidden_states[:, self.num_global:]
            
            # Global attention
            q_g = self.q_global(global_states).view(
                batch_size, self.num_global, self.num_heads, self.head_dim
            ).transpose(1, 2)
            k_g = self.k_global(hidden_states).view(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            v_g = self.v_global(hidden_states).view(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            
            global_scores = torch.matmul(q_g, k_g.transpose(-2, -1)) * self.scale
            global_weights = F.softmax(global_scores, dim=-1)
            global_output = torch.matmul(global_weights, v_g)
            
            # Local attention with local-global mask
            q_l = self.q_proj(hidden_states).view(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            k_l = self.k_proj(hidden_states).view(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            v_l = self.v_proj(hidden_states).view(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            
            local_mask = AttentionMask.local_global_mask(
                seq_len, self.num_global, self.local_window, self.config.use_causal_mask
            ).to(hidden_states.device)
            
            local_scores = torch.matmul(q_l, k_l.transpose(-2, -1)) * self.scale
            local_scores = local_scores.masked_fill(
                ~local_mask.unsqueeze(0).unsqueeze(0), float('-inf')
            )
            
            local_weights = F.softmax(local_scores, dim=-1)
            local_output = torch.matmul(local_weights, v_l)
            
            # Combine outputs
            output = local_output.clone()
            output[:, :, :self.num_global] = global_output
            
            output = output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.num_heads * self.head_dim
            )
            
            return self.o_proj(output)
    
    
    class DilatedAttention(nn.Module):
        """
        Dilated attention.
        
        Different heads use different dilation rates for
        multi-scale attention patterns.
        """
        
        def __init__(self, config: AttentionConfig):
            super().__init__()
            self.config = config
            
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_heads
            self.head_dim = config.head_dim
            self.window_size = config.window_size
            
            # Default dilation rates: exponential growth per head
            self.dilation_rates = config.dilation_rates or [
                2 ** (i % 4) for i in range(self.num_heads)
            ]
            
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)
            
            self.dropout = nn.Dropout(config.dropout)
            self.scale = self.head_dim ** -0.5
        
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor = None
        ) -> torch.Tensor:
            batch_size, seq_len, _ = hidden_states.shape
            
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention per head with different dilations
            outputs = []
            for h in range(self.num_heads):
                mask = AttentionMask.dilated_mask(
                    seq_len,
                    self.dilation_rates[h],
                    self.window_size,
                    self.config.use_causal_mask
                ).to(hidden_states.device)
                
                scores = torch.matmul(q[:, h:h+1], k[:, h:h+1].transpose(-2, -1)) * self.scale
                scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                
                weights = F.softmax(scores, dim=-1)
                weights = self.dropout(weights)
                
                out = torch.matmul(weights, v[:, h:h+1])
                outputs.append(out)
            
            output = torch.cat(outputs, dim=1)
            output = output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.num_heads * self.head_dim
            )
            
            return self.o_proj(output)
    
    
    def create_attention(config: AttentionConfig) -> nn.Module:
        """Factory function to create attention module."""
        if config.pattern_type == PatternType.SLIDING_WINDOW:
            return SlidingWindowAttention(config)
        elif config.pattern_type == PatternType.SPARSE:
            return SparseAttention(config)
        elif config.pattern_type == PatternType.LOCAL_GLOBAL:
            return LocalGlobalAttention(config)
        elif config.pattern_type == PatternType.DILATED:
            return DilatedAttention(config)
        else:
            # Full attention - use standard nn.MultiheadAttention
            return nn.MultiheadAttention(
                config.hidden_size,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True
            )

else:
    # Stubs when torch not available
    class AttentionMask:
        pass
    
    class SlidingWindowAttention:
        pass
    
    class SparseAttention:
        pass
    
    class LocalGlobalAttention:
        pass
    
    class DilatedAttention:
        pass
    
    def create_attention(config):
        raise ImportError("PyTorch required for attention patterns")
