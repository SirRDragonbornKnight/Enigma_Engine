"""
Sparse Attention for Enigma AI Engine

Memory-efficient attention patterns.

Features:
- Sliding window attention
- Block sparse attention
- Longformer-style global attention
- BigBird random + window + global
- Streaming attention

Usage:
    from enigma_engine.core.sparse_attention import SparseAttention, AttentionPattern
    
    attention = SparseAttention(
        pattern=AttentionPattern.SLIDING_WINDOW,
        window_size=256
    )
    
    output = attention(query, key, value)
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AttentionPattern(Enum):
    """Sparse attention patterns."""
    FULL = "full"  # Standard quadratic attention
    SLIDING_WINDOW = "sliding_window"  # Local window
    DILATED = "dilated"  # Dilated/strided
    BLOCK_SPARSE = "block_sparse"  # Fixed block patterns
    LONGFORMER = "longformer"  # Window + global tokens
    BIGBIRD = "bigbird"  # Window + random + global
    STREAMING = "streaming"  # Sink + window for streaming


@dataclass
class SparseConfig:
    """Configuration for sparse attention."""
    pattern: AttentionPattern = AttentionPattern.SLIDING_WINDOW
    window_size: int = 256
    dilate_rate: int = 1  # For dilated attention
    block_size: int = 64  # For block sparse
    global_tokens: int = 8  # Number of global attention tokens
    random_tokens: int = 64  # For BigBird random attention
    sink_tokens: int = 4  # For streaming attention


class SparseAttention(nn.Module):
    """Sparse attention with various patterns."""
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        config: Optional[SparseConfig] = None
    ):
        """
        Initialize sparse attention.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            config: Sparse attention configuration
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.config = config or SparseConfig()
        
        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Scale
        self.scale = self.head_dim ** -0.5
        
        logger.info(
            f"SparseAttention initialized: pattern={self.config.pattern.value}, "
            f"window_size={self.config.window_size}"
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply sparse attention.
        
        Args:
            query: Query tensor [batch, seq, hidden]
            key: Key tensor (default: query)
            value: Value tensor (default: query)
            attention_mask: Optional attention mask
            global_indices: Indices for global attention tokens
            
        Returns:
            Tuple of (output, attention_weights)
        """
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size, seq_len, _ = query.shape
        
        # Project
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply sparse pattern
        if self.config.pattern == AttentionPattern.FULL:
            output, weights = self._full_attention(q, k, v, attention_mask)
        elif self.config.pattern == AttentionPattern.SLIDING_WINDOW:
            output, weights = self._sliding_window_attention(q, k, v, attention_mask)
        elif self.config.pattern == AttentionPattern.DILATED:
            output, weights = self._dilated_attention(q, k, v, attention_mask)
        elif self.config.pattern == AttentionPattern.BLOCK_SPARSE:
            output, weights = self._block_sparse_attention(q, k, v, attention_mask)
        elif self.config.pattern == AttentionPattern.LONGFORMER:
            output, weights = self._longformer_attention(q, k, v, attention_mask, global_indices)
        elif self.config.pattern == AttentionPattern.BIGBIRD:
            output, weights = self._bigbird_attention(q, k, v, attention_mask, global_indices)
        elif self.config.pattern == AttentionPattern.STREAMING:
            output, weights = self._streaming_attention(q, k, v, attention_mask)
        else:
            output, weights = self._full_attention(q, k, v, attention_mask)
        
        # Reshape output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.o_proj(output)
        
        return output, weights
    
    def _full_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard full attention."""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        
        return output, weights
    
    def _sliding_window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sliding window attention."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        window = self.config.window_size
        
        # Create windowed attention mask
        window_mask = self._create_window_mask(seq_len, window, q.device)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply window mask
        scores = scores.masked_fill(window_mask == 0, float('-inf'))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        
        return output, weights
    
    def _create_window_mask(
        self,
        seq_len: int,
        window: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create sliding window mask."""
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        for i in range(seq_len):
            start = max(0, i - window // 2)
            end = min(seq_len, i + window // 2 + 1)
            mask[i, start:end] = 1
        
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
    
    def _dilated_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Dilated/strided attention."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        window = self.config.window_size
        dilate = self.config.dilate_rate
        
        # Create dilated mask
        dilate_mask = torch.zeros(seq_len, seq_len, device=q.device)
        
        for i in range(seq_len):
            # Local window
            for j in range(max(0, i - window // 2), min(seq_len, i + window // 2 + 1)):
                dilate_mask[i, j] = 1
            
            # Dilated positions
            for d in range(1, seq_len // dilate + 1):
                pos = i - d * dilate
                if pos >= 0:
                    dilate_mask[i, pos] = 1
                pos = i + d * dilate
                if pos < seq_len:
                    dilate_mask[i, pos] = 1
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(dilate_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        
        return output, weights
    
    def _block_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Block sparse attention."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        block_size = self.config.block_size
        
        # Pad sequence to block size
        pad_len = (block_size - seq_len % block_size) % block_size
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
        
        padded_len = q.shape[2]
        num_blocks = padded_len // block_size
        
        # Reshape to blocks
        q_blocks = q.view(batch_size, num_heads, num_blocks, block_size, head_dim)
        k_blocks = k.view(batch_size, num_heads, num_blocks, block_size, head_dim)
        v_blocks = v.view(batch_size, num_heads, num_blocks, block_size, head_dim)
        
        # Block-diagonal attention
        outputs = []
        for b in range(num_blocks):
            # Current block attends to current and previous block
            k_attend = [k_blocks[:, :, b, :, :]]
            v_attend = [v_blocks[:, :, b, :, :]]
            
            if b > 0:
                k_attend.insert(0, k_blocks[:, :, b - 1, :, :])
                v_attend.insert(0, v_blocks[:, :, b - 1, :, :])
            
            k_cat = torch.cat(k_attend, dim=2)
            v_cat = torch.cat(v_attend, dim=2)
            
            scores = torch.matmul(q_blocks[:, :, b, :, :], k_cat.transpose(-2, -1)) * self.scale
            weights = F.softmax(scores, dim=-1)
            out = torch.matmul(weights, v_cat)
            outputs.append(out)
        
        output = torch.stack(outputs, dim=2)
        output = output.view(batch_size, num_heads, padded_len, head_dim)
        
        # Remove padding
        output = output[:, :, :seq_len, :]
        
        return output, None
    
    def _longformer_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        global_indices: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Longformer-style attention with global tokens."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        window = self.config.window_size
        num_global = self.config.global_tokens
        
        # Default global indices: first N tokens
        if global_indices is None:
            global_indices = torch.arange(num_global, device=q.device)
        
        # Create combined mask
        combined_mask = torch.zeros(seq_len, seq_len, device=q.device)
        
        # Window attention
        for i in range(seq_len):
            start = max(0, i - window // 2)
            end = min(seq_len, i + window // 2 + 1)
            combined_mask[i, start:end] = 1
        
        # Global attention: global tokens attend to all, and all attend to global
        combined_mask[global_indices, :] = 1
        combined_mask[:, global_indices] = 1
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(combined_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        
        return output, weights
    
    def _bigbird_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        global_indices: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """BigBird attention: window + random + global."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        window = self.config.window_size
        num_random = self.config.random_tokens
        num_global = self.config.global_tokens
        
        # Create combined mask
        combined_mask = torch.zeros(seq_len, seq_len, device=q.device)
        
        # Window attention
        for i in range(seq_len):
            start = max(0, i - window // 2)
            end = min(seq_len, i + window // 2 + 1)
            combined_mask[i, start:end] = 1
        
        # Random attention
        for i in range(seq_len):
            random_indices = torch.randperm(seq_len, device=q.device)[:num_random]
            combined_mask[i, random_indices] = 1
        
        # Global attention
        if global_indices is None:
            global_indices = torch.arange(num_global, device=q.device)
        
        combined_mask[global_indices, :] = 1
        combined_mask[:, global_indices] = 1
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(combined_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        
        return output, weights
    
    def _streaming_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Streaming attention with attention sinks."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        window = self.config.window_size
        sink_tokens = self.config.sink_tokens
        
        # Create mask: sink tokens + recent window
        combined_mask = torch.zeros(seq_len, seq_len, device=q.device)
        
        for i in range(seq_len):
            # Sink tokens
            combined_mask[i, :sink_tokens] = 1
            
            # Recent window
            start = max(sink_tokens, i - window + 1)
            combined_mask[i, start:i + 1] = 1
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(combined_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        
        return output, weights


class EfficientAttention(nn.Module):
    """Memory-efficient attention using chunking."""
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        chunk_size: int = 512
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.chunk_size = chunk_size
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Memory-efficient attention via chunking.
        
        Args:
            x: Input tensor [batch, seq, hidden]
            attention_mask: Optional mask
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Chunk queries
        outputs = []
        for i in range(0, seq_len, self.chunk_size):
            end_i = min(i + self.chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i, :]
            
            # Compute attention for this chunk
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
            
            # Causal mask for current chunk
            chunk_len = end_i - i
            causal_mask = torch.triu(
                torch.ones(chunk_len, seq_len, device=x.device) * float('-inf'),
                diagonal=i + 1
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
            
            weights = F.softmax(scores, dim=-1)
            output = torch.matmul(weights, v)
            outputs.append(output)
        
        output = torch.cat(outputs, dim=2)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.o_proj(output)
