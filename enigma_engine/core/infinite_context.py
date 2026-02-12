"""
Infinite Context Support

Handle arbitrarily long contexts through various techniques:
- Streaming/chunked processing
- Memory-augmented attention
- Landmark/retrieval-based context
- Ring attention for distributed contexts

FILE: enigma_engine/core/infinite_context.py
TYPE: Core/Context
MAIN CLASSES: InfiniteContextModel, StreamingContext, LandmarkAttention
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Configuration for infinite context handling."""
    # Chunk settings
    chunk_size: int = 2048
    overlap_size: int = 128
    
    # Memory settings
    memory_size: int = 512  # Number of memory tokens
    memory_layers: int = 4  # Layers with memory
    
    # Landmark settings
    num_landmarks: int = 64
    landmark_selection: str = "attention"  # "attention", "uniform", "learned"
    
    # Ring attention
    ring_buffer_size: int = 8192
    
    # Compression
    compress_ratio: int = 4
    compress_method: str = "mean"  # "mean", "max", "learned"


class MemoryBank(nn.Module):
    """
    External memory bank for attention.
    
    Stores compressed representations from previous context.
    """
    
    def __init__(
        self,
        hidden_size: int,
        memory_size: int,
        num_heads: int
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        
        # Persistent memory parameters
        self.memory_keys = nn.Parameter(
            torch.randn(memory_size, hidden_size) * 0.02
        )
        self.memory_values = nn.Parameter(
            torch.randn(memory_size, hidden_size) * 0.02
        )
        
        # Gating for memory access
        self.gate = nn.Linear(hidden_size, 1)
        
        # Memory update projection
        self.update_proj = nn.Linear(hidden_size, hidden_size)
    
    def read(
        self,
        query: torch.Tensor,  # [batch, seq, hidden]
        top_k: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory.
        
        Returns:
            memory_output: Retrieved memory values
            attention_weights: Memory attention weights
        """
        batch_size = query.shape[0]
        
        # Expand memory for batch
        keys = self.memory_keys.unsqueeze(0).expand(batch_size, -1, -1)
        values = self.memory_values.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute attention scores
        scale = math.sqrt(self.hidden_size)
        scores = torch.bmm(query, keys.transpose(1, 2)) / scale
        
        # Top-k selection for efficiency
        if top_k < self.memory_size:
            top_scores, top_idx = scores.topk(top_k, dim=-1)
            
            # Gather top values
            batch_idx = torch.arange(batch_size, device=query.device)
            batch_idx = batch_idx.view(-1, 1, 1).expand(-1, query.shape[1], top_k)
            
            top_values = values[batch_idx, top_idx]
            
            # Softmax over top-k
            attn_weights = F.softmax(top_scores, dim=-1)
            memory_output = torch.bmm(attn_weights, top_values)
        else:
            attn_weights = F.softmax(scores, dim=-1)
            memory_output = torch.bmm(attn_weights, values)
        
        # Apply gating
        gate = torch.sigmoid(self.gate(query))
        memory_output = gate * memory_output
        
        return memory_output, attn_weights
    
    def write(
        self,
        key_values: torch.Tensor,  # [batch, seq, hidden]
        importance: Optional[torch.Tensor] = None
    ) -> None:
        """
        Write to memory (update memory content).
        
        Uses exponential moving average for stable updates.
        """
        # Project new content
        new_content = self.update_proj(key_values.mean(dim=(0, 1)))
        
        # Update with momentum
        with torch.no_grad():
            momentum = 0.9
            
            # Select random memory slots to update
            num_update = min(32, self.memory_size)
            update_idx = torch.randperm(self.memory_size)[:num_update]
            
            self.memory_keys.data[update_idx] = (
                momentum * self.memory_keys.data[update_idx] +
                (1 - momentum) * new_content.unsqueeze(0).expand(num_update, -1)
            )
            
            self.memory_values.data[update_idx] = (
                momentum * self.memory_values.data[update_idx] +
                (1 - momentum) * new_content.unsqueeze(0).expand(num_update, -1)
            )


class LandmarkAttention(nn.Module):
    """
    Landmark-based attention for long contexts.
    
    Selects important tokens as "landmarks" that all tokens can attend to.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_landmarks: int,
        selection_method: str = "attention"
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_landmarks = num_landmarks
        self.selection_method = selection_method
        
        # Learned landmark selector
        if selection_method == "learned":
            self.landmark_scorer = nn.Linear(hidden_size, 1)
        
        # Projection for landmark attention
        self.landmark_q = nn.Linear(hidden_size, hidden_size)
        self.landmark_k = nn.Linear(hidden_size, hidden_size)
        self.landmark_v = nn.Linear(hidden_size, hidden_size)
        self.landmark_out = nn.Linear(hidden_size, hidden_size)
    
    def select_landmarks(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select landmark positions.
        
        Returns:
            landmark_hidden: Hidden states at landmark positions
            landmark_idx: Indices of selected landmarks
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        if self.selection_method == "uniform":
            # Uniform spacing
            step = max(1, seq_len // self.num_landmarks)
            landmark_idx = torch.arange(0, seq_len, step, device=hidden_states.device)
            landmark_idx = landmark_idx[:self.num_landmarks]
            
        elif self.selection_method == "attention":
            # Select based on attention received
            if attention_weights is not None:
                # Sum attention received by each token
                importance = attention_weights.sum(dim=(1, 2))  # [batch, seq]
            else:
                # Fallback to norm-based importance
                importance = hidden_states.norm(dim=-1)
            
            _, landmark_idx = importance.topk(self.num_landmarks, dim=-1)
            landmark_idx = landmark_idx.sort(dim=-1).values
            
        elif self.selection_method == "learned":
            # Learned scoring
            scores = self.landmark_scorer(hidden_states).squeeze(-1)
            _, landmark_idx = scores.topk(self.num_landmarks, dim=-1)
            landmark_idx = landmark_idx.sort(dim=-1).values
        
        # Gather landmark hidden states
        batch_idx = torch.arange(batch_size, device=hidden_states.device)
        batch_idx = batch_idx.unsqueeze(1).expand(-1, self.num_landmarks)
        
        landmark_hidden = hidden_states[batch_idx, landmark_idx]
        
        return landmark_hidden, landmark_idx
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply landmark attention.
        
        All tokens attend to landmarks + local window.
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Select landmarks
        landmark_hidden, landmark_idx = self.select_landmarks(
            hidden_states, attention_weights
        )
        
        # Project queries, keys, values
        q = self.landmark_q(hidden_states)
        k = self.landmark_k(landmark_hidden)
        v = self.landmark_v(landmark_hidden)
        
        # Reshape for multi-head attention
        head_dim = self.hidden_size // self.num_heads
        
        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, self.num_landmarks, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, self.num_landmarks, self.num_heads, head_dim).transpose(1, 2)
        
        # Attention to landmarks
        scale = math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.landmark_out(context)


class ContextCompressor(nn.Module):
    """
    Compress context representations for efficiency.
    
    Reduces sequence length while preserving important information.
    """
    
    def __init__(
        self,
        hidden_size: int,
        compress_ratio: int = 4,
        method: str = "mean"
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.compress_ratio = compress_ratio
        self.method = method
        
        if method == "learned":
            self.compress_proj = nn.Linear(
                hidden_size * compress_ratio,
                hidden_size
            )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compress hidden states."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Pad to multiple of compress_ratio
        pad_len = (self.compress_ratio - seq_len % self.compress_ratio) % self.compress_ratio
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
        
        new_seq_len = (seq_len + pad_len) // self.compress_ratio
        
        # Reshape for compression
        reshaped = hidden_states.view(
            batch_size,
            new_seq_len,
            self.compress_ratio,
            hidden_size
        )
        
        if self.method == "mean":
            compressed = reshaped.mean(dim=2)
        elif self.method == "max":
            compressed = reshaped.max(dim=2).values
        elif self.method == "learned":
            flattened = reshaped.view(batch_size, new_seq_len, -1)
            compressed = self.compress_proj(flattened)
        
        return compressed


class StreamingContext:
    """
    Process arbitrarily long contexts in chunks.
    
    Maintains state between chunks for continuity.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ContextConfig = None
    ) -> None:
        self.model = model
        self.config = config or ContextConfig()
        
        # State between chunks
        self._kv_cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        self._memory_bank: Optional[MemoryBank] = None
        self._compressed_context: Optional[torch.Tensor] = None
        
        # Statistics
        self._total_tokens = 0
        self._chunks_processed = 0
    
    def reset(self) -> None:
        """Reset streaming state."""
        self._kv_cache = None
        self._compressed_context = None
        self._total_tokens = 0
        self._chunks_processed = 0
    
    def process_chunk(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process a chunk of input.
        
        Returns:
            Output hidden states for this chunk
        """
        device = input_ids.device
        batch_size, chunk_len = input_ids.shape
        
        # Get overlap from previous chunk
        if self._compressed_context is not None:
            # Prepend compressed context
            # Model should be modified to accept prefix_context
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                prefix_context=self._compressed_context,
                past_key_values=self._kv_cache
            )
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=self._kv_cache
            )
        
        # Extract outputs
        if hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs
        
        # Update KV cache (keep only recent)
        if hasattr(outputs, 'past_key_values'):
            self._kv_cache = outputs.past_key_values
            
            # Trim to max size
            if self._kv_cache is not None:
                max_cache = self.config.ring_buffer_size
                trimmed_cache = []
                for k, v in self._kv_cache:
                    if k.shape[2] > max_cache:
                        k = k[:, :, -max_cache:]
                        v = v[:, :, -max_cache:]
                    trimmed_cache.append((k, v))
                self._kv_cache = trimmed_cache
        
        # Compress processed chunk for memory
        compressor = ContextCompressor(
            hidden_states.shape[-1],
            self.config.compress_ratio,
            self.config.compress_method
        )
        
        compressed = compressor(hidden_states)
        
        if self._compressed_context is None:
            self._compressed_context = compressed
        else:
            # Concatenate and trim
            self._compressed_context = torch.cat([
                self._compressed_context, compressed
            ], dim=1)
            
            max_compressed = self.config.memory_size
            if self._compressed_context.shape[1] > max_compressed:
                self._compressed_context = self._compressed_context[:, -max_compressed:]
        
        # Update statistics
        self._total_tokens += chunk_len
        self._chunks_processed += 1
        
        return hidden_states
    
    def process_full(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process full input by chunking automatically.
        
        Returns:
            Complete output hidden states
        """
        self.reset()
        
        total_len = input_ids.shape[1]
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap_size
        
        all_outputs = []
        
        start = 0
        while start < total_len:
            end = min(start + chunk_size, total_len)
            
            chunk_ids = input_ids[:, start:end]
            chunk_mask = attention_mask[:, start:end] if attention_mask is not None else None
            
            outputs = self.process_chunk(chunk_ids, chunk_mask)
            
            # Remove overlap from output (except last chunk)
            if start > 0:
                outputs = outputs[:, overlap:]
            
            all_outputs.append(outputs)
            
            # Next chunk starts with overlap
            start = end - overlap
        
        # Concatenate all outputs
        return torch.cat(all_outputs, dim=1)


class RingAttention(nn.Module):
    """
    Ring attention for distributed long context processing.
    
    Splits sequence across devices and passes KV in a ring pattern.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ring_size: int = 2  # Number of devices in ring
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.ring_size = ring_size
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        local_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Ring attention forward pass.
        
        Note: In practice, this would use distributed primitives.
        This is a simplified single-device simulation.
        """
        batch_size, seq_len, _ = hidden_states.shape
        head_dim = self.hidden_size // self.num_heads
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        
        # Simulate ring pattern: each "device" processes its chunk
        chunk_size = seq_len // self.ring_size
        all_outputs = []
        
        for i in range(self.ring_size):
            start = i * chunk_size
            end = start + chunk_size if i < self.ring_size - 1 else seq_len
            
            q_chunk = q[:, :, start:end]
            
            # Attend to all K, V (in ring, these would come from other devices)
            scale = math.sqrt(head_dim)
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) / scale
            
            # Causal mask
            chunk_positions = torch.arange(start, end, device=hidden_states.device)
            all_positions = torch.arange(seq_len, device=hidden_states.device)
            causal_mask = chunk_positions.unsqueeze(1) < all_positions.unsqueeze(0)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            chunk_output = torch.matmul(attn_weights, v)
            all_outputs.append(chunk_output)
        
        # Concatenate chunks
        output = torch.cat(all_outputs, dim=2)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.out_proj(output)


class InfiniteContextModel(nn.Module):
    """
    Wrapper that adds infinite context capabilities to any transformer.
    
    Combines memory, landmarks, and streaming for handling any length.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: ContextConfig = None
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config or ContextConfig()
        
        # Detect hidden size from base model
        if hasattr(base_model, 'config'):
            hidden_size = getattr(base_model.config, 'hidden_size', 768)
            num_heads = getattr(base_model.config, 'num_attention_heads', 12)
        else:
            hidden_size = 768
            num_heads = 12
        
        # Memory bank
        self.memory_bank = MemoryBank(
            hidden_size,
            self.config.memory_size,
            num_heads
        )
        
        # Landmark attention
        self.landmark_attn = LandmarkAttention(
            hidden_size,
            num_heads,
            self.config.num_landmarks,
            self.config.landmark_selection
        )
        
        # Streaming context processor
        self.streaming = StreamingContext(base_model, config)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_size * 3, hidden_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_memory: bool = True,
        use_landmarks: bool = True,
        stream: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with infinite context support.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            use_memory: Whether to use memory bank
            use_landmarks: Whether to use landmark attention
            stream: Whether to use streaming mode
        """
        # Process through base model (optionally streaming)
        if stream:
            hidden_states = self.streaming.process_full(input_ids, attention_mask)
        else:
            outputs = self.base_model(input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs
        
        # Memory augmentation
        if use_memory:
            memory_output, _ = self.memory_bank.read(hidden_states)
            self.memory_bank.write(hidden_states)
        else:
            memory_output = torch.zeros_like(hidden_states)
        
        # Landmark attention
        if use_landmarks:
            landmark_output = self.landmark_attn(hidden_states)
        else:
            landmark_output = torch.zeros_like(hidden_states)
        
        # Fuse all sources
        fused = self.fusion(torch.cat([
            hidden_states,
            memory_output,
            landmark_output
        ], dim=-1))
        
        return fused


__all__ = [
    'InfiniteContextModel',
    'StreamingContext',
    'LandmarkAttention',
    'MemoryBank',
    'ContextCompressor',
    'RingAttention',
    'ContextConfig'
]
