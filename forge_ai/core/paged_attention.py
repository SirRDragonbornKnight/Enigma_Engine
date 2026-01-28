"""
================================================================================
PagedAttention - Memory-Efficient KV-Cache Management
================================================================================

Implements vLLM-style PagedAttention for efficient memory management.
Instead of pre-allocating max_seq_len for every request, allocates
fixed-size "pages" on demand.

BENEFITS:
    - 2-4x higher throughput
    - 3x more concurrent users with same GPU memory
    - Near-zero memory waste

📍 FILE: forge_ai/core/paged_attention.py
🏷️ TYPE: Memory Management System

USAGE:
    from forge_ai.core.paged_attention import PagedKVCache, PagedAttention
    
    # Create paged cache
    cache = PagedKVCache(
        num_layers=12,
        num_heads=8,
        head_dim=64,
        page_size=16,
        max_pages=1000
    )
    
    # Allocate pages for a sequence
    cache.allocate(seq_id=0, num_tokens=100)
    
    # Write to cache
    cache.write(seq_id=0, layer=0, keys=k, values=v)
    
    # Read from cache
    k, v = cache.read(seq_id=0, layer=0)
    
    # Free when done
    cache.free(seq_id=0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# Page Table Entry
# =============================================================================

@dataclass
class PageTableEntry:
    """Entry in the page table mapping logical to physical pages."""
    physical_page_id: int
    ref_count: int = 1
    is_copy_on_write: bool = False


# =============================================================================
# Block/Page Manager
# =============================================================================

class BlockManager:
    """
    Manages physical memory blocks (pages).
    
    Like an OS memory manager but for KV-cache.
    """
    
    def __init__(
        self,
        num_blocks: int,
        block_size: int,  # tokens per block
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the block manager.
        
        Args:
            num_blocks: Total number of physical blocks
            block_size: Tokens per block (page size)
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
            device: Device to allocate on
            dtype: Data type for cache
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        
        # Pre-allocate physical blocks for all layers
        # Shape: [num_blocks, block_size, num_heads, head_dim]
        self.key_cache = torch.zeros(
            (num_layers, num_blocks, block_size, num_heads, head_dim),
            dtype=dtype, device=device
        )
        self.value_cache = torch.zeros(
            (num_layers, num_blocks, block_size, num_heads, head_dim),
            dtype=dtype, device=device
        )
        
        # Free block list (stack for O(1) alloc/free)
        self.free_blocks = list(range(num_blocks))
        
        # Sequence to block mapping
        # seq_id -> list of physical block indices
        self.seq_blocks: Dict[int, List[int]] = {}
        
        # Track tokens written per sequence
        self.seq_lengths: Dict[int, int] = {}
        
        logger.info(f"[PagedAttn] Initialized {num_blocks} blocks x {block_size} tokens")
        logger.info(f"[PagedAttn] Total cache: {self._get_cache_size_mb():.1f} MB")
    
    def _get_cache_size_mb(self) -> float:
        """Get total cache size in MB."""
        bytes_per_element = 2 if self.dtype == torch.float16 else 4
        total_elements = (
            2 *  # key + value
            self.num_layers *
            self.num_blocks *
            self.block_size *
            self.num_heads *
            self.head_dim
        )
        return (total_elements * bytes_per_element) / (1024 * 1024)
    
    def can_allocate(self, num_tokens: int) -> bool:
        """Check if we can allocate blocks for num_tokens."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        return len(self.free_blocks) >= num_blocks_needed
    
    def allocate(self, seq_id: int, num_tokens: int) -> bool:
        """
        Allocate blocks for a sequence.
        
        Args:
            seq_id: Sequence identifier
            num_tokens: Number of tokens to allocate for
            
        Returns:
            True if allocation succeeded
        """
        if seq_id in self.seq_blocks:
            # Extend existing allocation
            return self._extend_allocation(seq_id, num_tokens)
        
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        if len(self.free_blocks) < num_blocks_needed:
            logger.warning(f"[PagedAttn] Cannot allocate {num_blocks_needed} blocks, "
                          f"only {len(self.free_blocks)} free")
            return False
        
        # Allocate blocks
        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            allocated.append(block_id)
        
        self.seq_blocks[seq_id] = allocated
        self.seq_lengths[seq_id] = 0  # No tokens written yet
        
        return True
    
    def _extend_allocation(self, seq_id: int, new_total_tokens: int) -> bool:
        """Extend allocation for existing sequence."""
        current_blocks = len(self.seq_blocks[seq_id])
        current_capacity = current_blocks * self.block_size
        
        if new_total_tokens <= current_capacity:
            return True  # Already have enough
        
        # Need more blocks
        additional_tokens = new_total_tokens - current_capacity
        additional_blocks = (additional_tokens + self.block_size - 1) // self.block_size
        
        if len(self.free_blocks) < additional_blocks:
            return False
        
        for _ in range(additional_blocks):
            block_id = self.free_blocks.pop()
            self.seq_blocks[seq_id].append(block_id)
        
        return True
    
    def free(self, seq_id: int):
        """Free all blocks for a sequence."""
        if seq_id not in self.seq_blocks:
            return
        
        # Return blocks to free list
        for block_id in self.seq_blocks[seq_id]:
            self.free_blocks.append(block_id)
        
        del self.seq_blocks[seq_id]
        if seq_id in self.seq_lengths:
            del self.seq_lengths[seq_id]
    
    def write(
        self,
        seq_id: int,
        layer: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ):
        """
        Write keys and values to cache.
        
        Args:
            seq_id: Sequence ID
            layer: Layer index
            keys: Key tensor [batch, seq_len, heads, dim] or [seq_len, heads, dim]
            values: Value tensor (same shape as keys)
            positions: Token positions (if not sequential)
        """
        if seq_id not in self.seq_blocks:
            raise ValueError(f"Sequence {seq_id} not allocated")
        
        # Handle batch dimension
        if keys.dim() == 4:
            keys = keys.squeeze(0)
            values = values.squeeze(0)
        
        seq_len = keys.size(0)
        blocks = self.seq_blocks[seq_id]
        
        # Determine positions
        if positions is None:
            start_pos = self.seq_lengths[seq_id]
            positions = torch.arange(start_pos, start_pos + seq_len, device=keys.device)
        
        # Write to appropriate blocks
        for i, pos in enumerate(positions):
            pos = pos.item() if hasattr(pos, 'item') else pos
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            
            if block_idx >= len(blocks):
                # Need to extend
                if not self._extend_allocation(seq_id, pos + 1):
                    raise RuntimeError("Failed to extend allocation")
                blocks = self.seq_blocks[seq_id]
            
            physical_block = blocks[block_idx]
            self.key_cache[layer, physical_block, offset] = keys[i]
            self.value_cache[layer, physical_block, offset] = values[i]
        
        # Update length
        max_pos = max(positions).item() if hasattr(positions, 'max') else max(positions)
        self.seq_lengths[seq_id] = max(self.seq_lengths[seq_id], max_pos + 1)
    
    def read(
        self,
        seq_id: int,
        layer: int,
        max_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read keys and values from cache.
        
        Args:
            seq_id: Sequence ID
            layer: Layer index
            max_len: Maximum length to read (default: all)
            
        Returns:
            (keys, values) tensors
        """
        if seq_id not in self.seq_blocks:
            raise ValueError(f"Sequence {seq_id} not allocated")
        
        length = self.seq_lengths.get(seq_id, 0)
        if max_len is not None:
            length = min(length, max_len)
        
        if length == 0:
            return (
                torch.empty(0, self.num_heads, self.head_dim, 
                           dtype=self.dtype, device=self.device),
                torch.empty(0, self.num_heads, self.head_dim,
                           dtype=self.dtype, device=self.device)
            )
        
        blocks = self.seq_blocks[seq_id]
        num_full_blocks = length // self.block_size
        remainder = length % self.block_size
        
        # Gather from blocks
        keys_list = []
        values_list = []
        
        for i in range(num_full_blocks):
            physical_block = blocks[i]
            keys_list.append(self.key_cache[layer, physical_block])
            values_list.append(self.value_cache[layer, physical_block])
        
        # Partial last block
        if remainder > 0 and num_full_blocks < len(blocks):
            physical_block = blocks[num_full_blocks]
            keys_list.append(self.key_cache[layer, physical_block, :remainder])
            values_list.append(self.value_cache[layer, physical_block, :remainder])
        
        if not keys_list:
            return (
                torch.empty(0, self.num_heads, self.head_dim,
                           dtype=self.dtype, device=self.device),
                torch.empty(0, self.num_heads, self.head_dim,
                           dtype=self.dtype, device=self.device)
            )
        
        keys = torch.cat(keys_list, dim=0)
        values = torch.cat(values_list, dim=0)
        
        return keys, values
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        used_blocks = self.num_blocks - len(self.free_blocks)
        return {
            "total_blocks": self.num_blocks,
            "used_blocks": used_blocks,
            "free_blocks": len(self.free_blocks),
            "utilization": used_blocks / self.num_blocks if self.num_blocks > 0 else 0,
            "active_sequences": len(self.seq_blocks),
            "cache_size_mb": self._get_cache_size_mb(),
        }


# =============================================================================
# Paged KV Cache
# =============================================================================

class PagedKVCache:
    """
    High-level paged KV-cache interface.
    
    Wraps BlockManager with a cleaner API.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,
        max_pages: int = 1000,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize paged KV cache.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads  
            head_dim: Dimension per head
            page_size: Tokens per page (default: 16)
            max_pages: Maximum number of pages
            device: Device for cache
            dtype: Data type
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.block_manager = BlockManager(
            num_blocks=max_pages,
            block_size=page_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype
        )
        
        self.page_size = page_size
        self.num_layers = num_layers
    
    def allocate(self, seq_id: int, num_tokens: int) -> bool:
        """Allocate pages for a sequence."""
        return self.block_manager.allocate(seq_id, num_tokens)
    
    def free(self, seq_id: int):
        """Free all pages for a sequence."""
        self.block_manager.free(seq_id)
    
    def write(self, seq_id: int, layer: int, keys: torch.Tensor, values: torch.Tensor):
        """Write to cache."""
        self.block_manager.write(seq_id, layer, keys, values)
    
    def read(self, seq_id: int, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read from cache."""
        return self.block_manager.read(seq_id, layer)
    
    def get_length(self, seq_id: int) -> int:
        """Get current length of a sequence."""
        return self.block_manager.seq_lengths.get(seq_id, 0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.block_manager.get_stats()


# =============================================================================
# Paged Attention Layer
# =============================================================================

class PagedAttention(nn.Module):
    """
    Attention layer with paged KV-cache support.
    
    Drop-in replacement for standard attention that uses
    paged memory management.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        page_size: int = 16,
        max_seq_len: int = 2048,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.page_size = page_size
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Paged cache (lazily initialized)
        self._cache: Optional[PagedKVCache] = None
        self._device = device
    
    def _get_or_create_cache(self, num_layers: int = 1) -> PagedKVCache:
        """Get or create the paged cache."""
        if self._cache is None:
            device = self._device or next(self.parameters()).device
            max_pages = 2048 // self.page_size * 10  # ~10 sequences
            self._cache = PagedKVCache(
                num_layers=num_layers,
                num_heads=self.n_heads,
                head_dim=self.head_dim,
                page_size=self.page_size,
                max_pages=max_pages,
                device=device
            )
        return self._cache
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        seq_id: int = 0,
        layer_id: int = 0,
        use_cache: bool = True,
        cache: Optional[PagedKVCache] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional paged caching.
        
        Args:
            x: Input tensor [batch, seq, dim]
            mask: Attention mask
            seq_id: Sequence ID for caching
            layer_id: Layer ID for caching
            use_cache: Whether to use KV cache
            cache: External cache (uses internal if None)
            
        Returns:
            Output tensor [batch, seq, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        if use_cache:
            # Get cache
            kv_cache = cache or self._get_or_create_cache()
            
            # Ensure sequence is allocated
            if not kv_cache.block_manager.seq_blocks.get(seq_id):
                kv_cache.allocate(seq_id, seq_len + 1024)  # Pre-allocate some extra
            
            # Write new K, V to cache
            kv_cache.write(seq_id, layer_id, k, v)
            
            # Read full K, V from cache
            cached_k, cached_v = kv_cache.read(seq_id, layer_id)
            
            # Use cached values
            k = cached_k.unsqueeze(0)  # Add batch dim
            v = cached_v.unsqueeze(0)
        
        # Transpose for attention: [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.o_proj(output)
        
        return output
    
    def clear_cache(self, seq_id: Optional[int] = None):
        """Clear cache for a sequence or all sequences."""
        if self._cache is None:
            return
        
        if seq_id is not None:
            self._cache.free(seq_id)
        else:
            # Clear all
            for sid in list(self._cache.block_manager.seq_blocks.keys()):
                self._cache.free(sid)


# =============================================================================
# Factory Functions
# =============================================================================

def create_paged_cache(
    model_config: Dict[str, Any],
    page_size: int = 16,
    max_pages: int = 1000,
    device: Optional[torch.device] = None
) -> PagedKVCache:
    """
    Create a paged KV cache from model config.
    
    Args:
        model_config: Model configuration dict with n_layers, n_heads, d_model
        page_size: Tokens per page
        max_pages: Maximum pages
        device: Device for cache
        
    Returns:
        PagedKVCache instance
    """
    n_layers = model_config.get("n_layers", 12)
    n_heads = model_config.get("n_heads", 8)
    d_model = model_config.get("d_model", 512)
    head_dim = d_model // n_heads
    
    return PagedKVCache(
        num_layers=n_layers,
        num_heads=n_heads,
        head_dim=head_dim,
        page_size=page_size,
        max_pages=max_pages,
        device=device
    )


def estimate_memory_savings(
    batch_size: int,
    max_seq_len: int,
    avg_seq_len: int,
    page_size: int = 16
) -> Dict[str, float]:
    """
    Estimate memory savings from paged attention.
    
    Args:
        batch_size: Number of sequences
        max_seq_len: Maximum sequence length
        avg_seq_len: Average actual sequence length
        page_size: Page size
        
    Returns:
        Dict with memory comparison
    """
    # Traditional: Allocate max for everyone
    traditional = batch_size * max_seq_len
    
    # Paged: Allocate only what's needed (rounded up to pages)
    pages_per_seq = (avg_seq_len + page_size - 1) // page_size
    paged = batch_size * pages_per_seq * page_size
    
    savings = 1.0 - (paged / traditional)
    
    return {
        "traditional_tokens": traditional,
        "paged_tokens": paged,
        "savings_percent": savings * 100,
        "throughput_multiplier": traditional / paged if paged > 0 else float('inf')
    }


# Example usage
if __name__ == "__main__":
    # Demo paged attention
    print("PagedAttention Demo")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create cache
    cache = PagedKVCache(
        num_layers=12,
        num_heads=8,
        head_dim=64,
        page_size=16,
        max_pages=1000,
        device=device
    )
    
    print(f"Cache stats: {cache.get_stats()}")
    
    # Allocate for 3 sequences
    cache.allocate(seq_id=0, num_tokens=100)
    cache.allocate(seq_id=1, num_tokens=50)
    cache.allocate(seq_id=2, num_tokens=200)
    
    print(f"\nAfter allocation: {cache.get_stats()}")
    
    # Simulate writing
    for seq_id in [0, 1, 2]:
        k = torch.randn(10, 8, 64, device=device)
        v = torch.randn(10, 8, 64, device=device)
        cache.write(seq_id, layer=0, keys=k, values=v)
    
    # Read back
    k0, v0 = cache.read(seq_id=0, layer=0)
    print(f"\nRead from seq 0: keys shape = {k0.shape}")
    
    # Free one sequence
    cache.free(seq_id=1)
    print(f"\nAfter freeing seq 1: {cache.get_stats()}")
    
    # Estimate savings
    savings = estimate_memory_savings(
        batch_size=32,
        max_seq_len=2048,
        avg_seq_len=256,
        page_size=16
    )
    print(f"\nMemory savings estimate:")
    print(f"  Savings: {savings['savings_percent']:.1f}%")
    print(f"  Throughput multiplier: {savings['throughput_multiplier']:.1f}x")
