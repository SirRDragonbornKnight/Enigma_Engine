"""
Activation Checkpointing Utilities for enigma_engine

Memory-efficient training by trading compute for memory:
- Gradient checkpointing at layer boundaries
- Selective checkpointing
- Automatic checkpoint placement

Usage:
    from enigma_engine.core.checkpointing import checkpoint_model
    
    model = checkpoint_model(model, checkpoint_ratio=0.5)
"""

import functools
import logging
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


def checkpoint_function(
    function: Callable,
    *args,
    use_reentrant: bool = False,
    preserve_rng_state: bool = True
) -> Any:
    """
    Apply gradient checkpointing to a function.
    
    Args:
        function: Function to checkpoint
        args: Arguments to pass to function
        use_reentrant: Use reentrant autograd (legacy)
        preserve_rng_state: Preserve RNG state for dropout
    
    Returns:
        Function output
    """
    if torch.is_grad_enabled():
        return checkpoint(
            function,
            *args,
            use_reentrant=use_reentrant,
            preserve_rng_state=preserve_rng_state
        )
    else:
        return function(*args)


class CheckpointedSequential(nn.Sequential):
    """
    Sequential module with gradient checkpointing.
    
    Checkpoints every `checkpoint_every` layers.
    """
    
    def __init__(
        self,
        *args,
        checkpoint_every: int = 1,
        use_reentrant: bool = False
    ):
        super().__init__(*args)
        self.checkpoint_every = checkpoint_every
        self.use_reentrant = use_reentrant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled() or self.checkpoint_every <= 0:
            return super().forward(x)
        
        # Group layers for checkpointing
        layers = list(self._modules.values())
        
        for i, layer in enumerate(layers):
            if i % self.checkpoint_every == 0:
                x = checkpoint(
                    layer,
                    x,
                    use_reentrant=self.use_reentrant
                )
            else:
                x = layer(x)
        
        return x


class GradientCheckpointer:
    """
    Utility class for applying gradient checkpointing to models.
    
    Supports:
    - Full checkpointing (all layers)
    - Selective checkpointing (every N layers)
    - Memory-based automatic checkpointing
    """
    
    def __init__(
        self,
        use_reentrant: bool = False,
        preserve_rng_state: bool = True
    ):
        self.use_reentrant = use_reentrant
        self.preserve_rng_state = preserve_rng_state
        self._original_forwards = {}
    
    def checkpoint_module(
        self,
        module: nn.Module,
        layer_types: Optional[tuple[type, ...]] = None
    ):
        """
        Apply checkpointing to a module's submodules.
        
        Args:
            module: Module to checkpoint
            layer_types: Types of layers to checkpoint (None = all)
        """
        for name, child in module.named_children():
            if layer_types is None or isinstance(child, layer_types):
                self._wrap_forward(child, name)
            
            # Recurse
            self.checkpoint_module(child, layer_types)
    
    def _wrap_forward(self, module: nn.Module, name: str):
        """Wrap a module's forward method with checkpointing."""
        original_forward = module.forward
        
        # Store original
        self._original_forwards[id(module)] = original_forward
        
        @functools.wraps(original_forward)
        def checkpointed_forward(*args, **kwargs):
            if not torch.is_grad_enabled():
                return original_forward(*args, **kwargs)
            
            # Checkpoint only accepts tensor args
            tensor_args = []
            non_tensor_args = {}
            
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    tensor_args.append(arg)
                else:
                    non_tensor_args[f'arg_{i}'] = arg
            
            non_tensor_args.update(kwargs)
            
            def forward_fn(*tensors):
                # Reconstruct args
                tensor_iter = iter(tensors)
                reconstructed = []
                
                for i, arg in enumerate(args):
                    if isinstance(arg, torch.Tensor):
                        reconstructed.append(next(tensor_iter))
                    else:
                        reconstructed.append(non_tensor_args[f'arg_{i}'])
                
                return original_forward(*reconstructed, **kwargs)
            
            return checkpoint(
                forward_fn,
                *tensor_args,
                use_reentrant=self.use_reentrant,
                preserve_rng_state=self.preserve_rng_state
            )
        
        module.forward = checkpointed_forward
    
    def remove_checkpointing(self, module: nn.Module):
        """Remove checkpointing from a module."""
        module_id = id(module)
        if module_id in self._original_forwards:
            module.forward = self._original_forwards[module_id]
            del self._original_forwards[module_id]
        
        for child in module.children():
            self.remove_checkpointing(child)


def checkpoint_model(
    model: nn.Module,
    checkpoint_ratio: float = 1.0,
    layer_class: Optional[type] = None,
    use_reentrant: bool = False
) -> nn.Module:
    """
    Apply gradient checkpointing to a model.
    
    Args:
        model: Model to checkpoint
        checkpoint_ratio: Fraction of layers to checkpoint (0-1)
        layer_class: Specific layer class to checkpoint
        use_reentrant: Use reentrant autograd
    
    Returns:
        Model with checkpointing applied
    """
    checkpointer = GradientCheckpointer(use_reentrant=use_reentrant)
    
    if layer_class:
        # Checkpoint specific layer type
        checkpointer.checkpoint_module(model, (layer_class,))
    elif checkpoint_ratio >= 1.0:
        # Checkpoint all layers
        checkpointer.checkpoint_module(model)
    else:
        # Selective checkpointing based on ratio
        layers = _get_transformer_layers(model)
        
        num_to_checkpoint = int(len(layers) * checkpoint_ratio)
        # Checkpoint evenly spaced layers
        step = len(layers) // max(num_to_checkpoint, 1)
        
        for i, layer in enumerate(layers):
            if i % step == 0:
                checkpointer._wrap_forward(layer, f"layer_{i}")
    
    logger.info(f"Applied gradient checkpointing to model (ratio={checkpoint_ratio})")
    return model


def _get_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """Find transformer layers in a model."""
    layers = []
    
    # Common transformer layer container names
    layer_names = ['layers', 'blocks', 'encoder', 'decoder', 'h']
    
    for name in layer_names:
        if hasattr(model, name):
            container = getattr(model, name)
            if isinstance(container, (nn.ModuleList, nn.Sequential)):
                layers.extend(list(container))
    
    return layers


class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention with automatic checkpointing.
    
    Uses chunked computation to reduce memory for long sequences.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        chunk_size: int = 1024,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Chunked attention for memory efficiency
        if seq_len > self.chunk_size:
            output = self._chunked_attention(q, k, v, attention_mask)
        else:
            output = self._standard_attention(q, k, v, attention_mask)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.out_proj(output)
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Standard attention computation."""
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_weights = attn_weights + mask
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        return torch.matmul(attn_weights, v)
    
    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Memory-efficient chunked attention."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        outputs = []
        
        for i in range(0, seq_len, self.chunk_size):
            chunk_end = min(i + self.chunk_size, seq_len)
            q_chunk = q[:, :, i:chunk_end]
            
            # Compute attention for this chunk
            attn_weights = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                # Extract relevant mask portion
                chunk_mask = mask[:, :, i:chunk_end, :]
                attn_weights = attn_weights + chunk_mask
            
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            chunk_output = torch.matmul(attn_weights, v)
            outputs.append(chunk_output)
        
        return torch.cat(outputs, dim=2)


def estimate_memory_savings(
    model: nn.Module,
    batch_size: int = 1,
    seq_len: int = 2048,
    checkpoint_ratio: float = 1.0
) -> dict:
    """
    Estimate memory savings from gradient checkpointing.
    
    Args:
        model: Model to analyze
        batch_size: Training batch size
        seq_len: Sequence length
        checkpoint_ratio: Fraction of layers to checkpoint
    
    Returns:
        Memory estimates
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count transformer layers
    layers = _get_transformer_layers(model)
    num_layers = len(layers)
    
    # Estimate activation memory per layer (rough)
    # Assuming hidden_size from first Linear layer
    hidden_size = None
    for layer in layers:
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                hidden_size = m.in_features
                break
        if hidden_size:
            break
    
    if hidden_size is None:
        hidden_size = 768  # Default
    
    # Bytes per activation: batch * seq * hidden * 4 (fp32) * 2 (for backward)
    activation_per_layer = batch_size * seq_len * hidden_size * 4 * 2
    
    # Without checkpointing: store all activations
    memory_no_checkpoint = num_layers * activation_per_layer
    
    # With checkpointing: only store checkpointed layer activations
    checkpointed_layers = int(num_layers * checkpoint_ratio)
    memory_with_checkpoint = checkpointed_layers * activation_per_layer / num_layers
    
    # But need to recompute, so add compute cost
    recompute_ratio = checkpointed_layers / num_layers if num_layers > 0 else 0
    
    return {
        'total_params': total_params,
        'num_layers': num_layers,
        'memory_without_checkpoint_mb': memory_no_checkpoint / (1024 * 1024),
        'memory_with_checkpoint_mb': memory_with_checkpoint / (1024 * 1024),
        'memory_savings_pct': (1 - memory_with_checkpoint / memory_no_checkpoint) * 100 if memory_no_checkpoint > 0 else 0,
        'additional_compute_ratio': recompute_ratio
    }
