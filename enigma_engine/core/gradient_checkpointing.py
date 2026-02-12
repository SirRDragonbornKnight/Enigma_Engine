"""
Gradient Checkpointing

Memory-efficient training by trading compute for memory.
Recomputes activations during backward pass instead of storing them.

FILE: enigma_engine/core/gradient_checkpointing.py
TYPE: Core/Training
MAIN CLASSES: CheckpointManager, ActivationCheckpointing, SelectiveCheckpointing
"""

import functools
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.checkpoint import checkpoint
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckpointStrategy(Enum):
    """Checkpointing strategy."""
    NONE = "none"
    FULL = "full"  # Checkpoint every layer
    SELECTIVE = "selective"  # Checkpoint specific layers
    GRADIENT_BASED = "gradient_based"  # Based on gradient memory
    UNIFORM = "uniform"  # Every N layers
    SQRT = "sqrt"  # sqrt(num_layers) checkpoints


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""
    strategy: CheckpointStrategy = CheckpointStrategy.NONE
    
    # Uniform strategy
    checkpoint_every_n: int = 2
    
    # Selective strategy
    checkpoint_layers: list[int] = None
    
    # Memory settings
    preserve_rng_state: bool = True
    use_reentrant: bool = True  # False is more memory efficient but stricter
    
    # Profiling
    profile_memory: bool = False
    memory_threshold_mb: float = 1000.0  # Memory per layer threshold


if HAS_TORCH:
    
    class CheckpointFunction(torch.autograd.Function):
        """
        Custom checkpoint function with memory profiling.
        
        Recomputes forward pass during backward to save activation memory.
        """
        
        @staticmethod
        def forward(ctx: Any, run_function: Callable, preserve_rng_state: bool, *args: Any) -> Any:
            ctx.run_function = run_function
            ctx.preserve_rng_state = preserve_rng_state
            
            # Save inputs for recomputation
            ctx.save_for_backward(*args)
            
            # Save RNG state
            if preserve_rng_state:
                ctx.fwd_cpu_state = torch.get_rng_state()
                ctx.had_cuda_in_fwd = False
                if torch.cuda.is_available():
                    ctx.had_cuda_in_fwd = True
                    ctx.fwd_gpu_state = torch.cuda.get_rng_state()
            
            with torch.no_grad():
                outputs = run_function(*args)
            
            return outputs
        
        @staticmethod
        def backward(ctx: Any, *args: Any) -> tuple[None, None, ...]:
            inputs = ctx.saved_tensors
            
            # Restore RNG state
            if ctx.preserve_rng_state:
                rng_devices = []
                if ctx.had_cuda_in_fwd:
                    rng_devices.append(torch.cuda.current_device())
                
                with torch.random.fork_rng(devices=rng_devices, enabled=True):
                    torch.set_rng_state(ctx.fwd_cpu_state)
                    if ctx.had_cuda_in_fwd:
                        torch.cuda.set_rng_state(ctx.fwd_gpu_state)
                    
                    # Recompute forward
                    detached_inputs = [
                        inp.detach().requires_grad_(inp.requires_grad)
                        for inp in inputs
                    ]
                    
                    with torch.enable_grad():
                        outputs = ctx.run_function(*detached_inputs)
            else:
                detached_inputs = [
                    inp.detach().requires_grad_(inp.requires_grad)
                    for inp in inputs
                ]
                with torch.enable_grad():
                    outputs = ctx.run_function(*detached_inputs)
            
            # Compute gradients
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            
            torch.autograd.backward(outputs, args)
            
            grads = tuple(
                inp.grad if isinstance(inp, torch.Tensor) else None
                for inp in detached_inputs
            )
            
            return (None, None) + grads
    
    
    def checkpoint_fn(
        function: Callable,
        *args,
        preserve_rng_state: bool = True,
        use_reentrant: bool = True
    ) -> Any:
        """
        Checkpoint a function.
        
        Args:
            function: Function to checkpoint
            *args: Arguments to function
            preserve_rng_state: Whether to preserve RNG state
            use_reentrant: Use reentrant checkpointing
        
        Returns:
            Function output
        """
        if use_reentrant:
            return CheckpointFunction.apply(function, preserve_rng_state, *args)
        else:
            return checkpoint(
                function, *args,
                preserve_rng_state=preserve_rng_state,
                use_reentrant=False
            )
    
    
    class CheckpointedModule(nn.Module):
        """
        Wrapper that makes a module use gradient checkpointing.
        """
        
        def __init__(
            self,
            module: nn.Module,
            config: Optional[CheckpointConfig] = None
        ) -> None:
            super().__init__()
            self.module = module
            self.config = config or CheckpointConfig()
        
        def forward(self, *args, **kwargs) -> Any:
            if self.training:
                # Wrap kwargs into a lambda
                def run_fn(*inputs):
                    return self.module(*inputs, **kwargs)
                
                return checkpoint_fn(
                    run_fn,
                    *args,
                    preserve_rng_state=self.config.preserve_rng_state,
                    use_reentrant=self.config.use_reentrant
                )
            else:
                return self.module(*args, **kwargs)
    
    
    class SelectiveCheckpointing:
        """
        Apply checkpointing selectively to specific layers.
        """
        
        def __init__(self, config: CheckpointConfig) -> None:
            self.config = config
            self._checkpointed_layers: set[int] = set()
        
        def apply(self, model: nn.Module, layers_attr: str = "layers") -> None:
            """
            Apply selective checkpointing to model.
            
            Args:
                model: Model to apply checkpointing to
                layers_attr: Attribute name containing layers
            """
            layers = getattr(model, layers_attr, None)
            if layers is None:
                logger.warning(f"Model has no '{layers_attr}' attribute")
                return
            
            num_layers = len(layers)
            
            # Determine which layers to checkpoint
            if self.config.strategy == CheckpointStrategy.FULL:
                self._checkpointed_layers = set(range(num_layers))
            
            elif self.config.strategy == CheckpointStrategy.UNIFORM:
                self._checkpointed_layers = set(
                    range(0, num_layers, self.config.checkpoint_every_n)
                )
            
            elif self.config.strategy == CheckpointStrategy.SELECTIVE:
                self._checkpointed_layers = set(self.config.checkpoint_layers or [])
            
            elif self.config.strategy == CheckpointStrategy.SQRT:
                import math
                n_checkpoints = int(math.sqrt(num_layers))
                step = max(1, num_layers // n_checkpoints)
                self._checkpointed_layers = set(range(0, num_layers, step))
            
            # Wrap layers
            new_layers = nn.ModuleList()
            for i, layer in enumerate(layers):
                if i in self._checkpointed_layers:
                    new_layers.append(CheckpointedModule(layer, self.config))
                else:
                    new_layers.append(layer)
            
            setattr(model, layers_attr, new_layers)
            
            logger.info(
                f"Applied checkpointing to {len(self._checkpointed_layers)}/{num_layers} layers"
            )
    
    
    class ActivationCheckpointing:
        """
        Activation checkpointing for transformer blocks.
        
        Implements various strategies for memory-compute tradeoff.
        """
        
        def __init__(self, config: Optional[CheckpointConfig] = None) -> None:
            self.config = config or CheckpointConfig()
            self._original_forwards: dict[int, Callable] = {}
        
        def enable(self, model: nn.Module) -> None:
            """Enable checkpointing on model."""
            if self.config.strategy == CheckpointStrategy.NONE:
                return
            
            layers = self._find_layers(model)
            
            for idx, layer in layers:
                if self._should_checkpoint(idx, len(layers)):
                    self._wrap_layer(layer, idx)
        
        def disable(self, model: nn.Module) -> None:
            """Disable checkpointing and restore original forwards."""
            for layer_id, original_forward in self._original_forwards.items():
                for module in model.modules():
                    if id(module) == layer_id:
                        module.forward = original_forward
            
            self._original_forwards.clear()
        
        def _find_layers(self, model: nn.Module) -> list[tuple[int, nn.Module]]:
            """Find transformer layers in model."""
            layers = []
            
            # Try common attribute names
            for attr in ['layers', 'blocks', 'encoder', 'decoder']:
                if hasattr(model, attr):
                    module_list = getattr(model, attr)
                    if isinstance(module_list, nn.ModuleList):
                        layers = list(enumerate(module_list))
                        break
            
            if not layers:
                # Fall back to finding by class name
                for name, module in model.named_modules():
                    if 'block' in name.lower() or 'layer' in name.lower():
                        layers.append((len(layers), module))
            
            return layers
        
        def _should_checkpoint(self, idx: int, total: int) -> bool:
            """Determine if layer should be checkpointed."""
            strategy = self.config.strategy
            
            if strategy == CheckpointStrategy.FULL:
                return True
            
            elif strategy == CheckpointStrategy.UNIFORM:
                return idx % self.config.checkpoint_every_n == 0
            
            elif strategy == CheckpointStrategy.SELECTIVE:
                return idx in (self.config.checkpoint_layers or [])
            
            elif strategy == CheckpointStrategy.SQRT:
                import math
                n_checkpoints = int(math.sqrt(total))
                step = max(1, total // n_checkpoints)
                return idx % step == 0
            
            return False
        
        def _wrap_layer(self, layer: nn.Module, idx: int) -> None:
            """Wrap layer forward with checkpointing."""
            original_forward = layer.forward
            self._original_forwards[id(layer)] = original_forward
            
            @functools.wraps(original_forward)
            def checkpointed_forward(*args, **kwargs):
                def run_fn(*inputs):
                    return original_forward(*inputs, **kwargs)
                
                if layer.training:
                    return checkpoint_fn(
                        run_fn,
                        *args,
                        preserve_rng_state=self.config.preserve_rng_state,
                        use_reentrant=self.config.use_reentrant
                    )
                else:
                    return original_forward(*args, **kwargs)
            
            layer.forward = checkpointed_forward
            logger.debug(f"Wrapped layer {idx} with checkpointing")
    
    
    class MemoryProfiler:
        """Profile memory usage for checkpointing decisions."""
        
        @staticmethod
        def get_activation_memory(model: nn.Module, sample_input: torch.Tensor) -> dict[str, float]:
            """
            Profile activation memory per layer.
            
            Args:
                model: Model to profile
                sample_input: Sample input tensor
            
            Returns:
                Dict mapping layer name to memory in MB
            """
            memories = {}
            handles = []
            
            def hook_factory(name):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        mem_mb = output.numel() * output.element_size() / 1024**2
                    elif isinstance(output, tuple):
                        mem_mb = sum(
                            t.numel() * t.element_size() / 1024**2
                            for t in output if isinstance(t, torch.Tensor)
                        )
                    else:
                        mem_mb = 0
                    memories[name] = mem_mb
                return hook
            
            for name, module in model.named_modules():
                handle = module.register_forward_hook(hook_factory(name))
                handles.append(handle)
            
            with torch.no_grad():
                model(sample_input)
            
            for handle in handles:
                handle.remove()
            
            return memories
        
        @staticmethod
        def suggest_checkpoints(
            memories: dict[str, float],
            threshold_mb: float = 100.0
        ) -> list[str]:
            """
            Suggest layers to checkpoint based on memory usage.
            
            Args:
                memories: Memory per layer from profiling
                threshold_mb: Memory threshold
            
            Returns:
                List of layer names to checkpoint
            """
            return [
                name for name, mem in memories.items()
                if mem > threshold_mb
            ]
    
    
    def apply_gradient_checkpointing(
        model: nn.Module,
        strategy: CheckpointStrategy = CheckpointStrategy.UNIFORM,
        **kwargs
    ) -> 'ActivationCheckpointing':
        """
        Apply gradient checkpointing to a model.
        
        Args:
            model: Model to apply checkpointing to
            strategy: Checkpointing strategy
            **kwargs: Additional config options
        """
        config = CheckpointConfig(strategy=strategy, **kwargs)
        checkpointing = ActivationCheckpointing(config)
        checkpointing.enable(model)
        return checkpointing
    
    
    def checkpoint_sequential_layers(
        layers: nn.ModuleList,
        segments: int,
        *inputs
    ) -> Any:
        """
        Checkpoint a sequential list of layers.
        
        Args:
            layers: ModuleList of layers
            segments: Number of checkpoint segments
            *inputs: Inputs to first layer
        
        Returns:
            Output from last layer
        """
        def run_function(start, end, *args):
            x = args[0] if len(args) == 1 else args
            for layer in layers[start:end]:
                x = layer(x)
            return x
        
        segment_size = max(1, len(layers) // segments)
        
        x = inputs[0] if len(inputs) == 1 else inputs
        
        for i in range(0, len(layers), segment_size):
            end = min(i + segment_size, len(layers))
            x = checkpoint_fn(
                functools.partial(run_function, i, end),
                x
            )
        
        return x

else:
    # Stubs when torch not available
    CheckpointFunction = None
    checkpoint_fn = None
    CheckpointedModule = None
    SelectiveCheckpointing = None
    ActivationCheckpointing = None
    MemoryProfiler = None
    apply_gradient_checkpointing = None
    checkpoint_sequential_layers = None
