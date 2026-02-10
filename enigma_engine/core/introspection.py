"""
Model Introspection for Enigma AI Engine

Analyze and visualize model internals.

Features:
- Layer inspection
- Attention visualization
- Weight statistics
- Gradient flow analysis
- Activation maps
- Layer-wise analysis

Usage:
    from enigma_engine.core.introspection import ModelIntrospector, get_introspector
    
    # Create introspector
    introspector = ModelIntrospector(model)
    
    # Get model summary
    summary = introspector.summary()
    
    # Analyze layers
    layer_info = introspector.analyze_layer("transformer.0.attention")
    
    # Visualize attention
    attention_maps = introspector.get_attention_maps(input_ids)
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


@dataclass
class LayerInfo:
    """Information about a model layer."""
    name: str
    type: str
    input_shape: Optional[Tuple] = None
    output_shape: Optional[Tuple] = None
    num_params: int = 0
    trainable_params: int = 0
    
    # Statistics
    weight_mean: float = 0.0
    weight_std: float = 0.0
    weight_min: float = 0.0
    weight_max: float = 0.0
    
    # Gradient info (if available)
    grad_mean: Optional[float] = None
    grad_norm: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "num_params": self.num_params,
            "trainable_params": self.trainable_params,
            "weight_mean": self.weight_mean,
            "weight_std": self.weight_std,
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "grad_mean": self.grad_mean,
            "grad_norm": self.grad_norm
        }


@dataclass
class ModelSummary:
    """Complete model summary."""
    name: str
    total_params: int
    trainable_params: int
    non_trainable_params: int
    layers: List[LayerInfo]
    memory_mb: float
    dtype: str
    device: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "non_trainable_params": self.non_trainable_params,
            "num_layers": len(self.layers),
            "memory_mb": self.memory_mb,
            "dtype": self.dtype,
            "device": self.device
        }
    
    def __str__(self) -> str:
        """Pretty print summary."""
        lines = [
            f"Model: {self.name}",
            f"Total Parameters: {self.total_params:,}",
            f"Trainable Parameters: {self.trainable_params:,}",
            f"Non-trainable Parameters: {self.non_trainable_params:,}",
            f"Memory: {self.memory_mb:.2f} MB",
            f"Device: {self.device}",
            f"Dtype: {self.dtype}",
            "",
            "Layers:",
        ]
        
        for layer in self.layers[:20]:  # Show first 20
            lines.append(f"  {layer.name}: {layer.type} ({layer.num_params:,} params)")
        
        if len(self.layers) > 20:
            lines.append(f"  ... and {len(self.layers) - 20} more layers")
        
        return "\n".join(lines)


@dataclass
class AttentionMap:
    """Attention visualization data."""
    layer_name: str
    head: int
    attention_weights: Any  # numpy array or tensor
    tokens: List[str]
    
    def to_heatmap_data(self) -> Dict[str, Any]:
        """Convert to heatmap visualization data."""
        if HAS_TORCH and torch.is_tensor(self.attention_weights):
            data = self.attention_weights.detach().cpu().numpy().tolist()
        else:
            data = list(self.attention_weights)
        
        return {
            "layer": self.layer_name,
            "head": self.head,
            "data": data,
            "x_labels": self.tokens,
            "y_labels": self.tokens
        }


class ActivationRecorder:
    """Records activations during forward pass."""
    
    def __init__(self):
        self._activations: Dict[str, Any] = {}
        self._hooks: List[Any] = []
    
    def attach(self, model: Any, layer_names: Optional[List[str]] = None):
        """Attach hooks to record activations."""
        if not HAS_TORCH:
            return
        
        def make_hook(name: str):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    self._activations[name] = out[0].detach()
                else:
                    self._activations[name] = out.detach()
            return hook
        
        for name, module in model.named_modules():
            if layer_names is None or name in layer_names:
                handle = module.register_forward_hook(make_hook(name))
                self._hooks.append(handle)
    
    def detach(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def get_activations(self) -> Dict[str, Any]:
        """Get recorded activations."""
        return self._activations
    
    def clear(self):
        """Clear recorded activations."""
        self._activations.clear()


class GradientAnalyzer:
    """Analyzes gradient flow."""
    
    def __init__(self, model: Any):
        self._model = model
        self._gradients: Dict[str, Any] = {}
        self._hooks: List[Any] = []
    
    def attach(self):
        """Attach gradient hooks."""
        if not HAS_TORCH:
            return
        
        def make_hook(name: str):
            def hook(grad):
                self._gradients[name] = grad.detach()
            return hook
        
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(make_hook(name))
                self._hooks.append(handle)
    
    def detach(self):
        """Remove gradient hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def get_gradient_stats(self) -> Dict[str, Dict[str, float]]:
        """Get gradient statistics per parameter."""
        stats = {}
        
        for name, grad in self._gradients.items():
            if HAS_TORCH and grad is not None:
                stats[name] = {
                    "mean": float(grad.mean()),
                    "std": float(grad.std()),
                    "min": float(grad.min()),
                    "max": float(grad.max()),
                    "norm": float(grad.norm())
                }
        
        return stats
    
    def check_vanishing_gradients(self, threshold: float = 1e-7) -> List[str]:
        """Check for vanishing gradients."""
        vanishing = []
        
        for name, grad in self._gradients.items():
            if HAS_TORCH and grad is not None:
                if grad.abs().max() < threshold:
                    vanishing.append(name)
        
        return vanishing
    
    def check_exploding_gradients(self, threshold: float = 100.0) -> List[str]:
        """Check for exploding gradients."""
        exploding = []
        
        for name, grad in self._gradients.items():
            if HAS_TORCH and grad is not None:
                if grad.abs().max() > threshold:
                    exploding.append(name)
        
        return exploding
    
    def clear(self):
        """Clear recorded gradients."""
        self._gradients.clear()


class ModelIntrospector:
    """
    Introspects and analyzes neural network models.
    """
    
    def __init__(self, model: Any):
        """
        Initialize introspector.
        
        Args:
            model: PyTorch model to analyze
        """
        self._model = model
        self._activation_recorder = ActivationRecorder()
        self._gradient_analyzer = GradientAnalyzer(model)
        
        # Cache
        self._summary_cache: Optional[ModelSummary] = None
    
    def summary(self, verbose: bool = False) -> ModelSummary:
        """
        Get model summary.
        
        Args:
            verbose: Include detailed layer info
            
        Returns:
            Model summary
        """
        if self._summary_cache and not verbose:
            return self._summary_cache
        
        if not HAS_TORCH:
            return ModelSummary(
                name="Unknown",
                total_params=0,
                trainable_params=0,
                non_trainable_params=0,
                layers=[],
                memory_mb=0,
                dtype="unknown",
                device="unknown"
            )
        
        # Count parameters
        total_params = 0
        trainable_params = 0
        layers = []
        
        for name, module in self._model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                params = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                layer_info = LayerInfo(
                    name=name,
                    type=module.__class__.__name__,
                    num_params=params,
                    trainable_params=trainable
                )
                
                # Weight statistics
                for p in module.parameters():
                    if p.numel() > 0:
                        layer_info.weight_mean = float(p.data.mean())
                        layer_info.weight_std = float(p.data.std())
                        layer_info.weight_min = float(p.data.min())
                        layer_info.weight_max = float(p.data.max())
                        break
                
                layers.append(layer_info)
                total_params += params
                trainable_params += trainable
        
        # Estimate memory
        memory_bytes = sum(
            p.numel() * p.element_size()
            for p in self._model.parameters()
        )
        memory_mb = memory_bytes / (1024 * 1024)
        
        # Get device and dtype
        try:
            first_param = next(self._model.parameters())
            device = str(first_param.device)
            dtype = str(first_param.dtype)
        except StopIteration:
            device = "unknown"
            dtype = "unknown"
        
        summary = ModelSummary(
            name=self._model.__class__.__name__,
            total_params=total_params,
            trainable_params=trainable_params,
            non_trainable_params=total_params - trainable_params,
            layers=layers,
            memory_mb=memory_mb,
            dtype=dtype,
            device=device
        )
        
        self._summary_cache = summary
        return summary
    
    def analyze_layer(self, layer_name: str) -> Optional[LayerInfo]:
        """
        Analyze a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Layer information
        """
        if not HAS_TORCH:
            return None
        
        # Find layer
        layer = None
        for name, module in self._model.named_modules():
            if name == layer_name:
                layer = module
                break
        
        if layer is None:
            return None
        
        # Get statistics
        params = sum(p.numel() for p in layer.parameters())
        trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        
        info = LayerInfo(
            name=layer_name,
            type=layer.__class__.__name__,
            num_params=params,
            trainable_params=trainable
        )
        
        # Detailed weight analysis
        for p in layer.parameters():
            if p.numel() > 0:
                info.weight_mean = float(p.data.mean())
                info.weight_std = float(p.data.std())
                info.weight_min = float(p.data.min())
                info.weight_max = float(p.data.max())
                
                if p.grad is not None:
                    info.grad_mean = float(p.grad.mean())
                    info.grad_norm = float(p.grad.norm())
                break
        
        return info
    
    def get_attention_maps(
        self,
        input_ids: Any,
        layer_names: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None
    ) -> List[AttentionMap]:
        """
        Get attention maps for input.
        
        Args:
            input_ids: Input token IDs
            layer_names: Specific attention layers to capture
            tokenizer: Tokenizer for token labels
            
        Returns:
            List of attention maps
        """
        if not HAS_TORCH:
            return []
        
        attention_maps = []
        
        # Find attention layers
        attention_layers = []
        for name, module in self._model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                if layer_names is None or name in layer_names:
                    attention_layers.append((name, module))
        
        if not attention_layers:
            logger.warning("No attention layers found")
            return []
        
        # Attach hooks
        attention_outputs = {}
        
        def make_hook(name: str):
            def hook(module, inp, out):
                if isinstance(out, tuple) and len(out) > 1:
                    # Attention weights are usually second output
                    attention_outputs[name] = out[1]
                else:
                    attention_outputs[name] = out
            return hook
        
        hooks = []
        for name, module in attention_layers:
            handle = module.register_forward_hook(make_hook(name))
            hooks.append(handle)
        
        try:
            # Forward pass
            with torch.no_grad():
                if isinstance(input_ids, torch.Tensor):
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    self._model(input_ids)
                else:
                    self._model(torch.tensor([input_ids]))
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
        finally:
            for hook in hooks:
                hook.remove()
        
        # Get token labels
        if tokenizer and hasattr(tokenizer, 'decode'):
            try:
                if isinstance(input_ids, torch.Tensor):
                    ids = input_ids[0].tolist()
                else:
                    ids = input_ids
                tokens = [tokenizer.decode([i]) for i in ids]
            except Exception:
                tokens = [f"[{i}]" for i in range(len(input_ids[0]) if hasattr(input_ids, '__len__') else 0)]
        else:
            tokens = []
        
        # Create attention maps
        for name, attn in attention_outputs.items():
            if attn is None:
                continue
            
            if attn.dim() >= 3:
                # Usually (batch, heads, seq, seq)
                num_heads = attn.size(1) if attn.dim() >= 4 else 1
                
                for head in range(min(num_heads, 4)):  # Limit to 4 heads
                    if attn.dim() == 4:
                        weights = attn[0, head]
                    else:
                        weights = attn[0]
                    
                    attention_maps.append(AttentionMap(
                        layer_name=name,
                        head=head,
                        attention_weights=weights,
                        tokens=tokens
                    ))
        
        return attention_maps
    
    def get_activations(
        self,
        input_ids: Any,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get layer activations for input.
        
        Args:
            input_ids: Input token IDs
            layer_names: Specific layers to capture
            
        Returns:
            Dictionary of layer name -> activations
        """
        if not HAS_TORCH:
            return {}
        
        self._activation_recorder.attach(self._model, layer_names)
        
        try:
            with torch.no_grad():
                if isinstance(input_ids, torch.Tensor):
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    self._model(input_ids)
                else:
                    self._model(torch.tensor([input_ids]))
        finally:
            self._activation_recorder.detach()
        
        return self._activation_recorder.get_activations()
    
    def analyze_gradients(self) -> Dict[str, Any]:
        """
        Analyze current gradients.
        
        Returns:
            Gradient analysis results
        """
        if not HAS_TORCH:
            return {}
        
        stats = {}
        
        for name, param in self._model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                stats[name] = {
                    "mean": float(grad.mean()),
                    "std": float(grad.std()),
                    "norm": float(grad.norm()),
                    "max_abs": float(grad.abs().max())
                }
        
        # Check for issues
        vanishing = [n for n, s in stats.items() if s["max_abs"] < 1e-7]
        exploding = [n for n, s in stats.items() if s["max_abs"] > 100]
        
        return {
            "per_layer": stats,
            "vanishing_gradients": vanishing,
            "exploding_gradients": exploding,
            "total_layers_with_grad": len(stats)
        }
    
    def compare_weights(
        self,
        other_model: Any,
        tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Compare weights with another model.
        
        Args:
            other_model: Model to compare with
            tolerance: Difference tolerance
            
        Returns:
            Comparison results
        """
        if not HAS_TORCH:
            return {}
        
        differences = {}
        matching = 0
        different = 0
        
        state_dict1 = self._model.state_dict()
        state_dict2 = other_model.state_dict()
        
        all_keys = set(state_dict1.keys()) | set(state_dict2.keys())
        
        for key in all_keys:
            if key not in state_dict1:
                differences[key] = "missing in model 1"
                different += 1
            elif key not in state_dict2:
                differences[key] = "missing in model 2"
                different += 1
            else:
                diff = (state_dict1[key] - state_dict2[key]).abs().max()
                if diff > tolerance:
                    differences[key] = float(diff)
                    different += 1
                else:
                    matching += 1
        
        return {
            "matching_layers": matching,
            "different_layers": different,
            "differences": differences,
            "identical": different == 0
        }
    
    def export_architecture(self, path: str):
        """Export model architecture to JSON."""
        summary = self.summary(verbose=True)
        
        data = {
            "summary": summary.to_dict(),
            "layers": [l.to_dict() for l in summary.layers]
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# Convenience function
def get_introspector(model: Any) -> ModelIntrospector:
    """Create introspector for model."""
    return ModelIntrospector(model)
