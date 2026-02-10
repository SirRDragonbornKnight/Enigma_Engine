"""
Model Debugging Tools

Comprehensive debugging toolkit for neural network inference,
including layer-by-layer analysis, gradient flow, and diagnostics.

FILE: enigma_engine/core/model_debugger.py
TYPE: Core/Debug
MAIN CLASSES: ModelDebugger, LayerInspector, GradientAnalyzer
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LayerDiagnostics:
    """Diagnostics for a single layer."""
    name: str
    module_type: str
    input_shape: tuple
    output_shape: tuple
    num_params: int
    has_nan_activations: bool
    has_inf_activations: bool
    activation_stats: dict[str, float]
    gradient_stats: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.module_type,
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "output_shape": list(self.output_shape) if self.output_shape else None,
            "num_params": self.num_params,
            "has_nan": self.has_nan_activations,
            "has_inf": self.has_inf_activations,
            "activation_stats": self.activation_stats,
            "gradient_stats": self.gradient_stats,
            "issues": self.issues
        }


@dataclass
class DebugReport:
    """Complete debug report."""
    model_name: str
    timestamp: datetime
    total_params: int
    layer_diagnostics: list[LayerDiagnostics]
    global_issues: list[str]
    recommendations: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "total_params": self.total_params,
            "layers": [d.to_dict() for d in self.layer_diagnostics],
            "global_issues": self.global_issues,
            "recommendations": self.recommendations
        }
    
    def save(self, filepath: str):
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


if HAS_TORCH:
    
    class LayerInspector:
        """
        Inspect individual layers during forward pass.
        """
        
        def __init__(self, model: nn.Module):
            self.model = model
            self._hooks: list[torch.utils.hooks.RemovableHandle] = []
            self._layer_data: dict[str, dict[str, Any]] = {}
        
        def inspect_layer(
            self,
            layer_name: str,
            input_data: torch.Tensor
        ) -> dict[str, Any]:
            """
            Inspect a specific layer's behavior.
            
            Args:
                layer_name: Name of layer to inspect
                input_data: Input tensor
            
            Returns:
                Layer inspection results
            """
            results = {
                "name": layer_name,
                "found": False,
                "input_shape": None,
                "output_shape": None,
                "activations": None,
                "stats": {}
            }
            
            for name, module in self.model.named_modules():
                if name == layer_name:
                    results["found"] = True
                    results["type"] = type(module).__name__
                    
                    # Register hook
                    captured = {}
                    
                    def hook(mod, inp, out, _cap=captured):
                        _cap["input"] = inp[0].detach() if isinstance(inp, tuple) else inp.detach()
                        _cap["output"] = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
                    
                    handle = module.register_forward_hook(hook)
                    
                    try:
                        with torch.no_grad():
                            self.model(input_data)
                        
                        if "input" in captured:
                            results["input_shape"] = tuple(captured["input"].shape)
                        if "output" in captured:
                            out = captured["output"].float()
                            results["output_shape"] = tuple(out.shape)
                            results["stats"] = {
                                "mean": out.mean().item(),
                                "std": out.std().item(),
                                "min": out.min().item(),
                                "max": out.max().item(),
                                "has_nan": torch.isnan(out).any().item(),
                                "has_inf": torch.isinf(out).any().item()
                            }
                    finally:
                        handle.remove()
                    
                    break
            
            return results
        
        def compare_layers(
            self,
            input_data: torch.Tensor,
            layer_names: list[str]
        ) -> dict[str, dict[str, Any]]:
            """Compare activations across multiple layers."""
            results = {}
            for name in layer_names:
                results[name] = self.inspect_layer(name, input_data)
            return results
    
    
    class GradientAnalyzer:
        """
        Analyze gradient flow through the network.
        """
        
        def __init__(self, model: nn.Module):
            self.model = model
            self._gradient_data: dict[str, torch.Tensor] = {}
            self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        
        def analyze_gradients(
            self,
            input_data: torch.Tensor,
            target: torch.Tensor = None,
            loss_fn: Callable = None
        ) -> dict[str, dict[str, float]]:
            """
            Analyze gradient flow.
            
            Args:
                input_data: Input tensor
                target: Target for loss computation
                loss_fn: Loss function (defaults to dummy loss)
            
            Returns:
                Gradient statistics per layer
            """
            self._gradient_data.clear()
            
            # Register hooks for all parameters
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.register_hook(
                        self._create_grad_hook(name)
                    )
            
            # Forward pass
            output = self.model(input_data)
            
            # Compute loss
            if loss_fn is None:
                if isinstance(output, tuple):
                    output = output[0]
                loss = output.sum()  # Dummy loss
            else:
                loss = loss_fn(output, target)
            
            # Backward pass
            loss.backward()
            
            # Compute statistics
            results = {}
            for name, grad in self._gradient_data.items():
                if grad is not None:
                    grad_flat = grad.float().flatten()
                    results[name] = {
                        "mean": grad_flat.mean().item(),
                        "std": grad_flat.std().item(),
                        "min": grad_flat.min().item(),
                        "max": grad_flat.max().item(),
                        "norm": grad_flat.norm().item(),
                        "has_nan": torch.isnan(grad_flat).any().item(),
                        "has_inf": torch.isinf(grad_flat).any().item(),
                        "near_zero": (grad_flat.abs() < 1e-7).float().mean().item()
                    }
            
            return results
        
        def _create_grad_hook(self, name: str):
            """Create gradient capture hook."""
            def hook(grad):
                self._gradient_data[name] = grad.detach().clone()
            return hook
        
        def find_gradient_issues(
            self,
            results: dict[str, dict[str, float]]
        ) -> dict[str, list[str]]:
            """
            Find gradient-related issues.
            
            Args:
                results: Gradient statistics
            
            Returns:
                Dict mapping issue type to layer names
            """
            issues = defaultdict(list)
            
            for name, stats in results.items():
                if stats["has_nan"]:
                    issues["nan_gradients"].append(name)
                
                if stats["has_inf"]:
                    issues["inf_gradients"].append(name)
                
                if stats["norm"] > 100:
                    issues["exploding_gradients"].append(name)
                
                if stats["norm"] < 1e-7:
                    issues["vanishing_gradients"].append(name)
                
                if stats["near_zero"] > 0.9:
                    issues["dead_gradients"].append(name)
            
            return dict(issues)
    
    
    class ModelDebugger:
        """
        Comprehensive model debugging toolkit.
        """
        
        def __init__(self, model: nn.Module, model_name: str = "model"):
            self.model = model
            self.model_name = model_name
            self.layer_inspector = LayerInspector(model)
            self.gradient_analyzer = GradientAnalyzer(model)
            
            self._hooks: list[torch.utils.hooks.RemovableHandle] = []
            self._layer_activations: dict[str, torch.Tensor] = {}
            self._layer_inputs: dict[str, torch.Tensor] = {}
        
        def full_diagnostic(
            self,
            input_data: torch.Tensor,
            run_gradients: bool = True
        ) -> DebugReport:
            """
            Run complete diagnostic on model.
            
            Args:
                input_data: Sample input
                run_gradients: Whether to analyze gradients
            
            Returns:
                DebugReport with findings
            """
            global_issues = []
            recommendations = []
            layer_diagnostics = []
            
            # Capture all activations
            self._capture_activations()
            
            try:
                with torch.no_grad():
                    self.model(input_data)
            except Exception as e:
                global_issues.append(f"Forward pass failed: {str(e)}")
            
            self._clear_hooks()
            
            # Analyze each layer
            for name, module in self.model.named_modules():
                if name in self._layer_activations:
                    diag = self._diagnose_layer(
                        name, 
                        module,
                        self._layer_inputs.get(name),
                        self._layer_activations.get(name)
                    )
                    layer_diagnostics.append(diag)
            
            # Gradient analysis
            if run_gradients:
                try:
                    self.model.train()
                    grad_results = self.gradient_analyzer.analyze_gradients(input_data)
                    grad_issues = self.gradient_analyzer.find_gradient_issues(grad_results)
                    
                    for issue_type, layers in grad_issues.items():
                        if layers:
                            global_issues.append(f"{issue_type}: {len(layers)} layers")
                    
                    # Update layer diagnostics with gradient info
                    for diag in layer_diagnostics:
                        param_name = f"{diag.name}.weight"
                        if param_name in grad_results:
                            diag.gradient_stats = grad_results[param_name]
                    
                    self.model.eval()
                except Exception as e:
                    global_issues.append(f"Gradient analysis failed: {str(e)}")
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                layer_diagnostics, global_issues
            )
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            
            return DebugReport(
                model_name=self.model_name,
                timestamp=datetime.now(),
                total_params=total_params,
                layer_diagnostics=layer_diagnostics,
                global_issues=global_issues,
                recommendations=recommendations
            )
        
        def _capture_activations(self):
            """Set up hooks to capture activations."""
            self._layer_activations.clear()
            self._layer_inputs.clear()
            
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules
                    handle = module.register_forward_hook(
                        self._create_activation_hook(name)
                    )
                    self._hooks.append(handle)
        
        def _create_activation_hook(self, name: str):
            """Create hook to capture layer activations."""
            def hook(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    inp = input[0]
                    if isinstance(inp, torch.Tensor):
                        self._layer_inputs[name] = inp.detach()
                
                if isinstance(output, torch.Tensor):
                    self._layer_activations[name] = output.detach()
                elif isinstance(output, tuple):
                    for out in output:
                        if isinstance(out, torch.Tensor):
                            self._layer_activations[name] = out.detach()
                            break
            
            return hook
        
        def _clear_hooks(self):
            """Remove all hooks."""
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()
        
        def _diagnose_layer(
            self,
            name: str,
            module: nn.Module,
            input_tensor: Optional[torch.Tensor],
            output_tensor: Optional[torch.Tensor]
        ) -> LayerDiagnostics:
            """Generate diagnostics for a single layer."""
            issues = []
            activation_stats = {}
            
            input_shape = tuple(input_tensor.shape) if input_tensor is not None else ()
            output_shape = tuple(output_tensor.shape) if output_tensor is not None else ()
            
            has_nan = False
            has_inf = False
            
            if output_tensor is not None:
                out = output_tensor.float()
                has_nan = torch.isnan(out).any().item()
                has_inf = torch.isinf(out).any().item()
                
                activation_stats = {
                    "mean": out.mean().item(),
                    "std": out.std().item(),
                    "min": out.min().item(),
                    "max": out.max().item(),
                    "sparsity": (out.abs() < 1e-6).float().mean().item()
                }
                
                # Check for issues
                if has_nan:
                    issues.append("NaN values in activations")
                if has_inf:
                    issues.append("Inf values in activations")
                if activation_stats["std"] < 1e-6:
                    issues.append("Near-constant activations (collapsed)")
                if activation_stats["sparsity"] > 0.9:
                    issues.append("Very sparse activations (>90% near zero)")
                if activation_stats["max"] > 1e4:
                    issues.append("Very large activation values")
            
            # Count parameters
            num_params = sum(p.numel() for p in module.parameters())
            
            return LayerDiagnostics(
                name=name,
                module_type=type(module).__name__,
                input_shape=input_shape,
                output_shape=output_shape,
                num_params=num_params,
                has_nan_activations=has_nan,
                has_inf_activations=has_inf,
                activation_stats=activation_stats,
                issues=issues
            )
        
        def _generate_recommendations(
            self,
            diagnostics: list[LayerDiagnostics],
            issues: list[str]
        ) -> list[str]:
            """Generate actionable recommendations."""
            recommendations = []
            
            # Check for NaN/Inf
            nan_layers = [d.name for d in diagnostics if d.has_nan_activations]
            inf_layers = [d.name for d in diagnostics if d.has_inf_activations]
            
            if nan_layers:
                recommendations.append(
                    f"NaN activations detected in {len(nan_layers)} layers. "
                    "Consider: gradient clipping, lower learning rate, or checking input data."
                )
            
            if inf_layers:
                recommendations.append(
                    f"Inf activations in {len(inf_layers)} layers. "
                    "Consider: mixed precision adjustments or numerical stability fixes."
                )
            
            # Check for collapsed layers
            collapsed = [
                d.name for d in diagnostics 
                if d.activation_stats.get("std", 1) < 1e-6
            ]
            if collapsed:
                recommendations.append(
                    f"{len(collapsed)} layers have collapsed activations. "
                    "Check for mode collapse, bad initialization, or architectural issues."
                )
            
            # Check gradients
            grad_issues = [i for i in issues if "gradient" in i.lower()]
            if grad_issues:
                recommendations.append(
                    "Gradient issues detected. Consider: gradient clipping, "
                    "residual connections, or layer normalization."
                )
            
            # General
            if not recommendations:
                recommendations.append("No critical issues detected.")
            
            return recommendations
        
        def watch(
            self,
            callback: Callable[[str, torch.Tensor], None],
            layer_names: list[str] = None
        ):
            """
            Set up continuous monitoring.
            
            Args:
                callback: Function called with (layer_name, activation)
                layer_names: Specific layers to watch (None = all)
            """
            for name, module in self.model.named_modules():
                if layer_names is None or name in layer_names:
                    def make_hook(n):
                        def hook(mod, inp, out):
                            if isinstance(out, torch.Tensor):
                                callback(n, out.detach())
                        return hook
                    
                    handle = module.register_forward_hook(make_hook(name))
                    self._hooks.append(handle)
        
        def stop_watching(self):
            """Stop continuous monitoring."""
            self._clear_hooks()


    def debug_model(
        model: nn.Module,
        sample_input: torch.Tensor,
        verbose: bool = True
    ) -> DebugReport:
        """
        Quick utility to debug a model.
        
        Args:
            model: Model to debug
            sample_input: Sample input tensor
            verbose: Print summary
        
        Returns:
            DebugReport
        """
        debugger = ModelDebugger(model)
        report = debugger.full_diagnostic(sample_input)
        
        if verbose:
            print(f"=== Debug Report: {report.model_name} ===")
            print(f"Total parameters: {report.total_params:,}")
            print(f"Layers analyzed: {len(report.layer_diagnostics)}")
            print(f"\nGlobal Issues ({len(report.global_issues)}):")
            for issue in report.global_issues:
                print(f"  - {issue}")
            print(f"\nRecommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")
        
        return report

else:
    class LayerInspector:
        pass
    
    class GradientAnalyzer:
        pass
    
    class ModelDebugger:
        pass
    
    def debug_model(*args, **kwargs):
        raise ImportError("PyTorch required for model debugging")
