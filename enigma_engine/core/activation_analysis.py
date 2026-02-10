"""
Activation Analysis

Tools for analyzing model activations, attention patterns,
and intermediate representations during inference.

FILE: enigma_engine/core/activation_analysis.py
TYPE: Core/Analysis
MAIN CLASSES: ActivationRecorder, AttentionAnalyzer, LayerProfiler
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ActivationStats:
    """Statistics for a layer's activations."""
    name: str
    shape: tuple[int, ...]
    mean: float
    std: float
    min_val: float
    max_val: float
    sparsity: float  # % of zeros
    nan_count: int
    inf_count: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "sparsity": self.sparsity,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count
        }


@dataclass
class AttentionPattern:
    """Extracted attention pattern."""
    layer_idx: int
    head_idx: int
    pattern: Any  # np.ndarray or torch.Tensor
    query_tokens: list[str] = field(default_factory=list)
    key_tokens: list[str] = field(default_factory=list)


if HAS_TORCH:
    
    class ActivationRecorder:
        """
        Record and analyze model activations.
        
        Uses forward hooks to capture intermediate activations
        for analysis and debugging.
        """
        
        def __init__(self, model: nn.Module):
            self.model = model
            self._hooks: list[torch.utils.hooks.RemovableHandle] = []
            self._activations: dict[str, torch.Tensor] = {}
            self._gradients: dict[str, torch.Tensor] = {}
            self._recording = False
        
        def start_recording(self, layer_names: list[str] = None):
            """
            Start recording activations.
            
            Args:
                layer_names: Specific layers to record (None = all)
            """
            self.stop_recording()
            self._activations.clear()
            self._gradients.clear()
            
            for name, module in self.model.named_modules():
                if layer_names is None or name in layer_names:
                    # Forward hook for activations
                    handle = module.register_forward_hook(
                        self._create_forward_hook(name)
                    )
                    self._hooks.append(handle)
                    
                    # Backward hook for gradients (if in training mode)
                    handle = module.register_full_backward_hook(
                        self._create_backward_hook(name)
                    )
                    self._hooks.append(handle)
            
            self._recording = True
            logger.debug(f"Recording activations for {len(self._hooks)//2} layers")
        
        def stop_recording(self):
            """Stop recording and remove hooks."""
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()
            self._recording = False
        
        def _create_forward_hook(self, name: str):
            """Create forward hook for a layer."""
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self._activations[name] = output.detach().cpu()
                elif isinstance(output, tuple):
                    # Store first tensor output
                    for out in output:
                        if isinstance(out, torch.Tensor):
                            self._activations[name] = out.detach().cpu()
                            break
            return hook
        
        def _create_backward_hook(self, name: str):
            """Create backward hook for gradients."""
            def hook(module, grad_input, grad_output):
                if grad_output and isinstance(grad_output[0], torch.Tensor):
                    self._gradients[name] = grad_output[0].detach().cpu()
            return hook
        
        def get_activation(self, name: str) -> Optional[torch.Tensor]:
            """Get recorded activation for a layer."""
            return self._activations.get(name)
        
        def get_gradient(self, name: str) -> Optional[torch.Tensor]:
            """Get recorded gradient for a layer."""
            return self._gradients.get(name)
        
        def get_all_activations(self) -> dict[str, torch.Tensor]:
            """Get all recorded activations."""
            return dict(self._activations)
        
        def compute_stats(self) -> list[ActivationStats]:
            """Compute statistics for all recorded activations."""
            stats = []
            
            for name, activation in self._activations.items():
                act_flat = activation.float().flatten()
                
                stats.append(ActivationStats(
                    name=name,
                    shape=tuple(activation.shape),
                    mean=act_flat.mean().item(),
                    std=act_flat.std().item(),
                    min_val=act_flat.min().item(),
                    max_val=act_flat.max().item(),
                    sparsity=(act_flat == 0).float().mean().item() * 100,
                    nan_count=torch.isnan(act_flat).sum().item(),
                    inf_count=torch.isinf(act_flat).sum().item()
                ))
            
            return stats
        
        def find_anomalies(self) -> dict[str, list[str]]:
            """
            Find layers with anomalous activations.
            
            Returns:
                Dict with keys: 'nan', 'inf', 'dead', 'exploding'
            """
            anomalies = defaultdict(list)
            
            for name, activation in self._activations.items():
                act = activation.float()
                
                if torch.isnan(act).any():
                    anomalies['nan'].append(name)
                
                if torch.isinf(act).any():
                    anomalies['inf'].append(name)
                
                # Dead neurons (all zeros)
                if (act == 0).all():
                    anomalies['dead'].append(name)
                
                # Exploding activations
                if act.abs().max() > 1e6:
                    anomalies['exploding'].append(name)
                
                # Vanishing activations
                if act.abs().max() < 1e-6:
                    anomalies['vanishing'].append(name)
            
            return dict(anomalies)
    
    
    class AttentionAnalyzer:
        """
        Analyze attention patterns in transformer models.
        """
        
        def __init__(self, model: nn.Module):
            self.model = model
            self._attention_weights: dict[str, torch.Tensor] = {}
            self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        
        def capture_attention(self):
            """Start capturing attention weights."""
            self._clear_hooks()
            self._attention_weights.clear()
            
            for name, module in self.model.named_modules():
                # Common attention module names
                if any(x in name.lower() for x in ['attention', 'attn']):
                    if hasattr(module, 'forward'):
                        handle = module.register_forward_hook(
                            self._attention_hook(name)
                        )
                        self._hooks.append(handle)
        
        def _attention_hook(self, name: str):
            """Hook to capture attention weights."""
            def hook(module, input, output):
                # Try to extract attention weights from output
                if isinstance(output, tuple) and len(output) > 1:
                    # Many models return (output, attention_weights)
                    for item in output:
                        if isinstance(item, torch.Tensor):
                            if len(item.shape) == 4:  # [batch, heads, seq, seq]
                                self._attention_weights[name] = item.detach().cpu()
                                break
            return hook
        
        def _clear_hooks(self):
            """Remove all hooks."""
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()
        
        def get_attention_patterns(
            self,
            layer_idx: int = None
        ) -> list[AttentionPattern]:
            """
            Get captured attention patterns.
            
            Args:
                layer_idx: Specific layer index (None = all)
            
            Returns:
                List of AttentionPattern objects
            """
            patterns = []
            
            for name, weights in self._attention_weights.items():
                # Extract layer index from name if possible
                idx = self._extract_layer_idx(name)
                
                if layer_idx is not None and idx != layer_idx:
                    continue
                
                # Shape: [batch, heads, seq_len, seq_len]
                batch_size, num_heads, seq_len, _ = weights.shape
                
                for h in range(num_heads):
                    patterns.append(AttentionPattern(
                        layer_idx=idx,
                        head_idx=h,
                        pattern=weights[0, h].numpy() if HAS_NUMPY else weights[0, h]
                    ))
            
            return patterns
        
        def _extract_layer_idx(self, name: str) -> int:
            """Extract layer index from module name."""
            import re
            match = re.search(r'\.(\d+)\.', name)
            if match:
                return int(match.group(1))
            return -1
        
        @staticmethod
        def compute_attention_entropy(pattern: torch.Tensor) -> float:
            """
            Compute entropy of attention distribution.
            
            Higher entropy = more uniform attention.
            """
            # Ensure proper probability distribution
            pattern = pattern.float()
            pattern = pattern / pattern.sum(dim=-1, keepdim=True)
            
            # Compute entropy
            log_pattern = torch.log(pattern + 1e-10)
            entropy = -(pattern * log_pattern).sum(dim=-1).mean()
            
            return entropy.item()
        
        @staticmethod
        def find_attention_heads(
            patterns: list[AttentionPattern],
            pattern_type: str = "local"
        ) -> list[tuple[int, int]]:
            """
            Find attention heads with specific patterns.
            
            Args:
                patterns: List of attention patterns
                pattern_type: 'local', 'global', 'diagonal', 'bos'
            
            Returns:
                List of (layer_idx, head_idx) tuples
            """
            matches = []
            
            for pattern in patterns:
                attn = pattern.pattern
                if isinstance(attn, torch.Tensor):
                    attn = attn.numpy()
                
                seq_len = attn.shape[0]
                
                if pattern_type == "local":
                    # Check if attention is concentrated near diagonal
                    bandwidth = 5
                    local_mass = 0
                    for i in range(seq_len):
                        start = max(0, i - bandwidth)
                        end = min(seq_len, i + bandwidth + 1)
                        local_mass += attn[i, start:end].sum()
                    
                    if local_mass / seq_len > 0.8:
                        matches.append((pattern.layer_idx, pattern.head_idx))
                
                elif pattern_type == "global":
                    # Check if attention is spread out
                    entropy = -(attn * np.log(attn + 1e-10)).sum(axis=-1).mean()
                    max_entropy = np.log(seq_len)
                    if entropy > 0.8 * max_entropy:
                        matches.append((pattern.layer_idx, pattern.head_idx))
                
                elif pattern_type == "diagonal":
                    # Check for strong diagonal
                    diag_mass = np.trace(attn) / seq_len
                    if diag_mass > 0.5:
                        matches.append((pattern.layer_idx, pattern.head_idx))
                
                elif pattern_type == "bos":
                    # Check if attending to first token
                    first_col_mass = attn[:, 0].mean()
                    if first_col_mass > 0.5:
                        matches.append((pattern.layer_idx, pattern.head_idx))
            
            return matches
    
    
    class LayerProfiler:
        """
        Profile computational cost of each layer.
        """
        
        def __init__(self, model: nn.Module):
            self.model = model
            self._layer_times: dict[str, list[float]] = defaultdict(list)
            self._layer_memory: dict[str, int] = {}
            self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        
        def profile(
            self,
            input_ids: torch.Tensor,
            num_runs: int = 10
        ) -> dict[str, dict[str, float]]:
            """
            Profile model layers.
            
            Args:
                input_ids: Sample input
                num_runs: Number of profiling runs
            
            Returns:
                Dict with timing and memory per layer
            """
            
            self._layer_times.clear()
            self._layer_memory.clear()
            
            # Register timing hooks
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules only
                    self._hooks.append(
                        module.register_forward_hook(
                            self._timing_hook(name)
                        )
                    )
            
            # Warm up
            with torch.no_grad():
                self.model(input_ids)
            
            # Profile runs
            for _ in range(num_runs):
                with torch.no_grad():
                    self.model(input_ids)
            
            # Clean up
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()
            
            # Compute stats
            results = {}
            total_time = 0
            
            for name, times in self._layer_times.items():
                avg_time = sum(times) / len(times)
                total_time += avg_time
                results[name] = {
                    "avg_time_ms": avg_time * 1000,
                    "std_time_ms": np.std(times) * 1000 if HAS_NUMPY else 0,
                    "memory_mb": self._layer_memory.get(name, 0) / 1024**2
                }
            
            # Add percentages
            for name in results:
                results[name]["percent_time"] = results[name]["avg_time_ms"] / (total_time * 1000) * 100
            
            return results
        
        def _timing_hook(self, name: str):
            """Create timing hook for a layer."""
            import time
            
            def hook(module, input, output):
                start = time.perf_counter()
                # The forward is already done, so we time by reference
                # This is a simplified version - real profiling would need pre-hooks
                end = time.perf_counter()
                self._layer_times[name].append(end - start)
                
                # Memory tracking
                if isinstance(output, torch.Tensor):
                    self._layer_memory[name] = output.numel() * output.element_size()
            
            return hook
        
        def get_bottlenecks(
            self,
            results: dict[str, dict[str, float]],
            top_k: int = 5
        ) -> list[tuple[str, float]]:
            """
            Get top computational bottlenecks.
            
            Args:
                results: Profiling results
                top_k: Number of top layers
            
            Returns:
                List of (layer_name, percent_time) tuples
            """
            sorted_layers = sorted(
                results.items(),
                key=lambda x: x[1]["percent_time"],
                reverse=True
            )
            return [(name, stats["percent_time"]) for name, stats in sorted_layers[:top_k]]
    
    
    def analyze_model(
        model: nn.Module,
        input_ids: torch.Tensor,
        tokenizer: Any = None
    ) -> dict[str, Any]:
        """
        Comprehensive model analysis.
        
        Args:
            model: Model to analyze
            input_ids: Sample input
            tokenizer: Optional tokenizer for token names
        
        Returns:
            Analysis report
        """
        report = {}
        
        # Activation analysis
        recorder = ActivationRecorder(model)
        recorder.start_recording()
        
        with torch.no_grad():
            model(input_ids)
        
        recorder.stop_recording()
        
        # Stats
        stats = recorder.compute_stats()
        report["activation_stats"] = [s.to_dict() for s in stats]
        
        # Anomalies
        report["anomalies"] = recorder.find_anomalies()
        
        # Layer profiling
        profiler = LayerProfiler(model)
        profile_results = profiler.profile(input_ids)
        report["layer_profile"] = profile_results
        report["bottlenecks"] = profiler.get_bottlenecks(profile_results)
        
        return report

else:
    # Stubs when torch not available
    class ActivationRecorder:
        pass
    
    class AttentionAnalyzer:
        pass
    
    class LayerProfiler:
        pass
    
    def analyze_model(*args, **kwargs):
        raise ImportError("PyTorch required for activation analysis")
