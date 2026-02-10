"""
Model Pruning

Remove unimportant weights to reduce model size and improve inference speed.
Supports structured and unstructured pruning with various criteria.

FILE: enigma_engine/core/pruning.py
TYPE: Core/Optimization
MAIN CLASSES: Pruner, StructuredPruner, UnstructuredPruner, PruningScheduler
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

try:
    import numpy as np
except ImportError:
    np = None


class PruningMethod(Enum):
    """Pruning methods."""
    MAGNITUDE = "magnitude"  # Remove smallest weights
    RANDOM = "random"  # Random pruning
    GRADIENT = "gradient"  # Based on gradient importance
    TAYLOR = "taylor"  # Taylor expansion importance
    MOVEMENT = "movement"  # Movement pruning during training


class PruningType(Enum):
    """Types of pruning."""
    UNSTRUCTURED = "unstructured"  # Individual weights
    STRUCTURED = "structured"  # Entire neurons/filters
    SEMI_STRUCTURED = "semi_structured"  # N:M sparsity patterns


@dataclass
class PruningConfig:
    """Configuration for pruning."""
    method: PruningMethod = PruningMethod.MAGNITUDE
    pruning_type: PruningType = PruningType.UNSTRUCTURED
    
    # Target sparsity
    target_sparsity: float = 0.5  # 50% of weights removed
    
    # Gradual pruning
    initial_sparsity: float = 0.0
    final_sparsity: float = 0.5
    pruning_steps: int = 100
    
    # Which layers to prune
    prune_embeddings: bool = False
    prune_attention: bool = True
    prune_ffn: bool = True
    prune_output: bool = False
    
    # Semi-structured settings
    n_sparse: int = 2
    m_block: int = 4  # 2:4 sparsity by default
    
    # Retraining
    retrain_epochs: int = 3
    learning_rate: float = 1e-5


@dataclass
class PruningMask:
    """Pruning mask for a layer."""
    name: str
    mask: Any  # torch.Tensor if available
    sparsity: float
    original_params: int
    pruned_params: int
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sparsity": self.sparsity,
            "original_params": self.original_params,
            "pruned_params": self.pruned_params
        }


class ImportanceEstimator:
    """Estimate importance of weights for pruning decisions."""
    
    def __init__(self, method: PruningMethod):
        self.method = method
        self._gradients: dict[str, Any] = {}
        self._activations: dict[str, Any] = {}
    
    def compute_importance(
        self,
        weight: "torch.Tensor",
        name: str = ""
    ) -> "torch.Tensor":
        """
        Compute importance scores for weights.
        
        Args:
            weight: Weight tensor
            name: Layer name
        
        Returns:
            Importance scores (same shape as weight)
        """
        if self.method == PruningMethod.MAGNITUDE:
            return weight.abs()
        
        elif self.method == PruningMethod.RANDOM:
            return torch.rand_like(weight)
        
        elif self.method == PruningMethod.GRADIENT:
            if name in self._gradients:
                return (weight * self._gradients[name]).abs()
            return weight.abs()
        
        elif self.method == PruningMethod.TAYLOR:
            # First-order Taylor expansion
            if name in self._gradients:
                return (weight * self._gradients[name]).abs()
            return weight.abs()
        
        elif self.method == PruningMethod.MOVEMENT:
            # Track movement from initial weights
            if name in self._activations:
                initial = self._activations[name]
                movement = (weight - initial).abs()
                return movement
            return weight.abs()
        
        return weight.abs()
    
    def collect_gradients(self, model: nn.Module):
        """Collect gradients for importance estimation."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self._gradients[name] = param.grad.clone()
    
    def save_initial_weights(self, model: nn.Module):
        """Save initial weights for movement pruning."""
        for name, param in model.named_parameters():
            self._activations[name] = param.data.clone()


class UnstructuredPruner:
    """Prune individual weights (fine-grained)."""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.importance = ImportanceEstimator(config.method)
        self.masks: dict[str, PruningMask] = {}
    
    def compute_mask(
        self,
        weight: "torch.Tensor",
        sparsity: float,
        name: str = ""
    ) -> "torch.Tensor":
        """
        Compute pruning mask for a weight tensor.
        
        Args:
            weight: Weight tensor to prune
            sparsity: Target sparsity (0.0-1.0)
            name: Layer name
        
        Returns:
            Binary mask (1 = keep, 0 = prune)
        """
        importance = self.importance.compute_importance(weight, name)
        
        # Flatten for threshold computation
        flat_importance = importance.flatten()
        k = int(len(flat_importance) * sparsity)
        
        if k <= 0:
            return torch.ones_like(weight)
        
        # Find threshold
        threshold = flat_importance.kthvalue(k).values
        
        # Create mask
        mask = (importance > threshold).float()
        
        # Store mask info
        self.masks[name] = PruningMask(
            name=name,
            mask=mask,
            sparsity=1 - mask.mean().item(),
            original_params=weight.numel(),
            pruned_params=int(weight.numel() * sparsity)
        )
        
        return mask
    
    def apply_mask(
        self,
        weight: "torch.Tensor",
        mask: "torch.Tensor"
    ) -> "torch.Tensor":
        """Apply pruning mask to weights."""
        return weight * mask
    
    def prune_layer(
        self,
        module: nn.Module,
        sparsity: float,
        name: str = ""
    ):
        """Prune a single layer."""
        if hasattr(module, "weight"):
            mask = self.compute_mask(module.weight.data, sparsity, name)
            module.weight.data = self.apply_mask(module.weight.data, mask)


class StructuredPruner:
    """Prune entire neurons/filters (coarse-grained)."""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.importance = ImportanceEstimator(config.method)
        self.pruned_indices: dict[str, list[int]] = {}
    
    def compute_neuron_importance(
        self,
        weight: "torch.Tensor",
        dim: int = 0
    ) -> "torch.Tensor":
        """
        Compute importance of each neuron/filter.
        
        Args:
            weight: Weight tensor (out_features, in_features, ...)
            dim: Dimension to prune (0 for output neurons)
        
        Returns:
            Importance score per neuron
        """
        # Sum absolute values along input dimension
        other_dims = list(range(weight.dim()))
        other_dims.remove(dim)
        
        return weight.abs().sum(dim=other_dims)
    
    def prune_linear(
        self,
        module: nn.Linear,
        sparsity: float,
        dim: str = "output"
    ) -> nn.Linear:
        """
        Prune a linear layer by removing neurons.
        
        Args:
            module: Linear layer
            sparsity: Target sparsity
            dim: "output" or "input"
        
        Returns:
            Pruned linear layer
        """
        weight = module.weight.data
        
        if dim == "output":
            importance = self.compute_neuron_importance(weight, dim=0)
            num_prune = int(module.out_features * sparsity)
            
            if num_prune <= 0:
                return module
            
            # Find neurons to keep
            _, keep_indices = importance.topk(module.out_features - num_prune)
            keep_indices = keep_indices.sort().values
            
            # Create new smaller layer
            new_out = len(keep_indices)
            new_module = nn.Linear(module.in_features, new_out, bias=module.bias is not None)
            new_module.weight.data = weight[keep_indices]
            
            if module.bias is not None:
                new_module.bias.data = module.bias.data[keep_indices]
            
            return new_module
        
        else:  # input
            importance = self.compute_neuron_importance(weight, dim=1)
            num_prune = int(module.in_features * sparsity)
            
            if num_prune <= 0:
                return module
            
            _, keep_indices = importance.topk(module.in_features - num_prune)
            keep_indices = keep_indices.sort().values
            
            new_in = len(keep_indices)
            new_module = nn.Linear(new_in, module.out_features, bias=module.bias is not None)
            new_module.weight.data = weight[:, keep_indices]
            
            if module.bias is not None:
                new_module.bias.data = module.bias.data
            
            return new_module


class SemiStructuredPruner:
    """N:M sparsity pruning (hardware-friendly patterns)."""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.n = config.n_sparse
        self.m = config.m_block
    
    def compute_nm_mask(self, weight: "torch.Tensor") -> "torch.Tensor":
        """
        Compute N:M sparsity mask.
        
        Keeps N largest values in each M consecutive elements.
        """
        shape = weight.shape
        flat = weight.flatten()
        
        # Pad to multiple of M
        padding = (self.m - len(flat) % self.m) % self.m
        if padding > 0:
            flat = torch.cat([flat, torch.zeros(padding, device=flat.device)])
        
        # Reshape to blocks
        blocks = flat.reshape(-1, self.m)
        
        # Keep top N in each block
        _, indices = blocks.abs().topk(self.n, dim=1)
        
        # Create mask
        mask = torch.zeros_like(blocks)
        mask.scatter_(1, indices, 1)
        
        # Reshape back
        mask = mask.flatten()[:shape.numel()].reshape(shape)
        
        return mask
    
    def prune_layer(self, module: nn.Module):
        """Apply N:M pruning to a layer."""
        if hasattr(module, "weight"):
            mask = self.compute_nm_mask(module.weight.data)
            module.weight.data *= mask


class PruningScheduler:
    """Schedule gradual pruning during training."""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.current_step = 0
    
    def get_sparsity(self, step: int = None) -> float:
        """
        Get sparsity for current/given step.
        
        Uses cubic interpolation for smooth pruning schedule.
        """
        step = step if step is not None else self.current_step
        
        if step >= self.config.pruning_steps:
            return self.config.final_sparsity
        
        if step <= 0:
            return self.config.initial_sparsity
        
        # Cubic interpolation
        progress = step / self.config.pruning_steps
        sparsity_delta = self.config.final_sparsity - self.config.initial_sparsity
        
        # Cubic ease-in-out
        if progress < 0.5:
            factor = 4 * progress ** 3
        else:
            factor = 1 - ((-2 * progress + 2) ** 3) / 2
        
        return self.config.initial_sparsity + sparsity_delta * factor
    
    def step(self):
        """Advance scheduler by one step."""
        self.current_step += 1
    
    def should_prune(self, step: int = None) -> bool:
        """Check if pruning should be applied at this step."""
        step = step if step is not None else self.current_step
        
        # Prune every 10 steps during pruning phase
        if step >= self.config.pruning_steps:
            return False
        
        return step % 10 == 0


class Pruner:
    """
    Main pruning coordinator.
    
    Handles model pruning with various methods and schedules.
    """
    
    def __init__(self, config: PruningConfig = None):
        self.config = config or PruningConfig()
        
        if self.config.pruning_type == PruningType.UNSTRUCTURED:
            self.pruner = UnstructuredPruner(self.config)
        elif self.config.pruning_type == PruningType.STRUCTURED:
            self.pruner = StructuredPruner(self.config)
        else:
            self.pruner = SemiStructuredPruner(self.config)
        
        self.scheduler = PruningScheduler(self.config)
    
    def prune(
        self,
        model: nn.Module,
        sparsity: float = None
    ) -> nn.Module:
        """
        Prune a model to target sparsity.
        
        Args:
            model: Model to prune
            sparsity: Target sparsity (uses config default if None)
        
        Returns:
            Pruned model
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for pruning")
        
        sparsity = sparsity if sparsity is not None else self.config.target_sparsity
        
        # Identify layers to prune
        for name, module in model.named_modules():
            if self._should_prune_layer(name, module):
                logger.debug(f"Pruning layer: {name}")
                
                if isinstance(self.pruner, UnstructuredPruner):
                    self.pruner.prune_layer(module, sparsity, name)
                elif isinstance(self.pruner, StructuredPruner):
                    if isinstance(module, nn.Linear):
                        # Replace with pruned version
                        parent = self._get_parent(model, name)
                        attr_name = name.split(".")[-1]
                        setattr(parent, attr_name, 
                               self.pruner.prune_linear(module, sparsity))
                elif isinstance(self.pruner, SemiStructuredPruner):
                    self.pruner.prune_layer(module)
        
        return model
    
    def _should_prune_layer(self, name: str, module: nn.Module) -> bool:
        """Check if a layer should be pruned."""
        if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            return False
        
        # Check layer type
        name_lower = name.lower()
        
        if "embed" in name_lower and not self.config.prune_embeddings:
            return False
        
        if ("attn" in name_lower or "attention" in name_lower) and not self.config.prune_attention:
            return False
        
        if ("mlp" in name_lower or "ffn" in name_lower or "fc" in name_lower):
            if not self.config.prune_ffn:
                return False
        
        if ("output" in name_lower or "head" in name_lower or "lm_head" in name_lower):
            if not self.config.prune_output:
                return False
        
        return True
    
    def _get_parent(self, model: nn.Module, name: str) -> nn.Module:
        """Get parent module of a named module."""
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent
    
    def get_sparsity_report(self, model: nn.Module) -> dict[str, Any]:
        """
        Get sparsity report for a model.
        
        Returns:
            Dict with layer-wise sparsity statistics
        """
        report = {
            "total_params": 0,
            "pruned_params": 0,
            "sparsity": 0.0,
            "layers": []
        }
        
        for name, module in model.named_modules():
            if hasattr(module, "weight"):
                weight = module.weight.data
                total = weight.numel()
                zeros = (weight == 0).sum().item()
                
                report["total_params"] += total
                report["pruned_params"] += zeros
                report["layers"].append({
                    "name": name,
                    "total": total,
                    "pruned": zeros,
                    "sparsity": zeros / total if total > 0 else 0
                })
        
        if report["total_params"] > 0:
            report["sparsity"] = report["pruned_params"] / report["total_params"]
        
        return report
    
    def save_pruned_model(
        self,
        model: nn.Module,
        path: str,
        sparse_format: bool = True
    ):
        """
        Save pruned model.
        
        Args:
            model: Pruned model
            path: Save path
            sparse_format: Use sparse tensor format for storage
        """
        state_dict = model.state_dict()
        
        if sparse_format:
            # Convert to sparse format for smaller file size
            for key in state_dict:
                if state_dict[key].is_sparse:
                    continue
                tensor = state_dict[key]
                sparsity = (tensor == 0).sum().item() / tensor.numel()
                
                if sparsity > 0.5:
                    state_dict[key] = tensor.to_sparse()
        
        torch.save({
            "model_state_dict": state_dict,
            "pruning_config": self.config.__dict__,
            "masks": {
                name: mask.to_dict() 
                for name, mask in getattr(self.pruner, "masks", {}).items()
            }
        }, path)
        
        logger.info(f"Saved pruned model to {path}")


def prune_model(
    model: nn.Module,
    sparsity: float = 0.5,
    method: str = "magnitude",
    pruning_type: str = "unstructured"
) -> nn.Module:
    """
    Convenience function to prune a model.
    
    Args:
        model: Model to prune
        sparsity: Target sparsity (0.0-1.0)
        method: Pruning method (magnitude, random, gradient, taylor, movement)
        pruning_type: Type of pruning (unstructured, structured, semi_structured)
    
    Returns:
        Pruned model
    """
    config = PruningConfig(
        method=PruningMethod(method),
        pruning_type=PruningType(pruning_type),
        target_sparsity=sparsity
    )
    
    pruner = Pruner(config)
    return pruner.prune(model, sparsity)
