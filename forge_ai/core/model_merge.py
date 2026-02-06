"""
Model Merging for Forge_AI

Combine multiple models using various merging strategies:
- Linear interpolation (LERP)
- SLERP (Spherical Linear Interpolation)
- TIES (Task Interpolation with Excluded Signs)
- DARE (Drop And REscale)
- Task Arithmetic

Useful for:
- Creating "frankenmodels" with combined capabilities
- Averaging fine-tuned checkpoints
- Model ensembling without inference overhead

Usage:
    from forge_ai.core.model_merge import ModelMerger, MergeMethod
    
    merger = ModelMerger()
    merged = merger.merge(
        models=[model_a, model_b, model_c],
        weights=[0.5, 0.3, 0.2],
        method=MergeMethod.SLERP
    )
"""

import copy
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MergeMethod(Enum):
    """Available model merging methods."""
    LINEAR = "linear"  # Simple weighted average
    SLERP = "slerp"   # Spherical linear interpolation
    TIES = "ties"     # Task interpolation with excluded signs
    DARE = "dare"     # Drop and rescale
    TASK_ARITHMETIC = "task_arithmetic"  # Add/subtract task vectors


@dataclass
class MergeConfig:
    """Configuration for model merging."""
    method: MergeMethod = MergeMethod.LINEAR
    weights: Optional[list[float]] = None  # Weights for each model
    base_model_index: int = 0  # Index of base model (for task arithmetic/TIES)
    density: float = 0.5  # For DARE: fraction of parameters to keep
    normalize: bool = True  # Normalize weights to sum to 1
    exclude_layers: list[str] = None  # Layer names to exclude from merging


class ModelMerger:
    """
    Merge multiple models into one.
    
    Supports various merging strategies optimized for LLMs.
    
    Usage:
        merger = ModelMerger()
        
        # Simple weighted average
        merged = merger.merge(
            models=[model_a, model_b],
            weights=[0.7, 0.3]
        )
        
        # SLERP merge
        merged = merger.merge(
            models=[model_a, model_b],
            method=MergeMethod.SLERP,
            t=0.5  # Interpolation factor
        )
        
        # TIES merge with base model
        merged = merger.merge(
            models=[base_model, fine_tuned_a, fine_tuned_b],
            method=MergeMethod.TIES,
            base_model_index=0
        )
    """
    
    def __init__(self, config: Optional[MergeConfig] = None):
        self.config = config or MergeConfig()
    
    def merge(
        self,
        models: list[nn.Module],
        weights: Optional[list[float]] = None,
        method: Optional[MergeMethod] = None,
        **kwargs
    ) -> nn.Module:
        """
        Merge multiple models into one.
        
        Args:
            models: List of models to merge
            weights: Weight for each model (must sum to 1 for LINEAR)
            method: Merging method to use
            **kwargs: Additional method-specific arguments
        
        Returns:
            Merged model
        """
        if len(models) < 2:
            raise ValueError("Need at least 2 models to merge")
        
        method = method or self.config.method
        weights = weights or self.config.weights
        
        # Normalize weights if needed
        if weights and self.config.normalize:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        logger.info(f"Merging {len(models)} models using {method.value}")
        
        if method == MergeMethod.LINEAR:
            return self._merge_linear(models, weights)
        elif method == MergeMethod.SLERP:
            return self._merge_slerp(models, kwargs.get('t', 0.5))
        elif method == MergeMethod.TIES:
            return self._merge_ties(models, weights, kwargs.get('density', 0.5))
        elif method == MergeMethod.DARE:
            return self._merge_dare(models, weights, kwargs.get('density', 0.5))
        elif method == MergeMethod.TASK_ARITHMETIC:
            return self._merge_task_arithmetic(models, weights)
        else:
            raise ValueError(f"Unknown merge method: {method}")
    
    def _merge_linear(
        self,
        models: list[nn.Module],
        weights: Optional[list[float]] = None
    ) -> nn.Module:
        """
        Linear interpolation (weighted average) of model parameters.
        
        merged_param = w1*param1 + w2*param2 + ... + wn*paramn
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Create copy of first model
        merged = copy.deepcopy(models[0])
        merged_state = merged.state_dict()
        
        # Get all state dicts
        state_dicts = [m.state_dict() for m in models]
        
        # Merge each parameter
        for key in merged_state.keys():
            if self._should_skip_layer(key):
                continue
            
            # Weighted sum
            merged_param = torch.zeros_like(merged_state[key], dtype=torch.float32)
            for state_dict, weight in zip(state_dicts, weights):
                merged_param += weight * state_dict[key].float()
            
            merged_state[key] = merged_param.to(merged_state[key].dtype)
        
        merged.load_state_dict(merged_state)
        return merged
    
    def _merge_slerp(
        self,
        models: list[nn.Module],
        t: float = 0.5
    ) -> nn.Module:
        """
        Spherical linear interpolation between two models.
        
        Preserves the "magnitude" of parameters better than linear interpolation.
        Works best for merging two models.
        """
        if len(models) != 2:
            logger.warning("SLERP works best with exactly 2 models. Using first two.")
            models = models[:2]
        
        merged = copy.deepcopy(models[0])
        state_a = models[0].state_dict()
        state_b = models[1].state_dict()
        merged_state = merged.state_dict()
        
        for key in merged_state.keys():
            if self._should_skip_layer(key):
                continue
            
            a = state_a[key].float().flatten()
            b = state_b[key].float().flatten()
            
            # Normalize
            a_norm = a / (a.norm() + 1e-8)
            b_norm = b / (b.norm() + 1e-8)
            
            # Compute angle between vectors
            dot = torch.clamp(torch.dot(a_norm, b_norm), -1.0, 1.0)
            theta = torch.acos(dot)
            
            if theta.abs() < 1e-6:
                # Vectors are parallel, use linear interpolation
                merged_param = (1 - t) * a + t * b
            else:
                # SLERP formula
                sin_theta = torch.sin(theta)
                merged_param = (
                    torch.sin((1 - t) * theta) / sin_theta * a +
                    torch.sin(t * theta) / sin_theta * b
                )
            
            merged_state[key] = merged_param.view_as(state_a[key]).to(state_a[key].dtype)
        
        merged.load_state_dict(merged_state)
        return merged
    
    def _merge_ties(
        self,
        models: list[nn.Module],
        weights: Optional[list[float]] = None,
        density: float = 0.5
    ) -> nn.Module:
        """
        TIES (Task Interpolation with Excluded Signs) merging.
        
        1. Compute task vectors (fine-tuned - base)
        2. Trim low-magnitude values
        3. Resolve sign conflicts
        4. Merge and add back to base
        """
        if weights is None:
            weights = [1.0 / (len(models) - 1)] * (len(models) - 1)
        
        # Base model is first
        base_state = models[0].state_dict()
        task_vectors = []
        
        # Compute task vectors
        for model in models[1:]:
            task_vec = {}
            model_state = model.state_dict()
            for key in base_state.keys():
                task_vec[key] = model_state[key].float() - base_state[key].float()
            task_vectors.append(task_vec)
        
        merged = copy.deepcopy(models[0])
        merged_state = merged.state_dict()
        
        for key in merged_state.keys():
            if self._should_skip_layer(key):
                continue
            
            # Stack task vectors for this parameter
            stacked = torch.stack([tv[key] for tv in task_vectors])
            
            # Trim: keep top density% by magnitude
            flat = stacked.abs().flatten()
            k = int(flat.numel() * density)
            if k > 0:
                threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
                mask = stacked.abs() >= threshold
                stacked = stacked * mask
            
            # Resolve sign conflicts: use sign of the sum
            signs = torch.sign(stacked.sum(dim=0))
            signs[signs == 0] = 1
            
            # Apply weights and sum
            weights_tensor = torch.tensor(weights).view(-1, *([1] * (stacked.dim() - 1)))
            merged_task = (stacked * weights_tensor.to(stacked.device)).sum(dim=0)
            
            # Apply majority sign
            merged_task = merged_task.abs() * signs
            
            # Add back to base
            merged_state[key] = (base_state[key].float() + merged_task).to(base_state[key].dtype)
        
        merged.load_state_dict(merged_state)
        return merged
    
    def _merge_dare(
        self,
        models: list[nn.Module],
        weights: Optional[list[float]] = None,
        density: float = 0.5
    ) -> nn.Module:
        """
        DARE (Drop And REscale) merging.
        
        1. For each task vector, randomly drop (1-density) of parameters
        2. Rescale remaining parameters by 1/density
        3. Merge with linear interpolation
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        merged = copy.deepcopy(models[0])
        merged_state = merged.state_dict()
        state_dicts = [m.state_dict() for m in models]
        
        for key in merged_state.keys():
            if self._should_skip_layer(key):
                continue
            
            merged_param = torch.zeros_like(merged_state[key], dtype=torch.float32)
            
            for state_dict, weight in zip(state_dicts, weights):
                param = state_dict[key].float()
                
                # Create random mask
                mask = torch.rand_like(param) < density
                
                # Drop and rescale
                dropped = param * mask / density
                
                merged_param += weight * dropped
            
            merged_state[key] = merged_param.to(merged_state[key].dtype)
        
        merged.load_state_dict(merged_state)
        return merged
    
    def _merge_task_arithmetic(
        self,
        models: list[nn.Module],
        weights: Optional[list[float]] = None
    ) -> nn.Module:
        """
        Task Arithmetic merging.
        
        merged = base + sum(weight_i * (model_i - base))
        
        Positive weights add capabilities, negative weights remove them.
        """
        if weights is None:
            weights = [1.0] * (len(models) - 1)
        
        # First model is base
        base_state = models[0].state_dict()
        
        merged = copy.deepcopy(models[0])
        merged_state = merged.state_dict()
        
        for key in merged_state.keys():
            if self._should_skip_layer(key):
                continue
            
            base_param = base_state[key].float()
            delta = torch.zeros_like(base_param)
            
            # Sum weighted task vectors
            for model, weight in zip(models[1:], weights):
                model_param = model.state_dict()[key].float()
                task_vector = model_param - base_param
                delta += weight * task_vector
            
            merged_state[key] = (base_param + delta).to(base_state[key].dtype)
        
        merged.load_state_dict(merged_state)
        return merged
    
    def _should_skip_layer(self, layer_name: str) -> bool:
        """Check if layer should be excluded from merging."""
        if self.config.exclude_layers:
            for pattern in self.config.exclude_layers:
                if pattern in layer_name:
                    return True
        return False
    
    @staticmethod
    def merge_from_files(
        paths: list[str],
        weights: Optional[list[float]] = None,
        method: MergeMethod = MergeMethod.LINEAR,
        output_path: Optional[str] = None,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Merge models from checkpoint files without loading full models.
        
        Memory-efficient for large models.
        
        Args:
            paths: List of checkpoint file paths
            weights: Weights for each model
            method: Merge method (only LINEAR supported for file-based)
            output_path: Optional path to save merged checkpoint
        
        Returns:
            Merged state dict
        """
        if method != MergeMethod.LINEAR:
            logger.warning("File-based merging only supports LINEAR method")
        
        if weights is None:
            weights = [1.0 / len(paths)] * len(paths)
        
        # Load first checkpoint (weights_only=True for security)
        merged_state = torch.load(paths[0], map_location='cpu', weights_only=True)
        
        # Scale by first weight
        for key in merged_state:
            if isinstance(merged_state[key], torch.Tensor):
                merged_state[key] = merged_state[key].float() * weights[0]
        
        # Add remaining checkpoints
        for path, weight in zip(paths[1:], weights[1:]):
            state = torch.load(path, map_location='cpu', weights_only=True)
            for key in merged_state:
                if isinstance(merged_state[key], torch.Tensor):
                    merged_state[key] += state[key].float() * weight
        
        # Convert back to original dtype
        ref_state = torch.load(paths[0], map_location='cpu', weights_only=True)
        for key in merged_state:
            if isinstance(merged_state[key], torch.Tensor):
                merged_state[key] = merged_state[key].to(ref_state[key].dtype)
        
        if output_path:
            torch.save(merged_state, output_path)
            logger.info(f"Saved merged checkpoint to {output_path}")
        
        return merged_state


def merge_models(
    models: list[nn.Module],
    weights: Optional[list[float]] = None,
    method: str = "linear"
) -> nn.Module:
    """
    Convenience function to merge models.
    
    Args:
        models: List of models
        weights: Optional weights (default: equal)
        method: "linear", "slerp", "ties", "dare", or "task_arithmetic"
    
    Returns:
        Merged model
    
    Example:
        merged = merge_models(
            [model_a, model_b, model_c],
            weights=[0.5, 0.3, 0.2],
            method="linear"
        )
    """
    merger = ModelMerger()
    method_enum = MergeMethod(method)
    return merger.merge(models, weights, method_enum)
