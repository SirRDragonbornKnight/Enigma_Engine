"""
Model Merging and Mixing for Enigma AI Engine

Combine multiple models into a single merged model.

Features:
- Weight interpolation (LERP, SLERP)
- Task arithmetic (addition, negation)
- TIES merging
- Frankenstein merging (layer mixing)
- Compatibility checking
- Quality validation

Usage:
    from enigma_engine.core.model_merging import ModelMerger, merge_models
    
    # Quick merge
    merged = merge_models(
        ["models/model_a", "models/model_b"],
        weights=[0.7, 0.3],
        output="models/merged"
    )
    
    # Advanced merging
    merger = ModelMerger()
    merger.add_model("models/base", weight=0.5)
    merger.add_model("models/creative", weight=0.3)
    merger.add_model("models/factual", weight=0.2)
    merger.merge(method="slerp", output="models/merged")
"""

import json
import logging
import math
import shutil
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class MergeMethod(Enum):
    """Model merging methods."""
    LERP = auto()           # Linear interpolation
    SLERP = auto()          # Spherical interpolation
    TIES = auto()           # Trim, Elect, Sign merge
    DARE = auto()           # Drop and Rescale
    FRANKENSTEIN = auto()   # Layer mixing
    TASK_ARITHMETIC = auto() # Addition/subtraction


@dataclass
class ModelSource:
    """A model to be merged."""
    path: str
    weight: float = 1.0
    
    # Layer selection for Frankenstein
    layers: Optional[List[int]] = None
    
    # For task arithmetic
    is_negative: bool = False
    
    # State
    state_dict: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None


@dataclass
class MergeConfig:
    """Configuration for model merging."""
    method: MergeMethod = MergeMethod.LERP
    
    # SLERP parameters
    slerp_t: float = 0.5  # Interpolation factor
    
    # TIES parameters
    ties_density: float = 0.5  # Density for trim
    ties_majority_sign: bool = True
    
    # DARE parameters
    dare_drop_rate: float = 0.1
    
    # Frankenstein parameters
    frank_layer_ranges: Dict[str, List[int]] = field(default_factory=dict)
    
    # Validation
    validate_compatibility: bool = True
    validate_quality: bool = False
    
    # Output
    output_format: str = "pytorch"  # pytorch, safetensors


class ModelMerger:
    """
    Merges multiple models into one.
    """
    
    def __init__(self, config: Optional[MergeConfig] = None):
        """
        Initialize merger.
        
        Args:
            config: Merge configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for model merging")
        
        self._config = config or MergeConfig()
        self._models: List[ModelSource] = []
        self._base_config: Optional[Dict[str, Any]] = None
    
    def add_model(
        self,
        path: str,
        weight: float = 1.0,
        layers: Optional[List[int]] = None,
        negative: bool = False
    ) -> ModelSource:
        """
        Add a model to merge.
        
        Args:
            path: Model path
            weight: Merge weight
            layers: Specific layers to use (for Frankenstein)
            negative: Use as negative (for task arithmetic)
            
        Returns:
            Model source object
        """
        source = ModelSource(
            path=path,
            weight=weight,
            layers=layers,
            is_negative=negative
        )
        
        self._models.append(source)
        logger.info(f"Added model: {path} (weight={weight})")
        
        return source
    
    def load_models(self):
        """Load all added models."""
        for source in self._models:
            source.state_dict, source.config = self._load_model(source.path)
            
            # Set base config from first model
            if self._base_config is None:
                self._base_config = source.config
    
    def _load_model(self, path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load a model's state dict and config."""
        path = Path(path)
        
        # Load state dict
        state_dict = None
        
        # Try different formats
        for filename in ["model.pt", "pytorch_model.bin", "model.safetensors"]:
            state_path = path / filename
            if state_path.exists():
                if filename.endswith(".safetensors"):
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(str(state_path))
                    except ImportError:
                        pass
                else:
                    state_dict = torch.load(state_path, map_location="cpu")
                break
        
        if state_dict is None:
            raise FileNotFoundError(f"No model weights found in {path}")
        
        # Load config
        config = {}
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        
        logger.debug(f"Loaded model from {path}")
        return state_dict, config
    
    def validate_compatibility(self) -> bool:
        """Check if models are compatible for merging."""
        if len(self._models) < 2:
            return True
        
        # Get architectures
        configs = [m.config for m in self._models if m.config]
        
        if not configs:
            logger.warning("No configs to validate")
            return True
        
        base = configs[0]
        
        for i, config in enumerate(configs[1:], 1):
            # Check key dimensions
            for key in ["d_model", "n_layers", "n_heads", "vocab_size"]:
                if key in base and key in config:
                    if base[key] != config[key]:
                        logger.error(
                            f"Incompatible {key}: model 0 has {base[key]}, "
                            f"model {i} has {config[key]}"
                        )
                        return False
        
        logger.info("All models are compatible")
        return True
    
    def merge(
        self,
        method: Optional[MergeMethod | str] = None,
        output: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Merge the models.
        
        Args:
            method: Merge method (overrides config)
            output: Output path (optional, to save directly)
            
        Returns:
            Merged state dict
        """
        if len(self._models) < 2:
            raise ValueError("Need at least 2 models to merge")
        
        # Load models if not loaded
        for source in self._models:
            if source.state_dict is None:
                source.state_dict, source.config = self._load_model(source.path)
                if self._base_config is None:
                    self._base_config = source.config
        
        # Validate
        if self._config.validate_compatibility:
            if not self.validate_compatibility():
                raise ValueError("Models are not compatible")
        
        # Parse method
        if method is None:
            method = self._config.method
        elif isinstance(method, str):
            method = MergeMethod[method.upper()]
        
        # Normalize weights
        total_weight = sum(m.weight for m in self._models)
        for m in self._models:
            m.weight /= total_weight
        
        # Merge based on method
        logger.info(f"Merging {len(self._models)} models using {method.name}")
        
        if method == MergeMethod.LERP:
            merged = self._merge_lerp()
        elif method == MergeMethod.SLERP:
            merged = self._merge_slerp()
        elif method == MergeMethod.TIES:
            merged = self._merge_ties()
        elif method == MergeMethod.DARE:
            merged = self._merge_dare()
        elif method == MergeMethod.FRANKENSTEIN:
            merged = self._merge_frankenstein()
        elif method == MergeMethod.TASK_ARITHMETIC:
            merged = self._merge_task_arithmetic()
        else:
            raise ValueError(f"Unknown merge method: {method}")
        
        # Save if output specified
        if output:
            self._save_merged(merged, output)
        
        return merged
    
    def _merge_lerp(self) -> Dict[str, Any]:
        """Linear interpolation merge."""
        merged = {}
        
        # Get all keys
        all_keys = set()
        for m in self._models:
            all_keys.update(m.state_dict.keys())
        
        for key in all_keys:
            tensors = []
            weights = []
            
            for m in self._models:
                if key in m.state_dict:
                    tensors.append(m.state_dict[key].float())
                    weights.append(m.weight)
            
            if tensors:
                # Weighted average
                total_weight = sum(weights)
                weighted_sum = sum(t * (w / total_weight) for t, w in zip(tensors, weights))
                merged[key] = weighted_sum
        
        return merged
    
    def _merge_slerp(self) -> Dict[str, Any]:
        """Spherical linear interpolation merge."""
        if len(self._models) != 2:
            logger.warning("SLERP works best with 2 models, falling back to LERP")
            return self._merge_lerp()
        
        m1, m2 = self._models
        t = self._config.slerp_t
        
        merged = {}
        
        for key in m1.state_dict.keys():
            if key not in m2.state_dict:
                merged[key] = m1.state_dict[key]
                continue
            
            v1 = m1.state_dict[key].float().flatten()
            v2 = m2.state_dict[key].float().flatten()
            
            # Normalize
            norm1 = torch.norm(v1)
            norm2 = torch.norm(v2)
            
            if norm1 < 1e-8 or norm2 < 1e-8:
                # Fall back to LERP for near-zero tensors
                merged[key] = (1 - t) * m1.state_dict[key].float() + t * m2.state_dict[key].float()
                continue
            
            v1_norm = v1 / norm1
            v2_norm = v2 / norm2
            
            # Compute angle
            dot = torch.clamp(torch.dot(v1_norm, v2_norm), -1.0, 1.0)
            omega = torch.acos(dot)
            
            if omega.abs() < 1e-8:
                # Vectors are parallel
                merged[key] = (1 - t) * m1.state_dict[key].float() + t * m2.state_dict[key].float()
            else:
                # SLERP
                sin_omega = torch.sin(omega)
                coeff1 = torch.sin((1 - t) * omega) / sin_omega
                coeff2 = torch.sin(t * omega) / sin_omega
                
                result = coeff1 * v1 + coeff2 * v2
                merged[key] = result.view_as(m1.state_dict[key])
        
        return merged
    
    def _merge_ties(self) -> Dict[str, Any]:
        """TIES merging: Trim, Elect by Sign, merge."""
        density = self._config.ties_density
        
        # Start with base model
        base_state = self._models[0].state_dict
        merged = {k: v.clone().float() for k, v in base_state.items()}
        
        # Compute task vectors (difference from base)
        task_vectors = []
        for m in self._models[1:]:
            tv = {}
            for key in base_state:
                if key in m.state_dict:
                    tv[key] = m.state_dict[key].float() - base_state[key].float()
            task_vectors.append((tv, m.weight))
        
        for key in base_state:
            deltas = []
            weights = []
            
            for tv, weight in task_vectors:
                if key in tv:
                    deltas.append(tv[key])
                    weights.append(weight)
            
            if not deltas:
                continue
            
            # Stack deltas
            stacked = torch.stack(deltas)
            
            # Trim: set small values to zero
            magnitudes = torch.abs(stacked)
            threshold = torch.quantile(magnitudes.flatten(), 1 - density)
            mask = magnitudes >= threshold
            trimmed = stacked * mask
            
            # Elect by sign: use majority sign
            if self._config.ties_majority_sign:
                signs = torch.sign(trimmed)
                # Count signs
                sign_sum = torch.sum(signs, dim=0)
                elected_sign = torch.sign(sign_sum)
                elected_sign[elected_sign == 0] = 1  # Default to positive
                
                # Only keep values matching elected sign
                sign_match = (signs == elected_sign.unsqueeze(0))
                trimmed = trimmed * sign_match
            
            # Weighted average of remaining
            weight_tensor = torch.tensor(weights).view(-1, *([1] * (trimmed.dim() - 1)))
            weighted = (trimmed * weight_tensor).sum(dim=0)
            
            merged[key] = base_state[key].float() + weighted
        
        return merged
    
    def _merge_dare(self) -> Dict[str, Any]:
        """DARE merging: Drop and Rescale."""
        drop_rate = self._config.dare_drop_rate
        
        # Start with base model
        base_state = self._models[0].state_dict
        merged = {k: v.clone().float() for k, v in base_state.items()}
        
        for m in self._models[1:]:
            for key in base_state:
                if key not in m.state_dict:
                    continue
                
                # Compute delta
                delta = m.state_dict[key].float() - base_state[key].float()
                
                # Random drop mask
                mask = torch.bernoulli(torch.ones_like(delta) * (1 - drop_rate))
                
                # Rescale to compensate for dropped values
                scale = 1.0 / (1 - drop_rate)
                
                # Apply with weight
                merged[key] = merged[key] + m.weight * delta * mask * scale
        
        return merged
    
    def _merge_frankenstein(self) -> Dict[str, Any]:
        """Frankenstein merging: combine layers from different models."""
        merged = {}
        layer_ranges = self._config.frank_layer_ranges
        
        # Get layer count from first model
        n_layers = self._base_config.get("n_layers", 12)
        
        # Build layer assignments
        layer_to_model = {}
        
        for model_name, layers in layer_ranges.items():
            for layer in layers:
                layer_to_model[layer] = model_name
        
        # Default: use first model for unspecified layers
        for i in range(n_layers):
            if i not in layer_to_model:
                layer_to_model[i] = str(self._models[0].path)
        
        # Build merged state dict
        for m in self._models:
            for key, value in m.state_dict.items():
                # Check if this is a layer parameter
                layer_match = None
                for i in range(n_layers):
                    if f"layers.{i}." in key or f"layer.{i}." in key:
                        layer_match = i
                        break
                
                if layer_match is not None:
                    # Use this model's value if it's assigned to this layer
                    if layer_to_model.get(layer_match, "") == str(m.path):
                        merged[key] = value.clone()
                elif key not in merged:
                    # Non-layer parameters (embeddings, head) from first model
                    merged[key] = value.clone()
        
        return merged
    
    def _merge_task_arithmetic(self) -> Dict[str, Any]:
        """Task arithmetic: add/subtract task vectors."""
        # Need a base model and at least one other
        base_state = self._models[0].state_dict
        merged = {k: v.clone().float() for k, v in base_state.items()}
        
        for m in self._models[1:]:
            sign = -1 if m.is_negative else 1
            
            for key in base_state:
                if key in m.state_dict:
                    # Compute and apply task vector
                    delta = m.state_dict[key].float() - base_state[key].float()
                    merged[key] = merged[key] + sign * m.weight * delta
        
        return merged
    
    def _save_merged(self, state_dict: Dict[str, Any], output: str):
        """Save merged model."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        
        # Save state dict
        if self._config.output_format == "safetensors":
            try:
                from safetensors.torch import save_file
                save_file(state_dict, str(output / "model.safetensors"))
            except ImportError:
                torch.save(state_dict, output / "model.pt")
        else:
            torch.save(state_dict, output / "model.pt")
        
        # Save config
        if self._base_config:
            config = self._base_config.copy()
            config["merged_from"] = [m.path for m in self._models]
            config["merge_method"] = self._config.method.name
            config["merge_weights"] = [m.weight for m in self._models]
            
            with open(output / "config.json", "w") as f:
                json.dump(config, f, indent=2)
        
        logger.info(f"Saved merged model to {output}")
    
    def get_models(self) -> List[ModelSource]:
        """Get added models."""
        return self._models.copy()
    
    def clear(self):
        """Clear added models."""
        self._models = []
        self._base_config = None


def merge_models(
    model_paths: List[str],
    weights: Optional[List[float]] = None,
    method: str = "lerp",
    output: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick function to merge models.
    
    Args:
        model_paths: List of model paths
        weights: Optional weights (default: equal)
        method: Merge method
        output: Optional output path
        
    Returns:
        Merged state dict
    """
    merger = ModelMerger()
    
    weights = weights or [1.0] * len(model_paths)
    
    for path, weight in zip(model_paths, weights):
        merger.add_model(path, weight=weight)
    
    return merger.merge(method=method, output=output)
