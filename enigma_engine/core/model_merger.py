"""
Model Merging for Enigma AI Engine

Combine multiple trained models' capabilities into one.

Provides:
- Weight averaging (simple merge)
- TIES merging (task-specific)
- DARE merging (drop and rescale)
- SLERP interpolation
- Linear interpolation with custom ratios

Usage:
    from enigma_engine.core.model_merger import ModelMerger, MergeMethod
    
    merger = ModelMerger()
    
    # Simple average of two models
    merged = merger.merge(
        models=["model_chat", "model_code"],
        method=MergeMethod.AVERAGE,
        output_name="merged_assistant"
    )
    
    # Weighted merge (70% chat, 30% code)
    merged = merger.merge(
        models=["model_chat", "model_code"],
        weights=[0.7, 0.3],
        method=MergeMethod.WEIGHTED,
        output_name="chat_focus"
    )
    
    # SLERP interpolation
    merged = merger.merge(
        models=["model_a", "model_b"],
        method=MergeMethod.SLERP,
        t=0.5,
        output_name="slerp_merged"
    )
"""

import logging
import math
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
    """Available merge methods."""
    AVERAGE = auto()       # Simple weight averaging
    WEIGHTED = auto()      # Weighted average with custom ratios
    SLERP = auto()         # Spherical linear interpolation
    TIES = auto()          # TIES-Merging (trim, elect, merge)
    DARE = auto()          # Drop And REscale
    TASK_ARITHMETIC = auto()  # Task vector arithmetic


@dataclass
class MergeConfig:
    """Configuration for model merging."""
    method: MergeMethod = MergeMethod.AVERAGE
    weights: List[float] = field(default_factory=list)  # For WEIGHTED
    t: float = 0.5                                       # For SLERP
    density: float = 0.5                                 # For DARE
    k: float = 20.0                                      # For TIES
    
    # Output settings
    output_name: str = "merged_model"
    output_dir: Optional[str] = None
    
    # Advanced
    exclude_layers: List[str] = field(default_factory=list)  # Layers to skip
    merge_embeddings: bool = True
    merge_lm_head: bool = True
    
    def __post_init__(self):
        if not self.output_dir:
            from enigma_engine.config import CONFIG
            self.output_dir = str(CONFIG.MODELS_DIR)


@dataclass
class MergeResult:
    """Result of a merge operation."""
    success: bool
    output_path: Optional[str] = None
    method: Optional[MergeMethod] = None
    source_models: List[str] = field(default_factory=list)
    parameter_count: int = 0
    merge_time_seconds: float = 0.0
    error: Optional[str] = None


class ModelMerger:
    """
    Merge multiple models into one.
    
    Supports various merge techniques for combining model capabilities.
    """
    
    def __init__(self):
        self._validate_dependencies()
        logger.info("ModelMerger initialized")
    
    def _validate_dependencies(self):
        """Check required libraries."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Limited functionality.")
    
    def merge(
        self,
        models: List[str],
        method: MergeMethod = MergeMethod.AVERAGE,
        weights: Optional[List[float]] = None,
        output_name: str = "merged_model",
        output_dir: Optional[str] = None,
        **kwargs
    ) -> MergeResult:
        """
        Merge multiple models.
        
        Args:
            models: List of model paths or names
            method: Merge method to use
            weights: Custom weights for weighted merge (sum to 1.0)
            output_name: Name for merged model
            output_dir: Directory to save merged model
            **kwargs: Additional method-specific parameters
                - t: interpolation factor for SLERP (0-1)
                - density: drop density for DARE (0-1)
                - k: scaling factor for TIES
                
        Returns:
            MergeResult with details
        """
        import time
        start_time = time.time()
        
        # Validate
        if len(models) < 2:
            return MergeResult(
                success=False,
                error="Need at least 2 models to merge"
            )
        
        # Create config
        config = MergeConfig(
            method=method,
            weights=weights or [],
            output_name=output_name,
            output_dir=output_dir,
            t=kwargs.get('t', 0.5),
            density=kwargs.get('density', 0.5),
            k=kwargs.get('k', 20.0),
            exclude_layers=kwargs.get('exclude_layers', []),
        )
        
        # Normalize weights if needed
        if method == MergeMethod.WEIGHTED:
            if not config.weights:
                # Equal weights
                config.weights = [1.0 / len(models)] * len(models)
            else:
                # Normalize to sum to 1
                total = sum(config.weights)
                config.weights = [w / total for w in config.weights]
        
        try:
            # Load models
            logger.info(f"Loading {len(models)} models for merging...")
            state_dicts = []
            
            for model_path in models:
                sd = self._load_model_state(model_path)
                if sd is None:
                    return MergeResult(
                        success=False,
                        error=f"Failed to load model: {model_path}"
                    )
                state_dicts.append(sd)
            
            # Verify compatible architectures
            if not self._verify_compatible(state_dicts):
                return MergeResult(
                    success=False,
                    error="Models have incompatible architectures"
                )
            
            # Perform merge
            logger.info(f"Merging with method: {method.name}")
            
            if method == MergeMethod.AVERAGE:
                merged_state = self._merge_average(state_dicts, config)
            elif method == MergeMethod.WEIGHTED:
                merged_state = self._merge_weighted(state_dicts, config)
            elif method == MergeMethod.SLERP:
                merged_state = self._merge_slerp(state_dicts, config)
            elif method == MergeMethod.TIES:
                merged_state = self._merge_ties(state_dicts, config)
            elif method == MergeMethod.DARE:
                merged_state = self._merge_dare(state_dicts, config)
            elif method == MergeMethod.TASK_ARITHMETIC:
                merged_state = self._merge_task_arithmetic(state_dicts, config)
            else:
                return MergeResult(
                    success=False,
                    error=f"Unknown merge method: {method}"
                )
            
            # Save merged model
            output_path = self._save_merged_model(merged_state, config)
            
            # Calculate stats
            param_count = sum(p.numel() for p in merged_state.values() if hasattr(p, 'numel'))
            
            elapsed = time.time() - start_time
            
            logger.info(f"Merge complete: {output_path}")
            logger.info(f"  Parameters: {param_count:,}")
            logger.info(f"  Time: {elapsed:.1f}s")
            
            return MergeResult(
                success=True,
                output_path=output_path,
                method=method,
                source_models=models,
                parameter_count=param_count,
                merge_time_seconds=elapsed
            )
            
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            return MergeResult(
                success=False,
                error=str(e),
                source_models=models
            )
    
    def _load_model_state(self, model_path: str) -> Optional[Dict[str, Any]]:
        """Load model state dict."""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch required for model loading")
            return None
        
        path = Path(model_path)
        
        # Try different locations
        search_paths = [
            path,
            path / "pytorch_model.bin",
            path / "model.pt",
            path / "model.pth",
        ]
        
        # Also check models directory
        from enigma_engine.config import CONFIG
        if not path.is_absolute():
            search_paths.extend([
                CONFIG.MODELS_DIR / model_path / "pytorch_model.bin",
                CONFIG.MODELS_DIR / model_path / "model.pt",
            ])
        
        for p in search_paths:
            if p.exists() and p.is_file():
                try:
                    state_dict = torch.load(p, map_location='cpu', weights_only=True)
                    
                    # Handle wrapped state dicts
                    if 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']
                    elif 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    
                    logger.debug(f"Loaded model from {p}")
                    return state_dict
                    
                except Exception as e:
                    logger.warning(f"Failed to load {p}: {e}")
                    continue
        
        logger.error(f"Could not find model: {model_path}")
        return None
    
    def _verify_compatible(self, state_dicts: List[Dict]) -> bool:
        """Verify models have compatible architectures."""
        if len(state_dicts) < 2:
            return True
        
        reference = state_dicts[0]
        ref_keys = set(reference.keys())
        
        for i, sd in enumerate(state_dicts[1:], 1):
            sd_keys = set(sd.keys())
            
            # Keys should match
            if sd_keys != ref_keys:
                missing = ref_keys - sd_keys
                extra = sd_keys - ref_keys
                
                if missing:
                    logger.warning(f"Model {i} missing keys: {missing}")
                if extra:
                    logger.warning(f"Model {i} has extra keys: {extra}")
                
                # Allow if mostly matching
                overlap = len(ref_keys & sd_keys) / max(len(ref_keys), len(sd_keys))
                if overlap < 0.9:
                    return False
            
            # Shapes should match
            for key in ref_keys & sd_keys:
                if reference[key].shape != sd[key].shape:
                    logger.warning(
                        f"Shape mismatch for {key}: "
                        f"{reference[key].shape} vs {sd[key].shape}"
                    )
                    return False
        
        return True
    
    def _merge_average(
        self,
        state_dicts: List[Dict],
        config: MergeConfig
    ) -> Dict[str, Any]:
        """Simple average of weights."""
        n = len(state_dicts)
        merged = {}
        
        for key in state_dicts[0].keys():
            if self._should_skip_layer(key, config):
                merged[key] = state_dicts[0][key].clone()
                continue
            
            # Average all models
            stacked = torch.stack([sd[key].float() for sd in state_dicts])
            merged[key] = stacked.mean(dim=0)
        
        return merged
    
    def _merge_weighted(
        self,
        state_dicts: List[Dict],
        config: MergeConfig
    ) -> Dict[str, Any]:
        """Weighted average of weights."""
        merged = {}
        weights = config.weights
        
        for key in state_dicts[0].keys():
            if self._should_skip_layer(key, config):
                merged[key] = state_dicts[0][key].clone()
                continue
            
            # Weighted sum
            result = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)
            for sd, w in zip(state_dicts, weights):
                result += sd[key].float() * w
            
            merged[key] = result
        
        return merged
    
    def _merge_slerp(
        self,
        state_dicts: List[Dict],
        config: MergeConfig
    ) -> Dict[str, Any]:
        """
        Spherical linear interpolation.
        
        Better preserves model geometry than linear interpolation.
        Only works with 2 models.
        """
        if len(state_dicts) != 2:
            logger.warning("SLERP requires exactly 2 models, falling back to weighted")
            return self._merge_weighted(state_dicts, config)
        
        t = config.t
        merged = {}
        
        for key in state_dicts[0].keys():
            if self._should_skip_layer(key, config):
                merged[key] = state_dicts[0][key].clone()
                continue
            
            v0 = state_dicts[0][key].float().flatten()
            v1 = state_dicts[1][key].float().flatten()
            
            # SLERP
            dot = torch.sum(v0 * v1)
            dot = torch.clamp(dot / (torch.norm(v0) * torch.norm(v1) + 1e-8), -1, 1)
            theta = torch.acos(dot)
            
            if theta.abs() < 1e-5:
                # Vectors nearly parallel, use linear
                result = (1 - t) * v0 + t * v1
            else:
                sin_theta = torch.sin(theta)
                result = (
                    torch.sin((1 - t) * theta) / sin_theta * v0 +
                    torch.sin(t * theta) / sin_theta * v1
                )
            
            merged[key] = result.reshape(state_dicts[0][key].shape)
        
        return merged
    
    def _merge_ties(
        self,
        state_dicts: List[Dict],
        config: MergeConfig
    ) -> Dict[str, Any]:
        """
        TIES-Merging: Trim, Elect, Merge
        
        1. Trim low-magnitude values
        2. Elect sign by majority vote
        3. Merge remaining values
        """
        k = config.k
        merged = {}
        
        # Use first model as base
        base = state_dicts[0]
        
        for key in base.keys():
            if self._should_skip_layer(key, config):
                merged[key] = base[key].clone()
                continue
            
            # Compute task vectors (difference from base)
            task_vectors = []
            for sd in state_dicts[1:]:
                task_vectors.append(sd[key].float() - base[key].float())
            
            if not task_vectors:
                merged[key] = base[key].clone()
                continue
            
            # Stack task vectors
            stacked = torch.stack(task_vectors)
            
            # Trim: Zero out bottom k% by magnitude
            magnitudes = stacked.abs()
            threshold = torch.quantile(magnitudes.flatten(), k / 100)
            trimmed = torch.where(magnitudes > threshold, stacked, torch.zeros_like(stacked))
            
            # Elect: Sign by majority
            signs = torch.sign(trimmed)
            elected_sign = torch.sign(signs.sum(dim=0) + 1e-8)
            
            # Merge: Average magnitude with elected sign
            mean_mag = trimmed.abs().mean(dim=0)
            merged_delta = elected_sign * mean_mag
            
            merged[key] = base[key].float() + merged_delta
        
        return merged
    
    def _merge_dare(
        self,
        state_dicts: List[Dict],
        config: MergeConfig
    ) -> Dict[str, Any]:
        """
        DARE: Drop And REscale.
        
        Randomly drop weights and rescale remaining.
        """
        density = config.density
        merged = {}
        
        base = state_dicts[0]
        
        for key in base.keys():
            if self._should_skip_layer(key, config):
                merged[key] = base[key].clone()
                continue
            
            # Compute deltas
            deltas = []
            for sd in state_dicts[1:]:
                deltas.append(sd[key].float() - base[key].float())
            
            if not deltas:
                merged[key] = base[key].clone()
                continue
            
            # Average deltas
            mean_delta = torch.stack(deltas).mean(dim=0)
            
            # Random drop mask
            mask = torch.rand_like(mean_delta) < density
            
            # Rescale
            scale = 1.0 / max(density, 1e-8)
            dropped_delta = torch.where(mask, mean_delta * scale, torch.zeros_like(mean_delta))
            
            merged[key] = base[key].float() + dropped_delta
        
        return merged
    
    def _merge_task_arithmetic(
        self,
        state_dicts: List[Dict],
        config: MergeConfig
    ) -> Dict[str, Any]:
        """
        Task arithmetic: Add/subtract task vectors.
        
        Uses weights to scale task contributions.
        """
        merged = {}
        base = state_dicts[0]
        weights = config.weights if config.weights else [1.0] * (len(state_dicts) - 1)
        
        for key in base.keys():
            if self._should_skip_layer(key, config):
                merged[key] = base[key].clone()
                continue
            
            result = base[key].float().clone()
            
            # Add weighted task vectors
            for i, sd in enumerate(state_dicts[1:]):
                if i < len(weights):
                    task_vector = sd[key].float() - base[key].float()
                    result += weights[i] * task_vector
            
            merged[key] = result
        
        return merged
    
    def _should_skip_layer(self, key: str, config: MergeConfig) -> bool:
        """Check if layer should be skipped in merge."""
        # Check exclude list
        for pattern in config.exclude_layers:
            if pattern in key:
                return True
        
        # Skip embeddings if configured
        if not config.merge_embeddings:
            if 'embed' in key.lower():
                return True
        
        # Skip lm_head if configured
        if not config.merge_lm_head:
            if 'lm_head' in key or 'output' in key:
                return True
        
        return False
    
    def _save_merged_model(
        self,
        state_dict: Dict[str, Any],
        config: MergeConfig
    ) -> str:
        """Save merged model to disk."""
        output_dir = Path(config.output_dir) / config.output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "pytorch_model.bin"
        
        # Convert back to appropriate dtype
        for key in state_dict:
            if state_dict[key].dtype == torch.float32:
                # Keep float32 for compatibility
                pass
        
        torch.save(state_dict, output_path)
        
        # Save config
        import json
        config_path = output_dir / "merge_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                "method": config.method.name,
                "weights": config.weights,
                "t": config.t,
                "density": config.density,
                "k": config.k,
            }, f, indent=2)
        
        return str(output_dir)
    
    def preview_merge(
        self,
        models: List[str],
        method: MergeMethod = MergeMethod.AVERAGE,
        layers: int = 5
    ) -> Dict[str, Any]:
        """
        Preview merge statistics without performing full merge.
        
        Args:
            models: Model paths
            method: Merge method
            layers: Number of layers to preview
            
        Returns:
            Preview statistics
        """
        preview = {
            "models": models,
            "method": method.name,
            "compatible": False,
            "layer_stats": []
        }
        
        try:
            state_dicts = []
            for model_path in models:
                sd = self._load_model_state(model_path)
                if sd:
                    state_dicts.append(sd)
            
            if len(state_dicts) < 2:
                return preview
            
            preview["compatible"] = self._verify_compatible(state_dicts)
            
            # Analyze first N layers
            keys = list(state_dicts[0].keys())[:layers]
            
            for key in keys:
                tensors = [sd[key].float() for sd in state_dicts]
                
                # Calculate statistics
                stacked = torch.stack(tensors)
                variance = stacked.var(dim=0).mean().item()
                
                preview["layer_stats"].append({
                    "name": key,
                    "shape": list(tensors[0].shape),
                    "variance": variance,
                    "mean_diff": (tensors[1] - tensors[0]).abs().mean().item()
                })
            
        except Exception as e:
            preview["error"] = str(e)
        
        return preview


# Convenience functions

def merge_models(
    models: List[str],
    output_name: str,
    method: str = "average",
    weights: Optional[List[float]] = None
) -> MergeResult:
    """
    Quick merge utility.
    
    Args:
        models: List of model paths
        output_name: Name for merged model
        method: Merge method (average, weighted, slerp, ties, dare)
        weights: Optional weights for weighted merge
        
    Returns:
        MergeResult
    """
    method_map = {
        "average": MergeMethod.AVERAGE,
        "weighted": MergeMethod.WEIGHTED,
        "slerp": MergeMethod.SLERP,
        "ties": MergeMethod.TIES,
        "dare": MergeMethod.DARE,
        "task_arithmetic": MergeMethod.TASK_ARITHMETIC,
    }
    
    merge_method = method_map.get(method.lower(), MergeMethod.AVERAGE)
    
    merger = ModelMerger()
    return merger.merge(
        models=models,
        method=merge_method,
        weights=weights,
        output_name=output_name
    )
