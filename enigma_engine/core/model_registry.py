"""
Model Registry - Safe Model Loading and Caching

Provides thread-safe model loading with security checks.
"""
import hashlib
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import torch

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing AI models on disk.
    
    Reads/writes to models/registry.json
    """
    
    def __init__(self, models_dir: Union[str, Path] = None):
        if models_dir is None:
            # Default to models/ in workspace root
            models_dir = Path(__file__).parent.parent.parent / "models"
        self.models_dir = Path(models_dir)
        self.registry_file = self.models_dir / "registry.json"
        self.registry = {"models": {}, "created": datetime.now().isoformat()}
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    self.registry = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load registry: {e}")
    
    def _save_registry(self):
        """Save registry to disk."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)
    
    def list_models(self) -> dict:
        """Get all registered models."""
        return self.registry.get("models", {})
    
    def get_model(self, name: str) -> Optional[dict]:
        """Get info for a specific model."""
        return self.registry.get("models", {}).get(name)
    
    def register_model(self, name: str, info: dict):
        """Register a new model."""
        if "models" not in self.registry:
            self.registry["models"] = {}
        self.registry["models"][name] = info
        self._save_registry()
    
    def remove_model(self, name: str):
        """Remove a model from registry."""
        if "models" in self.registry and name in self.registry["models"]:
            del self.registry["models"][name]
            self._save_registry()
    
    def model_exists(self, name: str) -> bool:
        """Check if a model exists."""
        return name in self.registry.get("models", {})

# Thread-safe lock for model loading
_load_lock = threading.Lock()

# Cache of loaded models
_model_cache: dict[str, Any] = {}


def safe_load_weights(
    path: Union[str, Path],
    map_location: Optional[str] = None,
    weights_only: bool = True,
) -> dict[str, Any]:
    """
    Safely load model weights from a file.
    
    Args:
        path: Path to the model file (.pth, .pt, .bin)
        map_location: Device to load to ('cpu', 'cuda', etc.)
        weights_only: If True, only load weights (safer, recommended)
        
    Returns:
        State dict or checkpoint dict
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        RuntimeError: If the file cannot be loaded
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # Determine map_location
    if map_location is None:
        if torch.cuda.is_available():
            map_location = "cuda"
        else:
            map_location = "cpu"
    
    logger.info(f"Loading weights from {path} to {map_location}")
    
    try:
        # Use weights_only=True for security (prevents arbitrary code execution)
        # Fall back to weights_only=False if it fails (for older checkpoints)
        try:
            checkpoint = torch.load(
                path,
                map_location=map_location,
                weights_only=weights_only
            )
        except Exception as e:
            if weights_only:
                logger.warning(f"weights_only=True failed, retrying with weights_only=False: {e}")
                checkpoint = torch.load(
                    path,
                    map_location=map_location,
                    weights_only=False
                )
            else:
                raise
        
        logger.info(f"Successfully loaded weights from {path}")
        return checkpoint
        
    except Exception as e:
        logger.error(f"Failed to load weights from {path}: {e}")
        raise RuntimeError(f"Failed to load model weights: {e}") from e


def get_state_dict(
    checkpoint: dict[str, Any],
    prefix: str = ""
) -> dict[str, torch.Tensor]:
    """
    Extract state dict from a checkpoint.
    
    Handles various checkpoint formats:
    - Direct state dict
    - {'model_state_dict': ...}
    - {'state_dict': ...}
    - {'model': ...}
    
    Args:
        checkpoint: Loaded checkpoint
        prefix: Optional prefix to strip from keys
        
    Returns:
        State dict
    """
    # Check for common keys
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        # Assume it's already a state dict
        state_dict = checkpoint
    
    # Strip prefix if present
    if prefix:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_state_dict[k[len(prefix):]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    return state_dict


def cache_model(key: str, model: Any) -> None:
    """Cache a model instance."""
    with _load_lock:
        _model_cache[key] = model


def get_cached_model(key: str) -> Optional[Any]:
    """Get a cached model instance."""
    with _load_lock:
        return _model_cache.get(key)


def clear_cache() -> None:
    """Clear the model cache."""
    with _load_lock:
        _model_cache.clear()


def get_model_hash(path: Union[str, Path]) -> str:
    """Get a hash of a model file for caching purposes."""
    path = Path(path)
    if not path.exists():
        return ""
    
    # Use file size and modification time for quick hash
    stat = path.stat()
    return hashlib.md5(f"{path}:{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()


__all__ = [
    'ModelRegistry',
    'safe_load_weights',
    'get_state_dict',
    'cache_model',
    'get_cached_model',
    'clear_cache',
    'get_model_hash',
]
