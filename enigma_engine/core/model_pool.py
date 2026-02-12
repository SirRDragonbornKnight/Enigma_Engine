"""
================================================================================
MODEL POOL - EFFICIENT MODEL LIFECYCLE MANAGEMENT
================================================================================

The Model Pool manages loaded AI models efficiently with features like:
- Lazy loading (load on first use)
- LRU eviction (unload least-used when memory tight)
- Resource tracking (GPU/CPU memory usage)
- Preloading hints (load models user will likely need)
- Shared resources (tokenizers, embeddings)

FILE: enigma_engine/core/model_pool.py
TYPE: Resource Management
MAIN CLASS: ModelPool

USAGE:
    from enigma_engine.core.model_pool import ModelPool, get_model_pool
    
    pool = get_model_pool()
    
    # Load a model (lazy - happens on first use)
    model = pool.get_model("forge:small")
    
    # Preload models you'll need soon
    pool.preload("huggingface:Qwen/Qwen2-1.5B-Instruct")
    
    # Release a model (returns to pool, doesn't unload)
    pool.release_model("forge:small")
    
    # Unload least-used models if memory is tight
    pool.evict_lru(target_memory_mb=4000)
"""

import gc
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import psutil

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL POOL CONFIGURATION
# =============================================================================

@dataclass
class ModelPoolConfig:
    """Configuration for the model pool."""
    
    # Maximum number of models to keep loaded
    max_loaded_models: int = 3
    
    # Maximum total GPU memory usage (MB, 0 = no limit)
    gpu_memory_limit_mb: int = 8000
    
    # Maximum total CPU memory usage (MB, 0 = no limit)
    cpu_memory_limit_mb: int = 16000
    
    # Enable automatic LRU eviction when memory is tight
    enable_auto_eviction: bool = True
    
    # Offload to CPU when GPU memory is full
    fallback_to_cpu: bool = True
    
    # Enable disk offloading for very large models
    enable_disk_offload: bool = False
    
    # Directory for disk-offloaded weights
    offload_dir: Optional[str] = None
    
    # Preload models on startup
    preload_models: list[str] = field(default_factory=list)
    
    # Share tokenizers across models when possible
    share_tokenizers: bool = True
    
    # Share embeddings across models when possible
    share_embeddings: bool = True


# =============================================================================
# MODEL ENTRY
# =============================================================================

@dataclass
class ModelEntry:
    """A model in the pool with its metadata."""
    
    model_id: str                              # Unique identifier
    model: Any                                 # The actual model instance
    model_type: str                            # "forge", "huggingface", "gguf", etc.
    device: str                                # "cpu", "cuda", "mps"
    memory_mb: float                           # Estimated memory usage
    loaded_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used_at: str = field(default_factory=lambda: datetime.now().isoformat())
    use_count: int = 0
    in_use: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def mark_used(self) -> None:
        """Mark model as being used."""
        self.last_used_at = datetime.now().isoformat()
        self.use_count += 1
        self.in_use = True
    
    def mark_released(self) -> None:
        """Mark model as released."""
        self.last_used_at = datetime.now().isoformat()
        self.in_use = False


# =============================================================================
# MODEL POOL
# =============================================================================

class ModelPool:
    """
    Manages a pool of loaded AI models with efficient resource management.
    
    Features:
    - Lazy loading: Models loaded on first use
    - LRU eviction: Unload least-recently-used when memory is tight
    - Resource tracking: Monitor GPU/CPU memory usage
    - Preloading: Load models before they're needed
    - Shared resources: Share tokenizers and embeddings
    """
    
    def __init__(self, config: Optional[ModelPoolConfig] = None):
        """
        Initialize the model pool.
        
        Args:
            config: Pool configuration
        """
        self.config = config or ModelPoolConfig()
        self._models: OrderedDict[str, ModelEntry] = OrderedDict()
        self._shared_tokenizers: dict[str, Any] = {}
        self._shared_embeddings: dict[str, Any] = {}
        self._lock = threading.RLock()
        
        # Preload models if configured
        if self.config.preload_models:
            for model_id in self.config.preload_models:
                try:
                    self.preload(model_id)
                except Exception as e:
                    logger.warning(f"Failed to preload {model_id}: {e}")
    
    # -------------------------------------------------------------------------
    # MODEL LOADING
    # -------------------------------------------------------------------------
    
    def get_model(
        self,
        model_id: str,
        load_args: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Get a model from the pool (loads if not already loaded).
        
        Args:
            model_id: Model identifier (e.g., "forge:small", "huggingface:model")
            load_args: Optional loading arguments
            
        Returns:
            The loaded model instance
        """
        with self._lock:
            # Check if already loaded
            if model_id in self._models:
                entry = self._models[model_id]
                entry.mark_used()
                # Move to end (most recently used)
                self._models.move_to_end(model_id)
                logger.debug(f"Retrieved cached model: {model_id}")
                return entry.model
            
            # Need to load the model
            logger.info(f"Loading model: {model_id}")
            
            # Check if we need to evict models first
            if self.config.enable_auto_eviction:
                self._auto_evict_if_needed()
            
            # Load the model
            model, model_type, device, memory_mb, metadata = self._load_model(
                model_id, load_args
            )
            
            # Create entry and add to pool
            entry = ModelEntry(
                model_id=model_id,
                model=model,
                model_type=model_type,
                device=device,
                memory_mb=memory_mb,
                metadata=metadata,
            )
            entry.mark_used()
            self._models[model_id] = entry
            
            logger.info(
                f"Loaded {model_id} ({model_type}) on {device} "
                f"using ~{memory_mb:.1f}MB"
            )
            
            return model
    
    def preload(
        self,
        model_id: str,
        load_args: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Preload a model before it's needed.
        
        Args:
            model_id: Model to preload
            load_args: Optional loading arguments
        """
        if model_id not in self._models:
            self.get_model(model_id, load_args)
    
    def release_model(self, model_id: str) -> None:
        """
        Release a model back to the pool (doesn't unload).
        
        Args:
            model_id: Model to release
        """
        with self._lock:
            if model_id in self._models:
                self._models[model_id].mark_released()
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_id: Model to unload
            
        Returns:
            True if model was unloaded, False if not found
        """
        with self._lock:
            if model_id not in self._models:
                return False
            
            entry = self._models[model_id]
            if entry.in_use:
                logger.warning(f"Cannot unload {model_id} - still in use")
                return False
            
            # Remove from pool
            del self._models[model_id]
            
            # Clean up the model
            del entry.model
            gc.collect()
            
            # Clear CUDA cache if using GPU
            try:
                import torch
                if torch.cuda.is_available() and entry.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except ImportError:
                pass  # Intentionally silent
            
            logger.info(f"Unloaded model: {model_id}")
            return True
    
    # -------------------------------------------------------------------------
    # RESOURCE MANAGEMENT
    # -------------------------------------------------------------------------
    
    def get_memory_usage(self) -> dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage in MB
        """
        with self._lock:
            total_mb = sum(entry.memory_mb for entry in self._models.values())
            
            gpu_mb = sum(
                entry.memory_mb
                for entry in self._models.values()
                if entry.device.startswith("cuda")
            )
            
            cpu_mb = sum(
                entry.memory_mb
                for entry in self._models.values()
                if entry.device == "cpu"
            )
            
            return {
                "total_mb": total_mb,
                "gpu_mb": gpu_mb,
                "cpu_mb": cpu_mb,
                "num_models": len(self._models),
            }
    
    def get_system_memory_info(self) -> dict[str, float]:
        """
        Get system-wide memory information.
        
        Returns:
            Dictionary with system memory stats in MB
        """
        vm = psutil.virtual_memory()
        info = {
            "system_total_mb": vm.total / (1024 * 1024),
            "system_available_mb": vm.available / (1024 * 1024),
            "system_used_mb": vm.used / (1024 * 1024),
            "system_percent": vm.percent,
        }
        
        # Add GPU memory if available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_total = torch.cuda.get_device_properties(0).total_memory
                gpu_allocated = torch.cuda.memory_allocated(0)
                gpu_reserved = torch.cuda.memory_reserved(0)
                
                info.update({
                    "gpu_total_mb": gpu_total / (1024 * 1024),
                    "gpu_allocated_mb": gpu_allocated / (1024 * 1024),
                    "gpu_reserved_mb": gpu_reserved / (1024 * 1024),
                    "gpu_free_mb": (gpu_total - gpu_reserved) / (1024 * 1024),
                })
        except ImportError:
            pass  # Intentionally silent
        
        return info
    
    def evict_lru(
        self,
        count: Optional[int] = None,
        target_memory_mb: Optional[float] = None,
    ) -> int:
        """
        Evict least-recently-used models.
        
        Args:
            count: Number of models to evict (or None)
            target_memory_mb: Target memory usage (evict until below this)
            
        Returns:
            Number of models evicted
        """
        with self._lock:
            evicted = 0
            
            # Build list of candidates (not in use)
            candidates = [
                (model_id, entry)
                for model_id, entry in self._models.items()
                if not entry.in_use
            ]
            
            # Sort by last used time (oldest first)
            candidates.sort(key=lambda x: x[1].last_used_at)
            
            # Evict by count
            if count is not None:
                for model_id, _ in candidates[:count]:
                    if self.unload_model(model_id):
                        evicted += 1
            
            # Evict by memory target
            elif target_memory_mb is not None:
                current_usage = self.get_memory_usage()["total_mb"]
                
                for model_id, entry in candidates:
                    if current_usage <= target_memory_mb:
                        break
                    
                    if self.unload_model(model_id):
                        current_usage -= entry.memory_mb
                        evicted += 1
            
            if evicted > 0:
                logger.info(f"Evicted {evicted} LRU models from pool")
            
            return evicted
    
    def _auto_evict_if_needed(self) -> None:
        """Automatically evict models if limits are exceeded."""
        # Check model count limit
        if len(self._models) >= self.config.max_loaded_models:
            logger.info("Model count limit reached, evicting LRU")
            self.evict_lru(count=1)
        
        # Check memory limits
        usage = self.get_memory_usage()
        
        if self.config.gpu_memory_limit_mb > 0:
            if usage["gpu_mb"] > self.config.gpu_memory_limit_mb:
                logger.info("GPU memory limit exceeded, evicting LRU")
                target = self.config.gpu_memory_limit_mb * 0.8  # 80% of limit
                self.evict_lru(target_memory_mb=target)
        
        if self.config.cpu_memory_limit_mb > 0:
            if usage["cpu_mb"] > self.config.cpu_memory_limit_mb:
                logger.info("CPU memory limit exceeded, evicting LRU")
                target = self.config.cpu_memory_limit_mb * 0.8
                self.evict_lru(target_memory_mb=target)
    
    # -------------------------------------------------------------------------
    # MODEL LOADING IMPLEMENTATION
    # -------------------------------------------------------------------------
    
    def _load_model(
        self,
        model_id: str,
        load_args: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, str, str, float, dict[str, Any]]:
        """
        Load a model based on its ID format.
        
        Args:
            model_id: Model identifier
            load_args: Optional loading arguments
            
        Returns:
            Tuple of (model, model_type, device, memory_mb, metadata)
        """
        load_args = load_args or {}
        
        # Parse model ID format
        if ":" in model_id:
            model_type, model_path = model_id.split(":", 1)
        else:
            # Default to forge model
            model_type = "forge"
            model_path = model_id
        
        # Load based on type
        if model_type == "forge":
            return self._load_forge_model(model_path, load_args)
        elif model_type == "huggingface":
            return self._load_huggingface_model(model_path, load_args)
        elif model_type == "gguf":
            return self._load_gguf_model(model_path, load_args)
        elif model_type == "local":
            return self._load_local_model(model_path, load_args)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _load_forge_model(
        self,
        model_path: str,
        load_args: dict[str, Any],
    ) -> tuple[Any, str, str, float, dict[str, Any]]:
        """Load a Forge model."""
        from ..config import CONFIG
        from .inference import load_engine

        # Build full path
        if not Path(model_path).is_absolute():
            model_path = str(Path(CONFIG["models_dir"]) / model_path)
        
        # Determine device
        device = load_args.get("device", CONFIG.get("device", "auto"))
        if device == "auto":
            device = self._get_best_device()
        
        # Load engine
        engine = load_engine(model_path, device=device)
        
        # Estimate memory usage
        memory_mb = self._estimate_model_memory(engine.model)
        
        metadata = {
            "path": model_path,
            "size": getattr(engine.model.config, "params", "unknown"),
        }
        
        return engine, "forge", device, memory_mb, metadata
    
    def _load_huggingface_model(
        self,
        model_id: str,
        load_args: dict[str, Any],
    ) -> tuple[Any, str, str, float, dict[str, Any]]:
        """Load a HuggingFace model."""
        try:
            from .huggingface_loader import load_huggingface_model
        except ImportError:
            raise ImportError(
                "HuggingFace support not available. "
                "Install with: pip install transformers"
            )
        
        # Determine device
        device = load_args.get("device")
        if not device:
            device = self._get_best_device()
        
        # Load model
        model = load_huggingface_model(model_id, device=device, **load_args)
        
        # Estimate memory
        memory_mb = self._estimate_hf_memory(model_id)
        
        metadata = {
            "model_id": model_id,
            "source": "huggingface",
        }
        
        return model, "huggingface", device, memory_mb, metadata
    
    def _load_gguf_model(
        self,
        model_path: str,
        load_args: dict[str, Any],
    ) -> tuple[Any, str, str, float, dict[str, Any]]:
        """Load a GGUF model."""
        try:
            from .gguf_loader import GGUFModel
        except ImportError:
            raise ImportError(
                "GGUF support not available. "
                "Install with: pip install gguf llama-cpp-python"
            )
        
        model = GGUFModel(model_path, **load_args)
        
        # GGUF models typically run on CPU
        device = "cpu"
        memory_mb = Path(model_path).stat().st_size / (1024 * 1024) * 0.7  # Rough estimate
        
        metadata = {
            "path": model_path,
            "format": "gguf",
        }
        
        return model, "gguf", device, memory_mb, metadata
    
    def _load_local_model(
        self,
        identifier: str,
        load_args: dict[str, Any],
    ) -> tuple[Any, str, str, float, dict[str, Any]]:
        """Load a local tool/model."""
        # This is a placeholder for local tools like Stable Diffusion
        # The actual implementation would depend on the tool
        
        metadata = {
            "identifier": identifier,
            "type": "local_tool",
        }
        
        return identifier, "local", "cpu", 100.0, metadata
    
    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------
    
    def _get_best_device(self) -> str:
        """Determine the best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                # Check if GPU has enough memory
                sys_info = self.get_system_memory_info()
                if sys_info.get("gpu_free_mb", 0) > 1000:  # At least 1GB free
                    return "cuda"
                elif self.config.fallback_to_cpu:
                    logger.info("GPU memory low, falling back to CPU")
                    return "cpu"
                else:
                    return "cuda"  # Try anyway
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass  # Intentionally silent
        
        return "cpu"
    
    def _estimate_model_memory(self, model: Any) -> float:
        """Estimate memory usage of a PyTorch model in MB."""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            # Assume float32 (4 bytes per param)
            bytes_per_param = 4
            total_bytes = total_params * bytes_per_param
            # Add overhead (activations, optimizer, etc.) - roughly 2x
            total_bytes *= 2
            return total_bytes / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Could not estimate model memory: {e}")
            return 500.0  # Default estimate
    
    def _estimate_hf_memory(self, model_id: str) -> float:
        """Estimate memory for a HuggingFace model based on size in name."""
        model_lower = model_id.lower()
        
        # Try to extract size from model name
        if "0.5b" in model_lower or "500m" in model_lower:
            return 1000  # 1GB
        elif "1b" in model_lower or "1.1b" in model_lower:
            return 2000  # 2GB
        elif "1.5b" in model_lower:
            return 3000  # 3GB
        elif "3b" in model_lower:
            return 6000  # 6GB
        elif "7b" in model_lower:
            return 14000  # 14GB
        elif "13b" in model_lower:
            return 26000  # 26GB
        else:
            return 4000  # Default 4GB
    
    # -------------------------------------------------------------------------
    # POOL INFORMATION
    # -------------------------------------------------------------------------
    
    def list_models(self) -> list[dict[str, Any]]:
        """Get list of all loaded models with info."""
        with self._lock:
            return [
                {
                    "model_id": model_id,
                    "model_type": entry.model_type,
                    "device": entry.device,
                    "memory_mb": entry.memory_mb,
                    "in_use": entry.in_use,
                    "use_count": entry.use_count,
                    "loaded_at": entry.loaded_at,
                    "last_used_at": entry.last_used_at,
                }
                for model_id, entry in self._models.items()
            ]
    
    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded."""
        return model_id in self._models
    
    def clear(self) -> None:
        """Unload all models from the pool."""
        with self._lock:
            model_ids = list(self._models.keys())
            for model_id in model_ids:
                self.unload_model(model_id)


# =============================================================================
# GLOBAL POOL INSTANCE
# =============================================================================

_global_pool: Optional[ModelPool] = None


def get_model_pool(config: Optional[ModelPoolConfig] = None) -> ModelPool:
    """
    Get the global model pool instance.
    
    Args:
        config: Optional pool configuration
        
    Returns:
        Global ModelPool instance
    """
    global _global_pool
    if _global_pool is None:
        # Load config from CONFIG if not provided
        if config is None:
            from ..config import CONFIG
            config = ModelPoolConfig(
                max_loaded_models=CONFIG.get("max_concurrent_models", 3),
                gpu_memory_limit_mb=int(CONFIG.get("gpu_memory_fraction", 0.85) * 
                                       CONFIG.get("memory_limit_mb", 8000)),
                enable_auto_eviction=True,
                fallback_to_cpu=CONFIG.get("fallback_to_cpu", True),
            )
        
        _global_pool = ModelPool(config)
    
    return _global_pool
