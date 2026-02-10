# type: ignore[all]
"""
Device Offloading - Split model between CPU and GPU.

This allows running models that are too large for GPU VRAM by:
1. Keeping some layers on GPU (fast)
2. Keeping other layers on CPU RAM (slower but more memory)
3. Optionally offloading to disk for very large models

How AI Uses RAM:
- When using CPU, the model weights are stored in system RAM
- RAM is typically 16-64GB+ vs GPU VRAM which is 4-24GB
- CPU is slower but can handle larger models

Usage:
    from enigma_engine.core.offloading import OffloadedModel, get_device_map
    
    # Automatic device mapping
    device_map = get_device_map(model, max_gpu_memory="6GB")
    
    # Or wrap existing model
    model = OffloadedModel(model, device_map="auto")
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch

logger = logging.getLogger(__name__)

# Check for accelerate library
try:
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_balanced_memory
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False
    logger.info("accelerate not installed. Install with: pip install accelerate")


def get_memory_info() -> dict[str, Any]:
    """Get current memory info for CPU and GPU."""
    import psutil
    
    info = {
        "cpu_total_gb": psutil.virtual_memory().total / (1024**3),
        "cpu_available_gb": psutil.virtual_memory().available / (1024**3),
        "cpu_used_gb": psutil.virtual_memory().used / (1024**3),
        "gpu_available": torch.cuda.is_available(),
        "gpus": []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / (1024**3)
            
            # Get allocated memory
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            free = total - reserved
            
            info["gpus"].append({
                "index": i,
                "name": props.name,
                "total_gb": total,
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "free_gb": free
            })
    
    return info


def estimate_model_memory(model: torch.nn.Module) -> dict[str, float]:
    """Estimate memory requirements for a model."""
    param_count = sum(p.numel() for p in model.parameters())
    
    # Calculate size in different precisions
    fp32_gb = (param_count * 4) / (1024**3)  # 4 bytes per float32
    fp16_gb = (param_count * 2) / (1024**3)  # 2 bytes per float16
    int8_gb = (param_count * 1) / (1024**3)  # 1 byte per int8
    int4_gb = (param_count * 0.5) / (1024**3)  # 0.5 bytes per int4
    
    return {
        "parameters": param_count,
        "fp32_gb": fp32_gb,
        "fp16_gb": fp16_gb,
        "int8_gb": int8_gb,
        "int4_gb": int4_gb,
        # Include overhead for optimizer states, activations, etc.
        "training_estimate_gb": fp32_gb * 4,  # Rough estimate
        "inference_estimate_gb": fp16_gb * 1.5
    }


def get_device_map(
    model: torch.nn.Module,
    max_gpu_memory: Optional[Union[str, dict[int, str]]] = None,
    max_cpu_memory: Optional[str] = None,
    offload_folder: Optional[str] = None
) -> dict[str, Union[int, str]]:
    """
    Create a device map for splitting model across devices.
    
    Args:
        model: The PyTorch model
        max_gpu_memory: Max GPU memory per device, e.g., "6GB" or {0: "6GB", 1: "8GB"}
        max_cpu_memory: Max CPU memory to use, e.g., "16GB"
        offload_folder: Folder for disk offloading (if needed)
        
    Returns:
        Dictionary mapping layer names to devices
    """
    if not HAS_ACCELERATE:
        logger.warning("accelerate not installed - cannot create device map")
        return {"": "cuda" if torch.cuda.is_available() else "cpu"}
    
    # Parse memory specs
    if isinstance(max_gpu_memory, str):
        # Convert "6GB" to {0: "6GB"}
        max_gpu_memory = {0: max_gpu_memory}
    
    try:
        # Get balanced memory allocation
        max_memory = get_balanced_memory(
            model,
            max_memory=max_gpu_memory,
            no_split_module_classes=["ForgeLayer", "TransformerBlock"]  # Don't split these
        )
        
        if max_cpu_memory:
            max_memory["cpu"] = max_cpu_memory
        
        # Infer device map
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["ForgeLayer", "TransformerBlock"]
        )
        
        return device_map
        
    except Exception as e:
        logger.warning(f"Could not create device map: {e}")
        return {"": "cuda" if torch.cuda.is_available() else "cpu"}


def apply_offloading(
    model: torch.nn.Module,
    device_map: Optional[dict[str, Union[int, str]]] = None,
    offload_folder: Optional[str] = None,
    offload_to_disk: bool = False,
    max_gpu_memory: Optional[str] = None
) -> torch.nn.Module:
    """
    Apply device offloading to a model.
    
    Args:
        model: The PyTorch model
        device_map: Explicit device map, or None for auto
        offload_folder: Folder for disk offloading
        offload_to_disk: Whether to also offload to disk
        max_gpu_memory: Max GPU memory to use (e.g., "6GB")
        
    Returns:
        Model with offloading applied
    """
    if not HAS_ACCELERATE:
        logger.warning("accelerate not installed - model will use single device")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return model.to(device)
    
    try:
        # Create device map if not provided
        if device_map is None or device_map == "auto":
            device_map = get_device_map(model, max_gpu_memory=max_gpu_memory)
        
        logger.info(f"Applying device map: {device_map}")
        
        # Apply the dispatch
        kwargs = {}
        if offload_folder:
            kwargs["offload_folder"] = offload_folder
        if offload_to_disk:
            kwargs["offload_state_dict"] = True
        
        model = dispatch_model(model, device_map=device_map, **kwargs)
        
        return model
        
    except Exception as e:
        logger.error(f"Could not apply offloading: {e}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return model.to(device)


class OffloadingConfig:
    """Configuration for model offloading."""
    
    def __init__(
        self,
        enabled: bool = False,
        max_gpu_memory: Optional[str] = None,
        max_cpu_memory: Optional[str] = None,
        offload_folder: Optional[str] = None,
        offload_to_disk: bool = False
    ):
        self.enabled = enabled
        self.max_gpu_memory = max_gpu_memory
        self.max_cpu_memory = max_cpu_memory
        self.offload_folder = offload_folder
        self.offload_to_disk = offload_to_disk
    
    @classmethod
    def from_config(cls) -> "OffloadingConfig":
        """Create from global CONFIG."""
        try:
            from ..config import CONFIG
            return cls(
                enabled=CONFIG.get("enable_offloading", False),
                max_gpu_memory=None,  # Let it auto-detect
                max_cpu_memory=None,
                offload_folder=CONFIG.get("offload_folder"),
                offload_to_disk=CONFIG.get("offload_to_disk", False)
            )
        except Exception:
            return cls()


def load_model_with_offloading(
    model_class,
    model_path: str,
    config: Optional[OffloadingConfig] = None,
    **model_kwargs
) -> torch.nn.Module:
    """
    Load a model with automatic offloading based on available memory.
    
    Args:
        model_class: The model class to instantiate
        model_path: Path to model weights
        config: Offloading configuration
        **model_kwargs: Additional arguments for model constructor
        
    Returns:
        Loaded model with offloading applied
    """
    config = config or OffloadingConfig.from_config()
    
    # Get memory info
    mem_info = get_memory_info()
    logger.info(f"Memory available - CPU: {mem_info['cpu_available_gb']:.1f}GB")
    
    if mem_info["gpus"]:
        for gpu in mem_info["gpus"]:
            logger.info(f"  GPU {gpu['index']} ({gpu['name']}): {gpu['free_gb']:.1f}GB free")
    
    # Create model
    model = model_class(**model_kwargs)
    
    # Load weights (weights_only=True for security against pickle attacks)
    if model_path and Path(model_path).exists():
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    
    # Estimate memory needs
    mem_estimate = estimate_model_memory(model)
    logger.info(f"Model needs ~{mem_estimate['inference_estimate_gb']:.1f}GB for inference")
    
    # Apply offloading if enabled
    if config.enabled:
        model = apply_offloading(
            model,
            device_map="auto",
            offload_folder=config.offload_folder,
            offload_to_disk=config.offload_to_disk,
            max_gpu_memory=config.max_gpu_memory
        )
    else:
        # Standard loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    
    return model


# Simple manual offloading (without accelerate)
class ManualOffload:
    """
    Manual layer-by-layer offloading for when accelerate isn't available.
    
    Moves layers to CPU when not in use, GPU when computing.
    Slower than accelerate but works without dependencies.
    """
    
    def __init__(self, model: torch.nn.Module, gpu_layers: int = 0):
        """
        Args:
            model: The model to offload
            gpu_layers: Number of layers to keep on GPU (0 = all on CPU)
        """
        self.model = model
        self.gpu_layers = gpu_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Move base components to GPU, layers to CPU
        self._setup_offloading()
    
    def _setup_offloading(self):
        """Set up initial device placement."""
        if not hasattr(self.model, 'layers'):
            # No layers attribute, just use regular device
            self.model.to(self.device)
            return
        
        # Move embedding and final layers to GPU
        if hasattr(self.model, 'embedding'):
            self.model.embedding.to(self.device)
        if hasattr(self.model, 'output_head'):
            self.model.output_head.to(self.device)
        if hasattr(self.model, 'norm'):
            self.model.norm.to(self.device)
        
        # Move layers based on gpu_layers setting
        for i, layer in enumerate(self.model.layers):
            if i < self.gpu_layers:
                layer.to(self.device)
            else:
                layer.to("cpu")
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with dynamic offloading."""
        if not hasattr(self.model, 'layers'):
            return self.model(x, **kwargs)
        
        # Embedding
        x = x.to(self.device)
        if hasattr(self.model, 'embedding'):
            x = self.model.embedding(x)
        
        # Process layers
        for i, layer in enumerate(self.model.layers):
            if i >= self.gpu_layers:
                # Move layer to GPU, process, move back
                layer.to(self.device)
                x = layer(x)
                layer.to("cpu")
                torch.cuda.empty_cache()
            else:
                x = layer(x)
        
        # Final norm and output
        if hasattr(self.model, 'norm'):
            x = self.model.norm(x)
        if hasattr(self.model, 'output_head'):
            x = self.model.output_head(x)
        
        return x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
