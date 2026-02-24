"""
Hardware Detection Module

Detects hardware capabilities and recommends optimal model configurations.
"""
import logging
import platform
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """Hardware profile with system capabilities."""
    device: str = "cpu"
    cpu_cores: int = 1
    cpu_threads: int = 1
    ram_gb: float = 4.0
    total_ram_gb: float = 4.0  # Alias for ram_gb
    available_ram_gb: float = 4.0  # Currently available RAM
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    is_apple_silicon: bool = False
    is_raspberry_pi: bool = False
    pi_model: str = ""
    cuda_version: str = ""
    hardware_type: str = "unknown"  # "raspberry_pi", "desktop", "server", etc.
    # Convenience attributes
    has_cuda: bool = False
    has_mps: bool = False
    is_arm: bool = False
    
    def __str__(self) -> str:
        if self.gpu_available:
            return f"{self.gpu_name} ({self.gpu_vram_gb:.1f}GB VRAM)"
        return f"{self.cpu_cores} CPU cores, {self.ram_gb:.1f}GB RAM"


_cached_profile: Optional[HardwareProfile] = None


def detect_hardware() -> HardwareProfile:
    """Detect hardware capabilities of the current system."""
    global _cached_profile
    if _cached_profile is not None:
        return _cached_profile
    
    profile = HardwareProfile()
    
    # CPU info
    import os
    profile.cpu_cores = os.cpu_count() or 1
    profile.cpu_threads = profile.cpu_cores
    
    # Check for ARM architecture
    machine = platform.machine().lower()
    profile.is_arm = machine in ('arm64', 'aarch64', 'armv7l', 'armv8')
    
    # RAM info
    try:
        import psutil
        mem = psutil.virtual_memory()
        profile.ram_gb = mem.total / (1024**3)
        profile.available_ram_gb = mem.available / (1024**3)
    except ImportError:
        profile.ram_gb = 8.0  # Default assumption
        profile.available_ram_gb = profile.ram_gb * 0.5
    profile.total_ram_gb = profile.ram_gb
    
    # Check for Raspberry Pi
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            if 'Raspberry Pi' in model:
                profile.is_raspberry_pi = True
                profile.pi_model = model.strip()
                profile.hardware_type = "raspberry_pi"
    except Exception:
        pass
    
    # Check for Apple Silicon
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        profile.is_apple_silicon = True
        profile.hardware_type = "apple_silicon"
    
    # GPU detection
    try:
        import torch
        if torch.cuda.is_available():
            profile.gpu_available = True
            profile.has_cuda = True
            profile.device = "cuda"
            profile.gpu_name = torch.cuda.get_device_name(0)
            profile.gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            profile.cuda_version = torch.version.cuda or ""
            profile.hardware_type = "desktop_gpu"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            profile.gpu_available = True
            profile.has_mps = True
            profile.device = "mps"
            profile.gpu_name = "Apple Silicon GPU"
            # MPS doesn't report VRAM, estimate from unified memory
            profile.gpu_vram_gb = profile.ram_gb * 0.75
    except ImportError:
        pass
    
    # Set default hardware type if not set
    if profile.hardware_type == "unknown":
        if profile.gpu_available:
            profile.hardware_type = "desktop_gpu"
        elif profile.ram_gb >= 32:
            profile.hardware_type = "server"
        else:
            profile.hardware_type = "desktop_cpu"
    
    _cached_profile = profile
    return profile


def get_cached_profile() -> Optional[HardwareProfile]:
    """Get cached hardware profile if available."""
    return _cached_profile


def clear_cached_profile() -> None:
    """Clear cached hardware profile."""
    global _cached_profile
    _cached_profile = None


def recommend_model_size(profile: Optional[HardwareProfile] = None) -> str:
    """
    Recommend optimal model size based on hardware.
    
    Returns:
        Model size string: 'pi_zero', 'nano', 'tiny', 'small', 'medium', 'large'
    """
    if profile is None:
        profile = detect_hardware()
    
    # Raspberry Pi or very low memory
    if profile.is_raspberry_pi or profile.ram_gb < 2:
        return "pi_zero"
    
    # Low memory systems
    if profile.ram_gb < 4:
        return "nano"
    
    # No GPU - use smaller models
    if not profile.gpu_available:
        if profile.ram_gb < 8:
            return "tiny"
        elif profile.ram_gb < 16:
            return "small"
        else:
            return "medium"
    
    # GPU available - use larger models
    if profile.gpu_vram_gb >= 24:
        return "large"
    elif profile.gpu_vram_gb >= 12:
        return "medium"
    elif profile.gpu_vram_gb >= 6:
        return "small"
    elif profile.gpu_vram_gb >= 4:
        return "tiny"
    else:
        return "nano"


def get_optimal_config(profile: Optional[HardwareProfile] = None) -> dict[str, Any]:
    """
    Get optimal configuration based on hardware.
    
    Returns:
        Dict with recommended settings: model_size, use_half, batch_size, etc.
    """
    if profile is None:
        profile = detect_hardware()
    
    model_size = recommend_model_size(profile)
    
    config = {
        "model_size": model_size,
        "device": profile.device,
        "use_half": profile.gpu_available and profile.device == "cuda",
        "batch_size": 1,
        "max_seq_len": 512,
    }
    
    # Adjust based on VRAM/RAM
    if profile.gpu_available and profile.gpu_vram_gb >= 8:
        config["batch_size"] = 4
        config["max_seq_len"] = 1024
    elif profile.gpu_available and profile.gpu_vram_gb >= 4:
        config["batch_size"] = 2
        config["max_seq_len"] = 512
    elif profile.ram_gb >= 16:
        config["batch_size"] = 2
        config["max_seq_len"] = 512
    
    return config


def estimate_memory_usage(
    model_size: str,
    batch_size: int = 1,
    seq_len: int = 512,
    use_half: bool = False
) -> dict[str, float]:
    """
    Estimate memory usage for a given configuration.
    
    Returns:
        Dict with estimated memory in GB: model_memory, kv_cache, total
    """
    # Approximate parameter counts
    param_counts = {
        "pi_zero": 0.5e6,
        "nano": 1e6,
        "tiny": 5e6,
        "small": 27e6,
        "medium": 85e6,
        "large": 200e6,
        "xl": 600e6,
        "xxl": 1.5e9,
    }
    
    params = param_counts.get(model_size, 27e6)
    bytes_per_param = 2 if use_half else 4
    
    # Model weight memory
    model_gb = (params * bytes_per_param) / (1024**3)
    
    # KV-cache estimate (rough)
    # For transformer: 2 * layers * 2 * hidden_dim * seq_len * batch_size
    hidden_dims = {
        "pi_zero": 128,
        "nano": 256,
        "tiny": 384,
        "small": 512,
        "medium": 768,
        "large": 1024,
        "xl": 1536,
        "xxl": 2048,
    }
    layers = {"pi_zero": 4, "nano": 6, "tiny": 8, "small": 12, "medium": 16, "large": 24, "xl": 32, "xxl": 48}
    
    dim = hidden_dims.get(model_size, 512)
    n_layers = layers.get(model_size, 12)
    
    kv_bytes = 2 * n_layers * 2 * dim * seq_len * batch_size * bytes_per_param
    kv_gb = kv_bytes / (1024**3)
    
    return {
        "model_memory": model_gb,
        "kv_cache": kv_gb,
        "total": model_gb + kv_gb,
    }


# Backward compatibility - get_hardware alias
def get_hardware() -> HardwareProfile:
    """Alias for detect_hardware()."""
    return detect_hardware()


__all__ = [
    'HardwareProfile',
    'detect_hardware',
    'get_hardware',
    'recommend_model_size',
    'get_optimal_config',
    'estimate_memory_usage',
    'get_cached_profile',
    'clear_cached_profile',
]
