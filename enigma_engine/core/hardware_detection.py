"""
Hardware Detection Module

Detects hardware capabilities and recommends optimal model configurations.
"""
import logging
import platform
import threading
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

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)


_cached_profile: Optional[HardwareProfile] = None
_profile_lock = threading.Lock()


def detect_hardware() -> HardwareProfile:
    """Detect hardware capabilities of the current system."""
    global _cached_profile
    with _profile_lock:
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
        logger.warning("psutil not installed — defaulting to 8 GB RAM")
        profile.ram_gb = 8.0  # Default assumption
        profile.available_ram_gb = profile.ram_gb * 0.5
    profile.total_ram_gb = profile.ram_gb

    # Check for Raspberry Pi
    try:
        with open('/proc/device-tree/model', 'r', encoding='utf-8') as f:
            model = f.read()
            if 'Raspberry Pi' in model:
                profile.is_raspberry_pi = True
                profile.pi_model = model.strip().rstrip('\x00')
                profile.hardware_type = "raspberry_pi"
    except (FileNotFoundError, OSError):
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

    with _profile_lock:
        _cached_profile = profile
    return profile


def get_cached_profile() -> Optional[HardwareProfile]:
    """Get cached hardware profile if available."""
    with _profile_lock:
        return _cached_profile


def clear_cached_profile() -> None:
    """Clear cached hardware profile."""
    global _cached_profile
    with _profile_lock:
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
        "precision": "auto",
        "batch_size": 1,
        "max_seq_len": 512,
    }

    # BF16 detection for Blackwell / Ampere+ GPUs
    if profile.gpu_available and profile.device == "cuda":
        try:
            import torch
            if torch.cuda.is_bf16_supported():
                config["precision"] = "bfloat16"
            else:
                config["precision"] = "float16"
        except Exception:
            config["precision"] = "float16"
    elif not profile.gpu_available:
        config["precision"] = "float32"

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

    Reads dim/n_layers from MODEL_PRESETS so estimates stay in
    sync with actual model definitions.

    Returns:
        Dict with estimated memory in GB: model_memory, kv_cache, total
    """
    # Pull real config from presets when available
    dim: int = 512
    n_layers: int = 12
    vocab_size: int = 32000
    try:
        from enigma_engine.core.model_presets import MODEL_PRESETS
        cfg = MODEL_PRESETS.get(model_size)
        if cfg is not None:
            dim = cfg.dim
            n_layers = cfg.n_layers
            vocab_size = cfg.vocab_size
    except ImportError:
        pass

    # Estimate parameter count from architecture:
    # embeddings + output + per-layer (QKV + out + FFN gate/up/down)
    ffn_dim = int(dim * 8 / 3)  # SwiGLU default
    per_layer = (
        3 * dim * dim      # Q, K, V projections
        + dim * dim         # output projection
        + 3 * dim * ffn_dim  # gate + up + down
    )
    params = 2 * vocab_size * dim + n_layers * per_layer

    bytes_per_param = 2 if use_half else 4

    # Model weight memory
    model_gb = (params * bytes_per_param) / (1024**3)

    # KV-cache estimate (rough)
    # For transformer: 2 * layers * 2 * hidden_dim * seq_len * batch_size
    kv_bytes = 2 * n_layers * 2 * dim * seq_len * batch_size * bytes_per_param
    kv_gb = kv_bytes / (1024**3)

    return {
        "model_memory": model_gb,
        "kv_cache": kv_gb,
        "total": model_gb + kv_gb,
    }


def recommend_training_batch_size(
    profile: Optional[HardwareProfile] = None,
) -> int:
    """Recommend a safe training batch size based on detected VRAM.

    Uses simple VRAM tiers rather than formula estimation.
    Conservative defaults to avoid OOM during training (training
    uses ~3x model memory due to gradients + optimizer states).

    Args:
        profile: Hardware profile.  Detected automatically if None.

    Returns:
        Recommended batch size (1, 2, 4, 8, or 16).
    """
    if profile is None:
        profile = detect_hardware()

    if not profile.gpu_available:
        # CPU-only: always use batch size 1
        if profile.ram_gb >= 32:
            return 2
        return 1

    vram = profile.gpu_vram_gb
    if vram >= 48:
        return 16
    if vram >= 24:
        return 8
    if vram >= 12:
        return 4
    if vram >= 6:
        return 2
    return 1


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
    'recommend_training_batch_size',
]
