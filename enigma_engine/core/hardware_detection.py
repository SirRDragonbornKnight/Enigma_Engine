"""
================================================================================
🔧 HARDWARE DETECTION - SMART DEVICE AWARENESS FOR EDGE DEPLOYMENT
================================================================================

Detects hardware capabilities and recommends optimal configurations for
deployment on any device from Raspberry Pi Zero to datacenter GPUs.

📍 FILE: enigma_engine/core/hardware_detection.py
🏷️ TYPE: Hardware Detection & Configuration
🎯 MAIN CLASSES: HardwareProfile, detect_hardware, recommend_model_size

┌─────────────────────────────────────────────────────────────────────────────┐
│  DETECTION FLOW:                                                            │
│                                                                             │
│  detect_hardware()                                                          │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. Detect RAM (total & available)                                   │   │
│  │  2. Detect GPU (CUDA, MPS, VRAM)                                    │   │
│  │  3. Detect CPU (cores, architecture)                                │   │
│  │  4. Detect platform (Pi, ARM, x86)                                  │   │
│  │  5. Recommend model size & quantization                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  HardwareProfile                                                           │
│    ├── total_ram_gb: 8.0                                                   │
│    ├── is_raspberry_pi: True                                               │
│    ├── pi_model: "Pi 5"                                                    │
│    ├── recommended_model_size: "pi_5"                                      │
│    └── recommended_quantization: "int8"                                    │
└─────────────────────────────────────────────────────────────────────────────┘

🍓 RASPBERRY PI MODELS:
    ┌──────────────┬─────────┬────────────────────────────────────────────┐
    │ Model        │ RAM     │ Recommended Config                          │
    ├──────────────┼─────────┼────────────────────────────────────────────┤
    │ Pi Zero 2W   │ 512MB   │ pi_zero preset, int4 quant                  │
    │ Pi 4 (2GB)   │ 2GB     │ pi_4 preset, int8 quant                     │
    │ Pi 4 (4GB)   │ 4GB     │ pi_4 preset, int8 quant                     │
    │ Pi 4 (8GB)   │ 8GB     │ pi_5 preset, dynamic quant                  │
    │ Pi 5 (4GB)   │ 4GB     │ pi_5 preset, int8 quant                     │
    │ Pi 5 (8GB)   │ 8GB     │ pi_5 preset, dynamic quant                  │
    └──────────────┴─────────┴────────────────────────────────────────────┘

🔗 CONNECTED FILES:
    → USES:      /proc/cpuinfo, /proc/meminfo (Linux)
    → USES:      torch.cuda, torch.backends.mps (GPU detection)
    ← USED BY:   enigma_engine/core/model.py (auto_configure)
    ← USED BY:   enigma_engine/core/inference.py (EnigmaEngine)

📖 USAGE:
    from enigma_engine.core.hardware_detection import (
        detect_hardware, recommend_model_size, get_optimal_config
    )
    
    # Detect hardware
    profile = detect_hardware()
    print(f"RAM: {profile.total_ram_gb}GB")
    print(f"Is Pi: {profile.is_raspberry_pi}")
    print(f"Recommended: {profile.recommended_model_size}")
    
    # Get optimal config for ForgeConfig
    config = get_optimal_config(profile)
    model = create_model(**config)

📖 SEE ALSO:
    • enigma_engine/core/model.py          - Model creation with presets
    • enigma_engine/core/hardware.py       - Legacy hardware detection
    • docs/multi_device_guide.md      - Multi-device setup
"""

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# 🔧 HARDWARE PROFILE DATACLASS
# =============================================================================

@dataclass
class HardwareProfile:
    """
    Complete hardware profile for a device.
    
    📖 WHAT THIS CONTAINS:
    All detected hardware capabilities needed to make intelligent decisions
    about model size, quantization, and inference settings.
    
    📐 RASPBERRY PI DETECTION:
    We check /proc/cpuinfo for BCM chips and "Raspberry Pi" strings.
    The pi_model field contains the specific model if detected.
    
    ⚡ RECOMMENDATIONS:
    The recommended_* fields are computed based on all hardware factors:
    - RAM limits what model size fits
    - GPU presence affects quantization choices
    - ARM architecture may prefer certain optimizations
    """
    # Memory
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    
    # GPU
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    
    # CPU
    cpu_cores: int = 1
    cpu_model: str = "Unknown"
    
    # Platform flags
    is_arm: bool = False
    is_raspberry_pi: bool = False
    pi_model: Optional[str] = None  # "Pi 4", "Pi 5", "Pi Zero 2W", etc.
    
    # GPU acceleration
    has_cuda: bool = False
    has_mps: bool = False  # Apple Silicon
    
    # Recommendations (computed)
    recommended_model_size: str = "tiny"
    recommended_quantization: str = "none"
    
    # Additional metadata
    system: str = ""
    architecture: str = ""
    python_version: str = ""
    
    def summary(self) -> str:
        """Get human-readable summary of hardware profile."""
        lines = [
            "=" * 60,
            "HARDWARE PROFILE",
            "=" * 60,
            f"System: {self.system} ({self.architecture})",
            f"CPU: {self.cpu_model} ({self.cpu_cores} cores)",
            f"RAM: {self.total_ram_gb:.1f} GB total, {self.available_ram_gb:.1f} GB available",
        ]
        
        if self.gpu_name:
            lines.append(f"GPU: {self.gpu_name} ({self.gpu_vram_gb:.1f} GB VRAM)")
        else:
            lines.append("GPU: None (CPU only)")
        
        if self.is_raspberry_pi:
            lines.append(f"Platform: Raspberry {self.pi_model}")
        elif self.is_arm:
            lines.append("Platform: ARM device")
        
        lines.extend([
            "-" * 60,
            f"Recommended Model: {self.recommended_model_size}",
            f"Recommended Quantization: {self.recommended_quantization}",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_ram_gb": self.total_ram_gb,
            "available_ram_gb": self.available_ram_gb,
            "gpu_name": self.gpu_name,
            "gpu_vram_gb": self.gpu_vram_gb,
            "cpu_cores": self.cpu_cores,
            "cpu_model": self.cpu_model,
            "is_arm": self.is_arm,
            "is_raspberry_pi": self.is_raspberry_pi,
            "pi_model": self.pi_model,
            "has_cuda": self.has_cuda,
            "has_mps": self.has_mps,
            "recommended_model_size": self.recommended_model_size,
            "recommended_quantization": self.recommended_quantization,
            "system": self.system,
            "architecture": self.architecture,
        }


# =============================================================================
# 🔍 DETECTION FUNCTIONS
# =============================================================================

def _detect_memory() -> tuple[float, float]:
    """
    Detect RAM total and available.
    
    Returns:
        (total_gb, available_gb) tuple
    """
    total_ram = 0.0
    available_ram = 0.0
    
    try:
        system = platform.system()
        
        if system == "Linux":
            # Parse /proc/meminfo (most accurate on Linux)
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        # Value is in KB
                        total_ram = int(line.split()[1]) / (1024 * 1024)
                    elif "MemAvailable" in line:
                        available_ram = int(line.split()[1]) / (1024 * 1024)
                        
        elif system == "Darwin":
            # macOS - use sysctl
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
            )
            total_ram = int(result.stdout.strip()) / (1024**3)
            # macOS doesn't easily report available, estimate as 60%
            available_ram = total_ram * 0.6
            
        elif system == "Windows":
            # Windows - use ctypes for accurate memory info
            import ctypes
            kernel32 = ctypes.windll.kernel32
            
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]
            
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_ram = stat.ullTotalPhys / (1024**3)
            available_ram = stat.ullAvailPhys / (1024**3)
            
    except Exception as e:
        logger.warning(f"Memory detection failed: {e}")
        # Safe defaults
        total_ram = 2.0
        available_ram = 1.0
    
    return total_ram, available_ram


def _detect_gpu() -> tuple[Optional[str], Optional[float], bool, bool]:
    """
    Detect GPU capabilities.
    
    Returns:
        (gpu_name, vram_gb, has_cuda, has_mps) tuple
    """
    gpu_name = None
    vram_gb = None
    has_cuda = False
    has_mps = False
    
    try:
        import torch

        # Check CUDA (NVIDIA)
        if torch.cuda.is_available():
            has_cuda = True
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Detected CUDA GPU: {gpu_name} with {vram_gb:.1f}GB VRAM")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            has_mps = True
            gpu_name = "Apple Silicon (MPS)"
            # MPS shares system RAM, estimate 50% available
            total_ram, _ = _detect_memory()
            vram_gb = total_ram * 0.5
            logger.info(f"Detected Apple Silicon MPS with ~{vram_gb:.1f}GB available")
            
    except ImportError:
        logger.info("PyTorch not available for GPU detection")
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}")
    
    # Fallback: Try nvidia-smi
    if gpu_name is None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                gpu_name = parts[0].strip()
                vram_str = parts[1].strip().replace("MiB", "").strip()
                vram_gb = int(vram_str) / 1024
                has_cuda = True
        except Exception:
            pass
    
    return gpu_name, vram_gb, has_cuda, has_mps


def _detect_cpu() -> tuple[int, str]:
    """
    Detect CPU cores and model.
    
    Returns:
        (cores, model_name) tuple
    """
    cores = os.cpu_count() or 1
    model = "Unknown"
    
    try:
        system = platform.system()
        
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line.lower():
                        model = line.split(":")[1].strip()
                        break
                        
        elif system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            model = result.stdout.strip()
            
        elif system == "Windows":
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            model = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            
    except Exception as e:
        logger.debug(f"CPU model detection failed: {e}")
    
    return cores, model


def _detect_raspberry_pi() -> tuple[bool, Optional[str]]:
    """
    Detect if running on Raspberry Pi and which model.
    
    Returns:
        (is_pi, model_name) tuple
        
    📐 PI MODEL DETECTION:
    We check multiple sources:
    1. /proc/cpuinfo for BCM chips (Broadcom)
    2. /proc/device-tree/model for exact model string
    3. /sys/firmware/devicetree/base/model
    """
    is_pi = False
    model = None
    
    if platform.system() != "Linux":
        return False, None
    
    # Check /proc/cpuinfo for BCM (Broadcom) chips
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read().lower()
            if "raspberry" in cpuinfo or "bcm" in cpuinfo:
                is_pi = True
    except Exception:
        pass
    
    # Try to get exact model from device tree
    model_paths = [
        "/proc/device-tree/model",
        "/sys/firmware/devicetree/base/model"
    ]
    
    for path in model_paths:
        try:
            with open(path) as f:
                model_str = f.read().strip().replace("\x00", "")
                if "Raspberry Pi" in model_str:
                    is_pi = True
                    # Extract model number
                    if "Zero 2" in model_str:
                        model = "Pi Zero 2W"
                    elif "Zero" in model_str:
                        model = "Pi Zero"
                    elif "5" in model_str:
                        model = "Pi 5"
                    elif "4" in model_str:
                        model = "Pi 4"
                    elif "3" in model_str:
                        model = "Pi 3"
                    elif "2" in model_str:
                        model = "Pi 2"
                    else:
                        model = model_str.replace("Raspberry ", "")
                    break
        except Exception:
            continue
    
    # Fallback model based on RAM if we know it's a Pi
    if is_pi and model is None:
        total_ram, _ = _detect_memory()
        if total_ram < 1:
            model = "Pi Zero"
        elif total_ram < 3:
            model = "Pi (unknown)"
        elif total_ram < 5:
            model = "Pi 4 (4GB)" 
        else:
            model = "Pi 4/5 (8GB)"
    
    return is_pi, model


def _recommend_model_and_quant(
    total_ram_gb: float,
    available_ram_gb: float,
    gpu_vram_gb: Optional[float],
    has_cuda: bool,
    has_mps: bool,
    is_raspberry_pi: bool,
    pi_model: Optional[str]
) -> tuple[str, str]:
    """
    Recommend model size and quantization based on hardware.
    
    Returns:
        (model_size, quantization) tuple
        
    📐 DECISION LOGIC:
    
    1. RASPBERRY PI SPECIFIC:
       - Pi Zero: pi_zero preset, int4 quantization
       - Pi 4 (4GB): pi_4 preset, int8 quantization
       - Pi 5 (8GB): pi_5 preset, dynamic quantization
    
    2. GPU AVAILABLE:
       - Use VRAM to determine size
       - No quantization needed (GPU is fast)
    
    3. CPU ONLY:
       - Use available RAM (need 2-3x model size for inference)
       - Dynamic quantization for larger models
    """
    model_size = "tiny"
    quantization = "none"
    
    # ─────────────────────────────────────────────────────────────────────────
    # RASPBERRY PI SPECIFIC RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    if is_raspberry_pi:
        if pi_model and "Zero" in pi_model:
            model_size = "pi_zero"
            quantization = "int4"
        elif total_ram_gb <= 2:
            model_size = "pi_zero"
            quantization = "int8"
        elif total_ram_gb <= 4:
            model_size = "pi_4"
            quantization = "int8"
        else:  # 8GB Pi
            model_size = "pi_5"
            quantization = "dynamic"
        return model_size, quantization
    
    # ─────────────────────────────────────────────────────────────────────────
    # GPU RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    if has_cuda and gpu_vram_gb is not None:
        if gpu_vram_gb >= 24:
            model_size = "xl"
        elif gpu_vram_gb >= 16:
            model_size = "large"
        elif gpu_vram_gb >= 8:
            model_size = "medium"
        elif gpu_vram_gb >= 4:
            model_size = "small"
        else:
            model_size = "tiny"
            quantization = "dynamic"
        return model_size, quantization
    
    if has_mps and gpu_vram_gb is not None:
        # Apple Silicon - shared memory, be conservative
        if gpu_vram_gb >= 16:
            model_size = "large"
        elif gpu_vram_gb >= 8:
            model_size = "medium"
        else:
            model_size = "small"
        return model_size, quantization
    
    # ─────────────────────────────────────────────────────────────────────────
    # CPU ONLY RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    # Need ~2-3x model size in RAM for inference
    usable_ram = available_ram_gb * 0.7  # Leave some for system
    
    if usable_ram >= 16:
        model_size = "large"
        quantization = "dynamic"
    elif usable_ram >= 8:
        model_size = "medium"
        quantization = "dynamic"
    elif usable_ram >= 4:
        model_size = "small"
        quantization = "dynamic"
    elif usable_ram >= 2:
        model_size = "tiny"
        quantization = "int8"
    else:
        model_size = "nano"
        quantization = "int8"
    
    return model_size, quantization


# =============================================================================
# 🔧 MAIN DETECTION FUNCTION
# =============================================================================

def detect_hardware() -> HardwareProfile:
    """
    Detect current hardware and return a complete profile.
    
    📖 WHAT THIS DOES:
    Runs all hardware detection routines and builds a HardwareProfile
    with recommendations for optimal model configuration.
    
    📐 EXAMPLE:
        profile = detect_hardware()
        print(profile.summary())
        
        # Output:
        # ============================================================
        # HARDWARE PROFILE
        # ============================================================
        # System: linux (aarch64)
        # CPU: ARM Cortex-A76 (4 cores)
        # RAM: 8.0 GB total, 6.5 GB available
        # GPU: None (CPU only)
        # Platform: Raspberry Pi 5
        # ------------------------------------------------------------
        # Recommended Model: pi_5
        # Recommended Quantization: dynamic
        # ============================================================
    
    Returns:
        HardwareProfile with all detected capabilities and recommendations
    """
    logger.info("Detecting hardware capabilities...")
    
    # Detect all components
    total_ram, available_ram = _detect_memory()
    gpu_name, gpu_vram, has_cuda, has_mps = _detect_gpu()
    cpu_cores, cpu_model = _detect_cpu()
    is_pi, pi_model = _detect_raspberry_pi()
    
    # Detect architecture
    machine = platform.machine().lower()
    is_arm = "arm" in machine or "aarch" in machine
    
    # Get recommendations
    rec_model, rec_quant = _recommend_model_and_quant(
        total_ram_gb=total_ram,
        available_ram_gb=available_ram,
        gpu_vram_gb=gpu_vram,
        has_cuda=has_cuda,
        has_mps=has_mps,
        is_raspberry_pi=is_pi,
        pi_model=pi_model
    )
    
    profile = HardwareProfile(
        total_ram_gb=round(total_ram, 2),
        available_ram_gb=round(available_ram, 2),
        gpu_name=gpu_name,
        gpu_vram_gb=round(gpu_vram, 2) if gpu_vram else None,
        cpu_cores=cpu_cores,
        cpu_model=cpu_model,
        is_arm=is_arm,
        is_raspberry_pi=is_pi,
        pi_model=pi_model,
        has_cuda=has_cuda,
        has_mps=has_mps,
        recommended_model_size=rec_model,
        recommended_quantization=rec_quant,
        system=platform.system().lower(),
        architecture=machine,
        python_version=platform.python_version(),
    )
    
    logger.info(f"Hardware detected: {profile.cpu_model}, {profile.total_ram_gb:.1f}GB RAM, "
                f"GPU={profile.gpu_name}, Pi={profile.is_raspberry_pi}")
    logger.info(f"Recommendations: model={rec_model}, quantization={rec_quant}")
    
    return profile


def recommend_model_size(profile: Optional[HardwareProfile] = None) -> str:
    """
    Get recommended model size for current or given hardware.
    
    Args:
        profile: Optional hardware profile. If None, detects current hardware.
    
    Returns:
        Model size preset name (e.g., "pi_5", "small", "medium")
    """
    if profile is None:
        profile = detect_hardware()
    return profile.recommended_model_size


# =============================================================================
# ⚙️ OPTIMAL CONFIG GENERATION
# =============================================================================

def get_optimal_config(profile: HardwareProfile) -> dict[str, Any]:
    """
    Get optimal ForgeConfig parameters for the given hardware profile.
    
    📖 WHAT THIS RETURNS:
    A dictionary of kwargs that can be passed to create_model() or ForgeConfig
    to create an optimally-configured model for this hardware.
    
    📐 EXAMPLE:
        profile = detect_hardware()
        config = get_optimal_config(profile)
        model = create_model(**config)
    
    Args:
        profile: Hardware profile from detect_hardware()
    
    Returns:
        Dictionary with keys: size, quantization, max_batch_size, use_half, etc.
    """
    config = {
        "size": profile.recommended_model_size,
        "quantization": profile.recommended_quantization,
    }
    
    # ─────────────────────────────────────────────────────────────────────────
    # BATCH SIZE RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    if profile.is_raspberry_pi:
        config["max_batch_size"] = 1
    elif profile.has_cuda and profile.gpu_vram_gb:
        if profile.gpu_vram_gb >= 16:
            config["max_batch_size"] = 8
        elif profile.gpu_vram_gb >= 8:
            config["max_batch_size"] = 4
        else:
            config["max_batch_size"] = 2
    else:
        config["max_batch_size"] = 1
    
    # ─────────────────────────────────────────────────────────────────────────
    # PRECISION RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    config["use_half"] = profile.has_cuda and profile.gpu_vram_gb and profile.gpu_vram_gb >= 4
    
    # ─────────────────────────────────────────────────────────────────────────
    # SEQUENCE LENGTH RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    if profile.is_raspberry_pi:
        if profile.pi_model and "Zero" in profile.pi_model:
            config["max_seq_len"] = 256
        elif profile.total_ram_gb <= 4:
            config["max_seq_len"] = 512
        else:
            config["max_seq_len"] = 1024
    elif profile.total_ram_gb < 4:
        config["max_seq_len"] = 512
    elif profile.total_ram_gb < 8:
        config["max_seq_len"] = 1024
    else:
        config["max_seq_len"] = 2048
    
    # ─────────────────────────────────────────────────────────────────────────
    # DEVICE SELECTION
    # ─────────────────────────────────────────────────────────────────────────
    if profile.has_cuda:
        config["device"] = "cuda"
    elif profile.has_mps:
        config["device"] = "mps"
    else:
        config["device"] = "cpu"
    
    return config


def estimate_memory_usage(size: str, quantization: str = "none") -> dict[str, float]:
    """
    Estimate RAM and VRAM requirements for a given model configuration.
    
    📖 WHAT THIS DOES:
    Calculates approximate memory requirements for loading and running
    a model of the given size with the specified quantization.
    
    📐 ESTIMATION FORMULA:
    - Base memory = parameters * bytes_per_param
    - FP32: 4 bytes/param
    - FP16: 2 bytes/param
    - INT8: 1 byte/param
    - INT4: 0.5 bytes/param
    - Dynamic: ~1.5 bytes/param (varies during inference)
    
    Args:
        size: Model size preset name
        quantization: Quantization type ("none", "int8", "int4", "dynamic")
    
    Returns:
        Dict with 'model_size_mb', 'inference_ram_mb', 'training_ram_mb'
    """
    # Approximate parameter counts for each preset
    PARAM_COUNTS = {
        "pi_zero": 500_000,       # ~500K params
        "pi_4": 3_000_000,        # ~3M params
        "pi_5": 8_000_000,        # ~8M params
        "nano": 1_000_000,        # ~1M params
        "micro": 2_000_000,       # ~2M params
        "tiny": 5_000_000,        # ~5M params
        "mini": 10_000_000,       # ~10M params
        "small": 27_000_000,      # ~27M params
        "medium": 85_000_000,     # ~85M params
        "base": 125_000_000,      # ~125M params
        "large": 200_000_000,     # ~200M params
        "xl": 600_000_000,        # ~600M params
        "xxl": 1_500_000_000,     # ~1.5B params
        "huge": 3_000_000_000,    # ~3B params
        "giant": 7_000_000_000,   # ~7B params
    }
    
    params = PARAM_COUNTS.get(size, 27_000_000)  # Default to small
    
    # Bytes per parameter based on quantization
    BYTES_PER_PARAM = {
        "none": 4.0,      # FP32
        "fp16": 2.0,      # FP16
        "int8": 1.0,      # INT8
        "int4": 0.5,      # INT4
        "dynamic": 1.5,   # Dynamic (approximate)
    }
    
    bytes_per = BYTES_PER_PARAM.get(quantization, 4.0)
    
    # Calculate sizes
    model_size_bytes = params * bytes_per
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    # Inference needs model + KV cache + activations (~2x model)
    inference_ram_mb = model_size_mb * 2.5
    
    # Training needs model + gradients + optimizer states + activations (~4-6x)
    training_ram_mb = model_size_mb * 5
    
    return {
        "model_size_mb": round(model_size_mb, 1),
        "inference_ram_mb": round(inference_ram_mb, 1),
        "training_ram_mb": round(training_ram_mb, 1),
        "parameters": params,
        "quantization": quantization,
    }


# =============================================================================
# 🔄 SINGLETON PATTERN
# =============================================================================

_cached_profile: Optional[HardwareProfile] = None


def get_cached_profile() -> HardwareProfile:
    """
    Get cached hardware profile (detects once, reuses).
    
    Use this for repeated access to avoid re-detecting hardware.
    """
    global _cached_profile
    if _cached_profile is None:
        _cached_profile = detect_hardware()
    return _cached_profile


def clear_cached_profile():
    """Clear cached profile to force re-detection."""
    global _cached_profile
    _cached_profile = None


# =============================================================================
# CLI TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FORGE AI HARDWARE DETECTION")
    print("=" * 60 + "\n")
    
    profile = detect_hardware()
    print(profile.summary())
    
    print("\nOptimal Configuration:")
    config = get_optimal_config(profile)
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nMemory Estimation:")
    memory = estimate_memory_usage(
        profile.recommended_model_size,
        profile.recommended_quantization
    )
    for key, value in memory.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f} MB")
        else:
            print(f"  {key}: {value}")
