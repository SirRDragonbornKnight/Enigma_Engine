"""
Device Profiles - Automatic optimization for any hardware.

Automatically detects hardware and provides optimal settings for:
  - Raspberry Pi (ARM, limited RAM)
  - Mobile (Android/iOS)
  - Desktop (CPU only)
  - Desktop (with GPU)
  - Workstation (multi-GPU)
  - Datacenter (high-end)

This is the foundation for running Enigma AI Engine on ANY device.
"""

import logging
import os
import platform
import sys
import threading
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DeviceClass(Enum):
    """Hardware classification for optimization."""
    EMBEDDED = auto()      # Raspberry Pi, microcontrollers
    MOBILE = auto()        # Android, iOS
    LAPTOP_LOW = auto()    # Basic laptop, no GPU
    LAPTOP_MID = auto()    # Laptop with integrated GPU
    DESKTOP_CPU = auto()   # Desktop, CPU only
    DESKTOP_GPU = auto()   # Desktop with dedicated GPU
    WORKSTATION = auto()   # High-end, multi-GPU capable
    DATACENTER = auto()    # Server-grade hardware


class OperationMode(Enum):
    """How the device will be used."""
    INFERENCE = auto()     # Just running models (default)
    TRAINING = auto()      # Training models
    HYBRID = auto()        # Both training and inference
    EDGE = auto()          # Edge deployment (optimized for latency)
    SERVER = auto()        # Serving requests (optimized for throughput)


@dataclass
class DeviceCapabilities:
    """Detected hardware capabilities."""
    # CPU
    cpu_cores: int = 1
    cpu_threads: int = 1
    cpu_arch: str = "unknown"
    cpu_freq_mhz: int = 0
    has_avx: bool = False
    has_avx2: bool = False
    has_avx512: bool = False
    has_neon: bool = False  # ARM
    
    # Memory
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    swap_total_mb: int = 0
    
    # GPU
    has_cuda: bool = False
    has_mps: bool = False  # Apple Silicon
    has_rocm: bool = False  # AMD
    gpu_count: int = 0
    gpu_name: str = ""
    vram_total_mb: int = 0
    vram_available_mb: int = 0
    cuda_version: str = ""
    
    # Platform
    platform_system: str = ""
    platform_release: str = ""
    is_raspberry_pi: bool = False
    is_android: bool = False
    is_ios: bool = False
    is_wsl: bool = False
    is_container: bool = False
    is_64bit: bool = True
    
    # Storage
    disk_free_gb: float = 0.0
    has_ssd: bool = False


@dataclass
class ProfileSettings:
    """Optimized settings for a device profile."""
    # Model settings
    recommended_model_size: str = "tiny"
    max_model_params: int = 1_000_000
    use_quantization: bool = False
    quantization_bits: int = 8
    
    # Memory management
    max_batch_size: int = 1
    max_sequence_length: int = 256
    gradient_checkpointing: bool = False
    cpu_offload: bool = False
    disk_offload: bool = False
    
    # Compute settings
    use_gpu: bool = False
    gpu_layers: int = 0
    num_threads: int = 1
    use_flash_attention: bool = False
    use_half_precision: bool = False
    use_compile: bool = False  # torch.compile
    
    # Inference optimization
    use_kv_cache: bool = True
    kv_cache_size_mb: int = 128
    use_speculative_decoding: bool = False
    
    # Network (for distributed)
    can_serve_inference: bool = True
    can_serve_training: bool = False
    recommended_role: str = "client"  # client, server, hybrid
    
    # Generation defaults
    default_max_tokens: int = 64
    default_temperature: float = 0.8


# Pre-defined profiles for common hardware
DEVICE_PROFILES: dict[DeviceClass, ProfileSettings] = {
    DeviceClass.EMBEDDED: ProfileSettings(
        recommended_model_size="nano",
        max_model_params=2_000_000,
        use_quantization=True,
        quantization_bits=4,
        max_batch_size=1,
        max_sequence_length=128,
        cpu_offload=True,
        num_threads=2,
        use_kv_cache=True,
        kv_cache_size_mb=32,
        can_serve_inference=True,
        can_serve_training=False,
        recommended_role="client",
        default_max_tokens=32,
    ),
    DeviceClass.MOBILE: ProfileSettings(
        recommended_model_size="tiny",
        max_model_params=10_000_000,
        use_quantization=True,
        quantization_bits=8,
        max_batch_size=1,
        max_sequence_length=256,
        num_threads=4,
        use_kv_cache=True,
        kv_cache_size_mb=64,
        can_serve_inference=True,
        can_serve_training=False,
        recommended_role="client",
        default_max_tokens=64,
    ),
    DeviceClass.LAPTOP_LOW: ProfileSettings(
        recommended_model_size="small",
        max_model_params=50_000_000,
        use_quantization=True,
        quantization_bits=8,
        max_batch_size=2,
        max_sequence_length=512,
        num_threads=4,
        use_kv_cache=True,
        kv_cache_size_mb=256,
        can_serve_inference=True,
        can_serve_training=False,
        recommended_role="hybrid",
        default_max_tokens=128,
    ),
    DeviceClass.LAPTOP_MID: ProfileSettings(
        recommended_model_size="small",
        max_model_params=100_000_000,
        max_batch_size=4,
        max_sequence_length=1024,
        use_gpu=True,
        gpu_layers=10,
        num_threads=6,
        use_half_precision=True,
        use_kv_cache=True,
        kv_cache_size_mb=512,
        can_serve_inference=True,
        can_serve_training=True,
        recommended_role="hybrid",
        default_max_tokens=256,
    ),
    DeviceClass.DESKTOP_CPU: ProfileSettings(
        recommended_model_size="medium",
        max_model_params=200_000_000,
        max_batch_size=4,
        max_sequence_length=2048,
        num_threads=8,
        use_kv_cache=True,
        kv_cache_size_mb=1024,
        can_serve_inference=True,
        can_serve_training=True,
        recommended_role="server",
        default_max_tokens=256,
    ),
    DeviceClass.DESKTOP_GPU: ProfileSettings(
        recommended_model_size="large",
        max_model_params=500_000_000,
        max_batch_size=8,
        max_sequence_length=4096,
        use_gpu=True,
        gpu_layers=999,  # All layers
        num_threads=8,
        use_flash_attention=True,
        use_half_precision=True,
        use_kv_cache=True,
        kv_cache_size_mb=2048,
        can_serve_inference=True,
        can_serve_training=True,
        recommended_role="server",
        default_max_tokens=512,
    ),
    DeviceClass.WORKSTATION: ProfileSettings(
        recommended_model_size="xl",
        max_model_params=3_000_000_000,
        max_batch_size=16,
        max_sequence_length=8192,
        use_gpu=True,
        gpu_layers=999,
        num_threads=16,
        use_flash_attention=True,
        use_half_precision=True,
        use_compile=True,
        use_kv_cache=True,
        kv_cache_size_mb=4096,
        use_speculative_decoding=True,
        can_serve_inference=True,
        can_serve_training=True,
        recommended_role="server",
        default_max_tokens=1024,
    ),
    DeviceClass.DATACENTER: ProfileSettings(
        recommended_model_size="omega",
        max_model_params=70_000_000_000,
        max_batch_size=32,
        max_sequence_length=32768,
        gradient_checkpointing=True,
        use_gpu=True,
        gpu_layers=999,
        num_threads=32,
        use_flash_attention=True,
        use_half_precision=True,
        use_compile=True,
        use_kv_cache=True,
        kv_cache_size_mb=16384,
        use_speculative_decoding=True,
        can_serve_inference=True,
        can_serve_training=True,
        recommended_role="server",
        default_max_tokens=2048,
    ),
}


class DeviceProfiler:
    """
    Automatic hardware detection and profile selection.
    
    Usage:
        profiler = DeviceProfiler()
        profile = profiler.get_profile()
        
        # Get optimal settings
        model_size = profile.recommended_model_size
        threads = profile.num_threads
    """
    
    _instance: Optional['DeviceProfiler'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._capabilities: Optional[DeviceCapabilities] = None
        self._device_class: Optional[DeviceClass] = None
        self._profile: Optional[ProfileSettings] = None
        self._custom_overrides: dict[str, Any] = {}
        
    def detect(self, force: bool = False) -> DeviceCapabilities:
        """
        Detect hardware capabilities.
        
        Args:
            force: Re-detect even if already cached
            
        Returns:
            DeviceCapabilities with all detected info
        """
        if self._capabilities and not force:
            return self._capabilities
        
        caps = DeviceCapabilities()
        
        # Platform detection
        caps.platform_system = platform.system().lower()
        caps.platform_release = platform.release()
        caps.cpu_arch = platform.machine().lower()
        caps.is_64bit = sys.maxsize > 2**32
        
        # Raspberry Pi detection
        if caps.platform_system == "linux":
            try:
                with open("/proc/cpuinfo") as f:
                    cpuinfo = f.read().lower()
                    if "raspberry" in cpuinfo or "bcm" in cpuinfo:
                        caps.is_raspberry_pi = True
                        caps.has_neon = True
            except OSError:
                pass  # /proc/cpuinfo not available on this system
            
            # Android detection
            if "ANDROID_ROOT" in os.environ or "TERMUX_VERSION" in os.environ:
                caps.is_android = True
            
            # WSL detection
            try:
                with open("/proc/version") as f:
                    if "microsoft" in f.read().lower():
                        caps.is_wsl = True
            except OSError:
                pass  # /proc/version not available
            
            # Container detection
            if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
                caps.is_container = True
        
        # iOS detection
        if caps.platform_system == "darwin" and "iP" in platform.platform():
            caps.is_ios = True
        
        # CPU detection
        caps.cpu_cores = os.cpu_count() or 1
        caps.cpu_threads = caps.cpu_cores  # Assume SMT
        
        try:
            if caps.platform_system == "linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "flags" in line.lower():
                            flags = line.lower()
                            caps.has_avx = "avx" in flags
                            caps.has_avx2 = "avx2" in flags
                            caps.has_avx512 = "avx512" in flags
                            break
        except OSError:
            pass  # CPU flags not available on this system
        
        # ARM detection
        if "arm" in caps.cpu_arch or "aarch" in caps.cpu_arch:
            caps.has_neon = True
        
        # Memory detection
        try:
            if caps.platform_system == "linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if "MemTotal" in line:
                            caps.ram_total_mb = int(line.split()[1]) // 1024
                        elif "MemAvailable" in line:
                            caps.ram_available_mb = int(line.split()[1]) // 1024
                        elif "SwapTotal" in line:
                            caps.swap_total_mb = int(line.split()[1]) // 1024
            elif caps.platform_system == "windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulonglong = ctypes.c_ulonglong
                
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', c_ulonglong),
                        ('ullAvailPhys', c_ulonglong),
                        ('ullTotalPageFile', c_ulonglong),
                        ('ullAvailPageFile', c_ulonglong),
                        ('ullTotalVirtual', c_ulonglong),
                        ('ullAvailVirtual', c_ulonglong),
                        ('ullAvailExtendedVirtual', c_ulonglong),
                    ]
                
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                caps.ram_total_mb = stat.ullTotalPhys // (1024 * 1024)
                caps.ram_available_mb = stat.ullAvailPhys // (1024 * 1024)
        except Exception:
            caps.ram_total_mb = 4096  # Assume 4GB minimum
            caps.ram_available_mb = 2048
        
        # GPU detection
        try:
            import torch
            caps.has_cuda = torch.cuda.is_available()
            caps.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            
            if caps.has_cuda:
                caps.gpu_count = torch.cuda.device_count()
                caps.gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                caps.vram_total_mb = props.total_memory // (1024 * 1024)
                caps.vram_available_mb = (props.total_memory - torch.cuda.memory_allocated(0)) // (1024 * 1024)
                caps.cuda_version = torch.version.cuda or ""
        except ImportError:
            pass  # Intentionally silent
        
        # Disk detection
        try:
            import shutil
            total, used, free = shutil.disk_usage(Path.home())
            caps.disk_free_gb = free / (1024**3)
        except OSError:
            caps.disk_free_gb = 10.0  # Assume 10GB
        
        self._capabilities = caps
        return caps
    
    def classify(self, caps: Optional[DeviceCapabilities] = None) -> DeviceClass:
        """
        Classify device based on capabilities.
        
        Args:
            caps: Capabilities to classify (auto-detect if None)
            
        Returns:
            DeviceClass enum value
        """
        if caps is None:
            caps = self.detect()
        
        # Embedded devices
        if caps.is_raspberry_pi:
            return DeviceClass.EMBEDDED
        
        # Mobile devices
        if caps.is_android or caps.is_ios:
            return DeviceClass.MOBILE
        
        # Check for powerful GPU
        if caps.has_cuda and caps.vram_total_mb > 0:
            if caps.gpu_count > 1 or caps.vram_total_mb > 24000:
                return DeviceClass.DATACENTER
            elif caps.vram_total_mb > 12000:
                return DeviceClass.WORKSTATION
            elif caps.vram_total_mb > 4000:
                return DeviceClass.DESKTOP_GPU
            else:
                return DeviceClass.LAPTOP_MID
        
        # Apple Silicon
        if caps.has_mps:
            if caps.ram_total_mb > 32000:
                return DeviceClass.WORKSTATION
            elif caps.ram_total_mb > 16000:
                return DeviceClass.DESKTOP_GPU
            else:
                return DeviceClass.LAPTOP_MID
        
        # CPU only - classify by RAM and cores
        if caps.ram_total_mb < 4000:
            return DeviceClass.EMBEDDED
        elif caps.ram_total_mb < 8000:
            return DeviceClass.LAPTOP_LOW
        elif caps.cpu_cores < 6:
            return DeviceClass.LAPTOP_LOW
        else:
            return DeviceClass.DESKTOP_CPU
    
    def get_profile(self, 
                    device_class: Optional[DeviceClass] = None,
                    mode: OperationMode = OperationMode.INFERENCE) -> ProfileSettings:
        """
        Get optimized settings for the device.
        
        Args:
            device_class: Override detected class
            mode: How the device will be used
            
        Returns:
            ProfileSettings with optimal configuration
        """
        if device_class is None:
            device_class = self.classify()
        
        self._device_class = device_class
        
        # Get base profile
        profile = DEVICE_PROFILES.get(device_class, DEVICE_PROFILES[DeviceClass.LAPTOP_LOW])
        
        # Make a copy for modification
        import copy
        profile = copy.deepcopy(profile)
        
        # Adjust for operation mode
        if mode == OperationMode.TRAINING:
            profile.gradient_checkpointing = True
            profile.max_batch_size = max(1, profile.max_batch_size // 2)
            profile.can_serve_training = True
        elif mode == OperationMode.EDGE:
            profile.max_batch_size = 1
            profile.default_max_tokens = max(32, profile.default_max_tokens // 2)
            profile.use_speculative_decoding = False
        elif mode == OperationMode.SERVER:
            profile.max_batch_size *= 2
            profile.recommended_role = "server"
        
        # Apply any custom overrides
        for key, value in self._custom_overrides.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        self._profile = profile
        return profile
    
    def override(self, **kwargs) -> 'DeviceProfiler':
        """
        Override specific profile settings.
        
        Args:
            **kwargs: Settings to override
            
        Returns:
            Self for chaining
        """
        self._custom_overrides.update(kwargs)
        if self._profile:
            for key, value in kwargs.items():
                if hasattr(self._profile, key):
                    setattr(self._profile, key, value)
        return self
    
    def get_torch_device(self) -> str:
        """Get the best PyTorch device string."""
        caps = self.detect()
        if caps.has_cuda:
            return "cuda"
        elif caps.has_mps:
            return "mps"
        return "cpu"
    
    def get_torch_dtype(self):
        """Get the best PyTorch dtype."""
        import torch
        profile = self.get_profile()
        
        if profile.use_half_precision:
            caps = self.detect()
            if caps.has_cuda:
                return torch.float16
            elif caps.has_mps:
                return torch.float16
        return torch.float32
    
    def summary(self) -> str:
        """Get a human-readable summary."""
        caps = self.detect()
        device_class = self.classify()
        profile = self.get_profile()
        
        lines = [
            "=" * 50,
            "DEVICE PROFILE SUMMARY",
            "=" * 50,
            f"Device Class: {device_class.name}",
            f"",
            f"Hardware:",
            f"  CPU: {caps.cpu_cores} cores, {caps.cpu_arch}",
            f"  RAM: {caps.ram_total_mb / 1024:.1f} GB",
        ]
        
        if caps.has_cuda:
            lines.append(f"  GPU: {caps.gpu_name} ({caps.vram_total_mb / 1024:.1f} GB VRAM)")
        elif caps.has_mps:
            lines.append(f"  GPU: Apple Silicon (MPS)")
        else:
            lines.append(f"  GPU: None (CPU only)")
        
        if caps.is_raspberry_pi:
            lines.append(f"  Platform: Raspberry Pi")
        elif caps.is_android:
            lines.append(f"  Platform: Android")
        
        lines.extend([
            f"",
            f"Recommended Settings:",
            f"  Model Size: {profile.recommended_model_size}",
            f"  Max Params: {profile.max_model_params:,}",
            f"  Batch Size: {profile.max_batch_size}",
            f"  Threads: {profile.num_threads}",
            f"  GPU Layers: {profile.gpu_layers if profile.use_gpu else 'N/A'}",
            f"  Quantization: {'Yes' if profile.use_quantization else 'No'}",
            f"  Role: {profile.recommended_role}",
            "=" * 50,
        ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """Export profile as dictionary."""
        caps = self.detect()
        device_class = self.classify()
        profile = self.get_profile()
        
        return {
            "device_class": device_class.name,
            "capabilities": {
                "cpu_cores": caps.cpu_cores,
                "ram_total_mb": caps.ram_total_mb,
                "has_cuda": caps.has_cuda,
                "has_mps": caps.has_mps,
                "gpu_name": caps.gpu_name,
                "vram_total_mb": caps.vram_total_mb,
                "is_raspberry_pi": caps.is_raspberry_pi,
                "is_android": caps.is_android,
            },
            "profile": {
                "recommended_model_size": profile.recommended_model_size,
                "max_model_params": profile.max_model_params,
                "max_batch_size": profile.max_batch_size,
                "num_threads": profile.num_threads,
                "use_gpu": profile.use_gpu,
                "gpu_layers": profile.gpu_layers,
                "use_quantization": profile.use_quantization,
                "recommended_role": profile.recommended_role,
            }
        }


# Global instance accessor
_profiler: Optional[DeviceProfiler] = None


def get_device_profiler() -> DeviceProfiler:
    """Get the global DeviceProfiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = DeviceProfiler()
    return _profiler


def get_optimal_settings() -> ProfileSettings:
    """Quick helper to get optimal settings for this device."""
    return get_device_profiler().get_profile()


def get_device_class() -> DeviceClass:
    """Quick helper to get device classification."""
    return get_device_profiler().classify()


def get_recommended_model_size() -> str:
    """Quick helper to get recommended model size."""
    return get_device_profiler().get_profile().recommended_model_size


if __name__ == "__main__":
    profiler = DeviceProfiler()
    print(profiler.summary())
