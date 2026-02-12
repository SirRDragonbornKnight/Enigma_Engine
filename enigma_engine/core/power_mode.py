"""
Power Mode Management - Control AI resource usage.

Modes:
  - FULL: Use all available resources (default)
  - BALANCED: Moderate resource usage
  - LOW: Minimal resources, slower responses
  - GAMING: Pause most AI functions, minimal CPU/GPU usage
  - BACKGROUND: Run in background with lowest priority
  - EMBEDDED: Optimized for Raspberry Pi and embedded devices
  - MOBILE: Optimized for phones/tablets

Supports automatic mode selection based on hardware detection.
"""

import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PowerLevel(Enum):
    """Available power levels."""
    FULL = "full"
    BALANCED = "balanced"
    LOW = "low"
    GAMING = "gaming"
    BACKGROUND = "background"
    EMBEDDED = "embedded"  # Raspberry Pi, microcontrollers
    MOBILE = "mobile"      # Android, iOS


@dataclass
class PowerSettings:
    """Settings for each power level."""
    max_batch_size: int
    max_tokens: int
    use_gpu: bool
    thread_count: int
    pause_background: bool
    response_delay: float  # Artificial delay to reduce CPU spikes
    use_quantization: bool = False
    quantization_bits: int = 8
    max_memory_mb: int = 0  # 0 = no limit
    offload_to_cpu: bool = False


POWER_PRESETS = {
    PowerLevel.FULL: PowerSettings(
        max_batch_size=16,
        max_tokens=512,
        use_gpu=True,
        thread_count=0,  # Use all
        pause_background=False,
        response_delay=0.0,
        use_quantization=False,
    ),
    PowerLevel.BALANCED: PowerSettings(
        max_batch_size=8,
        max_tokens=256,
        use_gpu=True,
        thread_count=4,
        pause_background=False,
        response_delay=0.1,
        use_quantization=False,
    ),
    PowerLevel.LOW: PowerSettings(
        max_batch_size=2,
        max_tokens=128,
        use_gpu=False,  # CPU only
        thread_count=2,
        pause_background=True,
        response_delay=0.5,
        use_quantization=True,
        quantization_bits=8,
    ),
    PowerLevel.GAMING: PowerSettings(
        max_batch_size=1,
        max_tokens=64,
        use_gpu=False,
        thread_count=1,
        pause_background=True,
        response_delay=1.0,
        use_quantization=True,
        quantization_bits=8,
    ),
    PowerLevel.BACKGROUND: PowerSettings(
        max_batch_size=1,
        max_tokens=32,
        use_gpu=False,
        thread_count=1,
        pause_background=True,
        response_delay=2.0,
        use_quantization=True,
        quantization_bits=8,
    ),
    PowerLevel.EMBEDDED: PowerSettings(
        max_batch_size=1,
        max_tokens=32,
        use_gpu=False,
        thread_count=2,  # Pi has 4 cores, use 2
        pause_background=True,
        response_delay=0.5,
        use_quantization=True,
        quantization_bits=4,  # More aggressive quantization
        max_memory_mb=1024,  # 1GB limit
        offload_to_cpu=True,
    ),
    PowerLevel.MOBILE: PowerSettings(
        max_batch_size=1,
        max_tokens=64,
        use_gpu=False,  # Most phones don't support CUDA
        thread_count=4,  # Modern phones have good CPUs
        pause_background=True,
        response_delay=0.2,
        use_quantization=True,
        quantization_bits=8,
        max_memory_mb=2048,  # 2GB limit for phones
    ),
}


class PowerManager:
    """Manage AI power consumption."""
    
    _instance: Optional['PowerManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._level = PowerLevel.FULL
        self._settings = POWER_PRESETS[PowerLevel.FULL]
        self._paused = False
        self._lock = threading.Lock()
        
        # Auto-detect optimal level based on hardware
        self._auto_detected_level = self._detect_optimal_level()
    
    def _detect_optimal_level(self) -> PowerLevel:
        """Detect optimal power level based on hardware."""
        try:
            from .device_profiles import DeviceClass, get_device_profiler
            profiler = get_device_profiler()
            device_class = profiler.classify()
            
            # Map device class to power level
            level_map = {
                DeviceClass.EMBEDDED: PowerLevel.EMBEDDED,
                DeviceClass.MOBILE: PowerLevel.MOBILE,
                DeviceClass.LAPTOP_LOW: PowerLevel.LOW,
                DeviceClass.LAPTOP_MID: PowerLevel.BALANCED,
                DeviceClass.DESKTOP_CPU: PowerLevel.BALANCED,
                DeviceClass.DESKTOP_GPU: PowerLevel.FULL,
                DeviceClass.WORKSTATION: PowerLevel.FULL,
                DeviceClass.DATACENTER: PowerLevel.FULL,
            }
            return level_map.get(device_class, PowerLevel.BALANCED)
        except ImportError:
            return PowerLevel.BALANCED
        except Exception:
            return PowerLevel.BALANCED
    
    def auto_configure(self) -> PowerLevel:
        """Auto-configure based on detected hardware. Returns the selected level."""
        self.set_level(self._auto_detected_level)
        return self._auto_detected_level
    
    @property
    def level(self) -> PowerLevel:
        return self._level
    
    @property
    def settings(self) -> PowerSettings:
        return self._settings
    
    @property
    def is_paused(self) -> bool:
        return self._paused
    
    def set_level(self, level: PowerLevel) -> None:
        """Set power level."""
        with self._lock:
            self._level = level
            self._settings = POWER_PRESETS[level]
            
            # Apply thread limit
            if self._settings.thread_count > 0:
                try:
                    import torch
                    torch.set_num_threads(self._settings.thread_count)
                except ImportError:
                    pass  # Intentionally silent
            
            # Apply memory limit for embedded/mobile
            if self._settings.max_memory_mb > 0:
                try:
                    import resource

                    # Set soft limit (Unix only)
                    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                    new_limit = self._settings.max_memory_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
                except (ImportError, ValueError, OSError):
                    pass  # Not supported on this platform
            
            # Set process priority
            if level in (PowerLevel.LOW, PowerLevel.GAMING, PowerLevel.BACKGROUND, 
                        PowerLevel.EMBEDDED, PowerLevel.MOBILE):
                try:
                    import psutil
                    p = psutil.Process()
                    if os.name == 'nt':
                        # Windows: use BELOW_NORMAL_PRIORITY_CLASS
                        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS') else 16384)
                    else:
                        # Linux/macOS: use nice value (10 = lower priority)
                        p.nice(10)
                except (ImportError, OSError, AttributeError):
                    # Fallback: try os.nice on Linux/macOS
                    if os.name != 'nt':
                        try:
                            os.nice(10)
                        except (OSError, AttributeError):
                            pass  # Intentionally silent
    
    def pause(self) -> None:
        """Pause AI operations."""
        self._paused = True
    
    def resume(self) -> None:
        """Resume AI operations."""
        self._paused = False
    
    def should_use_gpu(self) -> bool:
        """Check if GPU should be used in current mode."""
        return self._settings.use_gpu
    
    def get_max_batch_size(self) -> int:
        """Get max batch size for current mode."""
        return self._settings.max_batch_size
    
    def get_max_tokens(self) -> int:
        """Get max tokens for current mode."""
        return self._settings.max_tokens
    
    def should_quantize(self) -> bool:
        """Check if quantization should be used."""
        return self._settings.use_quantization
    
    def get_quantization_bits(self) -> int:
        """Get quantization bits for current mode."""
        return self._settings.quantization_bits
    
    def wait_if_needed(self) -> None:
        """Wait if in low power mode (to reduce CPU spikes)."""
        if self._settings.response_delay > 0:
            import time
            time.sleep(self._settings.response_delay)
        
        # Block if paused
        while self._paused:
            import time
            time.sleep(0.5)


# Global instance
def get_power_manager() -> PowerManager:
    """Get the global PowerManager instance."""
    return PowerManager()


def set_power_mode(mode: str) -> None:
    """Set power mode by name."""
    level = PowerLevel(mode.lower())
    get_power_manager().set_level(level)


def gaming_mode(enabled: bool = True) -> None:
    """Quick toggle for gaming mode."""
    if enabled:
        set_power_mode("gaming")
    else:
        set_power_mode("full")


def embedded_mode(enabled: bool = True) -> None:
    """Quick toggle for embedded/Pi mode."""
    if enabled:
        set_power_mode("embedded")
    else:
        set_power_mode("balanced")


def mobile_mode(enabled: bool = True) -> None:
    """Quick toggle for mobile mode."""
    if enabled:
        set_power_mode("mobile")
    else:
        set_power_mode("balanced")


def auto_power_mode() -> PowerLevel:
    """Auto-detect and set optimal power mode based on hardware."""
    return get_power_manager().auto_configure()
