"""
Power Mode Management - Control AI resource usage.

Modes:
  - FULL: Use all available resources (default)
  - BALANCED: Moderate resource usage
  - LOW: Minimal resources, slower responses
  - GAMING: Pause most AI functions, minimal CPU/GPU usage
  - BACKGROUND: Run in background with lowest priority
"""

import os
import threading
from enum import Enum
from typing import Optional
from dataclasses import dataclass


class PowerLevel(Enum):
    """Available power levels."""
    FULL = "full"
    BALANCED = "balanced"
    LOW = "low"
    GAMING = "gaming"
    BACKGROUND = "background"


@dataclass
class PowerSettings:
    """Settings for each power level."""
    max_batch_size: int
    max_tokens: int
    use_gpu: bool
    thread_count: int
    pause_background: bool
    response_delay: float  # Artificial delay to reduce CPU spikes


POWER_PRESETS = {
    PowerLevel.FULL: PowerSettings(
        max_batch_size=16,
        max_tokens=512,
        use_gpu=True,
        thread_count=0,  # Use all
        pause_background=False,
        response_delay=0.0
    ),
    PowerLevel.BALANCED: PowerSettings(
        max_batch_size=8,
        max_tokens=256,
        use_gpu=True,
        thread_count=4,
        pause_background=False,
        response_delay=0.1
    ),
    PowerLevel.LOW: PowerSettings(
        max_batch_size=2,
        max_tokens=128,
        use_gpu=False,  # CPU only
        thread_count=2,
        pause_background=True,
        response_delay=0.5
    ),
    PowerLevel.GAMING: PowerSettings(
        max_batch_size=1,
        max_tokens=64,
        use_gpu=False,
        thread_count=1,
        pause_background=True,
        response_delay=1.0
    ),
    PowerLevel.BACKGROUND: PowerSettings(
        max_batch_size=1,
        max_tokens=32,
        use_gpu=False,
        thread_count=1,
        pause_background=True,
        response_delay=2.0
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
    
    @property
    def level(self) -> PowerLevel:
        return self._level
    
    @property
    def settings(self) -> PowerSettings:
        return self._settings
    
    @property
    def is_paused(self) -> bool:
        return self._paused
    
    def set_level(self, level: PowerLevel):
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
                    pass
            
            # Set process priority on Windows
            if os.name == 'nt' and level in (PowerLevel.LOW, PowerLevel.GAMING, PowerLevel.BACKGROUND):
                try:
                    import psutil
                    p = psutil.Process()
                    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS') else 16384)
                except:
                    pass
    
    def pause(self):
        """Pause AI operations."""
        self._paused = True
    
    def resume(self):
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
    
    def wait_if_needed(self):
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


def set_power_mode(mode: str):
    """Set power mode by name."""
    level = PowerLevel(mode.lower())
    get_power_manager().set_level(level)


def gaming_mode(enabled: bool = True):
    """Quick toggle for gaming mode."""
    if enabled:
        set_power_mode("gaming")
    else:
        set_power_mode("full")
