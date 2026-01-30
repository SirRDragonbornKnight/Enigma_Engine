"""
================================================================================
Battery Manager - Optimize power usage for mobile/embedded devices.
================================================================================

Intelligent power management:
- Battery level monitoring
- Adaptive performance scaling
- Low power mode automation
- Thermal throttling

USAGE:
    from forge_ai.utils.battery_manager import BatteryManager
    
    manager = BatteryManager()
    manager.start()
    
    # Check status
    print(f"Battery: {manager.get_level()}%")
    print(f"Charging: {manager.is_charging()}")
    
    # Auto-adjusts AI performance based on battery
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


class PowerState(Enum):
    """Device power states."""
    FULL_POWER = auto()      # AC power, no limits
    HIGH_POWER = auto()      # Battery > 50%, minor limits
    BALANCED = auto()        # Battery 20-50%, moderate limits
    LOW_POWER = auto()       # Battery < 20%, significant limits
    CRITICAL = auto()        # Battery < 10%, emergency mode
    CHARGING = auto()        # Plugged in and charging


@dataclass
class PowerProfile:
    """Settings for a power state."""
    name: str
    cpu_limit: float        # 0.0 - 1.0, limit CPU usage
    gpu_enabled: bool       # Enable GPU acceleration
    sync_interval: float    # Seconds between network syncs
    voice_enabled: bool     # Enable voice processing
    avatar_fps: int         # Avatar animation FPS
    model_size: str         # AI model size to use
    background_tasks: bool  # Enable background processing


# Default profiles
POWER_PROFILES = {
    PowerState.FULL_POWER: PowerProfile(
        name="Full Power",
        cpu_limit=1.0,
        gpu_enabled=True,
        sync_interval=0.1,
        voice_enabled=True,
        avatar_fps=60,
        model_size="medium",
        background_tasks=True,
    ),
    PowerState.HIGH_POWER: PowerProfile(
        name="High Power",
        cpu_limit=0.8,
        gpu_enabled=True,
        sync_interval=0.2,
        voice_enabled=True,
        avatar_fps=30,
        model_size="small",
        background_tasks=True,
    ),
    PowerState.BALANCED: PowerProfile(
        name="Balanced",
        cpu_limit=0.5,
        gpu_enabled=False,
        sync_interval=0.5,
        voice_enabled=True,
        avatar_fps=15,
        model_size="small",
        background_tasks=False,
    ),
    PowerState.LOW_POWER: PowerProfile(
        name="Low Power",
        cpu_limit=0.3,
        gpu_enabled=False,
        sync_interval=1.0,
        voice_enabled=False,
        avatar_fps=10,
        model_size="tiny",
        background_tasks=False,
    ),
    PowerState.CRITICAL: PowerProfile(
        name="Critical",
        cpu_limit=0.1,
        gpu_enabled=False,
        sync_interval=5.0,
        voice_enabled=False,
        avatar_fps=5,
        model_size="nano",
        background_tasks=False,
    ),
    PowerState.CHARGING: PowerProfile(
        name="Charging",
        cpu_limit=0.9,
        gpu_enabled=True,
        sync_interval=0.2,
        voice_enabled=True,
        avatar_fps=30,
        model_size="small",
        background_tasks=True,
    ),
}


class BatteryManager:
    """
    Manages power usage and battery life.
    
    Features:
    - Real-time battery monitoring
    - Automatic profile switching
    - Performance scaling
    - Callbacks for power events
    """
    
    def __init__(
        self,
        check_interval: float = 30.0,
        custom_profiles: Dict[PowerState, PowerProfile] = None,
    ):
        self.check_interval = check_interval
        self.profiles = POWER_PROFILES.copy()
        if custom_profiles:
            self.profiles.update(custom_profiles)
        
        # Current state
        self._state = PowerState.FULL_POWER
        self._level = 100.0
        self._charging = False
        self._temperature = 0.0
        
        # Callbacks
        self._state_callbacks: List[Callable[[PowerState, PowerProfile], None]] = []
        self._level_callbacks: List[Callable[[float], None]] = []
        
        # Threading
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
        # Platform-specific battery reader
        self._battery_reader = self._init_battery_reader()
    
    def start(self):
        """Start battery monitoring."""
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Battery manager started")
    
    def stop(self):
        """Stop battery monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Battery manager stopped")
    
    def get_level(self) -> float:
        """Get battery level (0-100)."""
        with self._lock:
            return self._level
    
    def is_charging(self) -> bool:
        """Check if device is charging."""
        with self._lock:
            return self._charging
    
    def get_state(self) -> PowerState:
        """Get current power state."""
        with self._lock:
            return self._state
    
    def get_profile(self) -> PowerProfile:
        """Get current power profile."""
        with self._lock:
            return self.profiles[self._state]
    
    def get_temperature(self) -> float:
        """Get device temperature (Celsius)."""
        with self._lock:
            return self._temperature
    
    def on_state_change(self, callback: Callable[[PowerState, PowerProfile], None]):
        """Register callback for power state changes."""
        self._state_callbacks.append(callback)
    
    def on_level_change(self, callback: Callable[[float], None]):
        """Register callback for battery level changes."""
        self._level_callbacks.append(callback)
    
    def force_state(self, state: PowerState):
        """Force a specific power state."""
        with self._lock:
            if self._state != state:
                old_state = self._state
                self._state = state
                profile = self.profiles[state]
                logger.info(f"Power state forced: {old_state.name} -> {state.name}")
                
                for callback in self._state_callbacks:
                    try:
                        callback(state, profile)
                    except Exception as e:
                        logger.error(f"State callback error: {e}")
    
    def get_recommended_settings(self) -> Dict[str, Any]:
        """Get recommended settings for current power state."""
        profile = self.get_profile()
        return {
            "cpu_limit": profile.cpu_limit,
            "gpu_enabled": profile.gpu_enabled,
            "sync_interval": profile.sync_interval,
            "voice_enabled": profile.voice_enabled,
            "avatar_fps": profile.avatar_fps,
            "model_size": profile.model_size,
            "background_tasks": profile.background_tasks,
        }
    
    def _init_battery_reader(self):
        """Initialize platform-specific battery reader."""
        import platform
        system = platform.system().lower()
        
        if system == "windows":
            return self._read_battery_windows
        elif system == "linux":
            return self._read_battery_linux
        elif system == "darwin":
            return self._read_battery_macos
        else:
            return self._read_battery_fallback
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Read battery status
                level, charging, temp = self._battery_reader()
                
                with self._lock:
                    old_level = self._level
                    old_state = self._state
                    
                    self._level = level
                    self._charging = charging
                    self._temperature = temp
                    
                    # Determine new state
                    new_state = self._determine_state(level, charging, temp)
                    
                    if new_state != old_state:
                        self._state = new_state
                        profile = self.profiles[new_state]
                        logger.info(f"Power state: {old_state.name} -> {new_state.name}")
                        
                        for callback in self._state_callbacks:
                            try:
                                callback(new_state, profile)
                            except Exception as e:
                                logger.error(f"State callback error: {e}")
                    
                    # Notify level change if significant
                    if abs(old_level - level) >= 1.0:
                        for callback in self._level_callbacks:
                            try:
                                callback(level)
                            except Exception as e:
                                logger.error(f"Level callback error: {e}")
                
            except Exception as e:
                logger.error(f"Battery monitor error: {e}")
            
            time.sleep(self.check_interval)
    
    def _determine_state(self, level: float, charging: bool, temp: float) -> PowerState:
        """Determine power state from battery status."""
        # Thermal throttling
        if temp > 80:  # Celsius
            return PowerState.CRITICAL
        elif temp > 70:
            return PowerState.LOW_POWER
        
        # Charging
        if charging:
            return PowerState.CHARGING
        
        # Battery level
        if level >= 80:
            return PowerState.FULL_POWER
        elif level >= 50:
            return PowerState.HIGH_POWER
        elif level >= 20:
            return PowerState.BALANCED
        elif level >= 10:
            return PowerState.LOW_POWER
        else:
            return PowerState.CRITICAL
    
    def _read_battery_windows(self) -> tuple:
        """Read battery on Windows."""
        try:
            import ctypes
            
            class SYSTEM_POWER_STATUS(ctypes.Structure):
                _fields_ = [
                    ('ACLineStatus', ctypes.c_byte),
                    ('BatteryFlag', ctypes.c_byte),
                    ('BatteryLifePercent', ctypes.c_byte),
                    ('Reserved1', ctypes.c_byte),
                    ('BatteryLifeTime', ctypes.c_ulong),
                    ('BatteryFullLifeTime', ctypes.c_ulong),
                ]
            
            status = SYSTEM_POWER_STATUS()
            ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
            
            level = status.BatteryLifePercent
            if level == 255:
                level = 100.0  # No battery
            
            charging = status.ACLineStatus == 1
            temp = 40.0  # Default, Windows doesn't expose easily
            
            return float(level), charging, temp
            
        except Exception as e:
            logger.error(f"Windows battery read error: {e}")
            return 100.0, True, 40.0
    
    def _read_battery_linux(self) -> tuple:
        """Read battery on Linux."""
        try:
            from pathlib import Path
            
            # Check /sys/class/power_supply/
            power_supply = Path("/sys/class/power_supply")
            
            level = 100.0
            charging = True
            temp = 40.0
            
            for battery in power_supply.glob("BAT*"):
                # Read capacity
                capacity_file = battery / "capacity"
                if capacity_file.exists():
                    level = float(capacity_file.read_text().strip())
                
                # Read status
                status_file = battery / "status"
                if status_file.exists():
                    status = status_file.read_text().strip().lower()
                    charging = status in ("charging", "full")
                
                break
            
            # Try to read temperature
            for thermal in power_supply.parent.glob("thermal_zone*"):
                temp_file = thermal / "temp"
                if temp_file.exists():
                    temp = float(temp_file.read_text().strip()) / 1000
                    break
            
            return level, charging, temp
            
        except Exception as e:
            logger.error(f"Linux battery read error: {e}")
            return 100.0, True, 40.0
    
    def _read_battery_macos(self) -> tuple:
        """Read battery on macOS."""
        try:
            import subprocess
            
            result = subprocess.run(
                ["pmset", "-g", "batt"],
                capture_output=True,
                text=True
            )
            
            output = result.stdout
            
            # Parse output
            level = 100.0
            charging = True
            
            if "%" in output:
                # Extract percentage
                import re
                match = re.search(r'(\d+)%', output)
                if match:
                    level = float(match.group(1))
            
            charging = "charging" in output.lower() or "AC Power" in output
            
            return level, charging, 40.0
            
        except Exception as e:
            logger.error(f"macOS battery read error: {e}")
            return 100.0, True, 40.0
    
    def _read_battery_fallback(self) -> tuple:
        """Fallback battery reader."""
        # Try psutil if available
        try:
            import psutil
            battery = psutil.sensors_battery()
            if battery:
                return battery.percent, battery.power_plugged, 40.0
        except Exception:
            pass
        
        # Default: assume full power
        return 100.0, True, 40.0


# Global manager instance
_manager: Optional[BatteryManager] = None


def get_battery_manager(**kwargs) -> BatteryManager:
    """Get or create global battery manager."""
    global _manager
    if _manager is None:
        _manager = BatteryManager(**kwargs)
    return _manager


__all__ = [
    'BatteryManager',
    'PowerState',
    'PowerProfile',
    'POWER_PROFILES',
    'get_battery_manager',
]
