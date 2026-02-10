"""
Power Management for Edge Devices

Manage power consumption for Raspberry Pi and edge devices.
Reduces CPU/GPU usage when idle to extend battery life.

FILE: enigma_engine/edge/power_management.py
TYPE: Edge
MAIN CLASSES: PowerManager, ThermalManager, BatteryMonitor
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PowerState(Enum):
    """Power states."""
    PERFORMANCE = auto()  # Maximum performance
    BALANCED = auto()     # Balance performance/power
    POWER_SAVE = auto()   # Minimize power
    IDLE = auto()         # System idle
    SLEEP = auto()        # Deep sleep


class ThermalState(Enum):
    """Thermal states."""
    COOL = auto()       # < 50°C
    WARM = auto()       # 50-65°C
    HOT = auto()        # 65-80°C
    CRITICAL = auto()   # > 80°C


@dataclass
class PowerConfig:
    """Power management configuration."""
    # Idle detection
    idle_timeout_seconds: int = 60
    deep_sleep_timeout_seconds: int = 300
    
    # CPU scaling
    min_cpu_freq_mhz: int = 600
    max_cpu_freq_mhz: int = 1500
    balanced_cpu_freq_mhz: int = 1000
    
    # GPU (if available)
    min_gpu_freq_mhz: int = 300
    max_gpu_freq_mhz: int = 500
    
    # Thermal limits
    warm_threshold_c: float = 50.0
    hot_threshold_c: float = 65.0
    critical_threshold_c: float = 80.0
    
    # Battery (if available)
    low_battery_percent: int = 20
    critical_battery_percent: int = 10


@dataclass
class PowerMetrics:
    """Current power metrics."""
    state: PowerState = PowerState.BALANCED
    cpu_freq_mhz: int = 0
    gpu_freq_mhz: int = 0
    temperature_c: float = 0.0
    thermal_state: ThermalState = ThermalState.COOL
    battery_percent: Optional[int] = None
    is_charging: bool = False
    power_draw_watts: float = 0.0
    idle_seconds: float = 0.0


class CPUGovernor:
    """CPU frequency scaling."""
    
    SYSFS_CPU_PATH = Path("/sys/devices/system/cpu")
    GOVERNORS = ["performance", "ondemand", "powersave", "conservative", "schedutil"]
    
    def __init__(self):
        self.num_cores = self._detect_cores()
    
    def _detect_cores(self) -> int:
        """Detect number of CPU cores."""
        try:
            return os.cpu_count() or 1
        except (OSError, AttributeError):
            return 1
    
    def get_available_governors(self) -> list[str]:
        """Get available CPU governors."""
        try:
            path = self.SYSFS_CPU_PATH / "cpu0/cpufreq/scaling_available_governors"
            if path.exists():
                return path.read_text().strip().split()
        except OSError:
            pass
        return []
    
    def get_current_governor(self) -> str:
        """Get current CPU governor."""
        try:
            path = self.SYSFS_CPU_PATH / "cpu0/cpufreq/scaling_governor"
            if path.exists():
                return path.read_text().strip()
        except OSError:
            pass
        return "unknown"
    
    def set_governor(self, governor: str) -> bool:
        """Set CPU governor for all cores."""
        if governor not in self.get_available_governors():
            logger.warning(f"Governor {governor} not available")
            return False
        
        try:
            for i in range(self.num_cores):
                path = self.SYSFS_CPU_PATH / f"cpu{i}/cpufreq/scaling_governor"
                if path.exists():
                    path.write_text(governor)
            logger.info(f"Set CPU governor to {governor}")
            return True
        except PermissionError:
            logger.warning("Permission denied - run as root to change governor")
        except Exception as e:
            logger.error(f"Failed to set governor: {e}")
        return False
    
    def get_frequency(self, core: int = 0) -> int:
        """Get current CPU frequency in MHz."""
        try:
            path = self.SYSFS_CPU_PATH / f"cpu{core}/cpufreq/scaling_cur_freq"
            if path.exists():
                return int(path.read_text().strip()) // 1000
        except (OSError, ValueError):
            pass
        return 0
    
    def set_frequency_limits(self, min_mhz: int, max_mhz: int) -> bool:
        """Set CPU frequency limits."""
        try:
            for i in range(self.num_cores):
                min_path = self.SYSFS_CPU_PATH / f"cpu{i}/cpufreq/scaling_min_freq"
                max_path = self.SYSFS_CPU_PATH / f"cpu{i}/cpufreq/scaling_max_freq"
                
                if min_path.exists():
                    min_path.write_text(str(min_mhz * 1000))
                if max_path.exists():
                    max_path.write_text(str(max_mhz * 1000))
            
            logger.info(f"Set CPU frequency limits: {min_mhz}-{max_mhz} MHz")
            return True
        except PermissionError:
            logger.warning("Permission denied - run as root")
        except Exception as e:
            logger.error(f"Failed to set frequency: {e}")
        return False
    
    def disable_cores(self, cores_to_disable: list[int]) -> bool:
        """Disable CPU cores to save power."""
        try:
            for core in cores_to_disable:
                if core == 0:
                    continue  # Cannot disable core 0
                path = self.SYSFS_CPU_PATH / f"cpu{core}/online"
                if path.exists():
                    path.write_text("0")
            logger.info(f"Disabled cores: {cores_to_disable}")
            return True
        except Exception as e:
            logger.error(f"Failed to disable cores: {e}")
        return False
    
    def enable_all_cores(self) -> bool:
        """Enable all CPU cores."""
        try:
            for i in range(1, self.num_cores):
                path = self.SYSFS_CPU_PATH / f"cpu{i}/online"
                if path.exists():
                    path.write_text("1")
            logger.info("Enabled all CPU cores")
            return True
        except Exception as e:
            logger.error(f"Failed to enable cores: {e}")
        return False


class ThermalManager:
    """Thermal monitoring and management."""
    
    THERMAL_ZONES = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
        "/sys/class/hwmon/hwmon0/temp1_input",
    ]
    
    def __init__(self, config: PowerConfig):
        self.config = config
        self._zone_path: Optional[Path] = None
        self._find_thermal_zone()
    
    def _find_thermal_zone(self):
        """Find available thermal zone."""
        for zone in self.THERMAL_ZONES:
            path = Path(zone)
            if path.exists():
                self._zone_path = path
                logger.info(f"Found thermal zone: {path}")
                return
    
    def get_temperature(self) -> float:
        """Get current CPU temperature in Celsius."""
        if self._zone_path is None:
            return 0.0
        
        try:
            temp = int(self._zone_path.read_text().strip())
            # Most zones report in millidegrees
            if temp > 1000:
                temp = temp / 1000
            return temp
        except Exception as e:
            logger.error(f"Failed to read temperature: {e}")
            return 0.0
    
    def get_thermal_state(self) -> ThermalState:
        """Get current thermal state."""
        temp = self.get_temperature()
        
        if temp >= self.config.critical_threshold_c:
            return ThermalState.CRITICAL
        elif temp >= self.config.hot_threshold_c:
            return ThermalState.HOT
        elif temp >= self.config.warm_threshold_c:
            return ThermalState.WARM
        else:
            return ThermalState.COOL
    
    def should_throttle(self) -> bool:
        """Check if system should throttle due to heat."""
        return self.get_thermal_state() in (ThermalState.HOT, ThermalState.CRITICAL)


class BatteryMonitor:
    """Battery status monitoring."""
    
    BATTERY_PATHS = [
        "/sys/class/power_supply/BAT0",
        "/sys/class/power_supply/BAT1",
        "/sys/class/power_supply/battery",
    ]
    
    def __init__(self, config: PowerConfig):
        self.config = config
        self._battery_path: Optional[Path] = None
        self._find_battery()
    
    def _find_battery(self):
        """Find battery sysfs path."""
        for path in self.BATTERY_PATHS:
            p = Path(path)
            if p.exists():
                self._battery_path = p
                logger.info(f"Found battery: {p}")
                return
    
    @property
    def has_battery(self) -> bool:
        """Check if system has a battery."""
        return self._battery_path is not None
    
    def get_percent(self) -> Optional[int]:
        """Get battery percentage."""
        if not self._battery_path:
            return None
        
        try:
            capacity_path = self._battery_path / "capacity"
            if capacity_path.exists():
                return int(capacity_path.read_text().strip())
        except (OSError, ValueError):
            pass
        return None
    
    def is_charging(self) -> bool:
        """Check if battery is charging."""
        if not self._battery_path:
            return False
        
        try:
            status_path = self._battery_path / "status"
            if status_path.exists():
                status = status_path.read_text().strip().lower()
                return status in ("charging", "full")
        except OSError:
            pass
        return False
    
    def is_low(self) -> bool:
        """Check if battery is low."""
        percent = self.get_percent()
        if percent is None:
            return False
        return percent <= self.config.low_battery_percent
    
    def is_critical(self) -> bool:
        """Check if battery is critical."""
        percent = self.get_percent()
        if percent is None:
            return False
        return percent <= self.config.critical_battery_percent


class PowerManager:
    """Main power management controller."""
    
    def __init__(self, config: PowerConfig = None):
        self.config = config or PowerConfig()
        self.cpu = CPUGovernor()
        self.thermal = ThermalManager(self.config)
        self.battery = BatteryMonitor(self.config)
        
        self._state = PowerState.BALANCED
        self._last_activity = time.time()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Callbacks
        self._state_callbacks: list[Callable[[PowerState], None]] = []
    
    @property
    def current_state(self) -> PowerState:
        """Get current power state."""
        return self._state
    
    def get_metrics(self) -> PowerMetrics:
        """Get current power metrics."""
        return PowerMetrics(
            state=self._state,
            cpu_freq_mhz=self.cpu.get_frequency(),
            temperature_c=self.thermal.get_temperature(),
            thermal_state=self.thermal.get_thermal_state(),
            battery_percent=self.battery.get_percent(),
            is_charging=self.battery.is_charging(),
            idle_seconds=time.time() - self._last_activity,
        )
    
    def set_state(self, state: PowerState):
        """Set power state."""
        if state == self._state:
            return
        
        old_state = self._state
        self._state = state
        
        logger.info(f"Power state: {old_state.name} -> {state.name}")
        
        # Apply state
        if state == PowerState.PERFORMANCE:
            self._apply_performance()
        elif state == PowerState.BALANCED:
            self._apply_balanced()
        elif state == PowerState.POWER_SAVE:
            self._apply_power_save()
        elif state == PowerState.IDLE:
            self._apply_idle()
        elif state == PowerState.SLEEP:
            self._apply_sleep()
        
        # Notify callbacks
        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _apply_performance(self):
        """Apply performance mode."""
        self.cpu.set_governor("performance")
        self.cpu.set_frequency_limits(
            self.config.max_cpu_freq_mhz - 200,
            self.config.max_cpu_freq_mhz
        )
        self.cpu.enable_all_cores()
    
    def _apply_balanced(self):
        """Apply balanced mode."""
        self.cpu.set_governor("ondemand")
        self.cpu.set_frequency_limits(
            self.config.min_cpu_freq_mhz,
            self.config.max_cpu_freq_mhz
        )
        self.cpu.enable_all_cores()
    
    def _apply_power_save(self):
        """Apply power save mode."""
        self.cpu.set_governor("powersave")
        self.cpu.set_frequency_limits(
            self.config.min_cpu_freq_mhz,
            self.config.balanced_cpu_freq_mhz
        )
        # Disable half the cores
        cores_to_disable = list(range(self.cpu.num_cores // 2, self.cpu.num_cores))
        if cores_to_disable:
            self.cpu.disable_cores(cores_to_disable)
    
    def _apply_idle(self):
        """Apply idle mode."""
        self.cpu.set_governor("powersave")
        self.cpu.set_frequency_limits(
            self.config.min_cpu_freq_mhz,
            self.config.min_cpu_freq_mhz + 200
        )
        # Keep only 2 cores active
        cores_to_disable = list(range(2, self.cpu.num_cores))
        if cores_to_disable:
            self.cpu.disable_cores(cores_to_disable)
    
    def _apply_sleep(self):
        """Apply deep sleep mode."""
        self._apply_idle()
        # Additional sleep preparations if needed
    
    def activity(self):
        """Mark user activity (prevents idle/sleep)."""
        self._last_activity = time.time()
        
        if self._state in (PowerState.IDLE, PowerState.SLEEP):
            self.set_state(PowerState.BALANCED)
    
    def add_state_callback(self, callback: Callable[[PowerState], None]):
        """Add callback for state changes."""
        self._state_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start background power monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Power monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Power monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self._check_state()
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            time.sleep(5)  # Check every 5 seconds
    
    def _check_state(self):
        """Check and update power state based on conditions."""
        idle_time = time.time() - self._last_activity
        
        # Check thermal throttling
        if self.thermal.should_throttle():
            if self._state == PowerState.PERFORMANCE:
                logger.warning("Thermal throttling - switching to balanced")
                self.set_state(PowerState.BALANCED)
            elif self.thermal.get_thermal_state() == ThermalState.CRITICAL:
                logger.warning("Critical temperature - switching to power save")
                self.set_state(PowerState.POWER_SAVE)
            return
        
        # Check battery
        if self.battery.is_critical() and not self.battery.is_charging():
            logger.warning("Critical battery - switching to power save")
            self.set_state(PowerState.POWER_SAVE)
            return
        
        if self.battery.is_low() and not self.battery.is_charging():
            if self._state == PowerState.PERFORMANCE:
                self.set_state(PowerState.BALANCED)
            return
        
        # Check idle
        if idle_time > self.config.deep_sleep_timeout_seconds:
            if self._state != PowerState.SLEEP:
                self.set_state(PowerState.SLEEP)
        elif idle_time > self.config.idle_timeout_seconds:
            if self._state not in (PowerState.IDLE, PowerState.SLEEP, PowerState.POWER_SAVE):
                self.set_state(PowerState.IDLE)


# Raspberry Pi specific utilities
class RaspberryPiPower:
    """Raspberry Pi specific power management."""
    
    VCGENCMD = "/usr/bin/vcgencmd"
    
    @staticmethod
    def get_throttled_state() -> dict[str, bool]:
        """Get Pi throttling status."""
        result = {
            "under_voltage": False,
            "arm_freq_capped": False,
            "throttled": False,
            "soft_temp_limit": False,
        }
        
        try:
            import subprocess
            output = subprocess.check_output(
                [RaspberryPiPower.VCGENCMD, "get_throttled"],
                timeout=5
            ).decode()
            
            # Parse throttled=0x...
            if "=" in output:
                hex_value = int(output.split("=")[1].strip(), 16)
                result["under_voltage"] = bool(hex_value & 0x1)
                result["arm_freq_capped"] = bool(hex_value & 0x2)
                result["throttled"] = bool(hex_value & 0x4)
                result["soft_temp_limit"] = bool(hex_value & 0x8)
        except (subprocess.SubprocessError, FileNotFoundError, ValueError, OSError):
            pass
        
        return result
    
    @staticmethod
    def get_gpu_temperature() -> float:
        """Get GPU temperature on Pi."""
        try:
            import subprocess
            output = subprocess.check_output(
                [RaspberryPiPower.VCGENCMD, "measure_temp"],
                timeout=5
            ).decode()
            
            # Parse temp=XX.X'C
            if "temp=" in output:
                temp_str = output.split("=")[1].split("'")[0]
                return float(temp_str)
        except (subprocess.SubprocessError, FileNotFoundError, ValueError, OSError):
            pass
        return 0.0
    
    @staticmethod
    def set_gpu_frequency(freq_mhz: int) -> bool:
        """Set GPU frequency on Pi."""
        try:
            # This requires config.txt modification and reboot
            logger.info(f"GPU frequency change requires config.txt edit: gpu_freq={freq_mhz}")
            return False
        except Exception:
            return False


# Convenience functions
def get_power_manager() -> PowerManager:
    """Get global power manager."""
    global _power_manager
    if "_power_manager" not in globals():
        _power_manager = PowerManager()
    return _power_manager


def mark_activity():
    """Mark user activity."""
    get_power_manager().activity()


def set_power_mode(mode: str):
    """Set power mode by name."""
    manager = get_power_manager()
    modes = {
        "performance": PowerState.PERFORMANCE,
        "balanced": PowerState.BALANCED,
        "power_save": PowerState.POWER_SAVE,
        "powersave": PowerState.POWER_SAVE,
    }
    state = modes.get(mode.lower())
    if state:
        manager.set_state(state)
