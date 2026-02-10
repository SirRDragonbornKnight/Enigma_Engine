"""
Battery Management for Enigma AI Engine

Smart battery management for robots.

Features:
- Battery monitoring
- Charge estimation
- Auto-dock on low battery
- Power consumption prediction
- Charging scheduling

Usage:
    from enigma_engine.tools.battery_manager import BatteryManager, get_battery
    
    battery = get_battery()
    
    # Monitor battery
    battery.update(voltage=12.4, current=0.5)
    
    # Get status
    status = battery.get_status()
    print(f"Battery: {status.percentage}%")
    
    # Auto-dock
    if battery.should_dock():
        robot.return_to_dock()
"""

import logging
import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BatteryState(Enum):
    """Battery state."""
    UNKNOWN = "unknown"
    CHARGING = "charging"
    DISCHARGING = "discharging"
    FULL = "full"
    NOT_CHARGING = "not_charging"
    CRITICAL = "critical"


class ChargingStrategy(Enum):
    """Charging strategy."""
    IMMEDIATE = "immediate"  # Charge as soon as low
    SCHEDULED = "scheduled"  # Charge at specific times
    SMART = "smart"  # Balance tasks and charging


@dataclass
class BatteryConfig:
    """Battery configuration."""
    # Capacity
    capacity_mah: float = 5000.0
    
    # Voltage thresholds
    voltage_full: float = 12.6  # Volts
    voltage_empty: float = 10.5
    voltage_critical: float = 10.8
    
    # Percentage thresholds
    low_threshold: float = 20.0  # %
    critical_threshold: float = 10.0
    dock_threshold: float = 25.0  # Start returning to dock
    
    # Power
    nominal_voltage: float = 11.1  # Typical voltage
    
    # Charging
    charge_rate: float = 1.0  # C rate (1C = full charge in 1 hour)


@dataclass
class BatteryStatus:
    """Current battery status."""
    percentage: float = 100.0
    voltage: float = 12.6
    current: float = 0.0  # Positive = discharging
    temperature: float = 25.0
    state: BatteryState = BatteryState.UNKNOWN
    
    # Estimates
    time_remaining_minutes: float = 0.0
    time_to_full_minutes: float = 0.0
    
    # Health
    health_percentage: float = 100.0
    cycle_count: int = 0
    
    timestamp: float = 0.0


@dataclass
class PowerConsumer:
    """Power consumer device."""
    name: str
    power_watts: float  # Power consumption
    active: bool = True
    priority: int = 5  # Higher = more important (1-10)


class BatteryPredictor:
    """Predict battery consumption."""
    
    def __init__(self):
        self._history: List[Tuple[float, float, float]] = []  # time, voltage, current
        self._max_history = 1000
    
    def add_sample(self, voltage: float, current: float):
        """Add measurement sample."""
        self._history.append((time.time(), voltage, current))
        
        if len(self._history) > self._max_history:
            self._history.pop(0)
    
    def predict_remaining_time(
        self,
        current_capacity_mah: float,
        avg_current_ma: float
    ) -> float:
        """
        Predict remaining time in minutes.
        
        Returns:
            Minutes until empty
        """
        if avg_current_ma <= 0:
            return float('inf')
        
        hours = current_capacity_mah / avg_current_ma
        return hours * 60
    
    def get_average_current(self, window_seconds: float = 60.0) -> float:
        """Get average current over time window."""
        if not self._history:
            return 0.0
        
        now = time.time()
        samples = [
            current for t, _, current in self._history
            if now - t <= window_seconds
        ]
        
        return sum(samples) / len(samples) if samples else 0.0
    
    def get_consumption_rate(self) -> float:
        """Get consumption rate in %/hour."""
        if len(self._history) < 2:
            return 0.0
        
        # Use first and last samples
        t1, v1, _ = self._history[0]
        t2, v2, _ = self._history[-1]
        
        dt = (t2 - t1) / 3600  # Hours
        if dt < 0.01:
            return 0.0
        
        # Simple linear voltage-to-percentage
        # This is a rough estimate
        dv = v1 - v2  # Voltage drop
        dp = dv / 2.1 * 100  # ~2.1V range = 100%
        
        return dp / dt


class ChargingScheduler:
    """Schedule charging times."""
    
    def __init__(self):
        self._schedule: List[Tuple[int, int, int, int]] = []  # start_hour, start_min, end_hour, end_min
        self._enabled = False
    
    def enable(self):
        """Enable scheduled charging."""
        self._enabled = True
    
    def disable(self):
        """Disable scheduled charging."""
        self._enabled = False
    
    def add_window(
        self,
        start_hour: int,
        start_minute: int,
        end_hour: int,
        end_minute: int
    ):
        """Add charging time window."""
        self._schedule.append((start_hour, start_minute, end_hour, end_minute))
    
    def clear_schedule(self):
        """Clear all windows."""
        self._schedule.clear()
    
    def is_charging_time(self) -> bool:
        """Check if current time is in charging window."""
        if not self._enabled or not self._schedule:
            return True  # No restrictions
        
        now = time.localtime()
        current = now.tm_hour * 60 + now.tm_min
        
        for start_h, start_m, end_h, end_m in self._schedule:
            start = start_h * 60 + start_m
            end = end_h * 60 + end_m
            
            if start <= current <= end:
                return True
        
        return False
    
    def next_charging_window(self) -> Optional[Tuple[int, int]]:
        """Get next charging window start time."""
        if not self._schedule:
            return None
        
        now = time.localtime()
        current = now.tm_hour * 60 + now.tm_min
        
        # Find next window
        for start_h, start_m, end_h, end_m in self._schedule:
            start = start_h * 60 + start_m
            if start > current:
                return (start_h, start_m)
        
        # Wrap to first window tomorrow
        if self._schedule:
            return (self._schedule[0][0], self._schedule[0][1])
        
        return None


class BatteryManager:
    """Manage robot battery."""
    
    def __init__(self, config: Optional[BatteryConfig] = None):
        """
        Initialize battery manager.
        
        Args:
            config: Battery configuration
        """
        self._config = config or BatteryConfig()
        self._status = BatteryStatus()
        
        self._predictor = BatteryPredictor()
        self._scheduler = ChargingScheduler()
        
        # Power consumers
        self._consumers: Dict[str, PowerConsumer] = {}
        
        # Dock location
        self._dock_location: Optional[Tuple[float, float]] = None
        
        # Callbacks
        self._low_callbacks: List[Callable[[BatteryStatus], None]] = []
        self._critical_callbacks: List[Callable[[BatteryStatus], None]] = []
        self._state_callbacks: List[Callable[[BatteryState], None]] = []
        
        # Monitoring
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        logger.info("BatteryManager initialized")
    
    def update(
        self,
        voltage: float,
        current: float = 0.0,
        temperature: float = 25.0
    ):
        """
        Update battery readings.
        
        Args:
            voltage: Battery voltage
            current: Current (positive = discharging, negative = charging)
            temperature: Temperature in Celsius
        """
        old_state = self._status.state
        
        self._status.voltage = voltage
        self._status.current = current
        self._status.temperature = temperature
        self._status.timestamp = time.time()
        
        # Calculate percentage from voltage
        voltage_range = self._config.voltage_full - self._config.voltage_empty
        percentage = (voltage - self._config.voltage_empty) / voltage_range * 100
        self._status.percentage = max(0, min(100, percentage))
        
        # Determine state
        if current < -0.1:
            self._status.state = BatteryState.CHARGING
        elif current > 0.1:
            self._status.state = BatteryState.DISCHARGING
        else:
            if percentage >= 99:
                self._status.state = BatteryState.FULL
            else:
                self._status.state = BatteryState.NOT_CHARGING
        
        if percentage <= self._config.critical_threshold:
            self._status.state = BatteryState.CRITICAL
        
        # Add to predictor
        self._predictor.add_sample(voltage, current)
        
        # Calculate time remaining
        if current > 0:
            capacity_remaining = (self._status.percentage / 100) * self._config.capacity_mah
            self._status.time_remaining_minutes = self._predictor.predict_remaining_time(
                capacity_remaining, current * 1000
            )
        else:
            self._status.time_remaining_minutes = float('inf')
        
        # Calculate time to full
        if current < 0:
            capacity_needed = ((100 - self._status.percentage) / 100) * self._config.capacity_mah
            charge_current = abs(current) * 1000
            if charge_current > 0:
                self._status.time_to_full_minutes = (capacity_needed / charge_current) * 60
        else:
            self._status.time_to_full_minutes = 0
        
        # State change notifications
        if self._status.state != old_state:
            self._notify_state_change(self._status.state)
        
        # Low/critical notifications
        if self._status.percentage <= self._config.critical_threshold:
            self._notify_critical()
        elif self._status.percentage <= self._config.low_threshold:
            self._notify_low()
    
    def get_status(self) -> BatteryStatus:
        """Get current battery status."""
        return self._status
    
    def get_percentage(self) -> float:
        """Get battery percentage."""
        return self._status.percentage
    
    def is_charging(self) -> bool:
        """Check if battery is charging."""
        return self._status.state == BatteryState.CHARGING
    
    def is_low(self) -> bool:
        """Check if battery is low."""
        return self._status.percentage <= self._config.low_threshold
    
    def is_critical(self) -> bool:
        """Check if battery is critical."""
        return self._status.percentage <= self._config.critical_threshold
    
    def should_dock(self) -> bool:
        """Check if robot should return to dock."""
        return self._status.percentage <= self._config.dock_threshold
    
    def should_charge(self) -> bool:
        """Check if robot should charge (considers schedule)."""
        if self.is_critical():
            return True  # Always charge if critical
        
        if self.is_low() and self._scheduler.is_charging_time():
            return True
        
        return False
    
    def set_dock_location(self, x: float, y: float):
        """Set dock location."""
        self._dock_location = (x, y)
    
    def get_dock_location(self) -> Optional[Tuple[float, float]]:
        """Get dock location."""
        return self._dock_location
    
    # Power management
    def add_consumer(
        self,
        name: str,
        power_watts: float,
        priority: int = 5
    ):
        """Add power consumer."""
        self._consumers[name] = PowerConsumer(
            name=name,
            power_watts=power_watts,
            priority=priority
        )
    
    def set_consumer_active(self, name: str, active: bool):
        """Enable/disable power consumer."""
        if name in self._consumers:
            self._consumers[name].active = active
    
    def get_total_power(self) -> float:
        """Get total power consumption in watts."""
        return sum(
            c.power_watts for c in self._consumers.values()
            if c.active
        )
    
    def get_low_power_devices(self) -> List[str]:
        """Get devices that can be turned off to save power."""
        # Sort by priority (lowest first)
        consumers = sorted(
            self._consumers.values(),
            key=lambda c: c.priority
        )
        
        return [c.name for c in consumers if c.priority < 5 and c.active]
    
    # Scheduling
    def add_charging_window(
        self,
        start_hour: int,
        start_minute: int,
        end_hour: int,
        end_minute: int
    ):
        """Add scheduled charging window."""
        self._scheduler.add_window(start_hour, start_minute, end_hour, end_minute)
    
    def enable_scheduled_charging(self):
        """Enable scheduled charging."""
        self._scheduler.enable()
    
    def disable_scheduled_charging(self):
        """Disable scheduled charging."""
        self._scheduler.disable()
    
    # Callbacks
    def on_low_battery(self, callback: Callable[[BatteryStatus], None]):
        """Register low battery callback."""
        self._low_callbacks.append(callback)
    
    def on_critical_battery(self, callback: Callable[[BatteryStatus], None]):
        """Register critical battery callback."""
        self._critical_callbacks.append(callback)
    
    def on_state_change(self, callback: Callable[[BatteryState], None]):
        """Register state change callback."""
        self._state_callbacks.append(callback)
    
    def _notify_low(self):
        """Notify low battery callbacks."""
        for callback in self._low_callbacks:
            try:
                callback(self._status)
            except Exception as e:
                logger.error(f"Low battery callback error: {e}")
    
    def _notify_critical(self):
        """Notify critical battery callbacks."""
        for callback in self._critical_callbacks:
            try:
                callback(self._status)
            except Exception as e:
                logger.error(f"Critical battery callback error: {e}")
    
    def _notify_state_change(self, state: BatteryState):
        """Notify state change callbacks."""
        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    # Utility
    def estimate_range(
        self,
        power_per_meter: float = 1.0  # Watts per meter
    ) -> float:
        """
        Estimate remaining range in meters.
        
        Args:
            power_per_meter: Power consumption per meter traveled
            
        Returns:
            Estimated range in meters
        """
        if self._status.time_remaining_minutes <= 0:
            return 0.0
        
        total_power = self.get_total_power()
        if total_power <= 0:
            return float('inf')
        
        # Energy remaining (Wh)
        energy_wh = (
            self._status.percentage / 100 *
            self._config.capacity_mah / 1000 *
            self._config.nominal_voltage
        )
        
        # Range = energy / power per meter
        return energy_wh * 1000 / power_per_meter  # meters


# Global instance
_battery: Optional[BatteryManager] = None


def get_battery(config: Optional[BatteryConfig] = None) -> BatteryManager:
    """Get or create global battery manager."""
    global _battery
    if _battery is None:
        _battery = BatteryManager(config)
    return _battery
