"""
Edge Computing Support

Support for edge devices like Raspberry Pi.

Provides:
- CameraModule: Camera capture for edge devices
- GPIOController: GPIO pin control
- PowerManager: Power and thermal management
"""

from .camera_module import (
    CameraConfig,
    CameraModule,
    CameraType,
    CaptureResult,
    ImageFormat,
    MotionDetector,
)
from .gpio_control import (
    EdgeDetect,
    GPIOController,
    GPIOEvent,
    PinConfig,
    PinMode,
    PullUpDown,
)
from .power_management import (
    BatteryMonitor,
    CPUGovernor,
    PowerConfig,
    PowerManager,
    PowerMetrics,
    PowerState,
    RaspberryPiPower,
    ThermalManager,
    ThermalState,
)

__all__ = [
    # Camera
    "CameraModule",
    "CameraType",
    "CameraConfig",
    "CaptureResult",
    "ImageFormat",
    "MotionDetector",
    # GPIO
    "GPIOController",
    "GPIOEvent",
    "PinConfig",
    "PinMode",
    "PullUpDown",
    "EdgeDetect",
    # Power
    "PowerManager",
    "PowerState",
    "PowerConfig",
    "PowerMetrics",
    "ThermalState",
    "ThermalManager",
    "BatteryMonitor",
    "CPUGovernor",
    "RaspberryPiPower",
]
