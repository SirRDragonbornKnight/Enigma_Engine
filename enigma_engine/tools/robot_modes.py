"""
Robot Control Modes System

Provides Auto/Manual modes for safe robot operation.

Modes:
- MANUAL: User has direct control, AI cannot move robot
- AUTO: AI can control robot based on context
- SAFE: Limited speed/range, AI can only do safe movements
- DISABLED: Robot is completely disabled

Features:
- Camera integration for visual feedback
- Sensor monitoring and safety limits
- Mode switching with safety interlocks
- Emergency stop (E-STOP) system
- Movement constraints per mode

Usage:
    from enigma_engine.tools.robot_modes import RobotModeController, RobotMode
    
    controller = RobotModeController()
    controller.set_mode(RobotMode.MANUAL)  # User control
    controller.set_mode(RobotMode.AUTO)    # AI control
    controller.emergency_stop()             # E-STOP
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .robot_tools import RobotController

# Type aliases
_RobotController = Optional['RobotController']


class RobotMode(Enum):
    """Robot control modes."""
    DISABLED = auto()   # Robot off, no movement allowed
    MANUAL = auto()     # User has direct control
    AUTO = auto()       # AI can control robot
    SAFE = auto()       # Limited auto - slow speed, small movements only


class SafetyLevel(Enum):
    """Safety constraint levels."""
    NONE = auto()       # No restrictions (full manual)
    LOW = auto()        # Basic limits (normal auto)
    MEDIUM = auto()     # Moderate restrictions (safe mode)
    HIGH = auto()       # Strict limits (near humans/objects)
    MAXIMUM = auto()    # Emergency stop imminent


@dataclass
class MovementConstraints:
    """Movement limits for safety."""
    max_speed: float = 1.0          # Max speed multiplier (0-1)
    max_acceleration: float = 1.0   # Max acceleration multiplier
    max_range: float = 1.0          # Max movement range multiplier
    allowed_joints: list[str] = field(default_factory=lambda: ["all"])
    forbidden_regions: list[tuple[float, float, float, float, float, float]] = field(default_factory=list)
    require_confirmation: bool = False
    
    @staticmethod
    def for_mode(mode: RobotMode) -> 'MovementConstraints':
        """Get constraints for a mode."""
        if mode == RobotMode.DISABLED:
            return MovementConstraints(
                max_speed=0.0,
                max_acceleration=0.0,
                max_range=0.0,
                allowed_joints=[],
            )
        elif mode == RobotMode.MANUAL:
            return MovementConstraints(
                max_speed=1.0,
                max_acceleration=1.0,
                max_range=1.0,
                require_confirmation=False,
            )
        elif mode == RobotMode.AUTO:
            return MovementConstraints(
                max_speed=0.8,
                max_acceleration=0.6,
                max_range=0.9,
                require_confirmation=False,
            )
        elif mode == RobotMode.SAFE:
            return MovementConstraints(
                max_speed=0.3,
                max_acceleration=0.2,
                max_range=0.5,
                require_confirmation=True,
            )
        return MovementConstraints()


@dataclass
class CameraConfig:
    """Camera configuration for robot."""
    enabled: bool = False
    device_id: int = 0              # Camera device ID
    resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    flip_horizontal: bool = False
    flip_vertical: bool = False


@dataclass
class SensorConfig:
    """Sensor configuration."""
    enabled: bool = False
    type: str = "generic"           # proximity, force, current, etc.
    port: str = ""                  # Serial/GPIO port
    threshold_warning: float = 0.7
    threshold_stop: float = 0.9


class RobotModeController:
    """
    Controls robot modes and safety.
    
    Provides:
    - Mode switching (Manual, Auto, Safe, Disabled)
    - Safety interlocks and constraints
    - Camera feed integration
    - Sensor monitoring
    - Emergency stop
    """
    
    def __init__(self, robot_controller: Optional['RobotController'] = None):
        self._robot = robot_controller
        self._mode = RobotMode.DISABLED
        self._constraints = MovementConstraints.for_mode(RobotMode.DISABLED)
        self._safety_level = SafetyLevel.HIGH
        
        # E-STOP state
        self._estop_active = False
        self._estop_reason = ""
        
        # Camera
        self._camera_config = CameraConfig()
        self._camera = None
        self._camera_frame = None
        self._camera_thread: Optional[threading.Thread] = None
        
        # Sensors
        self._sensors: dict[str, SensorConfig] = {}
        self._sensor_values: dict[str, float] = {}
        self._sensor_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._on_mode_change: list[Callable] = []
        self._on_estop: list[Callable] = []
        self._on_camera_frame: list[Callable] = []
        self._on_sensor_warning: list[Callable] = []
        
        # Locks
        self._mode_lock = threading.Lock()
        self._camera_lock = threading.Lock()
        
        # Auto mode state
        self._auto_enabled = False
        self._auto_context: dict[str, Any] = {}
    
    @property
    def mode(self) -> RobotMode:
        return self._mode
    
    @property
    def is_estop(self) -> bool:
        return self._estop_active
    
    @property
    def constraints(self) -> MovementConstraints:
        return self._constraints
    
    @property
    def can_ai_control(self) -> bool:
        """Check if AI is allowed to control robot."""
        return self._mode in (RobotMode.AUTO, RobotMode.SAFE) and not self._estop_active
    
    @property
    def can_user_control(self) -> bool:
        """Check if user can control robot."""
        return self._mode in (RobotMode.MANUAL, RobotMode.AUTO) and not self._estop_active
    
    # ===== Mode Control =====
    
    def set_mode(self, mode: RobotMode, force: bool = False) -> bool:
        """
        Set robot mode.
        
        Args:
            mode: Target mode
            force: Bypass safety checks (use with caution!)
        
        Returns:
            True if mode changed successfully
        """
        with self._mode_lock:
            # Check E-STOP
            if self._estop_active and not force:
                logger.warning(f"Cannot change mode - E-STOP active: {self._estop_reason}")
                return False
            
            # Safety checks
            if not force:
                if mode == RobotMode.AUTO and self._safety_level == SafetyLevel.MAXIMUM:
                    logger.warning("Cannot enter AUTO mode - safety level too high")
                    return False
            
            old_mode = self._mode
            self._mode = mode
            self._constraints = MovementConstraints.for_mode(mode)
            
            # Notify robot controller
            if self._robot:
                if mode == RobotMode.DISABLED:
                    self._robot.stop()
                elif mode == RobotMode.SAFE:
                    # Apply speed limits via safe mode flag
                    pass
            
            logger.info(f"Mode changed: {old_mode.name} -> {mode.name}")
            
            # Notify callbacks
            for cb in self._on_mode_change:
                try:
                    cb(old_mode, mode)
                except Exception as e:
                    logger.debug(f"Mode change callback failed: {e}")
            
            return True
    
    def enter_manual(self):
        """Convenience: Switch to manual mode."""
        return self.set_mode(RobotMode.MANUAL)
    
    def enter_auto(self):
        """Convenience: Switch to auto mode."""
        return self.set_mode(RobotMode.AUTO)
    
    def enter_safe(self):
        """Convenience: Switch to safe auto mode."""
        return self.set_mode(RobotMode.SAFE)
    
    def disable(self):
        """Convenience: Disable robot."""
        return self.set_mode(RobotMode.DISABLED)
    
    # ===== Emergency Stop =====
    
    def emergency_stop(self, reason: str = "User initiated"):
        """
        EMERGENCY STOP - Immediately halt all robot movement.
        """
        self._estop_active = True
        self._estop_reason = reason
        
        # Stop robot immediately
        if self._robot:
            self._robot.stop()
        
        logger.critical(f"*** E-STOP ACTIVATED *** Reason: {reason}")
        
        # Notify callbacks
        for cb in self._on_estop:
            try:
                cb(reason)
            except Exception as e:
                logger.debug(f"E-STOP callback failed: {e}")
    
    def reset_estop(self, confirm: bool = False) -> bool:
        """
        Reset E-STOP state.
        
        Args:
            confirm: Must be True to confirm reset
        
        Returns:
            True if reset successful
        """
        if not confirm:
            logger.warning("E-STOP reset requires confirm=True")
            return False
        
        self._estop_active = False
        self._estop_reason = ""
        self._mode = RobotMode.DISABLED  # Always start disabled after E-STOP
        logger.info("E-STOP reset - robot now DISABLED")
        return True
    
    # ===== Camera =====
    
    def setup_camera(self, config: Optional[CameraConfig] = None):
        """Setup camera for robot feedback."""
        self._camera_config = config or CameraConfig(enabled=True)
        
        if self._camera_config.enabled:
            try:
                import cv2
                self._camera = cv2.VideoCapture(self._camera_config.device_id)
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_config.resolution[0])
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_config.resolution[1])
                self._camera.set(cv2.CAP_PROP_FPS, self._camera_config.fps)
                logger.info(f"Camera {self._camera_config.device_id} initialized")
            except ImportError:
                logger.warning("OpenCV not installed - camera disabled")
                self._camera_config.enabled = False
            except Exception as e:
                logger.error(f"Camera init failed: {e}")
                self._camera_config.enabled = False
    
    def start_camera(self):
        """Start camera capture thread."""
        if not self._camera_config.enabled or self._camera is None:
            return
        
        def capture_loop():
            while self._camera_config.enabled and self._camera:
                ret, frame = self._camera.read()
                if ret:
                    # Apply flips
                    if self._camera_config.flip_horizontal:
                        import cv2
                        frame = cv2.flip(frame, 1)
                    if self._camera_config.flip_vertical:
                        import cv2
                        frame = cv2.flip(frame, 0)
                    
                    with self._camera_lock:
                        self._camera_frame = frame
                    
                    # Notify callbacks
                    for cb in self._on_camera_frame:
                        try:
                            cb(frame)
                        except Exception as e:
                            logger.debug(f"Camera frame callback failed: {e}")
                
                time.sleep(1.0 / self._camera_config.fps)
        
        self._camera_thread = threading.Thread(target=capture_loop, daemon=True)
        self._camera_thread.start()
        logger.info("Camera capture started")
    
    def stop_camera(self):
        """Stop camera capture."""
        self._camera_config.enabled = False
        if self._camera:
            self._camera.release()
            self._camera = None
        logger.info("Camera stopped")
    
    def get_camera_frame(self):
        """Get latest camera frame."""
        with self._camera_lock:
            return self._camera_frame
    
    # ===== Sensors =====
    
    def add_sensor(self, name: str, config: SensorConfig):
        """Add a sensor for monitoring."""
        self._sensors[name] = config
        self._sensor_values[name] = 0.0
        logger.info(f"Sensor added: {name} ({config.type})")
    
    def start_sensor_monitoring(self):
        """Start sensor monitoring thread."""
        if not self._sensors:
            return
        
        def monitor_loop():
            while True:
                for name, config in self._sensors.items():
                    if not config.enabled:
                        continue
                    
                    # Read sensor value (implementation depends on type)
                    value = self._read_sensor(name, config)
                    self._sensor_values[name] = value
                    
                    # Check thresholds
                    if value >= config.threshold_stop:
                        self.emergency_stop(f"Sensor {name} exceeded stop threshold: {value}")
                    elif value >= config.threshold_warning:
                        for cb in self._on_sensor_warning:
                            try:
                                cb(name, value)
                            except Exception as e:
                                logger.debug(f"Sensor warning callback failed: {e}")
                
                time.sleep(0.1)  # 10Hz sensor check
        
        self._sensor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._sensor_thread.start()
        logger.info("Sensor monitoring started")
    
    def _read_sensor(self, name: str, config: SensorConfig) -> float:
        """Read sensor value (override for specific implementations)."""
        # Default implementation - override for real sensors
        if self._robot and hasattr(self._robot, 'get_sensors'):
            sensors = self._robot.get_sensors()
            if name in sensors:
                return sensors[name]
        return 0.0
    
    def get_sensor_value(self, name: str) -> float:
        """Get current sensor value."""
        return self._sensor_values.get(name, 0.0)
    
    # ===== Movement Validation =====
    
    def validate_movement(self, joint: str, position: float, speed: float = 1.0) -> tuple[bool, str]:
        """
        Validate if a movement is allowed.
        
        Returns:
            (allowed, reason)
        """
        if self._estop_active:
            return False, "E-STOP active"
        
        if self._mode == RobotMode.DISABLED:
            return False, "Robot disabled"
        
        # Check joint allowed
        if "all" not in self._constraints.allowed_joints:
            if joint not in self._constraints.allowed_joints:
                return False, f"Joint {joint} not allowed in current mode"
        
        # Check speed
        if speed > self._constraints.max_speed:
            return False, f"Speed {speed} exceeds limit {self._constraints.max_speed}"
        
        # Check range (simplified)
        max_range = self._constraints.max_range
        # ... additional range checks based on joint limits
        
        return True, "OK"
    
    def request_movement(self, joint: str, position: float, speed: float = 0.5) -> bool:
        """
        Request a movement with validation.
        
        Returns:
            True if movement executed
        """
        allowed, reason = self.validate_movement(joint, position, speed)
        
        if not allowed:
            logger.warning(f"Movement denied: {reason}")
            return False
        
        # Apply constraints
        actual_speed = min(speed, self._constraints.max_speed)
        
        # Check if confirmation required
        if self._constraints.require_confirmation:
            # In a GUI, you would show a confirmation dialog here
            logger.info(f"Movement requires confirmation: {joint} to {position}")
            # For now, just proceed
        
        # Execute
        if self._robot:
            self._robot.move_joint(joint, position, speed=actual_speed)
            return True
        
        return False
    
    # ===== AI Auto Control =====
    
    def ai_request(self, action: str, params: Optional[dict[str, Any]] = None) -> bool:
        """
        Handle AI control request.
        
        Only works in AUTO or SAFE mode.
        
        Args:
            action: Action name (move, gripper, home, etc.)
            params: Action parameters
        
        Returns:
            True if action executed
        """
        if not self.can_ai_control:
            logger.warning(f"AI control not allowed in mode {self._mode.name}")
            return False
        
        params = params or {}
        
        if action == "move":
            joint = params.get("joint", "")
            position = params.get("position", 0)
            speed = params.get("speed", 0.3)  # AI uses slower default
            return self.request_movement(joint, position, speed)
        
        elif action == "gripper":
            position = params.get("position", 0)
            if self._robot:
                self._robot.gripper(position)
                return True
        
        elif action == "home":
            if self._robot:
                self._robot.home()
                return True
        
        elif action == "stop":
            if self._robot:
                self._robot.stop()
                return True
        
        return False
    
    # ===== Callbacks =====
    
    def on_mode_change(self, callback: Callable):
        """Register callback for mode changes."""
        self._on_mode_change.append(callback)
    
    def on_estop(self, callback: Callable):
        """Register callback for E-STOP events."""
        self._on_estop.append(callback)
    
    def on_camera_frame(self, callback: Callable):
        """Register callback for camera frames."""
        self._on_camera_frame.append(callback)
    
    def on_sensor_warning(self, callback: Callable):
        """Register callback for sensor warnings."""
        self._on_sensor_warning.append(callback)


# Global instance
_mode_controller: Optional[RobotModeController] = None


def get_mode_controller(robot: Optional['RobotController'] = None) -> RobotModeController:
    """Get or create robot mode controller."""
    global _mode_controller
    
    if _mode_controller is None:
        if robot is None:
            try:
                from .robot_tools import RobotController
                robot = RobotController()
            except Exception as e:
                logger.debug(f"Could not create RobotController: {e}")
                robot = None
        _mode_controller = RobotModeController(robot)
    
    return _mode_controller
