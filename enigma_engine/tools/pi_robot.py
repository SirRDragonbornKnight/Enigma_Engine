"""
================================================================================
Pi Robot Controller - Optimized robot control for Raspberry Pi.
================================================================================

Lightweight robot control system designed for Raspberry Pi constraints:
- Minimal memory usage
- Efficient GPIO control
- Network-based offloading of heavy computation
- Real-time motor control
- Sensor integration

USAGE:
    from enigma_engine.tools.pi_robot import PiRobotController
    
    robot = PiRobotController()
    robot.connect_to_server("192.168.1.100:5000")  # Optional: offload AI
    
    # Direct control
    robot.move_forward(speed=0.5)
    robot.turn(angle=45)
    robot.stop()
    
    # AI-assisted (offloads to server)
    robot.execute_command("go to the red ball")

ARCHITECTURE:
    [Raspberry Pi]                  [Desktop PC]
    ┌─────────────────┐            ┌─────────────────┐
    │ PiRobotController│◀──network──│ AI Server       │
    │  - Motor control │            │  - Vision AI    │
    │  - Sensor read   │──camera───▶│  - Navigation   │
    │  - GPIO          │            │  - Planning     │
    └─────────────────┘            └─────────────────┘
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class MotorType(Enum):
    """Supported motor types."""
    DC = auto()         # DC motor with PWM
    SERVO = auto()      # Servo motor
    STEPPER = auto()    # Stepper motor


class SensorType(Enum):
    """Supported sensor types."""
    ULTRASONIC = auto()   # Distance sensor
    IR = auto()           # Infrared sensor
    CAMERA = auto()       # Camera (Pi Camera or USB)
    IMU = auto()          # Accelerometer/Gyroscope
    ENCODER = auto()      # Wheel encoder
    TEMPERATURE = auto()  # Temperature sensor
    LIGHT = auto()        # Light sensor


@dataclass
class MotorConfig:
    """Configuration for a motor."""
    name: str
    motor_type: MotorType = MotorType.DC
    gpio_pins: list[int] = field(default_factory=list)
    pwm_frequency: int = 1000
    min_duty: int = 0
    max_duty: int = 100
    inverted: bool = False


@dataclass
class SensorConfig:
    """Configuration for a sensor."""
    name: str
    sensor_type: SensorType
    gpio_pins: list[int] = field(default_factory=list)
    i2c_address: int = 0
    sample_rate_hz: float = 10.0


@dataclass
class PiRobotConfig:
    """Full robot configuration."""
    name: str = "pi_robot"
    
    # Motor configurations
    motors: dict[str, MotorConfig] = field(default_factory=dict)
    
    # Sensor configurations
    sensors: dict[str, SensorConfig] = field(default_factory=dict)
    
    # Network settings
    server_url: str = ""
    offload_vision: bool = True
    offload_navigation: bool = True
    
    # Safety settings
    max_speed: float = 1.0
    collision_distance_cm: float = 20.0
    emergency_stop_enabled: bool = True
    
    # Resource limits (for Pi)
    max_cpu_percent: float = 80.0
    max_memory_mb: int = 256


class PiRobotController:
    """
    Lightweight robot controller for Raspberry Pi.
    
    Features:
    - Direct motor control via GPIO
    - Sensor reading
    - Network offloading for AI tasks
    - Safety systems
    - Battery management
    """
    
    def __init__(self, config: PiRobotConfig = None):
        self.config = config or PiRobotConfig()
        
        # GPIO state
        self._gpio_initialized = False
        self._gpio = None  # RPi.GPIO module
        self._pwm_objects: dict[str, Any] = {}
        
        # Motor state
        self._motor_speeds: dict[str, float] = {}
        
        # Sensor state
        self._sensor_values: dict[str, Any] = {}
        self._sensor_threads: dict[str, threading.Thread] = {}
        
        # Network
        self._server_connected = False
        self._server_url = self.config.server_url
        
        # Safety
        self._emergency_stop = False
        self._collision_detected = False
        
        # Control loop
        self._running = False
        self._control_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Callbacks
        self._on_collision: list[Callable[[], None]] = []
        self._on_command_complete: list[Callable[[str, bool], None]] = []
        
        # Initialize
        self._init_gpio()
        self._init_default_config()
    
    def _init_default_config(self):
        """Set up default motor/sensor configuration for common robot."""
        if not self.config.motors:
            # Default: 2-wheel differential drive
            self.config.motors = {
                "left": MotorConfig(
                    name="left",
                    motor_type=MotorType.DC,
                    gpio_pins=[17, 18],  # Direction, PWM
                    pwm_frequency=1000,
                ),
                "right": MotorConfig(
                    name="right",
                    motor_type=MotorType.DC,
                    gpio_pins=[22, 23],  # Direction, PWM
                    pwm_frequency=1000,
                ),
            }
        
        if not self.config.sensors:
            # Default sensors
            self.config.sensors = {
                "front_distance": SensorConfig(
                    name="front_distance",
                    sensor_type=SensorType.ULTRASONIC,
                    gpio_pins=[24, 25],  # Trigger, Echo
                    sample_rate_hz=10.0,
                ),
            }
    
    def _init_gpio(self):
        """Initialize GPIO (with graceful fallback)."""
        try:
            import RPi.GPIO as GPIO  # type: ignore
            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            self._gpio_initialized = True
            logger.info("GPIO initialized successfully")
        except ImportError:
            logger.warning("RPi.GPIO not available - running in simulation mode")
            self._gpio = None
            self._gpio_initialized = False
        except Exception as e:
            logger.error(f"GPIO initialization failed: {e}")
            self._gpio = None
            self._gpio_initialized = False
    
    def _setup_motor(self, motor_config: MotorConfig):
        """Set up a motor's GPIO pins."""
        if not self._gpio_initialized:
            return
        
        GPIO = self._gpio
        
        if motor_config.motor_type == MotorType.DC:
            # DC motor: direction pin + PWM pin
            if len(motor_config.gpio_pins) >= 2:
                dir_pin, pwm_pin = motor_config.gpio_pins[:2]
                GPIO.setup(dir_pin, GPIO.OUT)
                GPIO.setup(pwm_pin, GPIO.OUT)
                
                pwm = GPIO.PWM(pwm_pin, motor_config.pwm_frequency)
                pwm.start(0)
                self._pwm_objects[motor_config.name] = pwm
        
        elif motor_config.motor_type == MotorType.SERVO:
            # Servo: single PWM pin
            if motor_config.gpio_pins:
                pin = motor_config.gpio_pins[0]
                GPIO.setup(pin, GPIO.OUT)
                pwm = GPIO.PWM(pin, 50)  # 50Hz for servo
                pwm.start(0)
                self._pwm_objects[motor_config.name] = pwm
    
    def start(self):
        """Start the robot controller."""
        if self._running:
            return
        
        # Set up motors
        for name, motor in self.config.motors.items():
            self._setup_motor(motor)
            self._motor_speeds[name] = 0.0
        
        # Start sensor threads
        for name, sensor in self.config.sensors.items():
            self._start_sensor(name, sensor)
        
        # Start control loop
        self._running = True
        self._control_thread = threading.Thread(
            target=self._control_loop,
            daemon=True,
            name="PiRobotControl",
        )
        self._control_thread.start()
        
        logger.info("Pi robot controller started")
    
    def stop(self):
        """Stop the robot controller."""
        self._running = False
        
        # Stop all motors
        self.emergency_stop()
        
        # Stop control thread
        if self._control_thread:
            self._control_thread.join(timeout=2.0)
        
        # Stop sensor threads
        for thread in self._sensor_threads.values():
            thread.join(timeout=1.0)
        
        # Clean up GPIO
        if self._gpio_initialized:
            for pwm in self._pwm_objects.values():
                pwm.stop()
            self._gpio.cleanup()
        
        logger.info("Pi robot controller stopped")
    
    def _control_loop(self):
        """Main control loop."""
        while self._running:
            try:
                # Check for collisions
                self._check_collisions()
                
                # Apply motor speeds (with safety limits)
                self._apply_motor_speeds()
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
            
            time.sleep(0.02)  # 50Hz control loop
    
    def _check_collisions(self):
        """Check for potential collisions."""
        if not self.config.emergency_stop_enabled:
            return
        
        front_distance = self._sensor_values.get("front_distance", 999)
        
        if front_distance < self.config.collision_distance_cm:
            if not self._collision_detected:
                self._collision_detected = True
                logger.warning(f"Collision detected! Distance: {front_distance}cm")
                
                # Stop forward movement
                for name, speed in self._motor_speeds.items():
                    if speed > 0:
                        self._motor_speeds[name] = 0
                
                # Notify callbacks
                for callback in self._on_collision:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Collision callback error: {e}")
        else:
            self._collision_detected = False
    
    def _apply_motor_speeds(self):
        """Apply current motor speeds to hardware."""
        if self._emergency_stop:
            return
        
        with self._lock:
            for name, speed in self._motor_speeds.items():
                motor = self.config.motors.get(name)
                if not motor:
                    continue
                
                # Apply speed limit
                speed = max(-self.config.max_speed, min(self.config.max_speed, speed))
                
                if self._gpio_initialized and name in self._pwm_objects:
                    pwm = self._pwm_objects[name]
                    
                    # Set direction
                    if len(motor.gpio_pins) >= 2:
                        dir_pin = motor.gpio_pins[0]
                        direction = 1 if speed >= 0 else 0
                        if motor.inverted:
                            direction = 1 - direction
                        self._gpio.output(dir_pin, direction)
                    
                    # Set speed (PWM duty cycle)
                    duty = abs(speed) * 100
                    duty = max(motor.min_duty, min(motor.max_duty, duty))
                    pwm.ChangeDutyCycle(duty)
    
    def _start_sensor(self, name: str, config: SensorConfig):
        """Start a sensor reading thread."""
        def sensor_loop():
            interval = 1.0 / config.sample_rate_hz
            while self._running:
                try:
                    value = self._read_sensor(config)
                    self._sensor_values[name] = value
                except Exception as e:
                    logger.debug(f"Sensor {name} read error: {e}")
                time.sleep(interval)
        
        thread = threading.Thread(target=sensor_loop, daemon=True)
        thread.start()
        self._sensor_threads[name] = thread
    
    def _read_sensor(self, config: SensorConfig) -> Any:
        """Read a sensor value."""
        if not self._gpio_initialized:
            return 0  # Simulation mode
        
        if config.sensor_type == SensorType.ULTRASONIC:
            return self._read_ultrasonic(config)
        elif config.sensor_type == SensorType.IR:
            return self._read_ir(config)
        
        return 0
    
    def _read_ultrasonic(self, config: SensorConfig) -> float:
        """Read ultrasonic distance sensor."""
        if not self._gpio_initialized or len(config.gpio_pins) < 2:
            return 999.0
        
        GPIO = self._gpio
        trigger_pin, echo_pin = config.gpio_pins[:2]
        
        GPIO.setup(trigger_pin, GPIO.OUT)
        GPIO.setup(echo_pin, GPIO.IN)
        
        # Send trigger pulse
        GPIO.output(trigger_pin, True)
        time.sleep(0.00001)
        GPIO.output(trigger_pin, False)
        
        # Wait for echo
        start_time = time.time()
        timeout = start_time + 0.1
        
        while GPIO.input(echo_pin) == 0:
            if time.time() > timeout:
                return 999.0
            start_time = time.time()
        
        while GPIO.input(echo_pin) == 1:
            if time.time() > timeout:
                return 999.0
            end_time = time.time()
        
        # Calculate distance
        duration = end_time - start_time
        distance = (duration * 34300) / 2  # cm
        
        return round(distance, 1)
    
    def _read_ir(self, config: SensorConfig) -> int:
        """Read IR sensor (digital)."""
        if not self._gpio_initialized or not config.gpio_pins:
            return 0
        
        self._gpio.setup(config.gpio_pins[0], self._gpio.IN)
        return self._gpio.input(config.gpio_pins[0])
    
    # ==========================================================================
    # Public Control Methods
    # ==========================================================================
    
    def move_forward(self, speed: float = 0.5):
        """Move forward at given speed (0-1)."""
        if self._emergency_stop:
            return
        
        with self._lock:
            self._motor_speeds["left"] = speed
            self._motor_speeds["right"] = speed
    
    def move_backward(self, speed: float = 0.5):
        """Move backward at given speed (0-1)."""
        if self._emergency_stop:
            return
        
        with self._lock:
            self._motor_speeds["left"] = -speed
            self._motor_speeds["right"] = -speed
    
    def turn_left(self, speed: float = 0.3):
        """Turn left in place."""
        if self._emergency_stop:
            return
        
        with self._lock:
            self._motor_speeds["left"] = -speed
            self._motor_speeds["right"] = speed
    
    def turn_right(self, speed: float = 0.3):
        """Turn right in place."""
        if self._emergency_stop:
            return
        
        with self._lock:
            self._motor_speeds["left"] = speed
            self._motor_speeds["right"] = -speed
    
    def set_motor(self, name: str, speed: float):
        """Set individual motor speed."""
        if self._emergency_stop:
            return
        
        with self._lock:
            if name in self._motor_speeds:
                self._motor_speeds[name] = speed
    
    def halt(self):
        """Stop all motors (normal stop)."""
        with self._lock:
            for name in self._motor_speeds:
                self._motor_speeds[name] = 0
    
    def emergency_stop(self):
        """Emergency stop - immediately halt all motors."""
        self._emergency_stop = True
        
        with self._lock:
            for name in self._motor_speeds:
                self._motor_speeds[name] = 0
        
        # Immediately stop PWM
        for pwm in self._pwm_objects.values():
            pwm.ChangeDutyCycle(0)
        
        logger.warning("EMERGENCY STOP activated")
    
    def reset_emergency_stop(self):
        """Reset emergency stop flag."""
        self._emergency_stop = False
        self._collision_detected = False
        logger.info("Emergency stop reset")
    
    def get_sensor(self, name: str) -> Any:
        """Get sensor value by name."""
        return self._sensor_values.get(name)
    
    def get_all_sensors(self) -> dict[str, Any]:
        """Get all sensor values."""
        return self._sensor_values.copy()
    
    # ==========================================================================
    # Network Offloading
    # ==========================================================================
    
    def connect_to_server(self, server_url: str) -> bool:
        """
        Connect to AI server for offloading.
        
        Args:
            server_url: Server URL (e.g., "192.168.1.100:5000")
            
        Returns:
            True if connected
        """
        import urllib.request
        
        self._server_url = server_url
        if not server_url.startswith("http"):
            self._server_url = f"http://{server_url}"
        
        try:
            req = urllib.request.Request(f"{self._server_url}/health")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                if data.get("ok"):
                    self._server_connected = True
                    logger.info(f"Connected to AI server: {server_url}")
                    return True
        except Exception as e:
            logger.warning(f"Could not connect to server: {e}")
        
        self._server_connected = False
        return False
    
    def execute_command(self, command: str) -> str:
        """
        Execute a natural language command.
        
        If connected to server, offloads AI processing.
        Otherwise, uses basic pattern matching.
        
        Args:
            command: Natural language command (e.g., "go forward")
            
        Returns:
            Result message
        """
        if self._server_connected and self.config.offload_navigation:
            return self._execute_remote(command)
        else:
            return self._execute_local(command)
    
    def _execute_remote(self, command: str) -> str:
        """Execute command via remote AI server."""
        import urllib.request
        
        try:
            data = json.dumps({
                "command": command,
                "sensors": self._sensor_values,
                "robot_type": "differential_drive",
            }).encode()
            
            req = urllib.request.Request(
                f"{self._server_url}/robot/command",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                actions = result.get("actions", [])
                
                # Execute returned actions
                for action in actions:
                    self._execute_action(action)
                
                return result.get("message", "Command executed")
                
        except Exception as e:
            logger.error(f"Remote execution failed: {e}")
            return self._execute_local(command)
    
    def _execute_local(self, command: str) -> str:
        """Execute command using local pattern matching."""
        cmd_lower = command.lower()
        
        if any(w in cmd_lower for w in ["forward", "ahead", "go"]):
            self.move_forward()
            return "Moving forward"
        elif any(w in cmd_lower for w in ["back", "reverse", "retreat"]):
            self.move_backward()
            return "Moving backward"
        elif "left" in cmd_lower:
            self.turn_left()
            return "Turning left"
        elif "right" in cmd_lower:
            self.turn_right()
            return "Turning right"
        elif any(w in cmd_lower for w in ["stop", "halt", "freeze"]):
            self.halt()
            return "Stopped"
        else:
            return f"Unknown command: {command}"
    
    def _execute_action(self, action: dict[str, Any]):
        """Execute a single action from server."""
        action_type = action.get("type", "")
        
        if action_type == "move":
            speed = action.get("speed", 0.5)
            if action.get("direction") == "forward":
                self.move_forward(speed)
            elif action.get("direction") == "backward":
                self.move_backward(speed)
        elif action_type == "turn":
            angle = action.get("angle", 0)
            speed = action.get("speed", 0.3)
            if angle > 0:
                self.turn_right(speed)
            else:
                self.turn_left(speed)
        elif action_type == "stop":
            self.halt()
        elif action_type == "wait":
            duration = action.get("duration", 1.0)
            time.sleep(duration)
    
    # ==========================================================================
    # Callbacks
    # ==========================================================================
    
    def on_collision(self, callback: Callable[[], None]):
        """Register collision callback."""
        self._on_collision.append(callback)
    
    def on_command_complete(self, callback: Callable[[str, bool], None]):
        """Register command completion callback."""
        self._on_command_complete.append(callback)
    
    # ==========================================================================
    # Status
    # ==========================================================================
    
    def get_status(self) -> dict[str, Any]:
        """Get robot status."""
        return {
            "running": self._running,
            "emergency_stop": self._emergency_stop,
            "collision_detected": self._collision_detected,
            "server_connected": self._server_connected,
            "motor_speeds": self._motor_speeds.copy(),
            "sensors": self._sensor_values.copy(),
            "gpio_available": self._gpio_initialized,
        }


def create_pi_robot(config: PiRobotConfig = None) -> PiRobotController:
    """Create and start a Pi robot controller."""
    robot = PiRobotController(config)
    robot.start()
    return robot


__all__ = [
    'PiRobotController',
    'PiRobotConfig',
    'MotorConfig',
    'SensorConfig',
    'MotorType',
    'SensorType',
    'create_pi_robot',
]
