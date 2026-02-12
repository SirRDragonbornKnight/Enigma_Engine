"""
Robot Control Tools for Forge

Allows AI to control physical robots via various interfaces:
  - Serial/USB (Arduino, microcontrollers)
  - GPIO (Raspberry Pi pins)
  - Network (WiFi robots, ESP32, HTTP APIs)
  - ROS (Robot Operating System)

This module provides INTERFACES - connect your actual hardware by
implementing the RobotInterface class for your specific robot.

USAGE:
    from enigma_engine.tools.robot_tools import RobotController, get_robot
    
    # Register your robot
    robot = get_robot()
    robot.register_interface("my_arm", SerialRobotInterface("/dev/ttyUSB0"))
    
    # Control it
    robot.move_joint("my_arm", "shoulder", 45)
    robot.gripper("my_arm", "close")
    
    # Or via tools
    from enigma_engine.tools import execute_tool
    execute_tool("robot_move", robot="my_arm", joint="shoulder", angle=45)

IMPLEMENTING YOUR OWN ROBOT:
    class MyRobot(RobotInterface):
        def connect(self):
            # Connect to your hardware
            pass
        
        def move_joint(self, joint, angle):
            # Send command to move joint
            pass
        
        def disconnect(self):
            # Clean up
            pass
"""

import json
import logging
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RobotState(Enum):
    """Robot connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class JointInfo:
    """Information about a robot joint."""
    name: str
    min_angle: float = -180
    max_angle: float = 180
    current_angle: float = 0
    speed: float = 1.0  # 0-1 normalized


@dataclass
class RobotInfo:
    """Information about a robot."""
    name: str
    type: str  # "arm", "mobile", "humanoid", etc.
    joints: dict[str, JointInfo]
    capabilities: list[str]  # ["move", "grip", "speak", "see"]
    state: RobotState = RobotState.DISCONNECTED


class RobotInterface(ABC):
    """
    Abstract base class for robot interfaces.
    
    Implement this for your specific robot hardware.
    """
    
    def __init__(self, name: str = "robot"):
        self.name = name
        self.state = RobotState.DISCONNECTED
        self.info: Optional[RobotInfo] = None
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the robot. Return True if successful."""
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the robot."""
    
    @abstractmethod
    def move_joint(self, joint: str, angle: float, speed: float = 1.0) -> bool:
        """Move a specific joint to an angle."""
    
    def get_joint_position(self, joint: str) -> Optional[float]:
        """Get current position of a joint."""
        return None
    
    def home(self) -> bool:
        """Move all joints to home position."""
        return False
    
    def stop(self) -> bool:
        """Emergency stop all movement."""
        return False
    
    def gripper(self, action: str) -> bool:
        """Control gripper: 'open', 'close', or percentage 0-100."""
        return False
    
    def move(self, x: float = 0, y: float = 0, z: float = 0) -> bool:
        """Move in 3D space (for mobile robots or arms)."""
        return False
    
    def rotate(self, angle: float) -> bool:
        """Rotate the robot (for mobile bases)."""
        return False
    
    def get_sensors(self) -> dict[str, Any]:
        """Read sensor values."""
        return {}


# === Example Implementations ===

class SerialRobotInterface(RobotInterface):
    """
    Robot interface via Serial/USB connection.
    Works with Arduino, ESP32, etc.
    """
    
    def __init__(self, port: str, baudrate: int = 9600, name: str = "serial_robot"):
        super().__init__(name)
        self.port = port
        self.baudrate = baudrate
        self._serial = None
    
    def connect(self) -> bool:
        try:
            import serial
            self._serial = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            self.state = RobotState.CONNECTED
            logger.info("Connected to %s on %s", self.name, self.port)
            return True
        except Exception as e:
            logger.error("Serial connection failed: %s", e)
            self.state = RobotState.ERROR
            return False
    
    def disconnect(self) -> bool:
        if self._serial:
            self._serial.close()
            self._serial = None
        self.state = RobotState.DISCONNECTED
        return True
    
    def move_joint(self, joint: str, angle: float, speed: float = 1.0) -> bool:
        if not self._serial:
            return False
        
        # Send command in format: "MOVE joint angle speed\n"
        cmd = f"MOVE {joint} {angle} {speed}\n"
        self._serial.write(cmd.encode())
        
        # Wait for acknowledgment
        response = self._serial.readline().decode().strip()
        return response == "OK"
    
    def gripper(self, action: str) -> bool:
        if not self._serial:
            return False
        
        cmd = f"GRIP {action}\n"
        self._serial.write(cmd.encode())
        response = self._serial.readline().decode().strip()
        return response == "OK"
    
    def send_raw(self, command: str) -> str:
        """Send raw command and get response."""
        if not self._serial:
            return ""
        
        self._serial.write(f"{command}\n".encode())
        return self._serial.readline().decode().strip()


class GPIORobotInterface(RobotInterface):
    """
    Robot interface via Raspberry Pi GPIO.
    Direct control of servos, motors, etc.
    """
    
    def __init__(self, name: str = "gpio_robot"):
        super().__init__(name)
        self._gpio_available = False
        self._servos: dict[str, int] = {}  # joint_name -> pin
    
    def connect(self) -> bool:
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            self._gpio_available = True
            self.state = RobotState.CONNECTED
            logger.info("GPIO robot '%s' connected", self.name)
            return True
        except Exception as e:
            logger.error("GPIO not available: %s", e)
            self.state = RobotState.ERROR
            return False
    
    def disconnect(self) -> bool:
        if self._gpio_available:
            import RPi.GPIO as GPIO
            GPIO.cleanup()
        self.state = RobotState.DISCONNECTED
        return True
    
    def add_servo(self, joint_name: str, pin: int):
        """Add a servo on a specific GPIO pin."""
        self._servos[joint_name] = pin
        if self._gpio_available:
            import RPi.GPIO as GPIO
            GPIO.setup(pin, GPIO.OUT)
    
    def move_joint(self, joint: str, angle: float, speed: float = 1.0) -> bool:
        if not self._gpio_available or joint not in self._servos:
            return False
        
        # Convert angle to PWM duty cycle (typical servo: 2.5-12.5% for 0-180 deg)
        duty = 2.5 + (angle / 180.0) * 10.0
        
        import RPi.GPIO as GPIO
        pin = self._servos[joint]
        pwm = GPIO.PWM(pin, 50)  # 50Hz for servos
        pwm.start(duty)
        time.sleep(0.5)  # Wait for movement
        pwm.stop()
        
        return True


class NetworkRobotInterface(RobotInterface):
    """
    Robot interface via HTTP/WebSocket.
    Works with WiFi robots, ESP32 web servers, etc.
    """
    
    def __init__(self, url: str, name: str = "network_robot"):
        super().__init__(name)
        self.url = url.rstrip('/')
    
    def connect(self) -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.url}/status")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                if data.get("ok"):
                    self.state = RobotState.CONNECTED
                    logger.info("Connected to network robot at %s", self.url)
                    return True
        except Exception as e:
            logger.error("Network robot connection failed: %s", e)
        
        self.state = RobotState.ERROR
        return False
    
    def disconnect(self) -> bool:
        self.state = RobotState.DISCONNECTED
        return True
    
    def move_joint(self, joint: str, angle: float, speed: float = 1.0) -> bool:
        try:
            import urllib.request
            data = json.dumps({"joint": joint, "angle": angle, "speed": speed}).encode()
            req = urllib.request.Request(
                f"{self.url}/move",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                return result.get("success", False)
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, socket.timeout, OSError):
            return False
    
    def gripper(self, action: str) -> bool:
        try:
            import urllib.request
            data = json.dumps({"action": action}).encode()
            req = urllib.request.Request(
                f"{self.url}/gripper",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                return result.get("success", False)
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, socket.timeout, OSError):
            return False


class SimulatedRobotInterface(RobotInterface):
    """
    Simulated robot for testing without hardware.
    Logs all commands and maintains virtual state.
    """
    
    def __init__(self, name: str = "sim_robot", joints: list[str] = None):
        super().__init__(name)
        self.joints = joints or ["base", "shoulder", "elbow", "wrist", "gripper"]
        self._positions: dict[str, float] = {j: 0.0 for j in self.joints}
        self._gripper_state = "open"
    
    def connect(self) -> bool:
        self.state = RobotState.CONNECTED
        logger.info("Robot '%s' connected (simulated)", self.name)
        logger.info("Joints: %s", self.joints)
        return True
    
    def disconnect(self) -> bool:
        self.state = RobotState.DISCONNECTED
        logger.info("Robot '%s' disconnected", self.name)
        return True
    
    def move_joint(self, joint: str, angle: float, speed: float = 1.0) -> bool:
        if joint not in self._positions:
            logger.warning("Unknown joint: %s", joint)
            return False
        
        old_pos = self._positions[joint]
        self._positions[joint] = angle
        logger.debug("%s: %s deg -> %s deg (speed=%s)", joint, old_pos, angle, speed)
        return True
    
    def get_joint_position(self, joint: str) -> Optional[float]:
        return self._positions.get(joint)
    
    def gripper(self, action: str) -> bool:
        old_state = self._gripper_state
        self._gripper_state = action
        logger.debug("Gripper: %s -> %s", old_state, action)
        return True
    
    def home(self) -> bool:
        logger.info("Homing all joints...")
        for joint in self._positions:
            self._positions[joint] = 0.0
        return True
    
    def get_sensors(self) -> dict[str, Any]:
        return {
            "positions": self._positions.copy(),
            "gripper": self._gripper_state,
            "simulated": True,
        }


# === Robot Controller (Main Interface) ===

class RobotController:
    """
    Main robot controller - manages multiple robot interfaces.
    """
    
    def __init__(self):
        self._robots: dict[str, RobotInterface] = {}
        self._default_robot: Optional[str] = None
    
    def register(self, name: str, interface: RobotInterface) -> bool:
        """Register a robot interface."""
        self._robots[name] = interface
        if self._default_robot is None:
            self._default_robot = name
        return True
    
    def connect(self, name: str = None) -> bool:
        """Connect to a robot."""
        robot = self._get_robot(name)
        if robot:
            return robot.connect()
        return False
    
    def disconnect(self, name: str = None) -> bool:
        """Disconnect from a robot."""
        robot = self._get_robot(name)
        if robot:
            return robot.disconnect()
        return False
    
    def _get_robot(self, name: str = None) -> Optional[RobotInterface]:
        """Get robot by name or default."""
        name = name or self._default_robot
        return self._robots.get(name)
    
    def move_joint(self, joint: str, angle: float, robot: str = None, speed: float = 1.0) -> bool:
        """Move a joint on a robot."""
        r = self._get_robot(robot)
        if r and r.state == RobotState.CONNECTED:
            return r.move_joint(joint, angle, speed)
        return False
    
    def gripper(self, action: str, robot: str = None) -> bool:
        """Control gripper."""
        r = self._get_robot(robot)
        if r and r.state == RobotState.CONNECTED:
            return r.gripper(action)
        return False
    
    def home(self, robot: str = None) -> bool:
        """Home a robot."""
        r = self._get_robot(robot)
        if r and r.state == RobotState.CONNECTED:
            return r.home()
        return False
    
    def stop(self, robot: str = None) -> bool:
        """Emergency stop."""
        r = self._get_robot(robot)
        if r:
            return r.stop()
        return False
    
    def list_robots(self) -> dict[str, str]:
        """List registered robots and their states."""
        return {name: robot.state.value for name, robot in self._robots.items()}
    
    def get_sensors(self, robot: str = None) -> dict[str, Any]:
        """Get sensor data from robot."""
        r = self._get_robot(robot)
        if r:
            return r.get_sensors()
        return {}


# Global instance
_robot_controller: Optional[RobotController] = None


def get_robot() -> RobotController:
    """Get or create global robot controller."""
    global _robot_controller
    if _robot_controller is None:
        _robot_controller = RobotController()
    return _robot_controller


# === Tool Classes for AI ===

from .tool_registry import Tool, RichParameter


class RobotMoveTool(Tool):
    """Tool for AI to move robot joints."""
    
    name = "robot_move"
    description = "Move a robot joint to a specific angle"
    parameters = {
        "joint": "Joint name (e.g., 'shoulder', 'elbow') - required",
        "angle": "Target angle in degrees - required",
        "robot": "Robot name (optional, uses default)",
        "speed": "Speed 0-1 (default 1.0)",
    }
    category = "robot"
    rich_parameters = [
        RichParameter(
            name="joint",
            type="string",
            description="Joint name (e.g., 'shoulder', 'elbow')",
            required=True,
        ),
        RichParameter(
            name="angle",
            type="number",
            description="Target angle in degrees",
            required=True,
            min_value=-180,
            max_value=180,
        ),
        RichParameter(
            name="robot",
            type="string",
            description="Robot name (uses default if not specified)",
            required=False,
        ),
        RichParameter(
            name="speed",
            type="number",
            description="Speed (0-1)",
            required=False,
            default=1.0,
            min_value=0,
            max_value=1,
        ),
    ]
    examples = [
        "robot_move(joint='shoulder', angle=45)",
        "robot_move(joint='elbow', angle=90, speed=0.5)",
    ]
    
    def execute(self, joint: str = "", angle: float = 0, robot: str = None, speed: float = 1.0, **kwargs) -> dict[str, Any]:
        controller = get_robot()
        success = controller.move_joint(joint, angle, robot, speed)
        return {
            "success": success,
            "joint": joint,
            "angle": angle,
            "robot": robot or "default",
        }


class RobotGripperTool(Tool):
    """Tool for AI to control robot gripper."""
    
    name = "robot_gripper"
    description = "Control robot gripper (open/close)"
    parameters = {
        "action": "'open', 'close', or percentage 0-100 - required",
        "robot": "Robot name (optional)",
    }
    category = "robot"
    rich_parameters = [
        RichParameter(
            name="action",
            type="string",
            description="Gripper action or percentage",
            required=True,
            enum=["open", "close"],
        ),
        RichParameter(
            name="robot",
            type="string",
            description="Robot name (uses default if not specified)",
            required=False,
        ),
    ]
    examples = [
        "robot_gripper(action='close') - Close gripper",
        "robot_gripper(action='open')",
    ]
    
    def execute(self, action: str = "open", robot: str = None, **kwargs) -> dict[str, Any]:
        controller = get_robot()
        success = controller.gripper(action, robot)
        return {"success": success, "action": action}


class RobotStatusTool(Tool):
    """Tool to get robot status."""
    
    name = "robot_status"
    description = "Get status and sensor data from robot"
    parameters = {
        "robot": "Robot name (optional)",
    }
    category = "robot"
    rich_parameters = [
        RichParameter(
            name="robot",
            type="string",
            description="Robot name (all robots if not specified)",
            required=False,
        ),
    ]
    examples = ["robot_status() - Get all robot status"]
    
    def execute(self, robot: str = None, **kwargs) -> dict[str, Any]:
        controller = get_robot()
        robots = controller.list_robots()
        sensors = controller.get_sensors(robot)
        return {
            "success": True,
            "robots": robots,
            "sensors": sensors,
        }


class RobotHomeTool(Tool):
    """Tool to home robot."""
    
    name = "robot_home"
    description = "Move robot to home position"
    parameters = {
        "robot": "Robot name (optional)",
    }
    category = "robot"
    rich_parameters = [
        RichParameter(
            name="robot",
            type="string",
            description="Robot name (uses default if not specified)",
            required=False,
        ),
    ]
    examples = ["robot_home() - Home the default robot"]
    
    def execute(self, robot: str = None, **kwargs) -> dict[str, Any]:
        controller = get_robot()
        success = controller.home(robot)
        return {"success": success, "action": "home"}
