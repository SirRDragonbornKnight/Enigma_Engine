"""
Robot Platform Support - Common robot platforms integration

Supports:
- TurtleBot (ROS2)
- Universal Robots (URx arms)
- Boston Dynamics Spot
- DJI Drones
- Raspberry Pi PiCar/PiArm
- Arduino-based robots
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Optional imports
try:
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

try:
    import urx
    HAS_URX = True
except ImportError:
    HAS_URX = False

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


class RobotType(Enum):
    """Supported robot types"""
    TURTLEBOT = "turtlebot"
    TURTLEBOT3 = "turtlebot3"
    TURTLEBOT4 = "turtlebot4"
    UR3 = "ur3"
    UR5 = "ur5"
    UR10 = "ur10"
    UR3E = "ur3e"
    UR5E = "ur5e"
    UR10E = "ur10e"
    UR16E = "ur16e"
    SPOT = "spot"
    DJI_TELLO = "dji_tello"
    DJI_MAVIC = "dji_mavic"
    PICAR = "picar"
    PIARM = "piarm"
    ARDUINO = "arduino"
    CUSTOM = "custom"


class RobotState(Enum):
    """Robot operational states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    IDLE = "idle"
    MOVING = "moving"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RobotCapabilities:
    """Robot capabilities description"""
    can_move: bool = False
    can_rotate: bool = False
    has_arm: bool = False
    has_gripper: bool = False
    has_camera: bool = False
    has_lidar: bool = False
    has_imu: bool = False
    can_fly: bool = False
    degrees_of_freedom: int = 0
    max_speed: float = 0.0  # m/s
    max_payload: float = 0.0  # kg


@dataclass
class Position:
    """3D position"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Orientation:
    """Quaternion orientation"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class Pose6D:
    """6D pose (position + orientation)"""
    position: Position = field(default_factory=Position)
    orientation: Orientation = field(default_factory=Orientation)


@dataclass
class JointConfiguration:
    """Robot joint configuration"""
    joint_names: list[str] = field(default_factory=list)
    joint_positions: list[float] = field(default_factory=list)
    joint_velocities: list[float] = field(default_factory=list)


class RobotPlatform(ABC):
    """Abstract base class for robot platforms"""
    
    def __init__(self, robot_type: RobotType):
        self.robot_type = robot_type
        self.state = RobotState.DISCONNECTED
        self._callbacks: dict[str, list[Callable]] = {}
    
    @abstractmethod
    async def connect(self, **kwargs) -> bool:
        """Connect to the robot"""
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the robot"""
    
    @abstractmethod
    def get_capabilities(self) -> RobotCapabilities:
        """Get robot capabilities"""
    
    @abstractmethod
    async def get_pose(self) -> Pose6D:
        """Get current robot pose"""
    
    @abstractmethod
    async def emergency_stop(self) -> None:
        """Emergency stop the robot"""
    
    def add_callback(self, event: str, callback: Callable) -> None:
        """Add event callback"""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, data: Any = None) -> None:
        """Trigger callbacks for an event"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")


class TurtleBotPlatform(RobotPlatform):
    """TurtleBot platform support (ROS2)"""
    
    def __init__(self, version: int = 3):
        robot_type = {
            2: RobotType.TURTLEBOT,
            3: RobotType.TURTLEBOT3,
            4: RobotType.TURTLEBOT4
        }.get(version, RobotType.TURTLEBOT3)
        
        super().__init__(robot_type)
        self.version = version
        self.node = None
        self._cmd_vel_pub = None
        self._odom_sub = None
        self._scan_sub = None
        self._current_pose = Pose6D()
        self._current_scan = None
    
    async def connect(self, node_name: str = "forge_turtlebot") -> bool:
        """Connect to TurtleBot via ROS2"""
        if not HAS_ROS2:
            logger.error("ROS2 not available")
            return False
        
        try:
            self.state = RobotState.CONNECTING
            
            if not rclpy.ok():
                rclpy.init()
            
            self.node = rclpy.create_node(node_name)
            
            # Publishers
            self._cmd_vel_pub = self.node.create_publisher(
                Twist, '/cmd_vel', 10
            )
            
            # Subscribers
            self._odom_sub = self.node.create_subscription(
                Odometry, '/odom', self._odom_callback, 10
            )
            
            self._scan_sub = self.node.create_subscription(
                LaserScan, '/scan', self._scan_callback, 10
            )
            
            self.state = RobotState.IDLE
            logger.info(f"Connected to TurtleBot{self.version}")
            return True
            
        except Exception as e:
            logger.error(f"TurtleBot connection failed: {e}")
            self.state = RobotState.ERROR
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from TurtleBot"""
        if self.node:
            self.node.destroy_node()
            self.node = None
        self.state = RobotState.DISCONNECTED
    
    def get_capabilities(self) -> RobotCapabilities:
        """Get TurtleBot capabilities"""
        return RobotCapabilities(
            can_move=True,
            can_rotate=True,
            has_camera=True,
            has_lidar=True,
            has_imu=True,
            max_speed=0.26 if self.version == 3 else 0.5
        )
    
    async def get_pose(self) -> Pose6D:
        """Get current pose from odometry"""
        return self._current_pose
    
    async def emergency_stop(self) -> None:
        """Emergency stop - send zero velocity"""
        await self.move(0, 0)
        self.state = RobotState.EMERGENCY_STOP
    
    async def move(self, linear: float, angular: float) -> None:
        """Move the robot with linear and angular velocities"""
        if not HAS_ROS2 or not self._cmd_vel_pub:
            return
        
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        
        self._cmd_vel_pub.publish(twist)
        self.state = RobotState.MOVING if (linear != 0 or angular != 0) else RobotState.IDLE
    
    async def navigate_to(self, x: float, y: float, theta: float = 0.0) -> bool:
        """Navigate to a goal position (requires Nav2)"""
        if not HAS_ROS2 or not self.node:
            return False
        
        try:
            from nav2_simple_commander.robot_navigator import BasicNavigator
            
            navigator = BasicNavigator(node=self.node)
            
            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.header.stamp = self.node.get_clock().now().to_msg()
            goal.pose.position.x = x
            goal.pose.position.y = y
            goal.pose.orientation.z = theta
            goal.pose.orientation.w = 1.0
            
            navigator.goToPose(goal)
            
            while not navigator.isTaskComplete():
                await asyncio.sleep(0.1)
            
            return navigator.getResult() == 0
            
        except ImportError:
            logger.warning("Nav2 not available for navigation")
            return False
    
    def _odom_callback(self, msg) -> None:
        """Handle odometry messages"""
        self._current_pose = Pose6D(
            position=Position(
                x=msg.pose.pose.position.x,
                y=msg.pose.pose.position.y,
                z=msg.pose.pose.position.z
            ),
            orientation=Orientation(
                x=msg.pose.pose.orientation.x,
                y=msg.pose.pose.orientation.y,
                z=msg.pose.pose.orientation.z,
                w=msg.pose.pose.orientation.w
            )
        )
        self._trigger_callbacks('odom', self._current_pose)
    
    def _scan_callback(self, msg) -> None:
        """Handle laser scan messages"""
        self._current_scan = {
            'ranges': list(msg.ranges),
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment
        }
        self._trigger_callbacks('scan', self._current_scan)


class UniversalRobotPlatform(RobotPlatform):
    """Universal Robots (UR) arm support"""
    
    # UR model specifications
    UR_SPECS = {
        RobotType.UR3: {'reach': 0.5, 'payload': 3.0, 'dof': 6},
        RobotType.UR5: {'reach': 0.85, 'payload': 5.0, 'dof': 6},
        RobotType.UR10: {'reach': 1.3, 'payload': 10.0, 'dof': 6},
        RobotType.UR3E: {'reach': 0.5, 'payload': 3.0, 'dof': 6},
        RobotType.UR5E: {'reach': 0.85, 'payload': 5.0, 'dof': 6},
        RobotType.UR10E: {'reach': 1.3, 'payload': 12.5, 'dof': 6},
        RobotType.UR16E: {'reach': 0.9, 'payload': 16.0, 'dof': 6},
    }
    
    def __init__(self, model: RobotType = RobotType.UR5E):
        super().__init__(model)
        self.robot = None
        self.ip_address = None
        self._joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow', 
                            'wrist_1', 'wrist_2', 'wrist_3']
    
    async def connect(self, ip_address: str = "192.168.1.100") -> bool:
        """Connect to UR robot via URX"""
        if not HAS_URX:
            logger.error("urx library not available")
            return False
        
        try:
            self.state = RobotState.CONNECTING
            self.ip_address = ip_address
            
            self.robot = urx.Robot(ip_address)
            
            self.state = RobotState.IDLE
            logger.info(f"Connected to {self.robot_type.value} at {ip_address}")
            return True
            
        except Exception as e:
            logger.error(f"UR connection failed: {e}")
            self.state = RobotState.ERROR
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from UR robot"""
        if self.robot:
            self.robot.close()
            self.robot = None
        self.state = RobotState.DISCONNECTED
    
    def get_capabilities(self) -> RobotCapabilities:
        """Get UR robot capabilities"""
        specs = self.UR_SPECS.get(self.robot_type, {})
        return RobotCapabilities(
            has_arm=True,
            has_gripper=True,  # Often has attached gripper
            degrees_of_freedom=specs.get('dof', 6),
            max_payload=specs.get('payload', 5.0)
        )
    
    async def get_pose(self) -> Pose6D:
        """Get current TCP pose"""
        if not self.robot:
            return Pose6D()
        
        try:
            pose = self.robot.getl()  # [x, y, z, rx, ry, rz]
            return Pose6D(
                position=Position(x=pose[0], y=pose[1], z=pose[2]),
                # Note: URX uses axis-angle, simplified conversion here
                orientation=Orientation(x=pose[3], y=pose[4], z=pose[5])
            )
        except Exception as e:
            logger.error(f"Failed to get UR pose: {e}")
            return Pose6D()
    
    async def get_joints(self) -> JointConfiguration:
        """Get current joint positions"""
        if not self.robot:
            return JointConfiguration()
        
        try:
            positions = self.robot.getj()
            return JointConfiguration(
                joint_names=self._joint_names,
                joint_positions=list(positions)
            )
        except Exception as e:
            logger.error(f"Failed to get UR joints: {e}")
            return JointConfiguration()
    
    async def emergency_stop(self) -> None:
        """Emergency stop the robot"""
        if self.robot:
            self.robot.stop()
        self.state = RobotState.EMERGENCY_STOP
    
    async def move_linear(self, x: float, y: float, z: float, 
                         rx: float = 0, ry: float = 0, rz: float = 0,
                         acceleration: float = 0.5, velocity: float = 0.1) -> bool:
        """Move to position in linear motion"""
        if not self.robot:
            return False
        
        try:
            self.state = RobotState.MOVING
            self.robot.movel([x, y, z, rx, ry, rz], acc=acceleration, vel=velocity)
            self.state = RobotState.IDLE
            return True
        except Exception as e:
            logger.error(f"UR linear move failed: {e}")
            self.state = RobotState.ERROR
            return False
    
    async def move_joints(self, joint_positions: list[float],
                         acceleration: float = 0.5, velocity: float = 0.5) -> bool:
        """Move to joint configuration"""
        if not self.robot or len(joint_positions) != 6:
            return False
        
        try:
            self.state = RobotState.MOVING
            self.robot.movej(joint_positions, acc=acceleration, vel=velocity)
            self.state = RobotState.IDLE
            return True
        except Exception as e:
            logger.error(f"UR joint move failed: {e}")
            self.state = RobotState.ERROR
            return False
    
    async def open_gripper(self) -> bool:
        """Open gripper (if attached)"""
        if not self.robot:
            return False
        
        try:
            # Standard URCap gripper command
            self.robot.send_program("rq_open()")
            return True
        except Exception as e:
            logger.error(f"Failed to open gripper: {e}")
            return False
    
    async def close_gripper(self) -> bool:
        """Close gripper (if attached)"""
        if not self.robot:
            return False
        
        try:
            self.robot.send_program("rq_close()")
            return True
        except Exception as e:
            logger.error(f"Failed to close gripper: {e}")
            return False
    
    async def set_freedrive(self, enabled: bool = True) -> None:
        """Enable/disable freedrive mode"""
        if self.robot:
            if enabled:
                self.robot.set_freedrive(True)
            else:
                self.robot.set_freedrive(False)


class DJITelloPlatform(RobotPlatform):
    """DJI Tello drone support"""
    
    def __init__(self):
        super().__init__(RobotType.DJI_TELLO)
        self.tello = None
        self._is_flying = False
    
    async def connect(self) -> bool:
        """Connect to DJI Tello"""
        try:
            from djitellopy import Tello
            
            self.state = RobotState.CONNECTING
            self.tello = Tello()
            self.tello.connect()
            
            self.state = RobotState.IDLE
            logger.info("Connected to DJI Tello")
            return True
            
        except ImportError:
            logger.error("djitellopy not available")
            return False
        except Exception as e:
            logger.error(f"Tello connection failed: {e}")
            self.state = RobotState.ERROR
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Tello"""
        if self.tello:
            if self._is_flying:
                self.tello.land()
            self.tello.end()
            self.tello = None
        self.state = RobotState.DISCONNECTED
    
    def get_capabilities(self) -> RobotCapabilities:
        """Get Tello capabilities"""
        return RobotCapabilities(
            can_move=True,
            can_rotate=True,
            has_camera=True,
            can_fly=True,
            max_speed=8.0  # m/s
        )
    
    async def get_pose(self) -> Pose6D:
        """Get current pose (limited on Tello)"""
        if not self.tello:
            return Pose6D()
        
        try:
            height = self.tello.get_height()
            return Pose6D(
                position=Position(z=height / 100.0)  # cm to m
            )
        except Exception:
            return Pose6D()
    
    async def emergency_stop(self) -> None:
        """Emergency stop - land immediately"""
        if self.tello:
            try:
                self.tello.emergency()
            except Exception:
                pass
        self._is_flying = False
        self.state = RobotState.EMERGENCY_STOP
    
    async def takeoff(self) -> bool:
        """Take off"""
        if not self.tello or self._is_flying:
            return False
        
        try:
            self.tello.takeoff()
            self._is_flying = True
            self.state = RobotState.MOVING
            return True
        except Exception as e:
            logger.error(f"Takeoff failed: {e}")
            return False
    
    async def land(self) -> bool:
        """Land the drone"""
        if not self.tello or not self._is_flying:
            return False
        
        try:
            self.tello.land()
            self._is_flying = False
            self.state = RobotState.IDLE
            return True
        except Exception as e:
            logger.error(f"Landing failed: {e}")
            return False
    
    async def move(self, x: int, y: int, z: int, speed: int = 50) -> bool:
        """Move in 3D space (cm)"""
        if not self.tello or not self._is_flying:
            return False
        
        try:
            self.tello.go_xyz_speed(x, y, z, speed)
            return True
        except Exception as e:
            logger.error(f"Move failed: {e}")
            return False
    
    async def rotate(self, angle: int) -> bool:
        """Rotate by angle (degrees)"""
        if not self.tello or not self._is_flying:
            return False
        
        try:
            if angle >= 0:
                self.tello.rotate_clockwise(angle)
            else:
                self.tello.rotate_counter_clockwise(abs(angle))
            return True
        except Exception as e:
            logger.error(f"Rotation failed: {e}")
            return False
    
    async def flip(self, direction: str = 'forward') -> bool:
        """Perform a flip"""
        if not self.tello or not self._is_flying:
            return False
        
        try:
            direction_map = {
                'forward': 'f',
                'back': 'b',
                'left': 'l',
                'right': 'r'
            }
            self.tello.flip(direction_map.get(direction, 'f'))
            return True
        except Exception as e:
            logger.error(f"Flip failed: {e}")
            return False
    
    def get_battery(self) -> int:
        """Get battery percentage"""
        if self.tello:
            try:
                return self.tello.get_battery()
            except Exception:
                pass
        return 0
    
    def get_frame(self):
        """Get current camera frame"""
        if self.tello:
            try:
                return self.tello.get_frame_read().frame
            except Exception:
                pass
        return None


class ArduinoRobotPlatform(RobotPlatform):
    """Arduino-based robot support via serial"""
    
    # Common commands for Arduino robots
    COMMANDS = {
        'forward': b'F',
        'backward': b'B',
        'left': b'L',
        'right': b'R',
        'stop': b'S',
        'speed': b'V',
        'servo': b'A',
        'read_sensors': b'?'
    }
    
    def __init__(self):
        super().__init__(RobotType.ARDUINO)
        self.serial = None
        self._port = None
        self._baudrate = 9600
    
    async def connect(self, port: str = "COM3", baudrate: int = 9600) -> bool:
        """Connect to Arduino via serial"""
        if not HAS_SERIAL:
            logger.error("pyserial not available")
            return False
        
        try:
            self.state = RobotState.CONNECTING
            self._port = port
            self._baudrate = baudrate
            
            self.serial = serial.Serial(port, baudrate, timeout=1)
            await asyncio.sleep(2)  # Wait for Arduino reset
            
            self.state = RobotState.IDLE
            logger.info(f"Connected to Arduino at {port}")
            return True
            
        except Exception as e:
            logger.error(f"Arduino connection failed: {e}")
            self.state = RobotState.ERROR
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Arduino"""
        if self.serial:
            self.serial.close()
            self.serial = None
        self.state = RobotState.DISCONNECTED
    
    def get_capabilities(self) -> RobotCapabilities:
        """Get Arduino robot capabilities (varies by setup)"""
        return RobotCapabilities(
            can_move=True,
            can_rotate=True,
            has_camera=False  # Usually no camera on Arduino
        )
    
    async def get_pose(self) -> Pose6D:
        """Get pose (not available for basic Arduino)"""
        return Pose6D()
    
    async def emergency_stop(self) -> None:
        """Emergency stop"""
        await self.send_command('stop')
        self.state = RobotState.EMERGENCY_STOP
    
    async def send_command(self, command: str, value: int = None) -> str:
        """Send command to Arduino"""
        if not self.serial:
            return ""
        
        try:
            cmd = self.COMMANDS.get(command, command.encode())
            if value is not None:
                cmd = cmd + str(value).encode()
            
            self.serial.write(cmd + b'\n')
            await asyncio.sleep(0.05)
            
            if self.serial.in_waiting:
                return self.serial.readline().decode().strip()
            return ""
            
        except Exception as e:
            logger.error(f"Arduino command failed: {e}")
            return ""
    
    async def move_forward(self, speed: int = 100) -> None:
        """Move forward"""
        await self.send_command('speed', speed)
        await self.send_command('forward')
        self.state = RobotState.MOVING
    
    async def move_backward(self, speed: int = 100) -> None:
        """Move backward"""
        await self.send_command('speed', speed)
        await self.send_command('backward')
        self.state = RobotState.MOVING
    
    async def turn_left(self, speed: int = 100) -> None:
        """Turn left"""
        await self.send_command('speed', speed)
        await self.send_command('left')
        self.state = RobotState.MOVING
    
    async def turn_right(self, speed: int = 100) -> None:
        """Turn right"""
        await self.send_command('speed', speed)
        await self.send_command('right')
        self.state = RobotState.MOVING
    
    async def stop(self) -> None:
        """Stop movement"""
        await self.send_command('stop')
        self.state = RobotState.IDLE
    
    async def set_servo(self, channel: int, angle: int) -> None:
        """Set servo angle"""
        await self.send_command('servo', channel * 1000 + angle)
    
    async def read_sensors(self) -> dict[str, Any]:
        """Read sensor values"""
        response = await self.send_command('read_sensors')
        
        # Parse response (format depends on Arduino code)
        try:
            # Assuming format: "D1:123,D2:456,A0:789"
            sensors = {}
            for part in response.split(','):
                if ':' in part:
                    name, value = part.split(':')
                    sensors[name] = int(value)
            return sensors
        except Exception:
            return {}


class PiCarPlatform(RobotPlatform):
    """Raspberry Pi PiCar platform support"""
    
    def __init__(self):
        super().__init__(RobotType.PICAR)
        self.car = None
    
    async def connect(self) -> bool:
        """Connect to PiCar"""
        try:
            from picar import PiCar
            
            self.state = RobotState.CONNECTING
            self.car = PiCar()
            
            self.state = RobotState.IDLE
            logger.info("Connected to PiCar")
            return True
            
        except ImportError:
            # Fallback to GPIO-based control
            return await self._connect_gpio()
        except Exception as e:
            logger.error(f"PiCar connection failed: {e}")
            self.state = RobotState.ERROR
            return False
    
    async def _connect_gpio(self) -> bool:
        """Fallback GPIO-based control"""
        try:
            import RPi.GPIO as GPIO
            self._gpio = GPIO
            self._gpio.setmode(GPIO.BCM)
            
            # Motor pins (adjust for your setup)
            self._motor_pins = {
                'left_forward': 17,
                'left_backward': 18,
                'right_forward': 22,
                'right_backward': 23,
                'pwm_left': 12,
                'pwm_right': 13
            }
            
            for pin in self._motor_pins.values():
                self._gpio.setup(pin, GPIO.OUT)
            
            self._pwm_left = self._gpio.PWM(self._motor_pins['pwm_left'], 1000)
            self._pwm_right = self._gpio.PWM(self._motor_pins['pwm_right'], 1000)
            self._pwm_left.start(0)
            self._pwm_right.start(0)
            
            self.state = RobotState.IDLE
            return True
            
        except Exception as e:
            logger.error(f"GPIO control failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from PiCar"""
        if hasattr(self, '_gpio'):
            self._pwm_left.stop()
            self._pwm_right.stop()
            self._gpio.cleanup()
        self.state = RobotState.DISCONNECTED
    
    def get_capabilities(self) -> RobotCapabilities:
        """Get PiCar capabilities"""
        return RobotCapabilities(
            can_move=True,
            can_rotate=True,
            has_camera=True,  # Pi Camera
            max_speed=0.5  # Approximate
        )
    
    async def get_pose(self) -> Pose6D:
        """Get pose (not available without encoders)"""
        return Pose6D()
    
    async def emergency_stop(self) -> None:
        """Emergency stop"""
        await self.stop()
        self.state = RobotState.EMERGENCY_STOP
    
    async def forward(self, speed: int = 50) -> None:
        """Move forward"""
        if self.car:
            self.car.forward(speed)
        else:
            self._set_motors(speed, speed)
        self.state = RobotState.MOVING
    
    async def backward(self, speed: int = 50) -> None:
        """Move backward"""
        if self.car:
            self.car.backward(speed)
        else:
            self._set_motors(-speed, -speed)
        self.state = RobotState.MOVING
    
    async def turn_left(self, speed: int = 50) -> None:
        """Turn left"""
        if self.car:
            self.car.turn_left(speed)
        else:
            self._set_motors(-speed, speed)
        self.state = RobotState.MOVING
    
    async def turn_right(self, speed: int = 50) -> None:
        """Turn right"""
        if self.car:
            self.car.turn_right(speed)
        else:
            self._set_motors(speed, -speed)
        self.state = RobotState.MOVING
    
    async def stop(self) -> None:
        """Stop movement"""
        if self.car:
            self.car.stop()
        else:
            self._set_motors(0, 0)
        self.state = RobotState.IDLE
    
    def _set_motors(self, left: int, right: int) -> None:
        """Set motor speeds via GPIO"""
        if not hasattr(self, '_gpio'):
            return
        
        # Left motor
        if left > 0:
            self._gpio.output(self._motor_pins['left_forward'], 1)
            self._gpio.output(self._motor_pins['left_backward'], 0)
        elif left < 0:
            self._gpio.output(self._motor_pins['left_forward'], 0)
            self._gpio.output(self._motor_pins['left_backward'], 1)
        else:
            self._gpio.output(self._motor_pins['left_forward'], 0)
            self._gpio.output(self._motor_pins['left_backward'], 0)
        
        # Right motor
        if right > 0:
            self._gpio.output(self._motor_pins['right_forward'], 1)
            self._gpio.output(self._motor_pins['right_backward'], 0)
        elif right < 0:
            self._gpio.output(self._motor_pins['right_forward'], 0)
            self._gpio.output(self._motor_pins['right_backward'], 1)
        else:
            self._gpio.output(self._motor_pins['right_forward'], 0)
            self._gpio.output(self._motor_pins['right_backward'], 0)
        
        self._pwm_left.ChangeDutyCycle(abs(left))
        self._pwm_right.ChangeDutyCycle(abs(right))


class RobotPlatformManager:
    """Manager for multiple robot platforms"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._platforms: dict[str, RobotPlatform] = {}
        self._active_platform: Optional[str] = None
        
        # Platform factories
        self._factories = {
            RobotType.TURTLEBOT: lambda: TurtleBotPlatform(2),
            RobotType.TURTLEBOT3: lambda: TurtleBotPlatform(3),
            RobotType.TURTLEBOT4: lambda: TurtleBotPlatform(4),
            RobotType.UR3: lambda: UniversalRobotPlatform(RobotType.UR3),
            RobotType.UR5: lambda: UniversalRobotPlatform(RobotType.UR5),
            RobotType.UR10: lambda: UniversalRobotPlatform(RobotType.UR10),
            RobotType.UR3E: lambda: UniversalRobotPlatform(RobotType.UR3E),
            RobotType.UR5E: lambda: UniversalRobotPlatform(RobotType.UR5E),
            RobotType.UR10E: lambda: UniversalRobotPlatform(RobotType.UR10E),
            RobotType.UR16E: lambda: UniversalRobotPlatform(RobotType.UR16E),
            RobotType.DJI_TELLO: DJITelloPlatform,
            RobotType.ARDUINO: ArduinoRobotPlatform,
            RobotType.PICAR: PiCarPlatform,
        }
    
    def create_platform(self, robot_type: RobotType, name: str = None) -> RobotPlatform:
        """Create a robot platform instance"""
        name = name or robot_type.value
        
        if robot_type not in self._factories:
            raise ValueError(f"Unsupported robot type: {robot_type}")
        
        platform = self._factories[robot_type]()
        self._platforms[name] = platform
        
        if self._active_platform is None:
            self._active_platform = name
        
        return platform
    
    def get_platform(self, name: str = None) -> Optional[RobotPlatform]:
        """Get a robot platform by name"""
        name = name or self._active_platform
        return self._platforms.get(name)
    
    def set_active(self, name: str) -> None:
        """Set the active platform"""
        if name in self._platforms:
            self._active_platform = name
    
    def list_platforms(self) -> list[str]:
        """List all registered platforms"""
        return list(self._platforms.keys())
    
    def list_supported(self) -> list[RobotType]:
        """List all supported robot types"""
        return list(self._factories.keys())
    
    async def disconnect_all(self) -> None:
        """Disconnect all platforms"""
        for platform in self._platforms.values():
            await platform.disconnect()
        self._platforms.clear()
        self._active_platform = None


def get_platform_manager() -> RobotPlatformManager:
    """Get the singleton platform manager"""
    return RobotPlatformManager()


# Convenience functions
async def connect_robot(robot_type: RobotType, name: str = None, **kwargs) -> Optional[RobotPlatform]:
    """Connect to a robot platform"""
    manager = get_platform_manager()
    platform = manager.create_platform(robot_type, name)
    
    if await platform.connect(**kwargs):
        return platform
    return None


async def quick_turtlebot(node_name: str = "enigma_engine") -> Optional[TurtleBotPlatform]:
    """Quick connect to TurtleBot3"""
    platform = await connect_robot(RobotType.TURTLEBOT3)
    return platform


async def quick_ur5(ip: str = "192.168.1.100") -> Optional[UniversalRobotPlatform]:
    """Quick connect to UR5e"""
    platform = await connect_robot(RobotType.UR5E, ip_address=ip)
    return platform


async def quick_tello() -> Optional[DJITelloPlatform]:
    """Quick connect to DJI Tello"""
    platform = await connect_robot(RobotType.DJI_TELLO)
    return platform
