"""
ROS/ROS2 Integration

Robot Operating System integration for Enigma AI Engine.
Enables AI-controlled robots with ROS1 and ROS2 support.

FILE: enigma_engine/robotics/ros_integration.py
TYPE: Robotics
MAIN CLASSES: ROSBridge, ROSNode, ROSTopicManager
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
from typing import Any, Callable, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ROSVersion(Enum):
    """ROS version."""
    ROS1 = "ros1"
    ROS2 = "ros2"


class MessageType(Enum):
    """Common ROS message types."""
    STRING = "std_msgs/String"
    INT32 = "std_msgs/Int32"
    FLOAT32 = "std_msgs/Float32"
    BOOL = "std_msgs/Bool"
    TWIST = "geometry_msgs/Twist"
    POSE = "geometry_msgs/Pose"
    POINT = "geometry_msgs/Point"
    QUATERNION = "geometry_msgs/Quaternion"
    ODOM = "nav_msgs/Odometry"
    LASER_SCAN = "sensor_msgs/LaserScan"
    IMAGE = "sensor_msgs/Image"
    JOINT_STATE = "sensor_msgs/JointState"
    CAMERA_INFO = "sensor_msgs/CameraInfo"


@dataclass
class ROSMessage:
    """A ROS message."""
    topic: str
    msg_type: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    frame_id: str = ""


@dataclass
class ROSConfig:
    """ROS configuration."""
    ros_version: ROSVersion = ROSVersion.ROS2
    rosbridge_url: str = "ws://localhost:9090"
    namespace: str = ""
    node_name: str = "enigma_engine_node"
    
    # Topics
    cmd_vel_topic: str = "/cmd_vel"
    odom_topic: str = "/odom"
    scan_topic: str = "/scan"
    camera_topic: str = "/camera/image_raw"
    
    # TF
    use_tf: bool = True
    base_frame: str = "base_link"
    odom_frame: str = "odom"
    map_frame: str = "map"


class ROSBridge:
    """
    Connect to ROS via rosbridge WebSocket protocol.
    Works with both ROS1 and ROS2.
    """
    
    def __init__(self, config: ROSConfig = None):
        self.config = config or ROSConfig()
        self._connected = False
        self._ws = None
        self._message_queue: Queue = Queue()
        self._callbacks: dict[str, list[Callable[[ROSMessage], None]]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._message_id = 0
    
    def connect(self) -> bool:
        """
        Connect to rosbridge server.
        
        Returns:
            True if connected successfully
        """
        try:
            import websocket
            
            self._ws = websocket.WebSocket()
            self._ws.connect(self.config.rosbridge_url)
            self._connected = True
            
            # Start receive thread
            self._running = True
            self._thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._thread.start()
            
            logger.info(f"Connected to rosbridge at {self.config.rosbridge_url}")
            return True
            
        except ImportError:
            logger.error("websocket-client required: pip install websocket-client")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to rosbridge: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from rosbridge."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception as e:
                logger.debug(f"Error closing websocket: {e}")
        self._connected = False
    
    def _receive_loop(self):
        """Background thread for receiving messages."""
        while self._running and self._connected:
            try:
                if self._ws:
                    data = self._ws.recv()
                    if data:
                        self._handle_message(json.loads(data))
            except Exception as e:
                if self._running:
                    logger.debug(f"Receive error: {e}")
                    time.sleep(0.1)
    
    def _handle_message(self, msg: dict[str, Any]):
        """Handle incoming rosbridge message."""
        op = msg.get("op")
        
        if op == "publish":
            topic = msg.get("topic")
            ros_msg = ROSMessage(
                topic=topic,
                msg_type=msg.get("type", ""),
                data=msg.get("msg", {})
            )
            
            # Call registered callbacks
            if topic in self._callbacks:
                for callback in self._callbacks[topic]:
                    try:
                        callback(ros_msg)
                    except Exception as e:
                        logger.error(f"Callback error for {topic}: {e}")
        
        elif op == "service_response":
            # Handle service responses
            pass
    
    def _send(self, message: dict[str, Any]):
        """Send message to rosbridge."""
        if not self._connected:
            return False
        
        try:
            self._ws.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Send error: {e}")
            return False
    
    def subscribe(
        self,
        topic: str,
        msg_type: str,
        callback: Callable[[ROSMessage], None],
        throttle_rate: int = 0,
        queue_length: int = 1
    ):
        """
        Subscribe to a ROS topic.
        
        Args:
            topic: Topic name
            msg_type: Message type
            callback: Function to call on new messages
            throttle_rate: Minimum ms between messages
            queue_length: Message queue length
        """
        # Add callback
        if topic not in self._callbacks:
            self._callbacks[topic] = []
        self._callbacks[topic].append(callback)
        
        # Send subscribe message
        self._message_id += 1
        msg = {
            "op": "subscribe",
            "id": f"subscribe:{topic}:{self._message_id}",
            "topic": topic,
            "type": msg_type,
            "throttle_rate": throttle_rate,
            "queue_length": queue_length
        }
        self._send(msg)
        
        logger.info(f"Subscribed to {topic}")
    
    def unsubscribe(self, topic: str):
        """Unsubscribe from a topic."""
        msg = {
            "op": "unsubscribe",
            "topic": topic
        }
        self._send(msg)
        
        if topic in self._callbacks:
            del self._callbacks[topic]
    
    def publish(self, topic: str, msg_type: str, data: dict[str, Any]):
        """
        Publish a message to a topic.
        
        Args:
            topic: Topic name
            msg_type: Message type
            data: Message data
        """
        msg = {
            "op": "publish",
            "topic": topic,
            "type": msg_type,
            "msg": data
        }
        self._send(msg)
    
    def advertise(self, topic: str, msg_type: str):
        """Advertise a topic for publishing."""
        msg = {
            "op": "advertise",
            "topic": topic,
            "type": msg_type
        }
        self._send(msg)
    
    def unadvertise(self, topic: str):
        """Stop advertising a topic."""
        msg = {
            "op": "unadvertise",
            "topic": topic
        }
        self._send(msg)
    
    def call_service(
        self,
        service: str,
        service_type: str,
        args: dict[str, Any] = None,
        callback: Callable[[dict[str, Any]], None] = None
    ):
        """
        Call a ROS service.
        
        Args:
            service: Service name
            service_type: Service type
            args: Service arguments
            callback: Callback for response
        """
        self._message_id += 1
        msg = {
            "op": "call_service",
            "id": f"service:{service}:{self._message_id}",
            "service": service,
            "type": service_type,
            "args": args or {}
        }
        self._send(msg)


class ROSTopicManager:
    """
    Manage ROS topics for common robot operations.
    """
    
    def __init__(self, bridge: ROSBridge, config: ROSConfig = None):
        self.bridge = bridge
        self.config = config or ROSConfig()
        
        self._odom_data: Optional[dict[str, Any]] = None
        self._scan_data: Optional[dict[str, Any]] = None
        self._joint_states: dict[str, float] = {}
    
    def setup_subscriptions(self):
        """Set up common topic subscriptions."""
        # Odometry
        self.bridge.subscribe(
            self.config.odom_topic,
            MessageType.ODOM.value,
            self._handle_odom
        )
        
        # Laser scan
        self.bridge.subscribe(
            self.config.scan_topic,
            MessageType.LASER_SCAN.value,
            self._handle_scan
        )
    
    def _handle_odom(self, msg: ROSMessage):
        """Handle odometry message."""
        self._odom_data = msg.data
    
    def _handle_scan(self, msg: ROSMessage):
        """Handle laser scan message."""
        self._scan_data = msg.data
    
    def send_velocity(
        self,
        linear_x: float = 0,
        linear_y: float = 0,
        linear_z: float = 0,
        angular_x: float = 0,
        angular_y: float = 0,
        angular_z: float = 0
    ):
        """
        Send velocity command to robot.
        
        Args:
            linear_x/y/z: Linear velocities
            angular_x/y/z: Angular velocities
        """
        twist_msg = {
            "linear": {"x": linear_x, "y": linear_y, "z": linear_z},
            "angular": {"x": angular_x, "y": angular_y, "z": angular_z}
        }
        
        self.bridge.publish(
            self.config.cmd_vel_topic,
            MessageType.TWIST.value,
            twist_msg
        )
    
    def move_forward(self, speed: float = 0.2):
        """Move robot forward."""
        self.send_velocity(linear_x=speed)
    
    def move_backward(self, speed: float = 0.2):
        """Move robot backward."""
        self.send_velocity(linear_x=-speed)
    
    def turn_left(self, speed: float = 0.5):
        """Turn robot left."""
        self.send_velocity(angular_z=speed)
    
    def turn_right(self, speed: float = 0.5):
        """Turn robot right."""
        self.send_velocity(angular_z=-speed)
    
    def stop(self):
        """Stop robot movement."""
        self.send_velocity()
    
    def get_position(self) -> Optional[tuple[float, float, float]]:
        """Get current robot position (x, y, z)."""
        if self._odom_data:
            pos = self._odom_data.get("pose", {}).get("pose", {}).get("position", {})
            return (pos.get("x", 0), pos.get("y", 0), pos.get("z", 0))
        return None
    
    def get_orientation(self) -> Optional[tuple[float, float, float, float]]:
        """Get current robot orientation as quaternion."""
        if self._odom_data:
            ori = self._odom_data.get("pose", {}).get("pose", {}).get("orientation", {})
            return (ori.get("x", 0), ori.get("y", 0), ori.get("z", 0), ori.get("w", 1))
        return None
    
    def get_scan_ranges(self) -> Optional[list[float]]:
        """Get laser scan ranges."""
        if self._scan_data:
            return self._scan_data.get("ranges", [])
        return None
    
    def get_closest_obstacle(self) -> Optional[tuple[float, float]]:
        """
        Get distance and angle to closest obstacle.
        
        Returns:
            Tuple of (distance, angle) or None
        """
        if not self._scan_data:
            return None
        
        ranges = self._scan_data.get("ranges", [])
        if not ranges:
            return None
        
        angle_min = self._scan_data.get("angle_min", 0)
        angle_increment = self._scan_data.get("angle_increment", 0.01)
        
        min_dist = float('inf')
        min_angle = 0
        
        for i, r in enumerate(ranges):
            if r > 0 and r < min_dist:
                min_dist = r
                min_angle = angle_min + i * angle_increment
        
        if min_dist < float('inf'):
            return (min_dist, min_angle)
        return None


class ROSNode:
    """
    High-level ROS node interface for Enigma AI Engine.
    """
    
    def __init__(self, config: ROSConfig = None):
        self.config = config or ROSConfig()
        self.bridge = ROSBridge(config)
        self.topics = ROSTopicManager(self.bridge, config)
        
        self._ai_command_callback: Optional[Callable[[str], str]] = None
    
    def start(self) -> bool:
        """Start the ROS node."""
        if self.bridge.connect():
            self.topics.setup_subscriptions()
            
            # Advertise cmd_vel
            self.bridge.advertise(
                self.config.cmd_vel_topic,
                MessageType.TWIST.value
            )
            
            return True
        return False
    
    def stop(self):
        """Stop the ROS node."""
        self.topics.stop()
        self.bridge.disconnect()
    
    def set_ai_command_handler(self, handler: Callable[[str], str]):
        """Set AI command handler for natural language control."""
        self._ai_command_callback = handler
    
    def process_ai_command(self, command: str) -> str:
        """
        Process a natural language command.
        
        Args:
            command: Natural language command
        
        Returns:
            Response string
        """
        command_lower = command.lower()
        
        # Simple command parsing
        if "forward" in command_lower or "ahead" in command_lower:
            self.topics.move_forward()
            return "Moving forward"
        
        elif "backward" in command_lower or "back" in command_lower:
            self.topics.move_backward()
            return "Moving backward"
        
        elif "left" in command_lower:
            self.topics.turn_left()
            return "Turning left"
        
        elif "right" in command_lower:
            self.topics.turn_right()
            return "Turning right"
        
        elif "stop" in command_lower or "halt" in command_lower:
            self.topics.stop()
            return "Stopped"
        
        elif "position" in command_lower or "where" in command_lower:
            pos = self.topics.get_position()
            if pos:
                return f"Position: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}"
            return "Position unknown"
        
        elif "obstacle" in command_lower or "scan" in command_lower:
            closest = self.topics.get_closest_obstacle()
            if closest:
                return f"Closest obstacle: {closest[0]:.2f}m at {closest[1]:.2f} rad"
            return "No obstacles detected"
        
        # Use AI if available
        if self._ai_command_callback:
            return self._ai_command_callback(command)
        
        return f"Unknown command: {command}"


def create_ros_node(
    rosbridge_url: str = "ws://localhost:9090",
    ros_version: ROSVersion = ROSVersion.ROS2
) -> ROSNode:
    """
    Create a ROS node.
    
    Args:
        rosbridge_url: URL of rosbridge server
        ros_version: ROS1 or ROS2
    
    Returns:
        ROSNode instance
    """
    config = ROSConfig(
        ros_version=ros_version,
        rosbridge_url=rosbridge_url
    )
    return ROSNode(config)
