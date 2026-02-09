"""
ROS2 Integration for Enigma AI Engine

Robot Operating System 2 integration for robotics.

Features:
- ROS2 node wrapper
- Topic publishing/subscribing
- Service calls
- Action clients
- Message conversion

Usage:
    from enigma_engine.tools.ros2_integration import ROS2Node, get_ros2_node
    
    node = get_ros2_node("enigma_ai")
    
    # Subscribe to topics
    node.subscribe("/camera/image", callback=process_image)
    
    # Publish messages
    node.publish("/cmd_vel", twist_msg)
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Common ROS2 message types."""
    STRING = "std_msgs/String"
    INT32 = "std_msgs/Int32"
    FLOAT32 = "std_msgs/Float32"
    BOOL = "std_msgs/Bool"
    TWIST = "geometry_msgs/Twist"
    POSE = "geometry_msgs/Pose"
    POINT = "geometry_msgs/Point"
    IMAGE = "sensor_msgs/Image"
    LASER_SCAN = "sensor_msgs/LaserScan"
    POINT_CLOUD = "sensor_msgs/PointCloud2"
    ODOMETRY = "nav_msgs/Odometry"
    PATH = "nav_msgs/Path"
    JOINT_STATE = "sensor_msgs/JointState"


@dataclass
class TopicInfo:
    """Information about a ROS2 topic."""
    name: str
    message_type: str
    is_publishing: bool = False
    is_subscribed: bool = False
    
    # Stats
    message_count: int = 0
    last_message_time: float = 0.0


@dataclass
class ServiceInfo:
    """Information about a ROS2 service."""
    name: str
    service_type: str
    
    # Stats
    call_count: int = 0
    last_call_time: float = 0.0


class MessageConverter:
    """Convert between Python dicts and ROS2 messages."""
    
    @staticmethod
    def to_twist(linear: Dict[str, float], angular: Dict[str, float]) -> Dict[str, Any]:
        """Create Twist message dict."""
        return {
            "linear": {
                "x": linear.get("x", 0.0),
                "y": linear.get("y", 0.0),
                "z": linear.get("z", 0.0)
            },
            "angular": {
                "x": angular.get("x", 0.0),
                "y": angular.get("y", 0.0),
                "z": angular.get("z", 0.0)
            }
        }
    
    @staticmethod
    def to_pose(position: Dict[str, float], orientation: Dict[str, float]) -> Dict[str, Any]:
        """Create Pose message dict."""
        return {
            "position": {
                "x": position.get("x", 0.0),
                "y": position.get("y", 0.0),
                "z": position.get("z", 0.0)
            },
            "orientation": {
                "x": orientation.get("x", 0.0),
                "y": orientation.get("y", 0.0),
                "z": orientation.get("z", 0.0),
                "w": orientation.get("w", 1.0)
            }
        }
    
    @staticmethod
    def from_odometry(msg: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from Odometry message."""
        return {
            "position": msg.get("pose", {}).get("pose", {}).get("position", {}),
            "orientation": msg.get("pose", {}).get("pose", {}).get("orientation", {}),
            "linear_velocity": msg.get("twist", {}).get("twist", {}).get("linear", {}),
            "angular_velocity": msg.get("twist", {}).get("twist", {}).get("angular", {})
        }
    
    @staticmethod
    def from_laser_scan(msg: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from LaserScan message."""
        return {
            "ranges": msg.get("ranges", []),
            "angle_min": msg.get("angle_min", 0.0),
            "angle_max": msg.get("angle_max", 0.0),
            "angle_increment": msg.get("angle_increment", 0.0),
            "range_min": msg.get("range_min", 0.0),
            "range_max": msg.get("range_max", 0.0)
        }


class ROS2Bridge:
    """Bridge to ROS2 system."""
    
    def __init__(self):
        self._rclpy = None
        self._node = None
        self._initialized = False
        
        self._publishers: Dict[str, Any] = {}
        self._subscribers: Dict[str, Any] = {}
        self._services: Dict[str, Any] = {}
        self._action_clients: Dict[str, Any] = {}
        
        self._message_queue: queue.Queue = queue.Queue()
    
    def initialize(self, node_name: str = "enigma_ai") -> bool:
        """
        Initialize ROS2.
        
        Returns:
            True if successful
        """
        try:
            import rclpy
            from rclpy.node import Node
            
            self._rclpy = rclpy
            
            if not rclpy.ok():
                rclpy.init()
            
            self._node = rclpy.create_node(node_name)
            self._initialized = True
            
            logger.info(f"ROS2 node '{node_name}' initialized")
            return True
            
        except ImportError:
            logger.warning("rclpy not available - ROS2 integration disabled")
            return False
        except Exception as e:
            logger.error(f"ROS2 initialization failed: {e}")
            return False
    
    def shutdown(self):
        """Shutdown ROS2."""
        if self._node:
            self._node.destroy_node()
        
        if self._rclpy and self._rclpy.ok():
            self._rclpy.shutdown()
        
        self._initialized = False
    
    def create_publisher(
        self,
        topic: str,
        message_type: str,
        qos_depth: int = 10
    ) -> bool:
        """Create a publisher."""
        if not self._initialized:
            return False
        
        try:
            msg_class = self._get_message_class(message_type)
            if msg_class:
                pub = self._node.create_publisher(msg_class, topic, qos_depth)
                self._publishers[topic] = pub
                logger.info(f"Created publisher for {topic}")
                return True
        except Exception as e:
            logger.error(f"Failed to create publisher: {e}")
        
        return False
    
    def create_subscriber(
        self,
        topic: str,
        message_type: str,
        callback: Callable[[Any], None],
        qos_depth: int = 10
    ) -> bool:
        """Create a subscriber."""
        if not self._initialized:
            return False
        
        try:
            msg_class = self._get_message_class(message_type)
            if msg_class:
                sub = self._node.create_subscription(
                    msg_class, topic, callback, qos_depth
                )
                self._subscribers[topic] = sub
                logger.info(f"Created subscriber for {topic}")
                return True
        except Exception as e:
            logger.error(f"Failed to create subscriber: {e}")
        
        return False
    
    def publish(self, topic: str, message: Any):
        """Publish a message."""
        if topic in self._publishers:
            self._publishers[topic].publish(message)
    
    def spin_once(self, timeout_sec: float = 0.0):
        """Process pending callbacks."""
        if self._rclpy and self._node:
            self._rclpy.spin_once(self._node, timeout_sec=timeout_sec)
    
    def _get_message_class(self, message_type: str) -> Optional[Type]:
        """Get message class from type string."""
        try:
            parts = message_type.split("/")
            if len(parts) == 2:
                pkg, msg = parts
                module = __import__(f"{pkg}.msg", fromlist=[msg])
                return getattr(module, msg)
        except Exception as e:
            logger.error(f"Failed to get message class: {e}")
        
        return None


class MockROS2Bridge(ROS2Bridge):
    """Mock ROS2 bridge for testing without ROS2."""
    
    def __init__(self):
        super().__init__()
        self._mock_topics: Dict[str, List[Any]] = {}
        self._callbacks: Dict[str, Callable] = {}
    
    def initialize(self, node_name: str = "enigma_ai") -> bool:
        self._initialized = True
        logger.info(f"Mock ROS2 node '{node_name}' initialized")
        return True
    
    def shutdown(self):
        self._initialized = False
    
    def create_publisher(
        self,
        topic: str,
        message_type: str,
        qos_depth: int = 10
    ) -> bool:
        if topic not in self._mock_topics:
            self._mock_topics[topic] = []
        logger.info(f"Mock publisher created for {topic}")
        return True
    
    def create_subscriber(
        self,
        topic: str,
        message_type: str,
        callback: Callable[[Any], None],
        qos_depth: int = 10
    ) -> bool:
        self._callbacks[topic] = callback
        if topic not in self._mock_topics:
            self._mock_topics[topic] = []
        logger.info(f"Mock subscriber created for {topic}")
        return True
    
    def publish(self, topic: str, message: Any):
        if topic in self._mock_topics:
            self._mock_topics[topic].append(message)
            
            # Call subscriber callback if exists
            if topic in self._callbacks:
                self._callbacks[topic](message)
    
    def spin_once(self, timeout_sec: float = 0.0):
        pass  # No-op for mock


class ROS2Node:
    """High-level ROS2 node interface."""
    
    def __init__(
        self,
        node_name: str = "enigma_ai",
        use_mock: bool = False
    ):
        """
        Initialize ROS2 node.
        
        Args:
            node_name: Name of the ROS2 node
            use_mock: Use mock bridge for testing
        """
        self._name = node_name
        
        if use_mock:
            self._bridge = MockROS2Bridge()
        else:
            self._bridge = ROS2Bridge()
        
        self._topics: Dict[str, TopicInfo] = {}
        self._services: Dict[str, ServiceInfo] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        
        # Thread for spinning
        self._running = False
        self._spin_thread: Optional[threading.Thread] = None
        
        # Initialize
        self._bridge.initialize(node_name)
    
    def subscribe(
        self,
        topic: str,
        callback: Callable[[Any], None],
        message_type: Optional[str] = None
    ):
        """
        Subscribe to a topic.
        
        Args:
            topic: Topic name
            callback: Callback for messages
            message_type: Message type (auto-detected if not provided)
        """
        # Auto-detect message type
        if not message_type:
            message_type = self._infer_message_type(topic)
        
        def wrapper(msg):
            # Convert to dict
            data = self._message_to_dict(msg)
            
            # Update stats
            if topic in self._topics:
                self._topics[topic].message_count += 1
                self._topics[topic].last_message_time = time.time()
            
            # Call user callback
            callback(data)
        
        if self._bridge.create_subscriber(topic, message_type, wrapper):
            self._topics[topic] = TopicInfo(
                name=topic,
                message_type=message_type,
                is_subscribed=True
            )
            
            if topic not in self._callbacks:
                self._callbacks[topic] = []
            self._callbacks[topic].append(callback)
    
    def publish(
        self,
        topic: str,
        data: Any,
        message_type: Optional[str] = None
    ):
        """
        Publish to a topic.
        
        Args:
            topic: Topic name
            data: Data to publish (dict or message)
            message_type: Message type
        """
        # Create publisher if needed
        if topic not in self._topics or not self._topics[topic].is_publishing:
            if not message_type:
                message_type = self._infer_message_type(topic)
            
            if self._bridge.create_publisher(topic, message_type):
                if topic not in self._topics:
                    self._topics[topic] = TopicInfo(
                        name=topic,
                        message_type=message_type
                    )
                self._topics[topic].is_publishing = True
        
        # Publish
        self._bridge.publish(topic, data)
        
        # Update stats
        if topic in self._topics:
            self._topics[topic].message_count += 1
            self._topics[topic].last_message_time = time.time()
    
    def publish_cmd_vel(
        self,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        angular_z: float = 0.0
    ):
        """Publish velocity command."""
        twist = MessageConverter.to_twist(
            {"x": linear_x, "y": linear_y, "z": 0.0},
            {"x": 0.0, "y": 0.0, "z": angular_z}
        )
        self.publish("/cmd_vel", twist, MessageType.TWIST.value)
    
    def stop_robot(self):
        """Send stop command."""
        self.publish_cmd_vel(0.0, 0.0, 0.0)
    
    def move_forward(self, speed: float = 0.5):
        """Move robot forward."""
        self.publish_cmd_vel(linear_x=speed)
    
    def turn(self, angular_speed: float = 0.5):
        """Turn robot."""
        self.publish_cmd_vel(angular_z=angular_speed)
    
    def start_spinning(self):
        """Start the spin loop in a thread."""
        if self._running:
            return
        
        self._running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()
    
    def stop_spinning(self):
        """Stop the spin loop."""
        self._running = False
    
    def _spin_loop(self):
        """Main spin loop."""
        while self._running:
            self._bridge.spin_once(timeout_sec=0.1)
            time.sleep(0.01)
    
    def _infer_message_type(self, topic: str) -> str:
        """Infer message type from topic name."""
        topic_lower = topic.lower()
        
        if "cmd_vel" in topic_lower or "velocity" in topic_lower:
            return MessageType.TWIST.value
        elif "odom" in topic_lower:
            return MessageType.ODOMETRY.value
        elif "scan" in topic_lower or "laser" in topic_lower:
            return MessageType.LASER_SCAN.value
        elif "image" in topic_lower or "camera" in topic_lower:
            return MessageType.IMAGE.value
        elif "pose" in topic_lower:
            return MessageType.POSE.value
        elif "path" in topic_lower:
            return MessageType.PATH.value
        elif "joint" in topic_lower:
            return MessageType.JOINT_STATE.value
        else:
            return MessageType.STRING.value
    
    def _message_to_dict(self, msg: Any) -> Dict[str, Any]:
        """Convert ROS2 message to dict."""
        if isinstance(msg, dict):
            return msg
        
        # Try to convert ROS message to dict
        try:
            if hasattr(msg, 'get_fields_and_field_types'):
                result = {}
                for field_name in msg.get_fields_and_field_types():
                    value = getattr(msg, field_name)
                    if hasattr(value, 'get_fields_and_field_types'):
                        result[field_name] = self._message_to_dict(value)
                    else:
                        result[field_name] = value
                return result
        except Exception as e:
            logger.debug(f"Message conversion failed: {e}")
        
        return {"data": str(msg)}
    
    def get_topic_info(self, topic: str) -> Optional[TopicInfo]:
        """Get information about a topic."""
        return self._topics.get(topic)
    
    def list_topics(self) -> List[str]:
        """List all known topics."""
        return list(self._topics.keys())
    
    def shutdown(self):
        """Shutdown the node."""
        self.stop_spinning()
        self._bridge.shutdown()


class RobotController:
    """High-level robot controller using ROS2."""
    
    def __init__(
        self,
        node: Optional[ROS2Node] = None
    ):
        """
        Initialize controller.
        
        Args:
            node: ROS2 node to use
        """
        self._node = node or ROS2Node("robot_controller")
        
        # State
        self._position: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._orientation: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        self._velocity: Dict[str, float] = {"linear": 0.0, "angular": 0.0}
        
        # Subscribe to odometry
        self._node.subscribe("/odom", self._odom_callback)
    
    def _odom_callback(self, msg: Dict[str, Any]):
        """Handle odometry updates."""
        odom = MessageConverter.from_odometry(msg)
        self._position = odom["position"]
        self._orientation = odom["orientation"]
        self._velocity = {
            "linear": odom["linear_velocity"].get("x", 0.0),
            "angular": odom["angular_velocity"].get("z", 0.0)
        }
    
    def get_position(self) -> Tuple[float, float, float]:
        """Get current position."""
        return (
            self._position.get("x", 0.0),
            self._position.get("y", 0.0),
            self._position.get("z", 0.0)
        )
    
    def get_heading(self) -> float:
        """Get current heading (yaw) in radians."""
        import math
        
        q = self._orientation
        # Convert quaternion to yaw
        siny_cosp = 2 * (q.get("w", 1) * q.get("z", 0) + q.get("x", 0) * q.get("y", 0))
        cosy_cosp = 1 - 2 * (q.get("y", 0)**2 + q.get("z", 0)**2)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def move_to(
        self,
        x: float,
        y: float,
        speed: float = 0.5
    ):
        """Move to a position (simple implementation)."""
        import math
        
        current_x, current_y, _ = self.get_position()
        
        dx = x - current_x
        dy = y - current_y
        
        distance = math.sqrt(dx**2 + dy**2)
        target_heading = math.atan2(dy, dx)
        
        # Turn to target
        current_heading = self.get_heading()
        angle_diff = target_heading - current_heading
        
        if abs(angle_diff) > 0.1:
            self._node.turn(0.5 if angle_diff > 0 else -0.5)
            time.sleep(abs(angle_diff) / 0.5)
        
        # Move forward
        if distance > 0.1:
            self._node.move_forward(speed)
            time.sleep(distance / speed)
        
        self._node.stop_robot()
    
    def stop(self):
        """Stop the robot."""
        self._node.stop_robot()


# Global instance
_node: Optional[ROS2Node] = None


def get_ros2_node(
    node_name: str = "enigma_ai",
    use_mock: bool = False
) -> ROS2Node:
    """Get or create global ROS2 node."""
    global _node
    if _node is None:
        _node = ROS2Node(node_name, use_mock)
    return _node
