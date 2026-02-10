"""
Robot Sensor Integration for Enigma AI Engine

Read and process data from robot sensors.

Features:
- Multiple sensor types (camera, lidar, IMU, etc.)
- Sensor fusion
- Real-time processing
- Event detection
- Data logging

Usage:
    from enigma_engine.tools.sensors import SensorHub, CameraSensor, LidarSensor
    
    # Create sensor hub
    hub = SensorHub()
    
    # Add sensors
    hub.add(CameraSensor("front_cam"))
    hub.add(LidarSensor("lidar"))
    
    # Start reading
    hub.start()
    
    # Get fused data
    state = hub.get_state()
"""

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Types of sensors."""
    CAMERA = auto()
    LIDAR = auto()
    IMU = auto()
    GPS = auto()
    ULTRASONIC = auto()
    INFRARED = auto()
    TOUCH = auto()
    TEMPERATURE = auto()
    HUMIDITY = auto()
    PRESSURE = auto()
    MICROPHONE = auto()
    ENCODER = auto()
    CUSTOM = auto()


@dataclass
class SensorConfig:
    """Sensor configuration."""
    name: str
    sensor_type: SensorType
    update_rate: float = 10.0  # Hz
    enabled: bool = True
    
    # Connection
    port: Optional[str] = None
    address: Optional[str] = None
    
    # Processing
    filter_enabled: bool = False
    filter_type: str = "lowpass"
    filter_cutoff: float = 1.0
    
    # Calibration
    offset: List[float] = field(default_factory=list)
    scale: List[float] = field(default_factory=list)


@dataclass
class SensorReading:
    """A single sensor reading."""
    sensor_name: str
    timestamp: float
    data: Dict[str, Any]
    raw_data: Optional[Any] = None
    is_valid: bool = True
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sensor": self.sensor_name,
            "timestamp": self.timestamp,
            "data": self.data,
            "valid": self.is_valid,
            "confidence": self.confidence
        }


@dataclass
class RobotState:
    """Fused robot state from all sensors."""
    timestamp: float
    
    # Position/orientation
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # roll, pitch, yaw
    
    # Motion
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Environment
    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    
    # Raw sensor data
    sensor_data: Dict[str, SensorReading] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "position": self.position,
            "orientation": self.orientation,
            "velocity": self.velocity,
            "obstacles": self.obstacles,
            "objects": self.detected_objects
        }


class Sensor(ABC):
    """Base class for sensors."""
    
    def __init__(self, config: SensorConfig):
        """Initialize sensor."""
        self._config = config
        self._is_running = False
        self._last_reading: Optional[SensorReading] = None
        self._callbacks: List[Callable[[SensorReading], None]] = []
    
    @property
    def name(self) -> str:
        """Get sensor name."""
        return self._config.name
    
    @property
    def sensor_type(self) -> SensorType:
        """Get sensor type."""
        return self._config.sensor_type
    
    @abstractmethod
    def read(self) -> SensorReading:
        """Read sensor data."""
    
    def start(self):
        """Start sensor."""
        self._is_running = True
        logger.info(f"Started sensor: {self.name}")
    
    def stop(self):
        """Stop sensor."""
        self._is_running = False
        logger.info(f"Stopped sensor: {self.name}")
    
    def calibrate(self) -> bool:
        """Calibrate sensor."""
        logger.info(f"Calibrating sensor: {self.name}")
        return True
    
    def on_reading(self, callback: Callable[[SensorReading], None]):
        """Register callback for new readings."""
        self._callbacks.append(callback)
    
    def _notify(self, reading: SensorReading):
        """Notify callbacks of new reading."""
        self._last_reading = reading
        for cb in self._callbacks:
            try:
                cb(reading)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    @property
    def last_reading(self) -> Optional[SensorReading]:
        """Get last reading."""
        return self._last_reading
    
    @property
    def is_running(self) -> bool:
        """Check if sensor is running."""
        return self._is_running


class CameraSensor(Sensor):
    """Camera/vision sensor."""
    
    def __init__(self, name: str, device_id: int = 0):
        """
        Initialize camera sensor.
        
        Args:
            name: Sensor name
            device_id: Camera device ID
        """
        super().__init__(SensorConfig(
            name=name,
            sensor_type=SensorType.CAMERA
        ))
        self._device_id = device_id
        self._capture = None
    
    def start(self):
        """Start camera."""
        try:
            import cv2
            self._capture = cv2.VideoCapture(self._device_id)
            super().start()
        except ImportError:
            logger.warning("OpenCV not available, using mock camera")
            super().start()
    
    def stop(self):
        """Stop camera."""
        if self._capture:
            self._capture.release()
        super().stop()
    
    def read(self) -> SensorReading:
        """Read camera frame."""
        data = {
            "has_frame": False,
            "width": 0,
            "height": 0,
            "frame": None
        }
        
        if self._capture and self._capture.isOpened():
            ret, frame = self._capture.read()
            if ret:
                data["has_frame"] = True
                data["height"], data["width"] = frame.shape[:2]
                data["frame"] = frame
        
        reading = SensorReading(
            sensor_name=self.name,
            timestamp=time.time(),
            data=data,
            is_valid=data["has_frame"]
        )
        
        self._notify(reading)
        return reading


class LidarSensor(Sensor):
    """LIDAR sensor for distance measurement."""
    
    def __init__(self, name: str, port: Optional[str] = None):
        """
        Initialize LIDAR sensor.
        
        Args:
            name: Sensor name
            port: Serial port
        """
        super().__init__(SensorConfig(
            name=name,
            sensor_type=SensorType.LIDAR,
            port=port
        ))
        self._serial = None
    
    def read(self) -> SensorReading:
        """Read LIDAR scan."""
        # Mock data for now
        import math
        import random
        
        # Generate mock scan (360 points)
        points = []
        for angle in range(360):
            distance = 2.0 + random.uniform(-0.5, 0.5)
            rad = math.radians(angle)
            x = distance * math.cos(rad)
            y = distance * math.sin(rad)
            points.append({"angle": angle, "distance": distance, "x": x, "y": y})
        
        data = {
            "points": points,
            "min_distance": min(p["distance"] for p in points),
            "max_distance": max(p["distance"] for p in points)
        }
        
        reading = SensorReading(
            sensor_name=self.name,
            timestamp=time.time(),
            data=data
        )
        
        self._notify(reading)
        return reading


class IMUSensor(Sensor):
    """IMU (Inertial Measurement Unit) sensor."""
    
    def __init__(self, name: str):
        """Initialize IMU sensor."""
        super().__init__(SensorConfig(
            name=name,
            sensor_type=SensorType.IMU
        ))
    
    def read(self) -> SensorReading:
        """Read IMU data."""
        import random
        
        # Mock IMU data
        data = {
            "accelerometer": {
                "x": random.uniform(-0.1, 0.1),
                "y": random.uniform(-0.1, 0.1),
                "z": 9.8 + random.uniform(-0.1, 0.1)
            },
            "gyroscope": {
                "x": random.uniform(-0.01, 0.01),
                "y": random.uniform(-0.01, 0.01),
                "z": random.uniform(-0.01, 0.01)
            },
            "magnetometer": {
                "x": random.uniform(-1, 1),
                "y": random.uniform(-1, 1),
                "z": random.uniform(-1, 1)
            }
        }
        
        reading = SensorReading(
            sensor_name=self.name,
            timestamp=time.time(),
            data=data
        )
        
        self._notify(reading)
        return reading


class GPSSensor(Sensor):
    """GPS sensor."""
    
    def __init__(self, name: str):
        """Initialize GPS sensor."""
        super().__init__(SensorConfig(
            name=name,
            sensor_type=SensorType.GPS
        ))
    
    def read(self) -> SensorReading:
        """Read GPS data."""
        # Mock GPS data
        data = {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "altitude": 10.0,
            "speed": 0.0,
            "heading": 0.0,
            "satellites": 8,
            "fix_quality": 1
        }
        
        reading = SensorReading(
            sensor_name=self.name,
            timestamp=time.time(),
            data=data
        )
        
        self._notify(reading)
        return reading


class UltrasonicSensor(Sensor):
    """Ultrasonic distance sensor."""
    
    def __init__(self, name: str, trigger_pin: int = 0, echo_pin: int = 0):
        """Initialize ultrasonic sensor."""
        super().__init__(SensorConfig(
            name=name,
            sensor_type=SensorType.ULTRASONIC
        ))
        self._trigger_pin = trigger_pin
        self._echo_pin = echo_pin
    
    def read(self) -> SensorReading:
        """Read distance."""
        import random
        
        # Mock distance
        distance = 50.0 + random.uniform(-5, 5)  # cm
        
        data = {
            "distance_cm": distance,
            "distance_m": distance / 100
        }
        
        reading = SensorReading(
            sensor_name=self.name,
            timestamp=time.time(),
            data=data
        )
        
        self._notify(reading)
        return reading


class SensorHub:
    """
    Central hub for managing multiple sensors.
    """
    
    def __init__(self, log_data: bool = False):
        """
        Initialize sensor hub.
        
        Args:
            log_data: Whether to log all sensor data
        """
        self._sensors: Dict[str, Sensor] = {}
        self._is_running = False
        self._update_thread: Optional[threading.Thread] = None
        self._state = RobotState(timestamp=time.time())
        self._lock = threading.Lock()
        
        self._log_data = log_data
        self._data_log: List[Dict[str, Any]] = []
        
        # Callbacks
        self._state_callbacks: List[Callable[[RobotState], None]] = []
        self._event_callbacks: List[Callable[[str, Any], None]] = []
    
    def add(self, sensor: Sensor):
        """Add a sensor to the hub."""
        self._sensors[sensor.name] = sensor
        sensor.on_reading(self._on_sensor_reading)
        logger.info(f"Added sensor: {sensor.name}")
    
    def remove(self, name: str):
        """Remove a sensor."""
        if name in self._sensors:
            self._sensors[name].stop()
            del self._sensors[name]
    
    def get_sensor(self, name: str) -> Optional[Sensor]:
        """Get sensor by name."""
        return self._sensors.get(name)
    
    def list_sensors(self) -> List[str]:
        """List all sensor names."""
        return list(self._sensors.keys())
    
    def start(self, update_rate: float = 10.0):
        """
        Start all sensors and update loop.
        
        Args:
            update_rate: State update rate (Hz)
        """
        self._is_running = True
        
        # Start all sensors
        for sensor in self._sensors.values():
            sensor.start()
        
        # Start update thread
        self._update_thread = threading.Thread(
            target=self._update_loop,
            args=(update_rate,),
            daemon=True
        )
        self._update_thread.start()
        
        logger.info(f"SensorHub started with {len(self._sensors)} sensors")
    
    def stop(self):
        """Stop all sensors."""
        self._is_running = False
        
        # Stop all sensors
        for sensor in self._sensors.values():
            sensor.stop()
        
        # Wait for thread
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
        
        logger.info("SensorHub stopped")
    
    def _update_loop(self, rate: float):
        """Main update loop."""
        period = 1.0 / rate
        
        while self._is_running:
            start = time.time()
            
            # Read all sensors
            for sensor in self._sensors.values():
                if sensor.is_running:
                    try:
                        sensor.read()
                    except Exception as e:
                        logger.error(f"Sensor read error ({sensor.name}): {e}")
            
            # Update fused state
            self._update_state()
            
            # Sleep for remaining time
            elapsed = time.time() - start
            if elapsed < period:
                time.sleep(period - elapsed)
    
    def _on_sensor_reading(self, reading: SensorReading):
        """Handle new sensor reading."""
        with self._lock:
            self._state.sensor_data[reading.sensor_name] = reading
        
        # Log if enabled
        if self._log_data:
            self._data_log.append(reading.to_dict())
        
        # Check for events
        self._check_events(reading)
    
    def _update_state(self):
        """Update fused robot state."""
        with self._lock:
            self._state.timestamp = time.time()
            
            # Fuse IMU data for orientation
            imu_readings = [
                r for r in self._state.sensor_data.values()
                if r.sensor_name.endswith("imu") or "IMU" in r.sensor_name
            ]
            
            if imu_readings:
                # Average accelerometer readings
                pass  # Placeholder for proper sensor fusion
            
            # Fuse GPS data for position
            gps_readings = [
                r for r in self._state.sensor_data.values()
                if r.sensor_name.endswith("gps") or "GPS" in r.sensor_name
            ]
            
            if gps_readings:
                latest = gps_readings[0]
                self._state.position = (
                    latest.data.get("latitude", 0),
                    latest.data.get("longitude", 0),
                    latest.data.get("altitude", 0)
                )
            
            # Detect obstacles from LIDAR/ultrasonic
            self._state.obstacles = self._detect_obstacles()
        
        # Notify callbacks
        for cb in self._state_callbacks:
            try:
                cb(self._state)
            except Exception as e:
                logger.error(f"State callback error: {e}")
    
    def _detect_obstacles(self) -> List[Dict[str, Any]]:
        """Detect obstacles from sensor data."""
        obstacles = []
        
        # Check LIDAR
        for name, reading in self._state.sensor_data.items():
            if reading.sensor_name.lower().find("lidar") >= 0:
                points = reading.data.get("points", [])
                for p in points:
                    if p["distance"] < 0.5:  # 50cm threshold
                        obstacles.append({
                            "type": "obstacle",
                            "angle": p["angle"],
                            "distance": p["distance"],
                            "source": name
                        })
        
        # Check ultrasonics
        for name, reading in self._state.sensor_data.items():
            if reading.sensor_name.lower().find("ultrasonic") >= 0:
                dist = reading.data.get("distance_m", 999)
                if dist < 0.3:
                    obstacles.append({
                        "type": "obstacle",
                        "distance": dist,
                        "source": name
                    })
        
        return obstacles
    
    def _check_events(self, reading: SensorReading):
        """Check for notable events."""
        events = []
        
        # Check for collision warning
        if reading.sensor_name.lower().find("ultrasonic") >= 0:
            dist = reading.data.get("distance_m", 999)
            if dist < 0.1:
                events.append(("collision_warning", {"distance": dist}))
        
        # Notify event callbacks
        for event_type, data in events:
            for cb in self._event_callbacks:
                try:
                    cb(event_type, data)
                except Exception as e:
                    logger.error(f"Event callback error: {e}")
    
    def get_state(self) -> RobotState:
        """Get current fused state."""
        with self._lock:
            return self._state
    
    def on_state_update(self, callback: Callable[[RobotState], None]):
        """Register state update callback."""
        self._state_callbacks.append(callback)
    
    def on_event(self, callback: Callable[[str, Any], None]):
        """Register event callback."""
        self._event_callbacks.append(callback)
    
    def save_log(self, path: str):
        """Save data log to file."""
        with open(path, "w") as f:
            json.dump(self._data_log, f, indent=2)
        logger.info(f"Saved {len(self._data_log)} readings to {path}")
    
    def get_log(self) -> List[Dict[str, Any]]:
        """Get data log."""
        return self._data_log.copy()


# Convenience function
def create_robot_sensors() -> SensorHub:
    """Create a standard robot sensor setup."""
    hub = SensorHub(log_data=True)
    
    hub.add(CameraSensor("front_camera"))
    hub.add(LidarSensor("lidar"))
    hub.add(IMUSensor("imu"))
    hub.add(UltrasonicSensor("ultrasonic_front"))
    
    return hub
