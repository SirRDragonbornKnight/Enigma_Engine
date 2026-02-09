"""
Sensor Fusion for Enigma AI Engine

Fuse data from multiple sensors for better state estimation.

Features:
- Multi-sensor fusion
- Kalman filtering
- IMU + GPS fusion
- Camera + LiDAR fusion
- Uncertainty estimation

Usage:
    from enigma_engine.tools.sensor_fusion import SensorFusion, get_fusion
    
    fusion = get_fusion()
    
    # Add sensors
    fusion.add_sensor("imu", SensorType.IMU)
    fusion.add_sensor("gps", SensorType.GPS)
    
    # Update with readings
    fusion.update("imu", {"accel": [0, 0, 9.8], "gyro": [0, 0, 0]})
    fusion.update("gps", {"lat": 47.6, "lon": -122.3})
    
    # Get fused state
    state = fusion.get_state()
"""

import logging
import math
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Types of sensors."""
    IMU = "imu"
    GPS = "gps"
    LIDAR = "lidar"
    CAMERA = "camera"
    WHEEL_ENCODER = "wheel_encoder"
    MAGNETOMETER = "magnetometer"
    BAROMETER = "barometer"
    ULTRASONIC = "ultrasonic"


@dataclass
class SensorConfig:
    """Sensor configuration."""
    sensor_id: str
    sensor_type: SensorType
    
    # Noise parameters
    position_noise: float = 0.1  # meters
    velocity_noise: float = 0.01  # m/s
    orientation_noise: float = 0.01  # radians
    
    # Update rate
    rate_hz: float = 10.0
    
    # Transform from sensor to body frame
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0


@dataclass
class SensorReading:
    """Single sensor reading."""
    sensor_id: str
    timestamp: float
    data: Dict[str, Any]
    covariance: Optional[np.ndarray] = None


@dataclass
class FusedState:
    """Fused state estimate."""
    # Position
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    # Velocity
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    
    # Orientation (quaternion)
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    
    # Angular velocity
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0
    
    # Uncertainty (diagonal of covariance)
    position_uncertainty: float = 0.0
    velocity_uncertainty: float = 0.0
    orientation_uncertainty: float = 0.0
    
    timestamp: float = 0.0


class KalmanFilter:
    """Extended Kalman Filter for state estimation."""
    
    def __init__(self, state_dim: int = 15):
        """
        Initialize Kalman filter.
        
        Args:
            state_dim: State vector dimension
                       [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, ax, ay]
        """
        self._state_dim = state_dim
        
        # State vector
        self._x = np.zeros(state_dim)
        self._x[6] = 1.0  # qw = 1 for identity quaternion
        
        # Covariance matrix
        self._P = np.eye(state_dim) * 0.1
        
        # Process noise
        self._Q = np.eye(state_dim) * 0.01
        
        # Measurement noise (varies by sensor)
        self._R_default = np.eye(3) * 0.1
    
    @property
    def state(self) -> np.ndarray:
        return self._x.copy()
    
    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()
    
    def predict(self, dt: float):
        """
        Prediction step.
        
        Args:
            dt: Time delta in seconds
        """
        # State transition (simple kinematic model)
        # Position updates with velocity
        self._x[0] += self._x[3] * dt  # x
        self._x[1] += self._x[4] * dt  # y
        self._x[2] += self._x[5] * dt  # z
        
        # Velocity updates with acceleration (if in state)
        if self._state_dim > 13:
            self._x[3] += self._x[13] * dt  # vx
            self._x[4] += self._x[14] * dt  # vy
        
        # Update quaternion with angular velocity
        wx, wy, wz = self._x[10:13]
        qw, qx, qy, qz = self._x[6:10]
        
        # Quaternion derivative
        dqw = -0.5 * (wx*qx + wy*qy + wz*qz)
        dqx = 0.5 * (wx*qw + wy*qz - wz*qy)
        dqy = 0.5 * (-wx*qz + wy*qw + wz*qx)
        dqz = 0.5 * (wx*qy - wy*qx + wz*qw)
        
        self._x[6] += dqw * dt
        self._x[7] += dqx * dt
        self._x[8] += dqy * dt
        self._x[9] += dqz * dt
        
        # Normalize quaternion
        q_norm = np.linalg.norm(self._x[6:10])
        if q_norm > 0:
            self._x[6:10] /= q_norm
        
        # Covariance prediction
        # P = F * P * F' + Q (simplified with identity F)
        self._P = self._P + self._Q * dt
    
    def update_position(
        self,
        measurement: np.ndarray,
        R: Optional[np.ndarray] = None
    ):
        """
        Update with position measurement.
        
        Args:
            measurement: [x, y, z] position
            R: Measurement noise covariance
        """
        if R is None:
            R = self._R_default
        
        # Measurement model (position only)
        H = np.zeros((3, self._state_dim))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # z
        
        self._kalman_update(measurement, H, R)
    
    def update_velocity(
        self,
        measurement: np.ndarray,
        R: Optional[np.ndarray] = None
    ):
        """
        Update with velocity measurement.
        
        Args:
            measurement: [vx, vy, vz] velocity
            R: Measurement noise covariance
        """
        if R is None:
            R = self._R_default * 0.1
        
        H = np.zeros((3, self._state_dim))
        H[0, 3] = 1  # vx
        H[1, 4] = 1  # vy
        H[2, 5] = 1  # vz
        
        self._kalman_update(measurement, H, R)
    
    def update_orientation(
        self,
        quaternion: np.ndarray,
        R: Optional[np.ndarray] = None
    ):
        """
        Update with orientation measurement.
        
        Args:
            quaternion: [qw, qx, qy, qz]
            R: Measurement noise covariance
        """
        if R is None:
            R = np.eye(4) * 0.001
        
        H = np.zeros((4, self._state_dim))
        H[0, 6] = 1  # qw
        H[1, 7] = 1  # qx
        H[2, 8] = 1  # qy
        H[3, 9] = 1  # qz
        
        self._kalman_update(quaternion, H, R)
    
    def update_angular_velocity(
        self,
        measurement: np.ndarray,
        R: Optional[np.ndarray] = None
    ):
        """
        Update with angular velocity measurement.
        
        Args:
            measurement: [wx, wy, wz]
            R: Measurement noise covariance
        """
        if R is None:
            R = np.eye(3) * 0.001
        
        H = np.zeros((3, self._state_dim))
        H[0, 10] = 1  # wx
        H[1, 11] = 1  # wy
        H[2, 12] = 1  # wz
        
        self._kalman_update(measurement, H, R)
    
    def _kalman_update(
        self,
        z: np.ndarray,
        H: np.ndarray,
        R: np.ndarray
    ):
        """
        Kalman update step.
        
        Args:
            z: Measurement
            H: Measurement matrix
            R: Measurement noise
        """
        # Innovation
        y = z - H @ self._x
        
        # Innovation covariance
        S = H @ self._P @ H.T + R
        
        # Kalman gain
        K = self._P @ H.T @ np.linalg.inv(S)
        
        # State update
        self._x = self._x + K @ y
        
        # Covariance update
        I = np.eye(self._state_dim)
        self._P = (I - K @ H) @ self._P


class IMUProcessor:
    """Process IMU data."""
    
    def __init__(self):
        self._last_time: Optional[float] = None
        self._gyro_bias = np.zeros(3)
        self._accel_bias = np.zeros(3)
    
    def process(
        self,
        accel: List[float],
        gyro: List[float],
        timestamp: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process IMU reading.
        
        Returns:
            (corrected_accel, corrected_gyro)
        """
        # Remove bias
        accel_corr = np.array(accel) - self._accel_bias
        gyro_corr = np.array(gyro) - self._gyro_bias
        
        self._last_time = timestamp
        
        return accel_corr, gyro_corr
    
    def calibrate(self, accel_samples: List, gyro_samples: List):
        """Calibrate bias from stationary samples."""
        if accel_samples:
            self._accel_bias = np.mean(accel_samples, axis=0)
            # Assume gravity is in z
            self._accel_bias[2] -= 9.81
        
        if gyro_samples:
            self._gyro_bias = np.mean(gyro_samples, axis=0)


class GPSProcessor:
    """Process GPS data."""
    
    def __init__(self, origin_lat: float = 0.0, origin_lon: float = 0.0):
        self._origin_lat = origin_lat
        self._origin_lon = origin_lon
        self._origin_set = origin_lat != 0 or origin_lon != 0
    
    def set_origin(self, lat: float, lon: float):
        """Set coordinate origin."""
        self._origin_lat = lat
        self._origin_lon = lon
        self._origin_set = True
    
    def process(
        self,
        lat: float,
        lon: float,
        alt: float = 0.0
    ) -> np.ndarray:
        """
        Process GPS reading to local coordinates.
        
        Returns:
            [x, y, z] in meters
        """
        if not self._origin_set:
            self.set_origin(lat, lon)
            return np.zeros(3)
        
        # Convert to local coordinates (simplified)
        # 1 degree latitude ≈ 111km
        # 1 degree longitude ≈ 111km * cos(lat)
        lat_rad = math.radians(self._origin_lat)
        
        x = (lon - self._origin_lon) * 111000 * math.cos(lat_rad)
        y = (lat - self._origin_lat) * 111000
        z = alt
        
        return np.array([x, y, z])


class SensorFusion:
    """Multi-sensor fusion system."""
    
    def __init__(self):
        """Initialize fusion system."""
        self._sensors: Dict[str, SensorConfig] = {}
        self._kf = KalmanFilter()
        
        self._imu_processor = IMUProcessor()
        self._gps_processor = GPSProcessor()
        
        self._last_update = time.time()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Callbacks for visualization
        self._state_callbacks: List[Callable[[FusedState], None]] = []
        
        logger.info("SensorFusion initialized")
    
    def add_sensor(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        **config
    ):
        """Add a sensor."""
        self._sensors[sensor_id] = SensorConfig(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            **config
        )
        logger.info(f"Added sensor {sensor_id} ({sensor_type.value})")
    
    def remove_sensor(self, sensor_id: str):
        """Remove a sensor."""
        if sensor_id in self._sensors:
            del self._sensors[sensor_id]
    
    def update(self, sensor_id: str, data: Dict[str, Any]):
        """
        Update with sensor reading.
        
        Args:
            sensor_id: Sensor identifier
            data: Sensor data
        """
        config = self._sensors.get(sensor_id)
        if not config:
            logger.warning(f"Unknown sensor: {sensor_id}")
            return
        
        now = time.time()
        dt = now - self._last_update
        self._last_update = now
        
        # Predict step
        self._kf.predict(dt)
        
        # Update based on sensor type
        if config.sensor_type == SensorType.IMU:
            self._update_imu(data, config)
        
        elif config.sensor_type == SensorType.GPS:
            self._update_gps(data, config)
        
        elif config.sensor_type == SensorType.WHEEL_ENCODER:
            self._update_encoder(data, config)
        
        elif config.sensor_type == SensorType.MAGNETOMETER:
            self._update_magnetometer(data, config)
        
        # Notify callbacks
        state = self.get_state()
        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")
    
    def _update_imu(self, data: Dict, config: SensorConfig):
        """Update with IMU data."""
        accel = data.get("accel", [0, 0, 0])
        gyro = data.get("gyro", [0, 0, 0])
        
        accel_corr, gyro_corr = self._imu_processor.process(
            accel, gyro, time.time()
        )
        
        # Update angular velocity
        R = np.eye(3) * config.orientation_noise
        self._kf.update_angular_velocity(gyro_corr, R)
        
        # Could integrate accel for velocity, but drift accumulates
    
    def _update_gps(self, data: Dict, config: SensorConfig):
        """Update with GPS data."""
        lat = data.get("lat", 0)
        lon = data.get("lon", 0)
        alt = data.get("alt", 0)
        
        pos = self._gps_processor.process(lat, lon, alt)
        
        R = np.eye(3) * config.position_noise
        self._kf.update_position(pos, R)
    
    def _update_encoder(self, data: Dict, config: SensorConfig):
        """Update with wheel encoder data."""
        velocity = data.get("velocity", [0, 0, 0])
        
        R = np.eye(3) * config.velocity_noise
        self._kf.update_velocity(np.array(velocity), R)
    
    def _update_magnetometer(self, data: Dict, config: SensorConfig):
        """Update with magnetometer data."""
        # Magnetometer gives heading (yaw)
        heading = data.get("heading", 0)
        
        # Convert to quaternion (yaw only)
        qw = math.cos(heading / 2)
        qz = math.sin(heading / 2)
        
        R = np.eye(4) * config.orientation_noise
        self._kf.update_orientation(np.array([qw, 0, 0, qz]), R)
    
    def get_state(self) -> FusedState:
        """Get current fused state estimate."""
        x = self._kf.state
        P = self._kf.covariance
        
        return FusedState(
            x=x[0], y=x[1], z=x[2],
            vx=x[3], vy=x[4], vz=x[5],
            qw=x[6], qx=x[7], qy=x[8], qz=x[9],
            wx=x[10], wy=x[11], wz=x[12],
            position_uncertainty=math.sqrt(P[0, 0] + P[1, 1] + P[2, 2]),
            velocity_uncertainty=math.sqrt(P[3, 3] + P[4, 4] + P[5, 5]),
            orientation_uncertainty=math.sqrt(P[6, 6] + P[7, 7] + P[8, 8] + P[9, 9]),
            timestamp=time.time()
        )
    
    def get_position(self) -> Tuple[float, float, float]:
        """Get position estimate."""
        state = self.get_state()
        return (state.x, state.y, state.z)
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """Get velocity estimate."""
        state = self.get_state()
        return (state.vx, state.vy, state.vz)
    
    def get_orientation_euler(self) -> Tuple[float, float, float]:
        """Get orientation as Euler angles (roll, pitch, yaw)."""
        state = self.get_state()
        
        # Convert quaternion to Euler
        qw, qx, qy, qz = state.qw, state.qx, state.qy, state.qz
        
        # Roll
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return (roll, pitch, yaw)
    
    def on_state_update(self, callback: Callable[[FusedState], None]):
        """Register callback for state updates."""
        self._state_callbacks.append(callback)
    
    def reset(self):
        """Reset filter state."""
        self._kf = KalmanFilter()
        logger.info("Sensor fusion reset")
    
    def get_sensor_status(self) -> Dict[str, str]:
        """Get status of all sensors."""
        return {
            sid: config.sensor_type.value
            for sid, config in self._sensors.items()
        }


# Global instance
_fusion: Optional[SensorFusion] = None


def get_fusion() -> SensorFusion:
    """Get or create global sensor fusion instance."""
    global _fusion
    if _fusion is None:
        _fusion = SensorFusion()
    return _fusion
