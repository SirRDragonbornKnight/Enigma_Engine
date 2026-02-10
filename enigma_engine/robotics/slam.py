"""
SLAM Support

Simultaneous Localization and Mapping for robotics.
Provides 2D occupancy grid mapping and localization.

FILE: enigma_engine/robotics/slam.py
TYPE: Robotics
MAIN CLASSES: SLAMEngine, OccupancyGrid, ParticleFilter
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CellState(Enum):
    """Occupancy grid cell state."""
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 100


@dataclass
class Pose2D:
    """2D robot pose."""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # radians
    
    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.theta)


@dataclass
class LaserScan:
    """Laser scan data."""
    ranges: list[float]
    angle_min: float = -math.pi
    angle_max: float = math.pi
    angle_increment: float = 0.01
    range_min: float = 0.1
    range_max: float = 30.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SLAMConfig:
    """SLAM configuration."""
    # Map parameters
    map_width: int = 100  # cells
    map_height: int = 100
    resolution: float = 0.05  # meters per cell
    origin_x: float = -2.5  # map origin in meters
    origin_y: float = -2.5
    
    # Update parameters
    hit_prob: float = 0.7  # Probability of hit
    miss_prob: float = 0.4  # Probability of miss
    max_range: float = 10.0  # Maximum sensor range
    
    # Particle filter parameters
    num_particles: int = 100
    motion_noise: tuple[float, float, float] = (0.05, 0.05, 0.02)
    resample_threshold: float = 0.5


if HAS_NUMPY:
    
    class OccupancyGrid:
        """
        2D occupancy grid map.
        """
        
        def __init__(self, config: SLAMConfig = None):
            self.config = config or SLAMConfig()
            
            # Initialize grid with unknown
            self.grid = np.full(
                (self.config.map_height, self.config.map_width),
                CellState.UNKNOWN.value,
                dtype=np.int8
            )
            
            # Log-odds representation for updates
            self.log_odds = np.zeros(
                (self.config.map_height, self.config.map_width),
                dtype=np.float32
            )
            
            # Precompute log-odds probabilities
            self._log_hit = math.log(self.config.hit_prob / (1 - self.config.hit_prob))
            self._log_miss = math.log(self.config.miss_prob / (1 - self.config.miss_prob))
        
        def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
            """Convert world coordinates to grid coordinates."""
            gx = int((x - self.config.origin_x) / self.config.resolution)
            gy = int((y - self.config.origin_y) / self.config.resolution)
            return (gx, gy)
        
        def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
            """Convert grid coordinates to world coordinates."""
            x = gx * self.config.resolution + self.config.origin_x
            y = gy * self.config.resolution + self.config.origin_y
            return (x, y)
        
        def in_bounds(self, gx: int, gy: int) -> bool:
            """Check if grid coordinates are in bounds."""
            return 0 <= gx < self.config.map_width and 0 <= gy < self.config.map_height
        
        def update_from_scan(self, pose: Pose2D, scan: LaserScan):
            """
            Update map from laser scan.
            
            Args:
                pose: Robot pose
                scan: Laser scan data
            """
            robot_gx, robot_gy = self.world_to_grid(pose.x, pose.y)
            
            angle = scan.angle_min
            
            for range_val in scan.ranges:
                # Skip invalid ranges
                if range_val < scan.range_min or range_val > min(scan.range_max, self.config.max_range):
                    angle += scan.angle_increment
                    continue
                
                # Calculate endpoint
                world_angle = pose.theta + angle
                end_x = pose.x + range_val * math.cos(world_angle)
                end_y = pose.y + range_val * math.sin(world_angle)
                
                end_gx, end_gy = self.world_to_grid(end_x, end_y)
                
                # Trace ray and mark free cells
                ray_cells = self._bresenham(robot_gx, robot_gy, end_gx, end_gy)
                
                for gx, gy in ray_cells[:-1]:  # All cells except last are free
                    if self.in_bounds(gx, gy):
                        self.log_odds[gy, gx] += self._log_miss
                
                # Last cell is occupied
                if self.in_bounds(end_gx, end_gy):
                    self.log_odds[end_gy, end_gx] += self._log_hit
                
                angle += scan.angle_increment
            
            # Convert log-odds to occupancy values
            self._update_grid()
        
        def _bresenham(
            self,
            x0: int, y0: int,
            x1: int, y1: int
        ) -> list[tuple[int, int]]:
            """Bresenham's line algorithm for ray tracing."""
            cells = []
            
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            
            err = dx - dy
            
            x, y = x0, y0
            
            while True:
                cells.append((x, y))
                
                if x == x1 and y == y1:
                    break
                
                e2 = 2 * err
                
                if e2 > -dy:
                    err -= dy
                    x += sx
                
                if e2 < dx:
                    err += dx
                    y += sy
            
            return cells
        
        def _update_grid(self):
            """Update grid from log-odds."""
            # Clip log-odds to prevent overflow
            self.log_odds = np.clip(self.log_odds, -10, 10)
            
            # Convert to probability
            prob = 1 - 1 / (1 + np.exp(self.log_odds))
            
            # Update grid
            self.grid = np.where(
                self.log_odds > 0.5,
                CellState.OCCUPIED.value,
                np.where(
                    self.log_odds < -0.5,
                    CellState.FREE.value,
                    CellState.UNKNOWN.value
                )
            ).astype(np.int8)
        
        def get_cell(self, x: float, y: float) -> int:
            """Get cell value at world coordinates."""
            gx, gy = self.world_to_grid(x, y)
            if self.in_bounds(gx, gy):
                return self.grid[gy, gx]
            return CellState.UNKNOWN.value
        
        def get_map_data(self) -> np.ndarray:
            """Get occupancy grid data."""
            return self.grid.copy()
        
        def save(self, filepath: str):
            """Save map to file."""
            np.savez(
                filepath,
                grid=self.grid,
                log_odds=self.log_odds,
                config={
                    "width": self.config.map_width,
                    "height": self.config.map_height,
                    "resolution": self.config.resolution,
                    "origin_x": self.config.origin_x,
                    "origin_y": self.config.origin_y
                }
            )
        
        def load(self, filepath: str):
            """Load map from file."""
            data = np.load(filepath, allow_pickle=True)
            self.grid = data["grid"]
            self.log_odds = data["log_odds"]
    
    
    class Particle:
        """Particle for particle filter localization."""
        
        def __init__(self, x: float, y: float, theta: float, weight: float = 1.0):
            self.pose = Pose2D(x, y, theta)
            self.weight = weight
    
    
    class ParticleFilter:
        """
        Particle filter for robot localization.
        """
        
        def __init__(
            self,
            occupancy_grid: OccupancyGrid,
            config: SLAMConfig = None
        ):
            self.grid = occupancy_grid
            self.config = config or SLAMConfig()
            
            self.particles: list[Particle] = []
            self._initialize_particles()
        
        def _initialize_particles(self):
            """Initialize particles uniformly."""
            self.particles = []
            
            for _ in range(self.config.num_particles):
                x = random.uniform(
                    self.config.origin_x,
                    self.config.origin_x + self.config.map_width * self.config.resolution
                )
                y = random.uniform(
                    self.config.origin_y,
                    self.config.origin_y + self.config.map_height * self.config.resolution
                )
                theta = random.uniform(-math.pi, math.pi)
                
                self.particles.append(Particle(x, y, theta))
        
        def initialize_at_pose(self, pose: Pose2D, spread: float = 0.5):
            """Initialize particles around a known pose."""
            self.particles = []
            
            for _ in range(self.config.num_particles):
                x = pose.x + random.gauss(0, spread)
                y = pose.y + random.gauss(0, spread)
                theta = pose.theta + random.gauss(0, spread / 2)
                
                self.particles.append(Particle(x, y, theta))
        
        def predict(self, delta_x: float, delta_y: float, delta_theta: float):
            """
            Motion model prediction step.
            
            Args:
                delta_x: Relative x movement
                delta_y: Relative y movement
                delta_theta: Relative rotation
            """
            noise = self.config.motion_noise
            
            for p in self.particles:
                # Add noise to motion
                noisy_dx = delta_x + random.gauss(0, noise[0])
                noisy_dy = delta_y + random.gauss(0, noise[1])
                noisy_dtheta = delta_theta + random.gauss(0, noise[2])
                
                # Apply motion in robot frame
                cos_t = math.cos(p.pose.theta)
                sin_t = math.sin(p.pose.theta)
                
                p.pose.x += noisy_dx * cos_t - noisy_dy * sin_t
                p.pose.y += noisy_dx * sin_t + noisy_dy * cos_t
                p.pose.theta += noisy_dtheta
                
                # Normalize angle
                p.pose.theta = math.atan2(
                    math.sin(p.pose.theta),
                    math.cos(p.pose.theta)
                )
        
        def update(self, scan: LaserScan):
            """
            Sensor model update step.
            
            Args:
                scan: Laser scan data
            """
            for p in self.particles:
                p.weight = self._compute_weight(p, scan)
            
            # Normalize weights
            total_weight = sum(p.weight for p in self.particles)
            if total_weight > 0:
                for p in self.particles:
                    p.weight /= total_weight
        
        def _compute_weight(self, particle: Particle, scan: LaserScan) -> float:
            """Compute particle weight based on scan match."""
            weight = 1.0
            angle = scan.angle_min
            
            # Sample subset of rays for efficiency
            step = max(1, len(scan.ranges) // 20)
            
            for i in range(0, len(scan.ranges), step):
                range_val = scan.ranges[i]
                
                if range_val < scan.range_min or range_val > scan.range_max:
                    angle += scan.angle_increment * step
                    continue
                
                # Expected range from map
                world_angle = particle.pose.theta + angle
                expected_range = self._ray_cast(particle.pose, world_angle)
                
                # Gaussian likelihood
                diff = range_val - expected_range
                sigma = 0.5
                likelihood = math.exp(-(diff * diff) / (2 * sigma * sigma))
                
                weight *= likelihood
                angle += scan.angle_increment * step
            
            return max(weight, 1e-10)
        
        def _ray_cast(self, pose: Pose2D, angle: float) -> float:
            """Cast ray from pose at angle and return distance to obstacle."""
            max_range = self.config.max_range
            step = self.config.resolution
            
            x, y = pose.x, pose.y
            dx = math.cos(angle) * step
            dy = math.sin(angle) * step
            
            distance = 0
            while distance < max_range:
                x += dx
                y += dy
                distance += step
                
                cell = self.grid.get_cell(x, y)
                if cell == CellState.OCCUPIED.value:
                    return distance
            
            return max_range
        
        def resample(self):
            """Resample particles based on weights."""
            # Calculate effective sample size
            weights = [p.weight for p in self.particles]
            n_eff = 1.0 / sum(w * w for w in weights)
            
            # Only resample if ESS is low
            if n_eff > self.config.num_particles * self.config.resample_threshold:
                return
            
            # Low variance resampling
            new_particles = []
            step = 1.0 / self.config.num_particles
            r = random.uniform(0, step)
            
            cumsum = weights[0]
            j = 0
            
            for i in range(self.config.num_particles):
                u = r + i * step
                
                while cumsum < u and j < len(self.particles) - 1:
                    j += 1
                    cumsum += weights[j]
                
                p = self.particles[j]
                new_particles.append(Particle(
                    p.pose.x,
                    p.pose.y,
                    p.pose.theta,
                    1.0 / self.config.num_particles
                ))
            
            self.particles = new_particles
        
        def get_estimate(self) -> Pose2D:
            """Get weighted mean pose estimate."""
            x = sum(p.pose.x * p.weight for p in self.particles)
            y = sum(p.pose.y * p.weight for p in self.particles)
            
            # Average angle using circular mean
            sin_sum = sum(math.sin(p.pose.theta) * p.weight for p in self.particles)
            cos_sum = sum(math.cos(p.pose.theta) * p.weight for p in self.particles)
            theta = math.atan2(sin_sum, cos_sum)
            
            return Pose2D(x, y, theta)
    
    
    class SLAMEngine:
        """
        Main SLAM engine combining mapping and localization.
        """
        
        def __init__(self, config: SLAMConfig = None):
            self.config = config or SLAMConfig()
            
            self.map = OccupancyGrid(config)
            self.localization = ParticleFilter(self.map, config)
            
            self._current_pose = Pose2D()
            self._last_odom = None
        
        def initialize(self, x: float = 0, y: float = 0, theta: float = 0):
            """Initialize SLAM at a known pose."""
            self._current_pose = Pose2D(x, y, theta)
            self.localization.initialize_at_pose(self._current_pose)
        
        def process_odom(self, x: float, y: float, theta: float):
            """
            Process odometry update.
            
            Args:
                x, y, theta: Current odometry pose
            """
            if self._last_odom is not None:
                # Calculate relative motion
                dx = x - self._last_odom[0]
                dy = y - self._last_odom[1]
                dtheta = theta - self._last_odom[2]
                
                # Predict particle filter
                self.localization.predict(dx, dy, dtheta)
            
            self._last_odom = (x, y, theta)
        
        def process_scan(self, scan: LaserScan):
            """
            Process laser scan for mapping and localization.
            
            Args:
                scan: Laser scan data
            """
            # Update localization
            self.localization.update(scan)
            self.localization.resample()
            
            # Get estimated pose
            self._current_pose = self.localization.get_estimate()
            
            # Update map
            self.map.update_from_scan(self._current_pose, scan)
        
        def get_pose(self) -> Pose2D:
            """Get current estimated pose."""
            return self._current_pose
        
        def get_map(self) -> OccupancyGrid:
            """Get current map."""
            return self.map
        
        def save_map(self, filepath: str):
            """Save current map."""
            self.map.save(filepath)

else:
    class OccupancyGrid:
        pass
    
    class ParticleFilter:
        pass
    
    class SLAMEngine:
        pass


def create_slam_engine(**config_kwargs) -> 'SLAMEngine':
    """Create SLAM engine with configuration."""
    if not HAS_NUMPY:
        raise ImportError("NumPy required for SLAM")
    
    config = SLAMConfig(**config_kwargs)
    return SLAMEngine(config)
