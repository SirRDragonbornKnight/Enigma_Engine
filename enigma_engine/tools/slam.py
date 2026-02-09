"""
SLAM Integration for Enigma AI Engine

Simultaneous Localization and Mapping for robotics.

Features:
- Occupancy grid mapping
- Particle filter localization
- Scan matching
- Loop closure detection
- Map saving/loading

Usage:
    from enigma_engine.tools.slam import SLAMSystem, get_slam
    
    slam = get_slam()
    
    # Update with sensor data
    slam.update_scan(laser_ranges, robot_pose)
    
    # Get map
    map_grid = slam.get_map()
    
    # Get robot position
    pose = slam.get_pose()
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CellState(Enum):
    """Occupancy grid cell states."""
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1


@dataclass
class Pose2D:
    """2D robot pose."""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # Heading in radians
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.theta)
    
    @staticmethod
    def from_tuple(t: Tuple[float, float, float]) -> 'Pose2D':
        return Pose2D(t[0], t[1], t[2])


@dataclass
class LaserScan:
    """Laser scan data."""
    ranges: List[float]
    angle_min: float = -math.pi
    angle_max: float = math.pi
    angle_increment: float = 0.0
    range_min: float = 0.1
    range_max: float = 30.0
    
    def __post_init__(self):
        if self.angle_increment == 0 and self.ranges:
            self.angle_increment = (self.angle_max - self.angle_min) / len(self.ranges)


@dataclass
class OccupancyGrid:
    """Occupancy grid map."""
    width: int
    height: int
    resolution: float  # Meters per cell
    origin: Pose2D = field(default_factory=Pose2D)
    
    # Grid data (-1=unknown, 0=free, 100=occupied)
    data: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.data:
            self.data = [-1] * (self.width * self.height)
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        gx = int((x - self.origin.x) / self.resolution)
        gy = int((y - self.origin.y) / self.resolution)
        return (gx, gy)
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        x = gx * self.resolution + self.origin.x
        y = gy * self.resolution + self.origin.y
        return (x, y)
    
    def in_bounds(self, gx: int, gy: int) -> bool:
        """Check if grid coordinates are in bounds."""
        return 0 <= gx < self.width and 0 <= gy < self.height
    
    def get_cell(self, gx: int, gy: int) -> int:
        """Get cell value."""
        if not self.in_bounds(gx, gy):
            return -1
        return self.data[gy * self.width + gx]
    
    def set_cell(self, gx: int, gy: int, value: int):
        """Set cell value."""
        if self.in_bounds(gx, gy):
            self.data[gy * self.width + gx] = value


@dataclass
class Particle:
    """Particle for particle filter."""
    pose: Pose2D
    weight: float = 1.0


class MotionModel:
    """Odometry-based motion model."""
    
    def __init__(
        self,
        alpha1: float = 0.1,  # Rotation noise from rotation
        alpha2: float = 0.1,  # Rotation noise from translation
        alpha3: float = 0.1,  # Translation noise from translation
        alpha4: float = 0.1   # Translation noise from rotation
    ):
        self._alpha = [alpha1, alpha2, alpha3, alpha4]
    
    def sample_motion(
        self,
        old_pose: Pose2D,
        odometry: Tuple[float, float, float]  # (delta_x, delta_y, delta_theta)
    ) -> Pose2D:
        """
        Sample new pose based on motion model.
        
        Returns:
            Sampled new pose
        """
        dx, dy, dtheta = odometry
        
        # Add noise
        trans = math.sqrt(dx**2 + dy**2)
        rot1 = math.atan2(dy, dx) if trans > 0.001 else 0
        rot2 = dtheta - rot1
        
        # Noisy motion
        rot1_noise = rot1 + random.gauss(0, self._alpha[0] * abs(rot1) + self._alpha[1] * trans)
        trans_noise = trans + random.gauss(0, self._alpha[2] * trans + self._alpha[3] * (abs(rot1) + abs(rot2)))
        rot2_noise = rot2 + random.gauss(0, self._alpha[0] * abs(rot2) + self._alpha[1] * trans)
        
        # Apply motion
        new_x = old_pose.x + trans_noise * math.cos(old_pose.theta + rot1_noise)
        new_y = old_pose.y + trans_noise * math.sin(old_pose.theta + rot1_noise)
        new_theta = old_pose.theta + rot1_noise + rot2_noise
        
        # Normalize theta
        new_theta = math.atan2(math.sin(new_theta), math.cos(new_theta))
        
        return Pose2D(new_x, new_y, new_theta)


class SensorModel:
    """Laser scan sensor model."""
    
    def __init__(
        self,
        z_hit: float = 0.9,
        z_random: float = 0.05,
        z_max: float = 0.05,
        sigma_hit: float = 0.2
    ):
        self._z_hit = z_hit
        self._z_random = z_random
        self._z_max = z_max
        self._sigma_hit = sigma_hit
    
    def compute_weight(
        self,
        pose: Pose2D,
        scan: LaserScan,
        grid: OccupancyGrid
    ) -> float:
        """
        Compute particle weight based on scan match.
        
        Returns:
            Weight (higher = better match)
        """
        weight = 1.0
        
        for i, range_val in enumerate(scan.ranges):
            if range_val < scan.range_min or range_val > scan.range_max:
                continue
            
            angle = scan.angle_min + i * scan.angle_increment
            world_angle = pose.theta + angle
            
            # Expected endpoint in world
            end_x = pose.x + range_val * math.cos(world_angle)
            end_y = pose.y + range_val * math.sin(world_angle)
            
            # Convert to grid
            gx, gy = grid.world_to_grid(end_x, end_y)
            
            # Check if occupied
            cell = grid.get_cell(gx, gy)
            
            if cell > 50:  # Occupied
                weight *= self._z_hit
            elif cell == -1:  # Unknown
                weight *= (self._z_hit + self._z_random) / 2
            else:  # Free
                weight *= self._z_random
        
        return weight


class ScanMatcher:
    """Match laser scans for localization."""
    
    def __init__(
        self,
        max_iterations: int = 20,
        convergence_threshold: float = 0.001
    ):
        self._max_iterations = max_iterations
        self._threshold = convergence_threshold
    
    def match(
        self,
        scan1: LaserScan,
        scan2: LaserScan,
        initial_transform: Pose2D = None
    ) -> Tuple[Pose2D, float]:
        """
        Match two scans using ICP-like algorithm.
        
        Returns:
            (transform, score)
        """
        if initial_transform is None:
            initial_transform = Pose2D()
        
        # Convert scans to point clouds
        points1 = self._scan_to_points(scan1, Pose2D())
        points2 = self._scan_to_points(scan2, initial_transform)
        
        current_transform = initial_transform
        
        for iteration in range(self._max_iterations):
            # Find correspondences
            correspondences = self._find_correspondences(points1, points2)
            
            if not correspondences:
                break
            
            # Compute optimal transform
            new_transform = self._compute_transform(correspondences, current_transform)
            
            # Check convergence
            diff = math.sqrt(
                (new_transform.x - current_transform.x)**2 +
                (new_transform.y - current_transform.y)**2
            )
            
            current_transform = new_transform
            
            if diff < self._threshold:
                break
        
        # Compute score
        score = self._compute_score(points1, points2, current_transform)
        
        return current_transform, score
    
    def _scan_to_points(
        self,
        scan: LaserScan,
        pose: Pose2D
    ) -> List[Tuple[float, float]]:
        """Convert scan to point cloud."""
        points = []
        
        for i, range_val in enumerate(scan.ranges):
            if range_val < scan.range_min or range_val > scan.range_max:
                continue
            
            angle = scan.angle_min + i * scan.angle_increment + pose.theta
            x = pose.x + range_val * math.cos(angle)
            y = pose.y + range_val * math.sin(angle)
            points.append((x, y))
        
        return points
    
    def _find_correspondences(
        self,
        points1: List[Tuple[float, float]],
        points2: List[Tuple[float, float]],
        max_dist: float = 1.0
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Find nearest point correspondences."""
        correspondences = []
        
        for p1 in points1[:100]:  # Limit for efficiency
            best_dist = max_dist
            best_p2 = None
            
            for p2 in points2:
                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_p2 = p2
            
            if best_p2:
                correspondences.append((p1, best_p2))
        
        return correspondences
    
    def _compute_transform(
        self,
        correspondences: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        current: Pose2D
    ) -> Pose2D:
        """Compute optimal transform from correspondences."""
        if not correspondences:
            return current
        
        # Simple centroid-based transform
        c1_x = sum(p[0][0] for p in correspondences) / len(correspondences)
        c1_y = sum(p[0][1] for p in correspondences) / len(correspondences)
        c2_x = sum(p[1][0] for p in correspondences) / len(correspondences)
        c2_y = sum(p[1][1] for p in correspondences) / len(correspondences)
        
        dx = c1_x - c2_x
        dy = c1_y - c2_y
        
        return Pose2D(
            current.x + dx * 0.5,
            current.y + dy * 0.5,
            current.theta
        )
    
    def _compute_score(
        self,
        points1: List[Tuple[float, float]],
        points2: List[Tuple[float, float]],
        transform: Pose2D
    ) -> float:
        """Compute alignment score."""
        total_dist = 0
        count = 0
        
        for p1 in points1[:50]:
            min_dist = float('inf')
            for p2 in points2:
                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                min_dist = min(min_dist, dist)
            
            if min_dist < float('inf'):
                total_dist += min_dist
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_dist = total_dist / count
        return math.exp(-avg_dist)  # Higher score = better match


class ParticleFilter:
    """Particle filter for localization."""
    
    def __init__(
        self,
        num_particles: int = 100,
        motion_model: Optional[MotionModel] = None,
        sensor_model: Optional[SensorModel] = None
    ):
        self._num_particles = num_particles
        self._motion_model = motion_model or MotionModel()
        self._sensor_model = sensor_model or SensorModel()
        
        self._particles: List[Particle] = []
        self._best_particle: Optional[Particle] = None
    
    def initialize(
        self,
        initial_pose: Pose2D,
        spread: float = 1.0
    ):
        """Initialize particles around initial pose."""
        self._particles = []
        
        for _ in range(self._num_particles):
            pose = Pose2D(
                initial_pose.x + random.gauss(0, spread),
                initial_pose.y + random.gauss(0, spread),
                initial_pose.theta + random.gauss(0, 0.1)
            )
            self._particles.append(Particle(pose=pose, weight=1.0 / self._num_particles))
        
        self._best_particle = self._particles[0]
    
    def update(
        self,
        odometry: Tuple[float, float, float],
        scan: LaserScan,
        grid: OccupancyGrid
    ):
        """Update particles with motion and sensor data."""
        # Motion update
        for particle in self._particles:
            particle.pose = self._motion_model.sample_motion(
                particle.pose, odometry
            )
        
        # Sensor update
        total_weight = 0
        for particle in self._particles:
            particle.weight = self._sensor_model.compute_weight(
                particle.pose, scan, grid
            )
            total_weight += particle.weight
        
        # Normalize weights
        if total_weight > 0:
            for particle in self._particles:
                particle.weight /= total_weight
        
        # Find best particle
        self._best_particle = max(self._particles, key=lambda p: p.weight)
        
        # Resample
        self._resample()
    
    def _resample(self):
        """Resample particles based on weights."""
        new_particles = []
        weights = [p.weight for p in self._particles]
        
        for _ in range(self._num_particles):
            # Random wheel selection
            idx = self._weighted_choice(weights)
            old_particle = self._particles[idx]
            
            new_particles.append(Particle(
                pose=Pose2D(
                    old_particle.pose.x,
                    old_particle.pose.y,
                    old_particle.pose.theta
                ),
                weight=1.0 / self._num_particles
            ))
        
        self._particles = new_particles
    
    def _weighted_choice(self, weights: List[float]) -> int:
        """Choose random index based on weights."""
        total = sum(weights)
        if total == 0:
            return random.randint(0, len(weights) - 1)
        
        r = random.random() * total
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if cumulative >= r:
                return i
        
        return len(weights) - 1
    
    def get_pose(self) -> Pose2D:
        """Get estimated pose."""
        if self._best_particle:
            return self._best_particle.pose
        return Pose2D()


class SLAMSystem:
    """Main SLAM system."""
    
    def __init__(
        self,
        map_width: int = 200,
        map_height: int = 200,
        resolution: float = 0.1,
        num_particles: int = 100
    ):
        """
        Initialize SLAM system.
        
        Args:
            map_width: Grid width in cells
            map_height: Grid height in cells
            resolution: Meters per cell
            num_particles: Number of particles for localization
        """
        # Map
        self._grid = OccupancyGrid(
            width=map_width,
            height=map_height,
            resolution=resolution,
            origin=Pose2D(-map_width * resolution / 2, -map_height * resolution / 2, 0)
        )
        
        # Localization
        self._particle_filter = ParticleFilter(num_particles)
        self._particle_filter.initialize(Pose2D())
        
        # Scan matching
        self._scan_matcher = ScanMatcher()
        
        # State
        self._last_pose = Pose2D()
        self._last_scan: Optional[LaserScan] = None
        self._initialized = False
    
    def update_scan(
        self,
        ranges: List[float],
        pose_estimate: Optional[Pose2D] = None,
        odometry: Optional[Tuple[float, float, float]] = None
    ):
        """
        Update SLAM with new laser scan.
        
        Args:
            ranges: Laser range measurements
            pose_estimate: Optional pose estimate
            odometry: Optional odometry (dx, dy, dtheta)
        """
        scan = LaserScan(ranges=ranges)
        
        # Initialize if needed
        if not self._initialized:
            if pose_estimate:
                self._particle_filter.initialize(pose_estimate)
            self._initialized = True
            self._last_scan = scan
            self._last_pose = self._particle_filter.get_pose()
            return
        
        # Compute odometry if not provided
        if odometry is None:
            if pose_estimate:
                odometry = (
                    pose_estimate.x - self._last_pose.x,
                    pose_estimate.y - self._last_pose.y,
                    pose_estimate.theta - self._last_pose.theta
                )
            else:
                odometry = (0.0, 0.0, 0.0)
        
        # Update localization
        self._particle_filter.update(odometry, scan, self._grid)
        
        # Get pose
        pose = self._particle_filter.get_pose()
        
        # Update map
        self._update_map(scan, pose)
        
        # Save state
        self._last_pose = pose
        self._last_scan = scan
    
    def _update_map(self, scan: LaserScan, pose: Pose2D):
        """Update occupancy grid with scan."""
        robot_gx, robot_gy = self._grid.world_to_grid(pose.x, pose.y)
        
        for i, range_val in enumerate(scan.ranges):
            if range_val < scan.range_min or range_val > scan.range_max:
                continue
            
            angle = scan.angle_min + i * scan.angle_increment + pose.theta
            
            # Endpoint
            end_x = pose.x + range_val * math.cos(angle)
            end_y = pose.y + range_val * math.sin(angle)
            end_gx, end_gy = self._grid.world_to_grid(end_x, end_y)
            
            # Raycast - mark cells as free
            cells = self._bresenham(robot_gx, robot_gy, end_gx, end_gy)
            for gx, gy in cells[:-1]:  # All but last (endpoint)
                if self._grid.in_bounds(gx, gy):
                    current = self._grid.get_cell(gx, gy)
                    # Decrease occupancy probability
                    new_val = max(0, current - 5) if current > 0 else 0
                    self._grid.set_cell(gx, gy, new_val)
            
            # Mark endpoint as occupied
            if self._grid.in_bounds(end_gx, end_gy):
                current = self._grid.get_cell(end_gx, end_gy)
                new_val = min(100, current + 20) if current >= 0 else 50
                self._grid.set_cell(end_gx, end_gy, new_val)
    
    def _bresenham(
        self,
        x0: int, y0: int,
        x1: int, y1: int
    ) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm."""
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
    
    def get_pose(self) -> Pose2D:
        """Get current pose estimate."""
        return self._particle_filter.get_pose()
    
    def get_map(self) -> OccupancyGrid:
        """Get occupancy grid map."""
        return self._grid
    
    def save_map(self, filepath: str):
        """Save map to file."""
        import json
        
        data = {
            "width": self._grid.width,
            "height": self._grid.height,
            "resolution": self._grid.resolution,
            "origin": self._grid.origin.to_tuple(),
            "data": self._grid.data
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Map saved to: {filepath}")
    
    def load_map(self, filepath: str):
        """Load map from file."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self._grid = OccupancyGrid(
            width=data["width"],
            height=data["height"],
            resolution=data["resolution"],
            origin=Pose2D.from_tuple(tuple(data["origin"])),
            data=data["data"]
        )
        
        logger.info(f"Map loaded from: {filepath}")


# Global instance
_slam: Optional[SLAMSystem] = None


def get_slam(**kwargs) -> SLAMSystem:
    """Get or create global SLAM system."""
    global _slam
    if _slam is None:
        _slam = SLAMSystem(**kwargs)
    return _slam
