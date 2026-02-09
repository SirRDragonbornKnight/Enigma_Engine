"""
Manipulation Planning for Enigma AI Engine

Robot arm manipulation and motion planning.

Features:
- Forward/inverse kinematics
- Path planning (RRT, RRT*)
- Collision checking
- Grasp planning
- Trajectory optimization

Usage:
    from enigma_engine.tools.manipulation import ManipulationPlanner, get_planner
    
    planner = get_planner()
    
    # Plan path to target
    path = planner.plan_path(start_config, goal_pose)
    
    # Execute path
    planner.execute(path)
"""

import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class JointType(Enum):
    """Types of robot joints."""
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    FIXED = "fixed"


@dataclass
class Joint:
    """Robot joint definition."""
    name: str
    joint_type: JointType
    
    # Limits
    lower_limit: float = -math.pi
    upper_limit: float = math.pi
    
    # For prismatic
    linear_lower: float = 0.0
    linear_upper: float = 1.0
    
    # DH parameters
    d: float = 0.0  # Link offset
    a: float = 0.0  # Link length
    alpha: float = 0.0  # Link twist
    theta_offset: float = 0.0  # Joint angle offset


@dataclass
class Pose3D:
    """3D pose (position + orientation)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    # Quaternion orientation
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    
    def to_matrix(self) -> List[List[float]]:
        """Convert to 4x4 transformation matrix."""
        # Rotation from quaternion
        xx = self.qx * self.qx
        xy = self.qx * self.qy
        xz = self.qx * self.qz
        xw = self.qx * self.qw
        yy = self.qy * self.qy
        yz = self.qy * self.qz
        yw = self.qy * self.qw
        zz = self.qz * self.qz
        zw = self.qz * self.qw
        
        return [
            [1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw), self.x],
            [2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw), self.y],
            [2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy), self.z],
            [0, 0, 0, 1]
        ]


@dataclass
class Configuration:
    """Joint configuration."""
    joint_values: List[float]
    
    def distance_to(self, other: 'Configuration') -> float:
        """Compute distance to another configuration."""
        return math.sqrt(sum(
            (a - b)**2 for a, b in zip(self.joint_values, other.joint_values)
        ))


@dataclass
class Trajectory:
    """Motion trajectory."""
    waypoints: List[Configuration]
    timestamps: List[float] = field(default_factory=list)
    
    # Velocity/acceleration limits
    max_velocity: List[float] = field(default_factory=list)
    max_acceleration: List[float] = field(default_factory=list)


@dataclass
class Obstacle:
    """Collision obstacle."""
    type: str  # "box", "sphere", "cylinder"
    pose: Pose3D
    
    # Dimensions
    size_x: float = 1.0
    size_y: float = 1.0
    size_z: float = 1.0
    radius: float = 0.5


class RobotArm:
    """Robot arm model."""
    
    def __init__(
        self,
        joints: List[Joint],
        name: str = "arm"
    ):
        self._joints = joints
        self._name = name
        self._dof = len(joints)
    
    @property
    def dof(self) -> int:
        return self._dof
    
    @property
    def joints(self) -> List[Joint]:
        return self._joints
    
    def forward_kinematics(
        self,
        config: Configuration
    ) -> Pose3D:
        """
        Compute end-effector pose from joint angles.
        
        Returns:
            End-effector pose
        """
        # Compute transformation matrix using DH parameters
        T = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        
        for i, (joint, value) in enumerate(zip(self._joints, config.joint_values)):
            theta = value + joint.theta_offset
            
            # DH transformation
            ct = math.cos(theta)
            st = math.sin(theta)
            ca = math.cos(joint.alpha)
            sa = math.sin(joint.alpha)
            
            T_i = [
                [ct, -st * ca, st * sa, joint.a * ct],
                [st, ct * ca, -ct * sa, joint.a * st],
                [0, sa, ca, joint.d],
                [0, 0, 0, 1]
            ]
            
            T = self._matrix_multiply(T, T_i)
        
        # Extract pose from transformation matrix
        return Pose3D(
            x=T[0][3],
            y=T[1][3],
            z=T[2][3],
            # Simplified quaternion extraction
            qw=1.0,
            qx=0.0,
            qy=0.0,
            qz=0.0
        )
    
    def inverse_kinematics(
        self,
        target_pose: Pose3D,
        seed: Optional[Configuration] = None,
        max_iterations: int = 100
    ) -> Optional[Configuration]:
        """
        Compute joint angles for target pose.
        
        Returns:
            Configuration or None if not found
        """
        # Numerical IK using Jacobian
        if seed is None:
            seed = Configuration([0.0] * self._dof)
        
        config = Configuration(list(seed.joint_values))
        
        for iteration in range(max_iterations):
            # Current pose
            current = self.forward_kinematics(config)
            
            # Error
            error = [
                target_pose.x - current.x,
                target_pose.y - current.y,
                target_pose.z - current.z
            ]
            
            error_norm = math.sqrt(sum(e**2 for e in error))
            
            if error_norm < 0.001:  # Converged
                return config
            
            # Compute Jacobian numerically
            J = self._compute_jacobian(config)
            
            # Pseudoinverse (simplified)
            delta = [0.0] * self._dof
            for j in range(self._dof):
                for i in range(3):
                    delta[j] += J[i][j] * error[i] * 0.1
            
            # Update config
            for j in range(self._dof):
                config.joint_values[j] += delta[j]
                
                # Clamp to limits
                config.joint_values[j] = max(
                    self._joints[j].lower_limit,
                    min(self._joints[j].upper_limit, config.joint_values[j])
                )
        
        return None  # Failed to converge
    
    def _compute_jacobian(
        self,
        config: Configuration,
        delta: float = 0.001
    ) -> List[List[float]]:
        """Compute numerical Jacobian."""
        J = [[0.0] * self._dof for _ in range(3)]
        
        base_pose = self.forward_kinematics(config)
        
        for j in range(self._dof):
            # Perturb joint
            perturbed = Configuration(list(config.joint_values))
            perturbed.joint_values[j] += delta
            
            new_pose = self.forward_kinematics(perturbed)
            
            J[0][j] = (new_pose.x - base_pose.x) / delta
            J[1][j] = (new_pose.y - base_pose.y) / delta
            J[2][j] = (new_pose.z - base_pose.z) / delta
        
        return J
    
    def _matrix_multiply(
        self,
        A: List[List[float]],
        B: List[List[float]]
    ) -> List[List[float]]:
        """Multiply 4x4 matrices."""
        result = [[0.0] * 4 for _ in range(4)]
        
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    result[i][j] += A[i][k] * B[k][j]
        
        return result


class CollisionChecker:
    """Check for collisions."""
    
    def __init__(self, obstacles: Optional[List[Obstacle]] = None):
        self._obstacles = obstacles or []
        self._arm: Optional[RobotArm] = None
    
    def set_arm(self, arm: RobotArm):
        """Set the robot arm for collision checking."""
        self._arm = arm
    
    def add_obstacle(self, obstacle: Obstacle):
        """Add an obstacle."""
        self._obstacles.append(obstacle)
    
    def clear_obstacles(self):
        """Clear all obstacles."""
        self._obstacles.clear()
    
    def check_collision(
        self,
        config: Configuration
    ) -> bool:
        """
        Check if configuration is in collision.
        
        Returns:
            True if collision detected
        """
        if not self._arm:
            return False
        
        # Get link positions
        ee_pose = self._arm.forward_kinematics(config)
        
        # Check against obstacles (simplified)
        for obstacle in self._obstacles:
            if self._check_point_collision(ee_pose, obstacle):
                return True
        
        return False
    
    def _check_point_collision(
        self,
        point: Pose3D,
        obstacle: Obstacle
    ) -> bool:
        """Check if point collides with obstacle."""
        if obstacle.type == "sphere":
            dist = math.sqrt(
                (point.x - obstacle.pose.x)**2 +
                (point.y - obstacle.pose.y)**2 +
                (point.z - obstacle.pose.z)**2
            )
            return dist < obstacle.radius
        
        elif obstacle.type == "box":
            # AABB check
            half_x = obstacle.size_x / 2
            half_y = obstacle.size_y / 2
            half_z = obstacle.size_z / 2
            
            return (
                abs(point.x - obstacle.pose.x) < half_x and
                abs(point.y - obstacle.pose.y) < half_y and
                abs(point.z - obstacle.pose.z) < half_z
            )
        
        return False


class RRTPlanner:
    """RRT (Rapidly-exploring Random Tree) path planner."""
    
    def __init__(
        self,
        arm: RobotArm,
        collision_checker: CollisionChecker,
        step_size: float = 0.1,
        goal_bias: float = 0.1,
        max_iterations: int = 1000
    ):
        self._arm = arm
        self._collision = collision_checker
        self._step_size = step_size
        self._goal_bias = goal_bias
        self._max_iterations = max_iterations
    
    def plan(
        self,
        start: Configuration,
        goal: Configuration
    ) -> Optional[List[Configuration]]:
        """
        Plan path from start to goal.
        
        Returns:
            List of configurations or None if failed
        """
        # Tree: config -> parent index
        tree: List[Tuple[Configuration, int]] = [(start, -1)]
        
        for iteration in range(self._max_iterations):
            # Sample random configuration
            if random.random() < self._goal_bias:
                q_rand = goal
            else:
                q_rand = self._random_config()
            
            # Find nearest node
            nearest_idx = self._find_nearest(tree, q_rand)
            nearest = tree[nearest_idx][0]
            
            # Extend towards random
            q_new = self._extend(nearest, q_rand)
            
            # Check collision
            if not self._collision.check_collision(q_new):
                tree.append((q_new, nearest_idx))
                
                # Check if reached goal
                if q_new.distance_to(goal) < self._step_size:
                    # Reconstruct path
                    return self._reconstruct_path(tree, len(tree) - 1)
        
        logger.warning("RRT planning failed to find path")
        return None
    
    def _random_config(self) -> Configuration:
        """Sample random configuration."""
        values = []
        for joint in self._arm.joints:
            value = random.uniform(joint.lower_limit, joint.upper_limit)
            values.append(value)
        return Configuration(values)
    
    def _find_nearest(
        self,
        tree: List[Tuple[Configuration, int]],
        target: Configuration
    ) -> int:
        """Find nearest node in tree."""
        best_idx = 0
        best_dist = float('inf')
        
        for i, (config, _) in enumerate(tree):
            dist = config.distance_to(target)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        return best_idx
    
    def _extend(
        self,
        from_config: Configuration,
        to_config: Configuration
    ) -> Configuration:
        """Extend from config towards target."""
        dist = from_config.distance_to(to_config)
        
        if dist <= self._step_size:
            return to_config
        
        # Interpolate
        ratio = self._step_size / dist
        values = [
            f + ratio * (t - f)
            for f, t in zip(from_config.joint_values, to_config.joint_values)
        ]
        
        return Configuration(values)
    
    def _reconstruct_path(
        self,
        tree: List[Tuple[Configuration, int]],
        goal_idx: int
    ) -> List[Configuration]:
        """Reconstruct path from tree."""
        path = []
        idx = goal_idx
        
        while idx != -1:
            config, parent_idx = tree[idx]
            path.append(config)
            idx = parent_idx
        
        path.reverse()
        return path


class GraspPlanner:
    """Plan grasps for objects."""
    
    def __init__(
        self,
        gripper_width: float = 0.1
    ):
        self._gripper_width = gripper_width
    
    def plan_grasp(
        self,
        object_pose: Pose3D,
        object_size: Tuple[float, float, float]
    ) -> List[Pose3D]:
        """
        Plan grasp poses for an object.
        
        Returns:
            List of candidate grasp poses
        """
        grasps = []
        
        # Top grasp
        grasps.append(Pose3D(
            x=object_pose.x,
            y=object_pose.y,
            z=object_pose.z + object_size[2] / 2 + 0.1,
            qw=1.0
        ))
        
        # Side grasps
        for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:
            offset = max(object_size[0], object_size[1]) / 2 + 0.1
            grasps.append(Pose3D(
                x=object_pose.x + offset * math.cos(angle),
                y=object_pose.y + offset * math.sin(angle),
                z=object_pose.z,
                qz=math.sin(angle / 2),
                qw=math.cos(angle / 2)
            ))
        
        return grasps
    
    def evaluate_grasp(
        self,
        grasp_pose: Pose3D,
        object_pose: Pose3D
    ) -> float:
        """
        Evaluate grasp quality.
        
        Returns:
            Score (0-1, higher = better)
        """
        # Simple scoring based on approach angle
        # Top grasp preferred
        height_diff = grasp_pose.z - object_pose.z
        if height_diff > 0:
            return 0.8 + 0.2 * min(1.0, height_diff / 0.2)
        else:
            return 0.5


class TrajectoryOptimizer:
    """Optimize trajectories for smoothness."""
    
    def __init__(
        self,
        max_velocity: float = 1.0,
        max_acceleration: float = 2.0
    ):
        self._max_vel = max_velocity
        self._max_acc = max_acceleration
    
    def optimize(
        self,
        path: List[Configuration],
        dt: float = 0.1
    ) -> Trajectory:
        """
        Optimize path into smooth trajectory.
        
        Returns:
            Optimized trajectory
        """
        if not path:
            return Trajectory(waypoints=[])
        
        waypoints = []
        timestamps = [0.0]
        
        # Interpolate for smoothness
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            
            # Number of interpolation steps
            dist = start.distance_to(end)
            steps = max(1, int(dist / (self._max_vel * dt)))
            
            for j in range(steps):
                ratio = j / steps
                values = [
                    s + ratio * (e - s)
                    for s, e in zip(start.joint_values, end.joint_values)
                ]
                waypoints.append(Configuration(values))
                timestamps.append(timestamps[-1] + dt)
        
        waypoints.append(path[-1])
        
        return Trajectory(
            waypoints=waypoints,
            timestamps=timestamps,
            max_velocity=[self._max_vel] * len(path[0].joint_values),
            max_acceleration=[self._max_acc] * len(path[0].joint_values)
        )


class ManipulationPlanner:
    """High-level manipulation planner."""
    
    def __init__(
        self,
        arm: Optional[RobotArm] = None
    ):
        """
        Initialize planner.
        
        Args:
            arm: Robot arm model
        """
        if arm is None:
            # Default 6-DOF arm
            arm = RobotArm([
                Joint("j1", JointType.REVOLUTE, d=0.1, a=0),
                Joint("j2", JointType.REVOLUTE, d=0, a=0.5),
                Joint("j3", JointType.REVOLUTE, d=0, a=0.4),
                Joint("j4", JointType.REVOLUTE, d=0.1, a=0),
                Joint("j5", JointType.REVOLUTE, d=0, a=0),
                Joint("j6", JointType.REVOLUTE, d=0.05, a=0)
            ])
        
        self._arm = arm
        self._collision = CollisionChecker()
        self._collision.set_arm(arm)
        
        self._rrt = RRTPlanner(arm, self._collision)
        self._grasp = GraspPlanner()
        self._optimizer = TrajectoryOptimizer()
        
        self._current_config = Configuration([0.0] * arm.dof)
    
    def add_obstacle(self, obstacle: Obstacle):
        """Add a collision obstacle."""
        self._collision.add_obstacle(obstacle)
    
    def clear_obstacles(self):
        """Clear all obstacles."""
        self._collision.clear_obstacles()
    
    def set_current_config(self, config: Configuration):
        """Set current joint configuration."""
        self._current_config = config
    
    def get_current_pose(self) -> Pose3D:
        """Get current end-effector pose."""
        return self._arm.forward_kinematics(self._current_config)
    
    def plan_to_pose(
        self,
        target_pose: Pose3D
    ) -> Optional[Trajectory]:
        """
        Plan motion to target pose.
        
        Returns:
            Trajectory or None if failed
        """
        # Compute IK
        goal_config = self._arm.inverse_kinematics(target_pose, self._current_config)
        
        if goal_config is None:
            logger.warning("IK failed for target pose")
            return None
        
        # Plan path
        path = self._rrt.plan(self._current_config, goal_config)
        
        if path is None:
            return None
        
        # Optimize trajectory
        return self._optimizer.optimize(path)
    
    def plan_pick(
        self,
        object_pose: Pose3D,
        object_size: Tuple[float, float, float] = (0.05, 0.05, 0.1)
    ) -> Optional[Trajectory]:
        """
        Plan pick operation.
        
        Returns:
            Trajectory or None if failed
        """
        # Get grasp candidates
        grasps = self._grasp.plan_grasp(object_pose, object_size)
        
        # Try each grasp
        for grasp_pose in grasps:
            trajectory = self.plan_to_pose(grasp_pose)
            if trajectory:
                return trajectory
        
        return None
    
    def plan_place(
        self,
        place_pose: Pose3D
    ) -> Optional[Trajectory]:
        """
        Plan place operation.
        
        Returns:
            Trajectory
        """
        return self.plan_to_pose(place_pose)
    
    def execute(
        self,
        trajectory: Trajectory,
        callback: Optional[Callable[[Configuration], None]] = None
    ):
        """
        Execute trajectory.
        
        Args:
            trajectory: Trajectory to execute
            callback: Called for each waypoint
        """
        for i, waypoint in enumerate(trajectory.waypoints):
            self._current_config = waypoint
            
            if callback:
                callback(waypoint)
            
            # Wait for next timestamp
            if i < len(trajectory.timestamps) - 1:
                dt = trajectory.timestamps[i + 1] - trajectory.timestamps[i]
                time.sleep(dt)


# Factory function for common arms
def create_panda_arm() -> RobotArm:
    """Create Franka Emika Panda arm model."""
    return RobotArm([
        Joint("panda_joint1", JointType.REVOLUTE, d=0.333, a=0, alpha=0,
              lower_limit=-2.8973, upper_limit=2.8973),
        Joint("panda_joint2", JointType.REVOLUTE, d=0, a=0, alpha=-math.pi/2,
              lower_limit=-1.7628, upper_limit=1.7628),
        Joint("panda_joint3", JointType.REVOLUTE, d=0.316, a=0, alpha=math.pi/2,
              lower_limit=-2.8973, upper_limit=2.8973),
        Joint("panda_joint4", JointType.REVOLUTE, d=0, a=0.0825, alpha=math.pi/2,
              lower_limit=-3.0718, upper_limit=-0.0698),
        Joint("panda_joint5", JointType.REVOLUTE, d=0.384, a=-0.0825, alpha=-math.pi/2,
              lower_limit=-2.8973, upper_limit=2.8973),
        Joint("panda_joint6", JointType.REVOLUTE, d=0, a=0, alpha=math.pi/2,
              lower_limit=-0.0175, upper_limit=3.7525),
        Joint("panda_joint7", JointType.REVOLUTE, d=0.107, a=0.088, alpha=math.pi/2,
              lower_limit=-2.8973, upper_limit=2.8973)
    ], name="panda")


# Global instance
_planner: Optional[ManipulationPlanner] = None


def get_planner(arm: Optional[RobotArm] = None) -> ManipulationPlanner:
    """Get or create global manipulation planner."""
    global _planner
    if _planner is None:
        _planner = ManipulationPlanner(arm)
    return _planner
