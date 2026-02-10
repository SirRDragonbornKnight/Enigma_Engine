"""
Robot Manipulation

Arm control, inverse kinematics, and grasping for robotics.

FILE: enigma_engine/robotics/manipulation.py
TYPE: Robotics
MAIN CLASSES: RobotArm, InverseKinematics, GraspPlanner
"""

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JointType(Enum):
    """Robot joint types."""
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    FIXED = "fixed"


class GripperState(Enum):
    """Gripper states."""
    OPEN = "open"
    CLOSED = "closed"
    GRASPING = "grasping"


@dataclass
class DHParameter:
    """Denavit-Hartenberg parameters for a joint."""
    d: float  # Link offset
    theta: float  # Joint angle
    a: float  # Link length
    alpha: float  # Link twist


@dataclass
class JointConfig:
    """Joint configuration."""
    name: str
    joint_type: JointType
    min_limit: float = -math.pi
    max_limit: float = math.pi
    max_velocity: float = 1.0
    max_effort: float = 100.0
    dh: DHParameter = None


@dataclass
class Pose3D:
    """3D pose with position and orientation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0  # rotation around x
    pitch: float = 0.0  # rotation around y
    yaw: float = 0.0  # rotation around z


@dataclass
class GraspCandidate:
    """Grasp candidate for planning."""
    pose: Pose3D
    score: float
    approach_vector: tuple[float, float, float]
    aperture: float  # gripper opening


@dataclass
class ManipulationConfig:
    """Manipulation configuration."""
    # IK solver settings
    max_iterations: int = 100
    tolerance: float = 0.001
    damping: float = 0.1
    
    # Motion planning
    interpolation_steps: int = 50
    collision_check: bool = True
    
    # Grasp planning
    num_candidates: int = 10
    approach_distance: float = 0.1


if HAS_NUMPY:
    
    class RobotArm:
        """
        Robot arm kinematics and control.
        """
        
        # Common arm configurations
        PRESETS = {
            "6dof": [
                JointConfig("joint_1", JointType.REVOLUTE, -math.pi, math.pi,
                           dh=DHParameter(0.089, 0, 0, math.pi/2)),
                JointConfig("joint_2", JointType.REVOLUTE, -math.pi/2, math.pi/2,
                           dh=DHParameter(0, 0, 0.425, 0)),
                JointConfig("joint_3", JointType.REVOLUTE, -math.pi, 0,
                           dh=DHParameter(0, 0, 0.392, 0)),
                JointConfig("joint_4", JointType.REVOLUTE, -math.pi, math.pi,
                           dh=DHParameter(0.109, 0, 0, math.pi/2)),
                JointConfig("joint_5", JointType.REVOLUTE, -math.pi, math.pi,
                           dh=DHParameter(0.095, 0, 0, -math.pi/2)),
                JointConfig("joint_6", JointType.REVOLUTE, -math.pi, math.pi,
                           dh=DHParameter(0.082, 0, 0, 0)),
            ],
            "4dof": [
                JointConfig("base", JointType.REVOLUTE, -math.pi, math.pi,
                           dh=DHParameter(0.1, 0, 0, math.pi/2)),
                JointConfig("shoulder", JointType.REVOLUTE, -math.pi/2, math.pi/2,
                           dh=DHParameter(0, 0, 0.3, 0)),
                JointConfig("elbow", JointType.REVOLUTE, -math.pi, 0,
                           dh=DHParameter(0, 0, 0.25, 0)),
                JointConfig("wrist", JointType.REVOLUTE, -math.pi, math.pi,
                           dh=DHParameter(0.1, 0, 0, 0)),
            ],
        }
        
        def __init__(self, joints: list[JointConfig] = None, preset: str = None):
            if preset and preset in self.PRESETS:
                self.joints = self.PRESETS[preset]
            elif joints:
                self.joints = joints
            else:
                self.joints = self.PRESETS["6dof"]
            
            self.num_joints = len(self.joints)
            self._joint_positions = np.zeros(self.num_joints)
        
        def get_joint_positions(self) -> np.ndarray:
            """Get current joint positions."""
            return self._joint_positions.copy()
        
        def set_joint_positions(self, positions: np.ndarray):
            """Set joint positions with limit checking."""
            for i, (pos, joint) in enumerate(zip(positions, self.joints)):
                self._joint_positions[i] = np.clip(
                    pos,
                    joint.min_limit,
                    joint.max_limit
                )
        
        def _dh_matrix(self, dh: DHParameter, theta_offset: float) -> np.ndarray:
            """Create DH transformation matrix."""
            theta = dh.theta + theta_offset
            
            ct = np.cos(theta)
            st = np.sin(theta)
            ca = np.cos(dh.alpha)
            sa = np.sin(dh.alpha)
            
            return np.array([
                [ct, -st * ca, st * sa, dh.a * ct],
                [st, ct * ca, -ct * sa, dh.a * st],
                [0, sa, ca, dh.d],
                [0, 0, 0, 1]
            ])
        
        def forward_kinematics(self, joint_positions: np.ndarray = None) -> np.ndarray:
            """
            Compute forward kinematics.
            
            Returns:
                4x4 homogeneous transformation matrix
            """
            if joint_positions is None:
                joint_positions = self._joint_positions
            
            T = np.eye(4)
            
            for i, joint in enumerate(self.joints):
                if joint.dh:
                    T = T @ self._dh_matrix(joint.dh, joint_positions[i])
            
            return T
        
        def get_end_effector_pose(self, joint_positions: np.ndarray = None) -> Pose3D:
            """Get end effector pose from joint positions."""
            T = self.forward_kinematics(joint_positions)
            
            # Extract position
            x, y, z = T[0, 3], T[1, 3], T[2, 3]
            
            # Extract Euler angles from rotation matrix
            if abs(T[2, 0]) != 1:
                pitch = -np.arcsin(T[2, 0])
                roll = np.arctan2(T[2, 1] / np.cos(pitch), T[2, 2] / np.cos(pitch))
                yaw = np.arctan2(T[1, 0] / np.cos(pitch), T[0, 0] / np.cos(pitch))
            else:
                yaw = 0
                if T[2, 0] == -1:
                    pitch = np.pi / 2
                    roll = np.arctan2(T[0, 1], T[0, 2])
                else:
                    pitch = -np.pi / 2
                    roll = np.arctan2(-T[0, 1], -T[0, 2])
            
            return Pose3D(x, y, z, roll, pitch, yaw)
        
        def jacobian(self, joint_positions: np.ndarray = None) -> np.ndarray:
            """
            Compute Jacobian matrix.
            
            Returns:
                6 x num_joints Jacobian matrix
            """
            if joint_positions is None:
                joint_positions = self._joint_positions
            
            J = np.zeros((6, self.num_joints))
            T = np.eye(4)
            
            # Get end effector position
            T_ee = self.forward_kinematics(joint_positions)
            p_ee = T_ee[:3, 3]
            
            for i, joint in enumerate(self.joints):
                # Get z-axis and origin of each frame
                z = T[:3, 2]
                p = T[:3, 3]
                
                if joint.joint_type == JointType.REVOLUTE:
                    # Linear velocity component
                    J[:3, i] = np.cross(z, p_ee - p)
                    # Angular velocity component
                    J[3:, i] = z
                elif joint.joint_type == JointType.PRISMATIC:
                    J[:3, i] = z
                    J[3:, i] = 0
                
                # Advance to next frame
                if joint.dh:
                    T = T @ self._dh_matrix(joint.dh, joint_positions[i])
            
            return J
    
    
    class InverseKinematics:
        """
        Inverse kinematics solver using damped least squares.
        """
        
        def __init__(self, arm: RobotArm, config: ManipulationConfig = None):
            self.arm = arm
            self.config = config or ManipulationConfig()
        
        def solve(
            self,
            target_pose: Pose3D,
            initial_guess: np.ndarray = None
        ) -> Optional[np.ndarray]:
            """
            Solve IK for target pose.
            
            Args:
                target_pose: Desired end effector pose
                initial_guess: Starting joint configuration
                
            Returns:
                Joint positions or None if failed
            """
            if initial_guess is None:
                q = self.arm.get_joint_positions()
            else:
                q = initial_guess.copy()
            
            target_pos = np.array([target_pose.x, target_pose.y, target_pose.z])
            target_rot = self._euler_to_rotation(
                target_pose.roll,
                target_pose.pitch,
                target_pose.yaw
            )
            
            for iteration in range(self.config.max_iterations):
                # Current pose
                T = self.arm.forward_kinematics(q)
                current_pos = T[:3, 3]
                current_rot = T[:3, :3]
                
                # Position error
                pos_error = target_pos - current_pos
                
                # Orientation error (angle-axis)
                R_error = target_rot @ current_rot.T
                angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
                
                if angle < 1e-6:
                    rot_error = np.zeros(3)
                else:
                    axis = np.array([
                        R_error[2, 1] - R_error[1, 2],
                        R_error[0, 2] - R_error[2, 0],
                        R_error[1, 0] - R_error[0, 1]
                    ]) / (2 * np.sin(angle))
                    rot_error = angle * axis
                
                # Combined error
                error = np.concatenate([pos_error, rot_error])
                
                # Check convergence
                if np.linalg.norm(error) < self.config.tolerance:
                    return q
                
                # Jacobian
                J = self.arm.jacobian(q)
                
                # Damped least squares
                JJT = J @ J.T
                lambda_sq = self.config.damping ** 2
                
                delta_q = J.T @ np.linalg.solve(
                    JJT + lambda_sq * np.eye(6),
                    error
                )
                
                # Update joints
                q = q + delta_q
                
                # Apply joint limits
                for i, joint in enumerate(self.arm.joints):
                    q[i] = np.clip(q[i], joint.min_limit, joint.max_limit)
            
            # Failed to converge
            return None
        
        def solve_position_only(
            self,
            target_position: tuple[float, float, float],
            initial_guess: np.ndarray = None
        ) -> Optional[np.ndarray]:
            """Solve IK for position only (ignoring orientation)."""
            if initial_guess is None:
                q = self.arm.get_joint_positions()
            else:
                q = initial_guess.copy()
            
            target = np.array(target_position)
            
            for _ in range(self.config.max_iterations):
                T = self.arm.forward_kinematics(q)
                current = T[:3, 3]
                
                error = target - current
                
                if np.linalg.norm(error) < self.config.tolerance:
                    return q
                
                J = self.arm.jacobian(q)[:3, :]  # Position only
                
                JJT = J @ J.T
                lambda_sq = self.config.damping ** 2
                
                delta_q = J.T @ np.linalg.solve(
                    JJT + lambda_sq * np.eye(3),
                    error
                )
                
                q = q + delta_q
                
                for i, joint in enumerate(self.arm.joints):
                    q[i] = np.clip(q[i], joint.min_limit, joint.max_limit)
            
            return None
        
        def _euler_to_rotation(
            self,
            roll: float,
            pitch: float,
            yaw: float
        ) -> np.ndarray:
            """Convert Euler angles to rotation matrix."""
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            
            return Rz @ Ry @ Rx
    
    
    class GraspPlanner:
        """
        Grasp planning for pick and place operations.
        """
        
        def __init__(
            self,
            arm: RobotArm,
            ik_solver: InverseKinematics = None,
            config: ManipulationConfig = None
        ):
            self.arm = arm
            self.config = config or ManipulationConfig()
            self.ik = ik_solver or InverseKinematics(arm, self.config)
            
            self.gripper_state = GripperState.OPEN
            self.gripper_aperture = 0.1  # meters
        
        def generate_grasp_candidates(
            self,
            object_pose: Pose3D,
            object_size: tuple[float, float, float] = (0.05, 0.05, 0.05)
        ) -> list[GraspCandidate]:
            """
            Generate grasp candidates for an object.
            
            Args:
                object_pose: Object position and orientation
                object_size: Object bounding box (width, height, depth)
                
            Returns:
                List of grasp candidates sorted by score
            """
            candidates = []
            
            # Top-down grasps
            for angle in np.linspace(0, math.pi, 4):
                grasp_pose = Pose3D(
                    x=object_pose.x,
                    y=object_pose.y,
                    z=object_pose.z + object_size[2] / 2 + 0.02,
                    roll=0,
                    pitch=math.pi,  # pointing down
                    yaw=angle
                )
                
                approach = (0, 0, -1)  # approach from above
                aperture = max(object_size[0], object_size[1]) * 1.2
                
                score = self._score_grasp(grasp_pose, approach)
                candidates.append(GraspCandidate(
                    pose=grasp_pose,
                    score=score,
                    approach_vector=approach,
                    aperture=aperture
                ))
            
            # Side grasps
            for angle in np.linspace(0, 2 * math.pi, 8):
                dx = math.cos(angle) * (object_size[0] / 2 + 0.02)
                dy = math.sin(angle) * (object_size[1] / 2 + 0.02)
                
                grasp_pose = Pose3D(
                    x=object_pose.x + dx,
                    y=object_pose.y + dy,
                    z=object_pose.z,
                    roll=0,
                    pitch=math.pi / 2,
                    yaw=angle + math.pi
                )
                
                approach = (-math.cos(angle), -math.sin(angle), 0)
                aperture = object_size[2] * 1.2
                
                score = self._score_grasp(grasp_pose, approach)
                candidates.append(GraspCandidate(
                    pose=grasp_pose,
                    score=score,
                    approach_vector=approach,
                    aperture=aperture
                ))
            
            # Sort by score
            candidates.sort(key=lambda c: c.score, reverse=True)
            return candidates[:self.config.num_candidates]
        
        def _score_grasp(
            self,
            grasp_pose: Pose3D,
            approach: tuple[float, float, float]
        ) -> float:
            """Score a grasp candidate."""
            score = 1.0
            
            # Check IK feasibility
            solution = self.ik.solve(grasp_pose)
            if solution is None:
                return 0.0
            
            # Favor grasps away from joint limits
            for i, (q, joint) in enumerate(zip(solution, self.arm.joints)):
                range_val = joint.max_limit - joint.min_limit
                mid = (joint.max_limit + joint.min_limit) / 2
                distance_from_mid = abs(q - mid) / (range_val / 2)
                score *= (1 - 0.3 * distance_from_mid)
            
            # Favor top-down grasps
            if approach[2] < 0:
                score *= 1.2
            
            return score
        
        def plan_pick(
            self,
            object_pose: Pose3D,
            object_size: tuple[float, float, float] = (0.05, 0.05, 0.05)
        ) -> Optional[list[np.ndarray]]:
            """
            Plan a pick trajectory.
            
            Returns:
                List of waypoints (joint configurations) or None
            """
            candidates = self.generate_grasp_candidates(object_pose, object_size)
            
            if not candidates:
                return None
            
            # Take best candidate
            grasp = candidates[0]
            
            # Plan approach
            approach_pose = Pose3D(
                x=grasp.pose.x + grasp.approach_vector[0] * self.config.approach_distance,
                y=grasp.pose.y + grasp.approach_vector[1] * self.config.approach_distance,
                z=grasp.pose.z + grasp.approach_vector[2] * self.config.approach_distance,
                roll=grasp.pose.roll,
                pitch=grasp.pose.pitch,
                yaw=grasp.pose.yaw
            )
            
            # Solve IK for waypoints
            current = self.arm.get_joint_positions()
            
            approach_q = self.ik.solve(approach_pose, current)
            if approach_q is None:
                return None
            
            grasp_q = self.ik.solve(grasp.pose, approach_q)
            if grasp_q is None:
                return None
            
            # Generate trajectory
            trajectory = []
            
            # Move to approach
            trajectory.extend(self._interpolate(current, approach_q))
            
            # Move to grasp
            trajectory.extend(self._interpolate(approach_q, grasp_q))
            
            return trajectory
        
        def plan_place(
            self,
            target_pose: Pose3D
        ) -> Optional[list[np.ndarray]]:
            """
            Plan a place trajectory.
            """
            current = self.arm.get_joint_positions()
            
            # Approach from above
            approach_pose = Pose3D(
                x=target_pose.x,
                y=target_pose.y,
                z=target_pose.z + self.config.approach_distance,
                roll=target_pose.roll,
                pitch=target_pose.pitch,
                yaw=target_pose.yaw
            )
            
            approach_q = self.ik.solve(approach_pose, current)
            if approach_q is None:
                return None
            
            place_q = self.ik.solve(target_pose, approach_q)
            if place_q is None:
                return None
            
            trajectory = []
            trajectory.extend(self._interpolate(current, approach_q))
            trajectory.extend(self._interpolate(approach_q, place_q))
            
            return trajectory
        
        def _interpolate(
            self,
            start: np.ndarray,
            end: np.ndarray
        ) -> list[np.ndarray]:
            """Interpolate between joint configurations."""
            waypoints = []
            
            for t in np.linspace(0, 1, self.config.interpolation_steps):
                q = start + t * (end - start)
                waypoints.append(q.copy())
            
            return waypoints
        
        def open_gripper(self, target_aperture: float = None):
            """Open gripper."""
            self.gripper_state = GripperState.OPEN
            if target_aperture is not None:
                self.gripper_aperture = target_aperture
        
        def close_gripper(self, target_aperture: float = 0.0):
            """Close gripper."""
            self.gripper_aperture = target_aperture
            self.gripper_state = GripperState.GRASPING if target_aperture > 0 else GripperState.CLOSED
    
    
    class TrajectoryExecutor:
        """
        Executes trajectories on the robot arm.
        """
        
        def __init__(self, arm: RobotArm, rate_hz: float = 100):
            self.arm = arm
            self.rate = rate_hz
            self._executing = False
        
        def execute(self, trajectory: list[np.ndarray], callback=None):
            """
            Execute a trajectory.
            
            Args:
                trajectory: List of joint configurations
                callback: Optional callback for each waypoint
            """
            self._executing = True
            dt = 1.0 / self.rate
            
            for i, waypoint in enumerate(trajectory):
                if not self._executing:
                    break
                
                self.arm.set_joint_positions(waypoint)
                
                if callback:
                    callback(i, len(trajectory), waypoint)
                
                time.sleep(dt)
            
            self._executing = False
        
        def stop(self):
            """Stop execution."""
            self._executing = False

else:
    class RobotArm:
        pass
    
    class InverseKinematics:
        pass
    
    class GraspPlanner:
        pass
    
    class TrajectoryExecutor:
        pass


def create_manipulation_system(
    preset: str = "6dof",
    **config_kwargs
) -> tuple['RobotArm', 'InverseKinematics', 'GraspPlanner']:
    """Create full manipulation system."""
    if not HAS_NUMPY:
        raise ImportError("NumPy required for manipulation")
    
    config = ManipulationConfig(**config_kwargs)
    arm = RobotArm(preset=preset)
    ik = InverseKinematics(arm, config)
    grasp = GraspPlanner(arm, ik, config)
    
    return arm, ik, grasp
