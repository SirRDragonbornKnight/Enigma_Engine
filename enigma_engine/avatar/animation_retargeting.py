"""
Animation Retargeting

Applies animations from one rig to another with different bone structures.
Handles bone mapping, scale adjustments, and motion adaptation.

FILE: enigma_engine/avatar/animation_retargeting.py
TYPE: Avatar Animation
MAIN CLASSES: AnimationRetargeter, BoneMapping, RetargetConfig
"""

import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BoneType(Enum):
    """Standard bone types for mapping."""
    # Core
    ROOT = "root"
    HIPS = "hips"
    SPINE = "spine"
    SPINE1 = "spine1"
    SPINE2 = "spine2"
    CHEST = "chest"
    NECK = "neck"
    HEAD = "head"
    
    # Arms
    LEFT_SHOULDER = "left_shoulder"
    LEFT_UPPER_ARM = "left_upper_arm"
    LEFT_LOWER_ARM = "left_lower_arm"
    LEFT_HAND = "left_hand"
    RIGHT_SHOULDER = "right_shoulder"
    RIGHT_UPPER_ARM = "right_upper_arm"
    RIGHT_LOWER_ARM = "right_lower_arm"
    RIGHT_HAND = "right_hand"
    
    # Legs
    LEFT_UPPER_LEG = "left_upper_leg"
    LEFT_LOWER_LEG = "left_lower_leg"
    LEFT_FOOT = "left_foot"
    LEFT_TOES = "left_toes"
    RIGHT_UPPER_LEG = "right_upper_leg"
    RIGHT_LOWER_LEG = "right_lower_leg"
    RIGHT_FOOT = "right_foot"
    RIGHT_TOES = "right_toes"
    
    # Fingers
    LEFT_THUMB_1 = "left_thumb_1"
    LEFT_THUMB_2 = "left_thumb_2"
    LEFT_THUMB_3 = "left_thumb_3"
    LEFT_INDEX_1 = "left_index_1"
    LEFT_INDEX_2 = "left_index_2"
    LEFT_INDEX_3 = "left_index_3"
    LEFT_MIDDLE_1 = "left_middle_1"
    LEFT_MIDDLE_2 = "left_middle_2"
    LEFT_MIDDLE_3 = "left_middle_3"
    LEFT_RING_1 = "left_ring_1"
    LEFT_RING_2 = "left_ring_2"
    LEFT_RING_3 = "left_ring_3"
    LEFT_PINKY_1 = "left_pinky_1"
    LEFT_PINKY_2 = "left_pinky_2"
    LEFT_PINKY_3 = "left_pinky_3"
    RIGHT_THUMB_1 = "right_thumb_1"
    RIGHT_THUMB_2 = "right_thumb_2"
    RIGHT_THUMB_3 = "right_thumb_3"
    RIGHT_INDEX_1 = "right_index_1"
    RIGHT_INDEX_2 = "right_index_2"
    RIGHT_INDEX_3 = "right_index_3"
    RIGHT_MIDDLE_1 = "right_middle_1"
    RIGHT_MIDDLE_2 = "right_middle_2"
    RIGHT_MIDDLE_3 = "right_middle_3"
    RIGHT_RING_1 = "right_ring_1"
    RIGHT_RING_2 = "right_ring_2"
    RIGHT_RING_3 = "right_ring_3"
    RIGHT_PINKY_1 = "right_pinky_1"
    RIGHT_PINKY_2 = "right_pinky_2"
    RIGHT_PINKY_3 = "right_pinky_3"


@dataclass
class Vector3:
    """3D vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalized(self) -> 'Vector3':
        length = self.length()
        if length < 0.0001:
            return Vector3()
        return self * (1.0 / length)


@dataclass
class Quaternion:
    """Rotation quaternion."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0
    
    @classmethod
    def identity(cls) -> 'Quaternion':
        return cls(0, 0, 0, 1)
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion multiplication."""
        return Quaternion(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        )
    
    def inverse(self) -> 'Quaternion':
        """Get inverse quaternion."""
        return Quaternion(-self.x, -self.y, -self.z, self.w)
    
    def normalized(self) -> 'Quaternion':
        """Normalize quaternion."""
        length = math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
        if length < 0.0001:
            return Quaternion.identity()
        return Quaternion(self.x/length, self.y/length, self.z/length, self.w/length)
    
    @classmethod
    def slerp(cls, a: 'Quaternion', b: 'Quaternion', t: float) -> 'Quaternion':
        """Spherical linear interpolation."""
        dot = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w
        
        if dot < 0:
            b = Quaternion(-b.x, -b.y, -b.z, -b.w)
            dot = -dot
        
        if dot > 0.9995:
            return Quaternion(
                a.x + t*(b.x - a.x),
                a.y + t*(b.y - a.y),
                a.z + t*(b.z - a.z),
                a.w + t*(b.w - a.w)
            ).normalized()
        
        theta_0 = math.acos(dot)
        theta = theta_0 * t
        
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)
        
        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return Quaternion(
            s0 * a.x + s1 * b.x,
            s0 * a.y + s1 * b.y,
            s0 * a.z + s1 * b.z,
            s0 * a.w + s1 * b.w
        )


@dataclass
class BoneTransform:
    """Transform for a single bone."""
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))


@dataclass
class AnimationKeyframe:
    """Single keyframe in animation."""
    time: float
    transforms: dict[str, BoneTransform] = field(default_factory=dict)


@dataclass
class Animation:
    """Animation clip."""
    name: str
    duration: float
    keyframes: list[AnimationKeyframe] = field(default_factory=list)
    fps: float = 30.0


@dataclass
class BoneMapping:
    """Maps bones between source and target rigs."""
    source_bone: str
    target_bone: str
    bone_type: BoneType
    rotation_offset: Quaternion = field(default_factory=Quaternion.identity)
    scale_factor: float = 1.0
    mirror: bool = False


@dataclass
class RetargetConfig:
    """Configuration for retargeting."""
    preserve_root_motion: bool = True
    apply_scale_correction: bool = True
    smooth_transitions: bool = True
    transition_frames: int = 3
    ignore_fingers: bool = False
    ignore_toes: bool = False


class AnimationRetargeter:
    """Retargets animations between different rigs."""
    
    # Common bone name patterns for auto-mapping
    BONE_PATTERNS = {
        BoneType.HIPS: ["hips", "pelvis", "root"],
        BoneType.SPINE: ["spine", "spine0", "spine_01"],
        BoneType.CHEST: ["chest", "spine2", "spine_02", "upper_spine"],
        BoneType.NECK: ["neck", "neck_01"],
        BoneType.HEAD: ["head"],
        BoneType.LEFT_SHOULDER: ["leftshoulder", "l_shoulder", "shoulder_l", "shoulder.l"],
        BoneType.LEFT_UPPER_ARM: ["leftupperarm", "l_upperarm", "arm_l", "arm.l"],
        BoneType.LEFT_LOWER_ARM: ["leftlowerarm", "l_forearm", "forearm_l", "forearm.l"],
        BoneType.LEFT_HAND: ["lefthand", "l_hand", "hand_l", "hand.l"],
        BoneType.RIGHT_SHOULDER: ["rightshoulder", "r_shoulder", "shoulder_r", "shoulder.r"],
        BoneType.RIGHT_UPPER_ARM: ["rightupperarm", "r_upperarm", "arm_r", "arm.r"],
        BoneType.RIGHT_LOWER_ARM: ["rightlowerarm", "r_forearm", "forearm_r", "forearm.r"],
        BoneType.RIGHT_HAND: ["righthand", "r_hand", "hand_r", "hand.r"],
        BoneType.LEFT_UPPER_LEG: ["leftupleg", "l_thigh", "thigh_l", "thigh.l", "leg_l"],
        BoneType.LEFT_LOWER_LEG: ["leftleg", "l_calf", "calf_l", "shin_l", "shin.l"],
        BoneType.LEFT_FOOT: ["leftfoot", "l_foot", "foot_l", "foot.l"],
        BoneType.RIGHT_UPPER_LEG: ["rightupleg", "r_thigh", "thigh_r", "thigh.r", "leg_r"],
        BoneType.RIGHT_LOWER_LEG: ["rightleg", "r_calf", "calf_r", "shin_r", "shin.r"],
        BoneType.RIGHT_FOOT: ["rightfoot", "r_foot", "foot_r", "foot.r"],
    }
    
    def __init__(self, config: Optional[RetargetConfig] = None):
        """
        Initialize retargeter.
        
        Args:
            config: Retargeting configuration
        """
        self._config = config or RetargetConfig()
        self._bone_mappings: list[BoneMapping] = []
        self._source_rest_pose: dict[str, BoneTransform] = {}
        self._target_rest_pose: dict[str, BoneTransform] = {}
    
    def auto_map_bones(self,
                       source_bones: list[str],
                       target_bones: list[str]) -> list[BoneMapping]:
        """
        Automatically map bones between rigs.
        
        Args:
            source_bones: Bone names in source rig
            target_bones: Bone names in target rig
            
        Returns:
            List of bone mappings
        """
        mappings = []
        
        for bone_type, patterns in self.BONE_PATTERNS.items():
            source_match = self._find_matching_bone(source_bones, patterns)
            target_match = self._find_matching_bone(target_bones, patterns)
            
            if source_match and target_match:
                mappings.append(BoneMapping(
                    source_bone=source_match,
                    target_bone=target_match,
                    bone_type=bone_type
                ))
        
        self._bone_mappings = mappings
        return mappings
    
    def _find_matching_bone(self, bones: list[str], patterns: list[str]) -> Optional[str]:
        """Find a bone matching any of the patterns."""
        for bone in bones:
            bone_lower = bone.lower().replace("_", "").replace(".", "")
            for pattern in patterns:
                if pattern.lower().replace("_", "") in bone_lower:
                    return bone
        return None
    
    def set_bone_mapping(self, mapping: BoneMapping):
        """Add or update a bone mapping."""
        # Remove existing mapping for this bone type
        self._bone_mappings = [m for m in self._bone_mappings 
                              if m.bone_type != mapping.bone_type]
        self._bone_mappings.append(mapping)
    
    def set_rest_poses(self,
                       source_rest: dict[str, BoneTransform],
                       target_rest: dict[str, BoneTransform]):
        """
        Set rest poses for source and target rigs.
        
        Args:
            source_rest: Rest pose transforms for source rig
            target_rest: Rest pose transforms for target rig
        """
        self._source_rest_pose = source_rest
        self._target_rest_pose = target_rest
    
    def retarget(self, animation: Animation) -> Animation:
        """
        Retarget an animation to the target rig.
        
        Args:
            animation: Source animation
            
        Returns:
            Retargeted animation
        """
        retargeted = Animation(
            name=f"{animation.name}_retargeted",
            duration=animation.duration,
            fps=animation.fps
        )
        
        for keyframe in animation.keyframes:
            retargeted_kf = self._retarget_keyframe(keyframe)
            retargeted.keyframes.append(retargeted_kf)
        
        # Smooth transitions if enabled
        if self._config.smooth_transitions:
            retargeted = self._smooth_animation(retargeted)
        
        return retargeted
    
    def _retarget_keyframe(self, keyframe: AnimationKeyframe) -> AnimationKeyframe:
        """Retarget a single keyframe."""
        retargeted = AnimationKeyframe(time=keyframe.time)
        
        for mapping in self._bone_mappings:
            # Skip fingers/toes if configured
            if self._config.ignore_fingers and "finger" in mapping.bone_type.value:
                continue
            if self._config.ignore_toes and "toes" in mapping.bone_type.value:
                continue
            
            source_transform = keyframe.transforms.get(mapping.source_bone)
            if not source_transform:
                continue
            
            # Get rest poses
            source_rest = self._source_rest_pose.get(mapping.source_bone, BoneTransform())
            target_rest = self._target_rest_pose.get(mapping.target_bone, BoneTransform())
            
            # Calculate relative rotation from rest pose
            relative_rotation = source_rest.rotation.inverse() * source_transform.rotation
            
            # Apply rotation offset
            adjusted_rotation = mapping.rotation_offset * relative_rotation
            
            # Handle mirroring
            if mapping.mirror:
                adjusted_rotation = Quaternion(
                    adjusted_rotation.x,
                    -adjusted_rotation.y,
                    -adjusted_rotation.z,
                    adjusted_rotation.w
                )
            
            # Apply to target rest pose
            target_rotation = target_rest.rotation * adjusted_rotation
            
            # Handle position (with scale)
            target_position = source_transform.position
            if self._config.apply_scale_correction:
                target_position = target_position * mapping.scale_factor
            
            # Handle root motion
            if mapping.bone_type == BoneType.HIPS and not self._config.preserve_root_motion:
                target_position = target_rest.position
            
            retargeted.transforms[mapping.target_bone] = BoneTransform(
                position=target_position,
                rotation=target_rotation,
                scale=source_transform.scale
            )
        
        return retargeted
    
    def _smooth_animation(self, animation: Animation) -> Animation:
        """Apply smoothing to animation."""
        if len(animation.keyframes) < 3:
            return animation
        
        smoothed = Animation(
            name=animation.name,
            duration=animation.duration,
            fps=animation.fps
        )
        
        frames = self._config.transition_frames
        
        for i, kf in enumerate(animation.keyframes):
            smoothed_kf = AnimationKeyframe(time=kf.time)
            
            for bone_name, transform in kf.transforms.items():
                # Get neighboring frames
                prev_frames = [animation.keyframes[max(0, i-j)] for j in range(1, frames+1)]
                next_frames = [animation.keyframes[min(len(animation.keyframes)-1, i+j)] for j in range(1, frames+1)]
                
                # Average rotations using slerp
                smoothed_rot = transform.rotation
                
                for j, (prev, next_kf) in enumerate(zip(prev_frames, next_frames)):
                    weight = 1.0 / (j + 2)
                    
                    prev_t = prev.transforms.get(bone_name)
                    next_t = next_kf.transforms.get(bone_name)
                    
                    if prev_t:
                        smoothed_rot = Quaternion.slerp(smoothed_rot, prev_t.rotation, weight * 0.3)
                    if next_t:
                        smoothed_rot = Quaternion.slerp(smoothed_rot, next_t.rotation, weight * 0.3)
                
                smoothed_kf.transforms[bone_name] = BoneTransform(
                    position=transform.position,
                    rotation=smoothed_rot.normalized(),
                    scale=transform.scale
                )
            
            smoothed.keyframes.append(smoothed_kf)
        
        return smoothed
    
    def calculate_scale_factors(self) -> dict[BoneType, float]:
        """
        Calculate bone length ratios between rigs.
        
        Returns:
            Scale factors by bone type
        """
        factors = {}
        
        bone_pairs = [
            (BoneType.LEFT_UPPER_ARM, BoneType.LEFT_LOWER_ARM),
            (BoneType.LEFT_LOWER_ARM, BoneType.LEFT_HAND),
            (BoneType.LEFT_UPPER_LEG, BoneType.LEFT_LOWER_LEG),
            (BoneType.LEFT_LOWER_LEG, BoneType.LEFT_FOOT),
            (BoneType.SPINE, BoneType.CHEST),
            (BoneType.NECK, BoneType.HEAD),
        ]
        
        for start_type, end_type in bone_pairs:
            source_length = self._get_bone_length(start_type, end_type, self._source_rest_pose)
            target_length = self._get_bone_length(start_type, end_type, self._target_rest_pose)
            
            if source_length > 0 and target_length > 0:
                factors[start_type] = target_length / source_length
        
        return factors
    
    def _get_bone_length(self, 
                         start_type: BoneType, 
                         end_type: BoneType,
                         rest_pose: dict[str, BoneTransform]) -> float:
        """Calculate length between two bones."""
        start_mapping = next((m for m in self._bone_mappings if m.bone_type == start_type), None)
        end_mapping = next((m for m in self._bone_mappings if m.bone_type == end_type), None)
        
        if not start_mapping or not end_mapping:
            return 0.0
        
        start_pos = rest_pose.get(start_mapping.target_bone, BoneTransform()).position
        end_pos = rest_pose.get(end_mapping.target_bone, BoneTransform()).position
        
        return (end_pos - start_pos).length()
    
    def save_mapping(self, path: Path):
        """Save bone mapping to file."""
        data = {
            "mappings": [
                {
                    "source_bone": m.source_bone,
                    "target_bone": m.target_bone,
                    "bone_type": m.bone_type.value,
                    "scale_factor": m.scale_factor,
                    "mirror": m.mirror
                }
                for m in self._bone_mappings
            ],
            "config": {
                "preserve_root_motion": self._config.preserve_root_motion,
                "apply_scale_correction": self._config.apply_scale_correction,
                "smooth_transitions": self._config.smooth_transitions,
                "transition_frames": self._config.transition_frames,
                "ignore_fingers": self._config.ignore_fingers,
                "ignore_toes": self._config.ignore_toes
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_mapping(self, path: Path):
        """Load bone mapping from file."""
        with open(path) as f:
            data = json.load(f)
        
        self._bone_mappings = [
            BoneMapping(
                source_bone=m["source_bone"],
                target_bone=m["target_bone"],
                bone_type=BoneType(m["bone_type"]),
                scale_factor=m.get("scale_factor", 1.0),
                mirror=m.get("mirror", False)
            )
            for m in data.get("mappings", [])
        ]
        
        config = data.get("config", {})
        self._config = RetargetConfig(
            preserve_root_motion=config.get("preserve_root_motion", True),
            apply_scale_correction=config.get("apply_scale_correction", True),
            smooth_transitions=config.get("smooth_transitions", True),
            transition_frames=config.get("transition_frames", 3),
            ignore_fingers=config.get("ignore_fingers", False),
            ignore_toes=config.get("ignore_toes", False)
        )


# Factory function
def create_retargeter(config: RetargetConfig = None) -> AnimationRetargeter:
    """Create an animation retargeter."""
    return AnimationRetargeter(config)


__all__ = [
    'AnimationRetargeter',
    'Animation',
    'AnimationKeyframe',
    'BoneMapping',
    'BoneTransform',
    'BoneType',
    'RetargetConfig',
    'Vector3',
    'Quaternion',
    'create_retargeter'
]
