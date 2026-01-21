"""
Avatar Bone Control System

Allows AI to control individual bones/joints of rigged 3D avatars
with built-in limits to prevent unnatural movements.

Each bone type has:
- Rotation limits (min/max for each axis)
- Movement speed limits (to prevent sudden jerky movements)
- Parent-child constraints (child bones follow parent)

Usage:
    from forge_ai.avatar.bone_control import BoneController, get_bone_controller
    
    controller = get_bone_controller()
    
    # AI can request bone movements
    controller.move_bone("left_arm", pitch=30, yaw=0, roll=10)
    
    # System clamps values to safe limits
    # Returns the actual applied values
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from threading import Lock
import time

logger = logging.getLogger(__name__)


@dataclass
class BoneLimits:
    """Rotation limits for a bone (in degrees)."""
    # Pitch (nodding up/down)
    pitch_min: float = -45.0
    pitch_max: float = 45.0
    
    # Yaw (turning left/right)
    yaw_min: float = -45.0
    yaw_max: float = 45.0
    
    # Roll (tilting side to side)
    roll_min: float = -30.0
    roll_max: float = 30.0
    
    # Speed limit (max degrees per second)
    speed_limit: float = 90.0
    
    def clamp(self, pitch: float, yaw: float, roll: float) -> Tuple[float, float, float]:
        """Clamp rotation values to limits."""
        pitch = max(self.pitch_min, min(self.pitch_max, pitch))
        yaw = max(self.yaw_min, min(self.yaw_max, yaw))
        roll = max(self.roll_min, min(self.roll_max, roll))
        return pitch, yaw, roll


# Standard bone limits based on human anatomy
# These prevent unnatural movements like elbows bending backwards
STANDARD_BONE_LIMITS: Dict[str, BoneLimits] = {
    # Head and neck
    "head": BoneLimits(pitch_min=-40, pitch_max=40, yaw_min=-80, yaw_max=80, roll_min=-30, roll_max=30),
    "neck": BoneLimits(pitch_min=-30, pitch_max=30, yaw_min=-60, yaw_max=60, roll_min=-20, roll_max=20),
    
    # Spine
    "spine": BoneLimits(pitch_min=-30, pitch_max=45, yaw_min=-30, yaw_max=30, roll_min=-20, roll_max=20),
    "spine1": BoneLimits(pitch_min=-20, pitch_max=30, yaw_min=-20, yaw_max=20, roll_min=-15, roll_max=15),
    "spine2": BoneLimits(pitch_min=-15, pitch_max=25, yaw_min=-15, yaw_max=15, roll_min=-10, roll_max=10),
    "chest": BoneLimits(pitch_min=-15, pitch_max=25, yaw_min=-20, yaw_max=20, roll_min=-10, roll_max=10),
    "hips": BoneLimits(pitch_min=-20, pitch_max=20, yaw_min=-30, yaw_max=30, roll_min=-15, roll_max=15),
    "pelvis": BoneLimits(pitch_min=-20, pitch_max=20, yaw_min=-30, yaw_max=30, roll_min=-15, roll_max=15),
    
    # Arms - Left
    "left_shoulder": BoneLimits(pitch_min=-30, pitch_max=30, yaw_min=-30, yaw_max=30, roll_min=-20, roll_max=20),
    "left_upper_arm": BoneLimits(pitch_min=-90, pitch_max=180, yaw_min=-90, yaw_max=90, roll_min=-90, roll_max=90),
    "left_arm": BoneLimits(pitch_min=-90, pitch_max=180, yaw_min=-90, yaw_max=90, roll_min=-90, roll_max=90),
    "left_forearm": BoneLimits(pitch_min=0, pitch_max=145, yaw_min=-90, yaw_max=90, roll_min=-5, roll_max=5),  # Elbow only bends one way!
    "left_lower_arm": BoneLimits(pitch_min=0, pitch_max=145, yaw_min=-90, yaw_max=90, roll_min=-5, roll_max=5),
    "left_hand": BoneLimits(pitch_min=-80, pitch_max=80, yaw_min=-20, yaw_max=45, roll_min=-45, roll_max=45),
    "left_wrist": BoneLimits(pitch_min=-80, pitch_max=80, yaw_min=-20, yaw_max=45, roll_min=-45, roll_max=45),
    
    # Arms - Right (mirrored)
    "right_shoulder": BoneLimits(pitch_min=-30, pitch_max=30, yaw_min=-30, yaw_max=30, roll_min=-20, roll_max=20),
    "right_upper_arm": BoneLimits(pitch_min=-90, pitch_max=180, yaw_min=-90, yaw_max=90, roll_min=-90, roll_max=90),
    "right_arm": BoneLimits(pitch_min=-90, pitch_max=180, yaw_min=-90, yaw_max=90, roll_min=-90, roll_max=90),
    "right_forearm": BoneLimits(pitch_min=0, pitch_max=145, yaw_min=-90, yaw_max=90, roll_min=-5, roll_max=5),
    "right_lower_arm": BoneLimits(pitch_min=0, pitch_max=145, yaw_min=-90, yaw_max=90, roll_min=-5, roll_max=5),
    "right_hand": BoneLimits(pitch_min=-80, pitch_max=80, yaw_min=-45, yaw_max=20, roll_min=-45, roll_max=45),
    "right_wrist": BoneLimits(pitch_min=-80, pitch_max=80, yaw_min=-45, yaw_max=20, roll_min=-45, roll_max=45),
    
    # Legs - Left
    "left_upper_leg": BoneLimits(pitch_min=-30, pitch_max=120, yaw_min=-45, yaw_max=45, roll_min=-45, roll_max=30),
    "left_thigh": BoneLimits(pitch_min=-30, pitch_max=120, yaw_min=-45, yaw_max=45, roll_min=-45, roll_max=30),
    "left_leg": BoneLimits(pitch_min=-30, pitch_max=120, yaw_min=-45, yaw_max=45, roll_min=-45, roll_max=30),
    "left_lower_leg": BoneLimits(pitch_min=-140, pitch_max=0, yaw_min=-5, yaw_max=5, roll_min=-5, roll_max=5),  # Knee only bends one way!
    "left_shin": BoneLimits(pitch_min=-140, pitch_max=0, yaw_min=-5, yaw_max=5, roll_min=-5, roll_max=5),
    "left_foot": BoneLimits(pitch_min=-45, pitch_max=45, yaw_min=-20, yaw_max=20, roll_min=-30, roll_max=30),
    "left_ankle": BoneLimits(pitch_min=-45, pitch_max=45, yaw_min=-20, yaw_max=20, roll_min=-30, roll_max=30),
    
    # Legs - Right (mirrored)
    "right_upper_leg": BoneLimits(pitch_min=-30, pitch_max=120, yaw_min=-45, yaw_max=45, roll_min=-30, roll_max=45),
    "right_thigh": BoneLimits(pitch_min=-30, pitch_max=120, yaw_min=-45, yaw_max=45, roll_min=-30, roll_max=45),
    "right_leg": BoneLimits(pitch_min=-30, pitch_max=120, yaw_min=-45, yaw_max=45, roll_min=-30, roll_max=45),
    "right_lower_leg": BoneLimits(pitch_min=-140, pitch_max=0, yaw_min=-5, yaw_max=5, roll_min=-5, roll_max=5),
    "right_shin": BoneLimits(pitch_min=-140, pitch_max=0, yaw_min=-5, yaw_max=5, roll_min=-5, roll_max=5),
    "right_foot": BoneLimits(pitch_min=-45, pitch_max=45, yaw_min=-20, yaw_max=20, roll_min=-30, roll_max=30),
    "right_ankle": BoneLimits(pitch_min=-45, pitch_max=45, yaw_min=-20, yaw_max=20, roll_min=-30, roll_max=30),
    
    # Fingers (simplified - same limits for all fingers)
    "finger": BoneLimits(pitch_min=-10, pitch_max=90, yaw_min=-15, yaw_max=15, roll_min=-5, roll_max=5),
    "thumb": BoneLimits(pitch_min=-30, pitch_max=60, yaw_min=-30, yaw_max=30, roll_min=-20, roll_max=20),
}

# Default limits for unknown bones (conservative)
DEFAULT_BONE_LIMITS = BoneLimits(
    pitch_min=-45, pitch_max=45,
    yaw_min=-45, yaw_max=45,
    roll_min=-30, roll_max=30,
    speed_limit=60.0
)


@dataclass
class BoneState:
    """Current state of a bone."""
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    last_update: float = 0.0


class BoneController:
    """
    Controls avatar bones with anatomical limits.
    
    Prevents unnatural movements by:
    1. Clamping rotations to realistic limits
    2. Limiting movement speed to prevent jerkiness
    3. Smoothing movements over time
    """
    
    def __init__(self):
        self._lock = Lock()
        self._bone_states: Dict[str, BoneState] = {}
        self._custom_limits: Dict[str, BoneLimits] = {}
        self._avatar_bones: List[str] = []
        self._callbacks: List[callable] = []
        
        # Command file for AI to write bone commands
        self._command_file = Path("data/avatar/bone_commands.json")
        self._command_file.parent.mkdir(parents=True, exist_ok=True)
        self._last_command_time = 0.0
    
    def set_avatar_bones(self, bone_names: List[str]) -> None:
        """Set the available bones for the current avatar."""
        with self._lock:
            self._avatar_bones = list(bone_names)
            # Initialize states for new bones
            for bone in bone_names:
                if bone not in self._bone_states:
                    self._bone_states[bone] = BoneState(last_update=time.time())
            logger.info(f"Bone controller initialized with {len(bone_names)} bones")
    
    def get_limits_for_bone(self, bone_name: str) -> BoneLimits:
        """Get the rotation limits for a bone."""
        # Check custom limits first
        if bone_name in self._custom_limits:
            return self._custom_limits[bone_name]
        
        # Normalize bone name for matching
        bone_lower = bone_name.lower().replace("_", "").replace("-", "").replace(" ", "")
        
        # Try to match against standard limits
        for standard_name, limits in STANDARD_BONE_LIMITS.items():
            standard_lower = standard_name.lower().replace("_", "")
            if standard_lower in bone_lower or bone_lower in standard_lower:
                return limits
        
        # Check for partial matches (e.g., "LeftArm" matches "left_arm")
        for standard_name, limits in STANDARD_BONE_LIMITS.items():
            parts = standard_name.lower().split("_")
            if all(part in bone_lower for part in parts):
                return limits
        
        return DEFAULT_BONE_LIMITS
    
    def move_bone(self, bone_name: str, pitch: float = None, yaw: float = None, 
                  roll: float = None, smooth: bool = True) -> Tuple[float, float, float]:
        """
        Move a bone to the specified rotation.
        
        Args:
            bone_name: Name of the bone to move
            pitch: Target pitch rotation (degrees)
            yaw: Target yaw rotation (degrees)
            roll: Target roll rotation (degrees)
            smooth: If True, apply smoothing to prevent jerky movements
        
        Returns:
            Tuple of (actual_pitch, actual_yaw, actual_roll) after clamping
        """
        with self._lock:
            if bone_name not in self._bone_states:
                self._bone_states[bone_name] = BoneState(last_update=time.time())
            
            state = self._bone_states[bone_name]
            limits = self.get_limits_for_bone(bone_name)
            current_time = time.time()
            
            # Get current values if not specified
            target_pitch = pitch if pitch is not None else state.pitch
            target_yaw = yaw if yaw is not None else state.yaw
            target_roll = roll if roll is not None else state.roll
            
            # Clamp to limits
            target_pitch, target_yaw, target_roll = limits.clamp(target_pitch, target_yaw, target_roll)
            
            # Apply speed limiting if smooth movement is enabled
            if smooth and state.last_update > 0:
                dt = current_time - state.last_update
                max_delta = limits.speed_limit * dt
                
                # Limit how fast each axis can change
                actual_pitch = self._limit_delta(state.pitch, target_pitch, max_delta)
                actual_yaw = self._limit_delta(state.yaw, target_yaw, max_delta)
                actual_roll = self._limit_delta(state.roll, target_roll, max_delta)
            else:
                actual_pitch, actual_yaw, actual_roll = target_pitch, target_yaw, target_roll
            
            # Update state
            state.pitch = actual_pitch
            state.yaw = actual_yaw
            state.roll = actual_roll
            state.last_update = current_time
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(bone_name, actual_pitch, actual_yaw, actual_roll)
                except Exception as e:
                    logger.error(f"Bone callback error: {e}")
            
            return actual_pitch, actual_yaw, actual_roll
    
    def _limit_delta(self, current: float, target: float, max_delta: float) -> float:
        """Limit how much a value can change."""
        delta = target - current
        if abs(delta) > max_delta:
            return current + (max_delta if delta > 0 else -max_delta)
        return target
    
    def get_bone_state(self, bone_name: str) -> Optional[BoneState]:
        """Get the current state of a bone."""
        with self._lock:
            return self._bone_states.get(bone_name)
    
    def get_all_states(self) -> Dict[str, Dict[str, float]]:
        """Get all bone states as a dictionary."""
        with self._lock:
            return {
                name: {"pitch": s.pitch, "yaw": s.yaw, "roll": s.roll}
                for name, s in self._bone_states.items()
            }
    
    def reset_all(self) -> None:
        """Reset all bones to neutral position."""
        with self._lock:
            for state in self._bone_states.values():
                state.pitch = 0.0
                state.yaw = 0.0
                state.roll = 0.0
                state.last_update = time.time()
    
    def add_callback(self, callback: callable) -> None:
        """Add a callback that's called when any bone moves."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: callable) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def check_ai_commands(self) -> None:
        """Check for AI bone commands from file."""
        if not self._command_file.exists():
            return
        
        try:
            with open(self._command_file, 'r') as f:
                cmd = json.load(f)
            
            timestamp = cmd.get("timestamp", 0)
            if timestamp <= self._last_command_time:
                return
            
            self._last_command_time = timestamp
            action = cmd.get("action", "")
            
            if action == "move_bone":
                bone_name = cmd.get("bone", "")
                pitch = cmd.get("pitch")
                yaw = cmd.get("yaw")
                roll = cmd.get("roll")
                smooth = cmd.get("smooth", True)
                
                if bone_name:
                    result = self.move_bone(bone_name, pitch, yaw, roll, smooth)
                    logger.info(f"AI moved bone {bone_name} to {result}")
            
            elif action == "pose":
                # Apply a full pose (multiple bones at once)
                bones = cmd.get("bones", {})
                for bone_name, rotations in bones.items():
                    self.move_bone(
                        bone_name,
                        pitch=rotations.get("pitch"),
                        yaw=rotations.get("yaw"),
                        roll=rotations.get("roll"),
                        smooth=cmd.get("smooth", True)
                    )
            
            elif action == "reset":
                self.reset_all()
                logger.info("AI reset all bones")
                
        except Exception as e:
            logger.error(f"Error reading bone commands: {e}")
    
    def get_bone_info_for_ai(self) -> Dict[str, Any]:
        """
        Get bone information formatted for AI.
        
        Returns a dict the AI can use to understand what movements are possible.
        """
        info = {
            "available_bones": [],
            "current_pose": {},
        }
        
        with self._lock:
            for bone_name in self._avatar_bones:
                limits = self.get_limits_for_bone(bone_name)
                state = self._bone_states.get(bone_name, BoneState())
                
                info["available_bones"].append({
                    "name": bone_name,
                    "limits": {
                        "pitch": {"min": limits.pitch_min, "max": limits.pitch_max},
                        "yaw": {"min": limits.yaw_min, "max": limits.yaw_max},
                        "roll": {"min": limits.roll_min, "max": limits.roll_max},
                    },
                    "current": {"pitch": state.pitch, "yaw": state.yaw, "roll": state.roll}
                })
                
                info["current_pose"][bone_name] = {
                    "pitch": state.pitch, "yaw": state.yaw, "roll": state.roll
                }
        
        return info
    
    def write_info_for_ai(self) -> None:
        """Write bone info to a file the AI can read."""
        info = self.get_bone_info_for_ai()
        info_path = Path("data/avatar/bone_info.json")
        info_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)


# Global instance
_bone_controller: Optional[BoneController] = None


def get_bone_controller() -> BoneController:
    """Get the global bone controller instance."""
    global _bone_controller
    if _bone_controller is None:
        _bone_controller = BoneController()
    return _bone_controller
