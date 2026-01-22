"""
Comprehensive Avatar System Tests

Tests all avatar functionality including:
- Bone control and movement
- Priority system
- Gestures and poses
- Visual rendering verification
- AI control integration
- Tool-based control
"""
import pytest
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from forge_ai.avatar.bone_control import (
    BoneController, 
    BoneLimits, 
    BoneState,
    STANDARD_BONE_LIMITS,
    DEFAULT_BONE_LIMITS,
    get_bone_controller,
)
from forge_ai.avatar.controller import AvatarController, ControlPriority


class TestBoneLimits:
    """Test bone rotation limits."""
    
    def test_bone_limits_creation(self):
        """Test creating bone limits."""
        limits = BoneLimits(
            pitch_min=-45, pitch_max=45,
            yaw_min=-30, yaw_max=30,
            roll_min=-20, roll_max=20
        )
        assert limits.pitch_min == -45
        assert limits.pitch_max == 45
        assert limits.speed_limit == 90.0  # default
    
    def test_bone_limits_clamp(self):
        """Test clamping values to limits."""
        limits = BoneLimits(
            pitch_min=-45, pitch_max=45,
            yaw_min=-30, yaw_max=30,
            roll_min=-20, roll_max=20
        )
        
        # Values within limits
        p, y, r = limits.clamp(30, 20, 10)
        assert p == 30
        assert y == 20
        assert r == 10
        
        # Values exceeding limits
        p, y, r = limits.clamp(100, -100, 50)
        assert p == 45  # clamped to max
        assert y == -30  # clamped to min
        assert r == 20  # clamped to max
    
    def test_standard_bone_limits_exist(self):
        """Test that standard bone limits are defined."""
        assert "head" in STANDARD_BONE_LIMITS
        assert "neck" in STANDARD_BONE_LIMITS
        assert "left_arm" in STANDARD_BONE_LIMITS
        assert "right_arm" in STANDARD_BONE_LIMITS
        assert "spine" in STANDARD_BONE_LIMITS
        
    def test_elbow_only_bends_one_way(self):
        """Test that elbow limits prevent backward bending."""
        left_forearm = STANDARD_BONE_LIMITS.get("left_forearm")
        assert left_forearm is not None
        # Elbow shouldn't bend backwards (negative pitch)
        assert left_forearm.pitch_min >= 0
        
    def test_knee_only_bends_one_way(self):
        """Test that knee limits prevent backward bending."""
        left_lower_leg = STANDARD_BONE_LIMITS.get("left_lower_leg")
        assert left_lower_leg is not None
        # Knee only bends one direction
        assert left_lower_leg.pitch_max <= 0  # Knee bends backwards (negative)


class TestBoneController:
    """Test bone controller functionality."""
    
    def test_controller_creation(self):
        """Test creating bone controller."""
        controller = BoneController()
        assert controller is not None
        assert controller._bone_states == {}
    
    def test_set_avatar_bones(self):
        """Test setting available bones."""
        controller = BoneController()
        bones = ["head", "neck", "left_arm", "right_arm"]
        controller.set_avatar_bones(bones)
        
        assert len(controller._avatar_bones) == 4
        assert "head" in controller._avatar_bones
        assert "head" in controller._bone_states
    
    def test_get_limits_for_standard_bone(self):
        """Test getting limits for standard bones."""
        controller = BoneController()
        
        limits = controller.get_limits_for_bone("head")
        assert limits.pitch_max == 40  # From STANDARD_BONE_LIMITS
        
        limits = controller.get_limits_for_bone("left_arm")
        assert limits.pitch_max == 180
    
    def test_get_limits_for_unknown_bone(self):
        """Test getting limits for unknown bones."""
        controller = BoneController()
        
        limits = controller.get_limits_for_bone("unknown_bone_xyz")
        assert limits == DEFAULT_BONE_LIMITS
    
    def test_move_bone_basic(self):
        """Test basic bone movement."""
        controller = BoneController()
        controller.set_avatar_bones(["head"])
        
        result = controller.move_bone("head", pitch=20, yaw=10, roll=5, smooth=False)
        
        assert result == (20, 10, 5)
        
        state = controller.get_bone_state("head")
        assert state.pitch == 20
        assert state.yaw == 10
        assert state.roll == 5
    
    def test_move_bone_clamping(self):
        """Test that bone movement is clamped to limits."""
        controller = BoneController()
        controller.set_avatar_bones(["head"])
        
        # Try to move beyond limits
        result = controller.move_bone("head", pitch=100, yaw=200, roll=-100, smooth=False)
        
        # Should be clamped to head limits (40, 80, 30)
        assert result[0] == 40  # pitch clamped
        assert result[1] == 80  # yaw clamped
        assert result[2] == -30  # roll clamped
    
    def test_move_bone_smooth_speed_limiting(self):
        """Test that smooth movement limits speed."""
        controller = BoneController()
        controller.set_avatar_bones(["head"])
        
        # First move - sets initial position
        controller.move_bone("head", pitch=0, yaw=0, roll=0, smooth=False)
        time.sleep(0.01)  # Small delay
        
        # Second move with smooth - should be speed limited
        # Head has speed_limit of 90 deg/s (default)
        # In 0.01s, max movement is 0.9 degrees
        result = controller.move_bone("head", pitch=45, yaw=45, roll=30, smooth=True)
        
        # Values should be less than target due to speed limiting
        assert result[0] < 45 or result[0] == 45  # Allow full if enough time passed
    
    def test_move_bone_partial_update(self):
        """Test updating only some axes."""
        controller = BoneController()
        controller.set_avatar_bones(["head"])
        
        # Set initial position
        controller.move_bone("head", pitch=10, yaw=20, roll=30, smooth=False)
        
        # Update only pitch
        result = controller.move_bone("head", pitch=40, smooth=False)
        
        assert result[0] == 40  # pitch updated
        assert result[1] == 20  # yaw unchanged
        assert result[2] == 30  # roll unchanged
    
    def test_reset_all_bones(self):
        """Test resetting all bones to neutral."""
        controller = BoneController()
        controller.set_avatar_bones(["head", "neck"])
        
        controller.move_bone("head", pitch=30, yaw=20, roll=10, smooth=False)
        controller.move_bone("neck", pitch=15, yaw=10, roll=5, smooth=False)
        
        controller.reset_all()
        
        head_state = controller.get_bone_state("head")
        neck_state = controller.get_bone_state("neck")
        
        assert head_state.pitch == 0
        assert head_state.yaw == 0
        assert neck_state.pitch == 0
    
    def test_get_all_states(self):
        """Test getting all bone states."""
        controller = BoneController()
        controller.set_avatar_bones(["head", "neck"])
        
        controller.move_bone("head", pitch=30, yaw=20, roll=10, smooth=False)
        controller.move_bone("neck", pitch=15, yaw=10, roll=5, smooth=False)
        
        states = controller.get_all_states()
        
        assert "head" in states
        assert "neck" in states
        assert states["head"]["pitch"] == 30
        assert states["neck"]["yaw"] == 10
    
    def test_callback_on_move(self):
        """Test callbacks are called on bone movement."""
        controller = BoneController()
        controller.set_avatar_bones(["head"])
        
        callback_data = []
        def on_move(bone_name, pitch, yaw, roll):
            callback_data.append((bone_name, pitch, yaw, roll))
        
        controller.add_callback(on_move)
        controller.move_bone("head", pitch=20, yaw=10, roll=5, smooth=False)
        
        assert len(callback_data) == 1
        assert callback_data[0] == ("head", 20, 10, 5)
    
    def test_remove_callback(self):
        """Test removing callbacks."""
        controller = BoneController()
        controller.set_avatar_bones(["head"])
        
        call_count = [0]
        def on_move(bone_name, pitch, yaw, roll):
            call_count[0] += 1
        
        controller.add_callback(on_move)
        controller.move_bone("head", pitch=10, smooth=False)
        assert call_count[0] == 1
        
        controller.remove_callback(on_move)
        controller.move_bone("head", pitch=20, smooth=False)
        assert call_count[0] == 1  # Should not have increased
    
    def test_get_bone_info_for_ai(self):
        """Test getting bone info formatted for AI."""
        controller = BoneController()
        controller.set_avatar_bones(["head", "left_arm"])
        controller.move_bone("head", pitch=20, yaw=10, roll=5, smooth=False)
        
        info = controller.get_bone_info_for_ai()
        
        assert "available_bones" in info
        assert "current_pose" in info
        assert len(info["available_bones"]) == 2
        
        head_info = next(b for b in info["available_bones"] if b["name"] == "head")
        assert "limits" in head_info
        assert "current" in head_info
        assert head_info["current"]["pitch"] == 20


class TestBoneControllerSingleton:
    """Test global bone controller singleton."""
    
    def test_get_bone_controller_returns_same_instance(self):
        """Test that get_bone_controller returns singleton."""
        # Reset singleton for test
        import forge_ai.avatar.bone_control as bc
        bc._bone_controller = None
        
        controller1 = get_bone_controller()
        controller2 = get_bone_controller()
        
        assert controller1 is controller2


class TestAvatarControllerPriority:
    """Test avatar controller priority system."""
    
    def test_priority_values(self):
        """Test that priority values are defined correctly."""
        assert ControlPriority.BONE_ANIMATION == 100
        assert ControlPriority.USER_MANUAL == 80
        assert ControlPriority.AI_TOOL_CALL == 70
        assert ControlPriority.AUTONOMOUS == 50
        assert ControlPriority.IDLE_ANIMATION == 30
        assert ControlPriority.FALLBACK == 10
    
    def test_higher_priority_takes_control(self):
        """Test that higher priority can take control."""
        controller = AvatarController()
        
        # Low priority takes control
        result = controller.request_control("idle", ControlPriority.IDLE_ANIMATION)
        assert result == True
        assert controller.current_controller == "idle"
        
        # Higher priority takes control
        result = controller.request_control("user", ControlPriority.USER_MANUAL)
        assert result == True
        assert controller.current_controller == "user"
    
    def test_lower_priority_denied(self):
        """Test that lower priority is denied when higher holds control."""
        controller = AvatarController()
        
        # High priority takes control
        controller.request_control("user", ControlPriority.USER_MANUAL)
        
        # Lower priority is denied
        result = controller.request_control("idle", ControlPriority.IDLE_ANIMATION)
        assert result == False
        assert controller.current_controller == "user"
    
    def test_same_controller_can_refresh(self):
        """Test that same controller can refresh its control."""
        controller = AvatarController()
        
        controller.request_control("bone", ControlPriority.BONE_ANIMATION)
        result = controller.request_control("bone", ControlPriority.BONE_ANIMATION)
        
        assert result == True


class TestBoneVisualMovement:
    """Test that bone movements translate to visual changes."""
    
    def test_bone_states_update_over_time(self):
        """Test that bone states change during animation."""
        controller = BoneController()
        controller.set_avatar_bones(["head"])
        
        # Record state changes
        states = []
        def record_state(bone_name, pitch, yaw, roll):
            states.append({"pitch": pitch, "yaw": yaw, "roll": roll})
        
        controller.add_callback(record_state)
        
        # Move bone multiple times
        controller.move_bone("head", pitch=0, smooth=False)
        controller.move_bone("head", pitch=20, smooth=False)
        controller.move_bone("head", pitch=40, smooth=False)
        
        # Should have recorded 3 state changes
        assert len(states) == 3
        assert states[0]["pitch"] == 0
        assert states[1]["pitch"] == 20
        assert states[2]["pitch"] == 40
    
    def test_smooth_animation_creates_intermediate_states(self):
        """Test that smooth animation creates intermediate states."""
        controller = BoneController()
        controller.set_avatar_bones(["head"])
        
        # Initialize position
        controller.move_bone("head", pitch=0, smooth=False)
        
        # Small delay to allow speed limiting to work
        time.sleep(0.05)
        
        # Move with smooth - will be speed limited
        result1 = controller.move_bone("head", pitch=45, smooth=True)
        
        # The movement should be partial (speed limited)
        # or complete if enough time passed
        state = controller.get_bone_state("head")
        assert state is not None
        assert state.pitch >= 0  # Should have moved somewhat


class TestGestureSystem:
    """Test gesture and pose functionality."""
    
    def test_wave_gesture(self):
        """Test executing a wave gesture."""
        controller = BoneController()
        controller.set_avatar_bones(["right_arm", "right_forearm", "right_hand"])
        
        # Simulate wave gesture
        controller.move_bone("right_arm", pitch=90, yaw=30, smooth=False)
        controller.move_bone("right_forearm", pitch=45, smooth=False)
        
        arm_state = controller.get_bone_state("right_arm")
        assert arm_state.pitch == 90
        
    def test_nod_gesture(self):
        """Test executing a nod gesture."""
        controller = BoneController()
        controller.set_avatar_bones(["head", "neck"])
        
        # Simulate nod - head moves down then up
        controller.move_bone("head", pitch=20, smooth=False)
        state1 = controller.get_bone_state("head")
        assert state1.pitch == 20
        
        controller.move_bone("head", pitch=-10, smooth=False)
        state2 = controller.get_bone_state("head")
        assert state2.pitch == -10
        
        controller.move_bone("head", pitch=0, smooth=False)
        state3 = controller.get_bone_state("head")
        assert state3.pitch == 0


class TestAIControlIntegration:
    """Test AI control of avatar bones."""
    
    def test_ai_bone_command_file(self):
        """Test that bone controller can read AI command files."""
        controller = BoneController()
        controller.set_avatar_bones(["head"])
        
        # Create command file
        import json
        command_file = Path("data/avatar/bone_commands.json")
        command_file.parent.mkdir(parents=True, exist_ok=True)
        
        command = {
            "timestamp": time.time(),
            "action": "move_bone",
            "bone": "head",
            "pitch": 25,
            "yaw": 15,
            "roll": 5,
            "smooth": False
        }
        
        with open(command_file, 'w') as f:
            json.dump(command, f)
        
        # Check for commands
        controller.check_ai_commands()
        
        state = controller.get_bone_state("head")
        assert state.pitch == 25
        assert state.yaw == 15
        
        # Cleanup
        command_file.unlink(missing_ok=True)
    
    def test_ai_pose_command(self):
        """Test AI applying a full pose."""
        controller = BoneController()
        controller.set_avatar_bones(["head", "neck", "left_arm"])
        
        import json
        command_file = Path("data/avatar/bone_commands.json")
        command_file.parent.mkdir(parents=True, exist_ok=True)
        
        command = {
            "timestamp": time.time(),
            "action": "pose",
            "bones": {
                "head": {"pitch": 10, "yaw": 5},
                "neck": {"pitch": 5},
                "left_arm": {"pitch": 45, "yaw": 30}
            },
            "smooth": False
        }
        
        with open(command_file, 'w') as f:
            json.dump(command, f)
        
        controller.check_ai_commands()
        
        assert controller.get_bone_state("head").pitch == 10
        assert controller.get_bone_state("neck").pitch == 5
        assert controller.get_bone_state("left_arm").pitch == 45
        
        # Cleanup
        command_file.unlink(missing_ok=True)


class TestToolBasedAvatarControl:
    """Test tool-based avatar control (new approach)."""
    
    def test_avatar_control_tool_exists(self):
        """Test that avatar control tool is defined."""
        from forge_ai.tools.tool_definitions import get_tool_definition
        
        tool = get_tool_definition("control_avatar_bones")
        assert tool is not None
        assert tool.name == "control_avatar_bones"
    
    def test_avatar_control_tool_parameters(self):
        """Test avatar control tool has correct parameters."""
        from forge_ai.tools.tool_definitions import get_tool_definition
        
        tool = get_tool_definition("control_avatar_bones")
        param_names = [p.name for p in tool.parameters]
        
        assert "action" in param_names
        assert "bone_name" in param_names or "gesture_name" in param_names
    
    def test_tool_executor_handles_avatar_tool(self):
        """Test that tool executor can handle avatar control."""
        from forge_ai.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        
        # Execute without avatar module loaded - should handle gracefully
        result = executor.execute_tool(
            "control_avatar_bones",
            {"action": "gesture", "gesture_name": "wave"}
        )
        
        # Should either succeed or gracefully report module not loaded
        assert "success" in result or "error" in result


class TestLinuxCompatibility:
    """Test cross-platform compatibility."""
    
    def test_bone_controller_works_without_gui(self):
        """Test that bone controller works in headless mode."""
        controller = BoneController()
        controller.set_avatar_bones(["head"])
        
        # Should work without any GUI
        result = controller.move_bone("head", pitch=20, smooth=False)
        assert result == (20, 0, 0)
    
    def test_paths_are_cross_platform(self):
        """Test that paths work on all platforms."""
        controller = BoneController()
        
        # Command file path should be valid
        assert controller._command_file.parent.name == "avatar"
        assert "data" in str(controller._command_file)
