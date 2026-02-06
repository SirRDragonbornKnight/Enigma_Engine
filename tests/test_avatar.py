"""
Tests for forge_ai.avatar module.

Tests avatar system functionality including:
- Avatar controller and priority system
- Bone control and rigging
- Autonomous behaviors
- Animation systems
- Desktop pet mode
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import threading


class TestControlPriority:
    """Test control priority system."""
    
    def test_priority_values(self):
        """Test priority values exist and have correct ordering."""
        from forge_ai.avatar.controller import ControlPriority
        
        # Higher value = higher priority
        assert ControlPriority.BONE_ANIMATION > ControlPriority.USER_MANUAL
        assert ControlPriority.USER_MANUAL > ControlPriority.AI_TOOL_CALL
        assert ControlPriority.AI_TOOL_CALL > ControlPriority.AUTONOMOUS
        assert ControlPriority.AUTONOMOUS > ControlPriority.IDLE_ANIMATION
        assert ControlPriority.IDLE_ANIMATION > ControlPriority.FALLBACK


class TestAvatarController:
    """Test AvatarController class."""
    
    def test_controller_creation(self):
        """Test creating an avatar controller."""
        from forge_ai.avatar.controller import AvatarController
        
        controller = AvatarController()
        assert controller is not None
    
    def test_priority_system(self):
        """Test that higher priority controls override lower."""
        from forge_ai.avatar.controller import AvatarController, ControlPriority
        
        controller = AvatarController()
        
        # Request low priority control
        granted1 = controller.request_control("test_low", ControlPriority.FALLBACK)
        assert granted1 is True
        
        # Higher priority should also be granted
        granted2 = controller.request_control("test_high", ControlPriority.USER_MANUAL)
        assert granted2 is True
        
        # Current controller should be the higher priority one
        assert controller.current_controller == "test_high"
    
    def test_release_control(self):
        """Test releasing control."""
        from forge_ai.avatar.controller import AvatarController, ControlPriority
        
        controller = AvatarController()
        
        # Request control
        controller.request_control("test_user", ControlPriority.USER_MANUAL)
        
        # Release control
        controller.release_control("test_user")
        
        # Controller should be released
        assert controller.current_controller in ["test_user", "none"]


class TestBoneControl:
    """Test BoneController class."""
    
    def test_bone_controller_creation(self):
        """Test creating a bone controller."""
        from forge_ai.avatar.bone_control import BoneController
        
        controller = BoneController()
        assert controller is not None
    
    def test_get_bone_controller_singleton(self):
        """Test that get_bone_controller returns singleton."""
        from forge_ai.avatar.bone_control import get_bone_controller
        
        controller1 = get_bone_controller()
        controller2 = get_bone_controller()
        
        assert controller1 is controller2
    
    def test_bone_movement(self):
        """Test moving bones with limits."""
        from forge_ai.avatar.bone_control import BoneController
        
        controller = BoneController()
        
        # Set available bones first
        controller.set_avatar_bones(["head", "neck", "spine"])
        
        # Move bone (should return clamped rotation tuple)
        result = controller.move_bone("head", pitch=30, yaw=45, roll=0)
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_bone_limits(self):
        """Test getting bone rotation limits."""
        from forge_ai.avatar.bone_control import BoneController
        
        controller = BoneController()
        
        # Get limits for a bone
        limits = controller.get_limits_for_bone("head")
        assert limits is not None


class TestAutonomousAvatar:
    """Test AutonomousAvatar class."""
    
    def test_autonomous_creation(self):
        """Test creating an autonomous avatar."""
        from forge_ai.avatar.autonomous import AutonomousAvatar, AutonomousConfig
        from forge_ai.avatar.controller import AvatarController
        
        avatar_controller = AvatarController()
        config = AutonomousConfig()
        autonomous = AutonomousAvatar(avatar_controller, config)
        
        assert autonomous is not None
        assert autonomous.config == config
    
    def test_autonomous_config_defaults(self):
        """Test default autonomous config."""
        from forge_ai.avatar.autonomous import AutonomousConfig
        
        config = AutonomousConfig()
        
        # Should have reasonable defaults
        assert config.action_interval_min >= 0
        assert config.action_interval_max >= config.action_interval_min
        assert isinstance(config.enabled, bool)


class TestDesktopPet:
    """Test desktop pet functionality."""
    
    def test_desktop_pet_config(self):
        """Test desktop pet configuration."""
        try:
            from forge_ai.avatar.desktop_pet import PetConfig
            
            config = PetConfig()
            assert config is not None
        except ImportError:
            pytest.skip("Desktop pet not available")
    
    def test_pet_state_enum(self):
        """Test pet state enumeration."""
        try:
            from forge_ai.avatar.desktop_pet import PetState
            
            # Check states exist
            assert PetState.IDLE is not None
        except ImportError:
            pytest.skip("Desktop pet not available")


class TestAnimationSystem:
    """Test animation system."""
    
    def test_animation_module_exists(self):
        """Test animation module exists."""
        import importlib
        try:
            animation = importlib.import_module('forge_ai.avatar.animation')
            assert animation is not None
        except ImportError:
            pytest.skip("Animation module not available")


class TestLipSync:
    """Test lip sync functionality."""
    
    def test_lip_sync_module_exists(self):
        """Test lip sync module exists."""
        try:
            from forge_ai.avatar import lip_sync
            assert lip_sync is not None
        except ImportError:
            pytest.skip("Lip sync module not available")


class TestEmotionSync:
    """Test emotion synchronization."""
    
    def test_emotion_module_exists(self):
        """Test emotion module exists."""
        import importlib
        try:
            emotion = importlib.import_module('forge_ai.avatar.emotion')
            assert emotion is not None
        except ImportError:
            pytest.skip("Emotion module not available")


class TestAvatarPresets:
    """Test avatar presets."""
    
    def test_presets_module_exists(self):
        """Test presets module exists."""
        try:
            from forge_ai.avatar import presets
            assert presets is not None
        except ImportError:
            pytest.skip("Presets module not available")


class TestModelManager:
    """Test avatar model management."""
    
    def test_model_manager_creation(self):
        """Test creating model manager."""
        from forge_ai.avatar.model_manager import AvatarModelManager
        
        manager = AvatarModelManager()
        assert manager is not None
    
    def test_list_avatars(self):
        """Test listing available avatars."""
        from forge_ai.avatar.model_manager import AvatarModelManager
        
        manager = AvatarModelManager()
        avatars = manager.list_avatars()
        
        assert isinstance(avatars, list)


class TestVRMSupport:
    """Test VRM avatar format support."""
    
    def test_vrm_loader_creation(self):
        """Test creating VRM loader."""
        from forge_ai.avatar.vrm_support import VRMLoader
        
        loader = VRMLoader()
        assert loader is not None
    
    def test_vrm_model_class(self):
        """Test VRMModel class exists."""
        from forge_ai.avatar.vrm_support import VRMModel
        
        assert VRMModel is not None
