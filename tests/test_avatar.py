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
        
        # Set low priority control
        controller.set_control(ControlPriority.FALLBACK, "idle")
        
        # High priority should override
        controller.set_control(ControlPriority.USER_MANUAL, "wave")
        
        # Current control should be the higher priority one
        assert controller.current_control == "wave"
    
    def test_release_control(self):
        """Test releasing control returns to lower priority."""
        from forge_ai.avatar.controller import AvatarController, ControlPriority
        
        controller = AvatarController()
        
        # Set base control
        controller.set_control(ControlPriority.IDLE_ANIMATION, "idle")
        
        # Set higher priority
        controller.set_control(ControlPriority.USER_MANUAL, "point")
        
        # Release high priority
        controller.release_control(ControlPriority.USER_MANUAL)
        
        # Should fall back to lower priority
        assert controller.current_control == "idle"


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
    
    def test_bone_rotation(self):
        """Test setting bone rotation."""
        from forge_ai.avatar.bone_control import BoneController
        
        controller = BoneController()
        
        # Set rotation (should not raise)
        controller.set_bone_rotation("head", (0, 45, 0))
        
        # Get rotation
        rotation = controller.get_bone_rotation("head")
        assert rotation is not None


class TestAutonomousAvatar:
    """Test AutonomousAvatar class."""
    
    def test_autonomous_creation(self):
        """Test creating an autonomous avatar."""
        from forge_ai.avatar.autonomous import AutonomousAvatar, AutonomousConfig
        
        config = AutonomousConfig()
        avatar = AutonomousAvatar(config)
        
        assert avatar is not None
        assert avatar.config == config
    
    def test_autonomous_config_defaults(self):
        """Test default autonomous config."""
        from forge_ai.avatar.autonomous import AutonomousConfig
        
        config = AutonomousConfig()
        
        # Should have reasonable defaults
        assert config.idle_timeout >= 0
        assert config.enabled is True or config.enabled is False


class TestDesktopPet:
    """Test desktop pet functionality."""
    
    def test_desktop_pet_creation(self):
        """Test creating a desktop pet."""
        from forge_ai.avatar.desktop_pet import DesktopPet
        
        # Should work without GUI in test mode
        pet = DesktopPet(headless=True)
        assert pet is not None
    
    def test_pet_state_management(self):
        """Test pet state transitions."""
        from forge_ai.avatar.desktop_pet import DesktopPet, PetState
        
        pet = DesktopPet(headless=True)
        
        # Should have initial state
        assert pet.state is not None


class TestAnimationSystem:
    """Test animation system."""
    
    def test_animation_loading(self):
        """Test loading animations."""
        from forge_ai.avatar.animation_system import AnimationSystem
        
        system = AnimationSystem()
        assert system is not None
    
    def test_animation_playback(self):
        """Test animation playback."""
        from forge_ai.avatar.animation_system import AnimationSystem
        
        system = AnimationSystem()
        
        # Play animation (should not raise even if animation doesn't exist)
        system.play("idle")


class TestLipSync:
    """Test lip sync functionality."""
    
    def test_lip_sync_creation(self):
        """Test creating lip sync controller."""
        from forge_ai.avatar.lip_sync import LipSyncController
        
        controller = LipSyncController()
        assert controller is not None
    
    def test_phoneme_detection(self):
        """Test phoneme detection from text."""
        from forge_ai.avatar.lip_sync import LipSyncController
        
        controller = LipSyncController()
        
        # Get phonemes for text
        phonemes = controller.text_to_phonemes("Hello")
        assert isinstance(phonemes, list)


class TestEmotionSync:
    """Test emotion synchronization."""
    
    def test_emotion_sync_creation(self):
        """Test creating emotion sync."""
        from forge_ai.avatar.emotion_sync import EmotionSync
        
        sync = EmotionSync()
        assert sync is not None
    
    def test_emotion_detection(self):
        """Test detecting emotion from text."""
        from forge_ai.avatar.emotion_sync import EmotionSync
        
        sync = EmotionSync()
        
        # Detect emotion (should return valid emotion)
        emotion = sync.detect_emotion("I'm so happy!")
        assert emotion is not None


class TestAvatarPresets:
    """Test avatar presets."""
    
    def test_preset_loading(self):
        """Test loading preset avatars."""
        from forge_ai.avatar.presets import get_preset_list, load_preset
        
        presets = get_preset_list()
        assert isinstance(presets, list)
    
    def test_preset_validation(self):
        """Test preset data validation."""
        from forge_ai.avatar.presets import validate_preset_data
        
        # Valid data
        valid_data = {
            'name': 'Test',
            'model': 'default',
            'scale': 1.0
        }
        assert validate_preset_data(valid_data) is True
        
        # Invalid data
        invalid_data = {'name': ''}
        assert validate_preset_data(invalid_data) is False


class TestModelManager:
    """Test avatar model management."""
    
    def test_model_manager_creation(self):
        """Test creating model manager."""
        from forge_ai.avatar.model_manager import AvatarModelManager
        
        manager = AvatarModelManager()
        assert manager is not None
    
    def test_list_models(self):
        """Test listing available models."""
        from forge_ai.avatar.model_manager import AvatarModelManager
        
        manager = AvatarModelManager()
        models = manager.list_models()
        
        assert isinstance(models, list)


class TestVRMSupport:
    """Test VRM avatar format support."""
    
    def test_vrm_loader_creation(self):
        """Test creating VRM loader."""
        from forge_ai.avatar.vrm_support import VRMLoader
        
        loader = VRMLoader()
        assert loader is not None
    
    def test_vrm_validation(self):
        """Test VRM file validation."""
        from forge_ai.avatar.vrm_support import VRMLoader
        
        loader = VRMLoader()
        
        # Invalid path should return False
        is_valid = loader.validate("nonexistent.vrm")
        assert is_valid is False
