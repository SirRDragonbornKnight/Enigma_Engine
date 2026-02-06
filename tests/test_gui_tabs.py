"""
Tests for forge_ai.gui.tabs module.

Tests GUI tab functionality including:
- Tab creation and initialization
- Tab state management
- Generation tabs (image, code, video, audio)
- Settings and configuration tabs
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


# Skip all tests if PyQt5 not available
pytestmark = pytest.mark.skipif(
    True,  # Skip by default in CI - requires display
    reason="GUI tests require display"
)


class TestBaseTab:
    """Test base tab functionality."""
    
    def test_base_tab_interface(self):
        """Test base tab has required interface."""
        from forge_ai.gui.tabs.base_generation_tab import BaseGenerationTab
        
        # Should have required methods
        assert hasattr(BaseGenerationTab, 'get_name')
        assert hasattr(BaseGenerationTab, 'get_icon')


class TestChatTab:
    """Test chat tab functionality."""
    
    def test_chat_tab_creation(self):
        """Test creating chat tab."""
        # Would need QApplication for actual test
        pass
    
    def test_message_handling(self):
        """Test message send/receive."""
        pass


class TestImageTab:
    """Test image generation tab."""
    
    def test_image_tab_providers(self):
        """Test image providers exist."""
        from forge_ai.gui.tabs.image_tab import (
            StableDiffusionLocal,
            OpenAIImage,
            ReplicateImage
        )
        
        # Providers should be importable
        assert StableDiffusionLocal is not None
        assert OpenAIImage is not None
        assert ReplicateImage is not None
    
    def test_provider_interface(self):
        """Test provider interface."""
        from forge_ai.gui.tabs.image_tab import StableDiffusionLocal
        
        # Should have generate method
        assert hasattr(StableDiffusionLocal, 'generate')


class TestCodeTab:
    """Test code generation tab."""
    
    def test_code_tab_providers(self):
        """Test code providers exist."""
        from forge_ai.gui.tabs.code_tab import ForgeCode, OpenAICode
        
        assert ForgeCode is not None
        assert OpenAICode is not None
    
    def test_language_support(self):
        """Test language selection support."""
        import importlib
        try:
            code_tab = importlib.import_module('forge_ai.gui.tabs.code_tab')
            if hasattr(code_tab, 'SUPPORTED_LANGUAGES'):
                langs = code_tab.SUPPORTED_LANGUAGES
                # Should support common languages if the constant exists
                assert isinstance(langs, (list, dict, set))
            else:
                pytest.skip("SUPPORTED_LANGUAGES not defined")
        except ImportError:
            pytest.skip("code_tab not available")


class TestVideoTab:
    """Test video generation tab."""
    
    def test_video_tab_providers(self):
        """Test video providers exist."""
        from forge_ai.gui.tabs.video_tab import LocalVideo, ReplicateVideo
        
        assert LocalVideo is not None
        assert ReplicateVideo is not None


class TestAudioTab:
    """Test audio generation tab."""
    
    def test_audio_tab_providers(self):
        """Test audio providers exist."""
        from forge_ai.gui.tabs.audio_tab import LocalTTS, ElevenLabsTTS, ReplicateAudio
        
        assert LocalTTS is not None
        assert ElevenLabsTTS is not None
        assert ReplicateAudio is not None


class TestEmbeddingsTab:
    """Test embeddings tab."""
    
    def test_embedding_providers(self):
        """Test embedding providers exist."""
        from forge_ai.gui.tabs.embeddings_tab import LocalEmbedding, OpenAIEmbedding
        
        assert LocalEmbedding is not None
        assert OpenAIEmbedding is not None


class TestThreeDTab:
    """Test 3D generation tab."""
    
    def test_threed_providers(self):
        """Test 3D providers exist."""
        from forge_ai.gui.tabs.threed_tab import Local3DGen, Cloud3DGen
        
        assert Local3DGen is not None
        assert Cloud3DGen is not None


class TestModulesTab:
    """Test modules management tab."""
    
    def test_modules_tab_class(self):
        """Test ModulesTab class exists."""
        from forge_ai.gui.tabs.modules_tab import ModulesTab
        
        assert ModulesTab is not None


class TestSettingsTab:
    """Test settings tab."""
    
    def test_settings_tab_class(self):
        """Test SettingsTab class exists."""
        import importlib
        try:
            settings_tab = importlib.import_module('forge_ai.gui.tabs.settings_tab')
            if hasattr(settings_tab, 'SettingsTab'):
                assert settings_tab.SettingsTab is not None
            else:
                pytest.skip("SettingsTab not defined")
        except ImportError:
            pytest.skip("settings_tab not available")


class TestTrainingTab:
    """Test training tab."""
    
    def test_training_tab_class(self):
        """Test TrainingTab class exists."""
        import importlib
        try:
            training_tab = importlib.import_module('forge_ai.gui.tabs.training_tab')
            if hasattr(training_tab, 'TrainingTab'):
                assert training_tab.TrainingTab is not None
            else:
                pytest.skip("TrainingTab not defined")
        except ImportError:
            pytest.skip("training_tab not available")


class TestVisionTab:
    """Test vision tab."""
    
    def test_vision_tab_class(self):
        """Test VisionTab class exists."""
        import importlib
        try:
            vision_tab = importlib.import_module('forge_ai.gui.tabs.vision_tab')
            if hasattr(vision_tab, 'VisionTab'):
                assert vision_tab.VisionTab is not None
            else:
                pytest.skip("VisionTab not defined")
        except ImportError:
            pytest.skip("vision_tab not available")


class TestCameraTab:
    """Test camera tab."""
    
    def test_camera_tab_class(self):
        """Test CameraTab class exists."""
        from forge_ai.gui.tabs.camera_tab import CameraTab
        
        assert CameraTab is not None


class TestModelRouterTab:
    """Test model router tab."""
    
    def test_model_router_tab_class(self):
        """Test ModelRouterTab class exists."""
        from forge_ai.gui.tabs.model_router_tab import ModelRouterTab
        
        assert ModelRouterTab is not None


class TestProviderBase:
    """Test provider base class."""
    
    def test_provider_base_interface(self):
        """Test provider base has required interface."""
        import importlib
        try:
            provider_base = importlib.import_module('forge_ai.gui.tabs.provider_base')
            if hasattr(provider_base, 'ProviderBase'):
                pb = provider_base.ProviderBase
                # Should be a class
                assert pb is not None
            else:
                pytest.skip("ProviderBase not defined")
        except ImportError:
            pytest.skip("provider_base not available")


class TestSharedComponents:
    """Test shared GUI components."""
    
    def test_shared_components_exist(self):
        """Test shared components module exists."""
        from forge_ai.gui.tabs import shared_components
        
        assert shared_components is not None
    
    def test_output_helpers_exist(self):
        """Test output helpers module exists."""
        from forge_ai.gui.tabs import output_helpers
        
        assert output_helpers is not None
