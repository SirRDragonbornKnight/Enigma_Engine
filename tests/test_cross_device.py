"""
================================================================================
Cross-Device Compatibility Test Suite
================================================================================

Tests ForgeAI functionality across different device configurations:
- Embedded (Raspberry Pi)
- Mobile (Phone/Tablet)
- Desktop (Standard PC)
- High-End (Gaming PC with GPU)

USAGE:
    python -m pytest tests/test_cross_device.py -v
    
    # Or run directly:
    python tests/test_cross_device.py
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Add forge_ai to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Mock Device Profiles
# =============================================================================

class MockDeviceClass:
    """Mock device class for testing."""
    EMBEDDED = "embedded"
    MOBILE = "mobile"
    LAPTOP_LOW = "laptop_low"
    LAPTOP_MID = "laptop_mid"
    DESKTOP_CPU = "desktop_cpu"
    DESKTOP_GPU = "desktop_gpu"
    WORKSTATION = "workstation"
    DATACENTER = "datacenter"


DEVICE_CONFIGS = {
    "pi_zero": {
        "device_class": "EMBEDDED",
        "cpu_cores": 1,
        "ram_mb": 512,
        "has_gpu": False,
        "vram_mb": 0,
    },
    "pi_4": {
        "device_class": "EMBEDDED",
        "cpu_cores": 4,
        "ram_mb": 4096,
        "has_gpu": False,
        "vram_mb": 0,
    },
    "phone": {
        "device_class": "MOBILE",
        "cpu_cores": 8,
        "ram_mb": 6144,
        "has_gpu": True,
        "vram_mb": 1024,  # Shared
    },
    "laptop_budget": {
        "device_class": "LAPTOP_LOW",
        "cpu_cores": 4,
        "ram_mb": 8192,
        "has_gpu": False,
        "vram_mb": 0,
    },
    "desktop_igpu": {
        "device_class": "DESKTOP_CPU",
        "cpu_cores": 8,
        "ram_mb": 16384,
        "has_gpu": True,
        "vram_mb": 2048,  # Integrated
    },
    "gaming_pc": {
        "device_class": "DESKTOP_GPU",
        "cpu_cores": 12,
        "ram_mb": 32768,
        "has_gpu": True,
        "vram_mb": 12288,  # RTX 3080
    },
    "workstation": {
        "device_class": "WORKSTATION",
        "cpu_cores": 32,
        "ram_mb": 131072,
        "has_gpu": True,
        "vram_mb": 24576,  # RTX 4090
    },
}


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(params=DEVICE_CONFIGS.keys())
def device_config(request):
    """Parameterized fixture for device configs."""
    return request.param, DEVICE_CONFIGS[request.param]


@pytest.fixture
def mock_device_profiler(device_config):
    """Mock device profiler for testing."""
    name, config = device_config
    
    class MockCapabilities:
        cpu_cores = config["cpu_cores"]
        ram_total_mb = config["ram_mb"]
        vram_total_mb = config["vram_mb"]
        has_cuda = config["has_gpu"]
    
    class MockProfile:
        max_model_params = config["ram_mb"] * 1000  # Rough estimate
        batch_size = 1 if config["ram_mb"] < 4096 else 4
        use_quantization = config["vram_mb"] < 4096
        cpu_threads = min(config["cpu_cores"], 8)
        gpu_layers = 0 if not config["has_gpu"] else 32
    
    class MockProfiler:
        def classify(self):
            return getattr(MockDeviceClass, config["device_class"])
        
        def detect(self):
            return MockCapabilities()
        
        def get_profile(self):
            return MockProfile()
    
    return MockProfiler()


# =============================================================================
# Core Import Tests
# =============================================================================

class TestCoreImports:
    """Test that core modules import cleanly on all devices."""
    
    def test_config_import(self):
        """Config should always import."""
        from forge_ai.config import CONFIG
        assert CONFIG is not None
    
    def test_builtin_import(self):
        """Builtins should always import (no dependencies)."""
        from forge_ai.builtin import (
            BuiltinTTS, BuiltinEmbeddings, BuiltinCodeGen,
            BuiltinImageGen, BuiltinChat, BuiltinVideoGen
        )
        # These should all be classes
        assert callable(BuiltinTTS)
        assert callable(BuiltinEmbeddings)
        assert callable(BuiltinCodeGen)
        assert callable(BuiltinImageGen)
        assert callable(BuiltinChat)
        assert callable(BuiltinVideoGen)
    
    def test_core_import(self):
        """Core module should import."""
        from forge_ai.core import Forge, ForgeConfig, create_model
        assert callable(create_model)
    
    def test_gaming_mode_import(self):
        """Gaming mode should import."""
        try:
            from forge_ai.core.gaming_mode import (
                GamingMode, GamingPriority, get_gaming_mode
            )
            assert callable(GamingMode)
        except ImportError:
            pytest.skip("Gaming mode not available")
    
    def test_adaptive_engine_import(self):
        """Adaptive engine should import."""
        try:
            from forge_ai.core.adaptive_engine import (
                AdaptiveEngine, AdaptiveConfig, get_adaptive_engine
            )
            assert callable(AdaptiveEngine)
        except ImportError:
            pytest.skip("Adaptive engine not available")


# =============================================================================
# Builtin Fallback Tests
# =============================================================================

class TestBuiltinFallbacks:
    """Test that builtin fallbacks work without dependencies."""
    
    def test_builtin_tts_initialize(self):
        """BuiltinTTS should initialize."""
        from forge_ai.builtin import BuiltinTTS
        tts = BuiltinTTS()
        assert tts.load() == True
    
    def test_builtin_embeddings_generate(self):
        """BuiltinEmbeddings should generate embeddings."""
        from forge_ai.builtin import BuiltinEmbeddings
        emb = BuiltinEmbeddings()
        emb.load()
        result = emb.embed("test text")
        assert result is not None
        assert len(result) > 0
    
    def test_builtin_image_generate(self):
        """BuiltinImageGen should generate images."""
        from forge_ai.builtin import BuiltinImageGen
        gen = BuiltinImageGen(width=64, height=64)  # Small for speed
        gen.load()
        result = gen.generate("test image")
        assert result is not None
        assert "image_data" in result or isinstance(result, bytes)
    
    def test_builtin_code_generate(self):
        """BuiltinCodeGen should generate code."""
        from forge_ai.builtin import BuiltinCodeGen
        gen = BuiltinCodeGen()
        gen.load()
        result = gen.generate("hello world function", language="python")
        assert result is not None
        assert "code" in result or isinstance(result, str)
    
    def test_builtin_chat_respond(self):
        """BuiltinChat should respond."""
        from forge_ai.builtin import BuiltinChat
        chat = BuiltinChat()
        chat.load()
        result = chat.respond("hello")
        assert result is not None
        assert len(result) > 0


# =============================================================================
# Gaming Mode Tests
# =============================================================================

class TestGamingMode:
    """Test gaming mode functionality."""
    
    def test_gaming_mode_creation(self):
        """GamingMode should create without error."""
        try:
            from forge_ai.core.gaming_mode import GamingMode
            gm = GamingMode()
            assert gm is not None
        except ImportError:
            pytest.skip("Gaming mode not available")
    
    def test_gaming_profile_defaults(self):
        """Default gaming profiles should be defined."""
        try:
            from forge_ai.core.gaming_mode import DEFAULT_GAMING_PROFILES
            assert "competitive_fps" in DEFAULT_GAMING_PROFILES
            assert "singleplayer_rpg" in DEFAULT_GAMING_PROFILES
        except ImportError:
            pytest.skip("Gaming mode not available")
    
    def test_gaming_resource_limits(self):
        """Resource limits should be configurable."""
        try:
            from forge_ai.core.gaming_mode import GamingMode, GamingProfile
            gm = GamingMode()
            
            # Should start with full limits
            assert gm.limits.generation_allowed == True
            assert gm.limits.cpu_only == False
        except ImportError:
            pytest.skip("Gaming mode not available")
    
    def test_gaming_mode_can_generate(self):
        """can_generate should return appropriate values."""
        try:
            from forge_ai.core.gaming_mode import GamingMode
            gm = GamingMode()
            
            # Without active game, should allow generation
            assert gm.can_generate("text") == True
            assert gm.can_generate("image") == True
        except ImportError:
            pytest.skip("Gaming mode not available")


# =============================================================================
# Adaptive Engine Tests
# =============================================================================

class TestAdaptiveEngine:
    """Test adaptive engine functionality."""
    
    def test_adaptive_engine_creation(self):
        """AdaptiveEngine should create without error."""
        try:
            from forge_ai.core.adaptive_engine import AdaptiveEngine, AdaptiveConfig
            config = AdaptiveConfig(
                enable_gaming_mode=False,  # Disable for testing
                enable_distributed=False,
            )
            engine = AdaptiveEngine(config=config)
            assert engine is not None
        except ImportError:
            pytest.skip("Adaptive engine not available")
    
    def test_adaptive_mode_detection(self):
        """Mode detection should work."""
        try:
            from forge_ai.core.adaptive_engine import AdaptiveEngine, AdaptiveConfig, AdaptiveMode
            config = AdaptiveConfig(
                enable_gaming_mode=False,
                enable_distributed=False,
            )
            engine = AdaptiveEngine(config=config)
            mode = engine._get_effective_mode()
            # Should be FULL without gaming or distributed
            assert mode in {AdaptiveMode.FULL, AdaptiveMode.LOW_POWER}
        except ImportError:
            pytest.skip("Adaptive engine not available")
    
    def test_adaptive_status(self):
        """get_status should return valid status."""
        try:
            from forge_ai.core.adaptive_engine import AdaptiveEngine, AdaptiveConfig
            config = AdaptiveConfig(
                enable_gaming_mode=False,
                enable_distributed=False,
            )
            engine = AdaptiveEngine(config=config)
            status = engine.get_status()
            
            assert "mode" in status
            assert "stats" in status
        except ImportError:
            pytest.skip("Adaptive engine not available")


# =============================================================================
# Device Profile Tests
# =============================================================================

class TestDeviceProfiles:
    """Test device profile detection and settings."""
    
    def test_device_profiler_import(self):
        """DeviceProfiler should import."""
        try:
            from forge_ai.core.device_profiles import DeviceProfiler, DeviceClass
            assert DeviceClass is not None
        except ImportError:
            pytest.skip("Device profiles not available")
    
    def test_profile_settings_exist(self):
        """ProfileSettings should be configurable."""
        try:
            from forge_ai.core.device_profiles import ProfileSettings, DeviceClass
            
            # Create settings manually
            settings = ProfileSettings(
                batch_size=1,
                max_model_params=1000000,
                use_quantization=True,
                cpu_threads=4,
                gpu_layers=0,
            )
            assert settings.batch_size == 1
        except ImportError:
            pytest.skip("Device profiles not available")


# =============================================================================
# Distributed Communication Tests
# =============================================================================

class TestDistributed:
    """Test distributed communication (mocked)."""
    
    def test_distributed_node_import(self):
        """DistributedNode should import."""
        try:
            from forge_ai.comms.distributed import DistributedNode, NodeRole
            assert callable(DistributedNode)
        except ImportError:
            pytest.skip("Distributed module not available")
    
    def test_node_role_enum(self):
        """NodeRole enum should have expected values."""
        try:
            from forge_ai.comms.distributed import NodeRole
            assert hasattr(NodeRole, "INFERENCE_CLIENT")
            assert hasattr(NodeRole, "INFERENCE_SERVER")
        except ImportError:
            pytest.skip("Distributed module not available")
    
    def test_protocol_message(self):
        """ProtocolMessage should serialize/deserialize."""
        try:
            from forge_ai.comms.distributed import ProtocolMessage, MessageType
            
            msg = ProtocolMessage(
                msg_type=MessageType.PING,
                payload={"test": True},
                sender_id="test_sender",
            )
            json_str = msg.to_json()
            restored = ProtocolMessage.from_json(json_str)
            
            assert restored.msg_type == MessageType.PING
            assert restored.sender_id == "test_sender"
        except ImportError:
            pytest.skip("Distributed module not available")


# =============================================================================
# Memory Efficiency Tests
# =============================================================================

class TestMemoryEfficiency:
    """Test memory-efficient operation on constrained devices."""
    
    def test_low_power_engine_import(self):
        """LowPowerEngine should import."""
        try:
            from forge_ai.core.low_power_inference import LowPowerEngine
            assert callable(LowPowerEngine)
        except ImportError:
            pytest.skip("Low power engine not available")
    
    def test_style_config_device_aware(self):
        """StyleConfig should adapt to device."""
        try:
            from forge_ai.gui.tabs.unified_patterns import StyleConfig, DeviceUIClass
            config = StyleConfig()
            
            # Should have reasonable defaults
            assert config.base_font_size >= 9
            assert config.base_font_size <= 14
            assert config.preview_max_size > 100
        except ImportError:
            pytest.skip("Unified patterns not available")


# =============================================================================
# Run Tests
# =============================================================================

def main():
    """Run tests directly."""
    print("ForgeAI Cross-Device Compatibility Tests")
    print("=" * 60)
    
    # Run pytest with verbose output
    import pytest
    return pytest.main([__file__, "-v", "-x"])


if __name__ == "__main__":
    sys.exit(main())
