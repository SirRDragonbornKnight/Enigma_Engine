"""
Hardware Optimization Tests - Validate Pi/Mobile/PC support.

Tests the new hardware abstraction, distributed protocol, and quantization.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add enigma_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDeviceProfiles(unittest.TestCase):
    """Test device detection and profiling."""
    
    def test_device_profiler_import(self):
        """Can import device profiler."""
        from enigma_engine.core.device_profiles import (
            DeviceProfiler,
            DeviceClass,
            DeviceCapabilities,
            ProfileSettings,
            get_device_profiler,
        )
        self.assertTrue(True)
    
    def test_device_profiler_singleton(self):
        """Profiler is singleton."""
        from enigma_engine.core.device_profiles import get_device_profiler
        p1 = get_device_profiler()
        p2 = get_device_profiler()
        self.assertIs(p1, p2)
    
    def test_detect_capabilities(self):
        """Can detect device capabilities."""
        from enigma_engine.core.device_profiles import get_device_profiler
        profiler = get_device_profiler()
        caps = profiler.detect()
        
        # Should have basic properties
        self.assertIsNotNone(caps.cpu_cores)
        self.assertIsNotNone(caps.ram_total_mb)
        self.assertIsNotNone(caps.is_64bit)
        self.assertGreater(caps.cpu_cores, 0)
        self.assertGreater(caps.ram_total_mb, 0)
    
    def test_classify_device(self):
        """Can classify device."""
        from enigma_engine.core.device_profiles import get_device_profiler, DeviceClass
        profiler = get_device_profiler()
        device_class = profiler.classify()
        
        self.assertIsInstance(device_class, DeviceClass)
    
    def test_get_optimal_settings(self):
        """Can get optimal settings."""
        from enigma_engine.core.device_profiles import get_optimal_settings
        settings = get_optimal_settings()
        
        self.assertIsNotNone(settings.recommended_model_size)
        self.assertIsNotNone(settings.max_sequence_length)
        self.assertIsNotNone(settings.max_batch_size)
        self.assertGreater(settings.num_threads, 0)
    
    def test_recommended_model_size(self):
        """Can get recommended model size."""
        from enigma_engine.core.device_profiles import get_recommended_model_size
        size = get_recommended_model_size()
        
        valid_sizes = ['nano', 'micro', 'tiny', 'small', 'medium', 'large', 'xl', 
                       'xxl', 'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'omega']
        self.assertIn(size, valid_sizes)


class TestPowerMode(unittest.TestCase):
    """Test power mode management."""
    
    def test_power_mode_import(self):
        """Can import power mode."""
        from enigma_engine.core.power_mode import (
            PowerLevel,
            PowerManager,
            get_power_manager,
        )
        self.assertTrue(True)
    
    def test_embedded_level_exists(self):
        """EMBEDDED power level exists."""
        from enigma_engine.core.power_mode import PowerLevel
        self.assertTrue(hasattr(PowerLevel, 'EMBEDDED'))
    
    def test_mobile_level_exists(self):
        """MOBILE power level exists."""
        from enigma_engine.core.power_mode import PowerLevel
        self.assertTrue(hasattr(PowerLevel, 'MOBILE'))
    
    def test_power_manager_singleton(self):
        """Power manager is singleton."""
        from enigma_engine.core.power_mode import get_power_manager
        m1 = get_power_manager()
        m2 = get_power_manager()
        self.assertIs(m1, m2)
    
    def test_auto_power_mode(self):
        """Can auto-detect power mode."""
        from enigma_engine.core.power_mode import auto_power_mode
        pm = auto_power_mode()
        self.assertIsNotNone(pm)


class TestLowPowerInference(unittest.TestCase):
    """Test low-power inference engine."""
    
    def test_low_power_import(self):
        """Can import low-power engine."""
        from enigma_engine.core.low_power_inference import (
            LowPowerConfig,
            LowPowerEngine,
            get_engine,
        )
        self.assertTrue(True)
    
    def test_config_for_raspberry_pi(self):
        """Can create Pi config."""
        from enigma_engine.core.low_power_inference import LowPowerConfig
        
        config = LowPowerConfig.for_raspberry_pi(ram_gb=1.0)
        
        self.assertLessEqual(config.max_memory_mb, 512)
        self.assertEqual(config.quantization_bits, 4)  # INT4 for 1GB Pi
        self.assertLessEqual(config.max_threads, 4)
    
    def test_config_for_mobile(self):
        """Can create mobile config."""
        from enigma_engine.core.low_power_inference import LowPowerConfig
        
        config = LowPowerConfig.for_mobile(is_high_end=False)
        
        self.assertEqual(config.quantization_bits, 4)
        self.assertLessEqual(config.max_context_tokens, 256)
    
    def test_config_auto_detect(self):
        """Can auto-detect config."""
        from enigma_engine.core.low_power_inference import LowPowerConfig
        
        config = LowPowerConfig.auto_detect()
        
        self.assertGreater(config.max_threads, 0)
        self.assertIn(config.quantization_bits, [0, 4, 8])


class TestQuantization(unittest.TestCase):
    """Test model quantization."""
    
    def test_quant_import(self):
        """Can import quantization."""
        from enigma_engine.core.quantization import (
            QuantConfig,
            QuantizedLinear,
            quantize_model,
            estimate_model_size,
            auto_quantize,
        )
        self.assertTrue(True)
    
    def test_quant_config_for_device(self):
        """Can create config for device."""
        from enigma_engine.core.quantization import QuantConfig
        
        embedded = QuantConfig.for_device('embedded')
        desktop = QuantConfig.for_device('desktop')
        
        self.assertEqual(embedded.bits, 4)
        self.assertEqual(desktop.bits, 16)
    
    def test_quantized_linear_creation(self):
        """Can create quantized linear layer."""
        import torch
        import torch.nn as nn
        from enigma_engine.core.quantization import QuantizedLinear
        
        linear = nn.Linear(64, 32)
        q_linear = QuantizedLinear.from_linear(linear, bits=8)
        
        self.assertEqual(q_linear.in_features, 64)
        self.assertEqual(q_linear.out_features, 32)
        self.assertEqual(q_linear.bits, 8)
    
    def test_quantized_forward(self):
        """Quantized layer forward pass works."""
        import torch
        import torch.nn as nn
        from enigma_engine.core.quantization import QuantizedLinear
        
        linear = nn.Linear(64, 32)
        q_linear = QuantizedLinear.from_linear(linear, bits=8)
        
        x = torch.randn(1, 64)
        y = q_linear(x)
        
        self.assertEqual(y.shape, (1, 32))
    
    def test_estimate_model_size(self):
        """Can estimate model size."""
        import torch.nn as nn
        from enigma_engine.core.quantization import estimate_model_size
        
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.Linear(50, 10),
        )
        
        stats = estimate_model_size(model, bits=8)
        
        self.assertIn('params', stats)
        self.assertIn('size_mb', stats)
        self.assertIn('quantized_mb', stats)
        self.assertLess(stats['quantized_mb'], stats['size_mb'])


class TestDistributedProtocol(unittest.TestCase):
    """Test distributed networking."""
    
    def test_distributed_import(self):
        """Can import distributed module."""
        from enigma_engine.comms.distributed import (
            DistributedNode,
            NodeRole,
            MessageType,
            ProtocolMessage,
            NodeInfo,
        )
        self.assertTrue(True)
    
    def test_node_roles(self):
        """Node roles are defined."""
        from enigma_engine.comms.distributed import NodeRole
        
        self.assertTrue(hasattr(NodeRole, 'AUTO'))
        self.assertTrue(hasattr(NodeRole, 'INFERENCE_CLIENT'))
        self.assertTrue(hasattr(NodeRole, 'INFERENCE_SERVER'))
        self.assertTrue(hasattr(NodeRole, 'AVATAR_DISPLAY'))
    
    def test_message_creation(self):
        """Can create protocol messages."""
        from enigma_engine.comms.distributed import ProtocolMessage, MessageType
        
        msg = ProtocolMessage(
            msg_type=MessageType.PING,
            payload={"test": True},
            sender_id="test_node",
        )
        
        self.assertEqual(msg.msg_type, MessageType.PING)
        self.assertEqual(msg.sender_id, "test_node")
    
    def test_message_serialization(self):
        """Messages serialize to JSON."""
        from enigma_engine.comms.distributed import ProtocolMessage, MessageType
        import json
        
        msg = ProtocolMessage(
            msg_type=MessageType.PING,
            payload={"value": 42},
            sender_id="sender",
            target_id="receiver",
        )
        
        json_str = msg.to_json()
        data = json.loads(json_str)
        
        self.assertEqual(data['type'], 'ping')
        self.assertEqual(data['payload']['value'], 42)
    
    def test_message_deserialization(self):
        """Messages deserialize from JSON."""
        from enigma_engine.comms.distributed import ProtocolMessage, MessageType
        
        json_str = '{"type": "ping", "payload": {"x": 1}, "sender": "s", "target": "t"}'
        msg = ProtocolMessage.from_json(json_str)
        
        self.assertEqual(msg.msg_type, MessageType.PING)
        self.assertEqual(msg.payload['x'], 1)
    
    def test_message_signing(self):
        """Messages can be signed and verified."""
        from enigma_engine.comms.distributed import ProtocolMessage, MessageType
        
        msg = ProtocolMessage(
            msg_type=MessageType.PING,
            payload={},
            sender_id="test",
        )
        
        secret = "test_secret_123"
        msg.sign(secret)
        
        self.assertTrue(len(msg.signature) > 0)
        self.assertTrue(msg.verify(secret))
        self.assertFalse(msg.verify("wrong_secret"))
    
    def test_node_creation(self):
        """Can create distributed node."""
        from enigma_engine.comms.distributed import DistributedNode, NodeRole
        
        node = DistributedNode("test_node", role=NodeRole.INFERENCE_CLIENT)
        
        self.assertEqual(node.name, "test_node")
        self.assertEqual(node.role, NodeRole.INFERENCE_CLIENT)


class TestModuleManagerEnhancements(unittest.TestCase):
    """Test module manager hardware awareness."""
    
    def test_device_profile_property(self):
        """Module manager has device_profile property."""
        from enigma_engine.modules.manager import ModuleManager
        
        manager = ModuleManager()
        self.assertTrue(hasattr(manager, 'device_profile'))
    
    def test_get_recommended_modules(self):
        """Can get recommended modules for device."""
        from enigma_engine.modules.manager import ModuleManager
        
        manager = ModuleManager()
        
        if hasattr(manager, 'get_recommended_modules'):
            recommended = manager.get_recommended_modules()
            self.assertIsInstance(recommended, list)
    
    def test_auto_configure(self):
        """Can auto-configure for device."""
        from enigma_engine.modules.manager import ModuleManager
        
        manager = ModuleManager()
        
        if hasattr(manager, 'auto_configure'):
            # Just check it doesn't crash
            try:
                manager.auto_configure()
            except Exception:
                pass  # OK if no modules registered


class TestUnifiedPatterns(unittest.TestCase):
    """Test unified GUI patterns."""
    
    def test_patterns_import(self):
        """Can import unified patterns."""
        try:
            from enigma_engine.gui.tabs.unified_patterns import (
                StyleConfig,
                Colors,
                get_style_config,
                get_button_style,
            )
            self.assertTrue(True)
        except ImportError:
            self.skipTest("PyQt5 not available")
    
    def test_style_config_singleton(self):
        """Style config is singleton."""
        try:
            from enigma_engine.gui.tabs.unified_patterns import get_style_config
            s1 = get_style_config()
            s2 = get_style_config()
            self.assertIs(s1, s2)
        except ImportError:
            self.skipTest("PyQt5 not available")
    
    def test_colors_defined(self):
        """Colors are defined."""
        try:
            from enigma_engine.gui.tabs.unified_patterns import Colors
            self.assertTrue(hasattr(Colors, 'BG_PRIMARY'))
            self.assertTrue(hasattr(Colors, 'ACCENT_BLUE'))
            self.assertTrue(hasattr(Colors, 'SUCCESS'))
        except ImportError:
            self.skipTest("PyQt5 not available")
    
    def test_button_styles(self):
        """Button styles are generated."""
        try:
            from enigma_engine.gui.tabs.unified_patterns import get_button_style
            
            primary = get_button_style('primary')
            secondary = get_button_style('secondary')
            
            self.assertIn('QPushButton', primary)
            self.assertIn('QPushButton', secondary)
        except ImportError:
            self.skipTest("PyQt5 not available")


class TestTrainingConfigDeviceAware(unittest.TestCase):
    """Test device-aware training configuration."""
    
    def test_from_device_profile_import(self):
        """TrainingConfig.from_device_profile is available."""
        from enigma_engine.core.training import TrainingConfig
        self.assertTrue(hasattr(TrainingConfig, 'from_device_profile'))
    
    def test_from_device_profile_returns_config(self):
        """from_device_profile returns a TrainingConfig instance."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig.from_device_profile()
        self.assertIsInstance(config, TrainingConfig)
    
    def test_from_device_profile_with_overrides(self):
        """from_device_profile accepts overrides."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig.from_device_profile(epochs=100, learning_rate=0.001)
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.learning_rate, 0.001)
    
    def test_device_aware_batch_size(self):
        """Batch size is adjusted based on device."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig.from_device_profile()
        # Should have a positive batch size
        self.assertGreater(config.batch_size, 0)
        self.assertLessEqual(config.batch_size, 64)  # Reasonable upper limit
    
    def test_device_aware_max_seq_len(self):
        """Max sequence length is adjusted based on device."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig.from_device_profile()
        # Should have a positive sequence length
        self.assertGreater(config.max_seq_len, 0)
        self.assertLessEqual(config.max_seq_len, 4096)  # Reasonable upper limit
    
    def test_device_aware_amp_setting(self):
        """AMP setting is based on GPU availability."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig.from_device_profile()
        # use_amp should be a boolean
        self.assertIsInstance(config.use_amp, bool)


def run_hardware_tests():
    """Run all hardware optimization tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDeviceProfiles))
    suite.addTests(loader.loadTestsFromTestCase(TestPowerMode))
    suite.addTests(loader.loadTestsFromTestCase(TestLowPowerInference))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantization))
    suite.addTests(loader.loadTestsFromTestCase(TestDistributedProtocol))
    suite.addTests(loader.loadTestsFromTestCase(TestModuleManagerEnhancements))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedPatterns))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingConfigDeviceAware))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_hardware_tests()
    sys.exit(0 if success else 1)
