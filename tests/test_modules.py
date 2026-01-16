"""
Tests for the Module Manager system.

Verifies:
- Module registration and discovery
- Dependency resolution
- Conflict detection
- Hardware checking
- Loading/unloading modules
"""

import unittest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from forge_ai.modules.manager import (
    ModuleManager, Module, ModuleInfo, ModuleState, 
    ModuleCategory
)
from forge_ai.modules.registry import (
    MODULE_REGISTRY, register_all, 
    ModelModule, TokenizerModule, InferenceModule
)


class TestModuleManager(unittest.TestCase):
    """Test the core module manager functionality."""
    
    def setUp(self):
        """Create a fresh module manager for each test."""
        self.manager = ModuleManager()
    
    def test_hardware_detection(self):
        """Test that hardware is detected."""
        hw = self.manager.hardware_profile
        self.assertIn('cpu_cores', hw)
        self.assertIn('ram_mb', hw)
        self.assertIn('gpu_available', hw)
        self.assertIsInstance(hw['cpu_cores'], int)
        self.assertIsInstance(hw['ram_mb'], int)
        self.assertIsInstance(hw['gpu_available'], bool)
    
    def test_register_module(self):
        """Test registering a module class."""
        result = self.manager.register(ModelModule)
        self.assertTrue(result)
        self.assertIn('model', self.manager.module_classes)
    
    def test_register_invalid(self):
        """Test that non-Module classes can't be registered."""
        class NotAModule:
            pass
        
        result = self.manager.register(NotAModule)
        self.assertFalse(result)
    
    def test_can_load_unregistered(self):
        """Test can_load on unregistered module."""
        can_load, reason = self.manager.can_load('nonexistent')
        self.assertFalse(can_load)
        self.assertIn('not registered', reason.lower())
    
    def test_register_all_builtin(self):
        """Test registering all built-in modules."""
        register_all(self.manager)
        
        # Check that core modules are registered
        self.assertIn('model', self.manager.module_classes)
        self.assertIn('tokenizer', self.manager.module_classes)
        self.assertIn('inference', self.manager.module_classes)
        self.assertIn('memory', self.manager.module_classes)
        
        # Check generation modules
        self.assertIn('image_gen_local', self.manager.module_classes)
        self.assertIn('image_gen_api', self.manager.module_classes)
        
        # Should have all modules from registry
        self.assertEqual(len(self.manager.module_classes), len(MODULE_REGISTRY))
    
    def test_list_modules(self):
        """Test listing registered modules."""
        register_all(self.manager)
        modules = self.manager.list_modules()
        
        self.assertIsInstance(modules, list)
        self.assertGreater(len(modules), 0)
        
        # Check structure
        for module_info in modules:
            self.assertIsInstance(module_info, ModuleInfo)
            self.assertIsInstance(module_info.id, str)
            self.assertIsInstance(module_info.category, ModuleCategory)
    
    def test_list_by_category(self):
        """Test filtering modules by category."""
        register_all(self.manager)
        
        core_modules = self.manager.list_modules(ModuleCategory.CORE)
        self.assertGreater(len(core_modules), 0)
        
        for module_info in core_modules:
            self.assertEqual(module_info.category, ModuleCategory.CORE)
    
    def test_dependency_checking(self):
        """Test that dependencies are checked before loading."""
        register_all(self.manager)
        
        # Try to load inference without its dependencies
        can_load, reason = self.manager.can_load('inference')
        self.assertFalse(can_load)
        self.assertIn('model', reason.lower())
    
    def test_conflict_prevention_explicit(self):
        """Test that explicit conflicts are prevented."""
        register_all(self.manager)
        
        # Create a test module with conflicts
        class TestModule(Module):
            INFO = ModuleInfo(
                id="test_conflict",
                name="Test Conflict",
                description="Test",
                category=ModuleCategory.EXTENSION,
                conflicts=["model"],
            )
        
        self.manager.register(TestModule)
        
        # Load model first
        self.manager.load('model', {'size': 'nano', 'vocab_size': 1000})
        
        # Try to load conflicting module
        can_load, reason = self.manager.can_load('test_conflict')
        self.assertFalse(can_load)
        self.assertIn('conflict', reason.lower())
    
    def test_capability_conflict_prevention(self):
        """Test that capability conflicts are detected."""
        register_all(self.manager)
        
        # Disable local_only mode so we can test cloud modules
        self.manager.local_only = False
        
        # Both provide 'image_generation'
        # Load one (without GPU check - use mock)
        img_local_info = self.manager.module_classes['image_gen_local'].get_info()
        
        # Manually set GPU available for test
        self.manager.hardware_profile['gpu_available'] = True
        self.manager.hardware_profile['vram_mb'] = 8000
        
        # Load first image gen module
        success = self.manager.load('image_gen_local')
        
        if success:
            # Try to load second one - should fail due to capability conflict
            can_load, reason = self.manager.can_load('image_gen_api')
            self.assertFalse(can_load)
            self.assertIn('image_generation', reason.lower())


class TestModuleLifecycle(unittest.TestCase):
    """Test module loading, activation, and unloading."""
    
    def setUp(self):
        """Create manager with registered modules."""
        self.manager = ModuleManager()
        register_all(self.manager)
    
    def test_load_simple_module(self):
        """Test loading a module with no dependencies."""
        # Memory module has no requirements
        success = self.manager.load('memory')
        self.assertTrue(success)
        self.assertIn('memory', self.manager.modules)
        
        module = self.manager.get_module('memory')
        self.assertIsNotNone(module)
        self.assertEqual(module.state, ModuleState.LOADED)
    
    def test_unload_module(self):
        """Test unloading a module."""
        self.manager.load('memory')
        
        success = self.manager.unload('memory')
        self.assertTrue(success)
        self.assertNotIn('memory', self.manager.modules)
    
    def test_load_with_dependencies(self):
        """Test loading modules with dependencies."""
        # Load dependencies first
        self.manager.load('model', {'size': 'nano', 'vocab_size': 1000})
        self.manager.load('tokenizer')
        
        # Now load inference
        success = self.manager.load('inference')
        self.assertTrue(success)
        self.assertIn('inference', self.manager.modules)
    
    def test_cannot_unload_with_dependents(self):
        """Test that modules can't be unloaded if others depend on them."""
        # Load chain: model -> tokenizer -> inference
        self.manager.load('model', {'size': 'nano', 'vocab_size': 1000})
        self.manager.load('tokenizer')
        self.manager.load('inference')
        
        # Try to unload model while inference depends on it
        success = self.manager.unload('model')
        self.assertFalse(success)
        self.assertIn('model', self.manager.modules)
    
    def test_get_interface(self):
        """Test getting module interface."""
        self.manager.load('memory')
        
        interface = self.manager.get_interface('memory')
        self.assertIsNotNone(interface)
    
    def test_list_loaded(self):
        """Test listing loaded modules."""
        self.manager.load('memory')
        self.manager.load('model', {'size': 'nano', 'vocab_size': 1000})
        
        loaded = self.manager.list_loaded()
        self.assertIn('memory', loaded)
        self.assertIn('model', loaded)
        self.assertEqual(len(loaded), 2)


class TestModuleRegistry(unittest.TestCase):
    """Test the module registry."""
    
    def test_registry_populated(self):
        """Test that built-in modules are in registry."""
        self.assertGreater(len(MODULE_REGISTRY), 0)
        
        # Core modules
        self.assertIn('model', MODULE_REGISTRY)
        self.assertIn('tokenizer', MODULE_REGISTRY)
        self.assertIn('training', MODULE_REGISTRY)
        self.assertIn('inference', MODULE_REGISTRY)
    
    def test_module_info_structure(self):
        """Test that modules have proper info structure."""
        for module_id, module_class in MODULE_REGISTRY.items():
            info = module_class.get_info()
            
            # Check required fields
            self.assertEqual(info.id, module_id)
            self.assertIsInstance(info.name, str)
            self.assertIsInstance(info.description, str)
            self.assertIsInstance(info.category, ModuleCategory)
            self.assertIsInstance(info.version, str)
            self.assertIsInstance(info.requires, list)
            self.assertIsInstance(info.provides, list)
    
    def test_generation_modules_provide_capabilities(self):
        """Test that generation modules declare what they provide."""
        # Image generation modules
        img_local = MODULE_REGISTRY['image_gen_local'].get_info()
        img_api = MODULE_REGISTRY['image_gen_api'].get_info()
        
        self.assertIn('image_generation', img_local.provides)
        self.assertIn('image_generation', img_api.provides)
        
        # Code generation modules
        code_local = MODULE_REGISTRY['code_gen_local'].get_info()
        code_api = MODULE_REGISTRY['code_gen_api'].get_info()
        
        self.assertIn('code_generation', code_local.provides)
        self.assertIn('code_generation', code_api.provides)


class TestGenerationModules(unittest.TestCase):
    """Test generation module wrapper functionality."""
    
    def setUp(self):
        """Create manager with registered modules."""
        self.manager = ModuleManager()
        register_all(self.manager)
    
    def test_generation_module_interface(self):
        """Test that generation modules expose generate() method."""
        from forge_ai.modules.registry import ImageGenLocalModule
        
        # Check the class has the method
        self.assertTrue(hasattr(ImageGenLocalModule, 'generate'))
        
        # Instance would need proper loading, so just verify structure
        info = ImageGenLocalModule.get_info()
        self.assertEqual(info.category, ModuleCategory.GENERATION)


class TestModuleConfiguration(unittest.TestCase):
    """Test module configuration system."""
    
    def setUp(self):
        """Create manager with registered modules."""
        self.manager = ModuleManager()
        register_all(self.manager)
    
    def test_module_has_config_schema(self):
        """Test that modules define configuration schemas."""
        model_info = ModelModule.get_info()
        self.assertIsInstance(model_info.config_schema, dict)
        self.assertIn('size', model_info.config_schema)
    
    def test_load_with_config(self):
        """Test loading module with configuration."""
        config = {
            'size': 'tiny',
            'vocab_size': 5000,
        }
        
        success = self.manager.load('model', config)
        self.assertTrue(success)
        
        module = self.manager.get_module('model')
        self.assertEqual(module.config['size'], 'tiny')
        self.assertEqual(module.config['vocab_size'], 5000)


if __name__ == '__main__':
    unittest.main()
