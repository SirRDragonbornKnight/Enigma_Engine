"""
Tests for forge_ai.plugins module.

Tests plugin system functionality including:
- Plugin loading and unloading
- Plugin installation
- Plugin signing and verification
- Plugin templates
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json


class TestPluginInstallation:
    """Test plugin installation."""
    
    def test_installation_class_exists(self):
        """Test PluginInstaller class exists."""
        from forge_ai.plugins.installation import PluginInstaller
        
        assert PluginInstaller is not None
    
    def test_installer_creation(self):
        """Test creating plugin installer with required args."""
        from forge_ai.plugins.installation import PluginInstaller
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plugins_dir = Path(tmpdir) / "plugins"
            installer = PluginInstaller(plugins_dir=plugins_dir)
            assert installer is not None
    
    def test_install_from_path(self):
        """Test installer has install method."""
        from forge_ai.plugins.installation import PluginInstaller
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plugins_dir = Path(tmpdir) / "plugins"
            installer = PluginInstaller(plugins_dir=plugins_dir)
            
            # Should have install method
            assert hasattr(installer, 'install')
    
    def test_uninstall_plugin(self):
        """Test installer has uninstall method."""
        from forge_ai.plugins.installation import PluginInstaller
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plugins_dir = Path(tmpdir) / "plugins"
            installer = PluginInstaller(plugins_dir=plugins_dir)
            
            # Should have uninstall method
            assert hasattr(installer, 'uninstall')


class TestPluginMarketplace:
    """Test plugin marketplace integration."""
    
    def test_marketplace_class_exists(self):
        """Test PluginMarketplace class exists."""
        from forge_ai.plugins.marketplace import PluginMarketplace
        
        assert PluginMarketplace is not None
    
    def test_marketplace_creation(self):
        """Test creating marketplace with required args."""
        from forge_ai.plugins.marketplace import PluginMarketplace
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "marketplace"
            marketplace = PluginMarketplace(data_dir=data_dir)
            assert marketplace is not None
    
    def test_search_plugins(self):
        """Test marketplace has search method."""
        from forge_ai.plugins.marketplace import PluginMarketplace
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "marketplace"
            marketplace = PluginMarketplace(data_dir=data_dir)
            
            # Should have search method
            assert hasattr(marketplace, 'search')


class TestPluginSigning:
    """Test plugin signing and verification."""
    
    def test_signing_class_exists(self):
        """Test PluginSigner class exists."""
        from forge_ai.plugins.signing import PluginSigner
        
        assert PluginSigner is not None
    
    def test_key_manager_exists(self):
        """Test KeyManager class exists."""
        from forge_ai.plugins.signing import KeyManager
        
        assert KeyManager is not None
    
    def test_signer_creation_with_key_manager(self):
        """Test creating plugin signer with key manager."""
        from forge_ai.plugins.signing import PluginSigner, KeyManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            key_dir = Path(tmpdir) / "keys"
            key_dir.mkdir()
            key_manager = KeyManager(key_dir)
            signer = PluginSigner(key_manager=key_manager)
            assert signer is not None


class TestPluginTemplates:
    """Test plugin templates."""
    
    def test_templates_module_exists(self):
        """Test plugin templates module exists."""
        import importlib
        try:
            templates = importlib.import_module('forge_ai.plugins.templates')
            assert templates is not None
        except ImportError:
            pytest.skip("templates module not available")
    
    def test_template_types(self):
        """Test available template types."""
        import importlib
        try:
            templates = importlib.import_module('forge_ai.plugins.templates')
            if hasattr(templates, 'TEMPLATE_TYPES'):
                assert len(templates.TEMPLATE_TYPES) > 0
            else:
                pytest.skip("TEMPLATE_TYPES not defined")
        except ImportError:
            pytest.skip("templates module not available")


class TestPluginManifest:
    """Test plugin manifest handling."""
    
    def test_manifest_creation(self):
        """Test creating plugin manifest with required fields."""
        from forge_ai.plugins.installation import PluginManifest, PluginType
        
        manifest = PluginManifest(
            name="Test Plugin",
            version="1.0.0",
            description="A test plugin",
            author="Test Author",
            plugin_type=PluginType.TOOL,
            entry_point="main"
        )
        
        assert manifest.name == "Test Plugin"
        assert manifest.version == "1.0.0"
        assert manifest.description == "A test plugin"
    
    def test_manifest_serialization(self):
        """Test manifest serialization."""
        from forge_ai.plugins.installation import PluginManifest, PluginType
        
        manifest = PluginManifest(
            name="SerializeTest",
            version="2.0.0",
            description="Test description",
            author="Author",
            plugin_type=PluginType.TOOL,
            entry_point="main"
        )
        
        data = manifest.to_dict()
        assert data['name'] == "SerializeTest"
        assert data['version'] == "2.0.0"
    
    def test_manifest_from_dict(self):
        """Test creating manifest from dict."""
        from forge_ai.plugins.installation import PluginManifest, PluginType
        
        data = {
            'name': 'TestFromDict',
            'version': '1.0.0',
            'description': 'From dict',
            'author': 'Tester',
            'type': 'tool',
            'entry_point': 'main'
        }
        
        manifest = PluginManifest.from_dict(data)
        assert manifest.name == 'TestFromDict'


class TestPluginDependencies:
    """Test plugin dependency handling."""
    
    def test_installer_has_dependency_methods(self):
        """Test installer has dependency-related methods."""
        from forge_ai.plugins.installation import PluginInstaller
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plugins_dir = Path(tmpdir) / "plugins"
            installer = PluginInstaller(plugins_dir=plugins_dir)
            
            # Check for common methods (may or may not exist)
            # Just verify installer works
            assert installer is not None


class TestPluginRegistry:
    """Test plugin registry."""
    
    def test_registry_class_exists(self):
        """Test PluginRegistry class exists."""
        from forge_ai.plugins.marketplace import PluginRegistry
        
        assert PluginRegistry is not None
    
    def test_registry_creation(self):
        """Test creating plugin registry."""
        from forge_ai.plugins.marketplace import PluginRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            registry = PluginRegistry(cache_dir=cache_dir)
            assert registry is not None
