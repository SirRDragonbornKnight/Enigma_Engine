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
        """Test creating plugin installer."""
        from forge_ai.plugins.installation import PluginInstaller
        
        installer = PluginInstaller()
        assert installer is not None
    
    def test_install_from_path(self):
        """Test installing plugin from path."""
        from forge_ai.plugins.installation import PluginInstaller
        
        installer = PluginInstaller()
        
        # Should have install method
        assert hasattr(installer, 'install')
    
    def test_uninstall_plugin(self):
        """Test uninstalling plugin."""
        from forge_ai.plugins.installation import PluginInstaller
        
        installer = PluginInstaller()
        
        # Should have uninstall method
        assert hasattr(installer, 'uninstall')
    
    def test_list_installed(self):
        """Test listing installed plugins."""
        from forge_ai.plugins.installation import PluginInstaller
        
        installer = PluginInstaller()
        
        plugins = installer.list_installed()
        assert isinstance(plugins, list)


class TestPluginMarketplace:
    """Test plugin marketplace integration."""
    
    def test_marketplace_class_exists(self):
        """Test PluginMarketplace class exists."""
        from forge_ai.plugins.marketplace import PluginMarketplace
        
        assert PluginMarketplace is not None
    
    def test_marketplace_creation(self):
        """Test creating marketplace client."""
        from forge_ai.plugins.marketplace import PluginMarketplace
        
        marketplace = PluginMarketplace()
        assert marketplace is not None
    
    def test_search_plugins(self):
        """Test searching for plugins."""
        from forge_ai.plugins.marketplace import PluginMarketplace
        
        marketplace = PluginMarketplace()
        
        # Should have search method
        assert hasattr(marketplace, 'search')
    
    def test_get_plugin_info(self):
        """Test getting plugin info."""
        from forge_ai.plugins.marketplace import PluginMarketplace
        
        marketplace = PluginMarketplace()
        
        # Should have get_info method
        assert hasattr(marketplace, 'get_info')


class TestPluginSigning:
    """Test plugin signing and verification."""
    
    def test_signing_class_exists(self):
        """Test PluginSigner class exists."""
        from forge_ai.plugins.signing import PluginSigner
        
        assert PluginSigner is not None
    
    def test_signer_creation(self):
        """Test creating plugin signer."""
        from forge_ai.plugins.signing import PluginSigner
        
        signer = PluginSigner()
        assert signer is not None
    
    def test_sign_plugin(self):
        """Test signing a plugin."""
        from forge_ai.plugins.signing import PluginSigner
        
        signer = PluginSigner()
        
        # Should have sign method
        assert hasattr(signer, 'sign')
    
    def test_verify_signature(self):
        """Test verifying plugin signature."""
        from forge_ai.plugins.signing import PluginSigner
        
        signer = PluginSigner()
        
        # Should have verify method
        assert hasattr(signer, 'verify')


class TestPluginTemplates:
    """Test plugin templates."""
    
    def test_templates_exist(self):
        """Test plugin templates exist."""
        from forge_ai.plugins.templates import PluginTemplate
        
        assert PluginTemplate is not None
    
    def test_create_from_template(self):
        """Test creating plugin from template."""
        from forge_ai.plugins.templates import PluginTemplate
        
        template = PluginTemplate()
        
        # Should have create method
        assert hasattr(template, 'create')
    
    def test_template_types(self):
        """Test available template types."""
        from forge_ai.plugins.templates import TEMPLATE_TYPES
        
        # Should have at least basic template
        assert len(TEMPLATE_TYPES) > 0


class TestPluginManifest:
    """Test plugin manifest handling."""
    
    def test_manifest_creation(self):
        """Test creating plugin manifest."""
        from forge_ai.plugins.installation import PluginManifest
        
        manifest = PluginManifest(
            name="Test Plugin",
            version="1.0.0",
            author="Test Author"
        )
        
        assert manifest.name == "Test Plugin"
        assert manifest.version == "1.0.0"
    
    def test_manifest_validation(self):
        """Test manifest validation."""
        from forge_ai.plugins.installation import PluginManifest, validate_manifest
        
        # Valid manifest
        valid = {
            'name': 'Test',
            'version': '1.0.0',
            'author': 'Tester'
        }
        assert validate_manifest(valid) is True
        
        # Invalid manifest (missing required field)
        invalid = {'name': 'Test'}
        assert validate_manifest(invalid) is False
    
    def test_manifest_serialization(self):
        """Test manifest serialization."""
        from forge_ai.plugins.installation import PluginManifest
        
        manifest = PluginManifest(
            name="SerializeTest",
            version="2.0.0",
            author="Author"
        )
        
        data = manifest.to_dict()
        assert data['name'] == "SerializeTest"
        assert data['version'] == "2.0.0"


class TestPluginDependencies:
    """Test plugin dependency handling."""
    
    def test_dependency_resolution(self):
        """Test resolving plugin dependencies."""
        from forge_ai.plugins.installation import PluginInstaller
        
        installer = PluginInstaller()
        
        # Should have dependency methods
        assert hasattr(installer, 'check_dependencies')
    
    def test_dependency_conflicts(self):
        """Test handling dependency conflicts."""
        from forge_ai.plugins.installation import PluginInstaller
        
        installer = PluginInstaller()
        
        # Should have conflict detection
        assert hasattr(installer, 'find_conflicts')


class TestPluginHooks:
    """Test plugin hook system."""
    
    def test_hook_registration(self):
        """Test registering plugin hooks."""
        from forge_ai.plugins.installation import PluginHookManager
        
        manager = PluginHookManager()
        
        # Register a hook
        callback = Mock()
        manager.register('on_load', callback)
        
        # Trigger hook
        manager.trigger('on_load')
        callback.assert_called_once()
    
    def test_hook_unregistration(self):
        """Test unregistering plugin hooks."""
        from forge_ai.plugins.installation import PluginHookManager
        
        manager = PluginHookManager()
        
        callback = Mock()
        manager.register('on_load', callback)
        manager.unregister('on_load', callback)
        
        # Should not be called after unregistration
        manager.trigger('on_load')
        callback.assert_not_called()
