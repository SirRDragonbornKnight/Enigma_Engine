"""
Tests for forge_ai.marketplace module.

Tests marketplace functionality including:
- Package browsing and search
- Package installation
- Repository management
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import importlib


class TestMarketplace:
    """Test marketplace main class."""
    
    def test_marketplace_class_exists(self):
        """Test Marketplace class exists."""
        from forge_ai.marketplace.marketplace import Marketplace
        
        assert Marketplace is not None
    
    def test_marketplace_creation(self):
        """Test creating marketplace instance."""
        from forge_ai.marketplace.marketplace import Marketplace
        
        marketplace = Marketplace()
        assert marketplace is not None
    
    def test_search_packages(self):
        """Test searching for packages."""
        from forge_ai.marketplace.marketplace import Marketplace
        
        marketplace = Marketplace()
        
        # Should have search method
        assert hasattr(marketplace, 'search')
    
    def test_refresh_method(self):
        """Test refresh method exists."""
        from forge_ai.marketplace.marketplace import Marketplace
        
        marketplace = Marketplace()
        
        # Should have refresh method
        assert hasattr(marketplace, 'refresh')


class TestMarketplaceInstaller:
    """Test marketplace package installer."""
    
    def test_installer_class_exists(self):
        """Test PluginInstaller class exists."""
        from forge_ai.marketplace.installer import PluginInstaller
        
        assert PluginInstaller is not None
    
    def test_installer_creation(self):
        """Test creating installer instance."""
        from forge_ai.marketplace.installer import PluginInstaller
        
        installer = PluginInstaller()
        assert installer is not None
    
    def test_install_package(self):
        """Test installing a package."""
        from forge_ai.marketplace.installer import PluginInstaller
        
        installer = PluginInstaller()
        
        # Should have install method
        assert hasattr(installer, 'install')
    
    def test_uninstall_package(self):
        """Test uninstalling a package."""
        from forge_ai.marketplace.installer import PluginInstaller
        
        installer = PluginInstaller()
        
        # Should have uninstall method
        assert hasattr(installer, 'uninstall')


class TestRepository:
    """Test repository management."""
    
    def test_repository_module_exists(self):
        """Test Repository module exists."""
        try:
            repo_mod = importlib.import_module('forge_ai.marketplace.repository')
            assert repo_mod is not None
        except ImportError:
            pytest.skip("Repository module not available")
    
    def test_repository_manager_exists(self):
        """Test RepositoryManager class exists."""
        try:
            repo_mod = importlib.import_module('forge_ai.marketplace.repository')
            if hasattr(repo_mod, 'RepositoryManager'):
                assert repo_mod.RepositoryManager is not None
            else:
                pytest.skip("RepositoryManager not defined")
        except ImportError:
            pytest.skip("Repository module not available")


class TestPackageInfo:
    """Test package data structures."""
    
    def test_plugin_info_class(self):
        """Test PluginInfo class exists."""
        from forge_ai.marketplace.marketplace import PluginInfo
        
        assert PluginInfo is not None


class TestDependencyResolver:
    """Test dependency resolution."""
    
    def test_resolver_class_exists(self):
        """Test DependencyResolver class exists."""
        from forge_ai.marketplace.installer import DependencyResolver
        
        assert DependencyResolver is not None
