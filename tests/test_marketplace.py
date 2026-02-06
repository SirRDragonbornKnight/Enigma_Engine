"""
Tests for forge_ai.marketplace module.

Tests marketplace functionality including:
- Marketplace browsing
- Package installation
- Repository management
- Package publishing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json


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
    
    def test_browse_packages(self):
        """Test browsing marketplace packages."""
        from forge_ai.marketplace.marketplace import Marketplace
        
        marketplace = Marketplace()
        
        # Should have browse method
        assert hasattr(marketplace, 'browse')
        
        # Browse should return list
        packages = marketplace.browse()
        assert isinstance(packages, list)
    
    def test_search_packages(self):
        """Test searching for packages."""
        from forge_ai.marketplace.marketplace import Marketplace
        
        marketplace = Marketplace()
        
        # Should have search method
        assert hasattr(marketplace, 'search')
    
    def test_get_package_details(self):
        """Test getting package details."""
        from forge_ai.marketplace.marketplace import Marketplace
        
        marketplace = Marketplace()
        
        # Should have get_details method
        assert hasattr(marketplace, 'get_details')
    
    def test_category_filtering(self):
        """Test filtering by category."""
        from forge_ai.marketplace.marketplace import Marketplace
        
        marketplace = Marketplace()
        
        # Should support category filtering
        assert hasattr(marketplace, 'filter_by_category')


class TestMarketplaceInstaller:
    """Test marketplace package installer."""
    
    def test_installer_class_exists(self):
        """Test MarketplaceInstaller class exists."""
        from forge_ai.marketplace.installer import MarketplaceInstaller
        
        assert MarketplaceInstaller is not None
    
    def test_installer_creation(self):
        """Test creating installer instance."""
        from forge_ai.marketplace.installer import MarketplaceInstaller
        
        installer = MarketplaceInstaller()
        assert installer is not None
    
    def test_install_package(self):
        """Test installing a package."""
        from forge_ai.marketplace.installer import MarketplaceInstaller
        
        installer = MarketplaceInstaller()
        
        # Should have install method
        assert hasattr(installer, 'install')
    
    def test_uninstall_package(self):
        """Test uninstalling a package."""
        from forge_ai.marketplace.installer import MarketplaceInstaller
        
        installer = MarketplaceInstaller()
        
        # Should have uninstall method
        assert hasattr(installer, 'uninstall')
    
    def test_update_package(self):
        """Test updating a package."""
        from forge_ai.marketplace.installer import MarketplaceInstaller
        
        installer = MarketplaceInstaller()
        
        # Should have update method
        assert hasattr(installer, 'update')
    
    def test_list_installed(self):
        """Test listing installed packages."""
        from forge_ai.marketplace.installer import MarketplaceInstaller
        
        installer = MarketplaceInstaller()
        
        installed = installer.list_installed()
        assert isinstance(installed, list)


class TestRepository:
    """Test repository management."""
    
    def test_repository_class_exists(self):
        """Test Repository class exists."""
        from forge_ai.marketplace.repository import Repository
        
        assert Repository is not None
    
    def test_repository_creation(self):
        """Test creating repository instance."""
        from forge_ai.marketplace.repository import Repository
        
        repo = Repository(url="https://example.com/repo")
        assert repo is not None
        assert repo.url == "https://example.com/repo"
    
    def test_fetch_index(self):
        """Test fetching repository index."""
        from forge_ai.marketplace.repository import Repository
        
        repo = Repository(url="https://example.com/repo")
        
        # Should have fetch method
        assert hasattr(repo, 'fetch_index')
    
    def test_repository_validation(self):
        """Test repository URL validation."""
        from forge_ai.marketplace.repository import Repository, validate_url
        
        # Valid URL
        assert validate_url("https://example.com/repo") is True
        
        # Invalid URL
        assert validate_url("not-a-url") is False


class TestRepositoryManager:
    """Test repository manager."""
    
    def test_manager_creation(self):
        """Test creating repository manager."""
        from forge_ai.marketplace.repository import RepositoryManager
        
        manager = RepositoryManager()
        assert manager is not None
    
    def test_add_repository(self):
        """Test adding a repository."""
        from forge_ai.marketplace.repository import RepositoryManager
        
        manager = RepositoryManager()
        
        # Should have add method
        assert hasattr(manager, 'add')
    
    def test_remove_repository(self):
        """Test removing a repository."""
        from forge_ai.marketplace.repository import RepositoryManager
        
        manager = RepositoryManager()
        
        # Should have remove method
        assert hasattr(manager, 'remove')
    
    def test_list_repositories(self):
        """Test listing repositories."""
        from forge_ai.marketplace.repository import RepositoryManager
        
        manager = RepositoryManager()
        
        repos = manager.list()
        assert isinstance(repos, list)


class TestPackage:
    """Test package data structure."""
    
    def test_package_creation(self):
        """Test creating a package object."""
        from forge_ai.marketplace.marketplace import Package
        
        package = Package(
            name="test-package",
            version="1.0.0",
            description="A test package",
            author="Tester"
        )
        
        assert package.name == "test-package"
        assert package.version == "1.0.0"
    
    def test_package_serialization(self):
        """Test package serialization."""
        from forge_ai.marketplace.marketplace import Package
        
        package = Package(
            name="test-package",
            version="1.0.0",
            description="Test",
            author="Author"
        )
        
        data = package.to_dict()
        assert data['name'] == "test-package"
        assert data['version'] == "1.0.0"
    
    def test_package_deserialization(self):
        """Test package deserialization."""
        from forge_ai.marketplace.marketplace import Package
        
        data = {
            'name': 'from-dict',
            'version': '2.0.0',
            'description': 'From dict',
            'author': 'Author'
        }
        
        package = Package.from_dict(data)
        assert package.name == "from-dict"
        assert package.version == "2.0.0"


class TestVersioning:
    """Test version comparison."""
    
    def test_version_comparison(self):
        """Test comparing versions."""
        from forge_ai.marketplace.installer import compare_versions
        
        # Greater than
        assert compare_versions("2.0.0", "1.0.0") > 0
        
        # Less than
        assert compare_versions("1.0.0", "2.0.0") < 0
        
        # Equal
        assert compare_versions("1.0.0", "1.0.0") == 0
    
    def test_version_parsing(self):
        """Test parsing version strings."""
        from forge_ai.marketplace.installer import parse_version
        
        major, minor, patch = parse_version("1.2.3")
        assert major == 1
        assert minor == 2
        assert patch == 3


class TestPackageCache:
    """Test package caching."""
    
    def test_cache_creation(self):
        """Test creating package cache."""
        from forge_ai.marketplace.installer import PackageCache
        
        cache = PackageCache()
        assert cache is not None
    
    def test_cache_operations(self):
        """Test cache get/set operations."""
        from forge_ai.marketplace.installer import PackageCache
        
        cache = PackageCache()
        
        # Set value
        cache.set("test-key", {"data": "value"})
        
        # Get value
        value = cache.get("test-key")
        assert value == {"data": "value"}
    
    def test_cache_expiry(self):
        """Test cache expiry."""
        from forge_ai.marketplace.installer import PackageCache
        
        cache = PackageCache(ttl=1)  # 1 second TTL
        
        cache.set("expire-test", "value")
        
        import time
        time.sleep(2)
        
        # Should be expired
        value = cache.get("expire-test")
        assert value is None
