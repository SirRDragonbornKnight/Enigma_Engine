"""
Plugin Marketplace

Browse, search, and install community plugins.
Handles plugin registry, versioning, and dependencies.

FILE: forge_ai/plugins/marketplace.py
TYPE: Plugin System
MAIN CLASSES: PluginMarketplace, PluginRegistry, PluginInstaller
"""

import hashlib
import json
import logging
import re
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PluginCategory(Enum):
    """Plugin categories."""
    TOOLS = "tools"
    MODELS = "models"
    UI = "ui"
    VOICE = "voice"
    AVATAR = "avatar"
    DATA = "data"
    INTEGRATION = "integration"
    OTHER = "other"


@dataclass
class PluginVersion:
    """Plugin version information."""
    version: str
    forge_version: str  # Minimum ForgeAI version
    release_date: str
    download_url: str
    checksum: str  # SHA256
    changelog: str = ""


@dataclass
class PluginInfo:
    """Plugin metadata."""
    id: str
    name: str
    description: str
    author: str
    category: PluginCategory
    homepage: str = ""
    repository: str = ""
    license: str = "MIT"
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # Other plugin IDs
    versions: list[PluginVersion] = field(default_factory=list)
    
    # Stats
    downloads: int = 0
    rating: float = 0.0
    rating_count: int = 0
    
    @property
    def latest_version(self) -> Optional[PluginVersion]:
        """Get latest version."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.version)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "category": self.category.value,
            "homepage": self.homepage,
            "repository": self.repository,
            "license": self.license,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "versions": [
                {
                    "version": v.version,
                    "forge_version": v.forge_version,
                    "release_date": v.release_date,
                    "download_url": v.download_url,
                    "checksum": v.checksum,
                    "changelog": v.changelog
                }
                for v in self.versions
            ],
            "downloads": self.downloads,
            "rating": self.rating,
            "rating_count": self.rating_count
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PluginInfo':
        """Create from dictionary."""
        versions = [
            PluginVersion(**v) for v in data.get("versions", [])
        ]
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            author=data["author"],
            category=PluginCategory(data.get("category", "other")),
            homepage=data.get("homepage", ""),
            repository=data.get("repository", ""),
            license=data.get("license", "MIT"),
            tags=data.get("tags", []),
            dependencies=data.get("dependencies", []),
            versions=versions,
            downloads=data.get("downloads", 0),
            rating=data.get("rating", 0.0),
            rating_count=data.get("rating_count", 0)
        )


class PluginRegistry:
    """
    Local plugin registry cache.
    
    Stores information about available and installed plugins.
    """
    
    def __init__(self, cache_dir: Path):
        """
        Initialize registry.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache_file = self.cache_dir / "registry.json"
        self._plugins: dict[str, PluginInfo] = {}
        
        self._load_cache()
    
    def _load_cache(self):
        """Load cached registry."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file) as f:
                    data = json.load(f)
                self._plugins = {
                    k: PluginInfo.from_dict(v) 
                    for k, v in data.get("plugins", {}).items()
                }
                logger.info(f"Loaded {len(self._plugins)} plugins from cache")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk."""
        data = {
            "plugins": {k: v.to_dict() for k, v in self._plugins.items()}
        }
        with open(self._cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def update_from_remote(self, registry_url: str):
        """
        Update registry from remote server.
        
        Args:
            registry_url: URL of plugin registry JSON
        """
        try:
            with urllib.request.urlopen(registry_url, timeout=30) as response:
                data = json.loads(response.read().decode())
            
            for plugin_data in data.get("plugins", []):
                plugin = PluginInfo.from_dict(plugin_data)
                self._plugins[plugin.id] = plugin
            
            self._save_cache()
            logger.info(f"Updated registry with {len(self._plugins)} plugins")
            
        except Exception as e:
            logger.error(f"Failed to update registry: {e}")
            raise
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get plugin by ID."""
        return self._plugins.get(plugin_id)
    
    def search(self,
               query: str = "",
               category: PluginCategory = None,
               tags: list[str] = None) -> list[PluginInfo]:
        """
        Search for plugins.
        
        Args:
            query: Search query (matches name, description)
            category: Filter by category
            tags: Filter by tags
            
        Returns:
            Matching plugins
        """
        results = []
        query_lower = query.lower()
        
        for plugin in self._plugins.values():
            # Category filter
            if category and plugin.category != category:
                continue
            
            # Tags filter
            if tags:
                if not any(t in plugin.tags for t in tags):
                    continue
            
            # Query filter
            if query:
                if query_lower not in plugin.name.lower() and \
                   query_lower not in plugin.description.lower():
                    continue
            
            results.append(plugin)
        
        # Sort by downloads
        results.sort(key=lambda p: p.downloads, reverse=True)
        return results
    
    def get_popular(self, limit: int = 10) -> list[PluginInfo]:
        """Get popular plugins."""
        plugins = list(self._plugins.values())
        plugins.sort(key=lambda p: p.downloads, reverse=True)
        return plugins[:limit]
    
    def get_by_category(self, category: PluginCategory) -> list[PluginInfo]:
        """Get plugins by category."""
        return [p for p in self._plugins.values() if p.category == category]


class PluginInstaller:
    """
    Install, update, and remove plugins.
    
    Handles downloading, verification, and dependency resolution.
    """
    
    def __init__(self, plugins_dir: Path, registry: PluginRegistry):
        """
        Initialize installer.
        
        Args:
            plugins_dir: Directory to install plugins
            registry: Plugin registry
        """
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry = registry
        self._installed: dict[str, str] = {}  # plugin_id -> version
        
        self._load_installed()
    
    def _load_installed(self):
        """Load installed plugins list."""
        manifest = self.plugins_dir / "installed.json"
        if manifest.exists():
            with open(manifest) as f:
                self._installed = json.load(f)
    
    def _save_installed(self):
        """Save installed plugins list."""
        manifest = self.plugins_dir / "installed.json"
        with open(manifest, 'w') as f:
            json.dump(self._installed, f, indent=2)
    
    def is_installed(self, plugin_id: str) -> bool:
        """Check if plugin is installed."""
        return plugin_id in self._installed
    
    def get_installed_version(self, plugin_id: str) -> Optional[str]:
        """Get installed version of plugin."""
        return self._installed.get(plugin_id)
    
    def get_installed_plugins(self) -> dict[str, str]:
        """Get all installed plugins."""
        return self._installed.copy()
    
    def install(self,
                plugin_id: str,
                version: str = None,
                progress_callback: Callable[[float], None] = None) -> bool:
        """
        Install a plugin.
        
        Args:
            plugin_id: Plugin ID to install
            version: Specific version (None = latest)
            progress_callback: Progress callback (0.0 - 1.0)
            
        Returns:
            True if successful
        """
        # Get plugin info
        plugin = self.registry.get_plugin(plugin_id)
        if not plugin:
            logger.error(f"Plugin not found: {plugin_id}")
            return False
        
        # Get version
        if version:
            plugin_version = next(
                (v for v in plugin.versions if v.version == version), None
            )
        else:
            plugin_version = plugin.latest_version
        
        if not plugin_version:
            logger.error(f"Version not found: {version}")
            return False
        
        # Check dependencies
        for dep_id in plugin.dependencies:
            if not self.is_installed(dep_id):
                logger.info(f"Installing dependency: {dep_id}")
                if not self.install(dep_id):
                    logger.error(f"Failed to install dependency: {dep_id}")
                    return False
        
        try:
            # Download
            if progress_callback:
                progress_callback(0.1)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                zip_path = tmp_path / f"{plugin_id}.zip"
                
                self._download_file(plugin_version.download_url, zip_path)
                
                if progress_callback:
                    progress_callback(0.5)
                
                # Verify checksum
                if not self._verify_checksum(zip_path, plugin_version.checksum):
                    logger.error("Checksum verification failed")
                    return False
                
                if progress_callback:
                    progress_callback(0.7)
                
                # Extract
                plugin_dir = self.plugins_dir / plugin_id
                if plugin_dir.exists():
                    shutil.rmtree(plugin_dir)
                
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(plugin_dir)
                
                if progress_callback:
                    progress_callback(0.9)
                
                # Update installed list
                self._installed[plugin_id] = plugin_version.version
                self._save_installed()
                
                if progress_callback:
                    progress_callback(1.0)
                
                logger.info(f"Installed {plugin_id} v{plugin_version.version}")
                return True
                
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return False
    
    def _download_file(self, url: str, path: Path):
        """Download file from URL."""
        urllib.request.urlretrieve(url, path)
    
    def _verify_checksum(self, path: Path, expected: str) -> bool:
        """Verify file SHA256 checksum."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest() == expected
    
    def uninstall(self, plugin_id: str) -> bool:
        """
        Uninstall a plugin.
        
        Args:
            plugin_id: Plugin ID to uninstall
            
        Returns:
            True if successful
        """
        if not self.is_installed(plugin_id):
            logger.warning(f"Plugin not installed: {plugin_id}")
            return False
        
        try:
            plugin_dir = self.plugins_dir / plugin_id
            if plugin_dir.exists():
                shutil.rmtree(plugin_dir)
            
            del self._installed[plugin_id]
            self._save_installed()
            
            logger.info(f"Uninstalled {plugin_id}")
            return True
            
        except Exception as e:
            logger.error(f"Uninstall failed: {e}")
            return False
    
    def update(self, plugin_id: str) -> bool:
        """Update plugin to latest version."""
        plugin = self.registry.get_plugin(plugin_id)
        if not plugin:
            return False
        
        latest = plugin.latest_version
        if not latest:
            return False
        
        current = self._installed.get(plugin_id)
        if current == latest.version:
            logger.info(f"{plugin_id} is already up to date")
            return True
        
        return self.install(plugin_id, latest.version)
    
    def check_updates(self) -> list[PluginInfo]:
        """Check for available updates."""
        updates = []
        
        for plugin_id, current_version in self._installed.items():
            plugin = self.registry.get_plugin(plugin_id)
            if plugin and plugin.latest_version:
                if plugin.latest_version.version != current_version:
                    updates.append(plugin)
        
        return updates


class PluginMarketplace:
    """
    Main interface for the plugin marketplace.
    
    Combines registry and installer functionality.
    """
    
    DEFAULT_REGISTRY_URL = "https://raw.githubusercontent.com/forge-ai/plugins/main/registry.json"
    
    def __init__(self, data_dir: Path):
        """
        Initialize marketplace.
        
        Args:
            data_dir: Base directory for plugin data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry = PluginRegistry(self.data_dir / "cache")
        self.installer = PluginInstaller(
            self.data_dir / "plugins",
            self.registry
        )
    
    def refresh(self, registry_url: str = None):
        """Refresh plugin registry from remote."""
        url = registry_url or self.DEFAULT_REGISTRY_URL
        self.registry.update_from_remote(url)
    
    def search(self,
               query: str = "",
               category: str = None,
               tags: list[str] = None) -> list[PluginInfo]:
        """Search for plugins."""
        cat = PluginCategory(category) if category else None
        return self.registry.search(query, cat, tags)
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get plugin details."""
        return self.registry.get_plugin(plugin_id)
    
    def install(self, plugin_id: str, version: str = None) -> bool:
        """Install a plugin."""
        return self.installer.install(plugin_id, version)
    
    def uninstall(self, plugin_id: str) -> bool:
        """Uninstall a plugin."""
        return self.installer.uninstall(plugin_id)
    
    def update(self, plugin_id: str) -> bool:
        """Update a plugin."""
        return self.installer.update(plugin_id)
    
    def update_all(self) -> dict[str, bool]:
        """Update all plugins."""
        results = {}
        for plugin_id in self.installer.get_installed_plugins():
            results[plugin_id] = self.update(plugin_id)
        return results
    
    def get_installed(self) -> list[Tuple[PluginInfo, str]]:
        """Get installed plugins with versions."""
        result = []
        for plugin_id, version in self.installer.get_installed_plugins().items():
            plugin = self.registry.get_plugin(plugin_id)
            if plugin:
                result.append((plugin, version))
        return result
    
    def check_updates(self) -> list[PluginInfo]:
        """Check for available updates."""
        return self.installer.check_updates()
    
    def get_popular(self, limit: int = 10) -> list[PluginInfo]:
        """Get popular plugins."""
        return self.registry.get_popular(limit)
    
    def get_categories(self) -> list[PluginCategory]:
        """Get available categories."""
        return list(PluginCategory)


__all__ = [
    'PluginMarketplace',
    'PluginRegistry',
    'PluginInstaller',
    'PluginInfo',
    'PluginVersion',
    'PluginCategory'
]
