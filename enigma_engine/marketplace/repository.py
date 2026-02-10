"""
================================================================================
PLUGIN REPOSITORY - MANAGE PLUGIN SOURCES
================================================================================

Handles plugin repositories (local and remote) for the marketplace.

FILE: enigma_engine/marketplace/repository.py
TYPE: Repository Management
MAIN CLASSES: PluginRepository, LocalRepository, RemoteRepository
"""

import json
import logging
import shutil
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import CONFIG
from .marketplace import PluginCategory, PluginInfo, PluginVersion

logger = logging.getLogger(__name__)


class PluginRepository(ABC):
    """Base class for plugin repositories."""
    
    @abstractmethod
    def get_plugins(self) -> list[PluginInfo]:
        """Get all plugins from this repository."""
    
    @abstractmethod
    def get_plugin(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get a specific plugin by ID."""
    
    @abstractmethod
    def download(self, plugin_id: str, version: str, dest: Path) -> bool:
        """Download a plugin to destination."""


class LocalRepository(PluginRepository):
    """
    Local file-based repository.
    
    Used for:
    - Development/testing plugins
    - Offline plugin storage
    - Local plugin distribution
    """
    
    def __init__(self, path: Path):
        """
        Initialize local repository.
        
        Args:
            path: Path to repository directory
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.path / "index.json"
        self._plugins: dict[str, PluginInfo] = {}
        self._load_index()
    
    def _load_index(self):
        """Load plugin index from file."""
        if self.index_file.exists():
            try:
                with open(self.index_file) as f:
                    data = json.load(f)
                for plugin_data in data.get('plugins', []):
                    plugin = PluginInfo.from_dict(plugin_data)
                    self._plugins[plugin.id] = plugin
            except Exception as e:
                logger.error(f"Error loading local index: {e}")
    
    def _save_index(self):
        """Save plugin index to file."""
        data = {
            'plugins': [p.to_dict() for p in self._plugins.values()],
            'updated_at': datetime.now().isoformat(),
        }
        with open(self.index_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_plugins(self) -> list[PluginInfo]:
        """Get all plugins."""
        return list(self._plugins.values())
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get plugin by ID."""
        return self._plugins.get(plugin_id)
    
    def download(self, plugin_id: str, version: str, dest: Path) -> bool:
        """Copy plugin to destination."""
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return False
        
        # Find version
        version_info = None
        for v in plugin.versions:
            if v.version == version:
                version_info = v
                break
        
        if not version_info:
            return False
        
        # Copy from local path
        src = self.path / "packages" / f"{plugin_id}-{version}.zip"
        if not src.exists():
            return False
        
        shutil.copy(src, dest)
        return True
    
    def add_plugin(self, plugin_path: Path, metadata: dict) -> PluginInfo:
        """
        Add a plugin to the local repository.
        
        Args:
            plugin_path: Path to plugin package (.zip)
            metadata: Plugin metadata
        
        Returns:
            PluginInfo for the added plugin
        """
        # Create PluginInfo
        plugin_id = metadata['id']
        version = metadata['version']
        
        plugin = PluginInfo(
            id=plugin_id,
            name=metadata['name'],
            description=metadata['description'],
            author=metadata.get('author', 'Unknown'),
            category=PluginCategory(metadata.get('category', 'other')),
            current_version=version,
            versions=[
                PluginVersion(
                    version=version,
                    release_date=datetime.now(),
                    changelog=metadata.get('changelog', ''),
                )
            ],
        )
        
        # Copy package
        packages_dir = self.path / "packages"
        packages_dir.mkdir(exist_ok=True)
        
        dest = packages_dir / f"{plugin_id}-{version}.zip"
        shutil.copy(plugin_path, dest)
        
        # Update index
        self._plugins[plugin_id] = plugin
        self._save_index()
        
        logger.info(f"Added plugin to local repository: {plugin_id} v{version}")
        return plugin
    
    def remove_plugin(self, plugin_id: str) -> bool:
        """Remove a plugin from the repository."""
        if plugin_id not in self._plugins:
            return False
        
        plugin = self._plugins[plugin_id]
        
        # Remove package files
        packages_dir = self.path / "packages"
        for version in plugin.versions:
            package = packages_dir / f"{plugin_id}-{version.version}.zip"
            if package.exists():
                package.unlink()
        
        # Update index
        del self._plugins[plugin_id]
        self._save_index()
        
        logger.info(f"Removed plugin from local repository: {plugin_id}")
        return True


class RemoteRepository(PluginRepository):
    """
    Remote HTTP-based repository.
    
    Fetches plugins from a remote server.
    """
    
    def __init__(self, base_url: str, cache_dir: Path = None):
        """
        Initialize remote repository.
        
        Args:
            base_url: Base URL of the repository
            cache_dir: Local cache directory
        """
        self.base_url = base_url.rstrip('/')
        self.cache_dir = cache_dir or Path(CONFIG.get("cache_dir", ".cache")) / "repos"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._plugins: dict[str, PluginInfo] = {}
        self._last_fetch: Optional[datetime] = None
    
    def _fetch_index(self, force: bool = False) -> bool:
        """Fetch the plugin index from remote."""
        # Check cache freshness (1 hour)
        if not force and self._last_fetch:
            age = (datetime.now() - self._last_fetch).total_seconds()
            if age < 3600:
                return True
        
        try:
            index_url = f"{self.base_url}/index.json"
            
            with urllib.request.urlopen(index_url, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            self._plugins = {}
            for plugin_data in data.get('plugins', []):
                plugin = PluginInfo.from_dict(plugin_data)
                self._plugins[plugin.id] = plugin
            
            self._last_fetch = datetime.now()
            
            # Cache locally
            cache_file = self.cache_dir / "remote_index.json"
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Error fetching remote index: {e}")
            
            # Try to load from cache
            cache_file = self.cache_dir / "remote_index.json"
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        data = json.load(f)
                    self._plugins = {}
                    for plugin_data in data.get('plugins', []):
                        plugin = PluginInfo.from_dict(plugin_data)
                        self._plugins[plugin.id] = plugin
                    return True
                except Exception:
                    pass
            
            return False
    
    def get_plugins(self) -> list[PluginInfo]:
        """Get all plugins."""
        self._fetch_index()
        return list(self._plugins.values())
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get plugin by ID."""
        self._fetch_index()
        return self._plugins.get(plugin_id)
    
    def download(self, plugin_id: str, version: str, dest: Path) -> bool:
        """Download plugin package."""
        plugin = self.get_plugin(plugin_id)
        if not plugin:
            return False
        
        # Find version
        version_info = None
        for v in plugin.versions:
            if v.version == version:
                version_info = v
                break
        
        if not version_info or not version_info.download_url:
            return False
        
        try:
            urllib.request.urlretrieve(version_info.download_url, dest)
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False


class RepositoryManager:
    """Manages multiple plugin repositories."""
    
    def __init__(self):
        """Initialize repository manager."""
        self.repositories: list[PluginRepository] = []
        self._local_repo: Optional[LocalRepository] = None
    
    def add_repository(self, repo: PluginRepository):
        """Add a repository."""
        self.repositories.append(repo)
    
    def add_local(self, path: Path) -> LocalRepository:
        """Add a local repository."""
        repo = LocalRepository(path)
        self.repositories.append(repo)
        self._local_repo = repo
        return repo
    
    def add_remote(self, url: str) -> RemoteRepository:
        """Add a remote repository."""
        repo = RemoteRepository(url)
        self.repositories.append(repo)
        return repo
    
    def get_all_plugins(self) -> list[PluginInfo]:
        """Get plugins from all repositories."""
        plugins = {}
        for repo in self.repositories:
            for plugin in repo.get_plugins():
                # Keep first occurrence (priority by order)
                if plugin.id not in plugins:
                    plugins[plugin.id] = plugin
        return list(plugins.values())
    
    def find_plugin(self, plugin_id: str) -> Optional[tuple]:
        """Find plugin and its repository."""
        for repo in self.repositories:
            plugin = repo.get_plugin(plugin_id)
            if plugin:
                return plugin, repo
        return None
    
    @property
    def local(self) -> Optional[LocalRepository]:
        """Get the local repository."""
        return self._local_repo
