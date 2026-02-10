"""
================================================================================
PLUGIN MARKETPLACE - CORE MARKETPLACE LOGIC
================================================================================

Browse, search, download, and share Enigma AI Engine modules with the community.

FILE: enigma_engine/marketplace/marketplace.py
TYPE: Marketplace Core
MAIN CLASSES: Marketplace, PluginInfo, PluginCategory

USAGE:
    market = Marketplace()
    plugins = market.search("voice", category=PluginCategory.VOICE)
    market.install(plugins[0].id)
"""

import hashlib
import json
import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import CONFIG

logger = logging.getLogger(__name__)


class PluginCategory(Enum):
    """Categories of plugins."""
    VOICE = "voice"
    AVATAR = "avatar"
    TOOLS = "tools"
    GENERATION = "generation"
    TRAINING = "training"
    MEMORY = "memory"
    INTERFACE = "interface"
    THEMES = "themes"
    INTEGRATIONS = "integrations"
    OTHER = "other"


@dataclass
class PluginVersion:
    """Version information for a plugin."""
    
    version: str
    release_date: datetime
    changelog: str = ""
    download_url: str = ""
    checksum: str = ""  # SHA256
    min_forge_version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'version': self.version,
            'release_date': self.release_date.isoformat(),
            'changelog': self.changelog,
            'download_url': self.download_url,
            'checksum': self.checksum,
            'min_forge_version': self.min_forge_version,
            'dependencies': self.dependencies,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PluginVersion':
        return cls(
            version=data['version'],
            release_date=datetime.fromisoformat(data['release_date']),
            changelog=data.get('changelog', ''),
            download_url=data.get('download_url', ''),
            checksum=data.get('checksum', ''),
            min_forge_version=data.get('min_forge_version', '0.1.0'),
            dependencies=data.get('dependencies', []),
        )


@dataclass
class PluginInfo:
    """Information about a marketplace plugin."""
    
    id: str
    name: str
    description: str
    author: str
    category: PluginCategory
    current_version: str
    versions: list[PluginVersion] = field(default_factory=list)
    
    # Metadata
    icon_url: str = ""
    screenshots: list[str] = field(default_factory=list)
    homepage: str = ""
    repository: str = ""
    license: str = "MIT"
    
    # Stats
    downloads: int = 0
    rating: float = 0.0
    reviews: int = 0
    
    # State
    installed: bool = False
    installed_version: str = ""
    update_available: bool = False
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'author': self.author,
            'category': self.category.value,
            'current_version': self.current_version,
            'versions': [v.to_dict() for v in self.versions],
            'icon_url': self.icon_url,
            'screenshots': self.screenshots,
            'homepage': self.homepage,
            'repository': self.repository,
            'license': self.license,
            'downloads': self.downloads,
            'rating': self.rating,
            'reviews': self.reviews,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PluginInfo':
        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            author=data['author'],
            category=PluginCategory(data.get('category', 'other')),
            current_version=data['current_version'],
            versions=[PluginVersion.from_dict(v) for v in data.get('versions', [])],
            icon_url=data.get('icon_url', ''),
            screenshots=data.get('screenshots', []),
            homepage=data.get('homepage', ''),
            repository=data.get('repository', ''),
            license=data.get('license', 'MIT'),
            downloads=data.get('downloads', 0),
            rating=data.get('rating', 0.0),
            reviews=data.get('reviews', 0),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now(),
            updated_at=datetime.fromisoformat(data['updated_at']) if 'updated_at' in data else datetime.now(),
        )


@dataclass
class InstallResult:
    """Result of plugin installation."""
    
    success: bool
    plugin_id: str
    version: str
    message: str
    installed_path: Optional[Path] = None
    dependencies_installed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class Marketplace:
    """
    Plugin marketplace for Enigma AI Engine.
    
    Features:
    - Browse and search plugins
    - Install/uninstall plugins
    - Update management
    - Publish your plugins
    - Rate and review
    """
    
    # Default repository URLs
    DEFAULT_REPOSITORIES = [
        "https://raw.githubusercontent.com/forge-ai-community/plugins/main/index.json",
    ]
    
    def __init__(self, plugins_dir: Path = None):
        """
        Initialize marketplace.
        
        Args:
            plugins_dir: Directory for installed plugins
        """
        self.plugins_dir = plugins_dir or Path(CONFIG.get("plugins_dir", "plugins"))
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = Path(CONFIG.get("cache_dir", ".cache")) / "marketplace"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Plugin registry
        self._plugins: dict[str, PluginInfo] = {}
        self._installed: dict[str, str] = {}  # id -> version
        
        # Repository URLs
        self._repositories = list(self.DEFAULT_REPOSITORIES)
        
        # Load installed plugins
        self._load_installed()
        
        logger.info(f"Marketplace initialized with {len(self._installed)} installed plugins")
    
    def _load_installed(self):
        """Load list of installed plugins."""
        installed_file = self.plugins_dir / "installed.json"
        if installed_file.exists():
            try:
                with open(installed_file) as f:
                    self._installed = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading installed plugins: {e}")
                self._installed = {}
    
    def _save_installed(self):
        """Save list of installed plugins."""
        installed_file = self.plugins_dir / "installed.json"
        with open(installed_file, 'w') as f:
            json.dump(self._installed, f, indent=2)
    
    def refresh(self, force: bool = False) -> int:
        """
        Refresh plugin list from repositories.
        
        Args:
            force: Force refresh even if cache is fresh
        
        Returns:
            Number of plugins found
        """
        cache_file = self.cache_dir / "plugins_cache.json"
        cache_age = 3600  # 1 hour
        
        # Check cache
        if not force and cache_file.exists():
            cache_mtime = cache_file.stat().st_mtime
            if (datetime.now().timestamp() - cache_mtime) < cache_age:
                try:
                    with open(cache_file) as f:
                        cached = json.load(f)
                    self._plugins = {
                        p['id']: PluginInfo.from_dict(p)
                        for p in cached.get('plugins', [])
                    }
                    self._update_installed_status()
                    return len(self._plugins)
                except Exception:
                    pass
        
        # Fetch from repositories
        all_plugins = []
        for repo_url in self._repositories:
            try:
                plugins = self._fetch_repository(repo_url)
                all_plugins.extend(plugins)
            except Exception as e:
                logger.warning(f"Error fetching {repo_url}: {e}")
        
        # Deduplicate by ID
        self._plugins = {}
        for plugin in all_plugins:
            if plugin.id not in self._plugins:
                self._plugins[plugin.id] = plugin
        
        # Update installed status
        self._update_installed_status()
        
        # Save cache
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'plugins': [p.to_dict() for p in self._plugins.values()],
                    'fetched_at': datetime.now().isoformat(),
                }, f)
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
        
        return len(self._plugins)
    
    def _fetch_repository(self, url: str) -> list[PluginInfo]:
        """Fetch plugins from a repository URL."""
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            plugins = []
            for plugin_data in data.get('plugins', []):
                try:
                    plugins.append(PluginInfo.from_dict(plugin_data))
                except Exception as e:
                    logger.warning(f"Error parsing plugin: {e}")
            
            return plugins
        except Exception as e:
            logger.error(f"Error fetching repository {url}: {e}")
            return []
    
    def _update_installed_status(self):
        """Update installed status for all plugins."""
        for plugin_id, plugin in self._plugins.items():
            if plugin_id in self._installed:
                plugin.installed = True
                plugin.installed_version = self._installed[plugin_id]
                # Check for updates
                if plugin.installed_version != plugin.current_version:
                    plugin.update_available = True
    
    def search(
        self,
        query: str = "",
        category: PluginCategory = None,
        installed_only: bool = False,
        sort_by: str = "downloads"
    ) -> list[PluginInfo]:
        """
        Search for plugins.
        
        Args:
            query: Search query (searches name and description)
            category: Filter by category
            installed_only: Only show installed plugins
            sort_by: Sort by "downloads", "rating", "name", "updated"
        
        Returns:
            List of matching plugins
        """
        results = list(self._plugins.values())
        
        # Filter by query
        if query:
            query_lower = query.lower()
            results = [
                p for p in results
                if query_lower in p.name.lower() or query_lower in p.description.lower()
            ]
        
        # Filter by category
        if category:
            results = [p for p in results if p.category == category]
        
        # Filter by installed
        if installed_only:
            results = [p for p in results if p.installed]
        
        # Sort
        sort_keys = {
            "downloads": lambda p: -p.downloads,
            "rating": lambda p: -p.rating,
            "name": lambda p: p.name.lower(),
            "updated": lambda p: -p.updated_at.timestamp(),
        }
        if sort_by in sort_keys:
            results.sort(key=sort_keys[sort_by])
        
        return results
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get plugin by ID."""
        return self._plugins.get(plugin_id)
    
    def install(
        self,
        plugin_id: str,
        version: str = None,
        progress_callback: Callable[[float, str], None] = None
    ) -> InstallResult:
        """
        Install a plugin.
        
        Args:
            plugin_id: Plugin ID to install
            version: Specific version (default: latest)
            progress_callback: Callback for progress updates (0-1, message)
        
        Returns:
            InstallResult with status
        """
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return InstallResult(
                success=False,
                plugin_id=plugin_id,
                version="",
                message=f"Plugin '{plugin_id}' not found"
            )
        
        # Get version info
        version = version or plugin.current_version
        version_info = None
        for v in plugin.versions:
            if v.version == version:
                version_info = v
                break
        
        if not version_info:
            return InstallResult(
                success=False,
                plugin_id=plugin_id,
                version=version,
                message=f"Version '{version}' not found"
            )
        
        if progress_callback:
            progress_callback(0.1, "Downloading plugin...")
        
        # Download plugin
        try:
            download_path = self._download_plugin(version_info)
        except Exception as e:
            return InstallResult(
                success=False,
                plugin_id=plugin_id,
                version=version,
                message=f"Download failed: {e}"
            )
        
        if progress_callback:
            progress_callback(0.5, "Verifying checksum...")
        
        # Verify checksum
        if version_info.checksum:
            if not self._verify_checksum(download_path, version_info.checksum):
                download_path.unlink()
                return InstallResult(
                    success=False,
                    plugin_id=plugin_id,
                    version=version,
                    message="Checksum verification failed"
                )
        
        if progress_callback:
            progress_callback(0.7, "Installing...")
        
        # Install dependencies
        deps_installed = []
        for dep in version_info.dependencies:
            if dep not in self._installed:
                dep_result = self.install(dep)
                if dep_result.success:
                    deps_installed.append(dep)
                else:
                    return InstallResult(
                        success=False,
                        plugin_id=plugin_id,
                        version=version,
                        message=f"Dependency '{dep}' failed: {dep_result.message}"
                    )
        
        # Extract to plugins directory
        install_path = self.plugins_dir / plugin_id
        try:
            if install_path.exists():
                shutil.rmtree(install_path)
            
            with zipfile.ZipFile(download_path, 'r') as zf:
                zf.extractall(install_path)
            
            download_path.unlink()
        except Exception as e:
            return InstallResult(
                success=False,
                plugin_id=plugin_id,
                version=version,
                message=f"Extraction failed: {e}"
            )
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        # Update installed list
        self._installed[plugin_id] = version
        self._save_installed()
        
        # Update plugin status
        plugin.installed = True
        plugin.installed_version = version
        plugin.update_available = False
        
        logger.info(f"Installed {plugin_id} v{version}")
        
        return InstallResult(
            success=True,
            plugin_id=plugin_id,
            version=version,
            message=f"Successfully installed {plugin.name} v{version}",
            installed_path=install_path,
            dependencies_installed=deps_installed
        )
    
    def _download_plugin(self, version_info: PluginVersion) -> Path:
        """Download plugin to temp file."""
        import urllib.request
        
        download_path = self.cache_dir / f"download_{datetime.now().timestamp()}.zip"
        
        urllib.request.urlretrieve(version_info.download_url, download_path)
        
        return download_path
    
    def _verify_checksum(self, file_path: Path, expected: str) -> bool:
        """Verify SHA256 checksum."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
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
        if plugin_id not in self._installed:
            logger.warning(f"Plugin '{plugin_id}' is not installed")
            return False
        
        install_path = self.plugins_dir / plugin_id
        if install_path.exists():
            shutil.rmtree(install_path)
        
        del self._installed[plugin_id]
        self._save_installed()
        
        # Update status
        if plugin_id in self._plugins:
            self._plugins[plugin_id].installed = False
            self._plugins[plugin_id].installed_version = ""
        
        logger.info(f"Uninstalled {plugin_id}")
        return True
    
    def get_updates(self) -> list[PluginInfo]:
        """Get list of plugins with available updates."""
        return [p for p in self._plugins.values() if p.update_available]
    
    def update_all(self) -> list[InstallResult]:
        """Update all plugins with available updates."""
        results = []
        for plugin in self.get_updates():
            result = self.install(plugin.id)
            results.append(result)
        return results
    
    def publish(
        self,
        plugin_path: Path,
        metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Prepare a plugin for publishing.
        
        Args:
            plugin_path: Path to plugin directory
            metadata: Plugin metadata (name, description, version, etc.)
        
        Returns:
            Publishing information
        """
        plugin_path = Path(plugin_path)
        if not plugin_path.exists():
            raise ValueError(f"Plugin path does not exist: {plugin_path}")
        
        # Validate required metadata
        required = ['id', 'name', 'description', 'version', 'author']
        for field in required:
            if field not in metadata:
                raise ValueError(f"Missing required field: {field}")
        
        # Create package
        package_name = f"{metadata['id']}-{metadata['version']}.zip"
        package_path = self.cache_dir / package_name
        
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in plugin_path.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(plugin_path)
                    zf.write(file, arcname)
        
        # Calculate checksum
        sha256 = hashlib.sha256()
        with open(package_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        
        return {
            'package_path': str(package_path),
            'checksum': sha256.hexdigest(),
            'size': package_path.stat().st_size,
            'metadata': metadata,
            'instructions': [
                "1. Upload the package to a public URL",
                "2. Create a pull request to add your plugin to the community index",
                "3. Include the checksum and download URL in your submission"
            ]
        }
    
    def add_repository(self, url: str):
        """Add a custom repository URL."""
        if url not in self._repositories:
            self._repositories.append(url)
            logger.info(f"Added repository: {url}")
    
    def remove_repository(self, url: str):
        """Remove a repository URL."""
        if url in self._repositories and url not in self.DEFAULT_REPOSITORIES:
            self._repositories.remove(url)
            logger.info(f"Removed repository: {url}")
    
    def get_installed_plugins(self) -> list[PluginInfo]:
        """Get all installed plugins."""
        return [p for p in self._plugins.values() if p.installed]
    
    def load_plugin(self, plugin_id: str) -> Optional[Any]:
        """
        Load an installed plugin.
        
        Args:
            plugin_id: Plugin ID to load
        
        Returns:
            Loaded plugin module or None
        """
        if plugin_id not in self._installed:
            logger.warning(f"Plugin '{plugin_id}' is not installed")
            return None
        
        plugin_path = self.plugins_dir / plugin_id
        if not plugin_path.exists():
            logger.error(f"Plugin directory not found: {plugin_path}")
            return None
        
        try:
            import importlib.util

            # Look for __init__.py or main.py
            init_file = plugin_path / "__init__.py"
            if not init_file.exists():
                init_file = plugin_path / "main.py"
            
            if not init_file.exists():
                logger.error(f"No entry point found for plugin: {plugin_id}")
                return None
            
            spec = importlib.util.spec_from_file_location(plugin_id, init_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            logger.info(f"Loaded plugin: {plugin_id}")
            return module
            
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_id}: {e}")
            return None


# Global marketplace instance
_marketplace: Optional[Marketplace] = None


def get_marketplace() -> Marketplace:
    """Get or create global marketplace instance."""
    global _marketplace
    if _marketplace is None:
        _marketplace = Marketplace()
    return _marketplace
