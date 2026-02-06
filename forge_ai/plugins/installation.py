"""
Plugin Installation System

Installs, manages, and updates plugins from local files or registry.
Handles dependencies, versioning, and plugin lifecycle.

FILE: forge_ai/plugins/installation.py
TYPE: Plugin System
MAIN CLASSES: PluginInstaller, InstalledPlugin, PluginRegistry
"""

import hashlib
import importlib.util
import json
import logging
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class InstallStatus(Enum):
    """Plugin installation status."""
    NOT_INSTALLED = "not_installed"
    INSTALLED = "installed"
    UPDATE_AVAILABLE = "update_available"
    INSTALLING = "installing"
    ERROR = "error"


class PluginType(Enum):
    """Types of plugins."""
    TOOL = "tool"
    TAB = "tab"
    THEME = "theme"
    PROCESSOR = "processor"
    MODEL = "model"
    INTEGRATION = "integration"


@dataclass
class PluginManifest:
    """Plugin manifest (plugin.json)."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    entry_point: str  # Module to load
    dependencies: list[str] = field(default_factory=list)  # Other plugins
    python_dependencies: list[str] = field(default_factory=list)  # pip packages
    min_forge_version: str = "0.0.0"
    homepage: str = ""
    license: str = ""
    tags: list[str] = field(default_factory=list)
    config_schema: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "type": self.plugin_type.value,
            "entry_point": self.entry_point,
            "dependencies": self.dependencies,
            "python_dependencies": self.python_dependencies,
            "min_forge_version": self.min_forge_version,
            "homepage": self.homepage,
            "license": self.license,
            "tags": self.tags,
            "config_schema": self.config_schema
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PluginManifest':
        return cls(
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            author=data.get("author", "Unknown"),
            plugin_type=PluginType(data.get("type", "tool")),
            entry_point=data.get("entry_point", "main"),
            dependencies=data.get("dependencies", []),
            python_dependencies=data.get("python_dependencies", []),
            min_forge_version=data.get("min_forge_version", "0.0.0"),
            homepage=data.get("homepage", ""),
            license=data.get("license", ""),
            tags=data.get("tags", []),
            config_schema=data.get("config_schema", {})
        )


@dataclass
class InstalledPlugin:
    """Represents an installed plugin."""
    manifest: PluginManifest
    install_path: Path
    installed_at: float
    status: InstallStatus = InstallStatus.INSTALLED
    enabled: bool = True
    config: dict = field(default_factory=dict)
    module: Any = None  # Loaded module
    error: Optional[str] = None
    
    @property
    def name(self) -> str:
        return self.manifest.name
    
    @property
    def version(self) -> str:
        return self.manifest.version


@dataclass
class RegistryPlugin:
    """Plugin info from registry."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    download_url: str
    checksum: str = ""
    downloads: int = 0
    rating: float = 0.0


class PluginInstaller:
    """Manages plugin installation and lifecycle."""
    
    def __init__(self, plugins_dir: Path, config_path: Optional[Path] = None):
        """
        Initialize plugin installer.
        
        Args:
            plugins_dir: Directory to install plugins
            config_path: Path to plugins config file
        """
        self._plugins_dir = Path(plugins_dir)
        self._plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self._config_path = config_path or (self._plugins_dir / "installed.json")
        self._installed: dict[str, InstalledPlugin] = {}
        self._callbacks: list[Callable[[str, str], None]] = []
        
        self._load_installed()
    
    def _load_installed(self):
        """Load list of installed plugins."""
        if not self._config_path.exists():
            return
        
        try:
            with open(self._config_path) as f:
                data = json.load(f)
            
            for plugin_data in data.get("plugins", []):
                name = plugin_data["name"]
                install_path = self._plugins_dir / name
                
                if install_path.exists():
                    manifest_path = install_path / "plugin.json"
                    if manifest_path.exists():
                        with open(manifest_path) as f:
                            manifest = PluginManifest.from_dict(json.load(f))
                        
                        self._installed[name] = InstalledPlugin(
                            manifest=manifest,
                            install_path=install_path,
                            installed_at=plugin_data.get("installed_at", 0),
                            enabled=plugin_data.get("enabled", True),
                            config=plugin_data.get("config", {})
                        )
        except Exception as e:
            logger.error(f"Failed to load installed plugins: {e}")
    
    def _save_installed(self):
        """Save list of installed plugins."""
        data = {
            "plugins": [
                {
                    "name": p.name,
                    "installed_at": p.installed_at,
                    "enabled": p.enabled,
                    "config": p.config
                }
                for p in self._installed.values()
            ]
        }
        
        with open(self._config_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def install(self, source: Path) -> tuple[bool, str]:
        """
        Install a plugin from a path (zip file or directory).
        
        Args:
            source: Path to plugin zip file or directory
            
        Returns:
            Tuple of (success, message)
        """
        source = Path(source)
        
        if not source.exists():
            return False, f"Path not found: {source}"
        
        if source.is_file() and source.suffix == '.zip':
            return self.install_from_zip(source)
        elif source.is_dir():
            return self.install_from_directory(source)
        else:
            return False, f"Invalid source: must be a .zip file or directory"
    
    def install_from_zip(self, zip_path: Path) -> tuple[bool, str]:
        """
        Install a plugin from a zip file.
        
        Args:
            zip_path: Path to plugin zip file
            
        Returns:
            Tuple of (success, message)
        """
        zip_path = Path(zip_path)
        
        if not zip_path.exists():
            return False, f"File not found: {zip_path}"
        
        try:
            # Extract to temp directory first
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(temp_path)
                
                # Find plugin.json
                manifest_path = temp_path / "plugin.json"
                if not manifest_path.exists():
                    # Check for nested directory
                    subdirs = [d for d in temp_path.iterdir() if d.is_dir()]
                    if len(subdirs) == 1:
                        manifest_path = subdirs[0] / "plugin.json"
                        temp_path = subdirs[0]
                
                if not manifest_path.exists():
                    return False, "plugin.json not found in archive"
                
                # Load and validate manifest
                with open(manifest_path) as f:
                    manifest = PluginManifest.from_dict(json.load(f))
                
                # Check dependencies
                missing_deps = self._check_dependencies(manifest)
                if missing_deps:
                    return False, f"Missing dependencies: {missing_deps}"
                
                # Install to plugins directory
                install_path = self._plugins_dir / manifest.name
                
                if install_path.exists():
                    # Backup existing
                    backup_path = install_path.with_suffix(".backup")
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    shutil.move(install_path, backup_path)
                
                shutil.copytree(temp_path, install_path)
                
                # Create installed plugin record
                self._installed[manifest.name] = InstalledPlugin(
                    manifest=manifest,
                    install_path=install_path,
                    installed_at=time.time()
                )
                
                self._save_installed()
                self._notify("installed", manifest.name)
                
                return True, f"Installed {manifest.name} v{manifest.version}"
                
        except Exception as e:
            logger.error(f"Install error: {e}")
            return False, str(e)
    
    def install_from_directory(self, source_dir: Path) -> tuple[bool, str]:
        """Install a plugin from a directory."""
        source_dir = Path(source_dir)
        
        manifest_path = source_dir / "plugin.json"
        if not manifest_path.exists():
            return False, "plugin.json not found"
        
        try:
            with open(manifest_path) as f:
                manifest = PluginManifest.from_dict(json.load(f))
            
            install_path = self._plugins_dir / manifest.name
            
            if install_path.exists():
                shutil.rmtree(install_path)
            
            shutil.copytree(source_dir, install_path)
            
            self._installed[manifest.name] = InstalledPlugin(
                manifest=manifest,
                install_path=install_path,
                installed_at=time.time()
            )
            
            self._save_installed()
            self._notify("installed", manifest.name)
            
            return True, f"Installed {manifest.name}"
            
        except Exception as e:
            return False, str(e)
    
    def uninstall(self, plugin_name: str) -> tuple[bool, str]:
        """
        Uninstall a plugin.
        
        Args:
            plugin_name: Name of plugin to uninstall
            
        Returns:
            Tuple of (success, message)
        """
        if plugin_name not in self._installed:
            return False, f"Plugin not installed: {plugin_name}"
        
        plugin = self._installed[plugin_name]
        
        # Unload if loaded
        if plugin.module:
            try:
                if hasattr(plugin.module, 'unload'):
                    plugin.module.unload()
            except Exception as e:
                logger.warning(f"Error unloading plugin {plugin_name}: {e}")
        
        # Remove files
        if plugin.install_path.exists():
            shutil.rmtree(plugin.install_path)
        
        del self._installed[plugin_name]
        self._save_installed()
        self._notify("uninstalled", plugin_name)
        
        return True, f"Uninstalled {plugin_name}"
    
    def enable(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        if plugin_name in self._installed:
            self._installed[plugin_name].enabled = True
            self._save_installed()
            return True
        return False
    
    def disable(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        if plugin_name in self._installed:
            self._installed[plugin_name].enabled = False
            self._save_installed()
            return True
        return False
    
    def load_plugin(self, plugin_name: str) -> tuple[bool, str]:
        """
        Load a plugin module.
        
        Args:
            plugin_name: Plugin to load
            
        Returns:
            Tuple of (success, message)
        """
        if plugin_name not in self._installed:
            return False, "Plugin not installed"
        
        plugin = self._installed[plugin_name]
        
        if not plugin.enabled:
            return False, "Plugin is disabled"
        
        try:
            # Find entry point
            entry_point = plugin.manifest.entry_point
            module_path = plugin.install_path / f"{entry_point}.py"
            
            if not module_path.exists():
                module_path = plugin.install_path / entry_point / "__init__.py"
            
            if not module_path.exists():
                return False, f"Entry point not found: {entry_point}"
            
            # Load module
            spec = importlib.util.spec_from_file_location(
                f"forge_plugins.{plugin_name}",
                module_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Initialize if has init function
            if hasattr(module, 'init'):
                module.init(plugin.config)
            
            plugin.module = module
            plugin.status = InstallStatus.INSTALLED
            plugin.error = None
            
            return True, f"Loaded {plugin_name}"
            
        except Exception as e:
            plugin.status = InstallStatus.ERROR
            plugin.error = str(e)
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return False, str(e)
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin module."""
        if plugin_name not in self._installed:
            return False
        
        plugin = self._installed[plugin_name]
        
        if plugin.module:
            try:
                if hasattr(plugin.module, 'unload'):
                    plugin.module.unload()
            except Exception as e:
                logger.warning(f"Error unloading plugin {plugin_name}: {e}")
            plugin.module = None
        
        return True
    
    def _check_dependencies(self, manifest: PluginManifest) -> list[str]:
        """Check for missing plugin dependencies."""
        missing = []
        for dep in manifest.dependencies:
            if dep not in self._installed:
                missing.append(dep)
        return missing
    
    def get_installed(self) -> list[InstalledPlugin]:
        """Get all installed plugins."""
        return list(self._installed.values())
    
    def get_plugin(self, name: str) -> Optional[InstalledPlugin]:
        """Get a specific installed plugin."""
        return self._installed.get(name)
    
    def is_installed(self, name: str) -> bool:
        """Check if a plugin is installed."""
        return name in self._installed
    
    def on_change(self, callback: Callable[[str, str], None]):
        """Register callback for install/uninstall events."""
        self._callbacks.append(callback)
    
    def _notify(self, action: str, plugin_name: str):
        """Notify callbacks of change."""
        for callback in self._callbacks:
            try:
                callback(action, plugin_name)
            except Exception as e:
                logger.warning(f"Plugin notification callback failed: {e}")


class PluginRegistryClient:
    """Client for remote plugin registry."""
    
    def __init__(self, registry_url: str = ""):
        """
        Initialize registry client.
        
        Args:
            registry_url: URL of plugin registry
        """
        self._registry_url = registry_url
        self._cache: dict[str, RegistryPlugin] = {}
        self._cache_time = 0.0
        self._cache_ttl = 3600  # 1 hour
    
    def search(self, 
               query: str = "",
               plugin_type: PluginType = None,
               tags: list[str] = None) -> list[RegistryPlugin]:
        """
        Search the registry.
        
        Note: Placeholder implementation without network calls.
        """
        # This would normally make HTTP requests to the registry
        # For now, return empty list
        return []
    
    def get_plugin_info(self, name: str) -> Optional[RegistryPlugin]:
        """Get information about a plugin from registry."""
        return self._cache.get(name)
    
    def download_plugin(self, 
                        name: str, 
                        destination: Path) -> tuple[bool, str]:
        """Download a plugin from registry."""
        # Placeholder - would download from registry_url
        return False, "Registry not configured"


# Singleton installer
_plugin_installer: Optional[PluginInstaller] = None


def get_plugin_installer(plugins_dir: Path = None) -> PluginInstaller:
    """Get the plugin installer singleton."""
    global _plugin_installer
    if _plugin_installer is None:
        if plugins_dir is None:
            plugins_dir = Path("plugins")
        _plugin_installer = PluginInstaller(plugins_dir)
    return _plugin_installer


__all__ = [
    'PluginInstaller',
    'InstalledPlugin',
    'PluginManifest',
    'PluginRegistryClient',
    'RegistryPlugin',
    'InstallStatus',
    'PluginType',
    'get_plugin_installer'
]
