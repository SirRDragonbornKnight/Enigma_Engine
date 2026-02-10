"""
================================================================================
PLUGIN INSTALLER - INSTALL AND MANAGE PLUGINS
================================================================================

Handles plugin installation, dependency resolution, and updates.

FILE: enigma_engine/marketplace/installer.py
TYPE: Plugin Installation
MAIN CLASSES: PluginInstaller, DependencyResolver
"""

import json
import logging
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import CONFIG
from .marketplace import InstallResult, PluginInfo

logger = logging.getLogger(__name__)


@dataclass
class DependencyNode:
    """Node in dependency graph."""
    plugin_id: str
    version: str
    dependencies: list[str]
    resolved: bool = False


class DependencyResolver:
    """
    Resolve plugin dependencies.
    
    Features:
    - Circular dependency detection
    - Version conflict detection
    - Installation order determination
    """
    
    def __init__(self, available_plugins: dict[str, PluginInfo]):
        """
        Initialize resolver.
        
        Args:
            available_plugins: Dictionary of available plugins
        """
        self.available = available_plugins
        self.installed: dict[str, str] = {}  # id -> version
    
    def set_installed(self, installed: dict[str, str]):
        """Set currently installed plugins."""
        self.installed = installed
    
    def resolve(self, plugin_id: str, version: str = None) -> tuple[list[str], list[str]]:
        """
        Resolve dependencies for a plugin.
        
        Args:
            plugin_id: Plugin to resolve
            version: Specific version (optional)
        
        Returns:
            Tuple of (install_order, errors)
        """
        errors = []
        
        # Get plugin info
        plugin = self.available.get(plugin_id)
        if not plugin:
            return [], [f"Plugin not found: {plugin_id}"]
        
        version = version or plugin.current_version
        
        # Build dependency graph
        graph: dict[str, DependencyNode] = {}
        to_process = [(plugin_id, version)]
        
        while to_process:
            pid, ver = to_process.pop(0)
            
            if pid in graph:
                continue
            
            p = self.available.get(pid)
            if not p:
                errors.append(f"Dependency not found: {pid}")
                continue
            
            # Find version info
            version_info = None
            for v in p.versions:
                if v.version == ver:
                    version_info = v
                    break
            
            if not version_info:
                # Use current version
                version_info = p.versions[0] if p.versions else None
            
            deps = version_info.dependencies if version_info else []
            
            graph[pid] = DependencyNode(
                plugin_id=pid,
                version=ver,
                dependencies=deps
            )
            
            # Add dependencies to process
            for dep in deps:
                if dep not in graph and dep not in self.installed:
                    dep_plugin = self.available.get(dep)
                    if dep_plugin:
                        to_process.append((dep, dep_plugin.current_version))
        
        # Check for circular dependencies
        circular = self._detect_circular(graph)
        if circular:
            errors.append(f"Circular dependency detected: {' -> '.join(circular)}")
            return [], errors
        
        # Topological sort for install order
        install_order = self._topological_sort(graph)
        
        # Filter out already installed
        install_order = [pid for pid in install_order if pid not in self.installed]
        
        return install_order, errors
    
    def _detect_circular(self, graph: dict[str, DependencyNode]) -> Optional[list[str]]:
        """Detect circular dependencies."""
        visited: set[str] = set()
        path: list[str] = []
        
        def dfs(node_id: str) -> Optional[list[str]]:
            if node_id in path:
                idx = path.index(node_id)
                return path[idx:] + [node_id]
            
            if node_id in visited:
                return None
            
            if node_id not in graph:
                return None
            
            visited.add(node_id)
            path.append(node_id)
            
            for dep in graph[node_id].dependencies:
                result = dfs(dep)
                if result:
                    return result
            
            path.pop()
            return None
        
        for node_id in graph:
            result = dfs(node_id)
            if result:
                return result
        
        return None
    
    def _topological_sort(self, graph: dict[str, DependencyNode]) -> list[str]:
        """Topological sort of dependency graph."""
        result = []
        visited: set[str] = set()
        
        def visit(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            
            if node_id in graph:
                for dep in graph[node_id].dependencies:
                    visit(dep)
            
            result.append(node_id)
        
        for node_id in graph:
            visit(node_id)
        
        return result


class PluginInstaller:
    """
    Install and manage plugins.
    
    Features:
    - Install with dependency resolution
    - Update plugins
    - Uninstall with cleanup
    - Backup before update
    """
    
    def __init__(self, plugins_dir: Path = None):
        """
        Initialize installer.
        
        Args:
            plugins_dir: Directory for installed plugins
        """
        self.plugins_dir = plugins_dir or Path(CONFIG.get("plugins_dir", "plugins"))
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self.backup_dir = self.plugins_dir / ".backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        self._installed: dict[str, str] = {}
        self._load_installed()
    
    def _load_installed(self):
        """Load installed plugins list."""
        manifest = self.plugins_dir / "manifest.json"
        if manifest.exists():
            try:
                with open(manifest) as f:
                    data = json.load(f)
                self._installed = data.get('installed', {})
            except Exception as e:
                logger.warning(f"Error loading manifest: {e}")
    
    def _save_installed(self):
        """Save installed plugins list."""
        manifest = self.plugins_dir / "manifest.json"
        with open(manifest, 'w') as f:
            json.dump({
                'installed': self._installed,
                'plugins_dir': str(self.plugins_dir),
            }, f, indent=2)
    
    @property
    def installed(self) -> dict[str, str]:
        """Get installed plugins."""
        return self._installed.copy()
    
    def install(self, source: Path, plugin_id: str = None, version: str = None) -> InstallResult:
        """
        Install a plugin from a path (package file or directory).
        
        Args:
            source: Path to plugin package or directory
            plugin_id: Optional plugin ID (extracted from source if not provided)
            version: Optional version (extracted from source if not provided)
            
        Returns:
            InstallResult with success status and details
        """
        source = Path(source)
        
        if not source.exists():
            return InstallResult(
                success=False,
                message=f"Source not found: {source}",
                plugin_id=plugin_id or "",
                version=version or "",
            )
        
        # Extract plugin_id and version from path if not provided
        if not plugin_id:
            plugin_id = source.stem
        if not version:
            version = "1.0.0"
        
        return self.install_package(source, plugin_id, version)
    
    def install_package(
        self,
        package_path: Path,
        plugin_id: str,
        version: str
    ) -> InstallResult:
        """
        Install a plugin from a package file.
        
        Args:
            package_path: Path to .zip package
            plugin_id: Plugin ID
            version: Version string
        
        Returns:
            InstallResult
        """
        install_path = self.plugins_dir / plugin_id
        
        # Backup existing if updating
        if install_path.exists():
            self._backup_plugin(plugin_id)
        
        try:
            # Create plugin directory
            install_path.mkdir(parents=True, exist_ok=True)
            
            # Extract package
            with zipfile.ZipFile(package_path, 'r') as zf:
                zf.extractall(install_path)
            
            # Update manifest
            self._installed[plugin_id] = version
            self._save_installed()
            
            logger.info(f"Installed {plugin_id} v{version}")
            
            return InstallResult(
                success=True,
                plugin_id=plugin_id,
                version=version,
                message=f"Successfully installed {plugin_id}",
                installed_path=install_path
            )
            
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            
            # Restore backup if exists
            self._restore_backup(plugin_id)
            
            return InstallResult(
                success=False,
                plugin_id=plugin_id,
                version=version,
                message=f"Installation failed: {e}",
                errors=[str(e)]
            )
    
    def uninstall(self, plugin_id: str, keep_backup: bool = True) -> bool:
        """
        Uninstall a plugin.
        
        Args:
            plugin_id: Plugin to uninstall
            keep_backup: Whether to keep a backup
        
        Returns:
            True if successful
        """
        if plugin_id not in self._installed:
            logger.warning(f"Plugin not installed: {plugin_id}")
            return False
        
        install_path = self.plugins_dir / plugin_id
        
        if keep_backup and install_path.exists():
            self._backup_plugin(plugin_id)
        
        # Remove plugin directory
        if install_path.exists():
            shutil.rmtree(install_path)
        
        # Update manifest
        del self._installed[plugin_id]
        self._save_installed()
        
        logger.info(f"Uninstalled {plugin_id}")
        return True
    
    def _backup_plugin(self, plugin_id: str):
        """Create backup of installed plugin."""
        install_path = self.plugins_dir / plugin_id
        if not install_path.exists():
            return
        
        version = self._installed.get(plugin_id, "unknown")
        backup_name = f"{plugin_id}-{version}.zip"
        backup_path = self.backup_dir / backup_name
        
        # Create zip backup
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in install_path.rglob('*'):
                if file.is_file():
                    zf.write(file, file.relative_to(install_path))
        
        logger.info(f"Created backup: {backup_path}")
    
    def _restore_backup(self, plugin_id: str) -> bool:
        """Restore plugin from backup."""
        # Find most recent backup
        backups = list(self.backup_dir.glob(f"{plugin_id}-*.zip"))
        if not backups:
            return False
        
        # Sort by modification time
        backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        backup_path = backups[0]
        
        install_path = self.plugins_dir / plugin_id
        
        # Remove current (broken) installation
        if install_path.exists():
            shutil.rmtree(install_path)
        
        # Restore from backup
        install_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(backup_path, 'r') as zf:
            zf.extractall(install_path)
        
        logger.info(f"Restored from backup: {backup_path}")
        return True
    
    def get_plugin_path(self, plugin_id: str) -> Optional[Path]:
        """Get installation path for a plugin."""
        if plugin_id not in self._installed:
            return None
        return self.plugins_dir / plugin_id
    
    def is_installed(self, plugin_id: str) -> bool:
        """Check if plugin is installed."""
        return plugin_id in self._installed
    
    def get_version(self, plugin_id: str) -> Optional[str]:
        """Get installed version of a plugin."""
        return self._installed.get(plugin_id)
    
    def list_backups(self) -> list[tuple[str, str, Path]]:
        """List available backups."""
        backups = []
        for backup in self.backup_dir.glob("*.zip"):
            # Parse plugin_id-version.zip
            name = backup.stem
            parts = name.rsplit('-', 1)
            if len(parts) == 2:
                plugin_id, version = parts
                backups.append((plugin_id, version, backup))
        return backups
    
    def cleanup_backups(self, keep: int = 3):
        """
        Clean up old backups.
        
        Args:
            keep: Number of backups to keep per plugin
        """
        # Group by plugin
        by_plugin: dict[str, list[Path]] = {}
        for backup in self.backup_dir.glob("*.zip"):
            name = backup.stem
            parts = name.rsplit('-', 1)
            if len(parts) == 2:
                plugin_id = parts[0]
                if plugin_id not in by_plugin:
                    by_plugin[plugin_id] = []
                by_plugin[plugin_id].append(backup)
        
        # Clean up each plugin's backups
        for plugin_id, backups in by_plugin.items():
            # Sort by modification time (newest first)
            backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            # Remove old backups
            for backup in backups[keep:]:
                backup.unlink()
                logger.info(f"Removed old backup: {backup}")
