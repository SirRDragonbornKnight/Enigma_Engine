"""
Tool Plugin Discovery and Loading
==================================

Discovers and loads custom tool plugins from configured directories.
Supports hot-reload for development without restarting the application.
"""

import importlib
import importlib.util
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class PluginFileWatcher:
    """
    Watch plugin files for changes and trigger hot-reload.
    
    Uses file modification time checking (portable, no dependencies).
    """
    
    def __init__(
        self,
        plugin_loader: 'ToolPluginLoader',
        poll_interval: float = 2.0
    ):
        self.plugin_loader = plugin_loader
        self.poll_interval = poll_interval
        self._watching = False
        self._watch_thread: Optional[threading.Thread] = None
        self._file_mtimes: Dict[str, float] = {}
        self._callbacks: List[Callable[[str, str], None]] = []
        
    def add_callback(self, callback: Callable[[str, str], None]):
        """
        Add a callback for file change events.
        
        Args:
            callback: Function taking (plugin_name, event_type) where
                     event_type is 'modified', 'added', or 'removed'
        """
        self._callbacks.append(callback)
    
    def start(self):
        """Start watching for file changes."""
        if self._watching:
            return
        
        self._watching = True
        
        # Initialize modification times
        self._refresh_mtimes()
        
        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()
        logger.info("Plugin file watcher started")
    
    def stop(self):
        """Stop watching for file changes."""
        self._watching = False
        if self._watch_thread:
            self._watch_thread.join(timeout=5.0)
            self._watch_thread = None
        logger.info("Plugin file watcher stopped")
    
    def _refresh_mtimes(self):
        """Refresh the modification time cache."""
        for plugin_name, plugin_data in self.plugin_loader.discovered_plugins.items():
            plugin_path: Path = plugin_data["path"]
            if plugin_path.exists():
                self._file_mtimes[plugin_name] = plugin_path.stat().st_mtime
    
    def _watch_loop(self):
        """Main watch loop - check for file changes periodically."""
        while self._watching:
            try:
                self._check_for_changes()
            except Exception as e:
                logger.debug(f"Plugin watch error: {e}")
            
            time.sleep(self.poll_interval)
    
    def _check_for_changes(self):
        """Check for file modifications and trigger reloads."""
        # Check for new plugins
        for plugin_dir in self.plugin_loader.plugin_dirs:
            if not plugin_dir.exists():
                continue
            
            for plugin_file in plugin_dir.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue
                
                plugin_name = plugin_file.stem
                current_mtime = plugin_file.stat().st_mtime
                
                if plugin_name not in self.plugin_loader.discovered_plugins:
                    # New plugin discovered
                    plugin_info = self.plugin_loader._inspect_plugin(plugin_file)
                    if plugin_info:
                        self.plugin_loader.discovered_plugins[plugin_name] = {
                            "path": plugin_file,
                            "info": plugin_info,
                        }
                        self._file_mtimes[plugin_name] = current_mtime
                        logger.info(f"New plugin discovered: {plugin_name}")
                        self._notify_callbacks(plugin_name, "added")
                
                elif plugin_name in self._file_mtimes:
                    # Check if modified
                    if current_mtime > self._file_mtimes[plugin_name]:
                        logger.info(f"Plugin modified: {plugin_name}")
                        self._file_mtimes[plugin_name] = current_mtime
                        
                        # Auto-reload if was loaded
                        if plugin_name in self.plugin_loader.loaded_plugins:
                            self.plugin_loader.reload_plugin(plugin_name)
                        
                        self._notify_callbacks(plugin_name, "modified")
        
        # Check for removed plugins
        to_remove = []
        for plugin_name, plugin_data in self.plugin_loader.discovered_plugins.items():
            plugin_path: Path = plugin_data["path"]
            if not plugin_path.exists():
                to_remove.append(plugin_name)
        
        for plugin_name in to_remove:
            logger.info(f"Plugin removed: {plugin_name}")
            
            # Unload if was loaded
            if plugin_name in self.plugin_loader.loaded_plugins:
                self.plugin_loader.unload_plugin(plugin_name)
            
            del self.plugin_loader.discovered_plugins[plugin_name]
            if plugin_name in self._file_mtimes:
                del self._file_mtimes[plugin_name]
            
            self._notify_callbacks(plugin_name, "removed")
    
    def _notify_callbacks(self, plugin_name: str, event_type: str):
        """Notify all callbacks of a plugin event."""
        for callback in self._callbacks:
            try:
                callback(plugin_name, event_type)
            except Exception as e:
                logger.debug(f"Plugin callback error: {e}")


class ToolPluginLoader:
    """
    Discover and load tool plugins.
    
    Features:
    - Auto-discovery from plugin directories
    - Dynamic loading and registration
    - Plugin validation
    - Dependency checking
    - Hot-reload support for development
    """
    
    def __init__(
        self,
        plugin_dirs: Optional[list[Path]] = None,
        auto_discover: bool = True,
        enable_hot_reload: bool = False,
        hot_reload_interval: float = 2.0
    ):
        """
        Initialize plugin loader.
        
        Args:
            plugin_dirs: List of directories to search for plugins
            auto_discover: Auto-discover plugins on init
            enable_hot_reload: Auto-reload plugins when files change
            hot_reload_interval: Seconds between file change checks
        """
        # Default plugin directories
        if plugin_dirs is None:
            plugin_dirs = [
                Path.home() / ".enigma_engine" / "tool_plugins",
                Path("./tool_plugins"),
            ]
        
        self.plugin_dirs = plugin_dirs
        self.discovered_plugins: dict[str, dict[str, Any]] = {}
        self.loaded_plugins: set[str] = set()
        
        # Hot-reload support
        self._file_watcher: Optional[PluginFileWatcher] = None
        
        # Callbacks
        self._on_loaded_callbacks: List[Callable[[str], None]] = []
        self._on_unloaded_callbacks: List[Callable[[str], None]] = []
        self._on_reloaded_callbacks: List[Callable[[str], None]] = []
        
        logger.info(f"ToolPluginLoader initialized with {len(plugin_dirs)} search paths")
        
        if auto_discover:
            self.discover_plugins()
        
        if enable_hot_reload:
            self.enable_hot_reload(hot_reload_interval)
    
    def discover_plugins(self) -> list[str]:
        """
        Discover plugins in configured directories.
        
        Returns:
            List of discovered plugin names
        """
        discovered = []
        
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                logger.debug(f"Plugin directory does not exist: {plugin_dir}")
                continue
            
            logger.info(f"Scanning for plugins in: {plugin_dir}")
            
            # Look for Python files
            for plugin_file in plugin_dir.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue  # Skip private files
                
                plugin_name = plugin_file.stem
                
                try:
                    plugin_info = self._inspect_plugin(plugin_file)
                    
                    if plugin_info:
                        self.discovered_plugins[plugin_name] = {
                            "path": plugin_file,
                            "info": plugin_info,
                        }
                        discovered.append(plugin_name)
                        logger.info(f"Discovered plugin: {plugin_name}")
                
                except Exception as e:
                    logger.warning(f"Failed to inspect plugin {plugin_name}: {e}")
        
        logger.info(f"Discovered {len(discovered)} plugin(s)")
        return discovered
    
    def _inspect_plugin(self, plugin_file: Path) -> Optional[dict[str, Any]]:
        """
        Inspect a plugin file for metadata.
        
        Args:
            plugin_file: Path to plugin file
            
        Returns:
            Plugin metadata dict or None if invalid
        """
        # Try to read plugin metadata without importing
        try:
            content = plugin_file.read_text()
            
            # Look for plugin metadata in comments or docstring
            plugin_info = {
                "name": plugin_file.stem,
                "description": "Custom tool plugin",
                "version": "1.0.0",
                "author": "Unknown",
            }
            
            # Parse docstring if present
            if '"""' in content:
                start = content.find('"""') + 3
                end = content.find('"""', start)
                if end > start:
                    docstring = content[start:end].strip()
                    plugin_info["description"] = docstring.split('\n')[0]
            
            return plugin_info
        
        except Exception as e:
            logger.warning(f"Failed to inspect {plugin_file}: {e}")
            return None
    
    def load_plugin(self, plugin_name: str) -> bool:
        """
        Load a specific plugin.
        
        Args:
            plugin_name: Name of the plugin to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if plugin_name not in self.discovered_plugins:
            logger.error(f"Plugin not found: {plugin_name}")
            return False
        
        if plugin_name in self.loaded_plugins:
            logger.info(f"Plugin already loaded: {plugin_name}")
            return True
        
        plugin_data = self.discovered_plugins[plugin_name]
        plugin_path = plugin_data["path"]
        
        try:
            # Add plugin directory to path if needed
            plugin_dir = plugin_path.parent
            if str(plugin_dir) not in sys.path:
                sys.path.insert(0, str(plugin_dir))
            
            # Import the plugin module
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if spec is None or spec.loader is None:
                logger.error(f"Failed to create module spec for {plugin_name}")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for tool definitions or registration function
            if hasattr(module, "register_tools"):
                # Plugin has a registration function
                tools = module.register_tools()
                logger.info(f"Loaded {len(tools) if isinstance(tools, list) else 0} tools from {plugin_name}")
            elif hasattr(module, "TOOLS"):
                # Plugin exports TOOLS list
                tools = module.TOOLS
                logger.info(f"Loaded {len(tools)} tools from {plugin_name}")
            else:
                logger.warning(f"Plugin {plugin_name} has no tools to register")
            
            self.loaded_plugins.add(plugin_name)
            
            # Store module reference for potential reload
            self.discovered_plugins[plugin_name]["module"] = module
            
            # Notify callbacks
            self._notify_load(plugin_name)
            
            logger.info(f"Successfully loaded plugin: {plugin_name}")
            return True
        
        except Exception as e:
            logger.exception(f"Failed to load plugin {plugin_name}: {e}")
            return False
    
    def load_all(self) -> dict[str, bool]:
        """
        Load all discovered plugins.
        
        Returns:
            Dictionary mapping plugin name to load success
        """
        results = {}
        
        for plugin_name in self.discovered_plugins:
            success = self.load_plugin(plugin_name)
            results[plugin_name] = success
        
        return results
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if unloaded successfully
        """
        if plugin_name not in self.loaded_plugins:
            logger.warning(f"Plugin not loaded: {plugin_name}")
            return False
        
        try:
            # Remove from loaded set
            self.loaded_plugins.remove(plugin_name)
            
            # Try to remove from sys.modules
            if plugin_name in sys.modules:
                del sys.modules[plugin_name]
            
            # Notify callbacks
            self._notify_unload(plugin_name)
            
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
        
        except Exception as e:
            logger.exception(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """
        Hot-reload a plugin without restarting the application.
        
        Args:
            plugin_name: Name of the plugin to reload
            
        Returns:
            True if reloaded successfully
        """
        if plugin_name not in self.discovered_plugins:
            logger.error(f"Plugin not found: {plugin_name}")
            return False
        
        was_loaded = plugin_name in self.loaded_plugins
        
        try:
            # Unload first if loaded
            if was_loaded:
                self.unload_plugin(plugin_name)
            
            # Re-inspect the plugin (metadata may have changed)
            plugin_path = self.discovered_plugins[plugin_name]["path"]
            plugin_info = self._inspect_plugin(plugin_path)
            if plugin_info:
                self.discovered_plugins[plugin_name]["info"] = plugin_info
            
            # Reload the module
            if was_loaded:
                success = self.load_plugin(plugin_name)
                if success:
                    logger.info(f"Hot-reloaded plugin: {plugin_name}")
                    # Notify reload callbacks
                    self._notify_reload(plugin_name)
                return success
            
            logger.info(f"Plugin refreshed (not loaded): {plugin_name}")
            return True
        
        except Exception as e:
            logger.exception(f"Failed to hot-reload plugin {plugin_name}: {e}")
            return False
    
    def reload_all(self) -> dict[str, bool]:
        """
        Hot-reload all loaded plugins.
        
        Returns:
            Dictionary mapping plugin name to reload success
        """
        results = {}
        
        for plugin_name in list(self.loaded_plugins):
            success = self.reload_plugin(plugin_name)
            results[plugin_name] = success
        
        return results
    
    def enable_hot_reload(self, poll_interval: float = 2.0) -> PluginFileWatcher:
        """
        Enable automatic hot-reload when plugin files change.
        
        Args:
            poll_interval: Seconds between file change checks
            
        Returns:
            The file watcher instance
        """
        if self._file_watcher is not None:
            # Already enabled
            return self._file_watcher
        
        self._file_watcher = PluginFileWatcher(self, poll_interval)
        self._file_watcher.start()
        logger.info(f"Plugin hot-reload enabled (poll interval: {poll_interval}s)")
        return self._file_watcher
    
    def disable_hot_reload(self):
        """Disable automatic hot-reload."""
        if self._file_watcher is not None:
            self._file_watcher.stop()
            self._file_watcher = None
            logger.info("Plugin hot-reload disabled")
    
    def is_hot_reload_enabled(self) -> bool:
        """Check if hot-reload is enabled."""
        return self._file_watcher is not None
    
    # ===== Callbacks =====
    
    def on_plugin_loaded(self, callback: Callable[[str], None]):
        """Register callback when a plugin is loaded."""
        self._on_loaded_callbacks.append(callback)
    
    def on_plugin_unloaded(self, callback: Callable[[str], None]):
        """Register callback when a plugin is unloaded."""
        self._on_unloaded_callbacks.append(callback)
    
    def on_plugin_reloaded(self, callback: Callable[[str], None]):
        """Register callback when a plugin is hot-reloaded."""
        self._on_reloaded_callbacks.append(callback)
    
    def _notify_load(self, plugin_name: str):
        for cb in self._on_loaded_callbacks:
            try:
                cb(plugin_name)
            except Exception as e:
                logger.debug(f"Load callback error: {e}")
    
    def _notify_unload(self, plugin_name: str):
        for cb in self._on_unloaded_callbacks:
            try:
                cb(plugin_name)
            except Exception as e:
                logger.debug(f"Unload callback error: {e}")
    
    def _notify_reload(self, plugin_name: str):
        for cb in self._on_reloaded_callbacks:
            try:
                cb(plugin_name)
            except Exception as e:
                logger.debug(f"Reload callback error: {e}")
    
    def get_plugin_info(self, plugin_name: str) -> Optional[dict[str, Any]]:
        """
        Get information about a plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin info dictionary or None
        """
        if plugin_name not in self.discovered_plugins:
            return None
        
        plugin_data = self.discovered_plugins[plugin_name]
        info = plugin_data["info"].copy()
        info["loaded"] = plugin_name in self.loaded_plugins
        info["path"] = str(plugin_data["path"])
        
        return info
    
    def list_plugins(self) -> list[dict[str, Any]]:
        """
        List all discovered plugins.
        
        Returns:
            List of plugin info dictionaries
        """
        plugins = []
        
        for plugin_name in self.discovered_plugins:
            info = self.get_plugin_info(plugin_name)
            if info:
                plugins.append(info)
        
        return plugins
    
    def get_statistics(self) -> dict[str, Any]:
        """Get plugin loader statistics."""
        return {
            "plugin_dirs": [str(p) for p in self.plugin_dirs],
            "discovered_plugins": len(self.discovered_plugins),
            "loaded_plugins": len(self.loaded_plugins),
            "plugin_names": list(self.discovered_plugins.keys()),
            "hot_reload_enabled": self.is_hot_reload_enabled(),
        }


__all__ = [
    "ToolPluginLoader",
    "PluginFileWatcher",
]
