"""
Tool Plugin Discovery and Loading
==================================

Discovers and loads custom tool plugins from configured directories.
"""

import logging
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Any

logger = logging.getLogger(__name__)


class ToolPluginLoader:
    """
    Discover and load tool plugins.
    
    Features:
    - Auto-discovery from plugin directories
    - Dynamic loading and registration
    - Plugin validation
    - Dependency checking
    """
    
    def __init__(
        self,
        plugin_dirs: Optional[List[Path]] = None,
        auto_discover: bool = True
    ):
        """
        Initialize plugin loader.
        
        Args:
            plugin_dirs: List of directories to search for plugins
            auto_discover: Auto-discover plugins on init
        """
        # Default plugin directories
        if plugin_dirs is None:
            plugin_dirs = [
                Path.home() / ".ai_tester" / "tool_plugins",
                Path("./tool_plugins"),
            ]
        
        self.plugin_dirs = plugin_dirs
        self.discovered_plugins: Dict[str, Dict[str, Any]] = {}
        self.loaded_plugins: Set[str] = set()
        
        logger.info(f"ToolPluginLoader initialized with {len(plugin_dirs)} search paths")
        
        if auto_discover:
            self.discover_plugins()
    
    def discover_plugins(self) -> List[str]:
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
    
    def _inspect_plugin(self, plugin_file: Path) -> Optional[Dict[str, Any]]:
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
            logger.info(f"Successfully loaded plugin: {plugin_name}")
            return True
        
        except Exception as e:
            logger.exception(f"Failed to load plugin {plugin_name}: {e}")
            return False
    
    def load_all(self) -> Dict[str, bool]:
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
            
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
        
        except Exception as e:
            logger.exception(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
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
    
    def list_plugins(self) -> List[Dict[str, Any]]:
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin loader statistics."""
        return {
            "plugin_dirs": [str(p) for p in self.plugin_dirs],
            "discovered_plugins": len(self.discovered_plugins),
            "loaded_plugins": len(self.loaded_plugins),
            "plugin_names": list(self.discovered_plugins.keys()),
        }


__all__ = [
    "ToolPluginLoader",
]
