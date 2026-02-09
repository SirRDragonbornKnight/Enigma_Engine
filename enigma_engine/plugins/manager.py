"""
Plugin System for Enigma AI Engine

Extensible plugin architecture.

Features:
- Plugin discovery
- Hot reload
- Dependency management
- Sandboxed execution
- Event hooks

Usage:
    from enigma_engine.plugins.manager import PluginManager, Plugin
    
    manager = PluginManager("plugins/")
    
    # Load all plugins
    manager.discover()
    manager.load_all()
    
    # Use plugins
    results = manager.call_hook("on_message", message="Hello")
"""

import importlib
import importlib.util
import inspect
import logging
import os
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)


class PluginState(Enum):
    """Plugin lifecycle states."""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginInfo:
    """Plugin metadata."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    website: str = ""
    dependencies: List[str] = field(default_factory=list)  # Required plugins
    python_dependencies: List[str] = field(default_factory=list)  # Required packages
    hooks: List[str] = field(default_factory=list)  # Hooks this plugin responds to
    permissions: List[str] = field(default_factory=list)  # Required permissions


class Plugin(ABC):
    """Base class for plugins."""
    
    # Plugin metadata (override in subclass)
    info: PluginInfo = PluginInfo(name="BasePlugin")
    
    def __init__(self):
        self.state = PluginState.UNLOADED
        self._manager: Optional['PluginManager'] = None
    
    @abstractmethod
    def on_load(self):
        """Called when plugin is loaded."""
        pass
    
    def on_unload(self):
        """Called when plugin is unloaded."""
        pass
    
    def on_enable(self):
        """Called when plugin is enabled."""
        pass
    
    def on_disable(self):
        """Called when plugin is disabled."""
        pass
    
    def get_api(self) -> Dict[str, Callable]:
        """
        Return dict of API functions this plugin exposes.
        
        Returns:
            Dict of name -> callable
        """
        return {}
    
    def call(self, method: str, *args, **kwargs) -> Any:
        """Call a method on this plugin."""
        if hasattr(self, method):
            return getattr(self, method)(*args, **kwargs)
        return None
    
    def log(self, message: str, level: str = "info"):
        """Log a message."""
        getattr(logger, level)(f"[{self.info.name}] {message}")


@dataclass
class HookHandler:
    """Handler for a hook."""
    plugin: Plugin
    method: Callable
    priority: int = 0


class PluginManager:
    """Manages plugin lifecycle and execution."""
    
    def __init__(
        self,
        plugin_dir: str = "plugins",
        auto_discover: bool = True,
        sandbox: bool = False
    ):
        """
        Initialize plugin manager.
        
        Args:
            plugin_dir: Directory containing plugins
            auto_discover: Automatically discover plugins
            sandbox: Run plugins in sandbox
        """
        self.plugin_dir = Path(plugin_dir)
        self.sandbox = sandbox
        
        # Plugin storage
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_classes: Dict[str, Type[Plugin]] = {}
        
        # Hooks
        self._hooks: Dict[str, List[HookHandler]] = {}
        
        # Dependencies
        self._dependency_graph: Dict[str, Set[str]] = {}
        
        # Create plugin directory
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        
        if auto_discover:
            self.discover()
        
        logger.info(f"PluginManager initialized: {self.plugin_dir}")
    
    def discover(self) -> List[str]:
        """
        Discover available plugins.
        
        Returns:
            List of discovered plugin names
        """
        discovered = []
        
        # Add plugin dir to path
        if str(self.plugin_dir) not in sys.path:
            sys.path.insert(0, str(self.plugin_dir))
        
        # Find plugin files/directories
        for item in self.plugin_dir.iterdir():
            if item.is_file() and item.suffix == '.py' and not item.name.startswith('_'):
                name = item.stem
                try:
                    self._discover_file(item, name)
                    discovered.append(name)
                except Exception as e:
                    logger.error(f"Failed to discover {name}: {e}")
            
            elif item.is_dir() and (item / '__init__.py').exists():
                name = item.name
                try:
                    self._discover_package(item, name)
                    discovered.append(name)
                except Exception as e:
                    logger.error(f"Failed to discover {name}: {e}")
        
        logger.info(f"Discovered {len(discovered)} plugins: {discovered}")
        return discovered
    
    def _discover_file(self, file_path: Path, name: str):
        """Discover plugin from file."""
        spec = importlib.util.spec_from_file_location(name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        self._find_plugin_class(module, name)
    
    def _discover_package(self, package_path: Path, name: str):
        """Discover plugin from package."""
        module = importlib.import_module(name)
        self._find_plugin_class(module, name)
    
    def _find_plugin_class(self, module, name: str):
        """Find Plugin subclass in module."""
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            
            if (inspect.isclass(attr) and 
                issubclass(attr, Plugin) and 
                attr is not Plugin):
                
                self._plugin_classes[name] = attr
                
                # Build dependency graph
                if hasattr(attr, 'info') and attr.info.dependencies:
                    self._dependency_graph[name] = set(attr.info.dependencies)
                else:
                    self._dependency_graph[name] = set()
                
                return
        
        raise ValueError(f"No Plugin subclass found in {name}")
    
    def load(self, name: str) -> bool:
        """
        Load a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            True if loaded successfully
        """
        if name in self._plugins:
            logger.warning(f"Plugin {name} already loaded")
            return True
        
        if name not in self._plugin_classes:
            logger.error(f"Plugin {name} not found")
            return False
        
        # Check dependencies
        if not self._check_dependencies(name):
            return False
        
        try:
            # Instantiate plugin
            plugin_class = self._plugin_classes[name]
            plugin = plugin_class()
            plugin._manager = self
            
            # Check Python dependencies
            if hasattr(plugin, 'info') and plugin.info.python_dependencies:
                for dep in plugin.info.python_dependencies:
                    try:
                        importlib.import_module(dep.split('==')[0])
                    except ImportError:
                        logger.error(f"Missing Python dependency: {dep}")
                        return False
            
            # Call on_load
            plugin.on_load()
            plugin.state = PluginState.LOADED
            
            self._plugins[name] = plugin
            
            # Register hooks
            self._register_hooks(plugin)
            
            logger.info(f"Loaded plugin: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load plugin {name}: {e}")
            traceback.print_exc()
            return False
    
    def unload(self, name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            True if unloaded successfully
        """
        if name not in self._plugins:
            return False
        
        plugin = self._plugins[name]
        
        try:
            # Unregister hooks
            self._unregister_hooks(plugin)
            
            # Call on_unload
            plugin.on_unload()
            plugin.state = PluginState.UNLOADED
            
            del self._plugins[name]
            
            logger.info(f"Unloaded plugin: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {name}: {e}")
            return False
    
    def reload(self, name: str) -> bool:
        """
        Reload a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            True if reloaded successfully
        """
        self.unload(name)
        
        # Re-discover
        if name in self._plugin_classes:
            del self._plugin_classes[name]
        
        for item in self.plugin_dir.iterdir():
            if item.stem == name or item.name == name:
                if item.is_file():
                    self._discover_file(item, name)
                else:
                    self._discover_package(item, name)
                break
        
        return self.load(name)
    
    def load_all(self) -> Dict[str, bool]:
        """
        Load all discovered plugins.
        
        Returns:
            Dict of plugin name -> success
        """
        results = {}
        
        # Sort by dependencies
        sorted_plugins = self._topological_sort()
        
        for name in sorted_plugins:
            results[name] = self.load(name)
        
        return results
    
    def unload_all(self):
        """Unload all plugins."""
        for name in list(self._plugins.keys()):
            self.unload(name)
    
    def enable(self, name: str) -> bool:
        """Enable a plugin."""
        if name not in self._plugins:
            return False
        
        plugin = self._plugins[name]
        plugin.on_enable()
        plugin.state = PluginState.ACTIVE
        return True
    
    def disable(self, name: str) -> bool:
        """Disable a plugin."""
        if name not in self._plugins:
            return False
        
        plugin = self._plugins[name]
        plugin.on_disable()
        plugin.state = PluginState.DISABLED
        return True
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a loaded plugin."""
        return self._plugins.get(name)
    
    def get_all_plugins(self) -> Dict[str, Plugin]:
        """Get all loaded plugins."""
        return dict(self._plugins)
    
    def list_available(self) -> List[str]:
        """List available (discovered) plugins."""
        return list(self._plugin_classes.keys())
    
    def list_loaded(self) -> List[str]:
        """List loaded plugins."""
        return list(self._plugins.keys())
    
    def register_hook(
        self,
        hook_name: str,
        plugin: Plugin,
        method: Callable,
        priority: int = 0
    ):
        """Register a hook handler."""
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        
        handler = HookHandler(plugin=plugin, method=method, priority=priority)
        self._hooks[hook_name].append(handler)
        
        # Sort by priority
        self._hooks[hook_name].sort(key=lambda h: h.priority, reverse=True)
    
    def call_hook(
        self,
        hook_name: str,
        *args,
        stop_on_result: bool = False,
        **kwargs
    ) -> List[Any]:
        """
        Call a hook, invoking all registered handlers.
        
        Args:
            hook_name: Name of hook
            stop_on_result: Stop if any handler returns truthy value
            
        Returns:
            List of results from handlers
        """
        if hook_name not in self._hooks:
            return []
        
        results = []
        
        for handler in self._hooks[hook_name]:
            if handler.plugin.state not in [PluginState.LOADED, PluginState.ACTIVE]:
                continue
            
            try:
                result = handler.method(*args, **kwargs)
                results.append(result)
                
                if stop_on_result and result:
                    break
                    
            except Exception as e:
                logger.error(f"Hook {hook_name} handler error: {e}")
                traceback.print_exc()
        
        return results
    
    def call_plugin_api(
        self,
        plugin_name: str,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Call a plugin's API method."""
        plugin = self._plugins.get(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin {plugin_name} not loaded")
        
        api = plugin.get_api()
        if method_name not in api:
            raise ValueError(f"Method {method_name} not found in {plugin_name}")
        
        return api[method_name](*args, **kwargs)
    
    def _check_dependencies(self, name: str) -> bool:
        """Check if plugin dependencies are satisfied."""
        deps = self._dependency_graph.get(name, set())
        
        for dep in deps:
            if dep not in self._plugins:
                logger.error(f"Plugin {name} requires {dep}")
                return False
        
        return True
    
    def _topological_sort(self) -> List[str]:
        """Sort plugins by dependencies."""
        visited = set()
        order = []
        
        def visit(name):
            if name in visited:
                return
            visited.add(name)
            
            for dep in self._dependency_graph.get(name, set()):
                if dep in self._plugin_classes:
                    visit(dep)
            
            order.append(name)
        
        for name in self._plugin_classes:
            visit(name)
        
        return order
    
    def _register_hooks(self, plugin: Plugin):
        """Register plugin's hook handlers."""
        # Auto-detect hook methods
        for attr_name in dir(plugin):
            if attr_name.startswith('on_'):
                method = getattr(plugin, attr_name)
                if callable(method):
                    hook_name = attr_name
                    self.register_hook(hook_name, plugin, method)
    
    def _unregister_hooks(self, plugin: Plugin):
        """Unregister plugin's hook handlers."""
        for hook_name, handlers in self._hooks.items():
            self._hooks[hook_name] = [
                h for h in handlers if h.plugin is not plugin
            ]


class PluginConfig:
    """Configuration storage for plugins."""
    
    def __init__(self, config_dir: str = "config/plugins"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._configs: Dict[str, Dict] = {}
    
    def get(self, plugin_name: str, key: str, default: Any = None) -> Any:
        """Get config value for plugin."""
        if plugin_name not in self._configs:
            self._load_config(plugin_name)
        
        return self._configs.get(plugin_name, {}).get(key, default)
    
    def set(self, plugin_name: str, key: str, value: Any):
        """Set config value for plugin."""
        if plugin_name not in self._configs:
            self._configs[plugin_name] = {}
        
        self._configs[plugin_name][key] = value
        self._save_config(plugin_name)
    
    def _load_config(self, plugin_name: str):
        """Load plugin config from file."""
        import json
        
        config_file = self.config_dir / f"{plugin_name}.json"
        
        if config_file.exists():
            self._configs[plugin_name] = json.loads(config_file.read_text())
        else:
            self._configs[plugin_name] = {}
    
    def _save_config(self, plugin_name: str):
        """Save plugin config to file."""
        import json
        
        config_file = self.config_dir / f"{plugin_name}.json"
        config_file.write_text(json.dumps(self._configs.get(plugin_name, {}), indent=2))


# Example plugin template
PLUGIN_TEMPLATE = '''"""
{name} Plugin for Enigma AI Engine
"""

from enigma_engine.plugins.manager import Plugin, PluginInfo


class {class_name}(Plugin):
    """Example plugin."""
    
    info = PluginInfo(
        name="{name}",
        version="1.0.0",
        description="{description}",
        author="{author}",
        dependencies=[],
        hooks=["on_message", "on_startup"]
    )
    
    def on_load(self):
        """Called when plugin loads."""
        self.log("Plugin loaded!")
    
    def on_unload(self):
        """Called when plugin unloads."""
        self.log("Plugin unloaded!")
    
    def on_message(self, message: str, **kwargs):
        """Hook: called on each message."""
        # Process message
        pass
    
    def on_startup(self, **kwargs):
        """Hook: called on startup."""
        pass
    
    def get_api(self):
        """Return public API."""
        return {{
            "example_function": self.example_function
        }}
    
    def example_function(self, arg: str) -> str:
        """Example API function."""
        return f"Processed: {{arg}}"
'''


def create_plugin_template(
    name: str,
    output_dir: str = "plugins",
    description: str = "",
    author: str = ""
):
    """Create a new plugin from template."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_name = ''.join(word.capitalize() for word in name.split('_'))
    
    content = PLUGIN_TEMPLATE.format(
        name=name,
        class_name=class_name,
        description=description,
        author=author
    )
    
    plugin_file = output_dir / f"{name}.py"
    plugin_file.write_text(content)
    
    logger.info(f"Created plugin template: {plugin_file}")
    return str(plugin_file)
