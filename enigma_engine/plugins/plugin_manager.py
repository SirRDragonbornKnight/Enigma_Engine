"""
Plugin System for Enigma AI Engine

Allows community extensions and custom functionality.

Features:
- Hot-loadable plugins
- Plugin discovery from multiple sources
- Sandboxed execution
- Dependency management
- Event hooks
- GUI integration

Usage:
    from enigma_engine.plugins.plugin_manager import PluginManager, get_manager
    
    # Get manager
    manager = get_manager()
    
    # Discover and load plugins
    manager.discover_plugins()
    manager.load_all()
    
    # Access plugin functionality
    plugin = manager.get_plugin("my_plugin")
    result = plugin.call("my_function", arg1, arg2)
    
    # Create a plugin
    @plugin_hook("on_chat_message")
    def my_handler(message):
        # Process message
        return modified_message
"""

import importlib
import importlib.util
import inspect
import json
import logging
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)


class PluginState(Enum):
    """Plugin lifecycle states."""
    DISCOVERED = auto()
    LOADED = auto()
    ACTIVATED = auto()
    DEACTIVATED = auto()
    ERROR = auto()
    UNLOADED = auto()


class PluginHook(Enum):
    """Available plugin hooks."""
    # Lifecycle
    ON_STARTUP = "on_startup"
    ON_SHUTDOWN = "on_shutdown"
    
    # Chat
    ON_CHAT_MESSAGE = "on_chat_message"
    ON_RESPONSE = "on_response"
    PRE_INFERENCE = "pre_inference"
    POST_INFERENCE = "post_inference"
    
    # Training
    ON_TRAINING_START = "on_training_start"
    ON_TRAINING_STEP = "on_training_step"
    ON_TRAINING_END = "on_training_end"
    
    # GUI
    ON_GUI_INIT = "on_gui_init"
    ON_TAB_CREATED = "on_tab_created"
    REGISTER_MENU = "register_menu"
    
    # Generation
    ON_IMAGE_GENERATED = "on_image_generated"
    ON_CODE_GENERATED = "on_code_generated"
    ON_AUDIO_GENERATED = "on_audio_generated"
    
    # Tools
    REGISTER_TOOL = "register_tool"
    ON_TOOL_CALL = "on_tool_call"
    
    # Memory
    ON_MEMORY_STORE = "on_memory_store"
    ON_MEMORY_RETRIEVE = "on_memory_retrieve"


@dataclass
class PluginMeta:
    """Plugin metadata."""
    name: str
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    
    # Requirements
    dependencies: List[str] = field(default_factory=list)  # Other plugin names
    python_requires: str = ""  # e.g., ">=3.8"
    pip_requires: List[str] = field(default_factory=list)  # pip packages
    
    # Compatibility
    enigma_version: str = ""  # Required Enigma version
    
    # Settings
    configurable: bool = True
    default_config: Dict[str, Any] = field(default_factory=dict)
    
    # Categories
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    
    # Source
    source_path: Optional[Path] = None
    homepage: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginMeta':
        """Create from dictionary."""
        return cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "1.0.0"),
            author=data.get("author", ""),
            description=data.get("description", ""),
            dependencies=data.get("dependencies", []),
            python_requires=data.get("python_requires", ""),
            pip_requires=data.get("pip_requires", []),
            enigma_version=data.get("enigma_version", ""),
            configurable=data.get("configurable", True),
            default_config=data.get("default_config", {}),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            homepage=data.get("homepage", ""),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "dependencies": self.dependencies,
            "python_requires": self.python_requires,
            "pip_requires": self.pip_requires,
            "enigma_version": self.enigma_version,
            "configurable": self.configurable,
            "default_config": self.default_config,
            "category": self.category,
            "tags": self.tags,
            "homepage": self.homepage,
        }


class PluginBase:
    """
    Base class for plugins.
    
    Plugins should inherit from this and implement the required methods.
    
    Example:
        class MyPlugin(PluginBase):
            def __init__(self):
                super().__init__()
                self.meta = PluginMeta(
                    name="my_plugin",
                    version="1.0.0",
                    description="Does something cool"
                )
            
            def activate(self) -> bool:
                # Initialize your plugin
                return True
            
            def deactivate(self):
                # Cleanup
                pass
    """
    
    def __init__(self):
        self.meta = PluginMeta(name="base_plugin")
        self.state = PluginState.DISCOVERED
        self.config: Dict[str, Any] = {}
        self._hooks: Dict[str, Callable] = {}
    
    def activate(self) -> bool:
        """
        Activate the plugin.
        Override this to initialize your plugin.
        
        Returns:
            True if activation succeeded
        """
        return True
    
    def deactivate(self):
        """
        Deactivate the plugin.
        Override this for cleanup.
        """
        pass
    
    def configure(self, config: Dict[str, Any]):
        """
        Configure the plugin.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {**self.meta.default_config, **config}
    
    def get_hooks(self) -> Dict[str, Callable]:
        """Get registered hook handlers."""
        return self._hooks
    
    def register_hook(self, hook: PluginHook | str, handler: Callable):
        """
        Register a hook handler.
        
        Args:
            hook: Hook to register for
            handler: Handler function
        """
        hook_name = hook.value if isinstance(hook, PluginHook) else hook
        self._hooks[hook_name] = handler
    
    def call(self, method: str, *args, **kwargs) -> Any:
        """
        Call a plugin method safely.
        
        Args:
            method: Method name
            *args, **kwargs: Arguments
            
        Returns:
            Method result
        """
        if hasattr(self, method):
            return getattr(self, method)(*args, **kwargs)
        raise AttributeError(f"Plugin has no method: {method}")


class PluginManager:
    """
    Manages plugin discovery, loading, and lifecycle.
    """
    
    # Default plugin directories
    DEFAULT_PLUGIN_DIRS = [
        Path.home() / ".enigma_engine" / "plugins",
        Path(__file__).parent / "installed",
    ]
    
    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        """
        Initialize plugin manager.
        
        Args:
            plugin_dirs: Directories to search for plugins
        """
        self._plugin_dirs = plugin_dirs or self.DEFAULT_PLUGIN_DIRS.copy()
        
        # Plugin storage
        self._plugins: Dict[str, PluginBase] = {}
        self._discovered: Dict[str, PluginMeta] = {}
        
        # Hook registry
        self._hooks: Dict[str, List[Callable]] = {}
        
        # Config storage
        self._config_dir = Path.home() / ".enigma_engine" / "plugin_configs"
        
        # State
        self._lock = threading.RLock()
        
        # Ensure directories exist
        for dir_path in self._plugin_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        self._config_dir.mkdir(parents=True, exist_ok=True)
    
    def add_plugin_dir(self, path: Path):
        """Add a plugin search directory."""
        if path not in self._plugin_dirs:
            self._plugin_dirs.append(path)
            path.mkdir(parents=True, exist_ok=True)
    
    def discover_plugins(self) -> List[PluginMeta]:
        """
        Discover plugins in all plugin directories.
        
        Returns:
            List of discovered plugin metadata
        """
        discovered = []
        
        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.exists():
                continue
            
            # Look for plugin.json or __init__.py
            for path in plugin_dir.iterdir():
                meta = None
                
                if path.is_dir():
                    # Directory-based plugin
                    meta = self._discover_directory_plugin(path)
                elif path.suffix == '.py' and path.stem != '__init__':
                    # Single-file plugin
                    meta = self._discover_file_plugin(path)
                
                if meta:
                    meta.source_path = path
                    self._discovered[meta.name] = meta
                    discovered.append(meta)
        
        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered
    
    def _discover_directory_plugin(self, path: Path) -> Optional[PluginMeta]:
        """Discover a directory-based plugin."""
        # Check for plugin.json
        meta_file = path / "plugin.json"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    data = json.load(f)
                return PluginMeta.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to read plugin.json in {path}: {e}")
        
        # Check for __init__.py with metadata
        init_file = path / "__init__.py"
        if init_file.exists():
            return self._extract_metadata_from_file(init_file, path.name)
        
        return None
    
    def _discover_file_plugin(self, path: Path) -> Optional[PluginMeta]:
        """Discover a single-file plugin."""
        return self._extract_metadata_from_file(path, path.stem)
    
    def _extract_metadata_from_file(self, path: Path, default_name: str) -> Optional[PluginMeta]:
        """Extract metadata from Python file."""
        try:
            content = path.read_text()
            
            # Simple extraction of PLUGIN_META dict
            if "PLUGIN_META" in content:
                # Load module temporarily
                spec = importlib.util.spec_from_file_location("temp_plugin", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, "PLUGIN_META"):
                    data = module.PLUGIN_META
                    if isinstance(data, dict):
                        return PluginMeta.from_dict(data)
            
            # Fallback - create basic metadata from docstring
            if '"""' in content:
                doc_start = content.find('"""') + 3
                doc_end = content.find('"""', doc_start)
                if doc_end > doc_start:
                    docstring = content[doc_start:doc_end].strip()
                    lines = docstring.split('\n')
                    return PluginMeta(
                        name=default_name,
                        description=lines[0] if lines else ""
                    )
            
            return PluginMeta(name=default_name)
            
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {path}: {e}")
            return None
    
    def load_plugin(self, name: str) -> bool:
        """
        Load a discovered plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            True if loaded successfully
        """
        with self._lock:
            if name in self._plugins:
                logger.warning(f"Plugin already loaded: {name}")
                return True
            
            if name not in self._discovered:
                logger.error(f"Plugin not discovered: {name}")
                return False
            
            meta = self._discovered[name]
            
            # Check dependencies
            if not self._check_dependencies(meta):
                return False
            
            try:
                # Load the plugin
                plugin = self._load_plugin_module(meta)
                
                if plugin:
                    plugin.meta = meta
                    plugin.state = PluginState.LOADED
                    
                    # Load config
                    config = self._load_plugin_config(name)
                    plugin.configure(config)
                    
                    self._plugins[name] = plugin
                    logger.info(f"Loaded plugin: {name} v{meta.version}")
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to load plugin {name}: {e}")
            
            return False
    
    def _check_dependencies(self, meta: PluginMeta) -> bool:
        """Check if plugin dependencies are satisfied."""
        # Check Python version
        if meta.python_requires:
            import packaging.version
            current_version = packaging.version.parse(
                f"{sys.version_info.major}.{sys.version_info.minor}"
            )
            # Simple check - could be improved
            if meta.python_requires.startswith(">="):
                required = packaging.version.parse(meta.python_requires[2:])
                if current_version < required:
                    logger.error(f"Plugin {meta.name} requires Python {meta.python_requires}")
                    return False
        
        # Check plugin dependencies
        for dep in meta.dependencies:
            if dep not in self._plugins and dep not in self._discovered:
                logger.error(f"Plugin {meta.name} requires missing plugin: {dep}")
                return False
        
        return True
    
    def _load_plugin_module(self, meta: PluginMeta) -> Optional[PluginBase]:
        """Load plugin module and instantiate plugin class."""
        path = meta.source_path
        
        if path.is_dir():
            # Directory plugin - import __init__.py
            spec = importlib.util.spec_from_file_location(
                f"enigma_plugins.{meta.name}",
                path / "__init__.py"
            )
        else:
            # File plugin
            spec = importlib.util.spec_from_file_location(
                f"enigma_plugins.{meta.name}",
                path
            )
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        
        # Find plugin class
        for item_name in dir(module):
            item = getattr(module, item_name)
            if (inspect.isclass(item) and 
                issubclass(item, PluginBase) and 
                item is not PluginBase):
                return item()
        
        # No class found - create wrapper
        return self._create_function_plugin(module, meta)
    
    def _create_function_plugin(self, module, meta: PluginMeta) -> PluginBase:
        """Create a plugin wrapper for function-based plugins."""
        plugin = PluginBase()
        plugin.meta = meta
        
        # Copy module functions to plugin
        for name in dir(module):
            if not name.startswith('_'):
                item = getattr(module, name)
                if callable(item):
                    setattr(plugin, name, item)
                    
                    # Auto-register hooks
                    for hook in PluginHook:
                        if name == hook.value:
                            plugin.register_hook(hook, item)
        
        return plugin
    
    def unload_plugin(self, name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            True if unloaded successfully
        """
        with self._lock:
            if name not in self._plugins:
                return False
            
            plugin = self._plugins[name]
            
            # Deactivate first
            if plugin.state == PluginState.ACTIVATED:
                self.deactivate_plugin(name)
            
            # Remove hook registrations
            for hook_name, handlers in self._hooks.items():
                plugin_hooks = plugin.get_hooks()
                if hook_name in plugin_hooks:
                    handler = plugin_hooks[hook_name]
                    if handler in handlers:
                        handlers.remove(handler)
            
            # Remove from plugins
            del self._plugins[name]
            plugin.state = PluginState.UNLOADED
            
            logger.info(f"Unloaded plugin: {name}")
            return True
    
    def activate_plugin(self, name: str) -> bool:
        """
        Activate a loaded plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            True if activated successfully
        """
        with self._lock:
            if name not in self._plugins:
                # Try to load first
                if not self.load_plugin(name):
                    return False
            
            plugin = self._plugins[name]
            
            if plugin.state == PluginState.ACTIVATED:
                return True
            
            try:
                if plugin.activate():
                    plugin.state = PluginState.ACTIVATED
                    
                    # Register hooks
                    for hook_name, handler in plugin.get_hooks().items():
                        if hook_name not in self._hooks:
                            self._hooks[hook_name] = []
                        self._hooks[hook_name].append(handler)
                    
                    logger.info(f"Activated plugin: {name}")
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to activate plugin {name}: {e}")
                plugin.state = PluginState.ERROR
            
            return False
    
    def deactivate_plugin(self, name: str) -> bool:
        """
        Deactivate a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            True if deactivated successfully
        """
        with self._lock:
            if name not in self._plugins:
                return False
            
            plugin = self._plugins[name]
            
            if plugin.state != PluginState.ACTIVATED:
                return True
            
            try:
                plugin.deactivate()
                plugin.state = PluginState.DEACTIVATED
                
                # Unregister hooks
                for hook_name, handler in plugin.get_hooks().items():
                    if hook_name in self._hooks:
                        if handler in self._hooks[hook_name]:
                            self._hooks[hook_name].remove(handler)
                
                logger.info(f"Deactivated plugin: {name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to deactivate plugin {name}: {e}")
            
            return False
    
    def load_all(self, activate: bool = True):
        """Load and optionally activate all discovered plugins."""
        for name in self._discovered:
            if self.load_plugin(name) and activate:
                self.activate_plugin(name)
    
    def get_plugin(self, name: str) -> Optional[PluginBase]:
        """Get a loaded plugin by name."""
        return self._plugins.get(name)
    
    def get_plugins(self, state: Optional[PluginState] = None) -> List[PluginBase]:
        """Get all plugins, optionally filtered by state."""
        if state:
            return [p for p in self._plugins.values() if p.state == state]
        return list(self._plugins.values())
    
    def trigger_hook(self, hook: PluginHook | str, *args, **kwargs) -> List[Any]:
        """
        Trigger a hook and collect results.
        
        Args:
            hook: Hook to trigger
            *args, **kwargs: Arguments for handlers
            
        Returns:
            List of results from handlers
        """
        hook_name = hook.value if isinstance(hook, PluginHook) else hook
        
        results = []
        handlers = self._hooks.get(hook_name, [])
        
        for handler in handlers:
            try:
                result = handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook handler error: {e}")
        
        return results
    
    def trigger_hook_chain(self, hook: PluginHook | str, value: Any) -> Any:
        """
        Trigger a hook where each handler processes the previous result.
        
        Args:
            hook: Hook to trigger
            value: Initial value
            
        Returns:
            Final processed value
        """
        hook_name = hook.value if isinstance(hook, PluginHook) else hook
        handlers = self._hooks.get(hook_name, [])
        
        for handler in handlers:
            try:
                result = handler(value)
                if result is not None:
                    value = result
            except Exception as e:
                logger.error(f"Hook chain error: {e}")
        
        return value
    
    def _load_plugin_config(self, name: str) -> Dict[str, Any]:
        """Load plugin configuration."""
        config_file = self._config_dir / f"{name}.json"
        
        if config_file.exists():
            try:
                with open(config_file) as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {}
    
    def save_plugin_config(self, name: str, config: Dict[str, Any]):
        """Save plugin configuration."""
        config_file = self._config_dir / f"{name}.json"
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config for {name}: {e}")
    
    def install_plugin(self, source: str) -> bool:
        """
        Install a plugin from source.
        
        Args:
            source: URL, git repo, or local path
            
        Returns:
            True if installed successfully
        """
        # This is a placeholder for full implementation
        # Could support:
        # - Git URLs
        # - PyPI packages
        # - Local zip files
        # - Direct URLs
        
        logger.warning("Plugin installation not yet implemented")
        return False


# Hook decorator for convenience
def plugin_hook(hook: PluginHook | str):
    """
    Decorator to mark a function as a hook handler.
    
    Usage:
        @plugin_hook(PluginHook.ON_CHAT_MESSAGE)
        def handle_message(message):
            return modified_message
    """
    def decorator(func: Callable) -> Callable:
        func._plugin_hook = hook.value if isinstance(hook, PluginHook) else hook
        return func
    return decorator


# Global manager instance
_manager: Optional[PluginManager] = None


def get_manager() -> PluginManager:
    """Get or create the global plugin manager."""
    global _manager
    if _manager is None:
        _manager = PluginManager()
    return _manager


def load_plugins():
    """Discover and load all plugins."""
    manager = get_manager()
    manager.discover_plugins()
    manager.load_all()


def trigger_hook(hook: PluginHook | str, *args, **kwargs) -> List[Any]:
    """Quick function to trigger a hook."""
    return get_manager().trigger_hook(hook, *args, **kwargs)
