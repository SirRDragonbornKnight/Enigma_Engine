"""
Plugin System for enigma_engine

Allows extending enigma_engine with custom:
- Inference backends
- Training methods
- Model architectures
- Tools/capabilities
- Pre/post processing hooks

Plugins are discovered from:
1. enigma_engine/plugins/ directory
2. User plugins directory (~/.forge/plugins/)
3. Installed packages with 'enigma_engine.plugins' entry point
"""

import importlib
import importlib.util
import inspect
import json
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    homepage: str = ""
    dependencies: list[str] = field(default_factory=list)
    forge_version: str = ">=0.1.0"  # Minimum enigma_engine version
    enabled: bool = True
    priority: int = 100  # Lower = higher priority


class PluginType:
    """Plugin type identifiers."""
    BACKEND = "backend"
    TRAINER = "trainer"
    MODEL = "model"
    TOOL = "tool"
    HOOK = "hook"
    TOKENIZER = "tokenizer"
    QUANTIZER = "quantizer"
    CUSTOM = "custom"


class ForgePlugin(ABC):
    """
    Base class for all enigma_engine plugins.
    
    To create a plugin:
    1. Subclass ForgePlugin
    2. Set metadata
    3. Implement required methods
    4. Place in plugins directory or install as package
    
    Example:
        class MyPlugin(ForgePlugin):
            metadata = PluginMetadata(
                name="my-plugin",
                version="1.0.0",
                description="My awesome plugin"
            )
            
            def initialize(self):
                print("Plugin loaded!")
            
            def cleanup(self):
                print("Plugin unloaded!")
    """
    
    metadata: PluginMetadata = PluginMetadata(
        name="base-plugin",
        version="0.0.0",
        description="Base plugin class"
    )
    
    plugin_type: str = PluginType.CUSTOM
    
    def __init__(self):
        self._initialized = False
        self._context: dict[str, Any] = {}
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the plugin. Called when plugin is loaded."""
    
    def cleanup(self) -> None:
        """Cleanup the plugin. Called when plugin is unloaded."""
    
    def configure(self, config: dict[str, Any]) -> None:
        """Configure the plugin with runtime settings."""
        self._context.update(config)
    
    def get_capabilities(self) -> list[str]:
        """Return list of capabilities this plugin provides."""
        return []
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized


class BackendPlugin(ForgePlugin):
    """Plugin for custom inference backends (CUDA, Metal, ROCm, etc.)."""
    
    plugin_type = PluginType.BACKEND
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
    
    @abstractmethod
    def get_device(self) -> Any:
        """Get the device for this backend."""
    
    @abstractmethod
    def prepare_model(self, model: Any) -> Any:
        """Prepare a model for inference on this backend."""
    
    @abstractmethod
    def generate(self, model: Any, inputs: Any, **kwargs) -> Any:
        """Run generation on this backend."""


class TrainerPlugin(ForgePlugin):
    """Plugin for custom training methods (SFT, DPO, RLHF, etc.)."""
    
    plugin_type = PluginType.TRAINER
    
    @abstractmethod
    def train(
        self,
        model: Any,
        dataset: Any,
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Train a model.
        
        Returns:
            Training results/metrics
        """
    
    def validate(self, model: Any, dataset: Any) -> dict[str, Any]:
        """Validate model on dataset."""
        return {}


class ModelPlugin(ForgePlugin):
    """Plugin for custom model architectures."""
    
    plugin_type = PluginType.MODEL
    
    @abstractmethod
    def create_model(self, config: dict[str, Any]) -> Any:
        """Create a model instance."""
    
    @abstractmethod
    def load_weights(self, model: Any, path: str) -> None:
        """Load weights into model."""
    
    def save_weights(self, model: Any, path: str) -> None:
        """Save model weights."""


class ToolPlugin(ForgePlugin):
    """Plugin for custom tools/capabilities."""
    
    plugin_type = PluginType.TOOL
    
    @abstractmethod
    def get_tool_definition(self) -> dict[str, Any]:
        """
        Return tool definition in OpenAI function format.
        
        Example:
            return {
                "name": "my_tool",
                "description": "Does something useful",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"}
                    }
                }
            }
        """
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""


class HookPlugin(ForgePlugin):
    """Plugin for pre/post processing hooks."""
    
    plugin_type = PluginType.HOOK
    
    hook_points: list[str] = []  # e.g., ["pre_inference", "post_inference"]
    
    def pre_inference(self, inputs: Any) -> Any:
        """Called before inference."""
        return inputs
    
    def post_inference(self, outputs: Any) -> Any:
        """Called after inference."""
        return outputs
    
    def pre_training(self, batch: Any) -> Any:
        """Called before each training step."""
        return batch
    
    def post_training(self, loss: Any) -> Any:
        """Called after each training step."""
        return loss


class PluginManager:
    """
    Manages plugin discovery, loading, and lifecycle.
    
    Usage:
        manager = PluginManager()
        manager.discover_plugins()
        manager.load_plugin("my-plugin")
        
        # Get plugin
        plugin = manager.get_plugin("my-plugin")
        
        # Get all plugins of a type
        backends = manager.get_plugins_by_type(PluginType.BACKEND)
    """
    
    def __init__(self, plugin_dirs: Optional[list[Path]] = None):
        self._plugins: dict[str, ForgePlugin] = {}
        self._plugin_classes: dict[str, type[ForgePlugin]] = {}
        self._hooks: dict[str, list[HookPlugin]] = {}
        
        # Default plugin directories
        self._plugin_dirs = plugin_dirs or [
            Path(__file__).parent.parent / "plugins",  # enigma_engine/plugins/
            Path.home() / ".forge" / "plugins",  # User plugins
        ]
        
        # Ensure directories exist
        for d in self._plugin_dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def discover_plugins(self) -> list[str]:
        """
        Discover all available plugins.
        
        Returns:
            List of discovered plugin names
        """
        discovered = []
        
        # Discover from plugin directories
        for plugin_dir in self._plugin_dirs:
            discovered.extend(self._discover_from_directory(plugin_dir))
        
        # Discover from installed packages (entry points)
        discovered.extend(self._discover_from_entry_points())
        
        logger.info(f"Discovered {len(discovered)} plugins: {discovered}")
        return discovered
    
    def _discover_from_directory(self, directory: Path) -> list[str]:
        """Discover plugins from a directory."""
        discovered = []
        
        if not directory.exists():
            return discovered
        
        for item in directory.iterdir():
            if item.is_file() and item.suffix == '.py' and not item.name.startswith('_'):
                # Single-file plugin
                plugin_name = item.stem
                try:
                    self._load_plugin_module(item, plugin_name)
                    discovered.append(plugin_name)
                except Exception as e:
                    logger.warning(f"Failed to load plugin {plugin_name}: {e}")
            
            elif item.is_dir() and (item / '__init__.py').exists():
                # Package plugin
                plugin_name = item.name
                try:
                    self._load_plugin_module(item / '__init__.py', plugin_name)
                    discovered.append(plugin_name)
                except Exception as e:
                    logger.warning(f"Failed to load plugin {plugin_name}: {e}")
        
        return discovered
    
    def _load_plugin_module(self, path: Path, name: str) -> None:
        """Load a plugin module and register its classes."""
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load plugin from {path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        
        # Find ForgePlugin subclasses
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                inspect.isclass(attr) and
                issubclass(attr, ForgePlugin) and
                attr is not ForgePlugin and
                not attr.__name__.startswith('_')
            ):
                plugin_name = attr.metadata.name
                self._plugin_classes[plugin_name] = attr
                logger.debug(f"Registered plugin class: {plugin_name}")
    
    def _discover_from_entry_points(self) -> list[str]:
        """Discover plugins from installed packages."""
        discovered = []
        
        try:
            if sys.version_info >= (3, 10):
                from importlib.metadata import entry_points
                eps = entry_points(group='enigma_engine.plugins')
            else:
                from importlib.metadata import entry_points
                eps = entry_points().get('enigma_engine.plugins', [])
            
            for ep in eps:
                try:
                    plugin_class = ep.load()
                    if issubclass(plugin_class, ForgePlugin):
                        plugin_name = plugin_class.metadata.name
                        self._plugin_classes[plugin_name] = plugin_class
                        discovered.append(plugin_name)
                except Exception as e:
                    logger.warning(f"Failed to load entry point {ep.name}: {e}")
        
        except Exception as e:
            logger.warning(f"Failed to discover entry points: {e}")
        
        return discovered
    
    def load_plugin(
        self,
        name: str,
        config: Optional[dict[str, Any]] = None
    ) -> ForgePlugin:
        """
        Load and initialize a plugin.
        
        Args:
            name: Plugin name
            config: Optional configuration
        
        Returns:
            Initialized plugin instance
        """
        if name in self._plugins:
            return self._plugins[name]
        
        if name not in self._plugin_classes:
            raise ValueError(f"Plugin not found: {name}")
        
        plugin_class = self._plugin_classes[name]
        plugin = plugin_class()
        
        if config:
            plugin.configure(config)
        
        plugin.initialize()
        plugin._initialized = True
        
        self._plugins[name] = plugin
        
        # Register hooks if it's a hook plugin
        if isinstance(plugin, HookPlugin):
            for hook_point in plugin.hook_points:
                if hook_point not in self._hooks:
                    self._hooks[hook_point] = []
                self._hooks[hook_point].append(plugin)
        
        logger.info(f"Loaded plugin: {name} v{plugin.metadata.version}")
        return plugin
    
    def unload_plugin(self, name: str) -> None:
        """Unload a plugin."""
        if name not in self._plugins:
            return
        
        plugin = self._plugins[name]
        
        # Remove from hooks
        if isinstance(plugin, HookPlugin):
            for hook_point in plugin.hook_points:
                if hook_point in self._hooks:
                    self._hooks[hook_point] = [
                        p for p in self._hooks[hook_point] if p is not plugin
                    ]
        
        plugin.cleanup()
        plugin._initialized = False
        del self._plugins[name]
        
        logger.info(f"Unloaded plugin: {name}")
    
    def get_plugin(self, name: str) -> Optional[ForgePlugin]:
        """Get a loaded plugin by name."""
        return self._plugins.get(name)
    
    def get_plugins_by_type(self, plugin_type: str) -> list[ForgePlugin]:
        """Get all loaded plugins of a specific type."""
        return [
            p for p in self._plugins.values()
            if p.plugin_type == plugin_type
        ]
    
    def list_plugins(self) -> dict[str, PluginMetadata]:
        """List all discovered plugins and their metadata."""
        return {
            name: cls.metadata
            for name, cls in self._plugin_classes.items()
        }
    
    def list_loaded(self) -> list[str]:
        """List currently loaded plugins."""
        return list(self._plugins.keys())
    
    def run_hooks(self, hook_point: str, data: Any) -> Any:
        """Run all hooks for a hook point."""
        if hook_point not in self._hooks:
            return data
        
        # Sort by priority
        hooks = sorted(
            self._hooks[hook_point],
            key=lambda p: p.metadata.priority
        )
        
        for hook in hooks:
            hook_method = getattr(hook, hook_point, None)
            if hook_method and callable(hook_method):
                try:
                    data = hook_method(data)
                except Exception as e:
                    logger.error(f"Hook {hook.metadata.name}.{hook_point} failed: {e}")
        
        return data
    
    def save_state(self, path: Path) -> None:
        """Save plugin manager state."""
        state = {
            'loaded': list(self._plugins.keys()),
            'configs': {
                name: plugin._context
                for name, plugin in self._plugins.items()
            }
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: Path) -> None:
        """Restore plugin manager state."""
        if not path.exists():
            return
        
        with open(path) as f:
            state = json.load(f)
        
        for name in state.get('loaded', []):
            config = state.get('configs', {}).get(name)
            try:
                self.load_plugin(name, config)
            except Exception as e:
                logger.error(f"Failed to restore plugin {name}: {e}")


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def register_plugin(plugin_class: type[ForgePlugin]) -> type[ForgePlugin]:
    """
    Decorator to register a plugin class.
    
    Example:
        @register_plugin
        class MyPlugin(ForgePlugin):
            metadata = PluginMetadata(name="my-plugin", version="1.0.0")
            ...
    """
    manager = get_plugin_manager()
    manager._plugin_classes[plugin_class.metadata.name] = plugin_class
    return plugin_class
