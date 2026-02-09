"""
Lazy Import Utilities
=====================

Provides utilities for deferring expensive imports until they're actually needed.
This significantly improves startup time when not all modules are used.

Usage:
    # In __init__.py files:
    from .lazy_import import lazy_module, LazyLoader
    
    # Define lazy imports
    _LAZY_IMPORTS = {
        'create_model': ('.model', 'create_model'),
        'EnigmaEngine': ('.inference', 'EnigmaEngine'),
    }
    
    def __getattr__(name):
        if name in _LAZY_IMPORTS:
            module_path, attr_name = _LAZY_IMPORTS[name]
            module = importlib.import_module(module_path, __package__)
            return getattr(module, attr_name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
"""

import importlib
import sys
from typing import Any, Callable, Dict, Optional, Tuple


class LazyModule:
    """
    A lazy module wrapper that defers importing until first attribute access.
    
    Usage:
        torch = LazyModule('torch')
        # torch is not imported until you actually use it
        x = torch.tensor([1, 2, 3])  # Now torch is imported
    """
    
    def __init__(self, module_name: str, package: Optional[str] = None):
        self._module_name = module_name
        self._package = package
        self._module: Optional[Any] = None
    
    def _load(self) -> Any:
        """Load the actual module."""
        if self._module is None:
            self._module = importlib.import_module(self._module_name, self._package)
        return self._module
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)
    
    def __dir__(self):
        return dir(self._load())
    
    def __repr__(self):
        if self._module is None:
            return f"<LazyModule '{self._module_name}' (not loaded)>"
        return f"<LazyModule '{self._module_name}' (loaded)>"


class LazyLoader:
    """
    Module-level lazy loader using __getattr__ pattern.
    
    Usage in __init__.py:
        _loader = LazyLoader(__name__)
        _loader.register('EnigmaEngine', '.inference', 'EnigmaEngine')
        _loader.register('create_model', '.model', 'create_model')
        
        def __getattr__(name):
            return _loader.load(name)
    """
    
    def __init__(self, package_name: str):
        self._package = package_name
        self._registry: Dict[str, Tuple[str, str]] = {}
        self._cache: Dict[str, Any] = {}
    
    def register(self, name: str, module_path: str, attr_name: Optional[str] = None):
        """
        Register a lazy import.
        
        Args:
            name: The name to expose in the module
            module_path: The module path (relative or absolute)
            attr_name: The attribute to import from the module (defaults to name)
        """
        self._registry[name] = (module_path, attr_name or name)
    
    def register_many(self, imports: Dict[str, Tuple[str, str]]):
        """Register multiple lazy imports at once."""
        self._registry.update(imports)
    
    def load(self, name: str) -> Any:
        """Load and return the requested attribute."""
        if name in self._cache:
            return self._cache[name]
        
        if name not in self._registry:
            raise AttributeError(f"module has no attribute {name!r}")
        
        module_path, attr_name = self._registry[name]
        module = importlib.import_module(module_path, self._package)
        value = getattr(module, attr_name)
        self._cache[name] = value
        return value
    
    def is_registered(self, name: str) -> bool:
        """Check if a name is registered for lazy loading."""
        return name in self._registry
    
    def get_all_names(self) -> list:
        """Get all registered names."""
        return list(self._registry.keys())


def lazy_module(module_name: str, package: Optional[str] = None) -> LazyModule:
    """
    Create a lazy module reference.
    
    Args:
        module_name: The module to lazily import
        package: The package context for relative imports
    
    Returns:
        A LazyModule instance that loads on first access
    
    Example:
        torch = lazy_module('torch')
        numpy = lazy_module('numpy')
        # Neither are imported yet...
        
        x = torch.tensor([1, 2, 3])  # Now torch imports
    """
    return LazyModule(module_name, package)


def lazy_import(
    module_name: str, 
    attr_name: Optional[str] = None,
    package: Optional[str] = None
) -> Callable[[], Any]:
    """
    Create a callable that imports on first call.
    
    Args:
        module_name: The module to import
        attr_name: Optional attribute to get from the module
        package: The package context for relative imports
    
    Returns:
        A callable that returns the imported module/attribute
    
    Example:
        get_torch = lazy_import('torch')
        # torch is not imported yet...
        
        torch = get_torch()  # Now it's imported
    """
    _cache = {}
    
    def loader():
        if 'value' not in _cache:
            module = importlib.import_module(module_name, package)
            if attr_name:
                _cache['value'] = getattr(module, attr_name)
            else:
                _cache['value'] = module
        return _cache['value']
    
    return loader


# Pre-defined lazy modules for common heavy imports
def get_torch():
    """Get torch module (lazy import)."""
    return importlib.import_module('torch')


def get_numpy():
    """Get numpy module (lazy import)."""
    return importlib.import_module('numpy')


def get_transformers():
    """Get transformers module (lazy import)."""
    return importlib.import_module('transformers')


# Type hints for lazy modules
TORCH_LAZY = LazyModule('torch')
NUMPY_LAZY = LazyModule('numpy')
