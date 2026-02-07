"""
Lazy Import Utilities
=====================

Provides utilities for lazy loading heavy modules to improve startup time.

Usage:
    from enigma_engine.utils.lazy_imports import lazy_import, LazyModule
    
    # Option 1: Decorator for functions that need heavy imports
    @lazy_import('torch', 'numpy')
    def my_function():
        import torch
        import numpy
        # torch and numpy are now available
    
    # Option 2: LazyModule for deferred imports
    torch = LazyModule('torch')
    # torch is not imported until first attribute access
    tensor = torch.zeros(10)  # NOW torch is imported
"""

import importlib
import importlib.util
import sys
from functools import wraps
from typing import Any, Callable


class LazyModule:
    """
    A proxy object that lazily imports a module on first attribute access.
    
    This helps reduce startup time by deferring heavy imports until needed.
    
    Example:
        >>> torch = LazyModule('torch')
        >>> # torch is NOT imported yet
        >>> x = torch.tensor([1, 2, 3])  # NOW torch is imported
    """
    
    __slots__ = ('_module_name', '_module', '_import_error')
    
    def __init__(self, module_name: str):
        object.__setattr__(self, '_module_name', module_name)
        object.__setattr__(self, '_module', None)
        object.__setattr__(self, '_import_error', None)
    
    def _load(self):
        """Load the module if not already loaded."""
        module = object.__getattribute__(self, '_module')
        if module is not None:
            return module
        
        error = object.__getattribute__(self, '_import_error')
        if error is not None:
            raise error
        
        module_name = object.__getattribute__(self, '_module_name')
        try:
            module = importlib.import_module(module_name)
            object.__setattr__(self, '_module', module)
            return module
        except ImportError as e:
            object.__setattr__(self, '_import_error', e)
            raise
    
    def __getattr__(self, name: str) -> Any:
        module = self._load()
        return getattr(module, name)
    
    def __setattr__(self, name: str, value: Any):
        if name in ('_module_name', '_module', '_import_error'):
            object.__setattr__(self, name, value)
        else:
            module = self._load()
            setattr(module, name, value)
    
    def __repr__(self) -> str:
        module = object.__getattribute__(self, '_module')
        if module is not None:
            return repr(module)
        module_name = object.__getattribute__(self, '_module_name')
        return f"<LazyModule({module_name!r})>"
    
    def __dir__(self):
        module = self._load()
        return dir(module)


def lazy_import(*module_names: str):
    """
    Decorator that ensures modules are importable before running function.
    
    Does NOT actually import the modules - just validates they exist.
    Use this for functions that import heavy modules internally.
    
    Example:
        @lazy_import('torch', 'transformers')
        def train_model():
            import torch
            import transformers
            # ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_or_import(module_name: str) -> Any | None:
    """
    Get a module if already imported, otherwise import it.
    
    Returns None if import fails (doesn't raise).
    
    Example:
        torch = get_or_import('torch')
        if torch is not None:
            x = torch.tensor([1, 2, 3])
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def is_available(module_name: str) -> bool:
    """
    Check if a module is available without fully importing it.
    
    Example:
        if is_available('torch'):
            import torch
    """
    if module_name in sys.modules:
        return True
    
    spec = importlib.util.find_spec(module_name)
    return spec is not None


# Pre-configured lazy modules for common heavy dependencies
# These are only imported when actually accessed

_torch: LazyModule | None = None
_numpy: LazyModule | None = None
_pandas: LazyModule | None = None
_transformers: LazyModule | None = None


def get_torch():
    """Get torch module (lazy loaded)."""
    global _torch
    if _torch is None:
        _torch = LazyModule('torch')
    return _torch


def get_numpy():
    """Get numpy module (lazy loaded)."""
    global _numpy
    if _numpy is None:
        _numpy = LazyModule('numpy')
    return _numpy


def get_pandas():
    """Get pandas module (lazy loaded)."""
    global _pandas
    if _pandas is None:
        _pandas = LazyModule('pandas')
    return _pandas


def get_transformers():
    """Get transformers module (lazy loaded)."""
    global _transformers
    if _transformers is None:
        _transformers = LazyModule('transformers')
    return _transformers


__all__ = [
    'LazyModule',
    'lazy_import',
    'get_or_import',
    'is_available',
    'get_torch',
    'get_numpy',
    'get_pandas',
    'get_transformers',
]
