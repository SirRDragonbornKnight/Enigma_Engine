"""
================================================================================
HOOK SYSTEM - Pre/Post Hooks for Extensibility
================================================================================

A hook system that allows intercepting and extending function/method behavior
without modifying the original code.

Features:
- Register pre/post hooks on any function
- Filter hooks by arguments
- Priority-based hook execution
- Context passing between hooks
- Async hook support
- Hook validation

FILE: forge_ai/utils/hooks.py
TYPE: System Utility
MAIN CLASSES: HookManager, Hook

USAGE:
    from forge_ai.utils.hooks import get_hook_manager, before, after
    
    # Using decorators
    @before("model.generate")
    def log_generation_start(args, kwargs, context):
        print(f"Starting generation: {kwargs.get('prompt', '')[:50]}...")
        context['start_time'] = time.time()
    
    @after("model.generate")
    def log_generation_end(result, args, kwargs, context):
        duration = time.time() - context['start_time']
        print(f"Generation completed in {duration:.2f}s")
        return result  # Return modified result or original
    
    # Or manually
    manager = get_hook_manager()
    manager.register("model.generate", before=my_pre_hook, after=my_post_hook)
    
    # Execute with hooks
    result = manager.call("model.generate", my_function, prompt="Hello")
"""

from __future__ import annotations

import functools
import inspect
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class Hook:
    """A registered hook."""
    
    callback: Callable
    hook_point: str
    phase: str  # "before" or "after"
    priority: int = 50  # Higher = runs first for before, last for after
    name: str = ""
    enabled: bool = True
    
    def __post_init__(self):
        if not self.name:
            self.name = self.callback.__name__
    
    def __lt__(self, other):
        # For "before": higher priority runs first
        # For "after": higher priority runs last (reverse order)
        if self.phase == "before":
            return self.priority > other.priority
        else:
            return self.priority < other.priority


@dataclass
class HookContext:
    """
    Context passed through hooks.
    
    Allows hooks to share data with each other and with the main function.
    """
    
    hook_point: str
    data: Dict[str, Any] = field(default_factory=dict)
    skip_remaining: bool = False  # If True, skip remaining hooks
    skip_function: bool = False   # If True, skip the main function (before hooks only)
    override_result: Any = None   # If set, use this instead of function result
    has_override: bool = False
    start_time: float = field(default_factory=time.time)
    
    def __getitem__(self, key: str) -> Any:
        return self.data[key]
    
    def __setitem__(self, key: str, value: Any):
        self.data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


class HookManager:
    """
    Manages hooks for extensible function behavior.
    
    Allows registering pre and post hooks that run before/after
    functions without modifying the original code.
    """
    
    _instance: Optional['HookManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the hook manager."""
        if HookManager._initialized:
            return
        HookManager._initialized = True
        
        # Hooks by hook point
        self._before_hooks: Dict[str, List[Hook]] = {}
        self._after_hooks: Dict[str, List[Hook]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._call_counts: Dict[str, int] = {}
        self._total_time: Dict[str, float] = {}
        
        logger.debug("HookManager initialized")
    
    def register(
        self,
        hook_point: str,
        before: Callable = None,
        after: Callable = None,
        priority: int = 50,
        name: str = None
    ) -> List[Callable[[], None]]:
        """
        Register hooks for a hook point.
        
        Args:
            hook_point: Name of the hook point (e.g., "model.generate")
            before: Function to call before the main function
            after: Function to call after the main function
            priority: Higher priority runs first for before, last for after
            name: Optional name for the hook
            
        Returns:
            List of unregister functions
        """
        unregisters = []
        
        if before:
            hook = Hook(
                callback=before,
                hook_point=hook_point,
                phase="before",
                priority=priority,
                name=name or before.__name__
            )
            unregisters.append(self._add_hook(hook, self._before_hooks))
        
        if after:
            hook = Hook(
                callback=after,
                hook_point=hook_point,
                phase="after",
                priority=priority,
                name=name or after.__name__
            )
            unregisters.append(self._add_hook(hook, self._after_hooks))
        
        return unregisters
    
    def _add_hook(
        self,
        hook: Hook,
        registry: Dict[str, List[Hook]]
    ) -> Callable[[], None]:
        """Add a hook to a registry and return unregister function."""
        with self._lock:
            if hook.hook_point not in registry:
                registry[hook.hook_point] = []
            
            registry[hook.hook_point].append(hook)
            registry[hook.hook_point].sort()
        
        logger.debug(f"Registered {hook.phase} hook '{hook.name}' for '{hook.hook_point}'")
        
        def unregister():
            with self._lock:
                try:
                    registry[hook.hook_point].remove(hook)
                except (KeyError, ValueError):
                    pass
        
        return unregister
    
    def before(
        self,
        hook_point: str,
        priority: int = 50,
        name: str = None
    ) -> Callable:
        """
        Decorator to register a before hook.
        
        The decorated function receives: (args, kwargs, context)
        
        Example:
            @hook_manager.before("model.generate")
            def my_hook(args, kwargs, context):
                context['start_time'] = time.time()
        """
        def decorator(func: Callable) -> Callable:
            self.register(hook_point, before=func, priority=priority, name=name)
            return func
        return decorator
    
    def after(
        self,
        hook_point: str,
        priority: int = 50,
        name: str = None
    ) -> Callable:
        """
        Decorator to register an after hook.
        
        The decorated function receives: (result, args, kwargs, context)
        Return value replaces the result if not None.
        
        Example:
            @hook_manager.after("model.generate")
            def my_hook(result, args, kwargs, context):
                print(f"Took {time.time() - context['start_time']:.2f}s")
                return result
        """
        def decorator(func: Callable) -> Callable:
            self.register(hook_point, after=func, priority=priority, name=name)
            return func
        return decorator
    
    def hookable(self, hook_point: str = None) -> Callable:
        """
        Decorator to make a function hookable.
        
        Example:
            @hook_manager.hookable("model.generate")
            def generate(prompt, max_tokens=100):
                return model.generate(prompt, max_tokens)
        """
        def decorator(func: Callable) -> Callable:
            point = hook_point or f"{func.__module__}.{func.__qualname__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.call(point, func, *args, **kwargs)
            
            # Store hook point for introspection
            wrapper._hook_point = point
            return wrapper
        
        return decorator
    
    def call(
        self,
        hook_point: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Call a function with its registered hooks.
        
        Args:
            hook_point: The hook point name
            func: The function to call
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            The function result (possibly modified by after hooks)
        """
        context = HookContext(hook_point=hook_point)
        
        # Track statistics
        with self._lock:
            self._call_counts[hook_point] = self._call_counts.get(hook_point, 0) + 1
        
        start_time = time.time()
        
        try:
            # Run before hooks
            self._run_before_hooks(hook_point, args, kwargs, context)
            
            # Check if we should skip the function
            if context.skip_function:
                if context.has_override:
                    result = context.override_result
                else:
                    result = None
            else:
                # Call the main function
                result = func(*args, **kwargs)
            
            # Run after hooks
            result = self._run_after_hooks(hook_point, result, args, kwargs, context)
            
            return result
            
        finally:
            # Track timing
            elapsed = time.time() - start_time
            with self._lock:
                self._total_time[hook_point] = self._total_time.get(hook_point, 0) + elapsed
    
    def _run_before_hooks(
        self,
        hook_point: str,
        args: tuple,
        kwargs: dict,
        context: HookContext
    ):
        """Run all before hooks for a hook point."""
        with self._lock:
            hooks = list(self._before_hooks.get(hook_point, []))
        
        for hook in hooks:
            if not hook.enabled:
                continue
            
            if context.skip_remaining:
                break
            
            try:
                # Call hook with (args, kwargs, context)
                hook.callback(args, kwargs, context)
                
            except Exception as e:
                logger.error(f"Before hook '{hook.name}' failed: {e}")
    
    def _run_after_hooks(
        self,
        hook_point: str,
        result: Any,
        args: tuple,
        kwargs: dict,
        context: HookContext
    ) -> Any:
        """Run all after hooks for a hook point."""
        with self._lock:
            hooks = list(self._after_hooks.get(hook_point, []))
        
        for hook in hooks:
            if not hook.enabled:
                continue
            
            if context.skip_remaining:
                break
            
            try:
                # Call hook with (result, args, kwargs, context)
                hook_result = hook.callback(result, args, kwargs, context)
                
                # If hook returns a value, use it as the new result
                if hook_result is not None:
                    result = hook_result
                
            except Exception as e:
                logger.error(f"After hook '{hook.name}' failed: {e}")
        
        return result
    
    def list_hooks(self, hook_point: str = None) -> Dict[str, Dict[str, List[str]]]:
        """
        List registered hooks.
        
        Args:
            hook_point: Filter to specific hook point (optional)
            
        Returns:
            Dict of hook points to their before/after hooks
        """
        with self._lock:
            result = {}
            
            all_points = set(self._before_hooks.keys()) | set(self._after_hooks.keys())
            
            for point in all_points:
                if hook_point and point != hook_point:
                    continue
                
                result[point] = {
                    'before': [h.name for h in self._before_hooks.get(point, [])],
                    'after': [h.name for h in self._after_hooks.get(point, [])]
                }
            
            return result
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get hook execution statistics."""
        with self._lock:
            stats = {}
            
            for point in set(self._call_counts.keys()) | set(self._total_time.keys()):
                count = self._call_counts.get(point, 0)
                total = self._total_time.get(point, 0)
                
                stats[point] = {
                    'call_count': count,
                    'total_time': total,
                    'avg_time': total / count if count > 0 else 0
                }
            
            return stats
    
    def enable_hook(self, hook_point: str, hook_name: str, enabled: bool = True):
        """Enable or disable a specific hook."""
        with self._lock:
            for hooks in [self._before_hooks.get(hook_point, []),
                         self._after_hooks.get(hook_point, [])]:
                for hook in hooks:
                    if hook.name == hook_name:
                        hook.enabled = enabled
    
    def clear_hooks(self, hook_point: str = None):
        """Clear all hooks, optionally for a specific hook point."""
        with self._lock:
            if hook_point:
                self._before_hooks.pop(hook_point, None)
                self._after_hooks.pop(hook_point, None)
            else:
                self._before_hooks.clear()
                self._after_hooks.clear()
    
    @classmethod
    def reset(cls):
        """Reset the singleton (for testing only)."""
        cls._instance = None
        cls._initialized = False


# =============================================================================
# Global access
# =============================================================================

_manager: Optional[HookManager] = None


def get_hook_manager() -> HookManager:
    """Get the global hook manager instance."""
    global _manager
    if _manager is None:
        _manager = HookManager()
    return _manager


def before(hook_point: str, priority: int = 50, name: str = None):
    """
    Decorator to register a before hook.
    
    Usage:
        @before("model.generate")
        def my_hook(args, kwargs, context):
            context['start_time'] = time.time()
    """
    return get_hook_manager().before(hook_point, priority, name)


def after(hook_point: str, priority: int = 50, name: str = None):
    """
    Decorator to register an after hook.
    
    Usage:
        @after("model.generate")
        def my_hook(result, args, kwargs, context):
            print(f"Result: {result[:50]}...")
            return result
    """
    return get_hook_manager().after(hook_point, priority, name)


def hookable(hook_point: str = None):
    """
    Decorator to make a function hookable.
    
    Usage:
        @hookable("model.generate")
        def generate(prompt):
            return model(prompt)
    """
    return get_hook_manager().hookable(hook_point)


__all__ = [
    'Hook',
    'HookContext',
    'HookManager',
    'get_hook_manager',
    'before',
    'after',
    'hookable',
]
