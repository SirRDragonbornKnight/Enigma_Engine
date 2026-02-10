"""
================================================================================
Singleton Factory - Reduce Boilerplate for get_* Functions
================================================================================

Provides a clean way to create singleton instances without writing
the same global variable + get_* function pattern over and over.

Before (common pattern):
    _instance: Optional[MyClass] = None
    
    def get_my_class(**kwargs) -> MyClass:
        global _instance
        if _instance is None:
            _instance = MyClass(**kwargs)
        return _instance

After (with singleton factory):
    from enigma_engine.utils.singleton import singleton
    
    @singleton
    class MyClass:
        def __init__(self, **kwargs):
            ...
    
    # Access instance
    instance = MyClass.instance()  # or get_my_class()

USAGE:
    # Option 1: Decorator
    @singleton
    class MyService:
        pass
    
    service = MyService.instance()
    
    # Option 2: Factory function
    from enigma_engine.utils.singleton import SingletonFactory
    
    class MyClass:
        pass
    
    get_my_class = SingletonFactory(MyClass)
    instance = get_my_class()
"""

import threading
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar

T = TypeVar('T')


class SingletonFactory(Generic[T]):
    """
    Factory for creating singleton instances.
    
    Thread-safe lazy initialization with support for kwargs.
    
    Usage:
        class MyService:
            def __init__(self, debug: bool = False):
                self.debug = debug
        
        get_service = SingletonFactory(MyService)
        service = get_service(debug=True)  # Creates instance
        service2 = get_service()  # Returns same instance
    """
    
    def __init__(
        self,
        cls: Type[T],
        *default_args,
        **default_kwargs
    ):
        self._cls = cls
        self._default_args = default_args
        self._default_kwargs = default_kwargs
        self._instance: Optional[T] = None
        self._lock = threading.Lock()
    
    def __call__(self, *args, **kwargs) -> T:
        """Get or create the singleton instance."""
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    # Merge default kwargs with provided kwargs
                    merged_args = args or self._default_args
                    merged_kwargs = {**self._default_kwargs, **kwargs}
                    self._instance = self._cls(*merged_args, **merged_kwargs)
        return self._instance
    
    def reset(self) -> None:
        """Reset the singleton instance (useful for testing)."""
        with self._lock:
            self._instance = None
    
    @property
    def instance(self) -> Optional[T]:
        """Get current instance without creating."""
        return self._instance
    
    @property
    def is_initialized(self) -> bool:
        """Check if instance has been created."""
        return self._instance is not None


class SingletonMeta(type):
    """
    Metaclass for creating singleton classes.
    
    Usage:
        class MyService(metaclass=SingletonMeta):
            def __init__(self, config=None):
                self.config = config
        
        # First call creates instance with args
        service = MyService(config={'debug': True})
        
        # Subsequent calls return same instance (args ignored)
        same_service = MyService()  # Returns existing instance
        
        # Reset if needed
        MyService._reset()
        new_service = MyService(config={'debug': False})
    """
    
    _instances: Dict[type, Any] = {}
    _locks: Dict[type, threading.Lock] = {}
    
    def __call__(cls, *args, **kwargs):
        # Create lock for this class if needed
        if cls not in cls._locks:
            cls._locks[cls] = threading.Lock()
        
        if cls not in cls._instances:
            with cls._locks[cls]:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]
    
    def _reset(cls) -> None:
        """Reset the singleton instance."""
        if cls in cls._locks:
            with cls._locks[cls]:
                if cls in cls._instances:
                    del cls._instances[cls]
    
    def _instance(cls) -> Optional[Any]:
        """Get current instance without creating."""
        return cls._instances.get(cls)


def singleton(cls: Type[T]) -> Type[T]:
    """
    Decorator to make a class a singleton.
    
    Adds:
    - .instance() class method to get/create instance
    - .reset() class method to reset instance (for testing)
    - .is_initialized property
    
    Usage:
        @singleton
        class MyService:
            def __init__(self, debug: bool = False):
                self.debug = debug
        
        # Get instance (creates on first call)
        service = MyService.instance(debug=True)
        
        # Subsequent calls return same instance
        same = MyService.instance()
        
        # Can also use direct instantiation (returns same instance)
        also_same = MyService()
        
        # Reset for testing
        MyService.reset()
    """
    lock = threading.Lock()
    _instance: Optional[T] = None
    
    original_init = cls.__init__
    
    def __init__(self, *args, **kwargs):
        nonlocal _instance
        # Only run original init if this is the first creation
        if _instance is None:
            original_init(self, *args, **kwargs)
    
    def __new__(klass, *args, **kwargs):
        nonlocal _instance
        if _instance is None:
            with lock:
                if _instance is None:
                    _instance = object.__new__(klass)
        return _instance
    
    @classmethod
    def instance(klass, *args, **kwargs) -> T:
        """Get or create the singleton instance."""
        nonlocal _instance
        if _instance is None:
            with lock:
                if _instance is None:
                    _instance = object.__new__(klass)
                    original_init(_instance, *args, **kwargs)
        return _instance
    
    @classmethod  
    def reset(klass) -> None:
        """Reset the singleton instance."""
        nonlocal _instance
        with lock:
            _instance = None
    
    @classmethod
    def is_initialized(klass) -> bool:
        """Check if instance exists."""
        return _instance is not None
    
    cls.__new__ = __new__
    cls.__init__ = __init__
    cls.instance = instance
    cls.reset = reset
    cls.is_initialized = is_initialized
    
    return cls


def create_getter(
    cls: Type[T],
    *default_args,
    **default_kwargs
) -> Callable[..., T]:
    """
    Create a get_* function for a class.
    
    Shortcut for creating a factory function.
    
    Usage:
        class MyService:
            pass
        
        get_my_service = create_getter(MyService)
        
        # Equivalent to:
        # _instance = None
        # def get_my_service(**kwargs):
        #     global _instance
        #     if _instance is None:
        #         _instance = MyService(**kwargs)
        #     return _instance
    """
    factory = SingletonFactory(cls, *default_args, **default_kwargs)
    return factory


__all__ = [
    'SingletonFactory',
    'SingletonMeta',
    'singleton',
    'create_getter',
]
