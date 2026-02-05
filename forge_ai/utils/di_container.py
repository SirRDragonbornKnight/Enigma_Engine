"""
Dependency Injection Container - Simple DI for better testability.

Provides a lightweight dependency injection container that supports:
- Service registration (singleton, transient, scoped)
- Constructor injection
- Interface to implementation mapping
- Automatic dependency resolution
- Factory functions

Part of the ForgeAI architecture patterns.
"""

import inspect
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, Type, TypeVar, Generic, get_type_hints
from enum import Enum
from threading import Lock
from contextlib import contextmanager


T = TypeVar('T')


class Lifetime(Enum):
    """Lifetime scope of registered services."""
    SINGLETON = "singleton"     # Single instance for container lifetime
    TRANSIENT = "transient"     # New instance every time
    SCOPED = "scoped"           # Single instance per scope


class DependencyError(Exception):
    """Base exception for dependency injection errors."""
    pass


class ServiceNotRegisteredError(DependencyError):
    """Raised when resolving an unregistered service."""
    pass


class CircularDependencyError(DependencyError):
    """Raised when circular dependency detected."""
    pass


class ScopeNotActiveError(DependencyError):
    """Raised when trying to resolve scoped service outside scope."""
    pass


@dataclass
class ServiceDescriptor:
    """Describes a registered service."""
    service_type: Type
    implementation: Any  # Type, factory, or instance
    lifetime: Lifetime
    is_factory: bool = False
    instance: Any = None  # For singletons
    
    def __post_init__(self):
        """Validate descriptor."""
        if self.lifetime == Lifetime.SINGLETON and self.instance is not None:
            # Pre-instantiated singleton
            pass
        elif not self.is_factory and not inspect.isclass(self.implementation):
            # Must be a class or factory
            if not callable(self.implementation):
                raise DependencyError(
                    f"Implementation must be a class or callable, got {type(self.implementation)}"
                )
            self.is_factory = True


class Scope:
    """A dependency injection scope for scoped services."""
    
    def __init__(self, container: 'Container'):
        """Initialize scope."""
        self._container = container
        self._instances: Dict[Type, Any] = {}
        self._lock = Lock()
    
    def get_or_create(
        self,
        descriptor: ServiceDescriptor,
        resolver: Callable
    ) -> Any:
        """Get existing instance or create new one for this scope."""
        with self._lock:
            if descriptor.service_type not in self._instances:
                self._instances[descriptor.service_type] = resolver()
            return self._instances[descriptor.service_type]
    
    def dispose(self):
        """Dispose of scoped instances."""
        for instance in self._instances.values():
            if hasattr(instance, 'dispose'):
                try:
                    instance.dispose()
                except Exception:
                    pass
            elif hasattr(instance, 'close'):
                try:
                    instance.close()
                except Exception:
                    pass
        self._instances.clear()


class Container:
    """
    Dependency injection container.
    
    Usage:
        # Create container
        container = Container()
        
        # Register services
        container.register(IDatabase, PostgresDatabase, Lifetime.SINGLETON)
        container.register(ICache, RedisCache, Lifetime.SCOPED)
        container.register(EmailService, lifetime=Lifetime.TRANSIENT)
        
        # Register with factory
        container.register_factory(ILogger, lambda c: FileLogger(c.resolve(IConfig)))
        
        # Register instance
        container.register_instance(IConfig, config_instance)
        
        # Resolve services
        db = container.resolve(IDatabase)
        
        # Use scopes
        with container.create_scope() as scope:
            cache = scope.resolve(ICache)
            # cache is reused within scope
    """
    
    def __init__(self):
        """Initialize container."""
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._lock = Lock()
        self._resolution_stack: list = []  # For circular dependency detection
        self._current_scope: Optional[Scope] = None
    
    def register(
        self,
        service_type: Type[T],
        implementation: Optional[Type[T]] = None,
        lifetime: Lifetime = Lifetime.TRANSIENT
    ) -> 'Container':
        """
        Register a service type.
        
        Args:
            service_type: The type/interface to register
            implementation: Implementation class (defaults to service_type)
            lifetime: Service lifetime
            
        Returns:
            Self for chaining
        """
        impl = implementation or service_type
        
        with self._lock:
            self._services[service_type] = ServiceDescriptor(
                service_type=service_type,
                implementation=impl,
                lifetime=lifetime
            )
        
        return self
    
    def register_singleton(
        self,
        service_type: Type[T],
        implementation: Optional[Type[T]] = None
    ) -> 'Container':
        """Register a singleton service."""
        return self.register(service_type, implementation, Lifetime.SINGLETON)
    
    def register_transient(
        self,
        service_type: Type[T],
        implementation: Optional[Type[T]] = None
    ) -> 'Container':
        """Register a transient service."""
        return self.register(service_type, implementation, Lifetime.TRANSIENT)
    
    def register_scoped(
        self,
        service_type: Type[T],
        implementation: Optional[Type[T]] = None
    ) -> 'Container':
        """Register a scoped service."""
        return self.register(service_type, implementation, Lifetime.SCOPED)
    
    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[['Container'], T],
        lifetime: Lifetime = Lifetime.TRANSIENT
    ) -> 'Container':
        """
        Register a factory function for a service.
        
        Args:
            service_type: The type/interface to register
            factory: Factory function taking container, returning instance
            lifetime: Service lifetime
            
        Returns:
            Self for chaining
        """
        with self._lock:
            self._services[service_type] = ServiceDescriptor(
                service_type=service_type,
                implementation=factory,
                lifetime=lifetime,
                is_factory=True
            )
        
        return self
    
    def register_instance(
        self,
        service_type: Type[T],
        instance: T
    ) -> 'Container':
        """
        Register an existing instance as singleton.
        
        Args:
            service_type: The type/interface to register
            instance: Existing instance
            
        Returns:
            Self for chaining
        """
        with self._lock:
            self._services[service_type] = ServiceDescriptor(
                service_type=service_type,
                implementation=type(instance),
                lifetime=Lifetime.SINGLETON,
                instance=instance
            )
            self._singletons[service_type] = instance
        
        return self
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        return service_type in self._services
    
    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service instance.
        
        Args:
            service_type: The type to resolve
            
        Returns:
            Service instance
            
        Raises:
            ServiceNotRegisteredError: If service not registered
            CircularDependencyError: If circular dependency detected
        """
        if service_type not in self._services:
            raise ServiceNotRegisteredError(
                f"Service '{service_type.__name__}' is not registered"
            )
        
        # Check for circular dependency
        if service_type in self._resolution_stack:
            cycle = ' -> '.join(t.__name__ for t in self._resolution_stack)
            raise CircularDependencyError(
                f"Circular dependency detected: {cycle} -> {service_type.__name__}"
            )
        
        descriptor = self._services[service_type]
        
        # Handle different lifetimes
        if descriptor.lifetime == Lifetime.SINGLETON:
            return self._resolve_singleton(descriptor)
        elif descriptor.lifetime == Lifetime.SCOPED:
            return self._resolve_scoped(descriptor)
        else:  # TRANSIENT
            return self._create_instance(descriptor)
    
    def _resolve_singleton(self, descriptor: ServiceDescriptor) -> Any:
        """Resolve singleton instance."""
        with self._lock:
            if descriptor.service_type not in self._singletons:
                self._singletons[descriptor.service_type] = self._create_instance(descriptor)
            return self._singletons[descriptor.service_type]
    
    def _resolve_scoped(self, descriptor: ServiceDescriptor) -> Any:
        """Resolve scoped instance."""
        if self._current_scope is None:
            raise ScopeNotActiveError(
                f"Cannot resolve scoped service '{descriptor.service_type.__name__}' "
                "outside of a scope. Use container.create_scope() context manager."
            )
        
        return self._current_scope.get_or_create(
            descriptor,
            lambda: self._create_instance(descriptor)
        )
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create a new service instance."""
        if descriptor.instance is not None:
            return descriptor.instance
        
        self._resolution_stack.append(descriptor.service_type)
        
        try:
            if descriptor.is_factory:
                return descriptor.implementation(self)
            else:
                return self._construct(descriptor.implementation)
        finally:
            self._resolution_stack.pop()
    
    def _construct(self, cls: Type) -> Any:
        """Construct an instance by injecting dependencies."""
        # Get constructor parameters
        try:
            hints = get_type_hints(cls.__init__)
        except Exception:
            hints = {}
        
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            
            # Get type from hints or annotations
            param_type = hints.get(name, param.annotation)
            
            if param_type == inspect.Parameter.empty:
                # No type hint, check if has default
                if param.default != inspect.Parameter.empty:
                    continue
                raise DependencyError(
                    f"Cannot inject parameter '{name}' of {cls.__name__}: no type hint"
                )
            
            # Try to resolve the dependency
            if self.is_registered(param_type):
                kwargs[name] = self.resolve(param_type)
            elif param.default != inspect.Parameter.empty:
                # Has default, skip
                continue
            else:
                raise ServiceNotRegisteredError(
                    f"Cannot inject '{name}' of type '{param_type.__name__}' "
                    f"into {cls.__name__}: not registered"
                )
        
        return cls(**kwargs)
    
    @contextmanager
    def create_scope(self):
        """
        Create a new dependency scope.
        
        Usage:
            with container.create_scope() as scope:
                service = scope.resolve(IScopedService)
        """
        scope = Scope(self)
        old_scope = self._current_scope
        self._current_scope = scope
        
        try:
            yield self
        finally:
            scope.dispose()
            self._current_scope = old_scope
    
    def clear(self):
        """Clear all registrations and instances."""
        with self._lock:
            # Dispose singletons
            for instance in self._singletons.values():
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception:
                        pass
            
            self._services.clear()
            self._singletons.clear()
            self._resolution_stack.clear()
            self._current_scope = None
    
    def get_registered_types(self) -> list[Type]:
        """Get all registered service types."""
        return list(self._services.keys())
    
    def get_service_info(self, service_type: Type) -> Optional[Dict[str, Any]]:
        """Get info about a registered service."""
        if service_type not in self._services:
            return None
        
        desc = self._services[service_type]
        return {
            "service_type": desc.service_type.__name__,
            "implementation": (
                desc.implementation.__name__ 
                if inspect.isclass(desc.implementation) 
                else "factory"
            ),
            "lifetime": desc.lifetime.value,
            "is_instantiated": desc.service_type in self._singletons
        }


# Decorators for automatic registration
def injectable(
    lifetime: Lifetime = Lifetime.TRANSIENT,
    interface: Optional[Type] = None
):
    """
    Decorator to mark a class as injectable.
    
    Usage:
        @injectable(Lifetime.SINGLETON)
        class MyService:
            pass
        
        @injectable(interface=IDatabase)
        class PostgresDatabase:
            pass
    """
    def decorator(cls: Type[T]) -> Type[T]:
        cls._di_lifetime = lifetime
        cls._di_interface = interface or cls
        return cls
    
    return decorator


def singleton(cls: Type[T]) -> Type[T]:
    """Decorator to mark a class as singleton."""
    cls._di_lifetime = Lifetime.SINGLETON
    cls._di_interface = cls
    return cls


def scoped(cls: Type[T]) -> Type[T]:
    """Decorator to mark a class as scoped."""
    cls._di_lifetime = Lifetime.SCOPED
    cls._di_interface = cls
    return cls


def transient(cls: Type[T]) -> Type[T]:
    """Decorator to mark a class as transient."""
    cls._di_lifetime = Lifetime.TRANSIENT
    cls._di_interface = cls
    return cls


class ContainerBuilder:
    """
    Builder for creating configured containers.
    
    Usage:
        container = (ContainerBuilder()
            .add_singleton(IConfig, Config)
            .add_scoped(IDatabase, PostgresDatabase)
            .add_transient(ILogger, ConsoleLogger)
            .scan_assembly(my_module)
            .build())
    """
    
    def __init__(self):
        """Initialize builder."""
        self._registrations: list = []
        self._modules_to_scan: list = []
    
    def add_singleton(
        self,
        service_type: Type[T],
        implementation: Optional[Type[T]] = None
    ) -> 'ContainerBuilder':
        """Add singleton registration."""
        self._registrations.append(
            (service_type, implementation, Lifetime.SINGLETON, None)
        )
        return self
    
    def add_scoped(
        self,
        service_type: Type[T],
        implementation: Optional[Type[T]] = None
    ) -> 'ContainerBuilder':
        """Add scoped registration."""
        self._registrations.append(
            (service_type, implementation, Lifetime.SCOPED, None)
        )
        return self
    
    def add_transient(
        self,
        service_type: Type[T],
        implementation: Optional[Type[T]] = None
    ) -> 'ContainerBuilder':
        """Add transient registration."""
        self._registrations.append(
            (service_type, implementation, Lifetime.TRANSIENT, None)
        )
        return self
    
    def add_factory(
        self,
        service_type: Type[T],
        factory: Callable[[Container], T],
        lifetime: Lifetime = Lifetime.TRANSIENT
    ) -> 'ContainerBuilder':
        """Add factory registration."""
        self._registrations.append(
            (service_type, None, lifetime, factory)
        )
        return self
    
    def add_instance(
        self,
        service_type: Type[T],
        instance: T
    ) -> 'ContainerBuilder':
        """Add instance registration."""
        self._registrations.append(
            (service_type, instance, Lifetime.SINGLETON, 'instance')
        )
        return self
    
    def scan_module(self, module) -> 'ContainerBuilder':
        """Scan module for @injectable decorated classes."""
        self._modules_to_scan.append(module)
        return self
    
    def build(self) -> Container:
        """Build the container."""
        container = Container()
        
        # Apply explicit registrations
        for service_type, impl_or_instance, lifetime, factory_or_flag in self._registrations:
            if factory_or_flag == 'instance':
                container.register_instance(service_type, impl_or_instance)
            elif factory_or_flag is not None:
                container.register_factory(service_type, factory_or_flag, lifetime)
            else:
                container.register(service_type, impl_or_instance, lifetime)
        
        # Scan modules
        for module in self._modules_to_scan:
            self._scan_and_register(container, module)
        
        return container
    
    def _scan_and_register(self, container: Container, module) -> None:
        """Scan module for injectable classes."""
        for name in dir(module):
            obj = getattr(module, name)
            
            if inspect.isclass(obj) and hasattr(obj, '_di_lifetime'):
                interface = getattr(obj, '_di_interface', obj)
                lifetime = obj._di_lifetime
                
                if not container.is_registered(interface):
                    container.register(interface, obj, lifetime)


# Global container
_global_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container."""
    global _global_container
    if _global_container is None:
        _global_container = Container()
    return _global_container


def set_container(container: Container) -> None:
    """Set the global container."""
    global _global_container
    _global_container = container


def resolve(service_type: Type[T]) -> T:
    """Resolve from global container."""
    return get_container().resolve(service_type)


def register(
    service_type: Type[T],
    implementation: Optional[Type[T]] = None,
    lifetime: Lifetime = Lifetime.TRANSIENT
) -> None:
    """Register in global container."""
    get_container().register(service_type, implementation, lifetime)
