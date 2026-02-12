"""
Resource Pool for Enigma AI Engine.

Pools expensive resources (database connections, model instances, etc.)
for efficient reuse. Supports min/max pool size, timeouts, and health checks.

Usage:
    from enigma_engine.utils.resource_pool import ResourcePool, PoolConfig
    
    # Create a pool with factory function
    pool = ResourcePool(
        factory=lambda: create_db_connection(),
        config=PoolConfig(min_size=2, max_size=10)
    )
    
    # Acquire and release resources
    conn = pool.acquire()
    try:
        result = conn.execute("SELECT * FROM users")
    finally:
        pool.release(conn)
    
    # Or use context manager
    with pool.acquire_context() as conn:
        result = conn.execute("SELECT * FROM users")
    
    # Model pool example
    model_pool = ResourcePool(
        factory=lambda: load_model("forge-small"),
        config=PoolConfig(min_size=1, max_size=3, name="models")
    )
"""

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PoolConfig:
    """Configuration for resource pools."""
    name: str = "default"
    min_size: int = 1
    max_size: int = 10
    acquire_timeout: float = 30.0
    idle_timeout: float = 300.0
    max_lifetime: float = 3600.0
    health_check_interval: float = 60.0
    validation_on_acquire: bool = True
    validation_on_release: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.min_size < 0:
            raise ValueError("min_size must be >= 0")
        if self.max_size < self.min_size:
            raise ValueError("max_size must be >= min_size")
        if self.acquire_timeout < 0:
            raise ValueError("acquire_timeout must be >= 0")


@dataclass
class PooledResource(Generic[T]):
    """Wrapper around a pooled resource with metadata."""
    resource: T
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    last_validated: float = field(default_factory=time.time)
    use_count: int = 0
    is_valid: bool = True
    
    def mark_used(self) -> None:
        """Mark resource as used."""
        self.last_used = time.time()
        self.use_count += 1
    
    def is_expired(self, max_lifetime: float) -> bool:
        """Check if resource has exceeded max lifetime."""
        return time.time() - self.created_at > max_lifetime
    
    def is_idle(self, idle_timeout: float) -> bool:
        """Check if resource has been idle too long."""
        return time.time() - self.last_used > idle_timeout


@dataclass
class PoolStats:
    """Pool statistics."""
    name: str = ""
    total_size: int = 0
    available: int = 0
    in_use: int = 0
    created_count: int = 0
    destroyed_count: int = 0
    acquire_count: int = 0
    release_count: int = 0
    timeout_count: int = 0
    validation_failures: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "total_size": self.total_size,
            "available": self.available,
            "in_use": self.in_use,
            "created_count": self.created_count,
            "destroyed_count": self.destroyed_count,
            "acquire_count": self.acquire_count,
            "release_count": self.release_count,
            "timeout_count": self.timeout_count,
            "validation_failures": self.validation_failures,
        }


class ResourceFactory(ABC, Generic[T]):
    """Abstract factory for creating and managing resources."""
    
    @abstractmethod
    def create(self) -> T:
        """Create a new resource."""
    
    @abstractmethod
    def destroy(self, resource: T) -> None:
        """Destroy a resource."""
    
    def validate(self, resource: T) -> bool:
        """Validate a resource is still usable. Override for custom validation."""
        return True
    
    def reset(self, resource: T) -> bool:
        """Reset resource state for reuse. Override for custom reset logic."""
        return True


class SimpleFactory(ResourceFactory[T]):
    """Simple factory using callable functions."""
    
    def __init__(
        self,
        create_func: Callable[[], T],
        destroy_func: Optional[Callable[[T], None]] = None,
        validate_func: Optional[Callable[[T], bool]] = None,
        reset_func: Optional[Callable[[T], bool]] = None
    ):
        self._create = create_func
        self._destroy = destroy_func
        self._validate = validate_func
        self._reset = reset_func
    
    def create(self) -> T:
        return self._create()
    
    def destroy(self, resource: T) -> None:
        if self._destroy:
            self._destroy(resource)
    
    def validate(self, resource: T) -> bool:
        if self._validate:
            return self._validate(resource)
        return True
    
    def reset(self, resource: T) -> bool:
        if self._reset:
            return self._reset(resource)
        return True


class ResourcePool(Generic[T]):
    """
    Generic resource pool for managing expensive resources.
    
    Features:
        - Min/max pool size
        - Resource validation
        - Idle timeout cleanup
        - Max lifetime expiry
        - Health check monitoring
        - Thread-safe
    """
    
    def __init__(
        self,
        factory: Optional[Callable[[], T]] = None,
        config: Optional[PoolConfig] = None,
        resource_factory: Optional[ResourceFactory[T]] = None,
        destroy: Optional[Callable[[T], None]] = None,
        validate: Optional[Callable[[T], bool]] = None,
    ):
        """
        Initialize pool.
        
        Args:
            factory: Function to create resources (simple mode)
            config: Pool configuration
            resource_factory: ResourceFactory instance (advanced mode)
            destroy: Function to destroy resources (simple mode)
            validate: Function to validate resources (simple mode)
        """
        self.config = config or PoolConfig()
        
        if resource_factory:
            self._factory = resource_factory
        elif factory:
            self._factory = SimpleFactory(factory, destroy, validate)
        else:
            raise ValueError("Must provide factory or resource_factory")
        
        self._pool: queue.Queue[PooledResource[T]] = queue.Queue()
        self._in_use: dict[int, PooledResource[T]] = {}
        self._lock = threading.RLock()
        self._stats = PoolStats(name=self.config.name)
        
        self._running = False
        self._maintenance_thread: Optional[threading.Thread] = None
        
        # Initialize minimum pool size
        self._ensure_min_size()
    
    def _ensure_min_size(self) -> None:
        """Ensure pool has minimum number of resources."""
        with self._lock:
            current = self._pool.qsize() + len(self._in_use)
            needed = self.config.min_size - current
            
            for _ in range(needed):
                try:
                    self._create_resource()
                except Exception as e:
                    logger.error(f"Pool '{self.config.name}': Failed to create resource: {e}")
                    break
    
    def _create_resource(self) -> PooledResource[T]:
        """Create a new pooled resource."""
        resource = self._factory.create()
        pooled = PooledResource(resource=resource)
        self._pool.put(pooled)
        self._stats.created_count += 1
        self._stats.total_size += 1
        self._stats.available += 1
        
        logger.debug(f"Pool '{self.config.name}': Created resource (total: {self._stats.total_size})")
        return pooled
    
    def _destroy_resource(self, pooled: PooledResource[T]) -> None:
        """Destroy a pooled resource."""
        try:
            self._factory.destroy(pooled.resource)
        except Exception as e:
            logger.warning(f"Pool '{self.config.name}': Error destroying resource: {e}")
        
        self._stats.destroyed_count += 1
        self._stats.total_size -= 1
        logger.debug(f"Pool '{self.config.name}': Destroyed resource (total: {self._stats.total_size})")
    
    def acquire(self, timeout: Optional[float] = None) -> T:
        """
        Acquire a resource from the pool.
        
        Args:
            timeout: Max time to wait (None uses config default)
        
        Returns:
            Resource instance
        
        Raises:
            TimeoutError: If no resource available within timeout
            RuntimeError: If pool is closed
        """
        timeout = timeout if timeout is not None else self.config.acquire_timeout
        deadline = time.time() + timeout
        
        while True:
            with self._lock:
                # Try to get from pool
                try:
                    pooled = self._pool.get_nowait()
                    self._stats.available -= 1
                except queue.Empty:
                    pooled = None
                
                # Check if we can create new resource
                if pooled is None and self._stats.total_size < self.config.max_size:
                    try:
                        pooled = self._create_resource()
                        self._pool.get_nowait()  # Remove from available
                        self._stats.available -= 1
                    except Exception as e:
                        logger.error(f"Pool '{self.config.name}': Failed to create resource: {e}")
            
            if pooled:
                # Validate resource if needed
                if self.config.validation_on_acquire:
                    if pooled.is_expired(self.config.max_lifetime):
                        logger.debug(f"Pool '{self.config.name}': Resource expired, destroying")
                        self._destroy_resource(pooled)
                        continue
                    
                    try:
                        if not self._factory.validate(pooled.resource):
                            logger.debug(f"Pool '{self.config.name}': Validation failed, destroying")
                            self._stats.validation_failures += 1
                            self._destroy_resource(pooled)
                            continue
                    except Exception as e:
                        logger.warning(f"Pool '{self.config.name}': Validation error: {e}")
                        self._stats.validation_failures += 1
                        self._destroy_resource(pooled)
                        continue
                    
                    pooled.last_validated = time.time()
                
                pooled.mark_used()
                
                with self._lock:
                    self._in_use[id(pooled.resource)] = pooled
                    self._stats.acquire_count += 1
                    self._stats.in_use = len(self._in_use)
                
                return pooled.resource
            
            # Wait for resource to become available
            remaining = deadline - time.time()
            if remaining <= 0:
                self._stats.timeout_count += 1
                raise TimeoutError(
                    f"Pool '{self.config.name}': Timeout waiting for resource "
                    f"(in_use: {len(self._in_use)}, max: {self.config.max_size})"
                )
            
            time.sleep(min(0.1, remaining))
    
    def release(self, resource: T) -> None:
        """
        Release a resource back to the pool.
        
        Args:
            resource: Resource to release
        """
        resource_id = id(resource)
        
        with self._lock:
            if resource_id not in self._in_use:
                logger.warning(f"Pool '{self.config.name}': Releasing unknown resource")
                return
            
            pooled = self._in_use.pop(resource_id)
            self._stats.release_count += 1
            self._stats.in_use = len(self._in_use)
        
        # Validate before returning to pool
        if self.config.validation_on_release:
            try:
                if not self._factory.validate(pooled.resource):
                    logger.debug(f"Pool '{self.config.name}': Release validation failed")
                    self._stats.validation_failures += 1
                    self._destroy_resource(pooled)
                    self._ensure_min_size()
                    return
            except Exception as e:
                logger.warning(f"Pool '{self.config.name}': Release validation error: {e}")
                self._destroy_resource(pooled)
                self._ensure_min_size()
                return
        
        # Reset resource state
        try:
            if not self._factory.reset(pooled.resource):
                logger.debug(f"Pool '{self.config.name}': Reset failed, destroying")
                self._destroy_resource(pooled)
                self._ensure_min_size()
                return
        except Exception as e:
            logger.warning(f"Pool '{self.config.name}': Reset error: {e}")
            self._destroy_resource(pooled)
            self._ensure_min_size()
            return
        
        # Return to pool
        pooled.is_valid = True
        self._pool.put(pooled)
        
        with self._lock:
            self._stats.available += 1
    
    @contextmanager
    def acquire_context(self, timeout: Optional[float] = None):
        """
        Context manager for acquiring resources.
        
        Args:
            timeout: Max time to wait
        
        Yields:
            Resource instance
        
        Example:
            with pool.acquire_context() as resource:
                result = resource.do_something()
        """
        resource = self.acquire(timeout)
        try:
            yield resource
        finally:
            self.release(resource)
    
    def stats(self) -> PoolStats:
        """Get pool statistics."""
        with self._lock:
            return PoolStats(
                name=self._stats.name,
                total_size=self._stats.total_size,
                available=self._pool.qsize(),
                in_use=len(self._in_use),
                created_count=self._stats.created_count,
                destroyed_count=self._stats.destroyed_count,
                acquire_count=self._stats.acquire_count,
                release_count=self._stats.release_count,
                timeout_count=self._stats.timeout_count,
                validation_failures=self._stats.validation_failures,
            )
    
    def resize(self, min_size: int, max_size: int) -> None:
        """
        Resize the pool.
        
        Args:
            min_size: New minimum size
            max_size: New maximum size
        """
        if max_size < min_size:
            raise ValueError("max_size must be >= min_size")
        
        self.config.min_size = min_size
        self.config.max_size = max_size
        
        # Shrink if needed
        with self._lock:
            while self._pool.qsize() > max_size - len(self._in_use):
                try:
                    pooled = self._pool.get_nowait()
                    self._stats.available -= 1
                    self._destroy_resource(pooled)
                except queue.Empty:
                    break
        
        # Grow if needed
        self._ensure_min_size()
    
    def clear(self) -> None:
        """Remove all idle resources from pool."""
        with self._lock:
            while True:
                try:
                    pooled = self._pool.get_nowait()
                    self._stats.available -= 1
                    self._destroy_resource(pooled)
                except queue.Empty:
                    break
    
    def close(self) -> None:
        """Close the pool and destroy all resources."""
        self.stop_maintenance()
        
        # Wait for in-use resources
        timeout = 30.0
        start = time.time()
        while self._in_use and time.time() - start < timeout:
            time.sleep(0.1)
        
        if self._in_use:
            logger.warning(f"Pool '{self.config.name}': {len(self._in_use)} resources still in use at close")
            with self._lock:
                for pooled in self._in_use.values():
                    self._destroy_resource(pooled)
                self._in_use.clear()
        
        self.clear()
        logger.info(f"Pool '{self.config.name}': Closed")
    
    def start_maintenance(self) -> None:
        """Start background maintenance thread."""
        if self._running:
            return
        
        self._running = True
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True,
            name=f"pool-{self.config.name}-maintenance"
        )
        self._maintenance_thread.start()
        logger.info(f"Pool '{self.config.name}': Started maintenance thread")
    
    def stop_maintenance(self) -> None:
        """Stop background maintenance thread."""
        self._running = False
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5.0)
            self._maintenance_thread = None
    
    def _maintenance_loop(self) -> None:
        """Background maintenance loop."""
        while self._running:
            try:
                self._run_maintenance()
            except Exception as e:
                logger.error(f"Pool '{self.config.name}': Maintenance error: {e}")
            
            time.sleep(self.config.health_check_interval)
    
    def _run_maintenance(self) -> None:
        """Run maintenance tasks."""
        # Remove expired/idle resources (keep minimum)
        to_destroy = []
        
        with self._lock:
            temp_list = []
            while True:
                try:
                    pooled = self._pool.get_nowait()
                    self._stats.available -= 1
                    
                    if (pooled.is_expired(self.config.max_lifetime) or
                        pooled.is_idle(self.config.idle_timeout)):
                        if self._stats.total_size > self.config.min_size:
                            to_destroy.append(pooled)
                        else:
                            temp_list.append(pooled)
                    else:
                        temp_list.append(pooled)
                except queue.Empty:
                    break
            
            # Return valid resources
            for pooled in temp_list:
                self._pool.put(pooled)
                self._stats.available += 1
        
        # Destroy expired resources
        for pooled in to_destroy:
            self._destroy_resource(pooled)
        
        if to_destroy:
            logger.debug(f"Pool '{self.config.name}': Cleaned up {len(to_destroy)} expired resources")
        
        # Ensure minimum size
        self._ensure_min_size()
    
    def __enter__(self) -> "ResourcePool[T]":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


# Global pool registry
_pools: dict[str, ResourcePool] = {}
_pools_lock = threading.Lock()


def get_pool(name: str) -> Optional[ResourcePool]:
    """Get a pool by name."""
    with _pools_lock:
        return _pools.get(name)


def register_pool(pool: ResourcePool) -> None:
    """Register a pool globally."""
    with _pools_lock:
        _pools[pool.config.name] = pool
        logger.info(f"Registered pool: {pool.config.name}")


def unregister_pool(name: str) -> bool:
    """Unregister and close a pool."""
    with _pools_lock:
        if name in _pools:
            pool = _pools.pop(name)
            pool.close()
            return True
        return False


def list_pools() -> dict[str, dict]:
    """List all pools and their stats."""
    with _pools_lock:
        return {
            name: pool.stats().to_dict()
            for name, pool in _pools.items()
        }


def close_all_pools() -> None:
    """Close all registered pools."""
    with _pools_lock:
        for name in list(_pools.keys()):
            _pools[name].close()
        _pools.clear()


# Pre-configured pools for Enigma AI Engine

def create_model_pool(
    model_loader: Callable[[], Any],
    min_size: int = 1,
    max_size: int = 3,
    name: str = "models"
) -> ResourcePool:
    """
    Create a pool for model instances.
    
    Args:
        model_loader: Function to load model
        min_size: Minimum pool size
        max_size: Maximum pool size
        name: Pool name
    
    Returns:
        Configured resource pool
    """
    config = PoolConfig(
        name=name,
        min_size=min_size,
        max_size=max_size,
        idle_timeout=600.0,  # 10 minutes
        max_lifetime=7200.0,  # 2 hours
        validation_on_acquire=False,  # Models don't need validation
    )
    
    pool = ResourcePool(
        factory=model_loader,
        config=config,
    )
    
    register_pool(pool)
    return pool


def create_connection_pool(
    connector: Callable[[], Any],
    validator: Optional[Callable[[Any], bool]] = None,
    min_size: int = 2,
    max_size: int = 10,
    name: str = "connections"
) -> ResourcePool:
    """
    Create a pool for database/network connections.
    
    Args:
        connector: Function to create connection
        validator: Function to validate connection
        min_size: Minimum pool size
        max_size: Maximum pool size
        name: Pool name
    
    Returns:
        Configured resource pool
    """
    config = PoolConfig(
        name=name,
        min_size=min_size,
        max_size=max_size,
        idle_timeout=300.0,  # 5 minutes
        max_lifetime=3600.0,  # 1 hour
        validation_on_acquire=True,
        validation_on_release=True,
    )
    
    def destroy_connection(conn):
        if hasattr(conn, 'close'):
            try:
                conn.close()
            except Exception:
                pass  # Intentionally silent
    
    pool = ResourcePool(
        factory=connector,
        config=config,
        destroy=destroy_connection,
        validate=validator,
    )
    
    register_pool(pool)
    return pool
