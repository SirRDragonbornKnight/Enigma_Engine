"""
Bulkhead Pattern - Isolate elements to prevent cascading failures.

The Bulkhead pattern isolates critical resources into separate pools
so that failure in one pool doesn't affect others. Named after
ship bulkheads that prevent flooding from spreading.

Features:
- Semaphore-based concurrency limiting
- Separate resource pools per partition
- Timeout and queue management
- Async and sync support
- Metrics and monitoring

Part of the ForgeAI resilience patterns.
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, TypeVar, Generic
from enum import Enum
from collections import deque
from contextlib import contextmanager, asynccontextmanager
from functools import wraps


T = TypeVar('T')


class BulkheadState(Enum):
    """State of a bulkhead partition."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    SATURATED = "saturated"


class RejectionReason(Enum):
    """Reason for request rejection."""
    BULKHEAD_FULL = "bulkhead_full"
    QUEUE_FULL = "queue_full"
    TIMEOUT = "timeout"
    PARTITION_DISABLED = "partition_disabled"


class BulkheadRejectedError(Exception):
    """Raised when a request is rejected by the bulkhead."""
    
    def __init__(self, reason: RejectionReason, partition: str, message: str):
        self.reason = reason
        self.partition = partition
        super().__init__(message)


@dataclass
class BulkheadMetrics:
    """Metrics for a bulkhead partition."""
    partition: str
    max_concurrent: int
    current_concurrent: int = 0
    max_queue_size: int = 0
    current_queue_size: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    rejected_requests: int = 0
    timed_out_requests: int = 0
    total_wait_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0
    
    @property
    def state(self) -> BulkheadState:
        """Get current state based on utilization."""
        utilization = self.current_concurrent / self.max_concurrent if self.max_concurrent > 0 else 0
        
        if utilization >= 1.0:
            return BulkheadState.SATURATED
        elif utilization >= 0.8:
            return BulkheadState.DEGRADED
        return BulkheadState.HEALTHY
    
    @property
    def avg_wait_time_ms(self) -> float:
        """Average time requests spend waiting."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_wait_time_ms / self.successful_requests
    
    @property
    def avg_execution_time_ms(self) -> float:
        """Average execution time."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_execution_time_ms / self.successful_requests
    
    @property
    def rejection_rate(self) -> float:
        """Percentage of rejected requests."""
        if self.total_requests == 0:
            return 0.0
        return (self.rejected_requests / self.total_requests) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "partition": self.partition,
            "state": self.state.value,
            "max_concurrent": self.max_concurrent,
            "current_concurrent": self.current_concurrent,
            "max_queue_size": self.max_queue_size,
            "current_queue_size": self.current_queue_size,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "rejected_requests": self.rejected_requests,
            "timed_out_requests": self.timed_out_requests,
            "avg_wait_time_ms": round(self.avg_wait_time_ms, 2),
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "rejection_rate": round(self.rejection_rate, 2)
        }


@dataclass
class BulkheadConfig:
    """Configuration for a bulkhead partition."""
    max_concurrent: int = 10
    max_queue_size: int = 0  # 0 = no queue
    queue_timeout_ms: float = 0  # 0 = no timeout
    enabled: bool = True
    
    def validate(self):
        """Validate configuration."""
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if self.max_queue_size < 0:
            raise ValueError("max_queue_size cannot be negative")
        if self.queue_timeout_ms < 0:
            raise ValueError("queue_timeout_ms cannot be negative")


class BulkheadPartition:
    """
    A single bulkhead partition with its own concurrency limit.
    
    Usage:
        partition = BulkheadPartition("database", max_concurrent=5)
        
        # Context manager
        with partition.acquire():
            # Do work
            pass
        
        # Async context manager
        async with partition.acquire_async():
            # Do async work
            pass
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[BulkheadConfig] = None
    ):
        """
        Initialize partition.
        
        Args:
            name: Partition name
            config: Optional configuration
        """
        self.name = name
        self.config = config or BulkheadConfig()
        self.config.validate()
        
        self._semaphore = threading.Semaphore(self.config.max_concurrent)
        self._async_semaphore: Optional[asyncio.Semaphore] = None
        self._queue: deque = deque()
        self._lock = threading.Lock()
        self._metrics = BulkheadMetrics(
            partition=name,
            max_concurrent=self.config.max_concurrent,
            max_queue_size=self.config.max_queue_size
        )
    
    @property
    def metrics(self) -> BulkheadMetrics:
        """Get current metrics."""
        return self._metrics
    
    @contextmanager
    def acquire(self, timeout_ms: Optional[float] = None):
        """
        Acquire a slot in the bulkhead (sync).
        
        Args:
            timeout_ms: Override queue timeout
            
        Yields:
            None when slot acquired
            
        Raises:
            BulkheadRejectedError: If request is rejected
        """
        if not self.config.enabled:
            raise BulkheadRejectedError(
                RejectionReason.PARTITION_DISABLED,
                self.name,
                f"Partition '{self.name}' is disabled"
            )
        
        self._metrics.total_requests += 1
        start_wait = time.time()
        timeout = (timeout_ms or self.config.queue_timeout_ms) / 1000 if (timeout_ms or self.config.queue_timeout_ms) else None
        
        # Try to acquire immediately
        if self._semaphore.acquire(blocking=False):
            self._metrics.current_concurrent += 1
            start_exec = time.time()
            self._metrics.total_wait_time_ms += (start_exec - start_wait) * 1000
            
            try:
                yield
                self._metrics.successful_requests += 1
            finally:
                exec_time = (time.time() - start_exec) * 1000
                self._metrics.total_execution_time_ms += exec_time
                self._metrics.current_concurrent -= 1
                self._semaphore.release()
            return
        
        # Check if we can queue
        if self.config.max_queue_size > 0:
            with self._lock:
                if len(self._queue) >= self.config.max_queue_size:
                    self._metrics.rejected_requests += 1
                    raise BulkheadRejectedError(
                        RejectionReason.QUEUE_FULL,
                        self.name,
                        f"Queue full for partition '{self.name}'"
                    )
                
                self._queue.append(threading.current_thread().ident)
                self._metrics.current_queue_size = len(self._queue)
            
            try:
                # Wait with timeout
                acquired = self._semaphore.acquire(blocking=True, timeout=timeout)
                
                if not acquired:
                    self._metrics.timed_out_requests += 1
                    self._metrics.rejected_requests += 1
                    raise BulkheadRejectedError(
                        RejectionReason.TIMEOUT,
                        self.name,
                        f"Timeout waiting for partition '{self.name}'"
                    )
                
                self._metrics.current_concurrent += 1
                start_exec = time.time()
                self._metrics.total_wait_time_ms += (start_exec - start_wait) * 1000
                
                try:
                    yield
                    self._metrics.successful_requests += 1
                finally:
                    exec_time = (time.time() - start_exec) * 1000
                    self._metrics.total_execution_time_ms += exec_time
                    self._metrics.current_concurrent -= 1
                    self._semaphore.release()
            finally:
                with self._lock:
                    if threading.current_thread().ident in self._queue:
                        self._queue.remove(threading.current_thread().ident)
                    self._metrics.current_queue_size = len(self._queue)
        else:
            # No queue, reject immediately
            self._metrics.rejected_requests += 1
            raise BulkheadRejectedError(
                RejectionReason.BULKHEAD_FULL,
                self.name,
                f"Bulkhead full for partition '{self.name}'"
            )
    
    @asynccontextmanager
    async def acquire_async(self, timeout_ms: Optional[float] = None):
        """
        Acquire a slot in the bulkhead (async).
        
        Args:
            timeout_ms: Override queue timeout
            
        Yields:
            None when slot acquired
            
        Raises:
            BulkheadRejectedError: If request is rejected
        """
        if not self.config.enabled:
            raise BulkheadRejectedError(
                RejectionReason.PARTITION_DISABLED,
                self.name,
                f"Partition '{self.name}' is disabled"
            )
        
        # Lazy init async semaphore
        if self._async_semaphore is None:
            self._async_semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        self._metrics.total_requests += 1
        start_wait = time.time()
        timeout = (timeout_ms or self.config.queue_timeout_ms) / 1000 if (timeout_ms or self.config.queue_timeout_ms) else None
        
        try:
            if timeout:
                await asyncio.wait_for(
                    self._async_semaphore.acquire(),
                    timeout=timeout
                )
            else:
                await self._async_semaphore.acquire()
            
            self._metrics.current_concurrent += 1
            start_exec = time.time()
            self._metrics.total_wait_time_ms += (start_exec - start_wait) * 1000
            
            try:
                yield
                self._metrics.successful_requests += 1
            finally:
                exec_time = (time.time() - start_exec) * 1000
                self._metrics.total_execution_time_ms += exec_time
                self._metrics.current_concurrent -= 1
                self._async_semaphore.release()
                
        except asyncio.TimeoutError:
            self._metrics.timed_out_requests += 1
            self._metrics.rejected_requests += 1
            raise BulkheadRejectedError(
                RejectionReason.TIMEOUT,
                self.name,
                f"Timeout waiting for partition '{self.name}'"
            )
    
    def reset_metrics(self):
        """Reset metrics to initial state."""
        self._metrics = BulkheadMetrics(
            partition=self.name,
            max_concurrent=self.config.max_concurrent,
            max_queue_size=self.config.max_queue_size
        )


class Bulkhead:
    """
    Bulkhead manager with multiple partitions.
    
    Usage:
        bulkhead = Bulkhead()
        
        # Create partitions
        bulkhead.create_partition("database", max_concurrent=5)
        bulkhead.create_partition("api", max_concurrent=10, max_queue_size=20)
        
        # Use partitions
        with bulkhead.acquire("database"):
            # Database work
            pass
        
        # Decorator
        @bulkhead.protect("api")
        def call_api():
            pass
        
        # Get metrics
        metrics = bulkhead.get_all_metrics()
    """
    
    def __init__(self):
        """Initialize bulkhead manager."""
        self._partitions: Dict[str, BulkheadPartition] = {}
        self._lock = threading.Lock()
    
    def create_partition(
        self,
        name: str,
        max_concurrent: int = 10,
        max_queue_size: int = 0,
        queue_timeout_ms: float = 0
    ) -> BulkheadPartition:
        """
        Create a new partition.
        
        Args:
            name: Partition name
            max_concurrent: Max concurrent requests
            max_queue_size: Max queue size (0 = no queue)
            queue_timeout_ms: Queue timeout in ms (0 = no timeout)
            
        Returns:
            Created partition
        """
        config = BulkheadConfig(
            max_concurrent=max_concurrent,
            max_queue_size=max_queue_size,
            queue_timeout_ms=queue_timeout_ms
        )
        
        partition = BulkheadPartition(name, config)
        
        with self._lock:
            self._partitions[name] = partition
        
        return partition
    
    def get_partition(self, name: str) -> Optional[BulkheadPartition]:
        """Get a partition by name."""
        return self._partitions.get(name)
    
    def remove_partition(self, name: str) -> bool:
        """Remove a partition."""
        with self._lock:
            if name in self._partitions:
                del self._partitions[name]
                return True
        return False
    
    @contextmanager
    def acquire(self, partition_name: str, timeout_ms: Optional[float] = None):
        """
        Acquire a slot in a partition.
        
        Args:
            partition_name: Name of partition
            timeout_ms: Optional timeout override
            
        Yields:
            None when slot acquired
        """
        partition = self._partitions.get(partition_name)
        if partition is None:
            raise ValueError(f"Unknown partition: {partition_name}")
        
        with partition.acquire(timeout_ms):
            yield
    
    @asynccontextmanager
    async def acquire_async(self, partition_name: str, timeout_ms: Optional[float] = None):
        """
        Acquire a slot in a partition (async).
        
        Args:
            partition_name: Name of partition
            timeout_ms: Optional timeout override
            
        Yields:
            None when slot acquired
        """
        partition = self._partitions.get(partition_name)
        if partition is None:
            raise ValueError(f"Unknown partition: {partition_name}")
        
        async with partition.acquire_async(timeout_ms):
            yield
    
    def protect(
        self,
        partition_name: str,
        timeout_ms: Optional[float] = None,
        fallback: Optional[Callable] = None
    ):
        """
        Decorator to protect a function with bulkhead.
        
        Args:
            partition_name: Name of partition
            timeout_ms: Optional timeout override
            fallback: Optional fallback function on rejection
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    with self.acquire(partition_name, timeout_ms):
                        return func(*args, **kwargs)
                except BulkheadRejectedError:
                    if fallback:
                        return fallback(*args, **kwargs)
                    raise
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    async with self.acquire_async(partition_name, timeout_ms):
                        return await func(*args, **kwargs)
                except BulkheadRejectedError:
                    if fallback:
                        if asyncio.iscoroutinefunction(fallback):
                            return await fallback(*args, **kwargs)
                        return fallback(*args, **kwargs)
                    raise
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return wrapper
        
        return decorator
    
    def get_metrics(self, partition_name: str) -> Optional[BulkheadMetrics]:
        """Get metrics for a partition."""
        partition = self._partitions.get(partition_name)
        return partition.metrics if partition else None
    
    def get_all_metrics(self) -> Dict[str, BulkheadMetrics]:
        """Get metrics for all partitions."""
        return {name: p.metrics for name, p in self._partitions.items()}
    
    def get_health(self) -> Dict[str, BulkheadState]:
        """Get health state of all partitions."""
        return {name: p.metrics.state for name, p in self._partitions.items()}
    
    def is_healthy(self, partition_name: Optional[str] = None) -> bool:
        """
        Check if partition(s) are healthy.
        
        Args:
            partition_name: Specific partition or all if None
            
        Returns:
            True if healthy
        """
        if partition_name:
            partition = self._partitions.get(partition_name)
            if partition is None:
                return False
            return partition.metrics.state == BulkheadState.HEALTHY
        
        return all(
            p.metrics.state == BulkheadState.HEALTHY
            for p in self._partitions.values()
        )
    
    def enable_partition(self, partition_name: str) -> bool:
        """Enable a partition."""
        partition = self._partitions.get(partition_name)
        if partition:
            partition.config.enabled = True
            return True
        return False
    
    def disable_partition(self, partition_name: str) -> bool:
        """Disable a partition."""
        partition = self._partitions.get(partition_name)
        if partition:
            partition.config.enabled = False
            return True
        return False
    
    def reset_metrics(self, partition_name: Optional[str] = None):
        """Reset metrics for partition(s)."""
        if partition_name:
            partition = self._partitions.get(partition_name)
            if partition:
                partition.reset_metrics()
        else:
            for partition in self._partitions.values():
                partition.reset_metrics()


# Global bulkhead instance
_global_bulkhead: Optional[Bulkhead] = None


def get_bulkhead() -> Bulkhead:
    """Get the global bulkhead instance."""
    global _global_bulkhead
    if _global_bulkhead is None:
        _global_bulkhead = Bulkhead()
    return _global_bulkhead


def create_partition(
    name: str,
    max_concurrent: int = 10,
    max_queue_size: int = 0,
    queue_timeout_ms: float = 0
) -> BulkheadPartition:
    """Create a partition in the global bulkhead."""
    return get_bulkhead().create_partition(
        name, max_concurrent, max_queue_size, queue_timeout_ms
    )


def protect(
    partition_name: str,
    timeout_ms: Optional[float] = None,
    fallback: Optional[Callable] = None
):
    """Decorator to protect with global bulkhead."""
    return get_bulkhead().protect(partition_name, timeout_ms, fallback)


@contextmanager
def acquire(partition_name: str, timeout_ms: Optional[float] = None):
    """Acquire from global bulkhead."""
    with get_bulkhead().acquire(partition_name, timeout_ms):
        yield


@asynccontextmanager
async def acquire_async(partition_name: str, timeout_ms: Optional[float] = None):
    """Acquire from global bulkhead (async)."""
    async with get_bulkhead().acquire_async(partition_name, timeout_ms):
        yield
