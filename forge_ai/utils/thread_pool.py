"""
Thread Pool - Efficient thread reuse.

Provides thread pool management for:
- Concurrent task execution
- Worker thread lifecycle
- Task queuing and prioritization
- Graceful shutdown
- Resource monitoring

Part of the ForgeAI concurrency utilities.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, TypeVar
from datetime import datetime
from queue import PriorityQueue, Empty
from enum import Enum
import atexit


T = TypeVar('T')


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get execution duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None


@dataclass(order=True)
class Task:
    """A task to execute in the thread pool."""
    priority: int
    task_id: str = field(compare=False)
    func: Callable = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: Dict[str, Any] = field(default_factory=dict, compare=False)
    callback: Optional[Callable[[TaskResult], None]] = field(default=None, compare=False)
    submitted_at: datetime = field(default_factory=datetime.now, compare=False)


class ThreadPoolManager:
    """
    Managed thread pool with monitoring and graceful shutdown.
    
    Usage:
        # Create pool
        pool = ThreadPoolManager(max_workers=4)
        
        # Submit tasks
        future = pool.submit(my_function, arg1, arg2)
        result = future.result()
        
        # With priority
        pool.submit_priority(TaskPriority.HIGH, my_function, arg1)
        
        # With callback
        pool.submit_with_callback(
            my_function,
            args=(arg1,),
            callback=lambda result: print(f"Done: {result}")
        )
        
        # Map over iterable
        results = pool.map(process_item, items)
        
        # Shutdown
        pool.shutdown()
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        thread_name_prefix: str = "forge-worker",
        auto_shutdown: bool = True
    ):
        """
        Initialize thread pool.
        
        Args:
            max_workers: Maximum worker threads (defaults to CPU count)
            thread_name_prefix: Prefix for worker thread names
            auto_shutdown: Register atexit handler for cleanup
        """
        self._max_workers = max_workers
        self._thread_name_prefix = thread_name_prefix
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._task_counter = 0
        
        # Task tracking
        self._pending_tasks: Dict[str, Task] = {}
        self._task_results: Dict[str, TaskResult] = {}
        self._results_limit = 1000
        
        # Statistics
        self._total_submitted = 0
        self._total_completed = 0
        self._total_failed = 0
        
        # Callbacks
        self._on_task_complete: Optional[Callable[[TaskResult], None]] = None
        self._on_task_error: Optional[Callable[[TaskResult], None]] = None
        
        # Initialize executor
        self._start_executor()
        
        # Register cleanup
        if auto_shutdown:
            atexit.register(self.shutdown)
    
    def _start_executor(self):
        """Start the thread pool executor."""
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix=self._thread_name_prefix
                )
    
    def _get_task_id(self) -> str:
        """Generate a unique task ID."""
        with self._lock:
            self._task_counter += 1
            return f"task-{self._task_counter}"
    
    def submit(
        self,
        fn: Callable[..., T],
        *args,
        **kwargs
    ) -> Future:
        """
        Submit a task to the pool.
        
        Args:
            fn: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Future for the task result
        """
        task_id = self._get_task_id()
        
        def wrapped():
            result = TaskResult(
                task_id=task_id,
                status=TaskStatus.RUNNING,
                started_at=datetime.now()
            )
            try:
                result.result = fn(*args, **kwargs)
                result.status = TaskStatus.COMPLETED
                self._total_completed += 1
                if self._on_task_complete:
                    self._on_task_complete(result)
            except Exception as e:
                result.error = str(e)
                result.status = TaskStatus.FAILED
                self._total_failed += 1
                if self._on_task_error:
                    self._on_task_error(result)
            finally:
                result.completed_at = datetime.now()
                self._record_result(result)
            
            return result.result
        
        with self._lock:
            self._total_submitted += 1
            if self._executor is None:
                raise RuntimeError("Thread pool not started")
            return self._executor.submit(wrapped)
    
    def submit_priority(
        self,
        priority: TaskPriority,
        fn: Callable[..., T],
        *args,
        **kwargs
    ) -> Future:
        """
        Submit a task with priority.
        
        Note: Python's ThreadPoolExecutor doesn't natively support priority.
        This is a wrapper that could be extended with a priority queue.
        
        Args:
            priority: Task priority
            fn: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Future for the task result
        """
        # For now, priority is informational
        # Could be extended with a custom priority queue implementation
        return self.submit(fn, *args, **kwargs)
    
    def submit_with_callback(
        self,
        fn: Callable[..., T],
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[TaskResult], None]] = None
    ) -> str:
        """
        Submit a task with completion callback.
        
        Args:
            fn: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            callback: Called when task completes
            
        Returns:
            Task ID
        """
        task_id = self._get_task_id()
        kwargs = kwargs or {}
        
        def wrapped():
            result = TaskResult(
                task_id=task_id,
                status=TaskStatus.RUNNING,
                started_at=datetime.now()
            )
            try:
                result.result = fn(*args, **kwargs)
                result.status = TaskStatus.COMPLETED
                self._total_completed += 1
            except Exception as e:
                result.error = str(e)
                result.status = TaskStatus.FAILED
                self._total_failed += 1
            finally:
                result.completed_at = datetime.now()
                self._record_result(result)
                
                if callback:
                    try:
                        callback(result)
                    except Exception:
                        pass
            
            return result.result
        
        with self._lock:
            self._total_submitted += 1
            if self._executor is not None:
                self._executor.submit(wrapped)
        
        return task_id
    
    def map(
        self,
        fn: Callable[[T], Any],
        iterable: List[T],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Map a function over an iterable using the pool.
        
        Args:
            fn: Function to apply
            iterable: Items to process
            timeout: Optional timeout in seconds
            
        Returns:
            List of results
        """
        futures = [self.submit(fn, item) for item in iterable]
        results = []
        
        for future in as_completed(futures, timeout=timeout):
            try:
                results.append(future.result())
            except Exception as e:
                results.append(None)
        
        return results
    
    def _record_result(self, result: TaskResult):
        """Record a task result."""
        with self._lock:
            self._task_results[result.task_id] = result
            
            # Trim old results
            if len(self._task_results) > self._results_limit:
                oldest = sorted(
                    self._task_results.keys(),
                    key=lambda k: self._task_results[k].completed_at or datetime.min
                )[:len(self._task_results) - self._results_limit]
                for key in oldest:
                    del self._task_results[key]
    
    def get_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result for a task ID."""
        return self._task_results.get(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "max_workers": self._max_workers,
                "total_submitted": self._total_submitted,
                "total_completed": self._total_completed,
                "total_failed": self._total_failed,
                "pending_results": len(self._task_results),
                "success_rate": (
                    self._total_completed / self._total_submitted
                    if self._total_submitted > 0 else 0.0
                )
            }
    
    def on_complete(self, callback: Callable[[TaskResult], None]):
        """Set callback for task completion."""
        self._on_task_complete = callback
    
    def on_error(self, callback: Callable[[TaskResult], None]):
        """Set callback for task errors."""
        self._on_task_error = callback
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """
        Shutdown the thread pool.
        
        Args:
            wait: Wait for pending tasks
            timeout: Shutdown timeout in seconds
        """
        with self._lock:
            if self._executor:
                self._executor.shutdown(wait=wait)
                self._executor = None
    
    def is_running(self) -> bool:
        """Check if pool is running."""
        return self._executor is not None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False


class WorkerPool:
    """
    Custom worker pool with priority queue.
    
    Usage:
        pool = WorkerPool(workers=4)
        pool.start()
        
        # Submit with priority
        pool.submit(TaskPriority.HIGH, my_function, arg1)
        
        pool.stop()
    """
    
    def __init__(self, workers: int = 4):
        """
        Initialize worker pool.
        
        Args:
            workers: Number of worker threads
        """
        self._num_workers = workers
        self._queue: PriorityQueue = PriorityQueue()
        self._workers: List[threading.Thread] = []
        self._running = False
        self._lock = threading.Lock()
        self._task_counter = 0
    
    def start(self):
        """Start worker threads."""
        if self._running:
            return
        
        self._running = True
        
        for i in range(self._num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"worker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
    
    def stop(self, timeout: float = 5.0):
        """Stop worker threads."""
        self._running = False
        
        # Send stop signals
        for _ in self._workers:
            self._queue.put((0, None))  # Sentinel
        
        # Wait for workers
        for worker in self._workers:
            worker.join(timeout=timeout)
        
        self._workers.clear()
    
    def _worker_loop(self):
        """Worker thread main loop."""
        while self._running:
            try:
                priority, task = self._queue.get(timeout=0.5)
                
                if task is None:  # Sentinel
                    break
                
                try:
                    result = task.func(*task.args, **task.kwargs)
                    if task.callback:
                        task_result = TaskResult(
                            task_id=task.task_id,
                            status=TaskStatus.COMPLETED,
                            result=result,
                            completed_at=datetime.now()
                        )
                        task.callback(task_result)
                except Exception as e:
                    if task.callback:
                        task_result = TaskResult(
                            task_id=task.task_id,
                            status=TaskStatus.FAILED,
                            error=str(e),
                            completed_at=datetime.now()
                        )
                        task.callback(task_result)
                
            except Empty:
                continue
    
    def submit(
        self,
        priority: TaskPriority,
        fn: Callable,
        *args,
        callback: Optional[Callable[[TaskResult], None]] = None,
        **kwargs
    ) -> str:
        """
        Submit a task with priority.
        
        Args:
            priority: Task priority
            fn: Function to execute
            *args: Positional arguments
            callback: Completion callback
            **kwargs: Keyword arguments
            
        Returns:
            Task ID
        """
        with self._lock:
            self._task_counter += 1
            task_id = f"task-{self._task_counter}"
        
        task = Task(
            priority=priority.value,
            task_id=task_id,
            func=fn,
            args=args,
            kwargs=kwargs,
            callback=callback
        )
        
        self._queue.put((priority.value, task))
        return task_id
    
    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# Global thread pool
_global_pool: Optional[ThreadPoolManager] = None


def get_thread_pool(max_workers: Optional[int] = None) -> ThreadPoolManager:
    """Get the global thread pool."""
    global _global_pool
    if _global_pool is None:
        _global_pool = ThreadPoolManager(max_workers=max_workers)
    return _global_pool


def submit_task(fn: Callable, *args, **kwargs) -> Future:
    """Submit task to global pool."""
    return get_thread_pool().submit(fn, *args, **kwargs)


def parallel_map(fn: Callable, iterable: List) -> List:
    """Parallel map using global pool."""
    return get_thread_pool().map(fn, iterable)


def shutdown_pool():
    """Shutdown global pool."""
    global _global_pool
    if _global_pool:
        _global_pool.shutdown()
        _global_pool = None
