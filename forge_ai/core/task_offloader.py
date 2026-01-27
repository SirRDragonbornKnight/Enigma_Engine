"""
================================================================================
TASK OFFLOADER - ASYNCHRONOUS TASK EXECUTION SYSTEM
================================================================================

The Task Offloader enables asynchronous and parallel task execution for the
Model Orchestrator. It allows tasks to be queued and processed in the background,
improving responsiveness and enabling concurrent model execution.

FILE: forge_ai/core/task_offloader.py
TYPE: Async Task Management
MAIN CLASS: TaskOffloader

KEY FEATURES:
- Async task queue with priority support
- Background worker threads for parallel execution
- Progress tracking and callbacks
- Task cancellation support
- Resource-aware scheduling
- Results caching

USAGE:
    from forge_ai.core.task_offloader import TaskOffloader, get_offloader
    
    offloader = get_offloader()
    
    # Submit a task for async execution
    task_id = offloader.submit_task(
        capability="code_generation",
        task="Write a Python function to sort a list",
        priority=5,
        callback=lambda result: print(f"Done: {result}")
    )
    
    # Check task status
    status = offloader.get_task_status(task_id)
    
    # Wait for result
    result = offloader.wait_for_task(task_id, timeout=30)
    
    # Cancel a task
    offloader.cancel_task(task_id)
"""

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# TASK STATUS
# =============================================================================

class TaskStatus(Enum):
    """Status of a task in the offloader."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# OFFLOADED TASK
# =============================================================================

@dataclass
class OffloadedTask:
    """A task submitted for async execution."""
    
    task_id: str
    capability: str
    task: Any
    context: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    model_id: Optional[str] = None
    priority: int = 5
    callback: Optional[Callable[[Any], None]] = None
    error_callback: Optional[Callable[[Exception], None]] = None
    
    # Status tracking
    status: TaskStatus = TaskStatus.PENDING
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Results
    result: Any = None
    error: Optional[Exception] = None
    
    # Metadata
    worker_id: Optional[int] = None
    processing_time: float = 0.0
    
    def mark_running(self, worker_id: int) -> None:
        """Mark task as running."""
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()
        self.worker_id = worker_id
    
    def mark_completed(self, result: Any) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = time.time()
        self.result = result
        if self.started_at:
            self.processing_time = self.completed_at - self.started_at
    
    def mark_failed(self, error: Exception) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = time.time()
        self.error = error
        if self.started_at:
            self.processing_time = self.completed_at - self.started_at
    
    def mark_cancelled(self) -> None:
        """Mark task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = time.time()
        if self.started_at:
            self.processing_time = self.completed_at - self.started_at


# =============================================================================
# TASK OFFLOADER CONFIGURATION
# =============================================================================

@dataclass
class OffloaderConfig:
    """Configuration for the task offloader."""
    
    # Number of worker threads
    num_workers: int = 2
    
    # Maximum queue size (0 = unlimited)
    max_queue_size: int = 100
    
    # Enable task prioritization
    enable_prioritization: bool = True
    
    # Keep completed task history
    keep_history: bool = True
    
    # Maximum history size
    max_history_size: int = 1000
    
    # Auto-cleanup completed tasks after N seconds
    auto_cleanup_seconds: float = 300.0
    
    # Enable result caching
    enable_caching: bool = False
    
    # Cache TTL in seconds
    cache_ttl: float = 3600.0


# =============================================================================
# WORKER THREAD
# =============================================================================

class WorkerThread(threading.Thread):
    """Background worker thread for executing tasks."""
    
    def __init__(
        self,
        worker_id: int,
        task_queue: queue.PriorityQueue,
        offloader: "TaskOffloader",
    ):
        """
        Initialize worker thread.
        
        Args:
            worker_id: Unique worker identifier
            task_queue: Shared task queue
            offloader: Parent offloader instance
        """
        super().__init__(daemon=True, name=f"TaskWorker-{worker_id}")
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.offloader = offloader
        self._stop_event = threading.Event()
        self._current_task: Optional[OffloadedTask] = None
    
    def run(self) -> None:
        """Main worker loop."""
        logger.info(f"Worker {self.worker_id} started")
        
        while not self._stop_event.is_set():
            try:
                # Get next task (with timeout to check stop event)
                try:
                    priority, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check if cancelled before starting
                if task.status == TaskStatus.CANCELLED:
                    self.task_queue.task_done()
                    continue
                
                # Mark as running
                self._current_task = task
                task.mark_running(self.worker_id)
                
                logger.debug(
                    f"Worker {self.worker_id} executing task {task.task_id} "
                    f"(priority {task.priority})"
                )
                
                # Execute the task
                try:
                    result = self.offloader._execute_task_impl(task)
                    task.mark_completed(result)
                    
                    # Call success callback
                    if task.callback:
                        try:
                            task.callback(result)
                        except Exception as e:
                            logger.error(f"Task callback failed: {e}")
                
                except Exception as e:
                    logger.error(
                        f"Worker {self.worker_id} task {task.task_id} failed: {e}"
                    )
                    task.mark_failed(e)
                    
                    # Call error callback
                    if task.error_callback:
                        try:
                            task.error_callback(e)
                        except Exception as cb_error:
                            logger.error(f"Error callback failed: {cb_error}")
                
                finally:
                    self._current_task = None
                    self.task_queue.task_done()
            
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    def stop(self) -> None:
        """Stop the worker thread."""
        self._stop_event.set()
    
    def get_current_task(self) -> Optional[OffloadedTask]:
        """Get the currently executing task."""
        return self._current_task


# =============================================================================
# TASK OFFLOADER
# =============================================================================

class TaskOffloader:
    """
    Asynchronous task execution system for the Model Orchestrator.
    
    Features:
    - Background worker threads for parallel execution
    - Priority-based task queue
    - Progress tracking and callbacks
    - Task cancellation support
    - Results caching
    """
    
    def __init__(
        self,
        config: Optional[OffloaderConfig] = None,
        orchestrator: Optional[Any] = None,
    ):
        """
        Initialize the task offloader.
        
        Args:
            config: Offloader configuration
            orchestrator: Model orchestrator instance for task execution
        """
        self.config = config or OffloaderConfig()
        self.orchestrator = orchestrator
        
        # Task queue (priority, task)
        max_size = self.config.max_queue_size if self.config.max_queue_size > 0 else 0
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_size)
        
        # Task tracking
        self._tasks: Dict[str, OffloadedTask] = {}
        self._task_lock = threading.RLock()
        
        # Result cache
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        # Workers
        self._workers: List[WorkerThread] = []
        self._start_workers()
        
        # Cleanup thread
        if self.config.keep_history and self.config.auto_cleanup_seconds > 0:
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True,
                name="TaskCleanup",
            )
            self._cleanup_thread.start()
        else:
            self._cleanup_thread = None
        
        # Statistics
        self._stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_cancelled": 0,
        }
        self._stats_lock = threading.Lock()
    
    def set_orchestrator(self, orchestrator: Any) -> None:
        """
        Set the orchestrator instance.
        
        Args:
            orchestrator: Model orchestrator for task execution
        """
        self.orchestrator = orchestrator
    
    # -------------------------------------------------------------------------
    # TASK SUBMISSION
    # -------------------------------------------------------------------------
    
    def submit_task(
        self,
        capability: str,
        task: Any = None,
        context: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        priority: int = 5,
        callback: Optional[Callable[[Any], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None,
        **kwargs,
    ) -> str:
        """
        Submit a task for async execution.
        
        Args:
            capability: Required capability
            task: Task data
            context: Optional context
            parameters: Optional parameters
            model_id: Optional specific model to use
            priority: Task priority (0=highest, 10=lowest)
            callback: Optional callback on success
            error_callback: Optional callback on error
            **kwargs: Additional arguments
            
        Returns:
            Task ID for tracking
        """
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Create task object
        offloaded_task = OffloadedTask(
            task_id=task_id,
            capability=capability,
            task=task or kwargs,
            context=context or {},
            parameters=parameters or {},
            model_id=model_id,
            priority=priority,
            callback=callback,
            error_callback=error_callback,
        )
        
        # Store task
        with self._task_lock:
            self._tasks[task_id] = offloaded_task
        
        # Add to queue
        # Note: PriorityQueue uses lower numbers = higher priority
        self._task_queue.put((priority, offloaded_task))
        
        # Update stats
        with self._stats_lock:
            self._stats["total_submitted"] += 1
        
        logger.debug(f"Submitted task {task_id} (priority {priority})")
        return task_id
    
    # -------------------------------------------------------------------------
    # TASK STATUS & RESULTS
    # -------------------------------------------------------------------------
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get the status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status or None if not found
        """
        with self._task_lock:
            task = self._tasks.get(task_id)
            return task.status if task else None
    
    def get_task(self, task_id: str) -> Optional[OffloadedTask]:
        """
        Get full task object.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task object or None if not found
        """
        with self._task_lock:
            return self._tasks.get(task_id)
    
    def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Wait for a task to complete and return the result.
        
        Args:
            task_id: Task identifier
            timeout: Maximum time to wait in seconds (None = wait forever)
            
        Returns:
            Task result
            
        Raises:
            TimeoutError: If timeout exceeded
            RuntimeError: If task failed or was cancelled
            KeyError: If task not found
        """
        start_time = time.time()
        
        while True:
            with self._task_lock:
                task = self._tasks.get(task_id)
                if not task:
                    raise KeyError(f"Task {task_id} not found")
                
                if task.status == TaskStatus.COMPLETED:
                    return task.result
                elif task.status == TaskStatus.FAILED:
                    raise RuntimeError(f"Task failed: {task.error}")
                elif task.status == TaskStatus.CANCELLED:
                    raise RuntimeError("Task was cancelled")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete in {timeout}s")
            
            # Sleep briefly before checking again
            time.sleep(0.1)
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was cancelled, False otherwise
        """
        with self._task_lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return False
            
            if task.status == TaskStatus.CANCELLED:
                return True
            
            task.mark_cancelled()
            
            # Update stats
            with self._stats_lock:
                self._stats["total_cancelled"] += 1
            
            logger.debug(f"Cancelled task {task_id}")
            return True
    
    # -------------------------------------------------------------------------
    # QUEUE MANAGEMENT
    # -------------------------------------------------------------------------
    
    def get_queue_size(self) -> int:
        """Get the number of pending tasks in the queue."""
        return self._task_queue.qsize()
    
    def get_pending_tasks(self) -> List[str]:
        """Get list of pending task IDs."""
        with self._task_lock:
            return [
                task_id
                for task_id, task in self._tasks.items()
                if task.status == TaskStatus.PENDING
            ]
    
    def get_running_tasks(self) -> List[str]:
        """Get list of running task IDs."""
        with self._task_lock:
            return [
                task_id
                for task_id, task in self._tasks.items()
                if task.status == TaskStatus.RUNNING
            ]
    
    def clear_queue(self) -> int:
        """
        Clear all pending tasks from the queue.
        
        Returns:
            Number of tasks cancelled
        """
        count = 0
        pending_ids = self.get_pending_tasks()
        
        for task_id in pending_ids:
            if self.cancel_task(task_id):
                count += 1
        
        logger.info(f"Cleared {count} pending tasks")
        return count
    
    # -------------------------------------------------------------------------
    # WORKER MANAGEMENT
    # -------------------------------------------------------------------------
    
    def _start_workers(self) -> None:
        """Start worker threads."""
        for i in range(self.config.num_workers):
            worker = WorkerThread(i, self._task_queue, self)
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started {self.config.num_workers} worker threads")
    
    def stop_workers(self) -> None:
        """Stop all worker threads."""
        logger.info("Stopping workers...")
        
        for worker in self._workers:
            worker.stop()
        
        for worker in self._workers:
            worker.join(timeout=5.0)
        
        self._workers.clear()
        logger.info("Workers stopped")
    
    # -------------------------------------------------------------------------
    # TASK EXECUTION
    # -------------------------------------------------------------------------
    
    def _execute_task_impl(self, task: OffloadedTask) -> Any:
        """
        Execute a task using the orchestrator.
        
        Args:
            task: Task to execute
            
        Returns:
            Task result
        """
        if not self.orchestrator:
            raise RuntimeError("No orchestrator set for task execution")
        
        # Check cache if enabled
        if self.config.enable_caching:
            cache_key = self._get_cache_key(task)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                logger.debug(f"Task {task.task_id} result from cache")
                return cached
        
        # Execute via orchestrator
        result = self.orchestrator.execute_task(
            capability=task.capability,
            task=task.task,
            context=task.context,
            parameters=task.parameters,
            model_id=task.model_id,
        )
        
        # Cache result if enabled
        if self.config.enable_caching:
            cache_key = self._get_cache_key(task)
            self._put_in_cache(cache_key, result)
        
        # Update stats
        with self._stats_lock:
            self._stats["total_completed"] += 1
        
        return result
    
    # -------------------------------------------------------------------------
    # CACHING
    # -------------------------------------------------------------------------
    
    def _get_cache_key(self, task: OffloadedTask) -> str:
        """Generate cache key for a task."""
        import hashlib
        import json
        
        # Create deterministic string from task
        cache_data = {
            "capability": task.capability,
            "task": str(task.task),
            "model_id": task.model_id,
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get result from cache if not expired."""
        if key not in self._cache:
            return None
        
        # Check TTL
        timestamp = self._cache_timestamps.get(key, 0)
        if time.time() - timestamp > self.config.cache_ttl:
            # Expired
            del self._cache[key]
            del self._cache_timestamps[key]
            return None
        
        return self._cache[key]
    
    def _put_in_cache(self, key: str, value: Any) -> None:
        """Put result in cache."""
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.debug("Cleared result cache")
    
    # -------------------------------------------------------------------------
    # CLEANUP
    # -------------------------------------------------------------------------
    
    def _cleanup_loop(self) -> None:
        """Background thread for cleaning up old completed tasks."""
        while True:
            time.sleep(self.config.auto_cleanup_seconds)
            self._cleanup_old_tasks()
    
    def _cleanup_old_tasks(self) -> None:
        """Remove old completed/failed/cancelled tasks from history."""
        cutoff_time = time.time() - self.config.auto_cleanup_seconds
        
        with self._task_lock:
            to_remove = []
            
            for task_id, task in self._tasks.items():
                if task.status in [
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                ]:
                    if task.completed_at and task.completed_at < cutoff_time:
                        to_remove.append(task_id)
            
            for task_id in to_remove:
                del self._tasks[task_id]
            
            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove)} old tasks")
            
            # Also enforce max history size
            if len(self._tasks) > self.config.max_history_size:
                # Remove oldest completed tasks
                completed_tasks = [
                    (task_id, task)
                    for task_id, task in self._tasks.items()
                    if task.status in [
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                        TaskStatus.CANCELLED,
                    ]
                ]
                completed_tasks.sort(key=lambda x: x[1].completed_at or 0)
                
                num_to_remove = len(self._tasks) - self.config.max_history_size
                for i in range(num_to_remove):
                    task_id = completed_tasks[i][0]
                    del self._tasks[task_id]
    
    # -------------------------------------------------------------------------
    # STATUS & STATISTICS
    # -------------------------------------------------------------------------
    
    def get_status(self) -> Dict[str, Any]:
        """Get offloader status and statistics."""
        with self._task_lock:
            pending = sum(
                1 for t in self._tasks.values() if t.status == TaskStatus.PENDING
            )
            running = sum(
                1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING
            )
            completed = sum(
                1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED
            )
            failed = sum(
                1 for t in self._tasks.values() if t.status == TaskStatus.FAILED
            )
            cancelled = sum(
                1 for t in self._tasks.values() if t.status == TaskStatus.CANCELLED
            )
        
        with self._stats_lock:
            stats = self._stats.copy()
        
        # Get worker info
        worker_info = []
        for worker in self._workers:
            current_task = worker.get_current_task()
            worker_info.append({
                "worker_id": worker.worker_id,
                "is_alive": worker.is_alive(),
                "current_task_id": current_task.task_id if current_task else None,
            })
        
        return {
            "num_workers": len(self._workers),
            "worker_info": worker_info,
            "queue_size": self.get_queue_size(),
            "tasks": {
                "pending": pending,
                "running": running,
                "completed": completed,
                "failed": failed,
                "cancelled": cancelled,
                "total": len(self._tasks),
            },
            "cache_size": len(self._cache),
            "statistics": stats,
        }
    
    # -------------------------------------------------------------------------
    # SHUTDOWN
    # -------------------------------------------------------------------------
    
    def shutdown(self, wait: bool = True, timeout: float = 10.0) -> None:
        """
        Shutdown the offloader and stop all workers.
        
        Args:
            wait: Wait for current tasks to complete
            timeout: Maximum time to wait in seconds
        """
        logger.info("Shutting down task offloader...")
        
        if wait:
            # Wait for queue to drain
            start_time = time.time()
            while not self._task_queue.empty():
                if time.time() - start_time > timeout:
                    logger.warning("Shutdown timeout reached, forcing stop")
                    break
                time.sleep(0.1)
        
        # Stop workers
        self.stop_workers()
        
        logger.info("Task offloader shutdown complete")


# =============================================================================
# GLOBAL OFFLOADER INSTANCE
# =============================================================================

_global_offloader: Optional[TaskOffloader] = None


def get_offloader(
    config: Optional[OffloaderConfig] = None,
    orchestrator: Optional[Any] = None,
) -> TaskOffloader:
    """
    Get the global task offloader instance.
    
    Args:
        config: Optional offloader configuration
        orchestrator: Optional orchestrator instance
        
    Returns:
        Global TaskOffloader instance
    """
    global _global_offloader
    if _global_offloader is None:
        # Load config from CONFIG if not provided
        if config is None:
            try:
                from ..config import CONFIG
                config = OffloaderConfig(
                    num_workers=CONFIG.get("offloader_num_workers", 2),
                    max_queue_size=CONFIG.get("offloader_max_queue_size", 100),
                    enable_caching=CONFIG.get("offloader_enable_caching", False),
                )
            except Exception:
                config = OffloaderConfig()
        
        _global_offloader = TaskOffloader(config, orchestrator)
    
    return _global_offloader
