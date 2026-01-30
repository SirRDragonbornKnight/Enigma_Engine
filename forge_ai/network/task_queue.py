"""
Network Task Queue - Queue tasks for remote execution

Manages a queue of tasks that need to be sent to remote servers,
with priority handling, retry logic, and result caching.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1    # Immediate execution
    HIGH = 2        # High priority
    NORMAL = 5      # Default priority
    LOW = 8         # Background task
    IDLE = 10       # Run when nothing else is waiting


@dataclass
class NetworkTask:
    """A task to be executed on a remote server."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Task definition
    capability: str = ""              # e.g., "text_generation", "image_generation"
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    
    # Routing
    target_server: Optional[str] = None   # Specific server, or None for auto-select
    allow_local_fallback: bool = True     # Fall back to local if remote fails
    
    # Retry settings
    max_retries: int = 3
    retry_delay_s: float = 1.0
    timeout_s: float = 60.0
    
    # Callbacks
    on_complete: Optional[Callable[[Any], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None
    on_progress: Optional[Callable[[float], None]] = None
    
    # State
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None
    attempts: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def __lt__(self, other):
        """For priority queue ordering."""
        if isinstance(other, NetworkTask):
            return self.priority.value < other.priority.value
        return NotImplemented


class NetworkTaskQueue:
    """
    Queue for network tasks with priority and worker management.
    
    Features:
    - Priority-based execution
    - Configurable worker count
    - Automatic retry on failure
    - Result caching
    - Progress tracking
    """
    
    def __init__(
        self,
        num_workers: int = 2,
        max_queue_size: int = 100
    ):
        """
        Initialize task queue.
        
        Args:
            num_workers: Number of worker threads
            max_queue_size: Maximum queue size
        """
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self._tasks: Dict[str, NetworkTask] = {}
        self._results: Dict[str, Any] = {}
        self._workers: List[threading.Thread] = []
        self._running = False
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "retried": 0,
        }
    
    def start(self):
        """Start the worker threads."""
        if self._running:
            return
        
        self._running = True
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
                name=f"NetworkTaskWorker-{i}"
            )
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started {self.num_workers} network task workers")
    
    def stop(self):
        """Stop the worker threads."""
        self._running = False
        
        # Clear queue to unblock workers
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=2.0)
        
        self._workers.clear()
        logger.info("Stopped network task workers")
    
    def submit(self, task: NetworkTask) -> str:
        """
        Submit a task to the queue.
        
        Args:
            task: Task to submit
            
        Returns:
            Task ID
        """
        with self._lock:
            self._tasks[task.task_id] = task
            self._stats["submitted"] += 1
        
        try:
            self._queue.put_nowait((task.priority.value, task))
            logger.debug(f"Submitted task {task.task_id}: {task.capability}")
        except queue.Full:
            task.status = "rejected"
            task.error = "Queue full"
            logger.warning(f"Task queue full, rejected task {task.task_id}")
        
        return task.task_id
    
    def get_task(self, task_id: str) -> Optional[NetworkTask]:
        """Get task by ID."""
        with self._lock:
            return self._tasks.get(task_id)
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get result for a task, waiting if necessary.
        
        Args:
            task_id: Task ID
            timeout: Maximum time to wait (None = no wait)
            
        Returns:
            Task result
            
        Raises:
            TimeoutError: If timeout reached
            RuntimeError: If task failed
        """
        start = time.time()
        
        while True:
            with self._lock:
                task = self._tasks.get(task_id)
                if task is None:
                    raise KeyError(f"Unknown task: {task_id}")
                
                if task.status == "completed":
                    return task.result
                
                if task.status in ("failed", "rejected"):
                    raise RuntimeError(f"Task failed: {task.error}")
            
            if timeout is not None and time.time() - start > timeout:
                raise TimeoutError(f"Task {task_id} did not complete in time")
            
            time.sleep(0.1)
    
    def cancel(self, task_id: str) -> bool:
        """
        Cancel a pending task.
        
        Returns:
            True if task was cancelled
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == "pending":
                task.status = "cancelled"
                return True
        return False
    
    def _worker_loop(self, worker_id: int):
        """Worker thread main loop."""
        logger.debug(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # Get next task
                try:
                    _, task = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Skip cancelled tasks
                if task.status == "cancelled":
                    continue
                
                # Execute task
                self._execute_task(task, worker_id)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.debug(f"Worker {worker_id} stopped")
    
    def _execute_task(self, task: NetworkTask, worker_id: int):
        """Execute a single task."""
        task.status = "running"
        task.started_at = time.time()
        task.attempts += 1
        
        try:
            # Import here to avoid circular imports
            from .load_balancer import LoadBalancer, BalancingStrategy
            from ..comms.remote_client import RemoteClient
            
            # Get server (use specific or load balance)
            if task.target_server:
                address, port = task.target_server.split(":")
                port = int(port)
            else:
                # Auto server selection using load balancer
                if not hasattr(self, '_load_balancer') or self._load_balancer is None:
                    self._load_balancer = LoadBalancer(strategy=BalancingStrategy.ADAPTIVE)
                
                # Check if we have servers registered
                server = self._load_balancer.get_server()
                if server is None:
                    # Try to discover servers from network discovery
                    try:
                        from .discovery import ServiceDiscovery
                        discovery = ServiceDiscovery()
                        services = discovery.find_services("forge_inference")
                        for svc in services[:5]:  # Add up to 5 discovered servers
                            self._load_balancer.add_server(svc.address, svc.port)
                        server = self._load_balancer.get_server()
                    except Exception:
                        pass
                
                if server is None:
                    # Fall back to localhost default
                    address = "localhost"
                    port = 5000
                    logger.warning(f"No servers available, falling back to {address}:{port}")
                else:
                    address = server.address
                    port = server.port
                    self._load_balancer.mark_request_start(server)
            
            # Create client and execute
            client = RemoteClient(address, port)
            
            # Route based on capability
            if task.capability == "text_generation":
                result = client.generate(
                    task.payload.get("prompt", ""),
                    max_tokens=task.payload.get("max_tokens", 100)
                )
            elif task.capability == "chat":
                result = client.chat(
                    task.payload.get("message", ""),
                    history=task.payload.get("history", [])
                )
            else:
                raise ValueError(f"Unknown capability: {task.capability}")
            
            # Success
            task.result = result
            task.status = "completed"
            task.completed_at = time.time()
            
            with self._lock:
                self._stats["completed"] += 1
            
            if task.on_complete:
                try:
                    task.on_complete(result)
                except Exception as e:
                    logger.error(f"Task callback error: {e}")
            
            logger.debug(f"Task {task.task_id} completed")
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # Retry logic
            if task.attempts < task.max_retries:
                task.status = "pending"
                time.sleep(task.retry_delay_s)
                self._queue.put_nowait((task.priority.value, task))
                
                with self._lock:
                    self._stats["retried"] += 1
                
                logger.debug(f"Retrying task {task.task_id} (attempt {task.attempts + 1})")
            else:
                task.status = "failed"
                task.error = str(e)
                task.completed_at = time.time()
                
                with self._lock:
                    self._stats["failed"] += 1
                
                if task.on_error:
                    try:
                        task.on_error(e)
                    except Exception:
                        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                **self._stats,
                "queue_size": self._queue.qsize(),
                "active_tasks": sum(
                    1 for t in self._tasks.values()
                    if t.status == "running"
                ),
            }
