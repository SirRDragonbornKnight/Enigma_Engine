"""
Request Queue System

Priority-based request queuing with rate limiting and fair scheduling.
Handles concurrent inference requests with configurable policies.

FILE: enigma_engine/comms/request_queue.py
TYPE: Communication/Infrastructure
MAIN CLASSES: RequestQueue, PriorityScheduler, RateLimiter
"""

import asyncio
import heapq
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueuePriority(Enum):
    """Request priority levels."""
    SYSTEM = 0  # System-level requests
    REALTIME = 1  # Real-time user interactions
    HIGH = 2  # Premium/paid users
    NORMAL = 3  # Standard users
    LOW = 4  # Background tasks
    BATCH = 5  # Batch processing


class RequestState(Enum):
    """Request state."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class QueuedRequest:
    """A queued request."""
    id: str
    payload: Any
    priority: QueuePriority = QueuePriority.NORMAL
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # State
    state: RequestState = RequestState.PENDING
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    
    # Callbacks
    on_complete: Optional[Callable[[Any], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None
    
    # Retry
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """Priority queue ordering."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        
        # Earlier deadline first
        if self.deadline and other.deadline:
            return self.deadline < other.deadline
        if self.deadline:
            return True
        if other.deadline:
            return False
        
        # FIFO for same priority
        return self.created_at < other.created_at
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class QueueConfig:
    """Queue configuration."""
    max_queue_size: int = 10000
    default_timeout: float = 30.0
    
    # Rate limiting
    enable_rate_limiting: bool = True
    requests_per_second: float = 100.0
    burst_size: int = 50
    
    # Per-user limits
    max_requests_per_user: int = 10
    user_rate_limit: float = 5.0  # requests per second per user
    
    # Fair scheduling
    enable_fair_scheduling: bool = True
    fair_share_interval: float = 1.0
    
    # Workers
    num_workers: int = 4
    worker_timeout: float = 60.0


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(
        self,
        rate: float,  # tokens per second
        burst: int  # max tokens
    ):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self, tokens: int = 1, block: bool = True) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            block: Whether to wait for tokens
        
        Returns:
            True if tokens acquired
        """
        with self._lock:
            now = time.time()
            
            # Refill tokens
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            if block:
                # Wait for tokens
                wait_time = (tokens - self.tokens) / self.rate
                time.sleep(wait_time)
                self.tokens = 0
                self.last_update = time.time()
                return True
            
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait for tokens."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            available = min(self.burst, self.tokens + elapsed * self.rate)
            
            if available >= tokens:
                return 0.0
            
            return (tokens - available) / self.rate


class PerUserRateLimiter:
    """Per-user rate limiting."""
    
    def __init__(self, rate: float, burst: int):
        self.rate = rate
        self.burst = burst
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()
    
    def acquire(self, user_id: str, tokens: int = 1, block: bool = False) -> bool:
        """Acquire tokens for a specific user."""
        with self._lock:
            if user_id not in self._limiters:
                self._limiters[user_id] = RateLimiter(self.rate, self.burst)
        
        return self._limiters[user_id].acquire(tokens, block)
    
    def cleanup(self, max_age_seconds: float = 3600):
        """Remove old user limiters."""
        now = time.time()
        with self._lock:
            to_remove = [
                uid for uid, limiter in self._limiters.items()
                if now - limiter.last_update > max_age_seconds
            ]
            for uid in to_remove:
                del self._limiters[uid]


class FairScheduler:
    """Fair scheduling across users/sessions."""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._user_queues: dict[str, list[QueuedRequest]] = defaultdict(list)
        self._round_robin_index = 0
        self._lock = threading.Lock()
    
    def add(self, request: QueuedRequest):
        """Add request to fair scheduling."""
        user_id = request.user_id or "anonymous"
        with self._lock:
            heapq.heappush(self._user_queues[user_id], request)
    
    def get_next(self) -> Optional[QueuedRequest]:
        """Get next request using round-robin across users."""
        with self._lock:
            if not self._user_queues:
                return None
            
            users = list(self._user_queues.keys())
            if not users:
                return None
            
            # Round-robin through users
            checked = 0
            while checked < len(users):
                user = users[self._round_robin_index % len(users)]
                self._round_robin_index += 1
                checked += 1
                
                if self._user_queues[user]:
                    request = heapq.heappop(self._user_queues[user])
                    
                    # Cleanup empty queues
                    if not self._user_queues[user]:
                        del self._user_queues[user]
                    
                    return request
            
            return None
    
    def get_queue_length(self, user_id: str = None) -> int:
        """Get queue length for a user or total."""
        with self._lock:
            if user_id:
                return len(self._user_queues.get(user_id, []))
            return sum(len(q) for q in self._user_queues.values())


class RequestQueue:
    """
    Priority-based request queue with rate limiting.
    
    Supports multiple concurrent workers, fair scheduling,
    and per-user rate limiting.
    """
    
    def __init__(self, config: QueueConfig = None):
        self.config = config or QueueConfig()
        
        # Request storage
        self._requests: dict[str, QueuedRequest] = {}
        self._priority_queue: list[QueuedRequest] = []
        self._lock = threading.Lock()
        
        # Rate limiting
        if self.config.enable_rate_limiting:
            self._global_limiter = RateLimiter(
                self.config.requests_per_second,
                self.config.burst_size
            )
            self._user_limiter = PerUserRateLimiter(
                self.config.user_rate_limit,
                self.config.burst_size
            )
        else:
            self._global_limiter = None
            self._user_limiter = None
        
        # Fair scheduling
        if self.config.enable_fair_scheduling:
            self._fair_scheduler = FairScheduler(self.config.fair_share_interval)
        else:
            self._fair_scheduler = None
        
        # Workers
        self._workers: list[threading.Thread] = []
        self._running = False
        self._processor: Optional[Callable[[Any], Any]] = None
        
        # Metrics
        self._metrics = {
            "enqueued": 0,
            "completed": 0,
            "failed": 0,
            "timeout": 0,
            "rejected": 0
        }
    
    def enqueue(
        self,
        payload: Any,
        priority: QueuePriority = QueuePriority.NORMAL,
        user_id: str = None,
        session_id: str = None,
        deadline: float = None,
        timeout: float = None,
        on_complete: Callable[[Any], None] = None,
        on_error: Callable[[Exception], None] = None
    ) -> Optional[str]:
        """
        Enqueue a request.
        
        Args:
            payload: Request payload
            priority: Request priority
            user_id: User identifier
            session_id: Session identifier
            deadline: Deadline timestamp
            timeout: Timeout in seconds
            on_complete: Completion callback
            on_error: Error callback
        
        Returns:
            Request ID or None if rejected
        """
        # Check queue size
        if len(self._requests) >= self.config.max_queue_size:
            self._metrics["rejected"] += 1
            logger.warning("Queue full, rejecting request")
            return None
        
        # Check per-user limits
        if user_id:
            user_pending = sum(
                1 for r in self._requests.values()
                if r.user_id == user_id and r.state == RequestState.PENDING
            )
            if user_pending >= self.config.max_requests_per_user:
                self._metrics["rejected"] += 1
                logger.warning(f"User {user_id} at request limit")
                return None
        
        # Rate limiting
        if self._global_limiter and not self._global_limiter.acquire(block=False):
            self._metrics["rejected"] += 1
            logger.warning("Rate limited (global)")
            return None
        
        if user_id and self._user_limiter:
            if not self._user_limiter.acquire(user_id, block=False):
                self._metrics["rejected"] += 1
                logger.warning(f"Rate limited (user: {user_id})")
                return None
        
        # Create request
        request = QueuedRequest(
            id=str(uuid.uuid4()),
            payload=payload,
            priority=priority,
            user_id=user_id,
            session_id=session_id,
            deadline=deadline or (time.time() + timeout if timeout else None),
            on_complete=on_complete,
            on_error=on_error
        )
        
        # Add to queue
        with self._lock:
            self._requests[request.id] = request
            
            if self._fair_scheduler and user_id:
                self._fair_scheduler.add(request)
            else:
                heapq.heappush(self._priority_queue, request)
        
        self._metrics["enqueued"] += 1
        logger.debug(f"Enqueued request {request.id} with priority {priority}")
        
        return request.id
    
    def get_next(self) -> Optional[QueuedRequest]:
        """Get next request to process."""
        with self._lock:
            # Try fair scheduler first
            if self._fair_scheduler:
                request = self._fair_scheduler.get_next()
                if request:
                    return request
            
            # Fall back to priority queue
            while self._priority_queue:
                request = heapq.heappop(self._priority_queue)
                
                # Check if still valid
                if request.state != RequestState.PENDING:
                    continue
                
                # Check deadline
                if request.deadline and time.time() > request.deadline:
                    request.state = RequestState.TIMEOUT
                    self._metrics["timeout"] += 1
                    continue
                
                return request
        
        return None
    
    def get_status(self, request_id: str) -> Optional[RequestState]:
        """Get request status."""
        request = self._requests.get(request_id)
        return request.state if request else None
    
    def get_result(self, request_id: str) -> tuple[Optional[Any], Optional[str]]:
        """Get request result and error."""
        request = self._requests.get(request_id)
        if request:
            return request.result, request.error
        return None, None
    
    def cancel(self, request_id: str) -> bool:
        """Cancel a pending request."""
        request = self._requests.get(request_id)
        if request and request.state == RequestState.PENDING:
            request.state = RequestState.CANCELLED
            return True
        return False
    
    def set_processor(self, processor: Callable[[Any], Any]):
        """Set the request processor function."""
        self._processor = processor
    
    def start_workers(self, num_workers: int = None):
        """Start worker threads."""
        if self._running:
            return
        
        self._running = True
        num = num_workers or self.config.num_workers
        
        for i in range(num):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"QueueWorker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started {num} queue workers")
    
    def stop_workers(self, timeout: float = 5.0):
        """Stop worker threads."""
        self._running = False
        
        for worker in self._workers:
            worker.join(timeout=timeout)
        
        self._workers.clear()
        logger.info("Stopped queue workers")
    
    def _worker_loop(self):
        """Worker processing loop."""
        while self._running:
            request = self.get_next()
            
            if not request:
                time.sleep(0.01)
                continue
            
            if not self._processor:
                logger.warning("No processor set")
                request.state = RequestState.FAILED
                request.error = "No processor"
                continue
            
            # Process request
            request.state = RequestState.PROCESSING
            request.started_at = time.time()
            
            try:
                result = self._processor(request.payload)
                
                request.result = result
                request.state = RequestState.COMPLETED
                request.completed_at = time.time()
                self._metrics["completed"] += 1
                
                # Callback
                if request.on_complete:
                    try:
                        request.on_complete(result)
                    except Exception as e:
                        logger.debug(f"Callback failed for {request.id}: {e}")
            
            except Exception as e:
                logger.error(f"Request {request.id} failed: {e}")
                
                request.error = str(e)
                request.retry_count += 1
                
                if request.retry_count < request.max_retries:
                    # Re-queue for retry
                    request.state = RequestState.PENDING
                    with self._lock:
                        heapq.heappush(self._priority_queue, request)
                else:
                    request.state = RequestState.FAILED
                    self._metrics["failed"] += 1
                    
                    if request.on_error:
                        try:
                            request.on_error(e)
                        except Exception as cb_err:
                            logger.debug(f"Error callback failed for {request.id}: {cb_err}")
    
    def wait_for(
        self,
        request_id: str,
        timeout: float = None
    ) -> tuple[Optional[Any], Optional[str]]:
        """Wait for a request to complete."""
        timeout = timeout or self.config.default_timeout
        start = time.time()
        
        while True:
            request = self._requests.get(request_id)
            
            if not request:
                return None, "Request not found"
            
            if request.state == RequestState.COMPLETED:
                return request.result, None
            
            if request.state in (RequestState.FAILED, RequestState.TIMEOUT, RequestState.CANCELLED):
                return None, request.error or request.state.value
            
            if time.time() - start > timeout:
                return None, "Timeout waiting for result"
            
            time.sleep(0.01)
    
    def process_sync(
        self,
        payload: Any,
        priority: QueuePriority = QueuePriority.NORMAL,
        timeout: float = None,
        **kwargs
    ) -> tuple[Optional[Any], Optional[str]]:
        """Enqueue and wait for result (synchronous)."""
        request_id = self.enqueue(payload, priority, timeout=timeout, **kwargs)
        
        if not request_id:
            return None, "Failed to enqueue"
        
        return self.wait_for(request_id, timeout)
    
    def get_metrics(self) -> dict[str, Any]:
        """Get queue metrics."""
        return {
            **self._metrics,
            "queue_length": len(self._priority_queue) + (
                self._fair_scheduler.get_queue_length() 
                if self._fair_scheduler else 0
            ),
            "active_workers": len([w for w in self._workers if w.is_alive()]),
            "processing": sum(
                1 for r in self._requests.values()
                if r.state == RequestState.PROCESSING
            )
        }
    
    def cleanup(self, max_age_seconds: float = 300):
        """Remove old completed requests."""
        now = time.time()
        with self._lock:
            to_remove = [
                rid for rid, req in self._requests.items()
                if req.state in (
                    RequestState.COMPLETED,
                    RequestState.FAILED,
                    RequestState.TIMEOUT,
                    RequestState.CANCELLED
                ) and req.completed_at and now - req.completed_at > max_age_seconds
            ]
            for rid in to_remove:
                del self._requests[rid]
        
        if self._user_limiter:
            self._user_limiter.cleanup()


class AsyncRequestQueue:
    """Async version of RequestQueue."""
    
    def __init__(self, config: QueueConfig = None):
        self._sync_queue = RequestQueue(config)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    async def enqueue(self, payload: Any, **kwargs) -> Optional[str]:
        """Async enqueue."""
        return await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self._sync_queue.enqueue(payload, **kwargs)
        )
    
    async def wait_for(
        self,
        request_id: str,
        timeout: float = None
    ) -> tuple[Optional[Any], Optional[str]]:
        """Async wait for result."""
        return await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self._sync_queue.wait_for(request_id, timeout)
        )
    
    async def process(
        self,
        payload: Any,
        **kwargs
    ) -> tuple[Optional[Any], Optional[str]]:
        """Async enqueue and wait."""
        request_id = await self.enqueue(payload, **kwargs)
        if not request_id:
            return None, "Failed to enqueue"
        return await self.wait_for(request_id, kwargs.get("timeout"))


# Global instance
_queue: Optional[RequestQueue] = None


def get_request_queue(config: QueueConfig = None) -> RequestQueue:
    """Get or create global request queue."""
    global _queue
    if _queue is None:
        _queue = RequestQueue(config)
    return _queue
