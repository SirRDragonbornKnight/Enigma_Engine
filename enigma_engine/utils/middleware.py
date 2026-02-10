"""
Middleware Pipeline for Enigma AI Engine.

Provides request/response middleware for processing messages through
a chain of handlers. Each middleware can transform, validate, or
augment the request/response.

Usage:
    from enigma_engine.utils.middleware import Pipeline, Middleware, Request, Response
    
    # Create pipeline
    pipeline = Pipeline()
    
    # Add middleware
    @pipeline.middleware(priority=100)
    async def log_requests(request, next_handler):
        print(f"Request: {request.data}")
        response = await next_handler(request)
        print(f"Response: {response.data}")
        return response
    
    # Or use class-based middleware
    class AuthMiddleware(Middleware):
        async def __call__(self, request, next_handler):
            if not request.context.get("api_key"):
                return Response(error="Unauthorized")
            return await next_handler(request)
    
    pipeline.use(AuthMiddleware())
    
    # Process request
    response = await pipeline.process(Request(data={"prompt": "Hello"}))
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class Request:
    """
    Request object flowing through the middleware pipeline.
    
    Attributes:
        data: The request payload (prompt, message, etc.)
        context: Mutable context dict for middleware to add metadata
        metadata: Immutable request metadata
        id: Unique request identifier
        timestamp: Request creation time
    """
    data: Any
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"req_{int(time.time() * 1000)}")
    timestamp: float = field(default_factory=time.time)
    
    def with_context(self, **kwargs) -> "Request":
        """Create new request with additional context."""
        new_context = {**self.context, **kwargs}
        return Request(
            data=self.data,
            context=new_context,
            metadata=self.metadata,
            id=self.id,
            timestamp=self.timestamp,
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context or metadata."""
        return self.context.get(key, self.metadata.get(key, default))


@dataclass
class Response:
    """
    Response object from the middleware pipeline.
    
    Attributes:
        data: The response payload
        success: Whether the request was successful
        error: Error message if unsuccessful
        context: Response context/metadata
        request_id: ID of the originating request
        duration: Processing time in seconds
    """
    data: Any = None
    success: bool = True
    error: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    duration: float = 0.0
    
    @classmethod
    def ok(cls, data: Any, **context) -> "Response":
        """Create successful response."""
        return cls(data=data, success=True, context=context)
    
    @classmethod
    def fail(cls, error: str, **context) -> "Response":
        """Create failed response."""
        return cls(data=None, success=False, error=error, context=context)


# Type for the next handler in the chain
NextHandler = Callable[[Request], Awaitable[Response]]


class Middleware(ABC):
    """
    Abstract base class for middleware.
    
    Implement __call__ to process requests. Call next_handler
    to continue the chain, or return early to short-circuit.
    """
    
    @abstractmethod
    async def __call__(self, request: Request, next_handler: NextHandler) -> Response:
        """
        Process request through this middleware.
        
        Args:
            request: Incoming request
            next_handler: Call to continue the chain
        
        Returns:
            Response object
        """
    
    @property
    def name(self) -> str:
        """Middleware name for logging."""
        return self.__class__.__name__


class FunctionMiddleware(Middleware):
    """Middleware wrapper for functions."""
    
    def __init__(
        self,
        func: Callable[[Request, NextHandler], Awaitable[Response]],
        name: Optional[str] = None
    ):
        self._func = func
        self._name = name or func.__name__
    
    async def __call__(self, request: Request, next_handler: NextHandler) -> Response:
        return await self._func(request, next_handler)
    
    @property
    def name(self) -> str:
        return self._name


@dataclass
class MiddlewareEntry:
    """Entry in the middleware stack."""
    middleware: Middleware
    priority: int = 0
    enabled: bool = True
    
    def __lt__(self, other: "MiddlewareEntry") -> bool:
        """Higher priority comes first."""
        return self.priority > other.priority


class Pipeline:
    """
    Middleware pipeline for processing requests.
    
    Middleware is executed in priority order (highest first).
    Each middleware can transform the request, short-circuit
    the chain, or modify the response.
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize pipeline.
        
        Args:
            name: Pipeline identifier
        """
        self.name = name
        self._middleware: list[MiddlewareEntry] = []
        self._final_handler: Optional[Callable[[Request], Awaitable[Response]]] = None
        self._error_handler: Optional[Callable[[Request, Exception], Awaitable[Response]]] = None
    
    def use(
        self,
        middleware: Union[Middleware, Callable],
        priority: int = 0,
        name: Optional[str] = None
    ) -> "Pipeline":
        """
        Add middleware to the pipeline.
        
        Args:
            middleware: Middleware instance or async function
            priority: Execution priority (higher = earlier)
            name: Optional name for function middleware
        
        Returns:
            Self for chaining
        """
        if not isinstance(middleware, Middleware):
            middleware = FunctionMiddleware(middleware, name)
        
        entry = MiddlewareEntry(middleware=middleware, priority=priority)
        self._middleware.append(entry)
        self._middleware.sort()
        
        logger.debug(f"Pipeline '{self.name}': Added middleware '{middleware.name}' (priority={priority})")
        return self
    
    def middleware(
        self,
        priority: int = 0,
        name: Optional[str] = None
    ) -> Callable:
        """
        Decorator to add middleware function.
        
        Args:
            priority: Execution priority
            name: Optional middleware name
        
        Returns:
            Decorator function
        
        Example:
            @pipeline.middleware(priority=100)
            async def my_middleware(request, next_handler):
                # Pre-processing
                response = await next_handler(request)
                # Post-processing
                return response
        """
        def decorator(func: Callable) -> Callable:
            self.use(func, priority=priority, name=name)
            return func
        return decorator
    
    def set_handler(self, handler: Callable[[Request], Awaitable[Response]]) -> "Pipeline":
        """
        Set the final request handler.
        
        Args:
            handler: Async function that processes the request
        
        Returns:
            Self for chaining
        """
        self._final_handler = handler
        return self
    
    def set_error_handler(
        self,
        handler: Callable[[Request, Exception], Awaitable[Response]]
    ) -> "Pipeline":
        """
        Set error handler for uncaught exceptions.
        
        Args:
            handler: Async function to handle errors
        
        Returns:
            Self for chaining
        """
        self._error_handler = handler
        return self
    
    def remove(self, name: str) -> bool:
        """
        Remove middleware by name.
        
        Args:
            name: Middleware name
        
        Returns:
            True if removed
        """
        for i, entry in enumerate(self._middleware):
            if entry.middleware.name == name:
                self._middleware.pop(i)
                logger.debug(f"Pipeline '{self.name}': Removed middleware '{name}'")
                return True
        return False
    
    def enable(self, name: str) -> bool:
        """Enable middleware by name."""
        for entry in self._middleware:
            if entry.middleware.name == name:
                entry.enabled = True
                return True
        return False
    
    def disable(self, name: str) -> bool:
        """Disable middleware by name."""
        for entry in self._middleware:
            if entry.middleware.name == name:
                entry.enabled = False
                return True
        return False
    
    def clear(self) -> None:
        """Remove all middleware."""
        self._middleware.clear()
    
    async def process(self, request: Request) -> Response:
        """
        Process request through the middleware chain.
        
        Args:
            request: Request to process
        
        Returns:
            Response from the chain
        """
        start_time = time.time()
        
        try:
            # Build chain of enabled middleware
            enabled = [e for e in self._middleware if e.enabled]
            
            # Build the handler chain (reversed so first middleware wraps last)
            async def final_handler(req: Request) -> Response:
                if self._final_handler:
                    return await self._final_handler(req)
                return Response.ok(req.data)
            
            handler = final_handler
            
            for entry in reversed(enabled):
                current_middleware = entry.middleware
                current_handler = handler
                
                async def make_handler(mw: Middleware, next_h: NextHandler):
                    async def wrapped(req: Request) -> Response:
                        return await mw(req, next_h)
                    return wrapped
                
                handler = await make_handler(current_middleware, current_handler)
            
            # Execute the chain
            response = await handler(request)
            response.request_id = request.id
            response.duration = time.time() - start_time
            
            return response
            
        except Exception as e:
            logger.error(f"Pipeline '{self.name}' error: {e}")
            
            if self._error_handler:
                try:
                    return await self._error_handler(request, e)
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {handler_error}")
            
            return Response.fail(
                error=str(e),
                request_id=request.id,
                duration=time.time() - start_time
            )
    
    def process_sync(self, request: Request) -> Response:
        """
        Synchronous wrapper for process().
        
        Args:
            request: Request to process
        
        Returns:
            Response from the chain
        """
        try:
            # Check if there's already a running loop
            loop = asyncio.get_running_loop()
            # Running in async context - schedule in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.process(request))
                return future.result()
        except RuntimeError:
            # No running loop - use asyncio.run()
            return asyncio.run(self.process(request))
    
    def list_middleware(self) -> list[dict[str, Any]]:
        """List all middleware with their status."""
        return [
            {
                "name": entry.middleware.name,
                "priority": entry.priority,
                "enabled": entry.enabled,
            }
            for entry in self._middleware
        ]


# Built-in middleware implementations

class LoggingMiddleware(Middleware):
    """Log requests and responses."""
    
    def __init__(self, log_data: bool = False):
        self.log_data = log_data
    
    async def __call__(self, request: Request, next_handler: NextHandler) -> Response:
        start = time.time()
        logger.info(f"Request {request.id} started")
        
        if self.log_data:
            logger.debug(f"Request data: {request.data}")
        
        response = await next_handler(request)
        
        duration = time.time() - start
        status = "OK" if response.success else f"ERROR: {response.error}"
        logger.info(f"Request {request.id} completed in {duration:.3f}s - {status}")
        
        return response


class TimingMiddleware(Middleware):
    """Add timing information to response context."""
    
    async def __call__(self, request: Request, next_handler: NextHandler) -> Response:
        start = time.time()
        request.context["_timing_start"] = start
        
        response = await next_handler(request)
        
        response.context["processing_time"] = time.time() - start
        return response


class ValidationMiddleware(Middleware):
    """Validate request data."""
    
    def __init__(self, validator: Callable[[Any], bool], error_message: str = "Validation failed"):
        self.validator = validator
        self.error_message = error_message
    
    async def __call__(self, request: Request, next_handler: NextHandler) -> Response:
        if not self.validator(request.data):
            return Response.fail(self.error_message)
        return await next_handler(request)


class RateLimitMiddleware(Middleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, max_requests: int = 100, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: list[float] = []
    
    async def __call__(self, request: Request, next_handler: NextHandler) -> Response:
        now = time.time()
        
        # Remove old requests
        self._requests = [t for t in self._requests if now - t < self.window_seconds]
        
        if len(self._requests) >= self.max_requests:
            return Response.fail(f"Rate limit exceeded ({self.max_requests} per {self.window_seconds}s)")
        
        self._requests.append(now)
        return await next_handler(request)


class RetryMiddleware(Middleware):
    """Retry failed requests."""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
    
    async def __call__(self, request: Request, next_handler: NextHandler) -> Response:
        last_error = None
        delay = self.delay
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await next_handler(request)
                if response.success:
                    return response
                last_error = response.error
            except Exception as e:
                last_error = str(e)
            
            if attempt < self.max_retries:
                logger.debug(f"Retry {attempt + 1}/{self.max_retries} after {delay}s")
                await asyncio.sleep(delay)
                delay *= self.backoff
        
        return Response.fail(f"Failed after {self.max_retries} retries: {last_error}")


class CacheMiddleware(Middleware):
    """Cache responses."""
    
    def __init__(self, ttl_seconds: float = 300.0, key_func: Optional[Callable[[Request], str]] = None):
        self.ttl_seconds = ttl_seconds
        self.key_func = key_func or (lambda r: str(r.data))
        self._cache: dict[str, tuple] = {}  # key -> (response, timestamp)
    
    async def __call__(self, request: Request, next_handler: NextHandler) -> Response:
        key = self.key_func(request)
        now = time.time()
        
        # Check cache
        if key in self._cache:
            response, timestamp = self._cache[key]
            if now - timestamp < self.ttl_seconds:
                logger.debug(f"Cache hit for key: {key[:50]}")
                response.context["from_cache"] = True
                return response
        
        # Miss - get fresh response
        response = await next_handler(request)
        
        if response.success:
            self._cache[key] = (response, now)
        
        # Cleanup old entries
        self._cleanup_cache(now)
        
        return response
    
    def _cleanup_cache(self, now: float) -> None:
        """Remove expired cache entries."""
        expired = [k for k, (_, ts) in self._cache.items() if now - ts >= self.ttl_seconds]
        for key in expired:
            del self._cache[key]


class TransformMiddleware(Middleware):
    """Transform request/response data."""
    
    def __init__(
        self,
        request_transform: Optional[Callable[[Any], Any]] = None,
        response_transform: Optional[Callable[[Any], Any]] = None
    ):
        self.request_transform = request_transform
        self.response_transform = response_transform
    
    async def __call__(self, request: Request, next_handler: NextHandler) -> Response:
        if self.request_transform:
            request = Request(
                data=self.request_transform(request.data),
                context=request.context,
                metadata=request.metadata,
                id=request.id,
                timestamp=request.timestamp,
            )
        
        response = await next_handler(request)
        
        if self.response_transform and response.data:
            response.data = self.response_transform(response.data)
        
        return response


# Global pipeline registry
_pipelines: dict[str, Pipeline] = {}


def get_pipeline(name: str = "default") -> Pipeline:
    """Get or create a pipeline by name."""
    if name not in _pipelines:
        _pipelines[name] = Pipeline(name)
    return _pipelines[name]


def list_pipelines() -> list[str]:
    """List all pipeline names."""
    return list(_pipelines.keys())


# Pre-configured pipelines for Enigma AI Engine
def create_chat_pipeline() -> Pipeline:
    """Create pipeline for chat requests."""
    pipeline = Pipeline("chat")
    pipeline.use(LoggingMiddleware(), priority=1000)
    pipeline.use(TimingMiddleware(), priority=900)
    pipeline.use(RateLimitMiddleware(max_requests=60, window_seconds=60), priority=800)
    return pipeline


def create_inference_pipeline() -> Pipeline:
    """Create pipeline for model inference."""
    pipeline = Pipeline("inference")
    pipeline.use(TimingMiddleware(), priority=900)
    pipeline.use(RetryMiddleware(max_retries=2, delay=0.5), priority=500)
    return pipeline


def create_api_pipeline() -> Pipeline:
    """Create pipeline for API requests."""
    pipeline = Pipeline("api")
    pipeline.use(LoggingMiddleware(log_data=False), priority=1000)
    pipeline.use(TimingMiddleware(), priority=900)
    pipeline.use(RateLimitMiddleware(max_requests=100, window_seconds=60), priority=800)
    pipeline.use(RetryMiddleware(max_retries=3, delay=1.0), priority=500)
    return pipeline
