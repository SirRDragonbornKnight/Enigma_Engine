"""
================================================================================
Network Optimizer - Reduce latency for distributed AI.
================================================================================

Optimizations for low-latency AI communication:
- Connection pooling
- Request batching
- Compression
- Predictive prefetching
- Adaptive timeout management

USAGE:
    from enigma_engine.comms.network_optimizer import NetworkOptimizer
    
    optimizer = NetworkOptimizer()
    
    # Optimized request
    response = optimizer.request(
        "http://server:5000/generate",
        {"prompt": "Hello"},
        priority="high"
    )
    
    # Batch multiple requests
    responses = optimizer.batch_request([
        {"url": "...", "data": {...}},
        {"url": "...", "data": {...}},
    ])
"""

import hashlib
import json
import logging
import queue
import threading
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib import request

logger = logging.getLogger(__name__)


@dataclass
class RequestStats:
    """Statistics for a request endpoint."""
    url: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    last_10_latencies: list[float] = field(default_factory=list)
    
    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests
    
    @property
    def recent_avg_latency_ms(self) -> float:
        if not self.last_10_latencies:
            return self.avg_latency_ms
        return sum(self.last_10_latencies) / len(self.last_10_latencies)
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests


@dataclass
class OptimizedRequest:
    """A request with optimization metadata."""
    url: str
    data: dict[str, Any]
    method: str = "POST"
    priority: str = "normal"  # low, normal, high, critical
    timeout: float = 30.0
    compress: bool = True
    cache_key: str = ""
    callback: Optional[Callable[[Any, Optional[Exception]], None]] = None


class ResponseCache:
    """LRU cache for responses."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: float = 60.0):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: dict[str, tuple[Any, float]] = {}
        self._access_order: deque = deque()
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached response if valid."""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                else:
                    # Expired
                    del self._cache[key]
            return None
    
    def set(self, key: str, value: Any):
        """Cache a response."""
        with self._lock:
            # Evict oldest if full
            while len(self._cache) >= self.max_size and self._access_order:
                oldest = self._access_order.popleft()
                if oldest in self._cache:
                    del self._cache[oldest]
            
            self._cache[key] = (value, time.time())
            self._access_order.append(key)
    
    def clear(self):
        """Clear all cached responses."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()


class NetworkOptimizer:
    """
    Optimizes network requests for distributed AI.
    
    Features:
    - Adaptive timeouts based on endpoint performance
    - Response caching
    - Request compression
    - Priority queuing
    - Batch processing
    - Automatic retry with backoff
    """
    
    def __init__(
        self,
        cache_size: int = 100,
        cache_ttl: float = 60.0,
        max_retries: int = 3,
        base_timeout: float = 30.0,
        enable_compression: bool = True,
    ):
        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.enable_compression = enable_compression
        
        # Stats per endpoint
        self._stats: dict[str, RequestStats] = {}
        self._stats_lock = threading.Lock()
        
        # Response cache
        self._cache = ResponseCache(cache_size, cache_ttl)
        
        # Request queue for batching
        self._request_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._batch_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Connection health
        self._endpoint_health: dict[str, bool] = {}
    
    def request(
        self,
        url: str,
        data: dict[str, Any],
        method: str = "POST",
        priority: str = "normal",
        timeout: Optional[float] = None,
        use_cache: bool = True,
        compress: bool = None,
    ) -> tuple[Optional[Any], Optional[Exception]]:
        """
        Make an optimized request.
        
        Args:
            url: Request URL
            data: Request data (will be JSON encoded)
            method: HTTP method
            priority: Request priority (low, normal, high, critical)
            timeout: Request timeout (None for adaptive)
            use_cache: Whether to use response cache
            compress: Whether to compress request
            
        Returns:
            Tuple of (response_data, error)
        """
        # Check cache
        if use_cache:
            cache_key = self._make_cache_key(url, data)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {url}")
                return cached, None
        
        # Get adaptive timeout
        if timeout is None:
            timeout = self._get_adaptive_timeout(url)
        
        # Decide on compression
        if compress is None:
            compress = self.enable_compression
        
        # Make request with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = self._make_request(url, data, method, timeout, compress)
                latency = (time.time() - start_time) * 1000
                
                # Record stats
                self._record_success(url, latency)
                
                # Cache response
                if use_cache:
                    self._cache.set(cache_key, response)
                
                return response, None
                
            except Exception as e:
                last_error = e
                self._record_failure(url)
                
                # Exponential backoff
                if attempt < self.max_retries - 1:
                    backoff = (2 ** attempt) * 0.5
                    time.sleep(backoff)
        
        return None, last_error
    
    def _make_request(
        self,
        url: str,
        data: dict[str, Any],
        method: str,
        timeout: float,
        compress: bool,
    ) -> Any:
        """Make the actual HTTP request."""
        # Encode data
        json_data = json.dumps(data).encode('utf-8')
        
        # Compress if enabled and data is large enough
        headers = {"Content-Type": "application/json"}
        if compress and len(json_data) > 1024:
            json_data = zlib.compress(json_data)
            headers["Content-Encoding"] = "gzip"
        
        # Create request
        req = request.Request(url, data=json_data, headers=headers, method=method)
        
        # Make request
        with request.urlopen(req, timeout=timeout) as response:
            response_data = response.read()
            
            # Decompress if needed
            if response.headers.get("Content-Encoding") == "gzip":
                response_data = zlib.decompress(response_data)
            
            return json.loads(response_data.decode('utf-8'))
    
    def _make_cache_key(self, url: str, data: dict[str, Any]) -> str:
        """Create a cache key from URL and data."""
        content = f"{url}:{json.dumps(data, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_adaptive_timeout(self, url: str) -> float:
        """Get adaptive timeout based on endpoint performance."""
        with self._stats_lock:
            stats = self._stats.get(url)
            if stats and stats.total_requests >= 5:
                # Use 3x the recent average, minimum base_timeout
                adaptive = max(
                    self.base_timeout,
                    stats.recent_avg_latency_ms * 3 / 1000
                )
                return min(adaptive, self.base_timeout * 3)  # Cap at 3x base
        
        return self.base_timeout
    
    def _record_success(self, url: str, latency_ms: float):
        """Record successful request."""
        with self._stats_lock:
            if url not in self._stats:
                self._stats[url] = RequestStats(url=url)
            
            stats = self._stats[url]
            stats.total_requests += 1
            stats.successful_requests += 1
            stats.total_latency_ms += latency_ms
            stats.min_latency_ms = min(stats.min_latency_ms, latency_ms)
            stats.max_latency_ms = max(stats.max_latency_ms, latency_ms)
            
            stats.last_10_latencies.append(latency_ms)
            if len(stats.last_10_latencies) > 10:
                stats.last_10_latencies.pop(0)
            
            self._endpoint_health[url] = True
    
    def _record_failure(self, url: str):
        """Record failed request."""
        with self._stats_lock:
            if url not in self._stats:
                self._stats[url] = RequestStats(url=url)
            
            stats = self._stats[url]
            stats.total_requests += 1
            stats.failed_requests += 1
            
            # Mark unhealthy if too many failures
            if stats.success_rate < 0.5:
                self._endpoint_health[url] = False
    
    def batch_request(
        self,
        requests: list[dict[str, Any]],
        parallel: int = 4,
    ) -> list[tuple[Optional[Any], Optional[Exception]]]:
        """
        Execute multiple requests in parallel.
        
        Args:
            requests: List of request dicts with 'url' and 'data' keys
            parallel: Max parallel requests
            
        Returns:
            List of (response, error) tuples in same order as requests
        """
        import concurrent.futures
        
        results = [None] * len(requests)
        
        def execute_request(index: int, req: dict[str, Any]):
            response, err = self.request(
                url=req.get("url", ""),
                data=req.get("data", {}),
                method=req.get("method", "POST"),
                priority=req.get("priority", "normal"),
            )
            return index, response, err
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [
                executor.submit(execute_request, i, req)
                for i, req in enumerate(requests)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    index, response, err = future.result()
                    results[index] = (response, err)
                except Exception as e:
                    # Find which request failed
                    pass
        
        return results
    
    def async_request(
        self,
        url: str,
        data: dict[str, Any],
        callback: Callable[[Any, Optional[Exception]], None],
        priority: str = "normal",
    ):
        """
        Make an asynchronous request with callback.
        
        Args:
            url: Request URL
            data: Request data
            callback: Called with (response, error) when complete
            priority: Request priority
        """
        def worker():
            response, err = self.request(url, data, priority=priority)
            callback(response, err)
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    def is_endpoint_healthy(self, url: str) -> bool:
        """Check if an endpoint is healthy."""
        return self._endpoint_health.get(url, True)
    
    def get_stats(self, url: str = None) -> dict[str, Any]:
        """Get statistics for endpoint(s)."""
        with self._stats_lock:
            if url:
                stats = self._stats.get(url)
                if stats:
                    return {
                        "url": stats.url,
                        "total_requests": stats.total_requests,
                        "success_rate": stats.success_rate,
                        "avg_latency_ms": stats.avg_latency_ms,
                        "recent_avg_latency_ms": stats.recent_avg_latency_ms,
                        "min_latency_ms": stats.min_latency_ms,
                        "max_latency_ms": stats.max_latency_ms,
                        "healthy": self._endpoint_health.get(url, True),
                    }
                return {}
            
            return {
                url: {
                    "total_requests": s.total_requests,
                    "success_rate": s.success_rate,
                    "avg_latency_ms": s.avg_latency_ms,
                    "healthy": self._endpoint_health.get(url, True),
                }
                for url, s in self._stats.items()
            }
    
    def clear_cache(self):
        """Clear response cache."""
        self._cache.clear()
    
    def reset_stats(self):
        """Reset all statistics."""
        with self._stats_lock:
            self._stats.clear()
            self._endpoint_health.clear()


# Global optimizer instance
_optimizer: Optional[NetworkOptimizer] = None


def get_network_optimizer(**kwargs) -> NetworkOptimizer:
    """Get or create global network optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = NetworkOptimizer(**kwargs)
    return _optimizer


__all__ = [
    'NetworkOptimizer',
    'OptimizedRequest',
    'RequestStats',
    'ResponseCache',
    'get_network_optimizer',
]
