"""
================================================================================
Network Fallback - Offline mode and network resilience.
================================================================================

Handles network connectivity issues gracefully:
- Detect network availability
- Queue requests when offline
- Automatic retry with backoff
- Fallback to local resources
- Connection status monitoring

USAGE:
    from enigma_engine.utils.network_fallback import NetworkManager, get_network_manager
    
    manager = get_network_manager()
    
    # Check if online
    if manager.is_online():
        # Make network request
        pass
    else:
        # Use local fallback
        pass
    
    # With automatic fallback
    @manager.with_fallback(local_func)
    def fetch_data():
        return requests.get("https://api.example.com/data")
    
    # Queue request for later
    manager.queue_request(method="POST", url="...", data={...})
"""

from __future__ import annotations

import functools
import json
import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Network connection status."""
    ONLINE = auto()
    OFFLINE = auto()
    LIMITED = auto()  # Can reach local network but not internet
    CHECKING = auto()


@dataclass
class QueuedRequest:
    """A network request queued for later execution."""
    id: str
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    data: Any | None = None
    json_data: dict | None = None
    created_at: str = ""
    priority: int = 0  # Higher = more important
    max_retries: int = 3
    retry_count: int = 0
    
    def __post_init__(self):
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class NetworkConfig:
    """Configuration for network manager."""
    check_interval: float = 30.0  # Seconds between connectivity checks
    timeout: float = 5.0  # Connection timeout
    retry_backoff: float = 2.0  # Exponential backoff multiplier
    max_retry_delay: float = 300.0  # Max delay between retries (5 min)
    queue_persist: bool = True  # Persist queue to disk
    test_hosts: list[str] = field(default_factory=lambda: [
        "8.8.8.8",  # Google DNS
        "1.1.1.1",  # Cloudflare DNS
        "www.google.com",
    ])


class NetworkManager:
    """
    Manages network connectivity and provides offline fallback.
    """
    
    def __init__(
        self,
        config: NetworkConfig | None = None,
        data_path: Path | None = None
    ):
        """
        Initialize the network manager.
        
        Args:
            config: Network configuration
            data_path: Path to store queued requests
        """
        self.config = config or NetworkConfig()
        self._data_path = data_path or Path("data/network")
        self._data_path.mkdir(parents=True, exist_ok=True)
        
        self._queue_file = self._data_path / "queued_requests.json"
        self._status = ConnectionStatus.CHECKING
        self._last_check = 0.0
        self._status_listeners: list[Callable[[ConnectionStatus], None]] = []
        
        self._request_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._queued_requests: dict[str, QueuedRequest] = {}
        
        self._monitor_thread: threading.Thread | None = None
        self._running = False
        
        self._load_queue()
        
        # Do initial check
        self._check_connectivity()
    
    def _load_queue(self) -> None:
        """Load queued requests from disk."""
        if not self.config.queue_persist or not self._queue_file.exists():
            return
        
        try:
            with open(self._queue_file, encoding='utf-8') as f:
                data = json.load(f)
                for item in data.get("requests", []):
                    req = QueuedRequest(**item)
                    self._queued_requests[req.id] = req
                    self._request_queue.put((-req.priority, req.created_at, req))
            logger.info(f"Loaded {len(self._queued_requests)} queued requests")
        except Exception as e:
            logger.error(f"Failed to load request queue: {e}")
    
    def _save_queue(self) -> None:
        """Save queued requests to disk."""
        if not self.config.queue_persist:
            return
        
        try:
            data = {
                "requests": [
                    {
                        "id": r.id,
                        "method": r.method,
                        "url": r.url,
                        "headers": r.headers,
                        "data": r.data,
                        "json_data": r.json_data,
                        "created_at": r.created_at,
                        "priority": r.priority,
                        "max_retries": r.max_retries,
                        "retry_count": r.retry_count
                    }
                    for r in self._queued_requests.values()
                ]
            }
            with open(self._queue_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save request queue: {e}")
    
    def _check_connectivity(self) -> ConnectionStatus:
        """Check network connectivity."""
        self._last_check = time.time()
        old_status = self._status
        
        # Try to connect to test hosts
        for host in self.config.test_hosts:
            if self._can_reach_host(host):
                self._status = ConnectionStatus.ONLINE
                break
        else:
            # Check if we can reach localhost (local network at least)
            if self._can_reach_host("127.0.0.1", port=80):
                self._status = ConnectionStatus.LIMITED
            else:
                self._status = ConnectionStatus.OFFLINE
        
        # Notify listeners if status changed
        if old_status != self._status:
            logger.info(f"Network status changed: {old_status.name} -> {self._status.name}")
            for listener in self._status_listeners:
                try:
                    listener(self._status)
                except Exception as e:
                    logger.error(f"Status listener error: {e}")
        
        return self._status
    
    def _can_reach_host(self, host: str, port: int = 53) -> bool:
        """Test if we can reach a host."""
        try:
            # Handle URLs
            if "://" in host:
                parsed = urlparse(host)
                host = parsed.hostname or host
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
            
            socket.create_connection(
                (host, port),
                timeout=self.config.timeout
            ).close()
            return True
        except (socket.timeout, OSError):
            return False
    
    def is_online(self, force_check: bool = False) -> bool:
        """
        Check if network is available.
        
        Args:
            force_check: Force a new connectivity check
            
        Returns:
            True if online
        """
        if force_check or time.time() - self._last_check > self.config.check_interval:
            self._check_connectivity()
        
        return self._status == ConnectionStatus.ONLINE
    
    def get_status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._status
    
    def add_status_listener(self, callback: Callable[[ConnectionStatus], None]) -> None:
        """Add a listener for status changes."""
        self._status_listeners.append(callback)
    
    def remove_status_listener(self, callback: Callable[[ConnectionStatus], None]) -> None:
        """Remove a status listener."""
        if callback in self._status_listeners:
            self._status_listeners.remove(callback)
    
    def queue_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        data: Any = None,
        json_data: dict | None = None,
        priority: int = 0,
        max_retries: int = 3
    ) -> str:
        """
        Queue a request for later execution.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            headers: Request headers
            data: Request body
            json_data: JSON body
            priority: Priority (higher = more important)
            max_retries: Maximum retry attempts
            
        Returns:
            Request ID for tracking
        """
        req = QueuedRequest(
            id="",  # Will be auto-generated
            method=method,
            url=url,
            headers=headers or {},
            data=data,
            json_data=json_data,
            priority=priority,
            max_retries=max_retries
        )
        
        self._queued_requests[req.id] = req
        self._request_queue.put((-req.priority, req.created_at, req))
        self._save_queue()
        
        logger.info(f"Queued request {req.id}: {method} {url}")
        return req.id
    
    def get_queue_size(self) -> int:
        """Get number of queued requests."""
        return len(self._queued_requests)
    
    def clear_queue(self) -> None:
        """Clear all queued requests."""
        self._queued_requests.clear()
        self._request_queue = queue.PriorityQueue()
        self._save_queue()
    
    def process_queue(self) -> list[tuple[str, bool, Any]]:
        """
        Process queued requests (when back online).
        
        Returns:
            List of (request_id, success, result/error) tuples
        """
        if not self.is_online():
            return []
        
        results = []
        
        try:
            import requests
        except ImportError:
            logger.error("requests library not available")
            return []
        
        while not self._request_queue.empty():
            try:
                _, _, req = self._request_queue.get_nowait()
            except queue.Empty:
                break
            
            if req.id not in self._queued_requests:
                continue  # Already processed or removed
            
            try:
                response = requests.request(
                    method=req.method,
                    url=req.url,
                    headers=req.headers,
                    data=req.data,
                    json=req.json_data,
                    timeout=self.config.timeout * 2
                )
                response.raise_for_status()
                
                # Success - remove from queue
                del self._queued_requests[req.id]
                results.append((req.id, True, response.text))
                logger.info(f"Successfully processed queued request: {req.id}")
                
            except Exception as e:
                req.retry_count += 1
                
                if req.retry_count >= req.max_retries:
                    # Max retries reached - remove
                    del self._queued_requests[req.id]
                    results.append((req.id, False, str(e)))
                    logger.warning(f"Request {req.id} failed after {req.max_retries} retries")
                else:
                    # Re-queue with backoff
                    self._request_queue.put((-req.priority, req.created_at, req))
                    logger.info(f"Request {req.id} failed, retry {req.retry_count}/{req.max_retries}")
        
        self._save_queue()
        return results
    
    def with_fallback(
        self,
        fallback_func: Callable | None = None,
        fallback_value: Any = None,
        queue_on_failure: bool = False
    ):
        """
        Decorator that provides fallback when offline.
        
        Args:
            fallback_func: Function to call when offline
            fallback_value: Value to return when offline
            queue_on_failure: Queue the request for later
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if self.is_online():
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        logger.warning(f"Network request failed: {e}")
                        # Check if we're now offline
                        self._check_connectivity()
                        if not self.is_online():
                            return self._handle_offline(fallback_func, fallback_value)
                        raise
                else:
                    return self._handle_offline(fallback_func, fallback_value)
            
            return wrapper
        return decorator
    
    def _handle_offline(
        self,
        fallback_func: Callable | None,
        fallback_value: Any
    ) -> Any:
        """Handle offline scenario."""
        if fallback_func:
            return fallback_func()
        return fallback_value
    
    def start_monitoring(self) -> None:
        """Start background connectivity monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Started network monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Stopped network monitoring")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            time.sleep(self.config.check_interval)
            
            old_status = self._status
            self._check_connectivity()
            
            # If we came back online, process queue
            if old_status != ConnectionStatus.ONLINE and self._status == ConnectionStatus.ONLINE:
                if self._queued_requests:
                    logger.info("Back online, processing queued requests...")
                    self.process_queue()
    
    def wait_for_connection(
        self,
        timeout: float | None = None,
        check_interval: float = 5.0
    ) -> bool:
        """
        Block until network is available.
        
        Args:
            timeout: Maximum wait time (None = wait forever)
            check_interval: Time between checks
            
        Returns:
            True if connection established, False on timeout
        """
        start = time.time()
        
        while True:
            if self.is_online(force_check=True):
                return True
            
            if timeout and (time.time() - start) >= timeout:
                return False
            
            time.sleep(check_interval)


# Singleton instance
_network_manager_instance: NetworkManager | None = None


def get_network_manager(config: NetworkConfig | None = None) -> NetworkManager:
    """Get or create the singleton network manager."""
    global _network_manager_instance
    if _network_manager_instance is None:
        _network_manager_instance = NetworkManager(config)
    return _network_manager_instance


# Convenience functions
def is_online(force_check: bool = False) -> bool:
    """Quick check if online."""
    return get_network_manager().is_online(force_check)


def get_connection_status() -> ConnectionStatus:
    """Get current connection status."""
    return get_network_manager().get_status()


def queue_for_later(method: str, url: str, **kwargs) -> str:
    """Queue a request for when we're back online."""
    return get_network_manager().queue_request(method, url, **kwargs)


# Offline resource cache
class OfflineCache:
    """
    Cache for offline access to resources.
    """
    
    def __init__(self, cache_path: Path | None = None):
        """
        Initialize offline cache.
        
        Args:
            cache_path: Path to store cached resources
        """
        self._cache_path = cache_path or Path("data/offline_cache")
        self._cache_path.mkdir(parents=True, exist_ok=True)
        
        self._index_file = self._cache_path / "index.json"
        self._index: dict[str, dict[str, Any]] = {}
        
        self._load_index()
    
    def _load_index(self) -> None:
        """Load cache index."""
        if self._index_file.exists():
            try:
                with open(self._index_file, encoding='utf-8') as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache index: {e}")
    
    def _save_index(self) -> None:
        """Save cache index."""
        try:
            with open(self._index_file, 'w', encoding='utf-8') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _url_to_filename(self, url: str) -> str:
        """Convert URL to safe filename."""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()
    
    def cache_response(self, url: str, content: str, content_type: str = "text/plain") -> None:
        """
        Cache a response for offline use.
        
        Args:
            url: Original URL
            content: Response content
            content_type: Content MIME type
        """
        filename = self._url_to_filename(url)
        cache_file = self._cache_path / filename
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self._index[url] = {
                "filename": filename,
                "content_type": content_type,
                "cached_at": datetime.now().isoformat(),
                "size": len(content)
            }
            self._save_index()
            
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")
    
    def get_cached(self, url: str) -> str | None:
        """
        Get cached response.
        
        Args:
            url: Original URL
            
        Returns:
            Cached content or None
        """
        if url not in self._index:
            return None
        
        filename = self._index[url]["filename"]
        cache_file = self._cache_path / filename
        
        if cache_file.exists():
            try:
                with open(cache_file, encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to read cache: {e}")
        
        return None
    
    def has_cached(self, url: str) -> bool:
        """Check if URL is cached."""
        return url in self._index
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        for info in self._index.values():
            cache_file = self._cache_path / info["filename"]
            if cache_file.exists():
                cache_file.unlink()
        
        self._index.clear()
        self._save_index()
    
    def get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        return sum(info.get("size", 0) for info in self._index.values())


# Global cache instance
_offline_cache_instance: OfflineCache | None = None


def get_offline_cache() -> OfflineCache:
    """Get the offline cache instance."""
    global _offline_cache_instance
    if _offline_cache_instance is None:
        _offline_cache_instance = OfflineCache()
    return _offline_cache_instance
