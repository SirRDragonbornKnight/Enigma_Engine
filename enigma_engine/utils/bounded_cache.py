"""
================================================================================
Bounded Cache - LRU Cache with Size and TTL Limits
================================================================================

Provides cache implementations that automatically evict entries to prevent
memory leaks from unbounded dictionaries.

USAGE:
    from enigma_engine.utils.bounded_cache import BoundedCache, TTLCache
    
    # LRU cache with max 1000 entries
    cache = BoundedCache(max_size=1000)
    cache["key"] = "value"
    
    # TTL cache with 60 second expiry
    ttl_cache = TTLCache(max_size=1000, ttl_seconds=60)
    ttl_cache["key"] = "value"  # Expires after 60s
    
    # Thread-safe usage
    from enigma_engine.utils.bounded_cache import ThreadSafeBoundedCache
    cache = ThreadSafeBoundedCache(max_size=1000)
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterator, Optional, TypeVar

K = TypeVar('K')
V = TypeVar('V')


class BoundedCache(Generic[K, V]):
    """
    LRU cache with maximum size limit.
    
    When the cache exceeds max_size, the least recently used entries
    are automatically evicted.
    
    Attributes:
        max_size: Maximum number of entries (default 1000)
        on_evict: Optional callback when entry is evicted
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        on_evict: Optional[Callable[[K, V], None]] = None
    ):
        self.max_size = max_size
        self.on_evict = on_evict
        self._data: OrderedDict[K, V] = OrderedDict()
    
    def __setitem__(self, key: K, value: V) -> None:
        """Set item, evicting LRU entries if needed."""
        if key in self._data:
            # Move to end (most recently used)
            self._data.move_to_end(key)
        self._data[key] = value
        
        # Evict oldest entries if over limit
        while len(self._data) > self.max_size:
            oldest_key, oldest_value = self._data.popitem(last=False)
            if self.on_evict:
                self.on_evict(oldest_key, oldest_value)
    
    def __getitem__(self, key: K) -> V:
        """Get item, marking it as recently used."""
        if key not in self._data:
            raise KeyError(key)
        # Move to end (most recently used)
        self._data.move_to_end(key)
        return self._data[key]
    
    def __delitem__(self, key: K) -> None:
        """Delete item."""
        del self._data[key]
    
    def __contains__(self, key: K) -> bool:
        """Check if key exists."""
        return key in self._data
    
    def __len__(self) -> int:
        """Return number of entries."""
        return len(self._data)
    
    def __iter__(self) -> Iterator[K]:
        """Iterate over keys."""
        return iter(self._data)
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get item or return default."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Remove and return item."""
        value = self._data.pop(key, default)
        return value
    
    def clear(self) -> None:
        """Clear all entries."""
        if self.on_evict:
            for key, value in self._data.items():
                self.on_evict(key, value)
        self._data.clear()
    
    def keys(self):
        """Return keys view."""
        return self._data.keys()
    
    def values(self):
        """Return values view."""
        return self._data.values()
    
    def items(self):
        """Return items view."""
        return self._data.items()


@dataclass
class CacheEntry(Generic[V]):
    """Entry with timestamp for TTL tracking."""
    value: V
    timestamp: float


class TTLCache(Generic[K, V]):
    """
    Cache with time-to-live expiration.
    
    Entries automatically expire after ttl_seconds.
    Also enforces a max_size limit.
    
    Attributes:
        max_size: Maximum number of entries
        ttl_seconds: Time-to-live in seconds
        on_evict: Optional callback when entry is evicted
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 300.0,  # 5 minutes default
        on_evict: Optional[Callable[[K, V], None]] = None
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.on_evict = on_evict
        self._data: OrderedDict[K, CacheEntry[V]] = OrderedDict()
    
    def _is_expired(self, entry: CacheEntry[V]) -> bool:
        """Check if entry has expired."""
        return time.time() - entry.timestamp > self.ttl_seconds
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired_keys = [
            key for key, entry in self._data.items()
            if now - entry.timestamp > self.ttl_seconds
        ]
        for key in expired_keys:
            entry = self._data.pop(key)
            if self.on_evict:
                self.on_evict(key, entry.value)
    
    def __setitem__(self, key: K, value: V) -> None:
        """Set item with current timestamp."""
        if key in self._data:
            self._data.move_to_end(key)
        
        self._data[key] = CacheEntry(value=value, timestamp=time.time())
        
        # Cleanup expired and enforce size limit
        self._cleanup_expired()
        while len(self._data) > self.max_size:
            oldest_key, entry = self._data.popitem(last=False)
            if self.on_evict:
                self.on_evict(oldest_key, entry.value)
    
    def __getitem__(self, key: K) -> V:
        """Get item, raising KeyError if expired or missing."""
        if key not in self._data:
            raise KeyError(key)
        
        entry = self._data[key]
        if self._is_expired(entry):
            del self._data[key]
            if self.on_evict:
                self.on_evict(key, entry.value)
            raise KeyError(key)
        
        # Refresh timestamp and move to end
        self._data.move_to_end(key)
        return entry.value
    
    def __delitem__(self, key: K) -> None:
        """Delete item."""
        del self._data[key]
    
    def __contains__(self, key: K) -> bool:
        """Check if key exists and is not expired."""
        if key not in self._data:
            return False
        if self._is_expired(self._data[key]):
            del self._data[key]
            return False
        return True
    
    def __len__(self) -> int:
        """Return number of non-expired entries."""
        self._cleanup_expired()
        return len(self._data)
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get item or return default."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def clear(self) -> None:
        """Clear all entries."""
        if self.on_evict:
            for key, entry in self._data.items():
                self.on_evict(key, entry.value)
        self._data.clear()


class ThreadSafeBoundedCache(Generic[K, V]):
    """
    Thread-safe version of BoundedCache.
    
    Uses a lock for all operations.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        on_evict: Optional[Callable[[K, V], None]] = None
    ):
        self._cache = BoundedCache[K, V](max_size=max_size, on_evict=on_evict)
        self._lock = threading.RLock()
    
    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            self._cache[key] = value
    
    def __getitem__(self, key: K) -> V:
        with self._lock:
            return self._cache[key]
    
    def __delitem__(self, key: K) -> None:
        with self._lock:
            del self._cache[key]
    
    def __contains__(self, key: K) -> bool:
        with self._lock:
            return key in self._cache
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            return self._cache.get(key, default)
    
    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            return self._cache.pop(key, default)
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class ThreadSafeTTLCache(Generic[K, V]):
    """
    Thread-safe version of TTLCache.
    
    Uses a lock for all operations.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 300.0,
        on_evict: Optional[Callable[[K, V], None]] = None
    ):
        self._cache = TTLCache[K, V](
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            on_evict=on_evict
        )
        self._lock = threading.RLock()
    
    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            self._cache[key] = value
    
    def __getitem__(self, key: K) -> V:
        with self._lock:
            return self._cache[key]
    
    def __delitem__(self, key: K) -> None:
        with self._lock:
            del self._cache[key]
    
    def __contains__(self, key: K) -> bool:
        with self._lock:
            return key in self._cache
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            return self._cache.get(key, default)
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# Convenience function for memoization with bounded cache
def bounded_lru_cache(
    max_size: int = 128,
    typed: bool = False
) -> Callable:
    """
    Decorator similar to functools.lru_cache but with bounded size.
    
    Args:
        max_size: Maximum cache entries
        typed: If True, arguments of different types are cached separately
        
    Usage:
        @bounded_lru_cache(max_size=100)
        def expensive_function(x):
            return x * 2
    """
    def decorator(func: Callable) -> Callable:
        cache: BoundedCache[tuple, Any] = BoundedCache(max_size=max_size)
        
        def wrapper(*args, **kwargs):
            # Create cache key
            if typed:
                key = (args, tuple(sorted(kwargs.items())), tuple(type(a) for a in args))
            else:
                key = (args, tuple(sorted(kwargs.items())))
            
            try:
                return cache[key]
            except KeyError:
                result = func(*args, **kwargs)
                cache[key] = result
                return result
        
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.__wrapped__ = func
        return wrapper
    
    return decorator


__all__ = [
    'BoundedCache',
    'TTLCache',
    'ThreadSafeBoundedCache',
    'ThreadSafeTTLCache',
    'bounded_lru_cache',
]
