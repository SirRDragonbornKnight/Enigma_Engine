"""
Caching Layer for enigma_engine

Multi-level caching for inference:
- In-memory LRU cache
- Redis integration
- Disk-based cache
- Semantic similarity cache

Usage:
    from enigma_engine.core.caching import InferenceCache
    
    cache = InferenceCache(backend='memory', max_size=1000)
    
    # Check cache
    cached = cache.get(prompt)
    if cached:
        return cached
    
    # Generate and cache
    response = model.generate(prompt)
    cache.set(prompt, response)
"""

import hashlib
import json
import logging
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    hits: int = 0
    metadata: Optional[dict[str, Any]] = None
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class LRUCache:
    """
    Thread-safe LRU cache.
    """
    
    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check expiration
                if entry.is_expired():
                    del self._cache[key]
                    self._misses += 1
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.hits += 1
                self._hits += 1
                
                return entry.value
            
            self._misses += 1
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Set value in cache."""
        with self._lock:
            # Remove if exists
            if key in self._cache:
                del self._cache[key]
            
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            # Add new entry
            expires_at = time.time() + ttl if ttl else None
            
            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                expires_at=expires_at,
                metadata=metadata
            )
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate
        }


class DiskCache:
    """
    Disk-based cache for persistence.
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_size_mb: int = 1000
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._index_path = self.cache_dir / 'index.json'
        self._index: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        self._load_index()
    
    def _load_index(self) -> None:
        """Load cache index from disk."""
        if self._index_path.exists():
            try:
                with open(self._index_path) as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self._index = {}
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self._index_path, 'w') as f:
                json.dump(self._index, f)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _key_to_path(self, key: str) -> Path:
        """Convert key to file path."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash[:2]}" / f"{key_hash}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self._lock:
            if key not in self._index:
                return None
            
            entry = self._index[key]
            
            # Check expiration
            if entry.get('expires_at') and time.time() > entry['expires_at']:
                self.delete(key)
                return None
            
            # Load from disk
            path = self._key_to_path(key)
            
            if not path.exists():
                del self._index[key]
                return None
            
            try:
                with open(path, 'rb') as f:
                    value = pickle.load(f)
                
                # Update hits
                self._index[key]['hits'] = entry.get('hits', 0) + 1
                
                return value
            except Exception as e:
                logger.error(f"Failed to load cache entry: {e}")
                return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Set value in disk cache."""
        with self._lock:
            # Serialize value
            path = self._key_to_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(path, 'wb') as f:
                    pickle.dump(value, f)
                
                # Update index
                self._index[key] = {
                    'path': str(path),
                    'created_at': time.time(),
                    'expires_at': time.time() + ttl if ttl else None,
                    'size': path.stat().st_size,
                    'hits': 0,
                    'metadata': metadata
                }
                
                self._save_index()
                
                # Evict if over size
                self._evict_if_needed()
                
            except Exception as e:
                logger.error(f"Failed to cache value: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key not in self._index:
                return False
            
            path = self._key_to_path(key)
            if path.exists():
                path.unlink()
            
            del self._index[key]
            self._save_index()
            
            return True
    
    def _evict_if_needed(self) -> None:
        """Evict entries if over size limit."""
        total_size = sum(e.get('size', 0) for e in self._index.values())
        
        if total_size <= self.max_size_bytes:
            return
        
        # Sort by last access (created_at as proxy)
        sorted_keys = sorted(
            self._index.keys(),
            key=lambda k: self._index[k].get('created_at', 0)
        )
        
        # Evict oldest until under limit
        for key in sorted_keys:
            if total_size <= self.max_size_bytes * 0.8:  # Leave 20% headroom
                break
            
            entry = self._index[key]
            total_size -= entry.get('size', 0)
            
            path = Path(entry['path'])
            if path.exists():
                path.unlink()
            
            del self._index[key]
        
        self._save_index()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            import shutil

            # Remove all cache files
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir() and subdir.name != 'index.json':
                    shutil.rmtree(subdir)
            
            self._index = {}
            self._save_index()


class RedisCache:
    """
    Redis-backed cache for distributed systems.
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = 'forge:'
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("redis not installed. Install with: pip install redis")
        
        self._client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False
        )
        self._prefix = prefix
    
    def _make_key(self, key: str) -> str:
        return f"{self._prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            data = self._client.get(self._make_key(key))
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Set value in Redis."""
        try:
            data = pickle.dumps(value)
            
            if ttl:
                self._client.setex(self._make_key(key), ttl, data)
            else:
                self._client.set(self._make_key(key), data)
            
            # Store metadata separately if needed
            if metadata:
                meta_key = f"{self._make_key(key)}:meta"
                self._client.set(meta_key, json.dumps(metadata))
                
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        try:
            result = self._client.delete(self._make_key(key))
            self._client.delete(f"{self._make_key(key)}:meta")
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear(self, pattern: str = '*') -> None:
        """Clear matching keys."""
        try:
            keys = self._client.keys(f"{self._prefix}{pattern}")
            if keys:
                self._client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear error: {e}")


class SemanticCache:
    """
    Cache with semantic similarity matching.
    
    Uses embeddings to find similar queries.
    """
    
    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        similarity_threshold: float = 0.95,
        max_size: int = 1000
    ):
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy not installed")
        
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        
        self._cache: list[tuple[str, np.ndarray, Any]] = []
        self._lock = threading.Lock()
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get(self, query: str) -> Optional[Any]:
        """Get semantically similar cached response."""
        query_embed = np.array(self.embed_fn(query))
        
        with self._lock:
            best_match = None
            best_sim = 0.0
            
            for key, embed, value in self._cache:
                sim = self._cosine_similarity(query_embed, embed)
                
                if sim > best_sim and sim >= self.similarity_threshold:
                    best_sim = sim
                    best_match = value
            
            return best_match
    
    def set(self, query: str, value: Any) -> None:
        """Cache a response with its embedding."""
        query_embed = np.array(self.embed_fn(query))
        
        with self._lock:
            # Check if similar query already exists
            for i, (key, embed, _) in enumerate(self._cache):
                sim = self._cosine_similarity(query_embed, embed)
                if sim >= self.similarity_threshold:
                    # Update existing
                    self._cache[i] = (query, query_embed, value)
                    return
            
            # Add new
            if len(self._cache) >= self.max_size:
                self._cache.pop(0)
            
            self._cache.append((query, query_embed, value))


class InferenceCache:
    """
    Unified inference cache interface.
    
    Supports multiple backends with consistent API.
    """
    
    def __init__(
        self,
        backend: str = 'memory',
        max_size: int = 1000,
        **kwargs
    ):
        """
        Initialize cache with specified backend.
        
        Args:
            backend: 'memory', 'disk', 'redis', or 'semantic'
            max_size: Maximum cache size
            **kwargs: Backend-specific options
        """
        self.backend_name = backend
        
        if backend == 'memory':
            self._backend = LRUCache(max_size=max_size)
        
        elif backend == 'disk':
            cache_dir = kwargs.get('cache_dir', '.cache/inference')
            max_size_mb = kwargs.get('max_size_mb', 1000)
            self._backend = DiskCache(cache_dir, max_size_mb)
        
        elif backend == 'redis':
            self._backend = RedisCache(
                host=kwargs.get('host', 'localhost'),
                port=kwargs.get('port', 6379),
                db=kwargs.get('db', 0),
                password=kwargs.get('password'),
                prefix=kwargs.get('prefix', 'forge:')
            )
        
        elif backend == 'semantic':
            embed_fn = kwargs.get('embed_fn')
            if not embed_fn:
                raise ValueError("semantic backend requires embed_fn")
            
            self._backend = SemanticCache(
                embed_fn=embed_fn,
                similarity_threshold=kwargs.get('similarity_threshold', 0.95),
                max_size=max_size
            )
        
        else:
            raise ValueError(f"Unknown cache backend: {backend}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._backend.get(key)
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Set value in cache."""
        self._backend.set(key, value, ttl=ttl, metadata=metadata)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        return self._backend.delete(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._backend.clear()
    
    def cached(
        self,
        ttl: Optional[int] = None,
        key_fn: Optional[Callable] = None
    ):
        """
        Decorator for caching function results.
        
        Usage:
            @cache.cached(ttl=3600)
            def generate(prompt):
                ...
        """
        def decorator(fn: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_fn:
                    key = key_fn(*args, **kwargs)
                else:
                    key = f"{fn.__name__}:{hash(str(args) + str(kwargs))}"
                
                # Check cache
                cached = self.get(key)
                if cached is not None:
                    return cached
                
                # Execute and cache
                result = fn(*args, **kwargs)
                self.set(key, result, ttl=ttl)
                
                return result
            
            return wrapper
        return decorator


def create_prompt_key(
    prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 256,
    **kwargs
) -> str:
    """
    Create a cache key from generation parameters.
    """
    key_data = {
        'prompt': prompt,
        'temperature': temperature,
        'max_tokens': max_tokens,
        **kwargs
    }
    
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()
