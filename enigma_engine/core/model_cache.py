"""
Model Caching System

LRU caching for loaded models with automatic memory management,
eviction policies, and persistence support.

FILE: enigma_engine/core/model_cache.py
TYPE: Core/Infrastructure
MAIN CLASSES: ModelCache, CachePolicy, MemoryManager
"""

import gc
import hashlib
import logging
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    """Cache eviction policy."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    SIZE = "size"  # Largest first
    TTL = "ttl"  # Time-to-live based


@dataclass
class CacheEntry:
    """A cached model entry."""
    key: str
    model: Any
    size_bytes: int
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None  # Seconds
    
    # Metadata
    model_type: str = ""
    device: str = "cpu"
    dtype: str = ""
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Mark as recently accessed."""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheConfig:
    """Cache configuration."""
    # Memory limits
    max_memory_bytes: int = 4 * 1024**3  # 4GB default
    max_memory_percent: float = 50.0  # Max % of system memory
    
    # Entry limits
    max_entries: int = 10
    default_ttl: Optional[float] = None  # Seconds, None = no expiry
    
    # Eviction
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    evict_on_low_memory: bool = True
    low_memory_threshold: float = 10.0  # % free memory
    
    # Persistence
    enable_persistence: bool = False
    cache_dir: Path = Path("cache/models")
    
    # Cleanup
    cleanup_interval: float = 60.0  # Seconds
    enable_gc_on_evict: bool = True


class MemoryManager:
    """Manages memory usage for caching."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
    
    def get_system_memory(self) -> tuple[int, int, float]:
        """
        Get system memory info.
        
        Returns:
            (total_bytes, available_bytes, percent_used)
        """
        if HAS_PSUTIL:
            mem = psutil.virtual_memory()
            return mem.total, mem.available, mem.percent
        
        # Fallback: assume 8GB, 50% used
        total = 8 * 1024**3
        return total, total // 2, 50.0
    
    def get_gpu_memory(self, device: str = "cuda:0") -> Optional[tuple[int, int]]:
        """
        Get GPU memory info.
        
        Returns:
            (total_bytes, free_bytes) or None
        """
        if not HAS_TORCH or not torch.cuda.is_available():
            return None
        
        try:
            device_idx = int(device.split(":")[-1]) if ":" in device else 0
            torch.cuda.set_device(device_idx)
            total = torch.cuda.get_device_properties(device_idx).total_memory
            free = total - torch.cuda.memory_allocated(device_idx)
            return total, free
        except (RuntimeError, ValueError, IndexError):
            return None
    
    def can_allocate(self, size_bytes: int, device: str = "cpu") -> bool:
        """Check if we can allocate the given size."""
        if "cuda" in device.lower():
            gpu_mem = self.get_gpu_memory(device)
            if gpu_mem:
                _, free = gpu_mem
                return size_bytes < free * 0.9  # 10% buffer
            return False
        
        _, available, _ = self.get_system_memory()
        return size_bytes < available * 0.9
    
    def is_memory_low(self) -> bool:
        """Check if system memory is low."""
        _, available, percent_used = self.get_system_memory()
        return (100 - percent_used) < self.config.low_memory_threshold
    
    def estimate_model_size(self, model: Any) -> int:
        """Estimate model memory footprint."""
        if HAS_TORCH and hasattr(model, 'parameters'):
            # PyTorch model
            size = 0
            for param in model.parameters():
                size += param.numel() * param.element_size()
            for buffer in model.buffers():
                size += buffer.numel() * buffer.element_size()
            return size
        
        if hasattr(model, '__sizeof__'):
            return model.__sizeof__()
        
        # Fallback: serialize and measure
        try:
            return len(pickle.dumps(model))
        except (TypeError, pickle.PicklingError):
            return 100 * 1024**2  # Assume 100MB
    
    def force_gc(self):
        """Force garbage collection."""
        gc.collect()
        
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()


class ModelCache:
    """
    LRU cache for loaded models.
    
    Provides automatic memory management, eviction policies,
    and optional disk persistence for model caching.
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.memory = MemoryManager(self.config)
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Stats
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0
        }
        
        # Cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Setup persistence
        if self.config.enable_persistence:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """
        Get a model from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
        
        Returns:
            Cached model or default
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats["misses"] += 1
                return default
            
            if entry.is_expired:
                self._evict(key)
                self._stats["misses"] += 1
                return default
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            
            self._stats["hits"] += 1
            return entry.model
    
    def put(
        self,
        key: str,
        model: Any,
        ttl: float = None,
        model_type: str = "",
        device: str = "cpu"
    ) -> bool:
        """
        Add a model to cache.
        
        Args:
            key: Cache key
            model: Model to cache
            ttl: Time-to-live in seconds
            model_type: Type of model
            device: Device model is on
        
        Returns:
            True if cached successfully
        """
        # Estimate size
        size = self.memory.estimate_model_size(model)
        
        # Check if we can fit it
        if not self._ensure_space(size, device):
            logger.warning(f"Cannot cache model {key}: insufficient space")
            return False
        
        # Get dtype info
        dtype = ""
        if HAS_TORCH and hasattr(model, 'parameters'):
            try:
                param = next(model.parameters())
                dtype = str(param.dtype)
            except StopIteration:
                pass
        
        entry = CacheEntry(
            key=key,
            model=model,
            size_bytes=size,
            ttl=ttl or self.config.default_ttl,
            model_type=model_type,
            device=device,
            dtype=dtype
        )
        
        with self._lock:
            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats["size_bytes"] -= old_entry.size_bytes
            
            self._cache[key] = entry
            self._stats["size_bytes"] += size
        
        logger.debug(f"Cached model {key} ({size / 1024**2:.1f}MB)")
        return True
    
    def remove(self, key: str) -> bool:
        """Remove a model from cache."""
        with self._lock:
            if key in self._cache:
                self._evict(key)
                return True
        return False
    
    def contains(self, key: str) -> bool:
        """Check if key is in cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired:
                return True
        return False
    
    def clear(self):
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            self._stats["size_bytes"] = 0
        
        if self.config.enable_gc_on_evict:
            self.memory.force_gc()
    
    def _ensure_space(self, needed_bytes: int, device: str = "cpu") -> bool:
        """Ensure we have enough space, evicting if necessary."""
        with self._lock:
            # Check entry limit
            while len(self._cache) >= self.config.max_entries:
                if not self._evict_one():
                    return False
            
            # Check memory limits
            max_bytes = min(
                self.config.max_memory_bytes,
                int(self.memory.get_system_memory()[0] * 
                    self.config.max_memory_percent / 100)
            )
            
            while self._stats["size_bytes"] + needed_bytes > max_bytes:
                if not self._evict_one():
                    return False
            
            # Check actual system memory
            if self.config.evict_on_low_memory:
                while self.memory.is_memory_low():
                    if not self._evict_one():
                        return False
            
            # Check device memory
            if not self.memory.can_allocate(needed_bytes, device):
                while not self.memory.can_allocate(needed_bytes, device):
                    if not self._evict_one():
                        return False
        
        return True
    
    def _evict_one(self) -> bool:
        """Evict one entry based on policy."""
        if not self._cache:
            return False
        
        key = self._select_victim()
        if key:
            self._evict(key)
            return True
        return False
    
    def _select_victim(self) -> Optional[str]:
        """Select entry to evict based on policy."""
        if not self._cache:
            return None
        
        policy = self.config.eviction_policy
        
        if policy == EvictionPolicy.LRU:
            # First item in OrderedDict is least recently used
            return next(iter(self._cache))
        
        elif policy == EvictionPolicy.LFU:
            # Least frequently used
            return min(
                self._cache.keys(),
                key=lambda k: self._cache[k].access_count
            )
        
        elif policy == EvictionPolicy.FIFO:
            # First created
            return min(
                self._cache.keys(),
                key=lambda k: self._cache[k].created_at
            )
        
        elif policy == EvictionPolicy.SIZE:
            # Largest first
            return max(
                self._cache.keys(),
                key=lambda k: self._cache[k].size_bytes
            )
        
        elif policy == EvictionPolicy.TTL:
            # Closest to expiry
            now = time.time()
            
            def remaining_ttl(k):
                entry = self._cache[k]
                if entry.ttl is None:
                    return float('inf')
                return entry.ttl - (now - entry.created_at)
            
            return min(self._cache.keys(), key=remaining_ttl)
        
        # Default to LRU
        return next(iter(self._cache))
    
    def _evict(self, key: str):
        """Evict a specific entry."""
        entry = self._cache.pop(key, None)
        if entry:
            self._stats["size_bytes"] -= entry.size_bytes
            self._stats["evictions"] += 1
            
            logger.debug(f"Evicted model {key}")
            
            # Persist if enabled
            if self.config.enable_persistence:
                self._persist_entry(entry)
            
            # Clear references
            del entry.model
        
        if self.config.enable_gc_on_evict:
            self.memory.force_gc()
    
    def _persist_entry(self, entry: CacheEntry):
        """Persist entry to disk."""
        try:
            cache_path = self.config.cache_dir / f"{entry.key}.pkl"
            
            # Don't persist full model, just metadata for reload hint
            metadata = {
                "key": entry.key,
                "model_type": entry.model_type,
                "size_bytes": entry.size_bytes,
                "device": entry.device,
                "dtype": entry.dtype
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(metadata, f)
        
        except Exception as e:
            logger.error(f"Failed to persist cache entry: {e}")
    
    def get_or_load(
        self,
        key: str,
        loader: Callable[[], Any],
        ttl: float = None,
        model_type: str = "",
        device: str = "cpu"
    ) -> Optional[Any]:
        """
        Get from cache or load and cache.
        
        Args:
            key: Cache key
            loader: Function to load model if not cached
            ttl: Time-to-live
            model_type: Type of model
            device: Device to load to
        
        Returns:
            Model or None
        """
        # Check cache
        model = self.get(key)
        if model is not None:
            return model
        
        # Load model
        try:
            model = loader()
        except Exception as e:
            logger.error(f"Failed to load model {key}: {e}")
            return None
        
        # Cache it
        self.put(key, model, ttl, model_type, device)
        
        return model
    
    def start_cleanup(self):
        """Start background cleanup thread."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def stop_cleanup(self):
        """Stop background cleanup thread."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2.0)
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while self._running:
            time.sleep(self.config.cleanup_interval)
            self._cleanup_expired()
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        with self._lock:
            expired = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired:
                self._evict(key)
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0
            
            return {
                **self._stats,
                "entries": len(self._cache),
                "hit_rate": hit_rate,
                "size_mb": self._stats["size_bytes"] / 1024**2
            }
    
    def get_entries(self) -> list[dict[str, Any]]:
        """Get info about cached entries."""
        with self._lock:
            return [
                {
                    "key": e.key,
                    "size_mb": e.size_bytes / 1024**2,
                    "model_type": e.model_type,
                    "device": e.device,
                    "dtype": e.dtype,
                    "access_count": e.access_count,
                    "age_seconds": time.time() - e.created_at
                }
                for e in self._cache.values()
            ]


class ModelCacheDecorator:
    """Decorator for caching model loading functions."""
    
    def __init__(
        self,
        cache: ModelCache,
        key_func: Callable[..., str] = None,
        ttl: float = None
    ):
        self.cache = cache
        self.key_func = key_func
        self.ttl = ttl
    
    def __call__(self, func: Callable) -> Callable:
        """Decorate a model loading function."""
        def wrapper(*args, **kwargs):
            # Generate cache key
            if self.key_func:
                key = self.key_func(*args, **kwargs)
            else:
                # Default key from function name and args
                key_parts = [func.__name__] + [str(a) for a in args]
                key_parts += [f"{k}={v}" for k, v in sorted(kwargs.items())]
                key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            return self.cache.get_or_load(
                key=key,
                loader=lambda: func(*args, **kwargs),
                ttl=self.ttl
            )
        
        return wrapper


# Global instance
_cache: Optional[ModelCache] = None


def get_model_cache(config: CacheConfig = None) -> ModelCache:
    """Get or create global model cache."""
    global _cache
    if _cache is None:
        _cache = ModelCache(config)
    return _cache


def cached_model(key_func: Callable = None, ttl: float = None):
    """Decorator to cache model loading functions."""
    cache = get_model_cache()
    return ModelCacheDecorator(cache, key_func, ttl)
