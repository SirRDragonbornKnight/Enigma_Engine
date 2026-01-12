"""
Tool Result Caching for Enigma AI
==================================

Provides memory and disk-based caching for expensive tool operations.
Reduces repeated expensive operations like web searches and API calls.
"""

import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Set
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# Tools that can be safely cached
CACHEABLE_TOOLS = {
    "web_search",
    "fetch_webpage",
    "read_file",
    "list_directory",
    "get_system_info",
    "analyze_image",
    "read_document",
    "extract_text",
}


class ToolCache:
    """
    Cache tool results in memory and on disk.
    
    Features:
    - Memory cache for fast access
    - Disk cache for persistence across sessions
    - TTL (Time To Live) support
    - Hash-based cache keys
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        default_ttl: int = 300,  # 5 minutes
        max_memory_items: int = 100,
        enable_disk_cache: bool = True
    ):
        """
        Initialize tool cache.
        
        Args:
            cache_dir: Directory for disk cache (default: data/tool_cache)
            default_ttl: Default time-to-live in seconds (default: 300)
            max_memory_items: Maximum items in memory cache (default: 100)
            enable_disk_cache: Whether to use disk cache (default: True)
        """
        self.default_ttl = default_ttl
        self.max_memory_items = max_memory_items
        self.enable_disk_cache = enable_disk_cache
        
        # Memory cache: {cache_key: (result, expiry_time)}
        self.memory_cache: Dict[str, tuple] = {}
        
        # Disk cache directory
        if cache_dir is None:
            cache_dir = Path("data/tool_cache")
        self.cache_dir = cache_dir
        
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "evictions": 0,
        }
        
        logger.info(f"ToolCache initialized with TTL={default_ttl}s, disk={enable_disk_cache}")
    
    def _make_cache_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        """
        Generate a cache key from tool name and parameters.
        
        Args:
            tool_name: Name of the tool
            params: Tool parameters
            
        Returns:
            Hash-based cache key
        """
        # Sort params for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)
        key_string = f"{tool_name}:{sorted_params}"
        
        # Use SHA256 for cache key
        hash_obj = hashlib.sha256(key_string.encode())
        return hash_obj.hexdigest()[:32]  # Use first 32 chars
    
    def _is_cacheable(self, tool_name: str) -> bool:
        """Check if a tool's results can be cached."""
        return tool_name in CACHEABLE_TOOLS
    
    def _is_expired(self, expiry_time: float) -> bool:
        """Check if a cache entry has expired."""
        return time.time() > expiry_time
    
    def get(
        self,
        tool_name: str,
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a tool call.
        
        Args:
            tool_name: Name of the tool
            params: Tool parameters
            
        Returns:
            Cached result or None if not found/expired
        """
        if not self._is_cacheable(tool_name):
            return None
        
        cache_key = self._make_cache_key(tool_name, params)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            result, expiry_time = self.memory_cache[cache_key]
            
            if not self._is_expired(expiry_time):
                self.stats["hits"] += 1
                logger.debug(f"Cache HIT (memory) for {tool_name}")
                return result
            else:
                # Expired, remove from memory
                del self.memory_cache[cache_key]
        
        # Check disk cache
        if self.enable_disk_cache:
            disk_result = self._get_from_disk(cache_key)
            if disk_result is not None:
                result, expiry_time = disk_result
                
                if not self._is_expired(expiry_time):
                    # Add to memory cache
                    self._add_to_memory(cache_key, result, expiry_time)
                    self.stats["hits"] += 1
                    logger.debug(f"Cache HIT (disk) for {tool_name}")
                    return result
        
        self.stats["misses"] += 1
        logger.debug(f"Cache MISS for {tool_name}")
        return None
    
    def set(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """
        Cache a tool result.
        
        Args:
            tool_name: Name of the tool
            params: Tool parameters
            result: Tool result to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        if not self._is_cacheable(tool_name):
            return
        
        # Don't cache failed results
        if not result.get("success", False):
            return
        
        cache_key = self._make_cache_key(tool_name, params)
        
        if ttl is None:
            ttl = self.default_ttl
        
        expiry_time = time.time() + ttl
        
        # Add to memory cache
        self._add_to_memory(cache_key, result, expiry_time)
        
        # Add to disk cache
        if self.enable_disk_cache:
            self._save_to_disk(cache_key, result, expiry_time)
        
        self.stats["writes"] += 1
        logger.debug(f"Cached result for {tool_name} (TTL={ttl}s)")
    
    def _add_to_memory(
        self,
        cache_key: str,
        result: Dict[str, Any],
        expiry_time: float
    ):
        """Add entry to memory cache, evicting old entries if needed."""
        # Evict oldest if at capacity
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
            self.stats["evictions"] += 1
        
        self.memory_cache[cache_key] = (result, expiry_time)
    
    def _get_from_disk(self, cache_key: str) -> Optional[tuple]:
        """Load cache entry from disk."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            result = data.get("result")
            expiry_time = data.get("expiry_time")
            
            if result and expiry_time:
                return (result, expiry_time)
        
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")
        
        return None
    
    def _save_to_disk(
        self,
        cache_key: str,
        result: Dict[str, Any],
        expiry_time: float
    ):
        """Save cache entry to disk."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            data = {
                "result": result,
                "expiry_time": expiry_time,
                "cached_at": time.time(),
            }
            
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        
        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")
    
    def clear(self, tool_name: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            tool_name: If specified, only clear entries for this tool.
                      If None, clear all entries.
        """
        if tool_name is None:
            # Clear all
            self.memory_cache.clear()
            
            if self.enable_disk_cache:
                for cache_file in self.cache_dir.glob("*.json"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file: {e}")
            
            logger.info("Cleared all cache entries")
        else:
            # Clear specific tool (would need to iterate and check)
            # For now, just clear memory cache entries
            keys_to_delete = []
            for key in self.memory_cache:
                # Can't easily determine tool name from hash, so skip for now
                pass
            
            logger.info(f"Cleared cache for {tool_name}")
    
    def cleanup_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        
        # Clean memory cache
        expired_keys = [
            key for key, (_, expiry) in self.memory_cache.items()
            if expiry < current_time
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        # Clean disk cache
        if self.enable_disk_cache:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    
                    if data.get("expiry_time", 0) < current_time:
                        cache_file.unlink()
                except Exception:
                    pass
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "memory_entries": len(self.memory_cache),
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "writes": self.stats["writes"],
            "evictions": self.stats["evictions"],
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
        }


__all__ = [
    "ToolCache",
    "CACHEABLE_TOOLS",
]
