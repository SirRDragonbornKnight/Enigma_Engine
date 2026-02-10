"""
Prefix Caching System

Caches processed system prompts and common prefixes for faster inference.
Reduces redundant computation by reusing KV cache states.

FILE: enigma_engine/core/prefix_cache.py
TYPE: Inference Optimization
MAIN CLASSES: PrefixCache, CachedPrefix, PrefixCacheManager
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from ..config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class CachedPrefix:
    """A cached prefix with its KV cache state."""
    prefix_hash: str
    prefix_text: str
    token_ids: list[int]
    kv_cache: Optional[Any] = None  # (key_cache, value_cache) tensors
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    memory_bytes: int = 0
    
    def update_usage(self):
        """Update usage statistics."""
        self.last_used = time.time()
        self.use_count += 1


class PrefixCache:
    """LRU cache for processed prefixes with KV cache states."""
    
    def __init__(self, 
                 max_entries: int = 32,
                 max_memory_mb: float = 512.0,
                 persist_dir: Optional[Path] = None):
        """
        Initialize prefix cache.
        
        Args:
            max_entries: Maximum number of cached prefixes
            max_memory_mb: Maximum memory usage in MB
            persist_dir: Directory for persistent cache (optional)
        """
        self._cache: OrderedDict[str, CachedPrefix] = OrderedDict()
        self._max_entries = max_entries
        self._max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self._current_memory = 0
        self._lock = threading.RLock()
        self._persist_dir = persist_dir
        self._hits = 0
        self._misses = 0
        
        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_metadata()
            
    def _compute_hash(self, text: str, model_id: str = "") -> str:
        """Compute hash for prefix text."""
        content = f"{model_id}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(self, prefix_text: str, model_id: str = "") -> Optional[CachedPrefix]:
        """
        Get cached prefix if available.
        
        Args:
            prefix_text: The prefix text to look up
            model_id: Optional model identifier for cache key
            
        Returns:
            CachedPrefix if found, None otherwise
        """
        prefix_hash = self._compute_hash(prefix_text, model_id)
        
        with self._lock:
            if prefix_hash in self._cache:
                entry = self._cache[prefix_hash]
                entry.update_usage()
                # Move to end (most recently used)
                self._cache.move_to_end(prefix_hash)
                self._hits += 1
                logger.debug(f"Prefix cache hit: {prefix_hash}")
                return entry
            
            self._misses += 1
            return None
    
    def put(self, prefix_text: str, token_ids: list[int],
            kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            model_id: str = "") -> CachedPrefix:
        """
        Store a prefix in the cache.
        
        Args:
            prefix_text: The prefix text
            token_ids: Tokenized prefix
            kv_cache: Optional KV cache tensors (key_cache, value_cache)
            model_id: Optional model identifier
            
        Returns:
            The cached prefix entry
        """
        prefix_hash = self._compute_hash(prefix_text, model_id)
        
        # Calculate memory usage
        memory_bytes = len(prefix_text.encode())
        memory_bytes += len(token_ids) * 4  # int32
        if kv_cache is not None:
            for tensor in kv_cache:
                if tensor is not None:
                    memory_bytes += tensor.numel() * tensor.element_size()
        
        with self._lock:
            # Evict if necessary
            self._evict_if_needed(memory_bytes)
            
            # Create entry
            entry = CachedPrefix(
                prefix_hash=prefix_hash,
                prefix_text=prefix_text,
                token_ids=token_ids,
                kv_cache=kv_cache,
                memory_bytes=memory_bytes
            )
            
            self._cache[prefix_hash] = entry
            self._current_memory += memory_bytes
            
            logger.debug(f"Cached prefix {prefix_hash}: {len(token_ids)} tokens, "
                        f"{memory_bytes / 1024:.1f} KB")
            
            return entry
    
    def _evict_if_needed(self, needed_bytes: int):
        """Evict entries to make room."""
        # Check entry count
        while len(self._cache) >= self._max_entries:
            self._evict_oldest()
            
        # Check memory
        while self._current_memory + needed_bytes > self._max_memory and self._cache:
            self._evict_oldest()
            
    def _evict_oldest(self):
        """Evict the least recently used entry."""
        if self._cache:
            oldest_key, oldest_entry = self._cache.popitem(last=False)
            self._current_memory -= oldest_entry.memory_bytes
            
            # Clear GPU memory if applicable
            if oldest_entry.kv_cache is not None:
                del oldest_entry.kv_cache
                
            logger.debug(f"Evicted prefix {oldest_key}")
            
    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            for entry in self._cache.values():
                if entry.kv_cache is not None:
                    del entry.kv_cache
            self._cache.clear()
            self._current_memory = 0
            self._hits = 0
            self._misses = 0
            
    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "memory_used_mb": self._current_memory / (1024 * 1024),
                "max_memory_mb": self._max_memory / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate
            }
            
    def _load_metadata(self):
        """Load cache metadata from disk."""
        if not self._persist_dir:
            return
            
        meta_path = self._persist_dir / "prefix_cache_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    data = json.load(f)
                    # Just log for now, actual KV cache loading would be complex
                    logger.info(f"Found {len(data.get('entries', []))} persisted prefix metadata")
            except Exception as e:
                logger.warning(f"Failed to load prefix cache metadata: {e}")
                
    def save_metadata(self):
        """Save cache metadata to disk."""
        if not self._persist_dir:
            return
            
        try:
            entries = []
            for entry in self._cache.values():
                entries.append({
                    "hash": entry.prefix_hash,
                    "text": entry.prefix_text[:100],  # Truncate for storage
                    "tokens": len(entry.token_ids),
                    "memory_bytes": entry.memory_bytes,
                    "use_count": entry.use_count
                })
                
            meta_path = self._persist_dir / "prefix_cache_meta.json"
            with open(meta_path, 'w') as f:
                json.dump({"entries": entries}, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save prefix cache metadata: {e}")


class PrefixCacheManager:
    """High-level manager for prefix caching with inference integration."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._cache = PrefixCache(
            max_entries=CONFIG.get("prefix_cache_entries", 32),
            max_memory_mb=CONFIG.get("prefix_cache_memory_mb", 512.0),
            persist_dir=Path(CONFIG.get("cache_dir", "cache")) / "prefix"
        )
        
        # Common system prompts to pre-cache
        self._common_prompts: dict[str, str] = {}
        self._initialized = True
        
    @property
    def cache(self) -> PrefixCache:
        """Get the underlying cache."""
        return self._cache
    
    def register_system_prompt(self, name: str, prompt: str):
        """Register a commonly-used system prompt.
        
        Args:
            name: Name identifier for the prompt
            prompt: The prompt text
        """
        self._common_prompts[name] = prompt
        
    def get_or_compute_prefix(self, 
                               prefix_text: str,
                               tokenizer,
                               model = None,
                               model_id: str = "") -> tuple[list[int], Optional[Any]]:
        """
        Get cached prefix or compute it.
        
        Args:
            prefix_text: The prefix text
            tokenizer: Tokenizer to use
            model: Model for computing KV cache (optional)
            model_id: Model identifier
            
        Returns:
            Tuple of (token_ids, kv_cache)
        """
        # Check cache
        cached = self._cache.get(prefix_text, model_id)
        if cached is not None:
            return cached.token_ids, cached.kv_cache
            
        # Tokenize
        token_ids = tokenizer.encode(prefix_text)
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        # Compute KV cache if model provided
        kv_cache = None
        if model is not None:
            try:
                kv_cache = self._compute_kv_cache(model, token_ids)
            except Exception as e:
                logger.warning(f"Failed to compute KV cache: {e}")
                
        # Store in cache
        self._cache.put(prefix_text, token_ids, kv_cache, model_id)
        
        return token_ids, kv_cache
    
    def _compute_kv_cache(self, model, token_ids: list[int]) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Compute KV cache for tokens using the model."""
        try:
            device = next(model.parameters()).device
            input_ids = torch.tensor([token_ids], device=device)
            
            # Run forward pass to populate cache
            with torch.no_grad():
                # Check if model supports KV cache output
                if hasattr(model, 'forward_with_cache'):
                    _, kv_cache = model.forward_with_cache(input_ids)
                    return kv_cache
                elif hasattr(model, 'forward'):
                    # Try to get cache from model state
                    model(input_ids)
                    if hasattr(model, 'kv_cache'):
                        return model.kv_cache
                        
            return None
        except Exception as e:
            logger.debug(f"KV cache computation skipped: {e}")
            return None
    
    def prefetch_system_prompts(self, tokenizer, model=None, model_id: str = ""):
        """Pre-cache registered system prompts.
        
        Args:
            tokenizer: Tokenizer to use
            model: Model for KV cache computation
            model_id: Model identifier
        """
        for name, prompt in self._common_prompts.items():
            logger.info(f"Pre-caching system prompt: {name}")
            self.get_or_compute_prefix(prompt, tokenizer, model, model_id)
            
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return self._cache.get_stats()
    
    def clear(self):
        """Clear all cached prefixes."""
        self._cache.clear()


def get_prefix_cache_manager() -> PrefixCacheManager:
    """Get the prefix cache manager singleton."""
    return PrefixCacheManager()


# Convenience function for direct use
def cache_prefix(text: str, tokenizer, model=None, model_id: str = "") -> tuple[list[int], Any]:
    """Cache a prefix and return tokens + KV cache.
    
    Args:
        text: Prefix text
        tokenizer: Tokenizer
        model: Model (optional)
        model_id: Model identifier
        
    Returns:
        (token_ids, kv_cache)
    """
    manager = get_prefix_cache_manager()
    return manager.get_or_compute_prefix(text, tokenizer, model, model_id)


__all__ = [
    'PrefixCache',
    'CachedPrefix',
    'PrefixCacheManager',
    'get_prefix_cache_manager',
    'cache_prefix'
]
