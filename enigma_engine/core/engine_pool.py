"""
================================================================================
ENGINE POOL - SHARED MODEL INSTANCE MANAGEMENT
================================================================================

Provides a centralized pool of EnigmaEngine instances for efficient reuse.
Prevents repeatedly loading models and manages engine lifecycle.

FILE: enigma_engine/core/engine_pool.py
TYPE: Resource Management Utility
MAIN CLASS: EnginePool

USAGE:
    from enigma_engine.core.engine_pool import get_engine, release_engine, EnginePool
    
    # Get a shared engine (creates if needed, reuses if available)
    engine = get_engine()
    response = engine.chat("Hello!")
    
    # Release when done (returns to pool, doesn't destroy)
    release_engine(engine)
    
    # Or use context manager
    with get_engine_context() as engine:
        response = engine.chat("Hello!")
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# ENGINE POOL CONFIGURATION
# =============================================================================

@dataclass
class EnginePoolConfig:
    """Configuration for the engine pool."""
    
    # Maximum number of engines to keep in pool
    max_pool_size: int = 2
    
    # Maximum idle time before engine is unloaded (seconds)
    max_idle_seconds: float = 300.0  # 5 minutes
    
    # Whether to preload an engine on first access
    preload_on_first_access: bool = True
    
    # Default model path (None = use CONFIG default)
    default_model_path: Optional[str] = None
    
    # Enable automatic cleanup of idle engines
    enable_auto_cleanup: bool = True
    
    # Cleanup check interval (seconds)
    cleanup_interval: float = 60.0


@dataclass
class PooledEngine:
    """Wrapper for a pooled engine with metadata."""
    
    engine: Any  # EnigmaEngine
    model_path: str
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    use_count: int = 0
    in_use: bool = False
    
    def mark_used(self) -> None:
        """Mark engine as being used."""
        self.last_used_at = time.time()
        self.use_count += 1
        self.in_use = True
    
    def mark_released(self) -> None:
        """Mark engine as released back to pool."""
        self.last_used_at = time.time()
        self.in_use = False
    
    @property
    def idle_seconds(self) -> float:
        """Time since last use."""
        return time.time() - self.last_used_at


# =============================================================================
# ENGINE POOL CLASS
# =============================================================================

class EnginePool:
    """
    Pool of EnigmaEngine instances for efficient reuse.
    
    Benefits:
    - Prevents repeated model loading (slow!)
    - Manages memory by limiting concurrent engines
    - Auto-cleans idle engines
    - Thread-safe access
    
    The pool maintains engines by model path, so different models
    can coexist in the pool.
    """
    
    _instance: Optional["EnginePool"] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[EnginePoolConfig] = None):
        """Initialize the engine pool."""
        if self._initialized:
            return
        
        self.config = config or EnginePoolConfig()
        self._engines: dict[str, list[PooledEngine]] = {}
        self._access_lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown = False
        
        if self.config.enable_auto_cleanup:
            self._start_cleanup_thread()
        
        self._initialized = True
        logger.info("EnginePool initialized")
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        def cleanup_loop():
            while not self._shutdown:
                time.sleep(self.config.cleanup_interval)
                if not self._shutdown:
                    self._cleanup_idle_engines()
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_idle_engines(self) -> None:
        """Remove engines that have been idle too long."""
        with self._access_lock:
            for model_path in list(self._engines.keys()):
                engines = self._engines[model_path]
                
                # Find idle engines
                to_remove = []
                for pe in engines:
                    if not pe.in_use and pe.idle_seconds > self.config.max_idle_seconds:
                        to_remove.append(pe)
                
                # Remove them
                for pe in to_remove:
                    engines.remove(pe)
                    logger.info(f"Cleaned up idle engine for {model_path} "
                               f"(idle {pe.idle_seconds:.0f}s)")
                    # Let garbage collection handle the engine
                
                # Remove empty lists
                if not engines:
                    del self._engines[model_path]
    
    def get_engine(
        self,
        model_path: Optional[str] = None,
        create_if_missing: bool = True,
        **engine_kwargs
    ) -> Optional[Any]:
        """
        Get an engine from the pool.
        
        Args:
            model_path: Path to model (None = use default)
            create_if_missing: Create new engine if none available
            **engine_kwargs: Arguments for engine creation
        
        Returns:
            EnigmaEngine instance or None if unavailable
        """
        # Import here to avoid circular imports
        from ..config import CONFIG
        from .inference import EnigmaEngine

        # Determine model path
        if model_path is None:
            model_path = self.config.default_model_path or str(CONFIG.models_dir / "default")
        
        with self._access_lock:
            # Look for available engine
            if model_path in self._engines:
                for pe in self._engines[model_path]:
                    if not pe.in_use:
                        pe.mark_used()
                        logger.debug(f"Reusing pooled engine for {model_path} "
                                    f"(use #{pe.use_count})")
                        return pe.engine
            
            # No available engine, create if allowed
            if not create_if_missing:
                return None
            
            # Check pool size limit
            total_engines = sum(len(engines) for engines in self._engines.values())
            if total_engines >= self.config.max_pool_size:
                logger.warning(f"Engine pool at capacity ({total_engines}/{self.config.max_pool_size})")
                # Try to evict oldest idle engine
                oldest_idle = self._find_oldest_idle()
                if oldest_idle:
                    self._evict_engine(oldest_idle)
                else:
                    logger.error("No idle engines to evict, cannot create new engine")
                    return None
            
            # Create new engine
            logger.info(f"Creating new engine for {model_path}")
            try:
                engine = EnigmaEngine(model_path=model_path, **engine_kwargs)
                pe = PooledEngine(engine=engine, model_path=model_path)
                pe.mark_used()
                
                if model_path not in self._engines:
                    self._engines[model_path] = []
                self._engines[model_path].append(pe)
                
                return engine
            
            except Exception as e:
                logger.error(f"Failed to create engine: {e}")
                return None
    
    def _find_oldest_idle(self) -> Optional[PooledEngine]:
        """Find the oldest idle engine in the pool."""
        oldest = None
        oldest_time = float('inf')
        
        for engines in self._engines.values():
            for pe in engines:
                if not pe.in_use and pe.last_used_at < oldest_time:
                    oldest = pe
                    oldest_time = pe.last_used_at
        
        return oldest
    
    def _evict_engine(self, pe: PooledEngine) -> None:
        """Remove an engine from the pool."""
        for model_path, engines in self._engines.items():
            if pe in engines:
                engines.remove(pe)
                logger.info(f"Evicted engine for {model_path}")
                if not engines:
                    del self._engines[model_path]
                return
    
    def release_engine(self, engine: Any) -> None:
        """
        Release an engine back to the pool.
        
        Args:
            engine: Engine to release
        """
        with self._access_lock:
            for engines in self._engines.values():
                for pe in engines:
                    if pe.engine is engine:
                        pe.mark_released()
                        logger.debug(f"Engine released back to pool")
                        return
            
            logger.warning("Released engine not found in pool")
    
    def clear_pool(self) -> None:
        """Clear all engines from the pool."""
        with self._access_lock:
            count = sum(len(engines) for engines in self._engines.values())
            self._engines.clear()
            logger.info(f"Cleared {count} engines from pool")
    
    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        with self._access_lock:
            stats = {
                "total_engines": 0,
                "in_use": 0,
                "idle": 0,
                "models": {},
            }
            
            for model_path, engines in self._engines.items():
                model_stats = {
                    "count": len(engines),
                    "in_use": sum(1 for pe in engines if pe.in_use),
                    "total_uses": sum(pe.use_count for pe in engines),
                }
                stats["models"][model_path] = model_stats
                stats["total_engines"] += model_stats["count"]
                stats["in_use"] += model_stats["in_use"]
            
            stats["idle"] = stats["total_engines"] - stats["in_use"]
            return stats
    
    def shutdown(self) -> None:
        """Shutdown the pool and cleanup."""
        self._shutdown = True
        self.clear_pool()
        logger.info("EnginePool shutdown complete")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_pool: Optional[EnginePool] = None


def get_pool() -> EnginePool:
    """Get the global engine pool instance."""
    global _pool
    if _pool is None:
        _pool = EnginePool()
    return _pool


def get_engine(model_path: Optional[str] = None, **kwargs) -> Optional[Any]:
    """
    Get an engine from the global pool.
    
    Args:
        model_path: Path to model (None = use default)
        **kwargs: Additional engine arguments
    
    Returns:
        EnigmaEngine instance or None
    """
    return get_pool().get_engine(model_path, **kwargs)


def release_engine(engine: Any) -> None:
    """Release an engine back to the global pool."""
    get_pool().release_engine(engine)


@contextmanager
def get_engine_context(model_path: Optional[str] = None, **kwargs):
    """
    Context manager for using a pooled engine.
    
    Usage:
        with get_engine_context() as engine:
            response = engine.chat("Hello!")
    """
    engine = get_engine(model_path, **kwargs)
    if engine is None:
        raise RuntimeError("Could not obtain engine from pool")
    try:
        yield engine
    finally:
        release_engine(engine)


def create_fallback_response(error: str = "AI unavailable") -> str:
    """
    Create a graceful fallback response when AI is unavailable.
    
    Args:
        error: Error description
    
    Returns:
        User-friendly fallback message
    """
    return (
        f"I'm having trouble processing that right now. "
        f"({error}) "
        f"Please try again in a moment, or check that a model is loaded."
    )
