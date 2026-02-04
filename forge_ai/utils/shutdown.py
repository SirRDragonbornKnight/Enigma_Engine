"""
================================================================================
GRACEFUL SHUTDOWN - Clean application termination
================================================================================

Handles clean shutdown of ForgeAI components when the application exits.

Features:
- Register cleanup callbacks from any module
- Handle SIGTERM, SIGINT, SIGHUP signals
- atexit handler for normal termination
- Timeout enforcement for stuck cleanups
- Thread-safe callback management

FILE: forge_ai/utils/shutdown.py
TYPE: System Utility
MAIN CLASSES: ShutdownManager

USAGE:
    from forge_ai.utils.shutdown import get_shutdown_manager, on_shutdown
    
    # Register a cleanup callback
    @on_shutdown
    def cleanup_my_resources():
        close_connections()
        save_state()
    
    # Or register manually
    manager = get_shutdown_manager()
    manager.register(cleanup_function, priority=100)
    
    # Trigger shutdown programmatically
    manager.shutdown()
"""

from __future__ import annotations

import atexit
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ShutdownCallback:
    """A registered shutdown callback."""
    
    callback: Callable[[], None]
    name: str
    priority: int = 50  # Higher = runs first
    timeout: float = 5.0  # Max time for this callback
    
    def __lt__(self, other):
        # Higher priority runs first
        return self.priority > other.priority


class ShutdownManager:
    """
    Manages graceful application shutdown.
    
    Coordinates cleanup across all ForgeAI components by:
    - Registering cleanup callbacks with priorities
    - Handling OS signals (SIGTERM, SIGINT)
    - Running cleanup in priority order with timeouts
    - Tracking shutdown state to prevent duplicate cleanup
    """
    
    _instance: Optional['ShutdownManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the shutdown manager."""
        if ShutdownManager._initialized:
            return
        ShutdownManager._initialized = True
        
        self._callbacks: List[ShutdownCallback] = []
        self._lock = threading.RLock()
        self._shutdown_started = False
        self._shutdown_complete = False
        self._default_timeout = 30.0  # Total shutdown timeout
        
        # Register signal handlers
        self._register_signals()
        
        # Register atexit handler
        atexit.register(self._atexit_handler)
        
        logger.debug("ShutdownManager initialized")
    
    def _register_signals(self):
        """Register handlers for termination signals."""
        # Only register signal handlers in main thread
        if threading.current_thread() is not threading.main_thread():
            logger.debug("Not in main thread, skipping signal registration")
            return
        
        try:
            # SIGTERM (kill command, systemd stop)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # SIGINT (Ctrl+C)
            signal.signal(signal.SIGINT, self._signal_handler)
            
            # SIGHUP (terminal closed) - Unix only
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, self._signal_handler)
            
            logger.debug("Signal handlers registered")
            
        except Exception as e:
            logger.warning(f"Could not register signal handlers: {e}")
    
    def _signal_handler(self, signum: int, frame):
        """Handle termination signals."""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name}, initiating shutdown...")
        
        # Run shutdown in a thread to not block signal handler
        thread = threading.Thread(target=self.shutdown, name="ShutdownThread")
        thread.start()
        
        # For SIGINT (Ctrl+C), wait a bit then force exit
        if signum == signal.SIGINT:
            thread.join(timeout=self._default_timeout)
            if thread.is_alive():
                logger.warning("Shutdown taking too long, forcing exit")
                sys.exit(1)
    
    def _atexit_handler(self):
        """Handle normal program termination."""
        if not self._shutdown_started:
            logger.debug("atexit handler triggered")
            self.shutdown()
    
    def register(
        self,
        callback: Callable[[], None],
        name: str = None,
        priority: int = 50,
        timeout: float = 5.0
    ) -> None:
        """
        Register a cleanup callback.
        
        Args:
            callback: Function to call during shutdown
            name: Descriptive name for logging
            priority: Higher priority runs first (default 50)
            timeout: Max seconds to wait for this callback
        
        Priority guidelines:
            100: Critical state saving (model checkpoints)
            80: Network connections close
            60: File handles close
            50: General cleanup (default)
            30: Optional cleanup
            10: Final logging
        """
        with self._lock:
            if self._shutdown_started:
                logger.warning(f"Cannot register callback '{name}' - shutdown already started")
                return
            
            cb = ShutdownCallback(
                callback=callback,
                name=name or callback.__name__,
                priority=priority,
                timeout=timeout
            )
            self._callbacks.append(cb)
            logger.debug(f"Registered shutdown callback: {cb.name} (priority={priority})")
    
    def unregister(self, callback: Callable[[], None]) -> bool:
        """
        Unregister a cleanup callback.
        
        Args:
            callback: The callback function to remove
            
        Returns:
            True if callback was found and removed
        """
        with self._lock:
            for i, cb in enumerate(self._callbacks):
                if cb.callback is callback:
                    del self._callbacks[i]
                    logger.debug(f"Unregistered shutdown callback: {cb.name}")
                    return True
            return False
    
    def shutdown(self, timeout: float = None) -> bool:
        """
        Execute graceful shutdown.
        
        Args:
            timeout: Total timeout in seconds (default: 30)
            
        Returns:
            True if all callbacks completed successfully
        """
        with self._lock:
            if self._shutdown_complete:
                return True
            
            if self._shutdown_started:
                logger.debug("Shutdown already in progress")
                return False
            
            self._shutdown_started = True
        
        timeout = timeout or self._default_timeout
        start_time = time.time()
        all_success = True
        
        logger.info("Starting graceful shutdown...")
        
        # Sort callbacks by priority (higher first)
        with self._lock:
            callbacks = sorted(self._callbacks.copy())
        
        for cb in callbacks:
            # Check total timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.warning(f"Shutdown timeout reached, skipping remaining callbacks")
                all_success = False
                break
            
            # Run callback with its individual timeout
            remaining = min(cb.timeout, timeout - elapsed)
            success = self._run_callback(cb, remaining)
            if not success:
                all_success = False
        
        elapsed = time.time() - start_time
        logger.info(f"Shutdown complete in {elapsed:.2f}s (success={all_success})")
        
        self._shutdown_complete = True
        return all_success
    
    def _run_callback(self, cb: ShutdownCallback, timeout: float) -> bool:
        """Run a single callback with timeout."""
        result = [False]  # Use list for thread-safe mutation
        exception = [None]
        
        def run_with_timeout():
            try:
                cb.callback()
                result[0] = True
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=run_with_timeout, name=f"Shutdown-{cb.name}")
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            logger.warning(f"Shutdown callback '{cb.name}' timed out after {timeout}s")
            return False
        
        if exception[0]:
            logger.error(f"Shutdown callback '{cb.name}' failed: {exception[0]}")
            return False
        
        logger.debug(f"Shutdown callback '{cb.name}' completed")
        return result[0]
    
    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._shutdown_started
    
    @property
    def is_shutdown_complete(self) -> bool:
        """Check if shutdown has completed."""
        return self._shutdown_complete
    
    @classmethod
    def reset(cls):
        """Reset the singleton (for testing only)."""
        cls._instance = None
        cls._initialized = False


# Global singleton access
_manager: Optional[ShutdownManager] = None


def get_shutdown_manager() -> ShutdownManager:
    """Get the global shutdown manager instance."""
    global _manager
    if _manager is None:
        _manager = ShutdownManager()
    return _manager


def on_shutdown(
    priority: int = 50,
    timeout: float = 5.0,
    name: str = None
):
    """
    Decorator to register a function as a shutdown callback.
    
    Usage:
        @on_shutdown(priority=80)
        def cleanup_connections():
            for conn in connections:
                conn.close()
    """
    def decorator(func: Callable[[], None]) -> Callable[[], None]:
        get_shutdown_manager().register(
            func,
            name=name or func.__name__,
            priority=priority,
            timeout=timeout
        )
        return func
    return decorator


def register_shutdown(
    callback: Callable[[], None],
    name: str = None,
    priority: int = 50,
    timeout: float = 5.0
) -> None:
    """
    Register a shutdown callback.
    
    Convenience function wrapping ShutdownManager.register().
    """
    get_shutdown_manager().register(callback, name, priority, timeout)


def shutdown_now(timeout: float = 30.0) -> bool:
    """
    Trigger immediate graceful shutdown.
    
    Returns:
        True if all cleanup completed successfully
    """
    return get_shutdown_manager().shutdown(timeout)


def is_shutting_down() -> bool:
    """Check if shutdown is in progress."""
    return get_shutdown_manager().is_shutting_down


# =============================================================================
# Built-in cleanup registrations
# =============================================================================

def _register_builtin_cleanups():
    """Register cleanup for core ForgeAI components."""
    manager = get_shutdown_manager()
    
    # Module manager cleanup (high priority - save state)
    def cleanup_modules():
        try:
            from forge_ai.modules.manager import get_manager
            mgr = get_manager()
            
            # Save module configuration
            try:
                mgr.save_config()
            except Exception as e:
                logger.warning(f"Failed to save module config: {e}")
            
            # Unload all modules
            for module_id in list(mgr.modules.keys()):
                try:
                    mgr.unload(module_id)
                except Exception as e:
                    logger.warning(f"Failed to unload module {module_id}: {e}")
                    
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Module cleanup failed: {e}")
    
    manager.register(cleanup_modules, "ModuleManager cleanup", priority=90, timeout=10.0)
    
    # Voice pipeline cleanup
    def cleanup_voice():
        try:
            from forge_ai.voice.voice_pipeline import _pipeline
            if _pipeline:
                _pipeline.stop()
        except Exception as e:
            logger.debug(f"Voice cleanup: {e}")
    
    manager.register(cleanup_voice, "VoicePipeline cleanup", priority=70, timeout=5.0)
    
    # API server cleanup
    def cleanup_api():
        try:
            from forge_ai.web.server import shutdown_server
            shutdown_server()
        except Exception as e:
            logger.debug(f"API cleanup: {e}")
    
    manager.register(cleanup_api, "API Server cleanup", priority=80, timeout=5.0)
    
    # Memory/conversation cleanup
    def cleanup_memory():
        try:
            from forge_ai.memory.manager import get_conversation_manager
            manager = get_conversation_manager()
            if hasattr(manager, 'save_all'):
                manager.save_all()
        except Exception as e:
            logger.debug(f"Memory cleanup: {e}")
    
    manager.register(cleanup_memory, "Memory cleanup", priority=85, timeout=5.0)
    
    logger.debug("Built-in cleanup handlers registered")


# Register builtin cleanups on module import
_register_builtin_cleanups()


__all__ = [
    'ShutdownManager',
    'get_shutdown_manager',
    'on_shutdown',
    'register_shutdown',
    'shutdown_now',
    'is_shutting_down',
]
