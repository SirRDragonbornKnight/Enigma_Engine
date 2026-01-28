"""
Resource Limiter - Enforce resource usage limits for game mode

Ensures AI stays within specified CPU, RAM, and GPU limits.
"""

import psutil
import threading
import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """Resource limits for game mode."""
    max_cpu_percent: float = 5.0      # Max CPU usage
    max_memory_mb: int = 500          # Max RAM usage
    gpu_allowed: bool = False          # Use GPU at all?
    background_tasks: bool = False     # Allow autonomous mode?
    inference_allowed: bool = True     # Can respond if asked?
    max_response_tokens: int = 50      # Shorter responses
    batch_processing: bool = False     # No batch operations


class ResourceLimiter:
    """
    Monitor and enforce resource limits.
    
    Tracks CPU and memory usage, triggers callbacks when limits exceeded.
    """
    
    def __init__(self, limits: ResourceLimits):
        """
        Initialize resource limiter.
        
        Args:
            limits: Resource limits to enforce
        """
        self.limits = limits
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._current_process = psutil.Process()
        
        # Callbacks
        self._on_limit_exceeded: list[Callable[[str, float], None]] = []
        self._on_limit_ok: list[Callable[[], None]] = []
    
    def start_monitoring(self, interval: float = 1.0):
        """
        Start monitoring resource usage.
        
        Args:
            interval: Check interval in seconds
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        self._stop_event.clear()
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True,
            name="ResourceLimiter"
        )
        self._monitor_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring resource usage."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._check_limits()
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
            
            self._stop_event.wait(interval)
    
    def _check_limits(self):
        """Check if resource usage is within limits."""
        # Get current usage
        cpu_percent = self._current_process.cpu_percent(interval=0.1)
        memory_mb = self._current_process.memory_info().rss / (1024 * 1024)
        
        # Check CPU limit
        if cpu_percent > self.limits.max_cpu_percent:
            self._notify_limit_exceeded("CPU", cpu_percent)
        
        # Check memory limit
        if memory_mb > self.limits.max_memory_mb:
            self._notify_limit_exceeded("Memory", memory_mb)
    
    def _notify_limit_exceeded(self, resource: str, value: float):
        """Notify that a limit was exceeded."""
        for callback in self._on_limit_exceeded:
            try:
                callback(resource, value)
            except Exception as e:
                logger.error(f"Limit exceeded callback error: {e}")
    
    def _notify_limit_ok(self):
        """Notify that limits are OK."""
        for callback in self._on_limit_ok:
            try:
                callback()
            except Exception as e:
                logger.error(f"Limit OK callback error: {e}")
    
    def update_limits(self, limits: ResourceLimits):
        """
        Update resource limits.
        
        Args:
            limits: New resource limits
        """
        self.limits = limits
        logger.info(f"Updated limits: CPU={limits.max_cpu_percent}%, RAM={limits.max_memory_mb}MB")
    
    def get_current_usage(self) -> dict:
        """
        Get current resource usage.
        
        Returns:
            Dictionary with cpu_percent and memory_mb
        """
        try:
            return {
                "cpu_percent": self._current_process.cpu_percent(interval=0.1),
                "memory_mb": self._current_process.memory_info().rss / (1024 * 1024),
            }
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return {"cpu_percent": 0.0, "memory_mb": 0.0}
    
    def is_within_limits(self) -> bool:
        """
        Check if current usage is within limits.
        
        Returns:
            True if within limits, False otherwise
        """
        usage = self.get_current_usage()
        return (
            usage["cpu_percent"] <= self.limits.max_cpu_percent and
            usage["memory_mb"] <= self.limits.max_memory_mb
        )
    
    def on_limit_exceeded(self, callback: Callable[[str, float], None]):
        """
        Register callback for limit exceeded events.
        
        Args:
            callback: Function(resource_name, value) called when limit exceeded
        """
        self._on_limit_exceeded.append(callback)
    
    def on_limit_ok(self, callback: Callable[[], None]):
        """
        Register callback for when limits are OK.
        
        Args:
            callback: Function() called when back within limits
        """
        self._on_limit_ok.append(callback)


__all__ = ['ResourceLimits', 'ResourceLimiter']
