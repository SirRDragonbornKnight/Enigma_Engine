"""
Progress Tracker for Long Operations

Provides progress feedback for operations like model loading, training, etc.
Can be used standalone or integrated with GUI.

Usage:
    from enigma_engine.utils.progress import ProgressTracker, track_progress
    
    # Simple usage with context manager
    with track_progress("Loading model", total=100) as progress:
        for i in range(100):
            do_work()
            progress.update(1)
    
    # Or manually
    tracker = ProgressTracker("Loading model", total=100)
    tracker.start()
    for i in range(100):
        do_work()
        tracker.update(1, status=f"Loading layer {i}")
    tracker.finish()
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProgressState:
    """Current state of a progress operation."""
    task_name: str
    total: Optional[int] = None
    current: int = 0
    status: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    
    @property
    def percent(self) -> Optional[float]:
        """Get completion percentage if total is known."""
        if self.total and self.total > 0:
            return min(100.0, (self.current / self.total) * 100)
        return None
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        end = self.finished_at or time.time()
        return end - self.started_at
    
    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        if self.total and self.current > 0:
            rate = self.current / self.elapsed_seconds
            remaining = self.total - self.current
            return remaining / rate if rate > 0 else None
        return None
    
    @property
    def is_finished(self) -> bool:
        return self.finished_at is not None


class ProgressTracker:
    """
    Track progress of long-running operations.
    
    Features:
    - Track progress with optional total
    - ETA estimation
    - Callbacks for GUI integration
    - Thread-safe updates
    """
    
    # Global callbacks for all trackers (for GUI integration)
    _global_callbacks: list = []
    
    def __init__(
        self,
        task_name: str,
        total: Optional[int] = None,
        callback: Optional[Callable[[ProgressState], None]] = None
    ):
        """
        Initialize progress tracker.
        
        Args:
            task_name: Description of the task
            total: Total steps (if known)
            callback: Function to call on each update
        """
        self.state = ProgressState(task_name=task_name, total=total)
        self._callback = callback
        self._lock = Lock()
    
    def start(self):
        """Mark task as started."""
        with self._lock:
            self.state.started_at = time.time()
            self.state.current = 0
            self.state.finished_at = None
        self._notify()
        logger.debug(f"Started: {self.state.task_name}")
    
    def update(
        self,
        increment: int = 1,
        status: Optional[str] = None,
        current: Optional[int] = None
    ):
        """
        Update progress.
        
        Args:
            increment: How much to add to current progress
            status: Optional status message
            current: Set absolute current value (overrides increment)
        """
        with self._lock:
            if current is not None:
                self.state.current = current
            else:
                self.state.current += increment
            
            if status is not None:
                self.state.status = status
        
        self._notify()
    
    def finish(self, status: str = "Complete"):
        """Mark task as finished."""
        with self._lock:
            self.state.finished_at = time.time()
            self.state.status = status
            if self.state.total:
                self.state.current = self.state.total
        
        self._notify()
        logger.debug(
            f"Finished: {self.state.task_name} "
            f"({self.state.elapsed_seconds:.1f}s)"
        )
    
    def _notify(self):
        """Notify callbacks of state change."""
        # Instance callback
        if self._callback:
            try:
                self._callback(self.state)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
        
        # Global callbacks
        for cb in ProgressTracker._global_callbacks:
            try:
                cb(self.state)
            except Exception as e:
                logger.warning(f"Global progress callback error: {e}")
    
    def format_status(self) -> str:
        """Format progress for terminal display."""
        parts = [self.state.task_name]
        
        if self.state.percent is not None:
            parts.append(f"{self.state.percent:.1f}%")
        elif self.state.current > 0:
            parts.append(f"{self.state.current}")
            if self.state.total:
                parts[-1] += f"/{self.state.total}"
        
        if self.state.status:
            parts.append(f"- {self.state.status}")
        
        if not self.state.is_finished and self.state.eta_seconds:
            parts.append(f"(ETA: {self.state.eta_seconds:.0f}s)")
        
        return " ".join(parts)
    
    @classmethod
    def add_global_callback(cls, callback: Callable[[ProgressState], None]):
        """Add a callback that receives all progress updates."""
        cls._global_callbacks.append(callback)
    
    @classmethod
    def remove_global_callback(cls, callback: Callable[[ProgressState], None]):
        """Remove a global callback."""
        if callback in cls._global_callbacks:
            cls._global_callbacks.remove(callback)


@contextmanager
def track_progress(
    task_name: str,
    total: Optional[int] = None,
    callback: Optional[Callable[[ProgressState], None]] = None
):
    """
    Context manager for tracking progress.
    
    Usage:
        with track_progress("Loading", total=100) as progress:
            for i in range(100):
                do_work()
                progress.update(1)
    """
    tracker = ProgressTracker(task_name, total, callback)
    tracker.start()
    try:
        yield tracker
    finally:
        if not tracker.state.is_finished:
            tracker.finish()


class ConsoleProgressBar:
    """Simple console progress bar for CLI use."""
    
    def __init__(self, total: int, width: int = 40, prefix: str = ""):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0
    
    def update(self, current: Optional[int] = None, increment: int = 1):
        """Update progress bar."""
        if current is not None:
            self.current = current
        else:
            self.current += increment
        
        self._draw()
    
    def _draw(self):
        """Draw the progress bar."""
        if self.total <= 0:
            return
        
        filled = int(self.width * self.current / self.total)
        bar = "=" * filled + "-" * (self.width - filled)
        percent = 100 * self.current / self.total
        
        print(f"\r{self.prefix}[{bar}] {percent:.1f}%", end="", flush=True)
    
    def finish(self):
        """Complete the progress bar."""
        self.current = self.total
        self._draw()
        print()  # Newline


# Integration helpers for module loading

def create_model_loading_progress(callback: Optional[Callable] = None) -> ProgressTracker:
    """Create a progress tracker configured for model loading."""
    return ProgressTracker(
        "Loading model",
        total=100,  # Percent-based
        callback=callback
    )


def model_loading_stages() -> dict:
    """Get standard stages for model loading progress."""
    return {
        "init": (0, "Initializing"),
        "config": (10, "Loading configuration"),
        "weights": (20, "Loading weights"),
        "layers": (60, "Building layers"),
        "compile": (80, "Compiling model"),
        "optimize": (90, "Optimizing"),
        "ready": (100, "Ready"),
    }
