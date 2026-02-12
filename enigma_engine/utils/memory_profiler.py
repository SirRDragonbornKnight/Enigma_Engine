"""
Memory Profiling Utility

Tools for profiling memory usage during inference and long conversations.
Helps identify memory leaks and optimize resource allocation.

Usage:
    from enigma_engine.utils.memory_profiler import MemoryProfiler, profile_function
    
    # Profile a function
    @profile_function
    def my_function():
        ...
    
    # Or use the profiler directly
    profiler = MemoryProfiler()
    profiler.start()
    # ... do work ...
    report = profiler.stop()
    print(report)
"""

import gc
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.debug("psutil not available - limited memory profiling")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class MemorySnapshot:
    """A snapshot of memory usage at a point in time."""
    timestamp: float
    label: str = ""
    
    # System memory (bytes)
    rss: int = 0                    # Resident Set Size
    vms: int = 0                    # Virtual Memory Size
    shared: int = 0                 # Shared memory
    
    # Python memory
    python_allocated: int = 0       # Python allocator
    gc_objects: int = 0             # Number of tracked objects
    
    # GPU memory (bytes)
    gpu_allocated: int = 0          # Currently allocated
    gpu_reserved: int = 0           # Reserved by caching allocator
    gpu_max_allocated: int = 0      # Peak allocated
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "label": self.label,
            "system_mb": {
                "rss": self.rss / (1024 * 1024),
                "vms": self.vms / (1024 * 1024),
                "shared": self.shared / (1024 * 1024),
            },
            "python": {
                "allocated_mb": self.python_allocated / (1024 * 1024),
                "gc_objects": self.gc_objects,
            },
            "gpu_mb": {
                "allocated": self.gpu_allocated / (1024 * 1024),
                "reserved": self.gpu_reserved / (1024 * 1024),
                "max_allocated": self.gpu_max_allocated / (1024 * 1024),
            },
        }
    
    def __str__(self) -> str:
        """Human-readable string."""
        parts = [f"[{self.label}]" if self.label else ""]
        parts.append(f"RSS: {self.rss / (1024**2):.1f}MB")
        if self.gpu_allocated > 0:
            parts.append(f"GPU: {self.gpu_allocated / (1024**2):.1f}MB")
        return " | ".join(filter(None, parts))


@dataclass
class MemoryReport:
    """Report of memory usage over a profiling session."""
    start_time: float
    end_time: float
    snapshots: list[MemorySnapshot] = field(default_factory=list)
    
    # Deltas
    rss_delta: int = 0
    gpu_delta: int = 0
    
    # Peak values
    peak_rss: int = 0
    peak_gpu: int = 0
    
    def summary(self) -> str:
        """Generate summary string."""
        duration = self.end_time - self.start_time
        lines = [
            "=" * 50,
            "MEMORY PROFILE REPORT",
            "=" * 50,
            f"Duration: {duration:.2f}s",
            f"Snapshots: {len(self.snapshots)}",
            "",
            "System Memory:",
            f"  RSS Delta: {self.rss_delta / (1024**2):+.1f} MB",
            f"  Peak RSS:  {self.peak_rss / (1024**2):.1f} MB",
        ]
        
        if self.peak_gpu > 0:
            lines.extend([
                "",
                "GPU Memory:",
                f"  Delta:     {self.gpu_delta / (1024**2):+.1f} MB",
                f"  Peak:      {self.peak_gpu / (1024**2):.1f} MB",
            ])
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "duration_seconds": self.end_time - self.start_time,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "summary": {
                "rss_delta_mb": self.rss_delta / (1024 * 1024),
                "peak_rss_mb": self.peak_rss / (1024 * 1024),
                "gpu_delta_mb": self.gpu_delta / (1024 * 1024),
                "peak_gpu_mb": self.peak_gpu / (1024 * 1024),
            }
        }


class MemoryProfiler:
    """
    Memory profiler for tracking usage during operations.
    
    Usage:
        profiler = MemoryProfiler()
        profiler.start()
        
        # ... do work ...
        profiler.snapshot("after_load")
        
        # ... more work ...
        profiler.snapshot("after_inference")
        
        report = profiler.stop()
        print(report.summary())
    """
    
    def __init__(self, interval: float = 0.0):
        """
        Initialize profiler.
        
        Args:
            interval: If > 0, automatically take snapshots at this interval (seconds)
        """
        self._interval = interval
        self._snapshots: list[MemorySnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._start_time = 0.0
        self._process = None
        
        if PSUTIL_AVAILABLE:
            self._process = psutil.Process(os.getpid())
    
    def _take_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a memory snapshot."""
        snap = MemorySnapshot(
            timestamp=time.time(),
            label=label
        )
        
        # System memory via psutil
        if self._process:
            try:
                mem_info = self._process.memory_info()
                snap.rss = mem_info.rss
                snap.vms = mem_info.vms
                if hasattr(mem_info, 'shared'):
                    snap.shared = mem_info.shared
            except Exception as e:
                logger.debug(f"Failed to get process memory: {e}")
        
        # Python memory
        gc.collect()
        snap.gc_objects = len(gc.get_objects())
        
        # GPU memory
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                snap.gpu_allocated = torch.cuda.memory_allocated()
                snap.gpu_reserved = torch.cuda.memory_reserved()
                snap.gpu_max_allocated = torch.cuda.max_memory_allocated()
            except Exception as e:
                logger.debug(f"Failed to get GPU memory: {e}")
        
        return snap
    
    def start(self):
        """Start profiling."""
        self._snapshots = []
        self._start_time = time.time()
        self._running = True
        
        # Take initial snapshot
        initial = self._take_snapshot("start")
        with self._lock:
            self._snapshots.append(initial)
        
        # Start background thread if interval specified
        if self._interval > 0:
            self._thread = threading.Thread(target=self._background_snapshots, daemon=True)
            self._thread.start()
        
        logger.debug("Memory profiler started")
    
    def _background_snapshots(self):
        """Background thread for automatic snapshots."""
        count = 0
        while self._running:
            time.sleep(self._interval)
            if self._running:
                count += 1
                self.snapshot(f"auto_{count}")
    
    def snapshot(self, label: str = ""):
        """Take a labeled snapshot."""
        snap = self._take_snapshot(label)
        with self._lock:
            self._snapshots.append(snap)
        logger.debug(f"Memory snapshot: {snap}")
    
    def stop(self) -> MemoryReport:
        """Stop profiling and generate report."""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        
        # Take final snapshot
        final = self._take_snapshot("end")
        with self._lock:
            self._snapshots.append(final)
        
        # Generate report
        report = self._generate_report()
        logger.debug("Memory profiler stopped")
        return report
    
    def _generate_report(self) -> MemoryReport:
        """Generate report from snapshots."""
        with self._lock:
            snapshots = list(self._snapshots)
        
        if not snapshots:
            return MemoryReport(start_time=self._start_time, end_time=time.time())
        
        first = snapshots[0]
        last = snapshots[-1]
        
        report = MemoryReport(
            start_time=first.timestamp,
            end_time=last.timestamp,
            snapshots=snapshots,
            rss_delta=last.rss - first.rss,
            gpu_delta=last.gpu_allocated - first.gpu_allocated,
            peak_rss=max(s.rss for s in snapshots),
            peak_gpu=max(s.gpu_allocated for s in snapshots),
        )
        
        return report
    
    def get_current(self) -> MemorySnapshot:
        """Get current memory usage without recording."""
        return self._take_snapshot()


@contextmanager
def profile_memory(label: str = "operation"):
    """
    Context manager for profiling a block of code.
    
    Usage:
        with profile_memory("model_loading"):
            model = load_model()
    """
    profiler = MemoryProfiler()
    profiler.start()
    try:
        yield profiler
    finally:
        report = profiler.stop()
        logger.info(f"Memory profile [{label}]:\n{report.summary()}")


def profile_function(func: Optional[Callable] = None, *, label: Optional[str] = None):
    """
    Decorator to profile a function's memory usage.
    
    Usage:
        @profile_function
        def my_function():
            ...
        
        @profile_function(label="custom_label")
        def another_function():
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            fn_label = label or fn.__name__
            with profile_memory(fn_label):
                return fn(*args, **kwargs)
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


def get_memory_usage() -> dict[str, float]:
    """
    Get current memory usage in MB.
    
    Returns:
        Dictionary with memory values in MB
    """
    result = {
        "rss_mb": 0.0,
        "vms_mb": 0.0,
        "gpu_allocated_mb": 0.0,
        "gpu_reserved_mb": 0.0,
    }
    
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            result["rss_mb"] = mem_info.rss / (1024 * 1024)
            result["vms_mb"] = mem_info.vms / (1024 * 1024)
        except Exception:
            pass  # Intentionally silent
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            result["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            result["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        except Exception:
            pass  # Intentionally silent
    
    return result


# Note: Use format_bytes from enigma_engine.utils for byte formatting


# Singleton profiler 
_global_profiler: Optional[MemoryProfiler] = None


def get_profiler() -> MemoryProfiler:
    """Get or create global memory profiler."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = MemoryProfiler()
    return _global_profiler
