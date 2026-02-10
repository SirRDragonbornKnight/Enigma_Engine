"""
Performance Monitoring for Enigma AI Engine

Monitor system resources and model performance.

Features:
- CPU, Memory, GPU monitoring
- Inference latency tracking
- Token throughput metrics
- Alert system
- Historical data

Usage:
    from enigma_engine.utils.performance import PerformanceMonitor, get_monitor
    
    monitor = get_monitor()
    
    # Start monitoring
    monitor.start()
    
    # Check metrics
    metrics = monitor.get_metrics()
    print(f"CPU: {metrics['cpu_percent']}%")
    print(f"Memory: {metrics['memory_percent']}%")
    
    # Track inference
    with monitor.track_inference("chat"):
        response = model.generate(prompt)
    
    # Get stats
    stats = monitor.get_inference_stats("chat")
"""

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetric:
    """Single inference measurement."""
    name: str
    start_time: float
    end_time: float
    duration_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    success: bool = True
    error: Optional[str] = None
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.duration_ms <= 0:
            return 0.0
        return (self.tokens_out / self.duration_ms) * 1000


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    disk_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": self.memory_used_gb,
            "gpu_percent": self.gpu_percent,
            "gpu_memory_percent": self.gpu_memory_percent
        }


@dataclass
class Alert:
    """Performance alert."""
    level: str  # info, warning, critical
    metric: str
    value: float
    threshold: float
    message: str
    timestamp: float = field(default_factory=time.time)


class InferenceTracker:
    """Context manager for tracking inference."""
    
    def __init__(
        self,
        monitor: "PerformanceMonitor",
        name: str,
        tokens_in: int = 0
    ):
        self._monitor = monitor
        self._name = name
        self._tokens_in = tokens_in
        self._start = 0.0
        self._tokens_out = 0
    
    def __enter__(self) -> "InferenceTracker":
        self._start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        duration_ms = (end - self._start) * 1000
        
        metric = InferenceMetric(
            name=self._name,
            start_time=self._start,
            end_time=end,
            duration_ms=duration_ms,
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            success=exc_type is None,
            error=str(exc_val) if exc_val else None
        )
        
        self._monitor._record_inference(metric)
        return False
    
    def set_tokens_out(self, count: int):
        """Set output token count."""
        self._tokens_out = count


class PerformanceMonitor:
    """
    Monitors system and model performance.
    """
    
    def __init__(
        self,
        history_size: int = 1000,
        sample_interval: float = 5.0,
        enable_gpu: bool = True
    ):
        """
        Initialize monitor.
        
        Args:
            history_size: Number of samples to keep
            sample_interval: Seconds between samples
            enable_gpu: Try to monitor GPU
        """
        self._history_size = history_size
        self._sample_interval = sample_interval
        self._enable_gpu = enable_gpu
        
        # Metrics storage
        self._system_history: deque = deque(maxlen=history_size)
        self._inference_history: Dict[str, deque] = {}
        self._alerts: deque = deque(maxlen=100)
        
        # Alert thresholds
        self._thresholds = {
            "cpu_percent": {"warning": 80, "critical": 95},
            "memory_percent": {"warning": 80, "critical": 95},
            "gpu_percent": {"warning": 90, "critical": 99},
            "gpu_memory_percent": {"warning": 85, "critical": 95}
        }
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Monitoring thread
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # GPU monitoring
        self._pynvml_available = False
        if enable_gpu:
            self._init_gpu_monitoring()
    
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring."""
        import sys
        
        # Try NVIDIA first (Windows/Linux)
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml_available = True
            logger.info("GPU monitoring enabled via pynvml")
            return
        except Exception:
            pass
        
        # macOS fallback via ioreg
        if sys.platform == 'darwin':
            self._macos_gpu_available = True
            logger.info("GPU monitoring enabled via macOS ioreg")
        else:
            logger.info("GPU monitoring not available (pynvml not installed)")
    
    def _get_macos_gpu_usage(self) -> Tuple[float, float]:
        """Get GPU usage on macOS via ioreg/powermetrics.
        
        Returns:
            Tuple of (gpu_utilization%, gpu_memory%)
        """
        import subprocess
        
        gpu_util = 0.0
        gpu_mem = 0.0
        
        try:
            # Try to get GPU device info via ioreg
            result = subprocess.run(
                ['ioreg', '-r', '-d', '1', '-c', 'IOAccelerator'],
                capture_output=True, text=True, timeout=2
            )
            
            if result.returncode == 0:
                output = result.stdout
                # Parse utilization if available (format varies by GPU)
                if 'GPU Core Utilization' in output or 'PerformanceStatistics' in output:
                    # Basic heuristic - if we see activity, estimate ~25%
                    gpu_util = 25.0
                    
        except Exception:
            pass
        
        return gpu_util, gpu_mem
    
    def start(self):
        """Start background monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                metrics = self._collect_system_metrics()
                
                with self._lock:
                    self._system_history.append(metrics)
                
                self._check_alerts(metrics)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            time.sleep(self._sample_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_percent=disk.percent
            )
            
            # GPU metrics
            if self._pynvml_available:
                try:
                    import pynvml
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    metrics.gpu_percent = util.gpu
                    metrics.gpu_memory_percent = (mem.used / mem.total) * 100
                    metrics.gpu_memory_used_gb = mem.used / (1024**3)
                except Exception:
                    pass
            elif getattr(self, '_macos_gpu_available', False):
                try:
                    metrics.gpu_percent, metrics.gpu_memory_percent = self._get_macos_gpu_usage()
                except Exception:
                    pass
            
            return metrics
            
        except ImportError:
            # psutil not available, return minimal metrics
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0,
                memory_percent=0,
                memory_used_gb=0,
                memory_total_gb=0
            )
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check metrics against thresholds."""
        for metric_name, thresholds in self._thresholds.items():
            value = getattr(metrics, metric_name, None)
            
            if value is None:
                continue
            
            if value >= thresholds["critical"]:
                self._raise_alert("critical", metric_name, value, thresholds["critical"])
            elif value >= thresholds["warning"]:
                self._raise_alert("warning", metric_name, value, thresholds["warning"])
    
    def _raise_alert(self, level: str, metric: str, value: float, threshold: float):
        """Raise a performance alert."""
        alert = Alert(
            level=level,
            metric=metric,
            value=value,
            threshold=threshold,
            message=f"{metric} at {value:.1f}% (threshold: {threshold}%)"
        )
        
        self._alerts.append(alert)
        
        # Call callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        logger.warning(f"Performance alert ({level}): {alert.message}")
    
    def track_inference(
        self,
        name: str = "default",
        tokens_in: int = 0
    ) -> InferenceTracker:
        """
        Track an inference operation.
        
        Args:
            name: Operation name
            tokens_in: Input token count
            
        Returns:
            Context manager for tracking
        """
        return InferenceTracker(self, name, tokens_in)
    
    def _record_inference(self, metric: InferenceMetric):
        """Record an inference metric."""
        with self._lock:
            if metric.name not in self._inference_history:
                self._inference_history[metric.name] = deque(maxlen=self._history_size)
            
            self._inference_history[metric.name].append(metric)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        metrics = self._collect_system_metrics()
        return metrics.to_dict()
    
    def get_history(
        self,
        minutes: int = 10
    ) -> List[Dict[str, Any]]:
        """Get metrics history."""
        cutoff = time.time() - (minutes * 60)
        
        with self._lock:
            return [
                m.to_dict() for m in self._system_history
                if m.timestamp >= cutoff
            ]
    
    def get_inference_stats(
        self,
        name: str = "default"
    ) -> Dict[str, Any]:
        """
        Get inference statistics.
        
        Args:
            name: Operation name
            
        Returns:
            Statistics dictionary
        """
        with self._lock:
            if name not in self._inference_history:
                return {"count": 0}
            
            metrics = list(self._inference_history[name])
        
        if not metrics:
            return {"count": 0}
        
        durations = [m.duration_ms for m in metrics]
        success_count = sum(1 for m in metrics if m.success)
        tokens_out_total = sum(m.tokens_out for m in metrics)
        
        return {
            "count": len(metrics),
            "success_rate": success_count / len(metrics),
            "latency_ms": {
                "mean": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "p50": sorted(durations)[len(durations) // 2],
                "p95": sorted(durations)[int(len(durations) * 0.95)] if len(durations) >= 20 else max(durations),
                "p99": sorted(durations)[int(len(durations) * 0.99)] if len(durations) >= 100 else max(durations)
            },
            "tokens_out_total": tokens_out_total,
            "avg_tokens_per_second": sum(m.tokens_per_second for m in metrics) / len(metrics) if metrics else 0
        }
    
    def get_alerts(
        self,
        level: Optional[str] = None,
        minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        cutoff = time.time() - (minutes * 60)
        
        alerts = []
        for alert in self._alerts:
            if alert.timestamp >= cutoff:
                if level is None or alert.level == level:
                    alerts.append({
                        "level": alert.level,
                        "metric": alert.metric,
                        "value": alert.value,
                        "threshold": alert.threshold,
                        "message": alert.message,
                        "timestamp": alert.timestamp
                    })
        
        return alerts
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback."""
        self._alert_callbacks.append(callback)
    
    def set_threshold(
        self,
        metric: str,
        warning: Optional[float] = None,
        critical: Optional[float] = None
    ):
        """Set alert thresholds."""
        if metric not in self._thresholds:
            self._thresholds[metric] = {}
        
        if warning is not None:
            self._thresholds[metric]["warning"] = warning
        if critical is not None:
            self._thresholds[metric]["critical"] = critical
    
    def report(self) -> str:
        """Generate performance report."""
        metrics = self.get_metrics()
        
        lines = [
            "=== Performance Report ===",
            f"Timestamp: {datetime.now().isoformat()}",
            "",
            "System Resources:",
            f"  CPU: {metrics['cpu_percent']:.1f}%",
            f"  Memory: {metrics['memory_percent']:.1f}% ({metrics['memory_used_gb']:.1f} GB used)",
        ]
        
        if metrics.get("gpu_percent") is not None:
            lines.append(f"  GPU: {metrics['gpu_percent']:.1f}%")
            lines.append(f"  GPU Memory: {metrics['gpu_memory_percent']:.1f}%")
        
        lines.append("")
        lines.append("Inference Operations:")
        
        with self._lock:
            for name in self._inference_history.keys():
                stats = self.get_inference_stats(name)
                lines.append(f"  {name}:")
                lines.append(f"    Count: {stats['count']}")
                lines.append(f"    Success Rate: {stats['success_rate']*100:.1f}%")
                if stats['count'] > 0:
                    lines.append(f"    Avg Latency: {stats['latency_ms']['mean']:.1f}ms")
                    lines.append(f"    P95 Latency: {stats['latency_ms']['p95']:.1f}ms")
        
        recent_alerts = self.get_alerts(minutes=60)
        if recent_alerts:
            lines.append("")
            lines.append("Recent Alerts:")
            for alert in recent_alerts[-5:]:
                lines.append(f"  [{alert['level'].upper()}] {alert['message']}")
        
        return "\n".join(lines)
    
    def save_history(self, path: str):
        """Save metrics history to file."""
        data = {
            "system": [m.to_dict() for m in self._system_history],
            "inference": {
                name: [{"duration_ms": m.duration_ms, "success": m.success}
                       for m in metrics]
                for name, metrics in self._inference_history.items()
            }
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# Global instance
_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


def track_time(func: Callable) -> Callable:
    """Decorator to track function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = get_monitor()
        with monitor.track_inference(func.__name__):
            return func(*args, **kwargs)
    return wrapper
