"""
================================================================================
Performance Monitor - Real-time system metrics with alerts.
================================================================================

Monitors CPU, RAM, GPU, and network for:
- Resource usage tracking
- Bottleneck detection
- Performance alerts
- Historical data for optimization

USAGE:
    from forge_ai.utils.performance_monitor import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    monitor.start()
    
    # Get metrics
    metrics = monitor.get_metrics()
    print(f"CPU: {metrics.cpu_percent}%")
    print(f"RAM: {metrics.ram_used_gb:.1f} GB")
    
    # Set alert
    monitor.set_alert("cpu", threshold=90.0, callback=lambda m: print("CPU high!"))
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Current system metrics."""
    # CPU
    cpu_percent: float = 0.0
    cpu_count: int = 1
    cpu_freq_mhz: float = 0.0
    cpu_temp: float = 0.0
    
    # Memory
    ram_total_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_percent: float = 0.0
    
    # GPU
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_memory_total_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_percent: float = 0.0
    gpu_temp: float = 0.0
    
    # Disk
    disk_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_percent: float = 0.0
    
    # Network
    net_sent_mb: float = 0.0
    net_recv_mb: float = 0.0
    net_sent_rate_mbps: float = 0.0
    net_recv_rate_mbps: float = 0.0
    
    # Process
    process_cpu: float = 0.0
    process_ram_mb: float = 0.0
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_percent": self.cpu_percent,
            "cpu_count": self.cpu_count,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "cpu_temp": self.cpu_temp,
            "ram_total_gb": self.ram_total_gb,
            "ram_used_gb": self.ram_used_gb,
            "ram_percent": self.ram_percent,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "gpu_memory_total_gb": self.gpu_memory_total_gb,
            "gpu_memory_used_gb": self.gpu_memory_used_gb,
            "gpu_percent": self.gpu_percent,
            "gpu_temp": self.gpu_temp,
            "disk_total_gb": self.disk_total_gb,
            "disk_used_gb": self.disk_used_gb,
            "disk_percent": self.disk_percent,
            "net_sent_mb": self.net_sent_mb,
            "net_recv_mb": self.net_recv_mb,
            "net_sent_rate_mbps": self.net_sent_rate_mbps,
            "net_recv_rate_mbps": self.net_recv_rate_mbps,
            "process_cpu": self.process_cpu,
            "process_ram_mb": self.process_ram_mb,
            "timestamp": self.timestamp,
        }


@dataclass
class PerformanceAlert:
    """Alert configuration."""
    metric: str           # cpu, ram, gpu, etc.
    threshold: float      # Alert when above this
    callback: Callable[[SystemMetrics], None]
    cooldown: float = 60.0  # Seconds between alerts
    last_triggered: float = 0.0


class PerformanceMonitor:
    """
    Real-time system performance monitoring.
    
    Features:
    - CPU, RAM, GPU, disk, network monitoring
    - Historical data collection
    - Alert system for thresholds
    - Bottleneck detection
    """
    
    def __init__(
        self,
        update_interval: float = 1.0,
        history_size: int = 3600,  # 1 hour at 1s interval
    ):
        self.update_interval = update_interval
        self.history_size = history_size
        
        # Current metrics
        self._metrics = SystemMetrics()
        self._lock = threading.Lock()
        
        # History
        self._history: deque = deque(maxlen=history_size)
        
        # Alerts
        self._alerts: List[PerformanceAlert] = []
        
        # Network tracking
        self._last_net_sent = 0.0
        self._last_net_recv = 0.0
        self._last_net_time = time.time()
        
        # Threading
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Initialize readers
        self._init_readers()
    
    def _init_readers(self):
        """Initialize metric readers."""
        # Check for psutil
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            self._psutil = None
            logger.warning("psutil not available, limited metrics")
        
        # Check for GPU monitoring
        self._gpu_reader = self._init_gpu_reader()
    
    def _init_gpu_reader(self):
        """Initialize GPU monitoring."""
        # Try pynvml for NVIDIA
        try:
            import pynvml
            pynvml.nvmlInit()
            return {"type": "nvidia", "lib": pynvml}
        except Exception:
            pass
        
        # Try GPUtil as fallback
        try:
            import GPUtil
            return {"type": "gputil", "lib": GPUtil}
        except Exception:
            pass
        
        return None
    
    def start(self):
        """Start monitoring."""
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitor started")
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        
        # Cleanup GPU
        if self._gpu_reader and self._gpu_reader["type"] == "nvidia":
            try:
                self._gpu_reader["lib"].nvmlShutdown()
            except Exception:
                pass
        
        logger.info("Performance monitor stopped")
    
    def get_metrics(self) -> SystemMetrics:
        """Get current metrics."""
        with self._lock:
            return SystemMetrics(**self._metrics.to_dict())
    
    def get_history(self, seconds: int = None) -> List[SystemMetrics]:
        """Get historical metrics."""
        with self._lock:
            if seconds is None:
                return list(self._history)
            
            cutoff = time.time() - seconds
            return [m for m in self._history if m.timestamp >= cutoff]
    
    def get_averages(self, seconds: int = 60) -> Dict[str, float]:
        """Get average metrics over time period."""
        history = self.get_history(seconds)
        if not history:
            return {}
        
        return {
            "cpu_percent": sum(m.cpu_percent for m in history) / len(history),
            "ram_percent": sum(m.ram_percent for m in history) / len(history),
            "gpu_percent": sum(m.gpu_percent for m in history) / len(history),
            "process_cpu": sum(m.process_cpu for m in history) / len(history),
            "process_ram_mb": sum(m.process_ram_mb for m in history) / len(history),
        }
    
    def set_alert(
        self,
        metric: str,
        threshold: float,
        callback: Callable[[SystemMetrics], None],
        cooldown: float = 60.0,
    ):
        """Set an alert for a metric threshold."""
        alert = PerformanceAlert(
            metric=metric,
            threshold=threshold,
            callback=callback,
            cooldown=cooldown,
        )
        self._alerts.append(alert)
        logger.info(f"Alert set: {metric} > {threshold}")
    
    def clear_alerts(self):
        """Clear all alerts."""
        self._alerts.clear()
    
    def detect_bottleneck(self) -> Optional[str]:
        """Detect current system bottleneck."""
        metrics = self.get_metrics()
        
        # Check for bottlenecks
        if metrics.cpu_percent > 90:
            return "CPU"
        if metrics.ram_percent > 90:
            return "RAM"
        if metrics.gpu_available and metrics.gpu_percent > 90:
            return "GPU"
        if metrics.gpu_available and metrics.gpu_memory_used_gb / max(metrics.gpu_memory_total_gb, 1) > 0.9:
            return "GPU Memory"
        if metrics.disk_percent > 95:
            return "Disk"
        
        return None
    
    def get_recommendations(self) -> List[str]:
        """Get performance recommendations."""
        metrics = self.get_metrics()
        recs = []
        
        if metrics.cpu_percent > 80:
            recs.append("CPU usage high - consider smaller model or distributed mode")
        
        if metrics.ram_percent > 80:
            recs.append("RAM usage high - reduce batch size or context length")
        
        if metrics.gpu_available:
            if metrics.gpu_memory_used_gb / max(metrics.gpu_memory_total_gb, 1) > 0.8:
                recs.append("GPU memory high - reduce model size or use CPU offload")
            if metrics.gpu_temp > 80:
                recs.append("GPU temperature high - improve cooling or reduce load")
        
        if metrics.cpu_temp > 80:
            recs.append("CPU temperature high - improve cooling")
        
        if not recs:
            recs.append("System running within normal parameters")
        
        return recs
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                with self._lock:
                    self._metrics = metrics
                    self._history.append(metrics)
                
                # Check alerts
                self._check_alerts(metrics)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            time.sleep(self.update_interval)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect all system metrics."""
        metrics = SystemMetrics()
        metrics.timestamp = time.time()
        
        if self._psutil:
            self._collect_cpu(metrics)
            self._collect_memory(metrics)
            self._collect_disk(metrics)
            self._collect_network(metrics)
            self._collect_process(metrics)
        
        if self._gpu_reader:
            self._collect_gpu(metrics)
        
        return metrics
    
    def _collect_cpu(self, metrics: SystemMetrics):
        """Collect CPU metrics."""
        try:
            metrics.cpu_percent = self._psutil.cpu_percent(interval=None)
            metrics.cpu_count = self._psutil.cpu_count()
            
            freq = self._psutil.cpu_freq()
            if freq:
                metrics.cpu_freq_mhz = freq.current
            
            # Temperature if available
            try:
                temps = self._psutil.sensors_temperatures()
                if temps:
                    # Get first available temperature
                    for name, entries in temps.items():
                        if entries:
                            metrics.cpu_temp = entries[0].current
                            break
            except Exception:
                pass
                
        except Exception as e:
            logger.debug(f"CPU metrics error: {e}")
    
    def _collect_memory(self, metrics: SystemMetrics):
        """Collect memory metrics."""
        try:
            mem = self._psutil.virtual_memory()
            metrics.ram_total_gb = mem.total / (1024 ** 3)
            metrics.ram_used_gb = mem.used / (1024 ** 3)
            metrics.ram_percent = mem.percent
        except Exception as e:
            logger.debug(f"Memory metrics error: {e}")
    
    def _collect_disk(self, metrics: SystemMetrics):
        """Collect disk metrics."""
        try:
            disk = self._psutil.disk_usage('/')
            metrics.disk_total_gb = disk.total / (1024 ** 3)
            metrics.disk_used_gb = disk.used / (1024 ** 3)
            metrics.disk_percent = disk.percent
        except Exception as e:
            logger.debug(f"Disk metrics error: {e}")
    
    def _collect_network(self, metrics: SystemMetrics):
        """Collect network metrics."""
        try:
            net = self._psutil.net_io_counters()
            
            current_time = time.time()
            time_diff = current_time - self._last_net_time
            
            metrics.net_sent_mb = net.bytes_sent / (1024 ** 2)
            metrics.net_recv_mb = net.bytes_recv / (1024 ** 2)
            
            if time_diff > 0:
                sent_diff = (net.bytes_sent - self._last_net_sent) / (1024 ** 2)
                recv_diff = (net.bytes_recv - self._last_net_recv) / (1024 ** 2)
                
                metrics.net_sent_rate_mbps = (sent_diff / time_diff) * 8
                metrics.net_recv_rate_mbps = (recv_diff / time_diff) * 8
            
            self._last_net_sent = net.bytes_sent
            self._last_net_recv = net.bytes_recv
            self._last_net_time = current_time
            
        except Exception as e:
            logger.debug(f"Network metrics error: {e}")
    
    def _collect_process(self, metrics: SystemMetrics):
        """Collect current process metrics."""
        try:
            import os
            process = self._psutil.Process(os.getpid())
            metrics.process_cpu = process.cpu_percent()
            metrics.process_ram_mb = process.memory_info().rss / (1024 ** 2)
        except Exception as e:
            logger.debug(f"Process metrics error: {e}")
    
    def _collect_gpu(self, metrics: SystemMetrics):
        """Collect GPU metrics."""
        try:
            if self._gpu_reader["type"] == "nvidia":
                self._collect_gpu_nvidia(metrics)
            elif self._gpu_reader["type"] == "gputil":
                self._collect_gpu_gputil(metrics)
        except Exception as e:
            logger.debug(f"GPU metrics error: {e}")
    
    def _collect_gpu_nvidia(self, metrics: SystemMetrics):
        """Collect NVIDIA GPU metrics via pynvml."""
        pynvml = self._gpu_reader["lib"]
        
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        metrics.gpu_available = True
        metrics.gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(metrics.gpu_name, bytes):
            metrics.gpu_name = metrics.gpu_name.decode('utf-8')
        
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        metrics.gpu_memory_total_gb = mem_info.total / (1024 ** 3)
        metrics.gpu_memory_used_gb = mem_info.used / (1024 ** 3)
        
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        metrics.gpu_percent = util.gpu
        
        try:
            metrics.gpu_temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            pass
    
    def _collect_gpu_gputil(self, metrics: SystemMetrics):
        """Collect GPU metrics via GPUtil."""
        GPUtil = self._gpu_reader["lib"]
        
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            metrics.gpu_available = True
            metrics.gpu_name = gpu.name
            metrics.gpu_memory_total_gb = gpu.memoryTotal / 1024
            metrics.gpu_memory_used_gb = gpu.memoryUsed / 1024
            metrics.gpu_percent = gpu.load * 100
            metrics.gpu_temp = gpu.temperature
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check and trigger alerts."""
        current_time = time.time()
        
        metric_values = {
            "cpu": metrics.cpu_percent,
            "ram": metrics.ram_percent,
            "gpu": metrics.gpu_percent,
            "gpu_mem": (metrics.gpu_memory_used_gb / max(metrics.gpu_memory_total_gb, 1)) * 100,
            "disk": metrics.disk_percent,
            "cpu_temp": metrics.cpu_temp,
            "gpu_temp": metrics.gpu_temp,
        }
        
        for alert in self._alerts:
            value = metric_values.get(alert.metric, 0)
            
            if value > alert.threshold:
                if current_time - alert.last_triggered >= alert.cooldown:
                    alert.last_triggered = current_time
                    try:
                        alert.callback(metrics)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")


# Global monitor instance
_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(**kwargs) -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor(**kwargs)
    return _monitor


__all__ = [
    'PerformanceMonitor',
    'SystemMetrics',
    'PerformanceAlert',
    'get_performance_monitor',
]
