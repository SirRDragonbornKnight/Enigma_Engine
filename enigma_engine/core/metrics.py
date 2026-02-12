"""
Prometheus Metrics for enigma_engine

Provides observability through Prometheus-compatible metrics:
- Request latency histograms
- Token throughput counters
- Active requests gauge
- GPU memory usage
- Model load times
- Error rates

Usage:
    from enigma_engine.core.metrics import MetricsCollector
    
    metrics = MetricsCollector()
    
    # Record inference
    with metrics.inference_timer():
        output = model.generate(...)
    
    # Export metrics
    metrics_text = metrics.export()
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class HistogramBuckets:
    """Standard histogram bucket configurations."""
    # Latency buckets in seconds
    LATENCY = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
    # Token count buckets
    TOKENS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 4096, float('inf')]
    # Memory buckets in GB
    MEMORY_GB = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, float('inf')]


class Counter:
    """Thread-safe counter metric."""
    
    def __init__(self, name: str, description: str, labels: Optional[list[str]] = None) -> None:
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment counter."""
        label_key = tuple(labels.get(l, '') for l in self.labels)
        with self._lock:
            self._values[label_key] += value
    
    def get(self, **labels) -> float:
        """Get current value."""
        label_key = tuple(labels.get(l, '') for l in self.labels)
        return self._values.get(label_key, 0.0)
    
    def export(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter"
        ]
        with self._lock:
            for label_key, value in self._values.items():
                if self.labels:
                    label_str = ','.join(f'{l}="{v}"' for l, v in zip(self.labels, label_key))
                    lines.append(f"{self.name}{{{label_str}}} {value}")
                else:
                    lines.append(f"{self.name} {value}")
        return '\n'.join(lines)


class Gauge:
    """Thread-safe gauge metric."""
    
    def __init__(self, name: str, description: str, labels: Optional[list[str]] = None) -> None:
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def set(self, value: float, **labels) -> None:
        """Set gauge value."""
        label_key = tuple(labels.get(l, '') for l in self.labels)
        with self._lock:
            self._values[label_key] = value
    
    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment gauge."""
        label_key = tuple(labels.get(l, '') for l in self.labels)
        with self._lock:
            self._values[label_key] += value
    
    def dec(self, value: float = 1.0, **labels) -> None:
        """Decrement gauge."""
        self.inc(-value, **labels)
    
    def get(self, **labels) -> float:
        """Get current value."""
        label_key = tuple(labels.get(l, '') for l in self.labels)
        return self._values.get(label_key, 0.0)
    
    def export(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge"
        ]
        with self._lock:
            for label_key, value in self._values.items():
                if self.labels:
                    label_str = ','.join(f'{l}="{v}"' for l, v in zip(self.labels, label_key))
                    lines.append(f"{self.name}{{{label_str}}} {value}")
                else:
                    lines.append(f"{self.name} {value}")
        return '\n'.join(lines)


class Histogram:
    """Thread-safe histogram metric."""
    
    def __init__(
        self,
        name: str,
        description: str,
        buckets: list[float],
        labels: Optional[list[str]] = None
    ) -> None:
        self.name = name
        self.description = description
        self.buckets = sorted(buckets)
        self.labels = labels or []
        self._bucket_counts: dict[tuple, dict[float, int]] = defaultdict(
            lambda: {b: 0 for b in self.buckets}
        )
        self._sums: dict[tuple, float] = defaultdict(float)
        self._counts: dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def observe(self, value: float, **labels) -> None:
        """Observe a value."""
        label_key = tuple(labels.get(l, '') for l in self.labels)
        with self._lock:
            self._sums[label_key] += value
            self._counts[label_key] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[label_key][bucket] += 1
    
    def export(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram"
        ]
        
        with self._lock:
            for label_key in self._bucket_counts:
                label_str = ''
                if self.labels:
                    label_str = ','.join(f'{l}="{v}"' for l, v in zip(self.labels, label_key))
                
                # Cumulative bucket counts
                cumulative = 0
                for bucket in self.buckets:
                    cumulative += self._bucket_counts[label_key][bucket]
                    le_value = '+Inf' if bucket == float('inf') else str(bucket)
                    if label_str:
                        lines.append(f'{self.name}_bucket{{{label_str},le="{le_value}"}} {cumulative}')
                    else:
                        lines.append(f'{self.name}_bucket{{le="{le_value}"}} {cumulative}')
                
                # Sum and count
                if label_str:
                    lines.append(f'{self.name}_sum{{{label_str}}} {self._sums[label_key]}')
                    lines.append(f'{self.name}_count{{{label_str}}} {self._counts[label_key]}')
                else:
                    lines.append(f'{self.name}_sum {self._sums[label_key]}')
                    lines.append(f'{self.name}_count {self._counts[label_key]}')
        
        return '\n'.join(lines)


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, histogram: Histogram, **labels) -> None:
        self.histogram = histogram
        self.labels = labels
        self.start_time: Optional[float] = None
    
    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.histogram.observe(duration, **self.labels)


class MetricsCollector:
    """
    Central metrics collector for enigma_engine.
    
    Collects and exports Prometheus-compatible metrics.
    
    Usage:
        metrics = MetricsCollector()
        
        # Time inference
        with metrics.inference_timer(model="forge-small"):
            output = model.generate(...)
        
        # Count tokens
        metrics.tokens_generated.inc(len(output_tokens), model="forge-small")
        
        # Get metrics text
        print(metrics.export())
    """
    
    def __init__(self) -> None:
        # Request metrics
        self.requests_total = Counter(
            "forge_requests_total",
            "Total number of requests",
            labels=["model", "endpoint", "status"]
        )
        
        self.requests_in_progress = Gauge(
            "forge_requests_in_progress",
            "Number of requests currently being processed",
            labels=["model"]
        )
        
        self.request_latency = Histogram(
            "forge_request_latency_seconds",
            "Request latency in seconds",
            buckets=HistogramBuckets.LATENCY,
            labels=["model", "endpoint"]
        )
        
        # Token metrics
        self.tokens_generated = Counter(
            "forge_tokens_generated_total",
            "Total number of tokens generated",
            labels=["model"]
        )
        
        self.tokens_per_request = Histogram(
            "forge_tokens_per_request",
            "Tokens generated per request",
            buckets=HistogramBuckets.TOKENS,
            labels=["model"]
        )
        
        self.tokens_per_second = Gauge(
            "forge_tokens_per_second",
            "Current token generation rate",
            labels=["model"]
        )
        
        # Batch metrics
        self.batch_size = Histogram(
            "forge_batch_size",
            "Batch sizes for inference",
            buckets=[1, 2, 4, 8, 16, 32, 64, 128, float('inf')],
            labels=["model"]
        )
        
        # Memory metrics
        self.gpu_memory_used = Gauge(
            "forge_gpu_memory_used_bytes",
            "GPU memory used in bytes",
            labels=["device"]
        )
        
        self.gpu_memory_total = Gauge(
            "forge_gpu_memory_total_bytes",
            "Total GPU memory in bytes",
            labels=["device"]
        )
        
        self.kv_cache_size = Gauge(
            "forge_kv_cache_size_bytes",
            "KV cache size in bytes",
            labels=["model"]
        )
        
        # Model metrics
        self.model_load_time = Histogram(
            "forge_model_load_time_seconds",
            "Time to load model",
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, float('inf')],
            labels=["model"]
        )
        
        self.models_loaded = Gauge(
            "forge_models_loaded",
            "Number of models currently loaded",
            labels=[]
        )
        
        # Error metrics
        self.errors_total = Counter(
            "forge_errors_total",
            "Total number of errors",
            labels=["model", "error_type"]
        )
        
        # Queue metrics (for continuous batching)
        self.queue_size = Gauge(
            "forge_queue_size",
            "Number of requests in queue",
            labels=["model"]
        )
        
        self.queue_wait_time = Histogram(
            "forge_queue_wait_time_seconds",
            "Time spent waiting in queue",
            buckets=HistogramBuckets.LATENCY,
            labels=["model"]
        )
        
        # All metrics for export
        self._metrics = [
            self.requests_total,
            self.requests_in_progress,
            self.request_latency,
            self.tokens_generated,
            self.tokens_per_request,
            self.tokens_per_second,
            self.batch_size,
            self.gpu_memory_used,
            self.gpu_memory_total,
            self.kv_cache_size,
            self.model_load_time,
            self.models_loaded,
            self.errors_total,
            self.queue_size,
            self.queue_wait_time,
        ]
        
        # Background thread for periodic updates
        self._stop_event = threading.Event()
        self._update_thread: Optional[threading.Thread] = None
    
    def start_background_updates(self, interval: float = 10.0) -> None:
        """Start background thread for periodic metric updates."""
        self._stop_event.clear()
        self._update_thread = threading.Thread(
            target=self._background_update_loop,
            args=(interval,),
            daemon=True
        )
        self._update_thread.start()
    
    def stop_background_updates(self) -> None:
        """Stop background update thread."""
        self._stop_event.set()
        if self._update_thread:
            self._update_thread.join(timeout=5.0)
    
    def _background_update_loop(self, interval: float) -> None:
        """Background loop for updating system metrics."""
        while not self._stop_event.wait(interval):
            self._update_gpu_metrics()
    
    def _update_gpu_metrics(self) -> None:
        """Update GPU memory metrics."""
        if not torch.cuda.is_available():
            return
        
        for i in range(torch.cuda.device_count()):
            try:
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                
                self.gpu_memory_used.set(memory_allocated, device=f"cuda:{i}")
                self.gpu_memory_total.set(memory_total, device=f"cuda:{i}")
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics for device {i}: {e}")
    
    def inference_timer(self, model: str = "default", endpoint: str = "generate") -> Timer:
        """Get a timer for inference latency."""
        return Timer(self.request_latency, model=model, endpoint=endpoint)
    
    def model_load_timer(self, model: str = "default") -> Timer:
        """Get a timer for model loading."""
        return Timer(self.model_load_time, model=model)
    
    def record_request(
        self,
        model: str,
        endpoint: str,
        status: str,
        tokens: int,
        latency: float
    ) -> None:
        """Record a completed request."""
        self.requests_total.inc(model=model, endpoint=endpoint, status=status)
        self.tokens_generated.inc(tokens, model=model)
        self.tokens_per_request.observe(tokens, model=model)
        self.request_latency.observe(latency, model=model, endpoint=endpoint)
        
        if latency > 0:
            self.tokens_per_second.set(tokens / latency, model=model)
    
    def record_error(self, model: str, error_type: str) -> None:
        """Record an error."""
        self.errors_total.inc(model=model, error_type=error_type)
    
    def export(self) -> str:
        """Export all metrics in Prometheus format."""
        # Update GPU metrics before export
        self._update_gpu_metrics()
        
        lines = []
        for metric in self._metrics:
            lines.append(metric.export())
            lines.append("")  # Blank line between metrics
        
        return '\n'.join(lines)


# Global metrics instance
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def metrics_endpoint() -> Any:
    """Flask endpoint handler for /metrics."""
    from flask import Response
    return Response(get_metrics().export(), mimetype='text/plain')


def add_metrics_to_flask_app(app: Any) -> None:
    """Add /metrics endpoint to a Flask app."""
    app.add_url_rule('/metrics', 'metrics', metrics_endpoint)
    logger.info("Added /metrics endpoint to Flask app")
