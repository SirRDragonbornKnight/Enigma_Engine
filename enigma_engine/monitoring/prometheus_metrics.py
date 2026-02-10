"""
Prometheus Metrics Export

Exports system metrics in Prometheus format for monitoring.
Tracks inference, memory, training, and system health metrics.

FILE: enigma_engine/monitoring/prometheus_metrics.py
TYPE: Monitoring/Observability
MAIN CLASSES: MetricsCollector, MetricType, PrometheusExporter
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Prometheus metric types."""
    COUNTER = "counter"       # Monotonically increasing
    GAUGE = "gauge"           # Can go up and down
    HISTOGRAM = "histogram"   # Distribution of values
    SUMMARY = "summary"       # Similar to histogram


@dataclass
class Metric:
    """A single metric definition."""
    name: str
    type: MetricType
    help: str
    labels: list[str] = field(default_factory=list)


@dataclass
class MetricValue:
    """A metric value with labels."""
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: Optional[float] = None


class Counter:
    """Thread-safe counter metric."""
    
    def __init__(self, name: str, help: str, labels: list[str] = None):
        self.name = name
        self.help = help
        self.labels = labels or []
        self._values: dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()
        
    def inc(self, value: float = 1.0, **labels):
        """Increment counter."""
        with self._lock:
            key = tuple(sorted(labels.items()))
            self._values[key] += value
            
    def get(self, **labels) -> float:
        """Get counter value."""
        key = tuple(sorted(labels.items()))
        return self._values.get(key, 0.0)
        
    def collect(self) -> list[MetricValue]:
        """Collect all values."""
        with self._lock:
            result = []
            for key, value in self._values.items():
                labels = dict(key)
                result.append(MetricValue(value=value, labels=labels))
            return result


class Gauge:
    """Thread-safe gauge metric."""
    
    def __init__(self, name: str, help: str, labels: list[str] = None):
        self.name = name
        self.help = help
        self.labels = labels or []
        self._values: dict[tuple, float] = {}
        self._lock = threading.Lock()
        
    def set(self, value: float, **labels):
        """Set gauge value."""
        with self._lock:
            key = tuple(sorted(labels.items()))
            self._values[key] = value
            
    def inc(self, value: float = 1.0, **labels):
        """Increment gauge."""
        with self._lock:
            key = tuple(sorted(labels.items()))
            self._values[key] = self._values.get(key, 0.0) + value
            
    def dec(self, value: float = 1.0, **labels):
        """Decrement gauge."""
        self.inc(-value, **labels)
        
    def get(self, **labels) -> float:
        """Get gauge value."""
        key = tuple(sorted(labels.items()))
        return self._values.get(key, 0.0)
        
    def collect(self) -> list[MetricValue]:
        """Collect all values."""
        with self._lock:
            result = []
            for key, value in self._values.items():
                labels = dict(key)
                result.append(MetricValue(value=value, labels=labels))
            return result


class Histogram:
    """Thread-safe histogram metric."""
    
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf'))
    
    def __init__(self, name: str, help: str, labels: list[str] = None, buckets: tuple = None):
        self.name = name
        self.help = help
        self.labels = labels or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._counts: dict[tuple, dict[float, int]] = defaultdict(lambda: defaultdict(int))
        self._sums: dict[tuple, float] = defaultdict(float)
        self._totals: dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()
        
    def observe(self, value: float, **labels):
        """Observe a value."""
        with self._lock:
            key = tuple(sorted(labels.items()))
            self._sums[key] += value
            self._totals[key] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[key][bucket] += 1
                    
    def collect(self) -> list[MetricValue]:
        """Collect all values."""
        with self._lock:
            result = []
            for key in self._counts:
                labels = dict(key)
                # Bucket values
                for bucket in self.buckets:
                    count = self._counts[key][bucket]
                    bucket_labels = {**labels, "le": str(bucket) if bucket != float('inf') else "+Inf"}
                    result.append(MetricValue(value=count, labels=bucket_labels))
                # Sum and count
                result.append(MetricValue(value=self._sums[key], labels={**labels, "_type": "sum"}))
                result.append(MetricValue(value=self._totals[key], labels={**labels, "_type": "count"}))
            return result


class Summary:
    """Thread-safe summary metric (tracks quantiles)."""
    
    DEFAULT_QUANTILES = (0.5, 0.9, 0.99)
    
    def __init__(self, name: str, help: str, labels: list[str] = None, 
                 quantiles: tuple = None, max_samples: int = 1000):
        self.name = name
        self.help = help
        self.labels = labels or []
        self.quantiles = quantiles or self.DEFAULT_QUANTILES
        self.max_samples = max_samples
        self._samples: dict[tuple, list[float]] = defaultdict(list)
        self._sums: dict[tuple, float] = defaultdict(float)
        self._counts: dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()
        
    def observe(self, value: float, **labels):
        """Observe a value."""
        with self._lock:
            key = tuple(sorted(labels.items()))
            self._samples[key].append(value)
            self._sums[key] += value
            self._counts[key] += 1
            # Trim old samples
            if len(self._samples[key]) > self.max_samples:
                self._samples[key] = self._samples[key][-self.max_samples:]
                
    def collect(self) -> list[MetricValue]:
        """Collect all values."""
        with self._lock:
            result = []
            for key in self._samples:
                labels = dict(key)
                samples = sorted(self._samples[key])
                if samples:
                    for q in self.quantiles:
                        idx = int(len(samples) * q)
                        idx = min(idx, len(samples) - 1)
                        result.append(MetricValue(
                            value=samples[idx],
                            labels={**labels, "quantile": str(q)}
                        ))
                result.append(MetricValue(value=self._sums[key], labels={**labels, "_type": "sum"}))
                result.append(MetricValue(value=self._counts[key], labels={**labels, "_type": "count"}))
            return result


class MetricsCollector:
    """Collects and manages metrics."""
    
    def __init__(self, prefix: str = "enigma_engine"):
        """
        Initialize collector.
        
        Args:
            prefix: Metric name prefix
        """
        self.prefix = prefix
        self._metrics: dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Register default metrics
        self._register_defaults()
        
    def _register_defaults(self):
        """Register default Enigma AI Engine metrics."""
        # Inference metrics
        self.inference_requests = self.counter(
            "inference_requests_total",
            "Total inference requests",
            ["model", "status"]
        )
        self.inference_duration = self.histogram(
            "inference_duration_seconds",
            "Inference request duration",
            ["model"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float('inf'))
        )
        self.tokens_generated = self.counter(
            "tokens_generated_total",
            "Total tokens generated",
            ["model"]
        )
        self.tokens_per_second = self.gauge(
            "tokens_per_second",
            "Current tokens per second rate",
            ["model"]
        )
        
        # Model metrics
        self.model_loaded = self.gauge(
            "model_loaded",
            "Model loaded status",
            ["model", "size"]
        )
        self.model_memory_bytes = self.gauge(
            "model_memory_bytes",
            "Model memory usage in bytes",
            ["model"]
        )
        
        # Memory system metrics
        self.memory_conversations = self.gauge(
            "memory_conversations_total",
            "Total stored conversations"
        )
        self.memory_messages = self.gauge(
            "memory_messages_total",
            "Total stored messages"
        )
        self.vector_db_entries = self.gauge(
            "vector_db_entries_total",
            "Total vector database entries"
        )
        
        # Module metrics
        self.modules_loaded = self.gauge(
            "modules_loaded",
            "Number of loaded modules",
            ["category"]
        )
        self.module_load_errors = self.counter(
            "module_load_errors_total",
            "Module load errors",
            ["module"]
        )
        
        # API metrics
        self.api_requests = self.counter(
            "api_requests_total",
            "Total API requests",
            ["endpoint", "method", "status"]
        )
        self.api_latency = self.histogram(
            "api_latency_seconds",
            "API request latency",
            ["endpoint"]
        )
        
        # Training metrics
        self.training_steps = self.counter(
            "training_steps_total",
            "Total training steps"
        )
        self.training_loss = self.gauge(
            "training_loss",
            "Current training loss"
        )
        
        # System metrics
        self.gpu_memory_used = self.gauge(
            "gpu_memory_used_bytes",
            "GPU memory used",
            ["device"]
        )
        self.cpu_percent = self.gauge(
            "cpu_percent",
            "CPU usage percentage"
        )
        self.memory_percent = self.gauge(
            "memory_percent",
            "System memory usage percentage"
        )
        
    def counter(self, name: str, help: str, labels: list[str] = None) -> Counter:
        """Create or get a counter."""
        full_name = f"{self.prefix}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Counter(full_name, help, labels)
            return self._metrics[full_name]
            
    def gauge(self, name: str, help: str, labels: list[str] = None) -> Gauge:
        """Create or get a gauge."""
        full_name = f"{self.prefix}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Gauge(full_name, help, labels)
            return self._metrics[full_name]
            
    def histogram(self, name: str, help: str, labels: list[str] = None, 
                  buckets: tuple = None) -> Histogram:
        """Create or get a histogram."""
        full_name = f"{self.prefix}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Histogram(full_name, help, labels, buckets)
            return self._metrics[full_name]
            
    def summary(self, name: str, help: str, labels: list[str] = None,
                quantiles: tuple = None) -> Summary:
        """Create or get a summary."""
        full_name = f"{self.prefix}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Summary(full_name, help, labels, quantiles)
            return self._metrics[full_name]
            
    def collect_all(self) -> dict[str, list[MetricValue]]:
        """Collect all metrics."""
        with self._lock:
            return {name: metric.collect() for name, metric in self._metrics.items()}


class PrometheusExporter:
    """Exports metrics in Prometheus format."""
    
    def __init__(self, collector: Optional[MetricsCollector] = None):
        """
        Initialize exporter.
        
        Args:
            collector: Metrics collector (creates default if None)
        """
        self._collector = collector or MetricsCollector()
        
    @property
    def collector(self) -> MetricsCollector:
        return self._collector
        
    def generate(self) -> str:
        """Generate Prometheus format output."""
        lines = []
        metrics = self._collector.collect_all()
        
        for name, values in metrics.items():
            metric = self._collector._metrics.get(name)
            if not metric:
                continue
                
            # Help and type
            lines.append(f"# HELP {name} {metric.help}")
            lines.append(f"# TYPE {name} {self._get_type_name(metric)}")
            
            # Values
            for mv in values:
                if mv.labels:
                    # Handle special labels for histogram/summary
                    if "_type" in mv.labels:
                        type_suffix = mv.labels.pop("_type")
                        label_str = self._format_labels(mv.labels)
                        lines.append(f"{name}_{type_suffix}{label_str} {mv.value}")
                    elif "le" in mv.labels:
                        label_str = self._format_labels(mv.labels)
                        lines.append(f"{name}_bucket{label_str} {mv.value}")
                    elif "quantile" in mv.labels:
                        label_str = self._format_labels(mv.labels)
                        lines.append(f"{name}{label_str} {mv.value}")
                    else:
                        label_str = self._format_labels(mv.labels)
                        lines.append(f"{name}{label_str} {mv.value}")
                else:
                    lines.append(f"{name} {mv.value}")
                    
        return "\n".join(lines) + "\n"
        
    def _get_type_name(self, metric) -> str:
        """Get Prometheus type name."""
        if isinstance(metric, Counter):
            return "counter"
        elif isinstance(metric, Gauge):
            return "gauge"
        elif isinstance(metric, Histogram):
            return "histogram"
        elif isinstance(metric, Summary):
            return "summary"
        return "untyped"
        
    def _format_labels(self, labels: dict[str, str]) -> str:
        """Format labels for Prometheus."""
        if not labels:
            return ""
        pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(pairs) + "}"


# Singleton
_collector: Optional[MetricsCollector] = None
_exporter: Optional[PrometheusExporter] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the metrics collector singleton."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


def get_prometheus_exporter() -> PrometheusExporter:
    """Get the prometheus exporter singleton."""
    global _exporter
    if _exporter is None:
        _exporter = PrometheusExporter(get_metrics_collector())
    return _exporter


def metrics_endpoint() -> str:
    """Generate metrics endpoint output."""
    return get_prometheus_exporter().generate()


# Convenience: record inference
def record_inference(model: str, duration: float, tokens: int, success: bool = True):
    """Record an inference request."""
    collector = get_metrics_collector()
    collector.inference_requests.inc(model=model, status="success" if success else "error")
    collector.inference_duration.observe(duration, model=model)
    collector.tokens_generated.inc(tokens, model=model)
    if duration > 0:
        collector.tokens_per_second.set(tokens / duration, model=model)


__all__ = [
    'MetricsCollector',
    'PrometheusExporter',
    'Counter',
    'Gauge', 
    'Histogram',
    'Summary',
    'MetricType',
    'get_metrics_collector',
    'get_prometheus_exporter',
    'metrics_endpoint',
    'record_inference'
]
