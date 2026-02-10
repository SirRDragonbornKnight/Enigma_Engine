"""
Telemetry Dashboard

Optional analytics and performance monitoring dashboard.
Privacy-respecting with local-first data collection.

FILE: enigma_engine/web/telemetry_dashboard.py
TYPE: Web/Analytics
MAIN CLASSES: TelemetryCollector, MetricsStore, TelemetryDashboard
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from flask import Flask, jsonify, request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class TelemetryLevel(Enum):
    """Telemetry collection levels."""
    OFF = "off"
    MINIMAL = "minimal"  # Only errors
    BASIC = "basic"  # Performance metrics
    FULL = "full"  # Everything


@dataclass
class Metric:
    """A single metric measurement."""
    name: str
    value: float
    type: MetricType
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.type.value,
            "timestamp": self.timestamp,
            "labels": self.labels
        }


@dataclass
class Event:
    """A telemetry event."""
    name: str
    timestamp: float = field(default_factory=time.time)
    properties: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "properties": self.properties
        }


@dataclass
class TelemetryConfig:
    """Telemetry configuration."""
    enabled: bool = True
    level: TelemetryLevel = TelemetryLevel.BASIC
    
    # Storage
    data_dir: Path = None
    max_events: int = 10000
    retention_days: int = 30
    
    # Privacy
    anonymize: bool = True
    collect_system_info: bool = True
    
    # Remote (opt-in only)
    remote_enabled: bool = False
    remote_endpoint: str = ""
    
    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = Path.home() / ".Enigma AI Engine" / "telemetry"


class MetricsStore:
    """Store and aggregate metrics."""
    
    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._metrics: dict[str, list[Metric]] = defaultdict(list)
        self._events: list[Event] = []
        self._lock = threading.Lock()
        
        # Load existing data
        self._load()
    
    def record_metric(self, metric: Metric):
        """Record a metric."""
        with self._lock:
            self._metrics[metric.name].append(metric)
            
            # Trim if too many
            if len(self._metrics[metric.name]) > 1000:
                self._metrics[metric.name] = self._metrics[metric.name][-500:]
    
    def record_event(self, event: Event):
        """Record an event."""
        with self._lock:
            self._events.append(event)
            
            # Trim if too many
            if len(self._events) > self.config.max_events:
                self._events = self._events[-self.config.max_events // 2:]
    
    def get_metrics(
        self,
        name: str = None,
        since: float = None
    ) -> list[Metric]:
        """Get metrics, optionally filtered."""
        with self._lock:
            if name:
                metrics = self._metrics.get(name, [])
            else:
                metrics = [m for ms in self._metrics.values() for m in ms]
            
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            return metrics
    
    def get_events(
        self,
        name: str = None,
        since: float = None
    ) -> list[Event]:
        """Get events, optionally filtered."""
        with self._lock:
            events = self._events.copy()
            
            if name:
                events = [e for e in events if e.name == name]
            
            if since:
                events = [e for e in events if e.timestamp >= since]
            
            return events
    
    def get_aggregates(
        self,
        metric_name: str,
        window_seconds: int = 3600
    ) -> dict[str, float]:
        """Get aggregated metrics."""
        since = time.time() - window_seconds
        metrics = self.get_metrics(metric_name, since)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1]
        }
    
    def save(self):
        """Save metrics to disk."""
        with self._lock:
            # Save metrics
            metrics_file = self.config.data_dir / "metrics.json"
            metrics_data = {
                name: [m.to_dict() for m in ms]
                for name, ms in self._metrics.items()
            }
            metrics_file.write_text(json.dumps(metrics_data, indent=2))
            
            # Save events
            events_file = self.config.data_dir / "events.json"
            events_data = [e.to_dict() for e in self._events]
            events_file.write_text(json.dumps(events_data, indent=2))
    
    def _load(self):
        """Load metrics from disk."""
        try:
            metrics_file = self.config.data_dir / "metrics.json"
            if metrics_file.exists():
                data = json.loads(metrics_file.read_text())
                for name, ms in data.items():
                    for m in ms:
                        metric = Metric(
                            name=m["name"],
                            value=m["value"],
                            type=MetricType(m["type"]),
                            timestamp=m["timestamp"],
                            labels=m.get("labels", {})
                        )
                        self._metrics[name].append(metric)
            
            events_file = self.config.data_dir / "events.json"
            if events_file.exists():
                data = json.loads(events_file.read_text())
                for e in data:
                    event = Event(
                        name=e["name"],
                        timestamp=e["timestamp"],
                        properties=e.get("properties", {})
                    )
                    self._events.append(event)
        
        except Exception as e:
            logger.warning(f"Failed to load telemetry data: {e}")
    
    def cleanup_old(self):
        """Remove data older than retention period."""
        cutoff = time.time() - (self.config.retention_days * 86400)
        
        with self._lock:
            for name in self._metrics:
                self._metrics[name] = [
                    m for m in self._metrics[name]
                    if m.timestamp >= cutoff
                ]
            
            self._events = [
                e for e in self._events
                if e.timestamp >= cutoff
            ]


class TelemetryCollector:
    """Collect telemetry data."""
    
    def __init__(self, config: TelemetryConfig = None):
        self.config = config or TelemetryConfig()
        self.store = MetricsStore(self.config)
        
        self._session_id = self._generate_session_id()
        self._device_id = self._get_device_id()
        
        # Timer contexts
        self._timers: dict[str, float] = {}
    
    def _generate_session_id(self) -> str:
        """Generate anonymous session ID."""
        return hashlib.sha256(
            f"{time.time()}{uuid.uuid4()}".encode()
        ).hexdigest()[:16]
    
    def _get_device_id(self) -> str:
        """Get anonymous device ID."""
        if not self.config.anonymize:
            return str(uuid.uuid4())
        
        # Create consistent but anonymous device ID
        import platform
        raw = f"{platform.node()}{platform.processor()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    def counter(
        self,
        name: str,
        value: float = 1,
        labels: dict[str, str] = None
    ):
        """Record a counter metric."""
        if not self._should_collect():
            return
        
        self.store.record_metric(Metric(
            name=name,
            value=value,
            type=MetricType.COUNTER,
            labels=labels or {}
        ))
    
    def gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] = None
    ):
        """Record a gauge metric."""
        if not self._should_collect():
            return
        
        self.store.record_metric(Metric(
            name=name,
            value=value,
            type=MetricType.GAUGE,
            labels=labels or {}
        ))
    
    def histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] = None
    ):
        """Record a histogram metric."""
        if not self._should_collect():
            return
        
        self.store.record_metric(Metric(
            name=name,
            value=value,
            type=MetricType.HISTOGRAM,
            labels=labels or {}
        ))
    
    def timer_start(self, name: str):
        """Start a timer."""
        self._timers[name] = time.time()
    
    def timer_stop(self, name: str, labels: dict[str, str] = None):
        """Stop a timer and record duration."""
        if name not in self._timers:
            return
        
        duration = time.time() - self._timers.pop(name)
        
        if self._should_collect():
            self.store.record_metric(Metric(
                name=name,
                value=duration,
                type=MetricType.TIMER,
                labels=labels or {}
            ))
    
    def event(
        self,
        name: str,
        properties: dict[str, Any] = None
    ):
        """Record an event."""
        if not self._should_collect():
            return
        
        props = properties or {}
        props["session_id"] = self._session_id
        
        if self.config.collect_system_info:
            import platform
            props["platform"] = platform.system()
            props["python_version"] = platform.python_version()
        
        self.store.record_event(Event(
            name=name,
            properties=props
        ))
    
    def error(self, name: str, error: Exception):
        """Record an error event."""
        if self.config.level == TelemetryLevel.OFF:
            return
        
        self.event(f"error.{name}", {
            "error_type": type(error).__name__,
            "error_message": str(error)
        })
    
    def _should_collect(self) -> bool:
        """Check if should collect based on level."""
        return (
            self.config.enabled and
            self.config.level != TelemetryLevel.OFF
        )
    
    def flush(self):
        """Flush collected data to storage."""
        self.store.save()


class TelemetryDashboard:
    """Web dashboard for telemetry visualization."""
    
    DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Enigma AI Engine Telemetry Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: system-ui; background: #f5f5f5; padding: 20px; }
        h1 { margin-bottom: 20px; color: #333; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card h2 { font-size: 14px; color: #666; margin-bottom: 10px; }
        .metric-value { font-size: 32px; font-weight: bold; color: #333; }
        .metric-change { font-size: 14px; color: #4caf50; }
        .metric-change.negative { color: #f44336; }
        canvas { max-height: 200px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #eee; }
        th { font-weight: 600; color: #666; }
        .status-card { display: flex; justify-content: space-between; align-items: center; }
        .status-dot { width: 12px; height: 12px; border-radius: 50%; background: #4caf50; }
        .status-dot.warning { background: #ff9800; }
        .status-dot.error { background: #f44336; }
        #refresh-btn { position: fixed; bottom: 20px; right: 20px; padding: 12px 24px; 
                       background: #2196f3; color: white; border: none; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Enigma AI Engine Telemetry Dashboard</h1>
    
    <div class="grid">
        <!-- Overview Cards -->
        <div class="card status-card">
            <div>
                <h2>System Status</h2>
                <div class="metric-value" id="status">Active</div>
            </div>
            <div class="status-dot" id="status-dot"></div>
        </div>
        
        <div class="card">
            <h2>Inference Requests (24h)</h2>
            <div class="metric-value" id="inference-count">-</div>
            <div class="metric-change" id="inference-change">-</div>
        </div>
        
        <div class="card">
            <h2>Average Response Time</h2>
            <div class="metric-value" id="avg-response">-</div>
            <div class="metric-change" id="response-change">-</div>
        </div>
        
        <div class="card">
            <h2>Error Rate</h2>
            <div class="metric-value" id="error-rate">-</div>
            <div class="metric-change" id="error-change">-</div>
        </div>
        
        <!-- Charts -->
        <div class="card">
            <h2>Requests Over Time</h2>
            <canvas id="requests-chart"></canvas>
        </div>
        
        <div class="card">
            <h2>Response Time Distribution</h2>
            <canvas id="response-chart"></canvas>
        </div>
        
        <div class="card">
            <h2>Memory Usage</h2>
            <canvas id="memory-chart"></canvas>
        </div>
        
        <div class="card">
            <h2>GPU Utilization</h2>
            <canvas id="gpu-chart"></canvas>
        </div>
        
        <!-- Recent Events -->
        <div class="card" style="grid-column: span 2;">
            <h2>Recent Events</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Event</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody id="events-table">
                </tbody>
            </table>
        </div>
    </div>
    
    <button id="refresh-btn" onclick="refreshData()">Refresh</button>
    
    <script>
        let charts = {};
        
        async function fetchData() {
            const response = await fetch('/api/telemetry/summary');
            return response.json();
        }
        
        function updateMetrics(data) {
            document.getElementById('inference-count').textContent = data.inference_count || '0';
            document.getElementById('avg-response').textContent = 
                (data.avg_response_time || 0).toFixed(2) + 'ms';
            document.getElementById('error-rate').textContent = 
                (data.error_rate || 0).toFixed(2) + '%';
        }
        
        function updateCharts(data) {
            // Requests chart
            if (charts.requests) {
                charts.requests.data.labels = data.requests_timeline?.labels || [];
                charts.requests.data.datasets[0].data = data.requests_timeline?.values || [];
                charts.requests.update();
            }
            
            // Response time chart
            if (charts.response) {
                charts.response.data.labels = data.response_distribution?.labels || [];
                charts.response.data.datasets[0].data = data.response_distribution?.values || [];
                charts.response.update();
            }
        }
        
        function updateEvents(events) {
            const tbody = document.getElementById('events-table');
            tbody.innerHTML = '';
            
            (events || []).slice(0, 10).forEach(event => {
                const tr = document.createElement('tr');
                const time = new Date(event.timestamp * 1000).toLocaleTimeString();
                tr.innerHTML = `
                    <td>${time}</td>
                    <td>${event.name}</td>
                    <td>${JSON.stringify(event.properties || {}).slice(0, 50)}</td>
                `;
                tbody.appendChild(tr);
            });
        }
        
        async function refreshData() {
            const data = await fetchData();
            updateMetrics(data);
            updateCharts(data);
            updateEvents(data.recent_events);
        }
        
        function initCharts() {
            charts.requests = new Chart(
                document.getElementById('requests-chart'),
                {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Requests',
                            data: [],
                            borderColor: '#2196f3',
                            fill: false
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                }
            );
            
            charts.response = new Chart(
                document.getElementById('response-chart'),
                {
                    type: 'bar',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Count',
                            data: [],
                            backgroundColor: '#4caf50'
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                }
            );
            
            charts.memory = new Chart(
                document.getElementById('memory-chart'),
                {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Memory (MB)',
                            data: [],
                            borderColor: '#ff9800',
                            fill: true,
                            backgroundColor: 'rgba(255, 152, 0, 0.1)'
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                }
            );
            
            charts.gpu = new Chart(
                document.getElementById('gpu-chart'),
                {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'GPU %',
                            data: [],
                            borderColor: '#9c27b0',
                            fill: true,
                            backgroundColor: 'rgba(156, 39, 176, 0.1)'
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                }
            );
        }
        
        // Initialize
        initCharts();
        refreshData();
        setInterval(refreshData, 30000);  // Auto-refresh every 30s
    </script>
</body>
</html>
"""
    
    def __init__(
        self,
        collector: TelemetryCollector,
        host: str = "127.0.0.1",
        port: int = 5001
    ):
        self.collector = collector
        self.host = host
        self.port = port
        self._app: Optional[Flask] = None
    
    def create_app(self) -> Flask:
        """Create Flask app for dashboard."""
        if not HAS_FLASK:
            raise ImportError("Flask required for dashboard")
        
        app = Flask(__name__)
        
        @app.route("/")
        def dashboard():
            return self.DASHBOARD_TEMPLATE
        
        @app.route("/api/telemetry/summary")
        def summary():
            return jsonify(self._get_summary())
        
        @app.route("/api/telemetry/metrics")
        def metrics():
            name = request.args.get("name")
            since = request.args.get("since", type=float)
            metrics = self.collector.store.get_metrics(name, since)
            return jsonify([m.to_dict() for m in metrics])
        
        @app.route("/api/telemetry/events")
        def events():
            name = request.args.get("name")
            since = request.args.get("since", type=float)
            events = self.collector.store.get_events(name, since)
            return jsonify([e.to_dict() for e in events])
        
        self._app = app
        return app
    
    def _get_summary(self) -> dict[str, Any]:
        """Get telemetry summary for dashboard."""
        now = time.time()
        day_ago = now - 86400
        
        # Inference metrics
        inference_metrics = self.collector.store.get_aggregates(
            "inference.duration", window_seconds=86400
        )
        
        # Error rate
        errors = len(self.collector.store.get_events("error", day_ago))
        total = inference_metrics.get("count", 0) or 1
        error_rate = (errors / total) * 100
        
        # Recent events
        recent_events = self.collector.store.get_events(since=now - 3600)[-10:]
        
        # Timeline data
        hour_buckets = defaultdict(int)
        for m in self.collector.store.get_metrics("inference.duration", day_ago):
            hour = int(m.timestamp // 3600)
            hour_buckets[hour] += 1
        
        sorted_hours = sorted(hour_buckets.keys())
        
        return {
            "inference_count": inference_metrics.get("count", 0),
            "avg_response_time": inference_metrics.get("avg", 0) * 1000,  # to ms
            "error_rate": error_rate,
            "recent_events": [e.to_dict() for e in recent_events],
            "requests_timeline": {
                "labels": [f"{h % 24}:00" for h in sorted_hours],
                "values": [hour_buckets[h] for h in sorted_hours]
            },
            "response_distribution": self._get_response_distribution(day_ago)
        }
    
    def _get_response_distribution(
        self,
        since: float
    ) -> dict[str, list]:
        """Get response time distribution."""
        metrics = self.collector.store.get_metrics("inference.duration", since)
        
        buckets = {"<50ms": 0, "50-100ms": 0, "100-500ms": 0, "500ms-1s": 0, ">1s": 0}
        
        for m in metrics:
            ms = m.value * 1000
            if ms < 50:
                buckets["<50ms"] += 1
            elif ms < 100:
                buckets["50-100ms"] += 1
            elif ms < 500:
                buckets["100-500ms"] += 1
            elif ms < 1000:
                buckets["500ms-1s"] += 1
            else:
                buckets[">1s"] += 1
        
        return {
            "labels": list(buckets.keys()),
            "values": list(buckets.values())
        }
    
    def run(self, debug: bool = False):
        """Run the dashboard server."""
        app = self.create_app()
        app.run(host=self.host, port=self.port, debug=debug)
    
    def run_background(self):
        """Run dashboard in background thread."""
        thread = threading.Thread(
            target=self.run,
            daemon=True
        )
        thread.start()
        logger.info(f"Telemetry dashboard started at http://{self.host}:{self.port}")


# Global instance
_collector: Optional[TelemetryCollector] = None


def get_collector(config: TelemetryConfig = None) -> TelemetryCollector:
    """Get or create global telemetry collector."""
    global _collector
    if _collector is None:
        _collector = TelemetryCollector(config)
    return _collector


# Convenience functions
def track_inference(duration: float, model: str = "default"):
    """Track an inference call."""
    collector = get_collector()
    collector.histogram("inference.duration", duration, {"model": model})
    collector.counter("inference.count", labels={"model": model})


def track_error(name: str, error: Exception):
    """Track an error."""
    get_collector().error(name, error)


def track_event(name: str, **properties):
    """Track an event."""
    get_collector().event(name, properties)
