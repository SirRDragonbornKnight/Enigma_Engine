"""
Health Check and Monitoring for enigma_engine

Production health monitoring:
- Liveness/readiness probes
- Model health checks
- System resource monitoring
- Performance tracking

Usage:
    from enigma_engine.core.health import HealthChecker, create_health_routes
    
    checker = HealthChecker(model)
    
    # Flask integration
    app = create_health_routes(app, checker)
"""

import gc
import logging
import os
import platform
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class HealthStatus:
    """Health check result."""
    status: str  # 'healthy', 'degraded', 'unhealthy'
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            'status': self.status,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'disk_percent': self.disk_percent,
            'gpu_memory_used_mb': self.gpu_memory_used_mb,
            'gpu_memory_total_mb': self.gpu_memory_total_mb,
            'gpu_utilization': self.gpu_utilization
        }


class HealthChecker:
    """
    Health check manager for production deployments.
    
    Provides:
    - Liveness checks (is the service running?)
    - Readiness checks (can the service handle requests?)
    - Model health checks
    - Custom health checks
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        check_interval: int = 30,
        unhealthy_threshold: int = 3
    ):
        self.model = model
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        
        self._custom_checks: dict[str, Callable[[], HealthStatus]] = {}
        self._last_check_time: Optional[datetime] = None
        self._consecutive_failures = 0
        self._is_ready = False
        
        # Performance tracking
        self._request_latencies: deque = deque(maxlen=1000)
        self._request_count = 0
        self._error_count = 0
        
        self._lock = threading.Lock()
    
    def register_check(
        self,
        name: str,
        check_fn: Callable[[], HealthStatus]
    ):
        """Register a custom health check."""
        self._custom_checks[name] = check_fn
    
    def check_liveness(self) -> HealthStatus:
        """
        Check if the service is alive.
        
        Returns healthy if the process is running.
        """
        try:
            # Basic process check
            pid = os.getpid()
            
            if PSUTIL_AVAILABLE:
                process = psutil.Process(pid)
                if process.is_running():
                    return HealthStatus(
                        status='healthy',
                        message='Service is alive',
                        details={'pid': pid, 'threads': process.num_threads()}
                    )
            else:
                return HealthStatus(
                    status='healthy',
                    message='Service is alive',
                    details={'pid': pid}
                )
            
        except Exception as e:
            return HealthStatus(
                status='unhealthy',
                message=f'Liveness check failed: {e}'
            )
    
    def check_readiness(self) -> HealthStatus:
        """
        Check if the service is ready to handle requests.
        
        Checks model availability and resource status.
        """
        issues = []
        details = {}
        
        # Check model
        if self.model is not None:
            try:
                # Try a simple forward pass
                model_status = self._check_model()
                details['model'] = model_status.to_dict()
                
                if model_status.status != 'healthy':
                    issues.append(f"Model: {model_status.message}")
                    
            except Exception as e:
                issues.append(f"Model check failed: {e}")
        
        # Check system resources
        try:
            metrics = self.get_system_metrics()
            details['system'] = metrics.to_dict()
            
            # Check memory
            if metrics.memory_percent > 90:
                issues.append(f"High memory usage: {metrics.memory_percent:.1f}%")
            
            # Check GPU memory
            if metrics.gpu_memory_used_mb and metrics.gpu_memory_total_mb:
                gpu_percent = metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb * 100
                if gpu_percent > 95:
                    issues.append(f"High GPU memory: {gpu_percent:.1f}%")
                    
        except Exception as e:
            logger.warning(f"System metrics check failed: {e}")
        
        # Run custom checks
        for name, check_fn in self._custom_checks.items():
            try:
                result = check_fn()
                details[name] = result.to_dict()
                
                if result.status != 'healthy':
                    issues.append(f"{name}: {result.message}")
                    
            except Exception as e:
                issues.append(f"{name} check failed: {e}")
        
        # Determine overall status
        if not issues:
            self._consecutive_failures = 0
            self._is_ready = True
            return HealthStatus(
                status='healthy',
                message='Service is ready',
                details=details
            )
        elif len(issues) <= 1:
            return HealthStatus(
                status='degraded',
                message='; '.join(issues),
                details=details
            )
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.unhealthy_threshold:
                self._is_ready = False
            
            return HealthStatus(
                status='unhealthy',
                message='; '.join(issues),
                details=details
            )
    
    def _check_model(self) -> HealthStatus:
        """Check model health."""
        if self.model is None:
            return HealthStatus(
                status='healthy',
                message='No model loaded'
            )
        
        try:
            # Check if model is on expected device
            if TORCH_AVAILABLE and hasattr(self.model, 'parameters'):
                device = next(self.model.parameters()).device
                
                # Try inference
                with torch.no_grad():
                    if hasattr(self.model, 'config'):
                        vocab_size = getattr(self.model.config, 'vocab_size', 1000)
                    else:
                        vocab_size = 1000
                    
                    dummy_input = torch.randint(0, vocab_size, (1, 10), device=device)
                    _ = self.model(dummy_input)
                
                return HealthStatus(
                    status='healthy',
                    message=f'Model operational on {device}',
                    details={'device': str(device)}
                )
            
            return HealthStatus(
                status='healthy',
                message='Model present'
            )
            
        except Exception as e:
            return HealthStatus(
                status='unhealthy',
                message=f'Model check failed: {e}'
            )
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        cpu_percent = 0.0
        memory_percent = 0.0
        memory_used_mb = 0.0
        memory_available_mb = 0.0
        disk_percent = None
        gpu_memory_used_mb = None
        gpu_memory_total_mb = None
        gpu_utilization = None
        
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            try:
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
            except Exception:
                pass  # Intentionally silent
        
        # GPU metrics
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_memory_used_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_memory_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                
                # Try to get GPU utilization via nvidia-smi
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                        capture_output=True,
                        text=True,
                        timeout=1
                    )
                    if result.returncode == 0:
                        gpu_utilization = float(result.stdout.strip().split('\n')[0])
                except Exception:
                    pass  # Intentionally silent
                    
            except Exception as e:
                logger.debug(f"GPU metrics unavailable: {e}")
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_percent=disk_percent,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            gpu_utilization=gpu_utilization
        )
    
    def record_request(
        self,
        latency_ms: float,
        success: bool = True
    ):
        """Record request metrics."""
        with self._lock:
            self._request_latencies.append(latency_ms)
            self._request_count += 1
            if not success:
                self._error_count += 1
    
    def get_request_stats(self) -> dict[str, Any]:
        """Get request statistics."""
        with self._lock:
            latencies = list(self._request_latencies)
        
        if not latencies:
            return {
                'total_requests': self._request_count,
                'error_count': self._error_count,
                'error_rate': 0.0
            }
        
        latencies.sort()
        
        return {
            'total_requests': self._request_count,
            'error_count': self._error_count,
            'error_rate': self._error_count / self._request_count if self._request_count > 0 else 0,
            'latency_avg_ms': sum(latencies) / len(latencies),
            'latency_p50_ms': latencies[len(latencies) // 2],
            'latency_p95_ms': latencies[int(len(latencies) * 0.95)],
            'latency_p99_ms': latencies[int(len(latencies) * 0.99)],
            'latency_max_ms': max(latencies)
        }
    
    def get_full_status(self) -> dict[str, Any]:
        """Get comprehensive status report."""
        liveness = self.check_liveness()
        readiness = self.check_readiness()
        metrics = self.get_system_metrics()
        stats = self.get_request_stats()
        
        return {
            'service': {
                'name': 'enigma_engine',
                'version': '1.0.0',
                'uptime_seconds': time.time() - self._start_time if hasattr(self, '_start_time') else 0
            },
            'liveness': liveness.to_dict(),
            'readiness': readiness.to_dict(),
            'system': metrics.to_dict(),
            'requests': stats,
            'environment': {
                'python_version': platform.python_version(),
                'platform': platform.platform(),
                'torch_version': torch.__version__ if TORCH_AVAILABLE else None,
                'cuda_available': torch.cuda.is_available() if TORCH_AVAILABLE else False
            }
        }


def create_health_routes(app: Any, checker: HealthChecker) -> Any:
    """
    Add health check routes to a Flask app.
    
    Args:
        app: Flask application
        checker: HealthChecker instance
    
    Returns:
        Flask app with health routes
    """
    from flask import jsonify
    
    @app.route('/health')
    @app.route('/health/live')
    def health_live() -> tuple[Any, int]:
        status = checker.check_liveness()
        code = 200 if status.status == 'healthy' else 503
        return jsonify(status.to_dict()), code
    
    @app.route('/health/ready')
    def health_ready() -> tuple[Any, int]:
        status = checker.check_readiness()
        code = 200 if status.status == 'healthy' else 503
        return jsonify(status.to_dict()), code
    
    @app.route('/health/full')
    def health_full() -> Any:
        return jsonify(checker.get_full_status())
    
    @app.route('/metrics')
    def metrics() -> Any:
        return jsonify({
            'system': checker.get_system_metrics().to_dict(),
            'requests': checker.get_request_stats()
        })
    
    return app


class HealthCheckMiddleware:
    """
    Middleware for automatic request tracking.
    
    Usage (Flask):
        app.wsgi_app = HealthCheckMiddleware(app.wsgi_app, checker)
    """
    
    def __init__(self, app: Any, checker: HealthChecker):
        self.app = app
        self.checker = checker
    
    def __call__(self, environ: dict, start_response: Callable) -> Any:
        start_time = time.time()
        success = True
        
        def custom_start_response(status: str, headers: Any, exc_info: Any = None) -> Any:
            nonlocal success
            status_code = int(status.split()[0])
            success = status_code < 500
            return start_response(status, headers, exc_info)
        
        try:
            response = self.app(environ, custom_start_response)
            return response
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self.checker.record_request(latency_ms, success)


def warmup_model(
    model: Any,
    tokenizer: Any,
    num_warmup: int = 3,
    warmup_text: str = "Hello, world!"
) -> None:
    """
    Warm up model for optimal performance.
    
    Runs several inference passes to:
    - JIT compile CUDA kernels
    - Populate caches
    - Stabilize memory allocation
    """
    if not TORCH_AVAILABLE:
        return
    
    logger.info(f"Warming up model with {num_warmup} passes...")
    
    device = next(model.parameters()).device
    
    for i in range(num_warmup):
        with torch.no_grad():
            input_ids = torch.tensor(
                [tokenizer.encode(warmup_text)],
                device=device
            )
            _ = model(input_ids)
        
        # Clear CUDA cache between warmup runs
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Model warmup complete")
