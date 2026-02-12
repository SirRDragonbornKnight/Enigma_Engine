"""
Performance Testing Suite

Benchmark critical paths, load testing, and profiling.

FILE: enigma_engine/testing/benchmarks.py
TYPE: Testing
MAIN CLASSES: BenchmarkSuite, ProfileRunner, LoadTester
"""

import gc
import json
import logging
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    iterations: int
    total_time: float
    mean_time: float
    min_time: float
    max_time: float
    std_dev: float
    throughput: float  # ops/sec
    memory_used_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_sec": self.total_time,
            "mean_time_sec": self.mean_time,
            "min_time_sec": self.min_time,
            "max_time_sec": self.max_time,
            "std_dev_sec": self.std_dev,
            "throughput_ops_sec": self.throughput,
            "memory_used_mb": self.memory_used_mb,
            "metadata": self.metadata
        }


@dataclass
class LoadTestResult:
    """Result of load testing."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    requests_per_second: float
    mean_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    max_latency: float
    errors: list[str] = field(default_factory=list)


def benchmark(iterations: int = 100, warmup: int = 10):
    """Decorator to benchmark a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Warmup
            for _ in range(warmup):
                func(*args, **kwargs)
            
            # Benchmark
            gc.collect()
            times = []
            
            for _ in range(iterations):
                start = time.perf_counter()
                func(*args, **kwargs)
                end = time.perf_counter()
                times.append(end - start)
            
            return BenchmarkResult(
                name=func.__name__,
                iterations=iterations,
                total_time=sum(times),
                mean_time=statistics.mean(times),
                min_time=min(times),
                max_time=max(times),
                std_dev=statistics.stdev(times) if len(times) > 1 else 0,
                throughput=iterations / sum(times) if sum(times) > 0 else 0
            )
        return wrapper
    return decorator


@contextmanager
def timer(name: str = "operation"):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    start_memory = 0
    
    if HAS_PSUTIL:
        process = psutil.Process()
        start_memory = process.memory_info().rss
    
    yield
    
    end = time.perf_counter()
    elapsed = end - start
    
    memory_delta = 0
    if HAS_PSUTIL:
        end_memory = process.memory_info().rss
        memory_delta = (end_memory - start_memory) / 1024 / 1024
    
    logger.info(f"{name}: {elapsed:.4f}s, memory delta: {memory_delta:.2f}MB")


class Profiler:
    """Simple profiler for code sections."""
    
    def __init__(self):
        self._sections: dict[str, list[float]] = {}
        self._active: dict[str, float] = {}
    
    def start(self, section: str):
        """Start timing a section."""
        self._active[section] = time.perf_counter()
    
    def stop(self, section: str):
        """Stop timing a section."""
        if section in self._active:
            elapsed = time.perf_counter() - self._active[section]
            
            if section not in self._sections:
                self._sections[section] = []
            self._sections[section].append(elapsed)
            
            del self._active[section]
    
    @contextmanager
    def section(self, name: str):
        """Context manager for timing."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)
    
    def get_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all sections."""
        stats = {}
        
        for name, times in self._sections.items():
            if times:
                stats[name] = {
                    "count": len(times),
                    "total": sum(times),
                    "mean": statistics.mean(times),
                    "min": min(times),
                    "max": max(times),
                    "std_dev": statistics.stdev(times) if len(times) > 1 else 0
                }
        
        return stats
    
    def reset(self):
        """Reset profiler."""
        self._sections.clear()
        self._active.clear()


class BenchmarkSuite:
    """
    Collection of benchmarks for Enigma AI Engine components.
    """
    
    def __init__(self, output_dir: str = "outputs/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._benchmarks: dict[str, Callable] = {}
        self._results: list[BenchmarkResult] = []
    
    def register(self, name: str = None):
        """Decorator to register a benchmark."""
        def decorator(func):
            bench_name = name or func.__name__
            self._benchmarks[bench_name] = func
            return func
        return decorator
    
    def run_all(
        self,
        iterations: int = 100,
        warmup: int = 10
    ) -> list[BenchmarkResult]:
        """Run all registered benchmarks."""
        self._results = []
        
        for name, func in self._benchmarks.items():
            logger.info(f"Running benchmark: {name}")
            
            try:
                result = self._run_benchmark(func, name, iterations, warmup)
                self._results.append(result)
                
                logger.info(
                    f"  {result.mean_time*1000:.2f}ms mean, "
                    f"{result.throughput:.1f} ops/sec"
                )
            except Exception as e:
                logger.error(f"  Failed: {e}")
                self._results.append(BenchmarkResult(
                    name=name,
                    iterations=0,
                    total_time=0,
                    mean_time=0,
                    min_time=0,
                    max_time=0,
                    std_dev=0,
                    throughput=0,
                    metadata={"error": str(e)}
                ))
        
        return self._results
    
    def run_benchmark(
        self,
        name: str,
        iterations: int = 100,
        warmup: int = 10
    ) -> Optional[BenchmarkResult]:
        """Run a specific benchmark."""
        if name not in self._benchmarks:
            logger.error(f"Unknown benchmark: {name}")
            return None
        
        return self._run_benchmark(
            self._benchmarks[name],
            name,
            iterations,
            warmup
        )
    
    def _run_benchmark(
        self,
        func: Callable,
        name: str,
        iterations: int,
        warmup: int
    ) -> BenchmarkResult:
        """Execute a benchmark."""
        # Warmup
        for _ in range(warmup):
            func()
        
        # Benchmark
        gc.collect()
        
        memory_before = 0
        if HAS_PSUTIL:
            process = psutil.Process()
            memory_before = process.memory_info().rss
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append(end - start)
        
        memory_after = 0
        if HAS_PSUTIL:
            memory_after = process.memory_info().rss
        
        memory_used = (memory_after - memory_before) / 1024 / 1024
        
        return BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time=sum(times),
            mean_time=statistics.mean(times),
            min_time=min(times),
            max_time=max(times),
            std_dev=statistics.stdev(times) if len(times) > 1 else 0,
            throughput=iterations / sum(times) if sum(times) > 0 else 0,
            memory_used_mb=memory_used
        )
    
    def save_results(self, filename: str = None):
        """Save results to JSON."""
        if filename is None:
            filename = f"benchmark_{int(time.time())}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(
                [r.to_dict() for r in self._results],
                f,
                indent=2
            )
        
        logger.info(f"Results saved to {filepath}")
    
    def compare(
        self,
        baseline_file: str,
        current_results: list[BenchmarkResult] = None
    ) -> dict[str, dict[str, float]]:
        """Compare current results to baseline."""
        results = current_results or self._results
        
        with open(baseline_file) as f:
            baseline = {r["name"]: r for r in json.load(f)}
        
        comparison = {}
        
        for result in results:
            if result.name in baseline:
                base = baseline[result.name]
                
                # Calculate differences
                time_diff = (result.mean_time - base["mean_time_sec"]) / base["mean_time_sec"] * 100
                throughput_diff = (result.throughput - base["throughput_ops_sec"]) / base["throughput_ops_sec"] * 100
                
                comparison[result.name] = {
                    "baseline_ms": base["mean_time_sec"] * 1000,
                    "current_ms": result.mean_time * 1000,
                    "time_change_pct": time_diff,
                    "throughput_change_pct": throughput_diff,
                    "regression": time_diff > 10  # >10% slower is regression
                }
        
        return comparison


class LoadTester:
    """
    Load testing for API and services.
    """
    
    def __init__(
        self,
        target: Callable,
        concurrency: int = 10
    ):
        self.target = target
        self.concurrency = concurrency
        
        self._results: list[float] = []
        self._errors: list[str] = []
        self._lock = threading.Lock()
    
    def run(
        self,
        requests: int,
        duration_sec: float = None
    ) -> LoadTestResult:
        """
        Run load test.
        
        Args:
            requests: Total number of requests
            duration_sec: Optional duration limit
        """
        self._results = []
        self._errors = []
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = []
            
            for i in range(requests):
                if duration_sec and (time.time() - start_time) > duration_sec:
                    break
                
                futures.append(executor.submit(self._execute_request, i))
            
            # Wait for completion
            for future in futures:
                future.result()
        
        total_time = time.time() - start_time
        
        # Calculate latency percentiles
        sorted_results = sorted(self._results)
        
        return LoadTestResult(
            total_requests=len(self._results) + len(self._errors),
            successful_requests=len(self._results),
            failed_requests=len(self._errors),
            total_time=total_time,
            requests_per_second=len(self._results) / total_time if total_time > 0 else 0,
            mean_latency=statistics.mean(self._results) if self._results else 0,
            p50_latency=self._percentile(sorted_results, 50),
            p95_latency=self._percentile(sorted_results, 95),
            p99_latency=self._percentile(sorted_results, 99),
            max_latency=max(self._results) if self._results else 0,
            errors=self._errors[:10]  # First 10 errors
        )
    
    def _execute_request(self, request_id: int):
        """Execute single request."""
        try:
            start = time.perf_counter()
            self.target()
            latency = time.perf_counter() - start
            
            with self._lock:
                self._results.append(latency)
        except Exception as e:
            with self._lock:
                self._errors.append(f"Request {request_id}: {str(e)}")
    
    def _percentile(self, data: list[float], p: int) -> float:
        """Calculate percentile."""
        if not data:
            return 0
        
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1
        
        if c >= len(data):
            return data[-1]
        
        return data[f] + (k - f) * (data[c] - data[f])


class MemoryProfiler:
    """Profile memory usage over time."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._samples: list[dict] = []
        self._running = False
        self._thread = None
    
    def start(self):
        """Start memory profiling."""
        if not HAS_PSUTIL:
            logger.warning("psutil not available for memory profiling")
            return
        
        self._running = True
        self._samples = []
        self._thread = threading.Thread(target=self._sample_loop)
        self._thread.start()
    
    def stop(self) -> list[dict]:
        """Stop profiling and return samples."""
        self._running = False
        
        if self._thread:
            self._thread.join()
        
        return self._samples
    
    def _sample_loop(self):
        """Sampling loop."""
        process = psutil.Process()
        start_time = time.time()
        
        while self._running:
            mem = process.memory_info()
            
            self._samples.append({
                "time": time.time() - start_time,
                "rss_mb": mem.rss / 1024 / 1024,
                "vms_mb": mem.vms / 1024 / 1024
            })
            
            time.sleep(self.interval)
    
    def get_peak_memory(self) -> float:
        """Get peak RSS memory in MB."""
        if not self._samples:
            return 0
        return max(s["rss_mb"] for s in self._samples)
    
    def get_memory_growth(self) -> float:
        """Get memory growth in MB."""
        if len(self._samples) < 2:
            return 0
        return self._samples[-1]["rss_mb"] - self._samples[0]["rss_mb"]


# Built-in benchmarks for Enigma AI Engine
def create_forge_benchmarks() -> BenchmarkSuite:
    """Create benchmark suite for Enigma AI Engine components."""
    suite = BenchmarkSuite()
    
    @suite.register("tokenizer_encode")
    def bench_tokenizer():
        """Benchmark tokenizer encoding."""
        try:
            from enigma_engine.core.tokenizer import get_tokenizer
            tokenizer = get_tokenizer()
            text = "Hello, this is a test sentence for benchmarking the tokenizer performance."
            tokenizer.encode(text)
        except ImportError:
            pass  # Intentionally silent
    
    @suite.register("embedding_compute")
    def bench_embedding():
        """Benchmark embedding computation."""
        try:
            import numpy as np

            # Simulate embedding computation
            x = np.random.randn(512)
            y = np.random.randn(512)
            np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        except ImportError:
            pass  # Intentionally silent
    
    @suite.register("json_serialize")
    def bench_json():
        """Benchmark JSON serialization."""
        data = {
            "messages": [
                {"role": "user", "content": f"Message {i}"}
                for i in range(100)
            ],
            "metadata": {"timestamp": time.time()}
        }
        json.dumps(data)
    
    @suite.register("matrix_multiply")
    def bench_matmul():
        """Benchmark matrix multiplication."""
        try:
            import numpy as np
            a = np.random.randn(256, 256)
            b = np.random.randn(256, 256)
            np.matmul(a, b)
        except ImportError:
            pass  # Intentionally silent
    
    return suite


def run_benchmarks():
    """Run all Enigma AI Engine benchmarks."""
    suite = create_forge_benchmarks()
    results = suite.run_all(iterations=100, warmup=10)
    suite.save_results()
    
    print("\nBenchmark Results:")
    print("-" * 60)
    
    for result in results:
        print(f"{result.name}:")
        print(f"  Mean: {result.mean_time*1000:.3f}ms")
        print(f"  Throughput: {result.throughput:.1f} ops/sec")
        print("")
    
    return results
