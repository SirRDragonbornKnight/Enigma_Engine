"""
Testing Package

Benchmarking and performance testing utilities.

Provides:
- BenchmarkSuite: Collection of performance benchmarks
- LoadTester: Concurrent request testing
- MemoryProfiler: Memory usage tracking
- Profiler: Timing profiler for code sections
- benchmark: Decorator for benchmarking functions
- timer: Context manager/decorator for timing
"""

from .benchmarks import (
    BenchmarkResult,
    LoadTestResult,
    benchmark,
    timer,
    Profiler,
    BenchmarkSuite,
    LoadTester,
    MemoryProfiler,
    create_forge_benchmarks,
    run_benchmarks,
)

__all__ = [
    # Result types
    "BenchmarkResult",
    "LoadTestResult",
    # Decorators/context managers
    "benchmark",
    "timer",
    # Profilers
    "Profiler",
    "MemoryProfiler",
    # Benchmark suite
    "BenchmarkSuite",
    "LoadTester",
    # Helpers
    "create_forge_benchmarks",
    "run_benchmarks",
]
