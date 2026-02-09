"""
Model Benchmarking System for Enigma AI Engine

Automated performance testing and comparison of models.

Usage:
    from enigma_engine.core.benchmarks import ModelBenchmark, run_benchmark
    
    # Run benchmark on a model
    results = run_benchmark(
        model_path="models/my_model",
        benchmark_suite="standard"
    )
    
    # Compare multiple models
    comparison = compare_models([
        "models/model_a",
        "models/model_b"
    ])
"""

import gc
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark test."""
    test_name: str
    score: float
    metric_unit: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


@dataclass
class ModelBenchmarkReport:
    """Complete benchmark report for a model."""
    model_name: str
    model_path: str
    model_size_mb: float
    param_count: int
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Results by category
    speed_results: List[BenchmarkResult] = field(default_factory=list)
    quality_results: List[BenchmarkResult] = field(default_factory=list)
    memory_results: List[BenchmarkResult] = field(default_factory=list)
    
    # Summary scores (0-100)
    speed_score: float = 0.0
    quality_score: float = 0.0
    memory_score: float = 0.0
    overall_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "model_size_mb": self.model_size_mb,
            "param_count": self.param_count,
            "timestamp": self.timestamp,
            "speed_results": [asdict(r) for r in self.speed_results],
            "quality_results": [asdict(r) for r in self.quality_results],
            "memory_results": [asdict(r) for r in self.memory_results],
            "speed_score": self.speed_score,
            "quality_score": self.quality_score,
            "memory_score": self.memory_score,
            "overall_score": self.overall_score
        }


# Standard test prompts
SPEED_TEST_PROMPTS = [
    "Hello",
    "What is 2 + 2?",
    "Explain the concept of machine learning in one sentence.",
    "Write a haiku about programming.",
    "List three benefits of open source software.",
]

QUALITY_TEST_CASES = [
    # (prompt, expected_contains, description)
    ("What is 2 + 2?", ["4"], "Basic arithmetic"),
    ("Is the sky blue? Answer yes or no.", ["yes"], "Simple fact"),
    ("Complete: The capital of France is", ["Paris"], "Basic knowledge"),
    ("What color do you get mixing red and blue?", ["purple", "violet"], "Basic reasoning"),
]


class ModelBenchmark:
    """
    Benchmarking suite for Enigma models.
    
    Tests:
    - Speed: Tokens per second, time to first token, latency
    - Quality: Accuracy on test cases, coherence
    - Memory: VRAM usage, peak memory
    """
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self._results: List[BenchmarkResult] = []
    
    def _load_model(self):
        """Load the model for benchmarking."""
        from .inference import EnigmaEngine
        
        logger.info(f"Loading model from {self.model_path}")
        self.engine = EnigmaEngine(str(self.model_path))
        
        # Get model info
        if hasattr(self.engine, 'model'):
            self.model = self.engine.model
            self.param_count = sum(p.numel() for p in self.model.parameters())
        else:
            self.param_count = 0
        
        # Get size on disk
        model_file = self.model_path / "forge_model.pt"
        if model_file.exists():
            self.model_size_mb = model_file.stat().st_size / (1024 * 1024)
        else:
            self.model_size_mb = 0
    
    def _unload_model(self):
        """Unload model and free memory."""
        if hasattr(self, 'engine'):
            del self.engine
        if self.model:
            del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def run_speed_benchmarks(self) -> List[BenchmarkResult]:
        """Run speed-related benchmarks."""
        results = []
        
        # Test 1: Time to first token
        logger.info("Running: Time to first token")
        times = []
        for prompt in SPEED_TEST_PROMPTS[:3]:
            start = time.perf_counter()
            # Generate minimal response
            try:
                response = self.engine.generate(prompt, max_gen=1)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
        
        if times:
            avg_ttft = sum(times) / len(times) * 1000  # ms
            results.append(BenchmarkResult(
                test_name="Time to First Token",
                score=avg_ttft,
                metric_unit="ms",
                passed=avg_ttft < 500,  # < 500ms is good
                duration_ms=sum(times) * 1000
            ))
        
        # Test 2: Tokens per second
        logger.info("Running: Tokens per second")
        total_tokens = 0
        total_time = 0
        
        for prompt in SPEED_TEST_PROMPTS:
            start = time.perf_counter()
            try:
                response = self.engine.generate(prompt, max_gen=50)
                elapsed = time.perf_counter() - start
                # Estimate tokens (rough)
                tokens = len(response.split()) * 1.3
                total_tokens += tokens
                total_time += elapsed
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
        
        if total_time > 0:
            tps = total_tokens / total_time
            results.append(BenchmarkResult(
                test_name="Tokens Per Second",
                score=tps,
                metric_unit="tokens/s",
                passed=tps > 10,  # > 10 tps is acceptable
                duration_ms=total_time * 1000
            ))
        
        # Test 3: Batch throughput (if supported)
        # Skipped for now - would need batch generation support
        
        return results
    
    def run_quality_benchmarks(self) -> List[BenchmarkResult]:
        """Run quality-related benchmarks."""
        results = []
        
        logger.info("Running: Quality tests")
        
        passed = 0
        total = 0
        
        for prompt, expected, description in QUALITY_TEST_CASES:
            start = time.perf_counter()
            try:
                response = self.engine.generate(prompt, max_gen=30)
                elapsed = time.perf_counter() - start
                
                # Check if response contains expected
                response_lower = response.lower()
                test_passed = any(exp.lower() in response_lower for exp in expected)
                
                if test_passed:
                    passed += 1
                total += 1
                
                results.append(BenchmarkResult(
                    test_name=f"Quality: {description}",
                    score=1.0 if test_passed else 0.0,
                    metric_unit="pass/fail",
                    passed=test_passed,
                    details={"prompt": prompt, "response": response[:100]},
                    duration_ms=elapsed * 1000
                ))
            except Exception as e:
                logger.warning(f"Quality test failed: {e}")
                total += 1
        
        # Overall accuracy
        if total > 0:
            accuracy = passed / total * 100
            results.insert(0, BenchmarkResult(
                test_name="Overall Accuracy",
                score=accuracy,
                metric_unit="%",
                passed=accuracy >= 50,
                details={"passed": passed, "total": total}
            ))
        
        return results
    
    def run_memory_benchmarks(self) -> List[BenchmarkResult]:
        """Run memory-related benchmarks."""
        results = []
        
        logger.info("Running: Memory benchmarks")
        
        # Test 1: Model memory footprint
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
            # Generate something to measure peak
            try:
                self.engine.generate("Test prompt for memory measurement", max_gen=50)
            except:
                pass
            
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            
            results.append(BenchmarkResult(
                test_name="Peak GPU Memory",
                score=peak_memory,
                metric_unit="MB",
                passed=peak_memory < 4096,  # < 4GB is acceptable
            ))
        
        # Test 2: Model size efficiency
        if self.param_count > 0 and self.model_size_mb > 0:
            params_per_mb = self.param_count / self.model_size_mb
            results.append(BenchmarkResult(
                test_name="Parameter Density",
                score=params_per_mb,
                metric_unit="params/MB",
                passed=True
            ))
        
        return results
    
    def run_all(self) -> ModelBenchmarkReport:
        """Run complete benchmark suite."""
        logger.info(f"Starting benchmark for {self.model_path}")
        
        self._load_model()
        
        try:
            # Run all benchmarks
            speed_results = self.run_speed_benchmarks()
            quality_results = self.run_quality_benchmarks()
            memory_results = self.run_memory_benchmarks()
            
            # Calculate scores (0-100 scale)
            speed_score = self._calculate_speed_score(speed_results)
            quality_score = self._calculate_quality_score(quality_results)
            memory_score = self._calculate_memory_score(memory_results)
            
            overall_score = (speed_score * 0.3 + quality_score * 0.5 + memory_score * 0.2)
            
            report = ModelBenchmarkReport(
                model_name=self.model_path.name,
                model_path=str(self.model_path),
                model_size_mb=self.model_size_mb,
                param_count=self.param_count,
                speed_results=speed_results,
                quality_results=quality_results,
                memory_results=memory_results,
                speed_score=speed_score,
                quality_score=quality_score,
                memory_score=memory_score,
                overall_score=overall_score
            )
            
            logger.info(f"Benchmark complete. Overall score: {overall_score:.1f}/100")
            return report
            
        finally:
            self._unload_model()
    
    def _calculate_speed_score(self, results: List[BenchmarkResult]) -> float:
        """Calculate speed score from results."""
        if not results:
            return 50.0
        
        score = 50.0
        for r in results:
            if r.test_name == "Time to First Token":
                # < 100ms = 100, > 1000ms = 0
                score += max(0, min(25, 25 * (1 - r.score / 1000)))
            elif r.test_name == "Tokens Per Second":
                # > 50 tps = 100, < 5 tps = 0
                score += max(0, min(25, r.score / 2))
        
        return min(100, max(0, score))
    
    def _calculate_quality_score(self, results: List[BenchmarkResult]) -> float:
        """Calculate quality score from results."""
        for r in results:
            if r.test_name == "Overall Accuracy":
                return r.score  # Already 0-100
        return 50.0
    
    def _calculate_memory_score(self, results: List[BenchmarkResult]) -> float:
        """Calculate memory score from results."""
        if not results:
            return 50.0
        
        for r in results:
            if r.test_name == "Peak GPU Memory":
                # < 1GB = 100, > 8GB = 0
                mb = r.score
                score = max(0, 100 - (mb / 80))
                return min(100, max(0, score))
        
        return 50.0


def run_benchmark(
    model_path: str,
    output_path: Optional[str] = None
) -> ModelBenchmarkReport:
    """
    Run benchmark on a model and optionally save results.
    
    Args:
        model_path: Path to model directory
        output_path: Optional path to save JSON report
        
    Returns:
        Benchmark report
    """
    benchmark = ModelBenchmark(model_path)
    report = benchmark.run_all()
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Saved benchmark report to {output_path}")
    
    return report


def compare_models(model_paths: List[str]) -> Dict[str, Any]:
    """
    Compare multiple models.
    
    Args:
        model_paths: List of model directories
        
    Returns:
        Comparison data with rankings
    """
    reports = []
    
    for path in model_paths:
        try:
            report = run_benchmark(path)
            reports.append(report)
        except Exception as e:
            logger.error(f"Failed to benchmark {path}: {e}")
    
    if not reports:
        return {"error": "No models benchmarked successfully"}
    
    # Sort by overall score
    reports.sort(key=lambda r: r.overall_score, reverse=True)
    
    return {
        "rankings": [
            {
                "rank": i + 1,
                "model": r.model_name,
                "overall_score": r.overall_score,
                "speed_score": r.speed_score,
                "quality_score": r.quality_score,
                "memory_score": r.memory_score,
                "size_mb": r.model_size_mb
            }
            for i, r in enumerate(reports)
        ],
        "best_overall": reports[0].model_name,
        "best_speed": max(reports, key=lambda r: r.speed_score).model_name,
        "best_quality": max(reports, key=lambda r: r.quality_score).model_name,
        "smallest": min(reports, key=lambda r: r.model_size_mb).model_name
    }


def quick_benchmark(model_path: str) -> Dict[str, float]:
    """
    Run a quick benchmark (speed only).
    
    Returns:
        Dict with tokens_per_second and time_to_first_token_ms
    """
    benchmark = ModelBenchmark(model_path)
    benchmark._load_model()
    
    try:
        # Single speed test
        start = time.perf_counter()
        response = benchmark.engine.generate("Hello, how are you?", max_gen=30)
        elapsed = time.perf_counter() - start
        
        tokens = len(response.split()) * 1.3
        
        return {
            "tokens_per_second": tokens / elapsed,
            "time_to_first_token_ms": elapsed * 1000 / 2,  # Estimate
            "response_length": len(response)
        }
    finally:
        benchmark._unload_model()
