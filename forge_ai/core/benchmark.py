"""
Benchmarking Suite for Forge_AI

Comprehensive benchmarks for:
- Inference throughput (tokens/sec)
- Latency (time to first token, total generation time)
- Memory usage (peak, average)
- Batch scaling efficiency
- Comparison with other frameworks

Usage:
    from forge_ai.core.benchmark import Benchmark
    
    bench = Benchmark(model, tokenizer)
    results = bench.run_all()
    bench.print_report(results)
    bench.save_report(results, "benchmark_results.json")
"""

import gc
import json
import logging
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""
    # Input settings
    prompt_lengths: list[int] = field(default_factory=lambda: [32, 128, 512, 1024])
    output_lengths: list[int] = field(default_factory=lambda: [32, 128, 256, 512])
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    
    # Run settings
    warmup_runs: int = 3
    benchmark_runs: int = 10
    timeout_seconds: float = 300.0
    
    # Memory settings
    track_memory: bool = True
    gc_between_runs: bool = True
    
    # Device
    device: str = "cuda"
    dtype: str = "float16"


@dataclass
class LatencyResult:
    """Latency benchmark results."""
    prompt_length: int
    output_length: int
    batch_size: int
    
    # Time metrics (seconds)
    time_to_first_token: float
    total_time: float
    per_token_latency: float
    
    # Throughput
    tokens_per_second: float
    
    # Statistics over runs
    total_time_std: float = 0.0
    ttft_std: float = 0.0


@dataclass
class ThroughputResult:
    """Throughput benchmark results."""
    batch_size: int
    prompt_length: int
    output_length: int
    
    # Throughput metrics
    tokens_per_second: float
    requests_per_second: float
    
    # Efficiency
    batch_efficiency: float  # Actual speedup vs linear
    
    # Statistics
    std_dev: float = 0.0


@dataclass
class MemoryResult:
    """Memory benchmark results."""
    model_memory_mb: float
    peak_memory_mb: float
    kv_cache_memory_mb: float
    
    # Per-batch memory
    memory_per_batch: dict[int, float] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    model_name: str
    model_params: int
    device: str
    dtype: str
    timestamp: str
    
    # Results
    latency_results: list[LatencyResult] = field(default_factory=list)
    throughput_results: list[ThroughputResult] = field(default_factory=list)
    memory_results: Optional[MemoryResult] = None
    
    # Summary
    summary: dict[str, Any] = field(default_factory=dict)


class Benchmark:
    """
    Benchmarking suite for Forge_AI models.
    
    Usage:
        bench = Benchmark(model, tokenizer)
        
        # Run all benchmarks
        report = bench.run_all()
        
        # Or run specific benchmarks
        latency = bench.benchmark_latency()
        throughput = bench.benchmark_throughput()
        memory = bench.benchmark_memory()
        
        # Print results
        bench.print_report(report)
        
        # Save results
        bench.save_report(report, "results.json")
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        config: Optional[BenchmarkConfig] = None,
        model_name: str = "forge_model"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or BenchmarkConfig()
        self.model_name = model_name
        
        # Determine device
        self.device = self._get_device()
        
        # Count parameters
        self.num_params = sum(p.numel() for p in model.parameters())
        
        # Sample prompts for benchmarking
        self._prompts: dict[int, str] = {}
    
    def _get_device(self) -> torch.device:
        """Get the device the model is on."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device(self.config.device)
    
    def _get_prompt(self, length: int) -> str:
        """Get or generate a prompt of specified token length."""
        if length not in self._prompts:
            # Generate a prompt of approximately the right length
            base = "The quick brown fox jumps over the lazy dog. "
            repeated = base * (length // 10 + 1)
            tokens = self.tokenizer.encode(repeated)[:length]
            self._prompts[length] = self.tokenizer.decode(tokens)
        return self._prompts[length]
    
    @contextmanager
    def _memory_tracker(self):
        """Context manager to track peak GPU memory."""
        if not torch.cuda.is_available():
            yield {"peak": 0, "allocated": 0}
            return
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        try:
            yield {
                "start_allocated": torch.cuda.memory_allocated(),
                "start_reserved": torch.cuda.memory_reserved()
            }
        finally:
            torch.cuda.synchronize()
    
    def _cleanup(self):
        """Clean up between benchmark runs."""
        if self.config.gc_between_runs:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @torch.inference_mode()
    def _run_inference(
        self,
        prompt: str,
        max_tokens: int,
        batch_size: int = 1
    ) -> tuple[float, float, int]:
        """
        Run a single inference and return timing.
        
        Returns:
            (time_to_first_token, total_time, tokens_generated)
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids] * batch_size).to(self.device)
        
        # Generate
        start_time = time.perf_counter()
        first_token_time = None
        
        generated = input_tensor.clone()
        
        for i in range(max_tokens):
            outputs = self.model(generated)
            logits = outputs[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            
            if first_token_time is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                first_token_time = time.perf_counter()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        ttft = (first_token_time - start_time) if first_token_time else 0
        total_time = end_time - start_time
        tokens_generated = max_tokens * batch_size
        
        return ttft, total_time, tokens_generated
    
    def benchmark_latency(self) -> list[LatencyResult]:
        """
        Benchmark inference latency for various configurations.
        """
        results = []
        
        for prompt_len in self.config.prompt_lengths:
            for output_len in self.config.output_lengths:
                for batch_size in [1]:  # Latency is typically measured at batch=1
                    prompt = self._get_prompt(prompt_len)
                    
                    # Warmup
                    for _ in range(self.config.warmup_runs):
                        self._run_inference(prompt, output_len, batch_size)
                        self._cleanup()
                    
                    # Benchmark runs
                    ttfts = []
                    total_times = []
                    tokens_list = []
                    
                    for _ in range(self.config.benchmark_runs):
                        ttft, total, tokens = self._run_inference(
                            prompt, output_len, batch_size
                        )
                        ttfts.append(ttft)
                        total_times.append(total)
                        tokens_list.append(tokens)
                        self._cleanup()
                    
                    avg_ttft = statistics.mean(ttfts)
                    avg_total = statistics.mean(total_times)
                    avg_tokens = statistics.mean(tokens_list)
                    
                    result = LatencyResult(
                        prompt_length=prompt_len,
                        output_length=output_len,
                        batch_size=batch_size,
                        time_to_first_token=avg_ttft,
                        total_time=avg_total,
                        per_token_latency=avg_total / output_len if output_len > 0 else 0,
                        tokens_per_second=avg_tokens / avg_total if avg_total > 0 else 0,
                        total_time_std=statistics.stdev(total_times) if len(total_times) > 1 else 0,
                        ttft_std=statistics.stdev(ttfts) if len(ttfts) > 1 else 0
                    )
                    results.append(result)
                    
                    logger.info(
                        f"Latency: prompt={prompt_len}, output={output_len}, "
                        f"ttft={avg_ttft:.3f}s, total={avg_total:.3f}s"
                    )
        
        return results
    
    def benchmark_throughput(self) -> list[ThroughputResult]:
        """
        Benchmark throughput at various batch sizes.
        """
        results = []
        baseline_tps: Optional[float] = None
        
        # Use medium prompt/output for throughput testing
        prompt_len = 128
        output_len = 128
        prompt = self._get_prompt(prompt_len)
        
        for batch_size in self.config.batch_sizes:
            # Skip if batch size might OOM
            if torch.cuda.is_available():
                free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                # Very rough estimate
                estimated_need = batch_size * prompt_len * 8 * 1e6  # 8MB per 1k tokens guess
                if estimated_need > free_mem * 0.8:
                    logger.warning(f"Skipping batch_size={batch_size}, might OOM")
                    continue
            
            # Warmup
            try:
                for _ in range(self.config.warmup_runs):
                    self._run_inference(prompt, output_len, batch_size)
                    self._cleanup()
            except Exception as e:
                logger.warning(f"Warmup failed for batch_size={batch_size}: {e}")
                continue
            
            # Benchmark
            times = []
            tokens_list = []
            
            try:
                for _ in range(self.config.benchmark_runs):
                    _, total, tokens = self._run_inference(prompt, output_len, batch_size)
                    times.append(total)
                    tokens_list.append(tokens)
                    self._cleanup()
            except Exception as e:
                logger.warning(f"Benchmark failed for batch_size={batch_size}: {e}")
                continue
            
            avg_time = statistics.mean(times)
            avg_tokens = statistics.mean(tokens_list)
            tps = avg_tokens / avg_time if avg_time > 0 else 0
            rps = batch_size / avg_time if avg_time > 0 else 0
            
            if baseline_tps is None:
                baseline_tps = tps
            
            efficiency = (tps / (baseline_tps * batch_size)) if baseline_tps > 0 else 0
            
            result = ThroughputResult(
                batch_size=batch_size,
                prompt_length=prompt_len,
                output_length=output_len,
                tokens_per_second=tps,
                requests_per_second=rps,
                batch_efficiency=efficiency,
                std_dev=statistics.stdev(times) if len(times) > 1 else 0
            )
            results.append(result)
            
            logger.info(
                f"Throughput: batch={batch_size}, "
                f"tps={tps:.1f}, rps={rps:.2f}, efficiency={efficiency:.2%}"
            )
        
        return results
    
    def benchmark_memory(self) -> MemoryResult:
        """
        Benchmark memory usage.
        """
        if not torch.cuda.is_available():
            return MemoryResult(
                model_memory_mb=0,
                peak_memory_mb=0,
                kv_cache_memory_mb=0
            )
        
        self._cleanup()
        torch.cuda.reset_peak_memory_stats()
        
        # Model memory
        model_mem = torch.cuda.memory_allocated() / 1e6
        
        # Run inference to measure peak
        prompt = self._get_prompt(512)
        self._run_inference(prompt, 256, batch_size=1)
        
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        
        # Estimate KV cache (peak - model - activations guess)
        kv_cache_mem = max(0, peak_mem - model_mem - 100)  # 100MB activation estimate
        
        # Memory per batch
        memory_per_batch = {}
        for batch_size in [1, 2, 4, 8]:
            self._cleanup()
            torch.cuda.reset_peak_memory_stats()
            
            try:
                self._run_inference(prompt, 256, batch_size)
                memory_per_batch[batch_size] = torch.cuda.max_memory_allocated() / 1e6
            except Exception:
                break
        
        return MemoryResult(
            model_memory_mb=model_mem,
            peak_memory_mb=peak_mem,
            kv_cache_memory_mb=kv_cache_mem,
            memory_per_batch=memory_per_batch
        )
    
    def run_all(self) -> BenchmarkReport:
        """Run all benchmarks and return a complete report."""
        import datetime
        
        logger.info(f"Starting benchmark for {self.model_name}")
        logger.info(f"Parameters: {self.num_params:,}")
        logger.info(f"Device: {self.device}")
        
        report = BenchmarkReport(
            model_name=self.model_name,
            model_params=self.num_params,
            device=str(self.device),
            dtype=self.config.dtype,
            timestamp=datetime.datetime.now().isoformat()
        )
        
        # Run benchmarks
        logger.info("Running latency benchmark...")
        report.latency_results = self.benchmark_latency()
        
        logger.info("Running throughput benchmark...")
        report.throughput_results = self.benchmark_throughput()
        
        if self.config.track_memory:
            logger.info("Running memory benchmark...")
            report.memory_results = self.benchmark_memory()
        
        # Generate summary
        report.summary = self._generate_summary(report)
        
        return report
    
    def _generate_summary(self, report: BenchmarkReport) -> dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        summary = {}
        
        # Best latency (batch=1)
        if report.latency_results:
            min_latency = min(r.total_time for r in report.latency_results)
            max_tps = max(r.tokens_per_second for r in report.latency_results)
            summary['min_latency_ms'] = min_latency * 1000
            summary['max_tokens_per_second_latency'] = max_tps
        
        # Best throughput
        if report.throughput_results:
            max_throughput = max(r.tokens_per_second for r in report.throughput_results)
            best_batch = max(report.throughput_results, key=lambda r: r.tokens_per_second)
            summary['max_tokens_per_second'] = max_throughput
            summary['optimal_batch_size'] = best_batch.batch_size
        
        # Memory
        if report.memory_results:
            summary['model_memory_mb'] = report.memory_results.model_memory_mb
            summary['peak_memory_mb'] = report.memory_results.peak_memory_mb
        
        return summary
    
    def print_report(self, report: BenchmarkReport):
        """Print a formatted benchmark report."""
        print("\n" + "=" * 60)
        print(f"BENCHMARK REPORT: {report.model_name}")
        print("=" * 60)
        print(f"Parameters: {report.model_params:,}")
        print(f"Device: {report.device}")
        print(f"Dtype: {report.dtype}")
        print(f"Timestamp: {report.timestamp}")
        
        print("\n--- LATENCY RESULTS ---")
        print(f"{'Prompt':>8} {'Output':>8} {'TTFT':>10} {'Total':>10} {'Tok/s':>10}")
        print("-" * 50)
        for r in report.latency_results:
            print(
                f"{r.prompt_length:>8} {r.output_length:>8} "
                f"{r.time_to_first_token*1000:>8.1f}ms {r.total_time*1000:>8.1f}ms "
                f"{r.tokens_per_second:>10.1f}"
            )
        
        print("\n--- THROUGHPUT RESULTS ---")
        print(f"{'Batch':>8} {'Tok/s':>12} {'Req/s':>10} {'Efficiency':>12}")
        print("-" * 45)
        for r in report.throughput_results:
            print(
                f"{r.batch_size:>8} {r.tokens_per_second:>12.1f} "
                f"{r.requests_per_second:>10.2f} {r.batch_efficiency:>11.1%}"
            )
        
        if report.memory_results:
            print("\n--- MEMORY RESULTS ---")
            print(f"Model Memory: {report.memory_results.model_memory_mb:.1f} MB")
            print(f"Peak Memory: {report.memory_results.peak_memory_mb:.1f} MB")
            print(f"KV Cache Est: {report.memory_results.kv_cache_memory_mb:.1f} MB")
        
        print("\n--- SUMMARY ---")
        for key, value in report.summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        print("=" * 60 + "\n")
    
    def save_report(self, report: BenchmarkReport, path: str):
        """Save benchmark report to JSON file."""
        data = {
            'model_name': report.model_name,
            'model_params': report.model_params,
            'device': report.device,
            'dtype': report.dtype,
            'timestamp': report.timestamp,
            'latency_results': [asdict(r) for r in report.latency_results],
            'throughput_results': [asdict(r) for r in report.throughput_results],
            'memory_results': asdict(report.memory_results) if report.memory_results else None,
            'summary': report.summary
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved benchmark report to {path}")


def run_quick_benchmark(model, tokenizer, model_name: str = "model") -> dict[str, float]:
    """
    Run a quick benchmark and return key metrics.
    
    Returns dict with:
    - tokens_per_second: Peak throughput
    - latency_ms: Average latency at batch=1
    - memory_mb: Peak memory usage
    """
    config = BenchmarkConfig(
        prompt_lengths=[128],
        output_lengths=[128],
        batch_sizes=[1, 4],
        warmup_runs=2,
        benchmark_runs=5
    )
    
    bench = Benchmark(model, tokenizer, config, model_name)
    report = bench.run_all()
    
    return {
        'tokens_per_second': report.summary.get('max_tokens_per_second', 0),
        'latency_ms': report.summary.get('min_latency_ms', 0),
        'memory_mb': report.summary.get('peak_memory_mb', 0)
    }
