"""
Inference Benchmark Utility

Measures and compares inference speed across different model sizes.
Useful for determining optimal model size for your hardware.

Usage:
    # From command line
    python -m enigma_engine.utils.benchmark_inference
    
    # From code
    from enigma_engine.utils.benchmark_inference import run_benchmark, quick_benchmark
    
    # Quick benchmark with current model
    results = quick_benchmark(engine)
    
    # Full benchmark across sizes
    results = run_benchmark(sizes=['tiny', 'small', 'medium'])
"""

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    model_size: str
    prompt_tokens: int
    generated_tokens: int
    total_time_ms: float
    tokens_per_second: float
    first_token_ms: float
    memory_mb: float
    device: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Full benchmark report with multiple runs."""
    results: list[BenchmarkResult]
    hardware_info: dict[str, Any]
    test_prompts: list[str]
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "FORGE AI INFERENCE BENCHMARK",
            "=" * 60,
            f"Device: {self.hardware_info.get('device', 'unknown')}",
            f"GPU: {self.hardware_info.get('gpu_name', 'N/A')}",
            f"RAM: {self.hardware_info.get('ram_gb', 0):.1f} GB",
            "",
            "-" * 60,
            f"{'Model Size':<12} {'Tokens/s':<12} {'First Token':<12} {'Memory MB':<12}",
            "-" * 60,
        ]
        
        # Group by model size
        by_size = {}
        for r in self.results:
            if r.model_size not in by_size:
                by_size[r.model_size] = []
            by_size[r.model_size].append(r)
        
        for size, runs in sorted(by_size.items()):
            avg_tps = statistics.mean(r.tokens_per_second for r in runs)
            avg_first = statistics.mean(r.first_token_ms for r in runs)
            avg_mem = statistics.mean(r.memory_mb for r in runs)
            lines.append(f"{size:<12} {avg_tps:<12.1f} {avg_first:<12.0f}ms {avg_mem:<12.0f}")
        
        lines.extend([
            "-" * 60,
            "",
            "Notes:",
            "- Tokens/s: Higher is better (generation speed)",
            "- First Token: Lower is better (response latency)",
            "- Memory: Lower is better (RAM/VRAM usage)",
        ])
        
        return "\n".join(lines)
    
    def save(self, path: Path):
        """Save benchmark results to JSON."""
        data = {
            'results': [r.to_dict() for r in self.results],
            'hardware_info': self.hardware_info,
            'test_prompts': self.test_prompts,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def get_hardware_info() -> dict[str, Any]:
    """Gather hardware information."""
    import torch
    
    info = {
        'device': 'cpu',
        'gpu_name': None,
        'gpu_memory_gb': 0,
        'ram_gb': 0,
        'cpu_count': 1,
    }
    
    try:
        import psutil
        info['ram_gb'] = psutil.virtual_memory().total / (1024**3)
        info['cpu_count'] = psutil.cpu_count()
    except ImportError:
        pass
    
    if torch.cuda.is_available():
        info['device'] = 'cuda'
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['device'] = 'mps'
        info['gpu_name'] = 'Apple MPS'
    
    return info


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    import torch
    
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)
    
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024**2)
    except ImportError:
        return 0


def benchmark_single(
    engine,
    prompt: str,
    max_tokens: int = 50,
    model_size: str = "unknown"
) -> BenchmarkResult:
    """
    Run a single benchmark on an inference engine.
    
    Args:
        engine: EnigmaEngine instance
        prompt: Text prompt to use
        max_tokens: Number of tokens to generate
        model_size: Label for the model size
        
    Returns:
        BenchmarkResult with timing information
    """
    import torch

    # Get prompt token count
    if hasattr(engine, 'tokenizer'):
        prompt_tokens = len(engine.tokenizer.encode(prompt))
    else:
        prompt_tokens = len(prompt.split())
    
    # Warm-up run
    _ = engine.generate(prompt, max_tokens=5, temperature=0.1)
    
    # Clear caches
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    memory_before = get_memory_usage()
    
    # Measure first token latency
    first_token_time = None
    tokens_generated = 0
    
    start_time = time.perf_counter()
    
    # Check if engine supports streaming
    if hasattr(engine, 'generate_stream'):
        for i, token in enumerate(engine.generate_stream(prompt, max_tokens=max_tokens)):
            if i == 0:
                first_token_time = time.perf_counter() - start_time
            tokens_generated += 1
    else:
        # Non-streaming fallback
        result = engine.generate(prompt, max_tokens=max_tokens, temperature=0.1)
        first_token_time = (time.perf_counter() - start_time) / 2  # Estimate
        if hasattr(engine, 'tokenizer'):
            tokens_generated = len(engine.tokenizer.encode(result)) - prompt_tokens
        else:
            tokens_generated = len(result.split()) - len(prompt.split())
    
    end_time = time.perf_counter()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    memory_after = get_memory_usage()
    
    total_time = (end_time - start_time) * 1000  # Convert to ms
    tokens_per_second = tokens_generated / (total_time / 1000) if total_time > 0 else 0
    
    hardware = get_hardware_info()
    
    return BenchmarkResult(
        model_size=model_size,
        prompt_tokens=prompt_tokens,
        generated_tokens=tokens_generated,
        total_time_ms=total_time,
        tokens_per_second=tokens_per_second,
        first_token_ms=(first_token_time or 0) * 1000,
        memory_mb=max(memory_after, memory_after - memory_before),
        device=hardware['device'],
    )


def quick_benchmark(
    engine,
    num_runs: int = 3,
    model_size: str = "current"
) -> dict[str, float]:
    """
    Quick benchmark of current engine.
    
    Args:
        engine: EnigmaEngine instance
        num_runs: Number of runs to average
        model_size: Label for the model
        
    Returns:
        Dict with average metrics
    """
    prompts = [
        "Once upon a time",
        "The future of AI is",
        "In a world where",
    ]
    
    results = []
    for i in range(num_runs):
        prompt = prompts[i % len(prompts)]
        result = benchmark_single(engine, prompt, max_tokens=30, model_size=model_size)
        results.append(result)
    
    return {
        'avg_tokens_per_second': statistics.mean(r.tokens_per_second for r in results),
        'avg_first_token_ms': statistics.mean(r.first_token_ms for r in results),
        'avg_memory_mb': statistics.mean(r.memory_mb for r in results),
        'runs': num_runs,
    }


def run_benchmark(
    sizes: Optional[list[str]] = None,
    num_runs: int = 3,
    max_tokens: int = 50,
    vocab_size: int = 8000,
) -> BenchmarkReport:
    """
    Run full benchmark across multiple model sizes.
    
    Args:
        sizes: List of model sizes to test (default: nano, tiny, small)
        num_runs: Number of runs per size
        max_tokens: Tokens to generate per run
        vocab_size: Vocabulary size for models
        
    Returns:
        BenchmarkReport with all results
    """
    from enigma_engine.core.inference import EnigmaEngine
    from enigma_engine.core.model import create_model
    from enigma_engine.core.tokenizer import get_tokenizer
    
    sizes = sizes or ['nano', 'tiny', 'small']
    
    test_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was nothing but darkness.",
        "Artificial intelligence will change the world by",
    ]
    
    hardware_info = get_hardware_info()
    results = []
    
    print(f"Running benchmark on {hardware_info['device']}...")
    print(f"Testing sizes: {sizes}")
    print("-" * 40)
    
    for size in sizes:
        print(f"\nTesting {size} model...")
        
        try:
            # Create model
            model = create_model(size, vocab_size=vocab_size)
            tokenizer = get_tokenizer(vocab_size=vocab_size)
            engine = EnigmaEngine(model, tokenizer)
            
            # Run benchmarks
            for i in range(num_runs):
                prompt = test_prompts[i % len(test_prompts)]
                result = benchmark_single(
                    engine, prompt, 
                    max_tokens=max_tokens, 
                    model_size=size
                )
                results.append(result)
                print(f"  Run {i+1}: {result.tokens_per_second:.1f} tok/s")
            
            # Cleanup
            del engine, model, tokenizer
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"  Error testing {size}: {e}")
            logger.exception(f"Benchmark error for {size}")
    
    return BenchmarkReport(
        results=results,
        hardware_info=hardware_info,
        test_prompts=test_prompts,
    )


def main():
    """Run benchmark from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Enigma AI Engine inference")
    parser.add_argument(
        '--sizes', 
        nargs='+', 
        default=['nano', 'tiny', 'small'],
        help='Model sizes to test'
    )
    parser.add_argument(
        '--runs', 
        type=int, 
        default=3,
        help='Number of runs per size'
    )
    parser.add_argument(
        '--tokens',
        type=int,
        default=50,
        help='Tokens to generate per run'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )
    
    args = parser.parse_args()
    
    report = run_benchmark(
        sizes=args.sizes,
        num_runs=args.runs,
        max_tokens=args.tokens,
    )
    
    print("\n")
    print(report.summary())
    
    if args.output:
        report.save(Path(args.output))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
