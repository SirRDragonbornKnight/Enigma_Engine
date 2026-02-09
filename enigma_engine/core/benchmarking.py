"""
Model Benchmarking for Enigma AI Engine

Performance comparison across models and configurations.

Features:
- Standardized benchmark suites
- Speed and quality metrics
- Memory profiling
- Parallel benchmark execution
- Historical tracking
- Exportable reports

Usage:
    from enigma_engine.core.benchmarking import ModelBenchmark, run_benchmarks
    
    # Quick benchmark
    results = run_benchmarks("models/my_model", benchmarks=["perplexity", "inference_speed"])
    
    # Detailed benchmark
    benchmark = ModelBenchmark()
    benchmark.load_model("models/my_model")
    benchmark.run_all()
    benchmark.save_report("benchmarks/report.json")
"""

import gc
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""
    name: str
    value: float
    unit: str
    higher_is_better: bool = True
    
    # Details
    samples: int = 1
    std_dev: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    
    # Context
    timestamp: str = ""
    duration_seconds: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "samples": self.samples,
            "std_dev": self.std_dev,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report for a model."""
    model_name: str
    model_path: str
    
    # Results
    results: List[BenchmarkResult] = field(default_factory=list)
    
    # System info
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    timestamp: str = ""
    total_duration: float = 0.0
    
    # Model info
    model_params: int = 0
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "results": [r.to_dict() for r in self.results],
            "system_info": self.system_info,
            "timestamp": self.timestamp,
            "total_duration": self.total_duration,
            "model_params": self.model_params,
            "model_config": self.model_config,
        }
    
    def get_result(self, name: str) -> Optional[BenchmarkResult]:
        """Get result by benchmark name."""
        for result in self.results:
            if result.name == name:
                return result
        return None
    
    def save(self, path: str):
        """Save report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BenchmarkReport':
        """Load report from file."""
        with open(path) as f:
            data = json.load(f)
        
        report = cls(
            model_name=data["model_name"],
            model_path=data["model_path"],
            system_info=data.get("system_info", {}),
            timestamp=data.get("timestamp", ""),
            total_duration=data.get("total_duration", 0),
            model_params=data.get("model_params", 0),
            model_config=data.get("model_config", {}),
        )
        
        for result_data in data.get("results", []):
            report.results.append(BenchmarkResult(**result_data))
        
        return report


# Benchmark test prompts
BENCHMARK_PROMPTS = {
    "short": [
        "The capital of France is",
        "Hello, my name is",
        "The weather today is",
        "I want to learn",
        "The best way to",
    ],
    "medium": [
        "Artificial intelligence is transforming the world by",
        "The process of photosynthesis involves several key steps:",
        "In computer science, algorithms are fundamental because",
        "The history of the internet began when",
        "Climate change affects ecosystems through",
    ],
    "long": [
        "Write a detailed explanation of how neural networks learn through backpropagation, including the mathematical foundations and practical considerations:",
        "Describe the complete process of software development from requirements gathering to deployment, covering all major phases and best practices:",
        "Explain the economic impact of technological innovation on developing countries, considering both positive and negative effects:",
    ],
}


class ModelBenchmark:
    """
    Comprehensive model benchmarking system.
    """
    
    def __init__(self, warmup_runs: int = 3, test_runs: int = 10):
        """
        Initialize benchmarker.
        
        Args:
            warmup_runs: Number of warmup iterations
            test_runs: Number of test iterations
        """
        self._warmup_runs = warmup_runs
        self._test_runs = test_runs
        
        self._model = None
        self._tokenizer = None
        self._model_path = ""
        self._model_name = ""
        
        self._report: Optional[BenchmarkReport] = None
        
        # Registered benchmarks
        self._benchmarks: Dict[str, Callable] = {
            "inference_speed": self._benchmark_inference_speed,
            "tokens_per_second": self._benchmark_tokens_per_second,
            "memory_usage": self._benchmark_memory,
            "first_token_latency": self._benchmark_first_token,
            "perplexity": self._benchmark_perplexity,
            "batch_throughput": self._benchmark_batch_throughput,
        }
    
    def load_model(self, model_path: str, model_name: str = ""):
        """
        Load model for benchmarking.
        
        Args:
            model_path: Path to model
            model_name: Optional friendly name
        """
        try:
            from enigma_engine.core.model import Forge
            from enigma_engine.core.tokenizer import get_tokenizer
            
            self._model_path = model_path
            self._model_name = model_name or Path(model_path).name
            
            logger.info(f"Loading model: {model_path}")
            self._model = Forge.from_pretrained(model_path)
            self._tokenizer = get_tokenizer()
            
            # Put in eval mode
            if TORCH_AVAILABLE and hasattr(self._model, 'eval'):
                self._model.eval()
            
            logger.info("Model loaded for benchmarking")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def run_benchmark(self, name: str) -> BenchmarkResult:
        """
        Run a single benchmark.
        
        Args:
            name: Benchmark name
            
        Returns:
            Benchmark result
        """
        if name not in self._benchmarks:
            raise ValueError(f"Unknown benchmark: {name}")
        
        if self._model is None:
            raise RuntimeError("No model loaded")
        
        logger.info(f"Running benchmark: {name}")
        
        start_time = time.time()
        result = self._benchmarks[name]()
        result.duration_seconds = time.time() - start_time
        result.timestamp = datetime.now().isoformat()
        
        logger.info(f"  Result: {result.value:.2f} {result.unit}")
        
        return result
    
    def run_all(self, benchmarks: Optional[List[str]] = None) -> BenchmarkReport:
        """
        Run all benchmarks.
        
        Args:
            benchmarks: Optional list to run (default: all)
            
        Returns:
            Complete benchmark report
        """
        if benchmarks is None:
            benchmarks = list(self._benchmarks.keys())
        
        # Initialize report
        self._report = BenchmarkReport(
            model_name=self._model_name,
            model_path=self._model_path,
            timestamp=datetime.now().isoformat(),
            system_info=self._get_system_info(),
        )
        
        # Get model info
        if self._model:
            self._report.model_params = self._count_parameters()
            if hasattr(self._model, 'config'):
                self._report.model_config = self._model.config.__dict__.copy() if hasattr(self._model.config, '__dict__') else {}
        
        # Run benchmarks
        start_time = time.time()
        
        for name in benchmarks:
            try:
                result = self.run_benchmark(name)
                self._report.results.append(result)
            except Exception as e:
                logger.error(f"Benchmark {name} failed: {e}")
        
        self._report.total_duration = time.time() - start_time
        
        return self._report
    
    def _benchmark_inference_speed(self) -> BenchmarkResult:
        """Benchmark average inference time per prompt."""
        prompts = BENCHMARK_PROMPTS["medium"]
        times = []
        
        # Warmup
        for _ in range(self._warmup_runs):
            self._generate(prompts[0], max_tokens=50)
        
        # Test runs
        for prompt in prompts * self._test_runs:
            start = time.perf_counter()
            self._generate(prompt, max_tokens=50)
            times.append(time.perf_counter() - start)
        
        return BenchmarkResult(
            name="inference_speed",
            value=statistics.mean(times) * 1000,  # Convert to ms
            unit="ms",
            higher_is_better=False,
            samples=len(times),
            std_dev=statistics.stdev(times) * 1000 if len(times) > 1 else 0,
            min_value=min(times) * 1000,
            max_value=max(times) * 1000,
        )
    
    def _benchmark_tokens_per_second(self) -> BenchmarkResult:
        """Benchmark token generation speed."""
        prompts = BENCHMARK_PROMPTS["medium"]
        tps_values = []
        
        # Warmup
        for _ in range(self._warmup_runs):
            self._generate(prompts[0], max_tokens=100)
        
        # Test runs
        for prompt in prompts * self._test_runs:
            start = time.perf_counter()
            output = self._generate(prompt, max_tokens=100)
            elapsed = time.perf_counter() - start
            
            # Count tokens in output
            tokens = len(self._tokenizer.encode(output)) if self._tokenizer else len(output.split())
            
            tps = tokens / elapsed if elapsed > 0 else 0
            tps_values.append(tps)
        
        return BenchmarkResult(
            name="tokens_per_second",
            value=statistics.mean(tps_values),
            unit="tok/s",
            higher_is_better=True,
            samples=len(tps_values),
            std_dev=statistics.stdev(tps_values) if len(tps_values) > 1 else 0,
            min_value=min(tps_values),
            max_value=max(tps_values),
        )
    
    def _benchmark_memory(self) -> BenchmarkResult:
        """Benchmark memory usage."""
        # Force garbage collection
        gc.collect()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # GPU memory
            memory_bytes = torch.cuda.memory_allocated()
            memory_mb = memory_bytes / (1024 * 1024)
            
            return BenchmarkResult(
                name="memory_usage",
                value=memory_mb,
                unit="MB (GPU)",
                higher_is_better=False,
                metadata={"device": "cuda"}
            )
        
        elif PSUTIL_AVAILABLE:
            # CPU memory
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            return BenchmarkResult(
                name="memory_usage",
                value=memory_mb,
                unit="MB (RAM)",
                higher_is_better=False,
                metadata={"device": "cpu"}
            )
        
        return BenchmarkResult(
            name="memory_usage",
            value=0,
            unit="MB",
            higher_is_better=False,
            metadata={"error": "Unable to measure memory"}
        )
    
    def _benchmark_first_token(self) -> BenchmarkResult:
        """Benchmark time to first token."""
        prompts = BENCHMARK_PROMPTS["medium"]
        times = []
        
        # Warmup
        for _ in range(self._warmup_runs):
            self._generate(prompts[0], max_tokens=1)
        
        # Test runs
        for prompt in prompts * self._test_runs:
            start = time.perf_counter()
            self._generate(prompt, max_tokens=1)
            times.append(time.perf_counter() - start)
        
        return BenchmarkResult(
            name="first_token_latency",
            value=statistics.mean(times) * 1000,
            unit="ms",
            higher_is_better=False,
            samples=len(times),
            std_dev=statistics.stdev(times) * 1000 if len(times) > 1 else 0,
            min_value=min(times) * 1000,
            max_value=max(times) * 1000,
        )
    
    def _benchmark_perplexity(self) -> BenchmarkResult:
        """Benchmark model perplexity."""
        # Use longer text samples
        test_texts = [
            "The quick brown fox jumps over the lazy dog. This is a common pangram used for testing.",
            "Machine learning models learn patterns from data through iterative optimization of parameters.",
            "The development of artificial intelligence has accelerated rapidly in recent years.",
        ]
        
        perplexities = []
        
        for text in test_texts:
            ppl = self._calculate_perplexity(text)
            if ppl > 0:
                perplexities.append(ppl)
        
        if not perplexities:
            return BenchmarkResult(
                name="perplexity",
                value=0,
                unit="PPL",
                higher_is_better=False,
                metadata={"error": "Unable to calculate perplexity"}
            )
        
        return BenchmarkResult(
            name="perplexity",
            value=statistics.mean(perplexities),
            unit="PPL",
            higher_is_better=False,
            samples=len(perplexities),
            std_dev=statistics.stdev(perplexities) if len(perplexities) > 1 else 0,
            min_value=min(perplexities),
            max_value=max(perplexities),
        )
    
    def _benchmark_batch_throughput(self) -> BenchmarkResult:
        """Benchmark batch processing throughput."""
        batch_sizes = [1, 2, 4, 8]
        prompts = BENCHMARK_PROMPTS["short"]
        
        best_throughput = 0
        best_batch_size = 1
        
        for batch_size in batch_sizes:
            try:
                batch = prompts[:batch_size] if batch_size <= len(prompts) else prompts * (batch_size // len(prompts) + 1)
                batch = batch[:batch_size]
                
                start = time.perf_counter()
                for prompt in batch:
                    self._generate(prompt, max_tokens=50)
                elapsed = time.perf_counter() - start
                
                throughput = batch_size / elapsed
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
                    
            except Exception:
                break
        
        return BenchmarkResult(
            name="batch_throughput",
            value=best_throughput,
            unit="prompts/s",
            higher_is_better=True,
            metadata={"best_batch_size": best_batch_size}
        )
    
    def _generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text from prompt."""
        try:
            from enigma_engine.core.inference import EnigmaEngine
            
            engine = EnigmaEngine(self._model, self._tokenizer)
            return engine.generate(prompt, max_gen=max_tokens)
            
        except Exception:
            # Fallback to direct model call (raw Forge model uses max_new_tokens)
            if hasattr(self._model, 'generate'):
                return self._model.generate(prompt, max_new_tokens=max_tokens)
            return ""
    
    def _calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity for text."""
        try:
            if not TORCH_AVAILABLE:
                return 0.0
            
            # Encode text
            tokens = self._tokenizer.encode(text)
            if len(tokens) < 2:
                return 0.0
            
            # Convert to tensor
            input_ids = torch.tensor([tokens])
            
            # Get model output
            with torch.no_grad():
                outputs = self._model(input_ids)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Calculate cross-entropy loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                perplexity = torch.exp(loss).item()
                return perplexity
                
        except Exception as e:
            logger.debug(f"Perplexity calculation failed: {e}")
            return 0.0
    
    def _count_parameters(self) -> int:
        """Count model parameters."""
        if TORCH_AVAILABLE and hasattr(self._model, 'parameters'):
            return sum(p.numel() for p in self._model.parameters())
        return 0
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "platform": __import__('platform').system(),
        }
        
        if TORCH_AVAILABLE:
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_device"] = torch.cuda.get_device_name(0)
                info["cuda_version"] = torch.version.cuda
        
        if PSUTIL_AVAILABLE:
            info["cpu_count"] = psutil.cpu_count()
            info["memory_total_gb"] = psutil.virtual_memory().total / (1024**3)
        
        return info
    
    def save_report(self, path: str):
        """Save current report to file."""
        if self._report:
            self._report.save(path)
    
    def compare_reports(
        self,
        reports: List[BenchmarkReport]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compare multiple benchmark reports.
        
        Args:
            reports: List of reports to compare
            
        Returns:
            Dict of benchmark name -> List of (model_name, value)
        """
        comparison = {}
        
        # Collect all benchmark names
        all_names = set()
        for report in reports:
            for result in report.results:
                all_names.add(result.name)
        
        # Build comparison
        for name in all_names:
            comparison[name] = []
            for report in reports:
                result = report.get_result(name)
                if result:
                    comparison[name].append((report.model_name, result.value))
        
        return comparison


def run_benchmarks(
    model_path: str,
    benchmarks: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> BenchmarkReport:
    """
    Quick function to run benchmarks on a model.
    
    Args:
        model_path: Path to model
        benchmarks: Optional list of benchmarks to run
        output_path: Optional path to save report
        
    Returns:
        Benchmark report
    """
    benchmark = ModelBenchmark()
    benchmark.load_model(model_path)
    report = benchmark.run_all(benchmarks)
    
    if output_path:
        report.save(output_path)
    
    return report
