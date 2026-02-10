"""
Evaluation Suite

Automated benchmarking for language models.
Supports MMLU, HellaSwag, HumanEval, and custom evaluations.

FILE: enigma_engine/core/eval_suite.py
TYPE: Training/Testing
MAIN CLASSES: EvaluationSuite, Benchmark, EvaluationResult
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks."""
    MULTIPLE_CHOICE = "multiple_choice"
    TEXT_COMPLETION = "text_completion"
    CODE_GENERATION = "code_generation"
    QUESTION_ANSWERING = "qa"
    REASONING = "reasoning"
    CUSTOM = "custom"


@dataclass
class BenchmarkSample:
    """A single benchmark sample."""
    id: str
    prompt: str
    choices: list[str] = field(default_factory=list)
    correct_answer: Any = None
    category: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result from evaluating a sample."""
    sample_id: str
    prediction: Any
    correct: bool
    score: float = 0.0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    benchmark_name: str
    total_samples: int
    correct_count: int
    accuracy: float
    avg_latency_ms: float
    results_by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    samples: list[EvaluationResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark": self.benchmark_name,
            "total_samples": self.total_samples,
            "correct": self.correct_count,
            "accuracy": self.accuracy,
            "avg_latency_ms": self.avg_latency_ms,
            "by_category": self.results_by_category,
            "timestamp": self.timestamp
        }


class Benchmark(ABC):
    """Abstract base class for benchmarks."""
    
    def __init__(self, name: str, benchmark_type: BenchmarkType):
        self.name = name
        self.benchmark_type = benchmark_type
        self._samples: list[BenchmarkSample] = []
    
    @abstractmethod
    def load_data(self, path: Optional[Path] = None):
        """Load benchmark data."""
    
    @abstractmethod
    def evaluate_sample(self, sample: BenchmarkSample, prediction: str) -> EvaluationResult:
        """Evaluate a single sample prediction."""
    
    def get_samples(self, limit: int = 0) -> list[BenchmarkSample]:
        """Get benchmark samples."""
        if limit > 0:
            return self._samples[:limit]
        return self._samples


class MMLUBenchmark(Benchmark):
    """Massive Multitask Language Understanding benchmark."""
    
    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging",
        "human_sexuality", "international_law", "jurisprudence",
        "logical_fallacies", "machine_learning", "management", "marketing",
        "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions"
    ]
    
    def __init__(self):
        super().__init__("MMLU", BenchmarkType.MULTIPLE_CHOICE)
    
    def load_data(self, path: Optional[Path] = None):
        """Load MMLU data."""
        sample_questions = [
            {
                "question": "What is the derivative of x^2?",
                "choices": ["x", "2x", "2", "x^2"],
                "answer": 1,
                "category": "college_mathematics"
            },
            {
                "question": "The mitochondria is often called the:",
                "choices": ["brain of the cell", "powerhouse of the cell", "wall of the cell", "nucleus"],
                "answer": 1,
                "category": "biology"
            },
            {
                "question": "In Python, which keyword is used to define a function?",
                "choices": ["function", "def", "define", "func"],
                "answer": 1,
                "category": "computer_science"
            }
        ]
        
        for i, q in enumerate(sample_questions):
            self._samples.append(BenchmarkSample(
                id=f"mmlu_{i}",
                prompt=self._format_prompt(q["question"], q["choices"]),
                choices=q["choices"],
                correct_answer=q["answer"],
                category=q["category"]
            ))
        
        logger.info(f"Loaded {len(self._samples)} MMLU samples")
    
    def _format_prompt(self, question: str, choices: list[str]) -> str:
        """Format question as multiple choice prompt."""
        prompt = f"Question: {question}\n\n"
        for i, choice in enumerate(choices):
            label = chr(65 + i)
            prompt += f"{label}. {choice}\n"
        prompt += "\nAnswer:"
        return prompt
    
    def evaluate_sample(self, sample: BenchmarkSample, prediction: str) -> EvaluationResult:
        """Evaluate MMLU prediction."""
        prediction = prediction.strip().upper()
        
        answer_match = re.search(r'\b([A-D])\b', prediction)
        if answer_match:
            predicted_idx = ord(answer_match.group(1)) - ord('A')
        else:
            predicted_idx = -1
        
        correct = predicted_idx == sample.correct_answer
        
        return EvaluationResult(
            sample_id=sample.id,
            prediction=predicted_idx,
            correct=correct,
            score=1.0 if correct else 0.0
        )


class HellaSwagBenchmark(Benchmark):
    """HellaSwag commonsense reasoning benchmark."""
    
    def __init__(self):
        super().__init__("HellaSwag", BenchmarkType.MULTIPLE_CHOICE)
    
    def load_data(self, path: Optional[Path] = None):
        """Load HellaSwag data."""
        sample_scenarios = [
            {
                "context": "A person is mixing ingredients in a bowl.",
                "choices": [
                    "They pour the mixture into a pan and put it in the oven.",
                    "They throw the bowl at the wall.",
                    "They start reading a book.",
                    "They begin to dance."
                ],
                "answer": 0
            },
            {
                "context": "Someone is sitting at a computer typing code.",
                "choices": [
                    "They run the program to test it.",
                    "They start swimming.",
                    "They climb a mountain.",
                    "They begin painting."
                ],
                "answer": 0
            }
        ]
        
        for i, s in enumerate(sample_scenarios):
            prompt = f"Context: {s['context']}\n\nWhat happens next?\n"
            for j, choice in enumerate(s['choices']):
                prompt += f"{chr(65+j)}. {choice}\n"
            prompt += "\nAnswer:"
            
            self._samples.append(BenchmarkSample(
                id=f"hellaswag_{i}",
                prompt=prompt,
                choices=s['choices'],
                correct_answer=s['answer'],
                category="commonsense"
            ))
        
        logger.info(f"Loaded {len(self._samples)} HellaSwag samples")
    
    def evaluate_sample(self, sample: BenchmarkSample, prediction: str) -> EvaluationResult:
        """Evaluate HellaSwag prediction."""
        prediction = prediction.strip().upper()
        
        answer_match = re.search(r'\b([A-D])\b', prediction)
        if answer_match:
            predicted_idx = ord(answer_match.group(1)) - ord('A')
        else:
            predicted_idx = -1
        
        correct = predicted_idx == sample.correct_answer
        
        return EvaluationResult(
            sample_id=sample.id,
            prediction=predicted_idx,
            correct=correct,
            score=1.0 if correct else 0.0
        )


class HumanEvalBenchmark(Benchmark):
    """HumanEval code generation benchmark."""
    
    def __init__(self):
        super().__init__("HumanEval", BenchmarkType.CODE_GENERATION)
    
    def load_data(self, path: Optional[Path] = None):
        """Load HumanEval problems."""
        sample_problems = [
            {
                "prompt": '''def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """''',
                "test": '''
assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False
assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
''',
                "entry_point": "has_close_elements"
            },
            {
                "prompt": '''def add(a: int, b: int) -> int:
    """Add two integers.
    >>> add(2, 3)
    5
    >>> add(-1, 1)
    0
    """''',
                "test": '''
assert add(2, 3) == 5
assert add(-1, 1) == 0
assert add(0, 0) == 0
''',
                "entry_point": "add"
            }
        ]
        
        for i, p in enumerate(sample_problems):
            self._samples.append(BenchmarkSample(
                id=f"humaneval_{i}",
                prompt=p['prompt'],
                correct_answer=p['test'],
                category="code",
                metadata={"entry_point": p['entry_point']}
            ))
        
        logger.info(f"Loaded {len(self._samples)} HumanEval problems")
    
    def evaluate_sample(self, sample: BenchmarkSample, prediction: str) -> EvaluationResult:
        """Evaluate code generation by running tests."""
        full_code = f"{sample.prompt}\n{prediction}\n{sample.correct_answer}"
        
        try:
            namespace = {"List": list}
            exec(full_code, namespace)
            correct = True
            score = 1.0
        except Exception:
            correct = False
            score = 0.0
        
        return EvaluationResult(
            sample_id=sample.id,
            prediction=prediction[:100],
            correct=correct,
            score=score
        )


class EvaluationSuite:
    """Runs multiple benchmarks on a model."""
    
    def __init__(self):
        """Initialize evaluation suite."""
        self._benchmarks: dict[str, Benchmark] = {}
        self._results: dict[str, BenchmarkResult] = {}
    
    def add_benchmark(self, benchmark: Benchmark):
        """Add a benchmark to the suite."""
        self._benchmarks[benchmark.name] = benchmark
    
    def add_standard_benchmarks(self):
        """Add standard benchmarks (MMLU, HellaSwag, HumanEval)."""
        mmlu = MMLUBenchmark()
        mmlu.load_data()
        self.add_benchmark(mmlu)
        
        hellaswag = HellaSwagBenchmark()
        hellaswag.load_data()
        self.add_benchmark(hellaswag)
        
        humaneval = HumanEvalBenchmark()
        humaneval.load_data()
        self.add_benchmark(humaneval)
    
    def evaluate(self,
                 model_fn: Callable[[str], str],
                 benchmarks: list[str] = None,
                 sample_limit: int = 0) -> dict[str, BenchmarkResult]:
        """Evaluate model on benchmarks."""
        results = {}
        
        benchmark_names = benchmarks or list(self._benchmarks.keys())
        
        for name in benchmark_names:
            if name not in self._benchmarks:
                logger.warning(f"Unknown benchmark: {name}")
                continue
            
            benchmark = self._benchmarks[name]
            result = self._run_benchmark(benchmark, model_fn, sample_limit)
            results[name] = result
            self._results[name] = result
            
            logger.info(f"{name}: {result.accuracy:.2%} accuracy ({result.correct_count}/{result.total_samples})")
        
        return results
    
    def _run_benchmark(self,
                       benchmark: Benchmark,
                       model_fn: Callable[[str], str],
                       sample_limit: int) -> BenchmarkResult:
        """Run a single benchmark."""
        samples = benchmark.get_samples(sample_limit)
        
        results = []
        correct_count = 0
        total_latency = 0.0
        category_results: dict[str, dict[str, int]] = {}
        
        for sample in samples:
            start = time.time()
            try:
                prediction = model_fn(sample.prompt)
            except Exception as e:
                logger.error(f"Model error on {sample.id}: {e}")
                prediction = ""
            latency = (time.time() - start) * 1000
            total_latency += latency
            
            result = benchmark.evaluate_sample(sample, prediction)
            result.latency_ms = latency
            results.append(result)
            
            if result.correct:
                correct_count += 1
            
            cat = sample.category or "uncategorized"
            if cat not in category_results:
                category_results[cat] = {"total": 0, "correct": 0}
            category_results[cat]["total"] += 1
            if result.correct:
                category_results[cat]["correct"] += 1
        
        category_accuracies = {}
        for cat, counts in category_results.items():
            category_accuracies[cat] = {
                "accuracy": counts["correct"] / counts["total"] if counts["total"] > 0 else 0,
                "total": counts["total"],
                "correct": counts["correct"]
            }
        
        return BenchmarkResult(
            benchmark_name=benchmark.name,
            total_samples=len(samples),
            correct_count=correct_count,
            accuracy=correct_count / len(samples) if samples else 0,
            avg_latency_ms=total_latency / len(samples) if samples else 0,
            results_by_category=category_accuracies,
            samples=results
        )
    
    def save_results(self, path: Path):
        """Save all results to JSON."""
        path = Path(path)
        
        data = {
            name: result.to_dict()
            for name, result in self._results.items()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved results to {path}")
    
    def get_summary(self) -> dict[str, float]:
        """Get summary of all benchmark results."""
        return {
            name: result.accuracy
            for name, result in self._results.items()
        }


def evaluate_model(model_fn: Callable[[str], str],
                   benchmarks: list[str] = None) -> dict[str, float]:
    """Quick model evaluation."""
    suite = EvaluationSuite()
    suite.add_standard_benchmarks()
    
    results = suite.evaluate(model_fn, benchmarks)
    return {name: r.accuracy for name, r in results.items()}


__all__ = [
    'EvaluationSuite',
    'Benchmark',
    'BenchmarkResult',
    'BenchmarkSample',
    'EvaluationResult',
    'BenchmarkType',
    'MMLUBenchmark',
    'HellaSwagBenchmark',
    'HumanEvalBenchmark',
    'evaluate_model'
]
