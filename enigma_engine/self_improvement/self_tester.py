"""
Self-Tester Module for Autonomous AI Improvement

Verifies that the AI has learned new features correctly
by running test questions and checking response quality.

Features:
- Automatic test generation from training data
- Quality scoring (relevance, accuracy, coherence)
- Comparison with baseline performance
- Regression detection
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single test case."""
    question: str
    expected_keywords: List[str] = field(default_factory=list)
    expected_pattern: str = ""
    category: str = "general"
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "expected_keywords": self.expected_keywords,
            "expected_pattern": self.expected_pattern,
            "category": self.category,
            "weight": self.weight,
        }


@dataclass
class TestResult:
    """Result of a single test."""
    test_case: TestCase
    response: str
    score: float = 0.0  # 0-1
    passed: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.test_case.question,
            "response": self.response[:500],  # Truncate for storage
            "score": self.score,
            "passed": self.passed,
            "metrics": self.metrics,
            "error": self.error,
        }


@dataclass
class TestSuiteResult:
    """Result of running a full test suite."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    quality_score: float = 0.0  # 0-1, weighted average
    category_scores: Dict[str, float] = field(default_factory=dict)
    results: List[TestResult] = field(default_factory=list)
    baseline_score: Optional[float] = None
    regression_detected: bool = False
    test_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "quality_score": self.quality_score,
            "category_scores": self.category_scores,
            "baseline_score": self.baseline_score,
            "regression_detected": self.regression_detected,
            "test_time": self.test_time,
            "results": [r.to_dict() for r in self.results],
        }


class ResponseScorer:
    """
    Scores AI responses for quality.
    
    Evaluates:
    - Keyword presence (does response contain expected terms)
    - Pattern matching (does response follow expected format)
    - Coherence (is response well-formed)
    - Relevance (does response address the question)
    - Length appropriateness
    """
    
    # Minimum response length (too short = bad)
    MIN_LENGTH = 20
    
    # Maximum response length (too long may indicate rambling)
    MAX_LENGTH = 2000
    
    # Weight for different score components
    WEIGHTS = {
        "keyword_score": 0.35,
        "pattern_score": 0.25,
        "coherence_score": 0.20,
        "relevance_score": 0.15,
        "length_score": 0.05,
    }
    
    def score(self, response: str, test_case: TestCase) -> Tuple[float, Dict[str, float]]:
        """
        Score a response against a test case.
        
        Returns:
            Tuple of (overall_score, component_scores)
        """
        metrics = {}
        
        # Keyword score
        if test_case.expected_keywords:
            metrics["keyword_score"] = self._score_keywords(
                response, test_case.expected_keywords
            )
        else:
            metrics["keyword_score"] = 1.0  # No keywords to check
        
        # Pattern score
        if test_case.expected_pattern:
            metrics["pattern_score"] = self._score_pattern(
                response, test_case.expected_pattern
            )
        else:
            metrics["pattern_score"] = 1.0  # No pattern to check
        
        # Coherence score
        metrics["coherence_score"] = self._score_coherence(response)
        
        # Relevance score
        metrics["relevance_score"] = self._score_relevance(
            response, test_case.question
        )
        
        # Length score
        metrics["length_score"] = self._score_length(response)
        
        # Calculate weighted average
        total_score = sum(
            metrics[key] * self.WEIGHTS[key]
            for key in self.WEIGHTS
        )
        
        return total_score, metrics
    
    def _score_keywords(self, response: str, keywords: List[str]) -> float:
        """Score based on keyword presence."""
        if not keywords:
            return 1.0
        
        response_lower = response.lower()
        found = sum(1 for kw in keywords if kw.lower() in response_lower)
        
        return found / len(keywords)
    
    def _score_pattern(self, response: str, pattern: str) -> float:
        """Score based on regex pattern match."""
        try:
            if re.search(pattern, response, re.IGNORECASE | re.DOTALL):
                return 1.0
            return 0.0
        except re.error:
            logger.warning(f"Invalid regex pattern: {pattern}")
            return 0.5  # Neutral if pattern is invalid
    
    def _score_coherence(self, response: str) -> float:
        """Score response coherence."""
        score = 1.0
        
        # Check for incomplete sentences
        if response and not response.strip()[-1] in '.!?':
            score -= 0.1
        
        # Check for repeated words (sign of generation issues)
        words = response.lower().split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                score -= 0.3
        
        # Check for obvious errors
        error_patterns = [
            r'\b(\w+)\s+\1\s+\1\b',  # Same word 3+ times
            r'[^\x00-\x7F]{10,}',  # Long non-ASCII sequences
        ]
        for pattern in error_patterns:
            if re.search(pattern, response):
                score -= 0.2
        
        return max(0, score)
    
    def _score_relevance(self, response: str, question: str) -> float:
        """Score how relevant the response is to the question."""
        # Extract key terms from question
        question_words = set(
            word.lower() for word in re.findall(r'\b\w+\b', question)
            if len(word) > 3 and word.lower() not in {'what', 'when', 'where', 'which', 'does', 'have', 'this', 'that', 'with'}
        )
        
        if not question_words:
            return 0.8  # Can't determine relevance
        
        response_lower = response.lower()
        found = sum(1 for word in question_words if word in response_lower)
        
        # At least some question terms should appear
        return min(1.0, found / max(1, len(question_words) * 0.5))
    
    def _score_length(self, response: str) -> float:
        """Score response length appropriateness."""
        length = len(response.strip())
        
        if length < self.MIN_LENGTH:
            return length / self.MIN_LENGTH
        elif length > self.MAX_LENGTH:
            return max(0.5, 1 - (length - self.MAX_LENGTH) / self.MAX_LENGTH)
        
        return 1.0


class SelfTester:
    """
    Tests AI quality after self-training.
    
    Usage:
        tester = SelfTester()
        
        # Run tests
        result = tester.test()
        
        # Check quality
        if result.quality_score >= 0.7:
            print("Training successful!")
        elif result.regression_detected:
            print("Warning: Quality decreased!")
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        baseline_path: Optional[str] = None,
        test_path: Optional[str] = None,
    ):
        base_path = Path(__file__).parent.parent.parent
        
        self.model_path = model_path or str(base_path / "models" / "enigma")
        self.baseline_path = baseline_path or str(base_path / "data" / "test_baseline.json")
        self.test_path = test_path or str(base_path / "data" / "self_improvement_tests.json")
        
        self.scorer = ResponseScorer()
        self.engine = None
        self._default_tests: List[TestCase] = []
        
        self._setup_default_tests()
    
    def _setup_default_tests(self):
        """Set up default test cases."""
        self._default_tests = [
            # Basic knowledge
            TestCase(
                question="What is Enigma AI Engine?",
                expected_keywords=["AI", "engine", "module", "modular"],
                category="core",
            ),
            TestCase(
                question="How do I load a module?",
                expected_keywords=["ModuleManager", "load", "module"],
                category="usage",
            ),
            TestCase(
                question="What is the Model Manager?",
                expected_keywords=["model", "manage", "load"],
                category="core",
            ),
            # GUI knowledge
            TestCase(
                question="How do I open the GUI?",
                expected_keywords=["gui", "window", "run"],
                category="gui",
            ),
            TestCase(
                question="What tabs are available?",
                expected_keywords=["tab", "module", "chat"],
                category="gui",
            ),
            # Training knowledge
            TestCase(
                question="How do I train a model?",
                expected_keywords=["train", "data", "model"],
                category="training",
            ),
        ]
    
    def test(
        self,
        test_cases: Optional[List[TestCase]] = None,
        pass_threshold: float = 0.6,
        callback: Optional[Callable[[str, TestResult], None]] = None,
    ) -> TestSuiteResult:
        """
        Run test suite on the AI.
        
        Args:
            test_cases: Custom test cases, or None for defaults
            pass_threshold: Score threshold for passing (0-1)
            callback: Called after each test with (test_name, result)
            
        Returns:
            TestSuiteResult with all metrics
        """
        start_time = time.time()
        test_cases = test_cases or self._load_or_default_tests()
        
        logger.info(f"Running {len(test_cases)} tests")
        
        # Load engine
        if self.engine is None:
            self._load_engine()
        
        # Run tests
        results: List[TestResult] = []
        category_totals: Dict[str, Tuple[float, int]] = {}
        
        for test_case in test_cases:
            try:
                result = self._run_single_test(test_case, pass_threshold)
            except Exception as e:
                result = TestResult(
                    test_case=test_case,
                    response="",
                    error=str(e),
                )
            
            results.append(result)
            
            # Track category scores
            cat = test_case.category
            if cat not in category_totals:
                category_totals[cat] = (0.0, 0)
            current_total, current_count = category_totals[cat]
            category_totals[cat] = (
                current_total + result.score * test_case.weight,
                current_count + 1,
            )
            
            if callback:
                callback(test_case.question[:50], result)
        
        # Calculate overall scores
        total_weight = sum(tc.weight for tc in test_cases)
        weighted_score = sum(r.score * r.test_case.weight for r in results) / total_weight if total_weight > 0 else 0
        
        category_scores = {
            cat: total / count if count > 0 else 0
            for cat, (total, count) in category_totals.items()
        }
        
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        
        # Check for regression
        baseline = self._load_baseline()
        regression = False
        if baseline is not None and weighted_score < baseline - 0.1:
            regression = True
            logger.warning(f"Regression detected! Score: {weighted_score:.2f}, Baseline: {baseline:.2f}")
        
        suite_result = TestSuiteResult(
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=failed,
            quality_score=weighted_score,
            category_scores=category_scores,
            results=results,
            baseline_score=baseline,
            regression_detected=regression,
            test_time=time.time() - start_time,
        )
        
        # Save new baseline if improved
        if baseline is None or weighted_score > baseline:
            self._save_baseline(weighted_score)
        
        # Log summary
        logger.info(f"Tests complete: {passed}/{len(results)} passed, "
                   f"quality={weighted_score:.2f}")
        
        return suite_result
    
    def _load_engine(self):
        """Load AI engine for testing."""
        try:
            from enigma_engine.core.inference import EnigmaEngine
            self.engine = EnigmaEngine(self.model_path)
            logger.info("Loaded Enigma engine for testing")
        except Exception as e:
            logger.warning(f"Failed to load engine: {e}")
            # Create mock engine for testing
            self.engine = self._create_mock_engine()
    
    def _create_mock_engine(self):
        """Create mock engine for testing without model."""
        class MockEngine:
            def generate(self, prompt, max_length=100):
                # Return generic response
                return f"This is about {prompt.split()[-1] if prompt.split() else 'AI'}."
        
        return MockEngine()
    
    def _run_single_test(self, test_case: TestCase, threshold: float) -> TestResult:
        """Run a single test case."""
        if self.engine is None:
            raise RuntimeError("Engine not loaded. Call _load_engine() first.")
        
        # Generate response
        response = self.engine.generate(test_case.question, max_length=500)
        
        # Score response
        score, metrics = self.scorer.score(response, test_case)
        
        return TestResult(
            test_case=test_case,
            response=response,
            score=score,
            passed=score >= threshold,
            metrics=metrics,
        )
    
    def _load_or_default_tests(self) -> List[TestCase]:
        """Load test cases from file or use defaults."""
        try:
            if Path(self.test_path).exists():
                with open(self.test_path, 'r') as f:
                    data = json.load(f)
                
                tests = []
                for item in data.get("tests", []):
                    tests.append(TestCase(
                        question=item["question"],
                        expected_keywords=item.get("expected_keywords", []),
                        expected_pattern=item.get("expected_pattern", ""),
                        category=item.get("category", "general"),
                        weight=item.get("weight", 1.0),
                    ))
                
                if tests:
                    return tests
        except Exception as e:
            logger.warning(f"Failed to load tests: {e}")
        
        return self._default_tests
    
    def _load_baseline(self) -> Optional[float]:
        """Load baseline score."""
        try:
            if Path(self.baseline_path).exists():
                with open(self.baseline_path, 'r') as f:
                    data = json.load(f)
                return data.get("score")
        except Exception as e:
            logger.warning(f"Failed to load baseline: {e}")
        return None
    
    def _save_baseline(self, score: float):
        """Save new baseline score."""
        try:
            Path(self.baseline_path).parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "score": score,
                "timestamp": datetime.now().isoformat(),
            }
            
            with open(self.baseline_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved new baseline: {score:.2f}")
        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")
    
    def generate_tests_from_training_data(self, training_path: str) -> List[TestCase]:
        """Generate test cases from training data."""
        tests = []
        
        try:
            with open(training_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse Q&A pairs
            pattern = r'Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for question, answer in matches[:20]:  # Limit to 20 tests
                question = question.strip()
                answer = answer.strip()
                
                # Extract keywords from answer
                keywords = self._extract_keywords(answer)
                
                tests.append(TestCase(
                    question=question,
                    expected_keywords=keywords,
                    category="auto_generated",
                ))
            
            logger.info(f"Generated {len(tests)} test cases from training data")
            
        except Exception as e:
            logger.error(f"Failed to generate tests: {e}")
        
        return tests
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract key terms from text."""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'this', 'that', 'these', 'those', 'it', 'its', 'to', 'for',
            'of', 'in', 'on', 'at', 'by', 'with', 'from', 'as', 'or',
            'and', 'but', 'not', 'can', 'will', 'would', 'could', 'should',
            'you', 'your', 'how', 'what', 'when', 'where', 'which', 'who',
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
        
        # Filter and count
        word_counts: Dict[str, int] = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, count in sorted_words[:max_keywords]]
        
        return keywords
    
    def save_tests(self, tests: List[TestCase], output_path: Optional[str] = None):
        """Save test cases to file."""
        output_path = output_path or self.test_path
        
        data = {
            "generated": datetime.now().isoformat(),
            "count": len(tests),
            "tests": [tc.to_dict() for tc in tests],
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(tests)} tests to {output_path}")


# CLI interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-Tester")
    parser.add_argument("--generate", help="Generate tests from training data")
    parser.add_argument("--run", action="store_true", help="Run test suite")
    parser.add_argument("--threshold", type=float, default=0.6)
    
    args = parser.parse_args()
    
    tester = SelfTester()
    
    if args.generate:
        tests = tester.generate_tests_from_training_data(args.generate)
        tester.save_tests(tests)
        print(f"Generated {len(tests)} tests")
    
    if args.run or not args.generate:
        def callback(name, result):
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {name}... Score: {result.score:.2f}")
        
        result = tester.test(pass_threshold=args.threshold, callback=callback)
        
        print(f"\n{'='*50}")
        print(f"Test Results:")
        print(f"  Total: {result.total_tests}")
        print(f"  Passed: {result.passed_tests}")
        print(f"  Failed: {result.failed_tests}")
        print(f"  Quality Score: {result.quality_score:.2f}")
        
        if result.regression_detected:
            print(f"  WARNING: Regression detected!")
            print(f"  Baseline: {result.baseline_score:.2f}")


if __name__ == "__main__":
    main()
