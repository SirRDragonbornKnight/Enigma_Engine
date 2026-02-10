"""
Reasoning Monitor

Monitor and analyze chain-of-thought reasoning in LLM outputs.
Track reasoning steps, detect errors, and measure reasoning quality.

FILE: enigma_engine/core/reasoning_monitor.py
TYPE: Core/Analysis
MAIN CLASSES: ReasoningMonitor, StepExtractor, ReasoningEvaluator
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StepType(Enum):
    """Types of reasoning steps."""
    SETUP = auto()        # Problem setup/understanding
    ANALYSIS = auto()     # Breaking down the problem
    CALCULATION = auto()  # Mathematical computation
    INFERENCE = auto()    # Logical deduction
    EXAMPLE = auto()      # Using examples
    VERIFICATION = auto() # Checking work
    CONCLUSION = auto()   # Final answer
    QUESTION = auto()     # Self-questioning
    CORRECTION = auto()   # Self-correction
    UNKNOWN = auto()


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    index: int
    text: str
    step_type: StepType
    confidence: float = 0.0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "type": self.step_type.name,
            "confidence": self.confidence,
            "errors": self.errors,
            "metadata": self.metadata
        }


@dataclass
class ReasoningChain:
    """Complete chain of reasoning."""
    steps: list[ReasoningStep]
    input_text: str
    output_text: str
    final_answer: str = ""
    quality_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "input": self.input_text,
            "output": self.output_text,
            "final_answer": self.final_answer,
            "quality_score": self.quality_score,
            "timestamp": self.timestamp.isoformat(),
            "num_steps": len(self.steps)
        }


class StepExtractor:
    """
    Extract reasoning steps from LLM output.
    """
    
    # Common step markers
    STEP_PATTERNS = [
        # Numbered steps
        r'^(\d+)\.\s*(.+)$',
        r'^Step\s+(\d+)[:.\s]\s*(.+)$',
        # Bullet points
        r'^[-*]\s*(.+)$',
        # CoT markers
        r'^(?:First|Second|Third|Then|Next|Finally|Therefore|Thus|Hence)[,:]?\s*(.+)$',
        # Let's think patterns
        r'^Let\'?s?\s+(?:think|consider|analyze|examine)[^:]*[:.]?\s*(.+)?$',
    ]
    
    # Step type indicators
    TYPE_INDICATORS = {
        StepType.SETUP: [
            r'given|problem|question|understand|interpret|define',
        ],
        StepType.ANALYSIS: [
            r'break down|analyze|consider|identify|examine|look at',
        ],
        StepType.CALCULATION: [
            r'calculate|compute|=|plus|minus|times|divide|\d+\s*[+\-*/]\s*\d+',
        ],
        StepType.INFERENCE: [
            r'therefore|thus|hence|conclude|implies|means that|so\s+',
        ],
        StepType.EXAMPLE: [
            r'for example|e\.g\.|instance|such as|like\s+',
        ],
        StepType.VERIFICATION: [
            r'check|verify|confirm|validate|make sure|double.?check',
        ],
        StepType.CONCLUSION: [
            r'final|answer|result|solution|in conclusion|ultimately',
        ],
        StepType.QUESTION: [
            r'what if|how|why|should we|\?$',
        ],
        StepType.CORRECTION: [
            r'wait|actually|mistake|wrong|correct|oops|sorry',
        ],
    }
    
    def __init__(
        self,
        custom_patterns: list[str] = None
    ):
        self.patterns = self.STEP_PATTERNS.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)
        
        # Compile type indicators
        self._type_patterns = {
            stype: [re.compile(p, re.IGNORECASE) for p in patterns]
            for stype, patterns in self.TYPE_INDICATORS.items()
        }
    
    def extract_steps(
        self,
        text: str
    ) -> list[ReasoningStep]:
        """
        Extract reasoning steps from text.
        
        Args:
            text: LLM output text
        
        Returns:
            List of ReasoningStep objects
        """
        steps = []
        
        # Try structured extraction first
        lines = text.strip().split('\n')
        
        current_step_text = []
        step_index = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this starts a new step
            is_new_step = False
            for pattern in self.patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous step
                    if current_step_text:
                        step_text = ' '.join(current_step_text)
                        steps.append(self._create_step(step_index, step_text))
                        step_index += 1
                    
                    current_step_text = [line]
                    is_new_step = True
                    break
            
            if not is_new_step:
                current_step_text.append(line)
        
        # Save last step
        if current_step_text:
            step_text = ' '.join(current_step_text)
            steps.append(self._create_step(step_index, step_text))
        
        # If no steps found, treat whole text as one step
        if not steps:
            steps.append(self._create_step(0, text.strip()))
        
        return steps
    
    def _create_step(
        self,
        index: int,
        text: str
    ) -> ReasoningStep:
        """Create a reasoning step with type classification."""
        step_type = self._classify_step(text)
        
        return ReasoningStep(
            index=index,
            text=text,
            step_type=step_type
        )
    
    def _classify_step(self, text: str) -> StepType:
        """Classify the type of reasoning step."""
        scores = {}
        
        for stype, patterns in self._type_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.search(text):
                    score += 1
            scores[stype] = score
        
        if not scores or max(scores.values()) == 0:
            return StepType.UNKNOWN
        
        return max(scores.items(), key=lambda x: x[1])[0]


class ReasoningEvaluator:
    """
    Evaluate quality of reasoning chains.
    """
    
    def __init__(self):
        self.validators: list[Callable[[ReasoningChain], tuple[float, list[str]]]] = []
        self._add_default_validators()
    
    def _add_default_validators(self):
        """Add default validation checks."""
        self.validators.append(self._check_structure)
        self.validators.append(self._check_progression)
        self.validators.append(self._check_conclusion)
        self.validators.append(self._check_consistency)
    
    def add_validator(
        self,
        validator: Callable[[ReasoningChain], tuple[float, list[str]]]
    ):
        """Add custom validator function."""
        self.validators.append(validator)
    
    def evaluate(
        self,
        chain: ReasoningChain
    ) -> tuple[float, dict[str, Any]]:
        """
        Evaluate a reasoning chain.
        
        Args:
            chain: ReasoningChain to evaluate
        
        Returns:
            Tuple of (score, details)
        """
        all_errors = []
        scores = []
        
        for validator in self.validators:
            score, errors = validator(chain)
            scores.append(score)
            all_errors.extend(errors)
        
        final_score = sum(scores) / len(scores) if scores else 0.0
        
        return final_score, {
            "score": final_score,
            "errors": all_errors,
            "num_steps": len(chain.steps),
            "step_types": [s.step_type.name for s in chain.steps]
        }
    
    def _check_structure(
        self,
        chain: ReasoningChain
    ) -> tuple[float, list[str]]:
        """Check if reasoning has proper structure."""
        errors = []
        score = 1.0
        
        # Should have multiple steps
        if len(chain.steps) < 2:
            score -= 0.3
            errors.append("Reasoning lacks multiple distinct steps")
        
        # Steps should not be too short
        short_steps = sum(1 for s in chain.steps if len(s.text) < 10)
        if short_steps > len(chain.steps) / 2:
            score -= 0.2
            errors.append("Many steps are too brief")
        
        return max(0.0, score), errors
    
    def _check_progression(
        self,
        chain: ReasoningChain
    ) -> tuple[float, list[str]]:
        """Check if reasoning progresses logically."""
        errors = []
        score = 1.0
        
        if not chain.steps:
            return 0.0, ["No reasoning steps found"]
        
        # Check for logical flow
        step_types = [s.step_type for s in chain.steps]
        
        # Should ideally start with setup/analysis
        if step_types[0] not in [StepType.SETUP, StepType.ANALYSIS, StepType.QUESTION]:
            score -= 0.1
        
        # Should end with conclusion/verification
        if step_types[-1] not in [StepType.CONCLUSION, StepType.VERIFICATION]:
            score -= 0.2
            errors.append("Reasoning doesn't end with clear conclusion")
        
        # Check for sudden jumps
        for i in range(len(step_types) - 1):
            if step_types[i] == StepType.SETUP and step_types[i+1] == StepType.CONCLUSION:
                score -= 0.3
                errors.append("Reasoning jumps from setup to conclusion")
        
        return max(0.0, score), errors
    
    def _check_conclusion(
        self,
        chain: ReasoningChain
    ) -> tuple[float, list[str]]:
        """Check if there's a clear conclusion."""
        errors = []
        score = 1.0
        
        # Check for final answer
        conclusion_markers = ['answer', 'result', 'therefore', 'conclusion', 'final']
        
        has_conclusion = any(
            any(marker in s.text.lower() for marker in conclusion_markers)
            for s in chain.steps
        )
        
        if not has_conclusion and not chain.final_answer:
            score -= 0.4
            errors.append("No clear final answer or conclusion")
        
        return max(0.0, score), errors
    
    def _check_consistency(
        self,
        chain: ReasoningChain
    ) -> tuple[float, list[str]]:
        """Check for internal consistency."""
        errors = []
        score = 1.0
        
        # Check for self-corrections (not necessarily bad, but note them)
        corrections = sum(1 for s in chain.steps if s.step_type == StepType.CORRECTION)
        if corrections > 2:
            score -= 0.1
            errors.append(f"Multiple self-corrections ({corrections})")
        
        # Check for repetition
        texts = [s.text.lower() for s in chain.steps]
        for i, text in enumerate(texts):
            for j, other in enumerate(texts[i+1:], i+1):
                # Simple similarity check
                if len(text) > 20 and text in other:
                    score -= 0.1
                    errors.append(f"Repetition between steps {i+1} and {j+1}")
        
        return max(0.0, score), errors


class ReasoningMonitor:
    """
    Monitor reasoning quality during inference.
    """
    
    def __init__(
        self,
        extractor: StepExtractor = None,
        evaluator: ReasoningEvaluator = None
    ):
        self.extractor = extractor or StepExtractor()
        self.evaluator = evaluator or ReasoningEvaluator()
        
        self._history: list[ReasoningChain] = []
        self._max_history: int = 100
        self._callbacks: list[Callable[[ReasoningChain], None]] = []
    
    def add_callback(
        self,
        callback: Callable[[ReasoningChain], None]
    ):
        """Add callback to be called after each analysis."""
        self._callbacks.append(callback)
    
    def analyze(
        self,
        input_text: str,
        output_text: str,
        extract_answer: bool = True
    ) -> ReasoningChain:
        """
        Analyze reasoning in model output.
        
        Args:
            input_text: Original prompt/question
            output_text: Model's response
            extract_answer: Try to extract final answer
        
        Returns:
            ReasoningChain with analysis
        """
        # Extract steps
        steps = self.extractor.extract_steps(output_text)
        
        # Extract final answer
        final_answer = ""
        if extract_answer:
            final_answer = self._extract_answer(output_text)
        
        # Create chain
        chain = ReasoningChain(
            steps=steps,
            input_text=input_text,
            output_text=output_text,
            final_answer=final_answer
        )
        
        # Evaluate
        score, details = self.evaluator.evaluate(chain)
        chain.quality_score = score
        
        # Update step errors from evaluation
        for step in steps:
            step.confidence = score
        
        # Store and callback
        self._history.append(chain)
        
        # Trim history if too long
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        for callback in self._callbacks:
            try:
                callback(chain)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        return chain
    
    def _extract_answer(self, text: str) -> str:
        """Extract final answer from text."""
        # Common answer patterns
        patterns = [
            r'(?:final answer|answer)(?:\s+is)?[:\s]+([^\n]+)',
            r'(?:therefore|thus|hence)[,:\s]+([^\n]+)',
            r'(?:the result is|result)[:\s]+([^\n]+)',
            r'=\s*(\d+(?:\.\d+)?)',  # Math result
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Return last sentence as fallback
        sentences = re.split(r'[.!?]', text)
        if sentences:
            return sentences[-1].strip() or (sentences[-2].strip() if len(sentences) > 1 else "")
        
        return ""
    
    def get_statistics(self) -> dict[str, Any]:
        """Get statistics across all analyzed chains."""
        if not self._history:
            return {"count": 0}
        
        scores = [c.quality_score for c in self._history]
        step_counts = [len(c.steps) for c in self._history]
        
        type_counts = {}
        for chain in self._history:
            for step in chain.steps:
                type_counts[step.step_type.name] = type_counts.get(step.step_type.name, 0) + 1
        
        return {
            "count": len(self._history),
            "avg_quality": sum(scores) / len(scores),
            "min_quality": min(scores),
            "max_quality": max(scores),
            "avg_steps": sum(step_counts) / len(step_counts),
            "step_type_distribution": type_counts
        }
    
    def export_history(self, filepath: str):
        """Export analysis history to JSON."""
        data = [c.to_dict() for c in self._history]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported {len(data)} chains to {filepath}")
    
    def clear_history(self):
        """Clear analysis history."""
        self._history.clear()


def monitor_reasoning(
    input_text: str,
    output_text: str
) -> dict[str, Any]:
    """
    Quick utility to analyze reasoning.
    
    Args:
        input_text: Question/prompt
        output_text: Model response
    
    Returns:
        Analysis results
    """
    monitor = ReasoningMonitor()
    chain = monitor.analyze(input_text, output_text)
    
    return {
        "quality_score": chain.quality_score,
        "num_steps": len(chain.steps),
        "final_answer": chain.final_answer,
        "steps": [s.to_dict() for s in chain.steps]
    }
