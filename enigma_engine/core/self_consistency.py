"""
Self-Consistency Sampling

Implements self-consistency decoding strategy.
Samples multiple responses and selects by consistency/voting.

FILE: enigma_engine/core/self_consistency.py
TYPE: Advanced Decoding
MAIN CLASSES: SelfConsistency, ConsistencyResult, AnswerCluster
PAPER: "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (Wang et al. 2023)
"""

import difflib
import logging
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Methods for aggregating responses."""
    MAJORITY_VOTE = "majority_vote"           # Simple majority voting
    WEIGHTED_VOTE = "weighted_vote"           # Vote weighted by confidence
    SEMANTIC_CLUSTER = "semantic_cluster"     # Cluster by semantic similarity
    ANSWER_EXTRACTION = "answer_extraction"   # Extract and compare final answers


@dataclass
class SampledResponse:
    """A single sampled response."""
    id: int
    text: str
    extracted_answer: Optional[str] = None
    confidence: float = 1.0
    generation_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerCluster:
    """A cluster of similar answers."""
    answer: str
    count: int
    responses: list[SampledResponse]
    confidence: float = 0.0
    
    @property
    def percentage(self) -> float:
        """Get percentage of total responses."""
        if not self.responses:
            return 0.0
        return len(self.responses) / max(1, sum(1 for _ in self.responses))


@dataclass
class ConsistencyResult:
    """Result of self-consistency sampling."""
    final_answer: str
    confidence: float
    num_samples: int
    clusters: list[AnswerCluster]
    all_responses: list[SampledResponse]
    aggregation_method: AggregationMethod
    generation_time: float
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "num_samples": self.num_samples,
            "num_clusters": len(self.clusters),
            "method": self.aggregation_method.value,
            "generation_time": self.generation_time,
            "cluster_sizes": [c.count for c in self.clusters]
        }


class AnswerExtractor:
    """Extracts answers from reasoning chains."""
    
    # Common answer markers
    ANSWER_PATTERNS = [
        r"(?:the answer is|answer:?)\s*[:\-]?\s*(.+?)(?:\.|$)",
        r"(?:therefore|thus|so|hence),?\s*(.+?)(?:\.|$)",
        r"(?:in conclusion|finally),?\s*(.+?)(?:\.|$)",
        r"(?:=|equals?)\s*(.+?)(?:\.|$)",
        r"(?:result:?|solution:?)\s*(.+?)(?:\.|$)",
    ]
    
    def __init__(self, custom_patterns: list[str] = None):
        """
        Initialize extractor.
        
        Args:
            custom_patterns: Additional regex patterns for extraction
        """
        self._patterns = self.ANSWER_PATTERNS.copy()
        if custom_patterns:
            self._patterns.extend(custom_patterns)
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self._patterns]
        
    def extract(self, text: str) -> Optional[str]:
        """
        Extract answer from text.
        
        Args:
            text: Response text with reasoning chain
            
        Returns:
            Extracted answer or None
        """
        # Try each pattern
        for pattern in self._compiled:
            match = pattern.search(text)
            if match:
                answer = match.group(1).strip()
                # Clean up
                answer = re.sub(r'\s+', ' ', answer)
                return answer
                
        # Fallback: last sentence
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            last = sentences[-1].strip() or (sentences[-2].strip() if len(sentences) > 1 else "")
            return last if last else None
            
        return None


class AnswerComparator:
    """Compares answers for similarity."""
    
    def __init__(self, 
                 similarity_threshold: float = 0.8,
                 normalize: bool = True):
        """
        Initialize comparator.
        
        Args:
            similarity_threshold: Threshold for considering answers similar
            normalize: Whether to normalize answers before comparison
        """
        self._threshold = similarity_threshold
        self._normalize = normalize
        
    def normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        if not answer:
            return ""
        # Lowercase
        normalized = answer.lower()
        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        return normalized
    
    def similarity(self, answer1: str, answer2: str) -> float:
        """Calculate similarity between two answers."""
        if self._normalize:
            answer1 = self.normalize_answer(answer1)
            answer2 = self.normalize_answer(answer2)
            
        if not answer1 or not answer2:
            return 0.0
            
        if answer1 == answer2:
            return 1.0
            
        # Sequence similarity
        return difflib.SequenceMatcher(None, answer1, answer2).ratio()
    
    def are_similar(self, answer1: str, answer2: str) -> bool:
        """Check if two answers are similar."""
        return self.similarity(answer1, answer2) >= self._threshold


class SelfConsistency:
    """Self-consistency sampling for improved accuracy."""
    
    def __init__(self,
                 generator: Callable[[str, float], str] = None,
                 num_samples: int = 5,
                 temperature: float = 0.7,
                 method: AggregationMethod = AggregationMethod.MAJORITY_VOTE,
                 similarity_threshold: float = 0.8):
        """
        Initialize self-consistency.
        
        Args:
            generator: Function to generate responses. Takes (prompt, temperature) and returns text.
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            method: Aggregation method
            similarity_threshold: Threshold for answer similarity
        """
        self._generator = generator or self._default_generator
        self._num_samples = num_samples
        self._temperature = temperature
        self._method = method
        self._extractor = AnswerExtractor()
        self._comparator = AnswerComparator(similarity_threshold)
        
    def _default_generator(self, prompt: str, temperature: float) -> str:
        """Default generator (placeholder)."""
        return f"Sample response at temperature {temperature}: The answer would be..."
    
    def sample(self, prompt: str) -> ConsistencyResult:
        """
        Generate samples and aggregate via self-consistency.
        
        Args:
            prompt: Input prompt
            
        Returns:
            ConsistencyResult with final answer and metadata
        """
        start_time = time.time()
        responses: list[SampledResponse] = []
        
        # Generate samples
        for i in range(self._num_samples):
            try:
                sample_start = time.time()
                text = self._generator(prompt, self._temperature)
                sample_time = time.time() - sample_start
                
                # Extract answer
                extracted = self._extractor.extract(text)
                
                response = SampledResponse(
                    id=i,
                    text=text,
                    extracted_answer=extracted,
                    generation_time=sample_time
                )
                responses.append(response)
                
            except Exception as e:
                logger.error(f"Sample {i} generation error: {e}")
        
        if not responses:
            return ConsistencyResult(
                final_answer="",
                confidence=0.0,
                num_samples=0,
                clusters=[],
                all_responses=[],
                aggregation_method=self._method,
                generation_time=time.time() - start_time
            )
        
        # Aggregate based on method
        if self._method == AggregationMethod.MAJORITY_VOTE:
            final, confidence, clusters = self._majority_vote(responses)
        elif self._method == AggregationMethod.WEIGHTED_VOTE:
            final, confidence, clusters = self._weighted_vote(responses)
        elif self._method == AggregationMethod.SEMANTIC_CLUSTER:
            final, confidence, clusters = self._semantic_cluster(responses)
        elif self._method == AggregationMethod.ANSWER_EXTRACTION:
            final, confidence, clusters = self._answer_extraction(responses)
        else:
            final, confidence, clusters = self._majority_vote(responses)
        
        return ConsistencyResult(
            final_answer=final,
            confidence=confidence,
            num_samples=len(responses),
            clusters=clusters,
            all_responses=responses,
            aggregation_method=self._method,
            generation_time=time.time() - start_time
        )
    
    def _majority_vote(self, responses: list[SampledResponse]) -> tuple[str, float, list[AnswerCluster]]:
        """Simple majority voting on extracted answers."""
        # Count extracted answers
        answer_counts: dict[str, list[SampledResponse]] = defaultdict(list)
        
        for resp in responses:
            answer = resp.extracted_answer or ""
            normalized = self._comparator.normalize_answer(answer)
            if normalized:
                answer_counts[normalized].append(resp)
        
        if not answer_counts:
            # Fallback to full response comparison
            for resp in responses:
                normalized = self._comparator.normalize_answer(resp.text[:100])
                answer_counts[normalized].append(resp)
        
        # Create clusters
        clusters = []
        for answer, resps in answer_counts.items():
            cluster = AnswerCluster(
                answer=resps[0].extracted_answer or answer,
                count=len(resps),
                responses=resps,
                confidence=len(resps) / len(responses)
            )
            clusters.append(cluster)
        
        # Sort by count
        clusters.sort(key=lambda c: c.count, reverse=True)
        
        if clusters:
            final = clusters[0].answer
            confidence = clusters[0].count / len(responses)
        else:
            final = responses[0].text if responses else ""
            confidence = 1.0 / len(responses) if responses else 0.0
        
        return final, confidence, clusters
    
    def _weighted_vote(self, responses: list[SampledResponse]) -> tuple[str, float, list[AnswerCluster]]:
        """Voting weighted by response confidence."""
        # For now, use base confidence * length heuristic
        for resp in responses:
            length_factor = min(len(resp.text) / 200, 1.0)
            resp.confidence = 0.5 + 0.5 * length_factor
        
        # Weighted counting
        answer_weights: dict[str, tuple[float, list[SampledResponse]]] = defaultdict(lambda: (0.0, []))
        
        for resp in responses:
            answer = resp.extracted_answer or ""
            normalized = self._comparator.normalize_answer(answer)
            if normalized:
                weight, resps = answer_weights[normalized]
                answer_weights[normalized] = (weight + resp.confidence, resps + [resp])
        
        # Create clusters
        clusters = []
        total_weight = sum(weight for weight, _ in answer_weights.values())
        
        for answer, (weight, resps) in answer_weights.items():
            cluster = AnswerCluster(
                answer=resps[0].extracted_answer or answer,
                count=len(resps),
                responses=resps,
                confidence=weight / total_weight if total_weight > 0 else 0.0
            )
            clusters.append(cluster)
        
        clusters.sort(key=lambda c: c.confidence, reverse=True)
        
        if clusters:
            final = clusters[0].answer
            confidence = clusters[0].confidence
        else:
            final = responses[0].text if responses else ""
            confidence = 1.0 / len(responses) if responses else 0.0
        
        return final, confidence, clusters
    
    def _semantic_cluster(self, responses: list[SampledResponse]) -> tuple[str, float, list[AnswerCluster]]:
        """Cluster answers by semantic similarity."""
        # Group similar answers
        clusters: list[AnswerCluster] = []
        assigned = set()
        
        for i, resp in enumerate(responses):
            if i in assigned:
                continue
            
            answer = resp.extracted_answer or resp.text[:100]
            similar_responses = [resp]
            assigned.add(i)
            
            # Find similar answers
            for j, other in enumerate(responses):
                if j in assigned:
                    continue
                other_answer = other.extracted_answer or other.text[:100]
                if self._comparator.are_similar(answer, other_answer):
                    similar_responses.append(other)
                    assigned.add(j)
            
            cluster = AnswerCluster(
                answer=answer,
                count=len(similar_responses),
                responses=similar_responses,
                confidence=len(similar_responses) / len(responses)
            )
            clusters.append(cluster)
        
        clusters.sort(key=lambda c: c.count, reverse=True)
        
        if clusters:
            final = clusters[0].answer
            confidence = clusters[0].confidence
        else:
            final = responses[0].text if responses else ""
            confidence = 1.0 / len(responses) if responses else 0.0
        
        return final, confidence, clusters
    
    def _answer_extraction(self, responses: list[SampledResponse]) -> tuple[str, float, list[AnswerCluster]]:
        """Focus on extracted answers only."""
        # Filter to responses with extracted answers
        with_answers = [r for r in responses if r.extracted_answer]
        
        if not with_answers:
            return self._majority_vote(responses)
        
        # Count extracted answers
        answer_counts = Counter(r.extracted_answer for r in with_answers)
        
        # Create clusters
        clusters = []
        answer_to_responses = defaultdict(list)
        for r in with_answers:
            answer_to_responses[r.extracted_answer].append(r)
        
        for answer, count in answer_counts.items():
            cluster = AnswerCluster(
                answer=answer,
                count=count,
                responses=answer_to_responses[answer],
                confidence=count / len(with_answers)
            )
            clusters.append(cluster)
        
        clusters.sort(key=lambda c: c.count, reverse=True)
        
        final = clusters[0].answer if clusters else ""
        confidence = clusters[0].confidence if clusters else 0.0
        
        return final, confidence, clusters


# Integration with Enigma AI Engine
class ForgeSelfConsistency:
    """Self-consistency integration with Enigma AI Engine inference."""
    
    def __init__(self, engine=None, num_samples: int = 5):
        """
        Initialize integration.
        
        Args:
            engine: EnigmaEngine instance
            num_samples: Number of samples per query
        """
        self._engine = engine
        self._num_samples = num_samples
        self._sc: Optional[SelfConsistency] = None
        
    def set_engine(self, engine):
        """Set the inference engine."""
        self._engine = engine
        self._sc = None
        
    def _create_generator(self):
        """Create generator using Enigma AI Engine."""
        def generator(prompt: str, temperature: float) -> str:
            if not self._engine:
                return f"[No engine available] Response to: {prompt[:50]}..."
            
            try:
                return self._engine.generate(
                    prompt,
                    temperature=temperature,
                    max_tokens=500
                )
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return f"Error: {e}"
        
        return generator
    
    def query(self, 
              prompt: str,
              method: AggregationMethod = AggregationMethod.MAJORITY_VOTE) -> ConsistencyResult:
        """
        Query with self-consistency.
        
        Args:
            prompt: Input prompt
            method: Aggregation method
            
        Returns:
            ConsistencyResult
        """
        if self._sc is None or self._sc._method != method:
            self._sc = SelfConsistency(
                generator=self._create_generator(),
                num_samples=self._num_samples,
                method=method
            )
        
        return self._sc.sample(prompt)


# Singleton
_forge_sc: Optional[ForgeSelfConsistency] = None


def get_self_consistency(engine=None, num_samples: int = 5) -> ForgeSelfConsistency:
    """Get the self-consistency singleton."""
    global _forge_sc
    if _forge_sc is None:
        _forge_sc = ForgeSelfConsistency(engine, num_samples)
    elif engine:
        _forge_sc.set_engine(engine)
    return _forge_sc


__all__ = [
    'SelfConsistency',
    'ConsistencyResult',
    'AnswerCluster',
    'SampledResponse',
    'AggregationMethod',
    'AnswerExtractor',
    'AnswerComparator',
    'ForgeSelfConsistency',
    'get_self_consistency'
]
