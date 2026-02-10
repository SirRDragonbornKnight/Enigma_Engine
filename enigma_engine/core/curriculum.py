"""
Curriculum Learning for Enigma AI Engine

Train models with progressively harder examples.

Features:
- Difficulty estimation
- Curriculum scheduling
- Anti-curriculum support
- Self-paced learning
- Multi-criteria sorting

Usage:
    from enigma_engine.core.curriculum import CurriculumScheduler, get_scheduler
    
    scheduler = get_scheduler()
    
    # Score difficulty
    scheduler.score_dataset(dataset)
    
    # Get curriculum batches
    for batch in scheduler.get_batches(dataset, epoch=1):
        train(batch)
"""

import json
import logging
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class DifficultyMetric(Enum):
    """Metrics for estimating difficulty."""
    LENGTH = "length"  # Text length
    VOCABULARY = "vocabulary"  # Vocabulary complexity
    PERPLEXITY = "perplexity"  # Model perplexity
    LOSS = "loss"  # Training loss
    SYNTAX = "syntax"  # Syntactic complexity
    SEMANTIC = "semantic"  # Semantic complexity


class CurriculumStrategy(Enum):
    """Curriculum strategies."""
    EASY_TO_HARD = "easy_to_hard"  # Standard curriculum
    HARD_TO_EASY = "hard_to_easy"  # Anti-curriculum
    SELF_PACED = "self_paced"  # Adaptive based on model performance
    RANDOM = "random"  # No curriculum (baseline)
    BALANCED = "balanced"  # Mix of difficulties
    PROBABILISTIC = "probabilistic"  # Sample based on competence


@dataclass
class DifficultyScore:
    """Difficulty score for a sample."""
    sample_id: int
    overall_score: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "overall_score": self.overall_score,
            "components": self.component_scores
        }


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    strategy: CurriculumStrategy = CurriculumStrategy.EASY_TO_HARD
    
    # Metrics to use (with weights)
    metrics: Dict[DifficultyMetric, float] = field(default_factory=lambda: {
        DifficultyMetric.LENGTH: 0.3,
        DifficultyMetric.VOCABULARY: 0.3,
        DifficultyMetric.SYNTAX: 0.2,
        DifficultyMetric.PERPLEXITY: 0.2
    })
    
    # Scheduling
    warmup_epochs: int = 1  # Epochs before full difficulty
    competence_growth: float = 0.1  # How fast competence grows
    
    # Self-paced
    threshold_increase: float = 0.1  # Difficulty threshold increase per epoch
    min_competence: float = 0.1
    max_competence: float = 1.0
    
    # Balanced
    difficulty_bins: int = 5  # Number of difficulty buckets


class DifficultyEstimator:
    """Estimates difficulty of training samples."""
    
    # Common/simple words
    SIMPLE_WORDS = {
        "the", "a", "is", "are", "was", "were", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may",
        "be", "been", "being", "i", "you", "he", "she", "it", "we", "they",
        "what", "who", "how", "when", "where", "why", "this", "that"
    }
    
    def __init__(self):
        self._model = None
        self._vocab_cache: Dict[str, float] = {}
    
    def estimate_length(self, text: str) -> float:
        """
        Estimate difficulty based on length.
        
        Longer texts are considered harder.
        """
        words = text.split()
        # Normalize to 0-1 (assuming max useful length ~500 words)
        return min(1.0, len(words) / 500)
    
    def estimate_vocabulary(self, text: str) -> float:
        """
        Estimate difficulty based on vocabulary complexity.
        
        More complex words = harder.
        """
        words = text.lower().split()
        
        if not words:
            return 0.0
        
        # Count non-simple words
        complex_count = sum(1 for w in words if w not in self.SIMPLE_WORDS and len(w) > 4)
        
        # Average word length
        avg_word_len = sum(len(w) for w in words) / len(words)
        
        # Unique word ratio (higher = more complex)
        unique_ratio = len(set(words)) / len(words)
        
        # Combine metrics
        complexity = (
            (complex_count / len(words)) * 0.4 +
            (min(1.0, avg_word_len / 10)) * 0.3 +
            unique_ratio * 0.3
        )
        
        return complexity
    
    def estimate_syntax(self, text: str) -> float:
        """
        Estimate syntactic complexity.
        
        More clauses, punctuation = harder.
        """
        # Count sentences
        sentences = text.count('.') + text.count('!') + text.count('?')
        sentences = max(1, sentences)
        
        # Words per sentence
        words = len(text.split())
        words_per_sentence = words / sentences
        
        # Punctuation density
        punctuation = sum(1 for c in text if c in '.,;:!?-()[]{}')
        punct_density = punctuation / max(1, len(text))
        
        # Combine
        complexity = (
            min(1.0, words_per_sentence / 30) * 0.5 +  # Longer sentences harder
            min(1.0, punct_density * 20) * 0.5  # More punctuation harder
        )
        
        return complexity
    
    def estimate_perplexity(
        self,
        text: str,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ) -> float:
        """
        Estimate difficulty using model perplexity.
        
        Higher perplexity = harder for model.
        """
        if model is None:
            return 0.5  # Default if no model
        
        try:
            import torch
            
            # Tokenize
            if tokenizer:
                input_ids = tokenizer.encode(text, return_tensors="pt")
            else:
                return 0.5
            
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                perplexity = torch.exp(loss).item()
            
            # Normalize (assuming perplexity typically 1-1000)
            normalized = min(1.0, math.log(perplexity) / 10)
            return normalized
            
        except Exception as e:
            logger.debug(f"Perplexity estimation failed: {e}")
            return 0.5
    
    def estimate(
        self,
        text: str,
        metrics: Dict[DifficultyMetric, float],
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ) -> DifficultyScore:
        """
        Estimate overall difficulty.
        
        Args:
            text: Input text
            metrics: Metrics to use with weights
            model: Optional model for perplexity
            tokenizer: Optional tokenizer
            
        Returns:
            Difficulty score
        """
        component_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, weight in metrics.items():
            if metric == DifficultyMetric.LENGTH:
                score = self.estimate_length(text)
            elif metric == DifficultyMetric.VOCABULARY:
                score = self.estimate_vocabulary(text)
            elif metric == DifficultyMetric.SYNTAX:
                score = self.estimate_syntax(text)
            elif metric == DifficultyMetric.PERPLEXITY:
                score = self.estimate_perplexity(text, model, tokenizer)
            else:
                continue
            
            component_scores[metric.value] = score
            weighted_sum += score * weight
            total_weight += weight
        
        overall = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return DifficultyScore(
            sample_id=hash(text) % 1000000,
            overall_score=overall,
            component_scores=component_scores
        )


class CurriculumScheduler:
    """
    Schedules training data according to curriculum.
    """
    
    def __init__(self, config: Optional[CurriculumConfig] = None):
        """
        Initialize scheduler.
        
        Args:
            config: Curriculum configuration
        """
        self._config = config or CurriculumConfig()
        self._estimator = DifficultyEstimator()
        
        # Cached scores
        self._scores: Dict[int, DifficultyScore] = {}
        
        # Training state
        self._current_epoch = 0
        self._current_competence = self._config.min_competence
    
    def score_sample(
        self,
        text: str,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ) -> DifficultyScore:
        """Score a single sample."""
        return self._estimator.estimate(
            text,
            self._config.metrics,
            model,
            tokenizer
        )
    
    def score_dataset(
        self,
        dataset: List[Dict[str, Any]],
        text_key: str = "text",
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ) -> List[DifficultyScore]:
        """
        Score entire dataset.
        
        Args:
            dataset: List of data samples
            text_key: Key for text content
            model: Optional model for perplexity
            tokenizer: Optional tokenizer
            
        Returns:
            List of difficulty scores
        """
        scores = []
        
        for i, sample in enumerate(dataset):
            text = sample.get(text_key, str(sample))
            score = self.score_sample(text, model, tokenizer)
            score.sample_id = i
            scores.append(score)
            self._scores[i] = score
        
        logger.info(f"Scored {len(scores)} samples")
        return scores
    
    def get_competence(self, epoch: int) -> float:
        """
        Get competence level for epoch.
        
        Competence determines what difficulty level the model can handle.
        """
        if epoch < self._config.warmup_epochs:
            return self._config.min_competence
        
        # Linear growth
        progress = min(1.0, (epoch - self._config.warmup_epochs) * self._config.competence_growth)
        
        return self._config.min_competence + progress * (
            self._config.max_competence - self._config.min_competence
        )
    
    def filter_by_competence(
        self,
        dataset: List[Dict[str, Any]],
        competence: float
    ) -> List[int]:
        """
        Filter samples by competence level.
        
        Returns indices of samples within competence.
        """
        if not self._scores:
            # If not scored, include all
            return list(range(len(dataset)))
        
        indices = []
        for i, sample in enumerate(dataset):
            if i in self._scores:
                if self._scores[i].overall_score <= competence:
                    indices.append(i)
        
        return indices
    
    def sort_by_difficulty(
        self,
        dataset: List[Dict[str, Any]],
        reverse: bool = False
    ) -> List[int]:
        """
        Sort dataset indices by difficulty.
        
        Args:
            dataset: Dataset
            reverse: If True, sort hard-to-easy
            
        Returns:
            Sorted indices
        """
        # Score if needed
        if not self._scores:
            for i, sample in enumerate(dataset):
                text = sample.get("text", str(sample))
                score = self.score_sample(text)
                score.sample_id = i
                self._scores[i] = score
        
        indices = list(range(len(dataset)))
        indices.sort(
            key=lambda i: self._scores.get(i, DifficultyScore(i, 0.5)).overall_score,
            reverse=reverse
        )
        
        return indices
    
    def get_batches(
        self,
        dataset: List[Dict[str, Any]],
        batch_size: int = 32,
        epoch: int = 0
    ) -> Iterator[List[int]]:
        """
        Get training batches according to curriculum.
        
        Args:
            dataset: Training data
            batch_size: Batch size
            epoch: Current epoch
            
        Yields:
            Batch indices
        """
        self._current_epoch = epoch
        competence = self.get_competence(epoch)
        self._current_competence = competence
        
        strategy = self._config.strategy
        
        if strategy == CurriculumStrategy.EASY_TO_HARD:
            indices = self.sort_by_difficulty(dataset, reverse=False)
            # Filter by competence
            indices = [i for i in indices if self._scores.get(i, DifficultyScore(i, 0)).overall_score <= competence]
        
        elif strategy == CurriculumStrategy.HARD_TO_EASY:
            indices = self.sort_by_difficulty(dataset, reverse=True)
        
        elif strategy == CurriculumStrategy.SELF_PACED:
            indices = self.filter_by_competence(dataset, competence)
            random.shuffle(indices)
        
        elif strategy == CurriculumStrategy.BALANCED:
            indices = self._balanced_sampling(dataset)
        
        elif strategy == CurriculumStrategy.PROBABILISTIC:
            indices = self._probabilistic_sampling(dataset, competence)
        
        else:  # RANDOM
            indices = list(range(len(dataset)))
            random.shuffle(indices)
        
        # Yield batches
        for i in range(0, len(indices), batch_size):
            yield indices[i:i + batch_size]
    
    def _balanced_sampling(self, dataset: List[Dict[str, Any]]) -> List[int]:
        """Sample evenly from difficulty bins."""
        n_bins = self._config.difficulty_bins
        bins: List[List[int]] = [[] for _ in range(n_bins)]
        
        for i in range(len(dataset)):
            score = self._scores.get(i, DifficultyScore(i, 0.5)).overall_score
            bin_idx = min(n_bins - 1, int(score * n_bins))
            bins[bin_idx].append(i)
        
        # Shuffle within bins
        for bin_list in bins:
            random.shuffle(bin_list)
        
        # Interleave from bins
        indices = []
        max_len = max(len(b) for b in bins) if bins else 0
        
        for j in range(max_len):
            for bin_list in bins:
                if j < len(bin_list):
                    indices.append(bin_list[j])
        
        return indices
    
    def _probabilistic_sampling(
        self,
        dataset: List[Dict[str, Any]],
        competence: float
    ) -> List[int]:
        """Sample with probability based on difficulty and competence."""
        indices = []
        
        for i in range(len(dataset)):
            score = self._scores.get(i, DifficultyScore(i, 0.5)).overall_score
            
            # Higher probability for samples near competence level
            distance = abs(score - competence)
            prob = math.exp(-distance * 5)  # Exponential decay
            
            if random.random() < prob:
                indices.append(i)
        
        random.shuffle(indices)
        return indices
    
    def update_competence(self, loss: float, threshold: float = 0.5):
        """
        Update competence based on training loss (for self-paced).
        
        Args:
            loss: Recent training loss
            threshold: Loss threshold for increasing competence
        """
        if loss < threshold:
            self._current_competence = min(
                self._config.max_competence,
                self._current_competence + self._config.threshold_increase
            )
            logger.info(f"Competence increased to {self._current_competence:.3f}")
    
    def get_difficulty_distribution(
        self,
        dataset: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Get distribution of difficulty levels."""
        if not self._scores:
            self.score_dataset(dataset)
        
        bins = {
            "very_easy": 0,
            "easy": 0,
            "medium": 0,
            "hard": 0,
            "very_hard": 0
        }
        
        for score in self._scores.values():
            s = score.overall_score
            if s < 0.2:
                bins["very_easy"] += 1
            elif s < 0.4:
                bins["easy"] += 1
            elif s < 0.6:
                bins["medium"] += 1
            elif s < 0.8:
                bins["hard"] += 1
            else:
                bins["very_hard"] += 1
        
        return bins
    
    def export_scores(self, path: str):
        """Export difficulty scores to file."""
        data = {
            str(k): v.to_dict()
            for k, v in self._scores.items()
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# Global instance
_scheduler: Optional[CurriculumScheduler] = None


def get_scheduler() -> CurriculumScheduler:
    """Get or create global curriculum scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = CurriculumScheduler()
    return _scheduler
