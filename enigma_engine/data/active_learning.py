"""
Active Learning

Prioritizes unlabeled data for annotation based on model uncertainty.
Implements various active learning strategies for efficient labeling.

FILE: enigma_engine/data/active_learning.py
TYPE: Data Management
MAIN CLASSES: ActiveLearner, UncertaintySampler, DiversitySampler
"""

import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Active learning sampling strategies."""
    RANDOM = "random"
    UNCERTAINTY = "uncertainty"  # Least confident predictions
    MARGIN = "margin"  # Smallest margin between top 2 predictions
    ENTROPY = "entropy"  # Highest prediction entropy
    DIVERSITY = "diversity"  # Maximize coverage
    HYBRID = "hybrid"  # Combination of uncertainty + diversity


@dataclass
class Sample:
    """An unlabeled sample for consideration."""
    id: str
    data: Any
    features: Optional[list[float]] = None
    prediction: Optional[dict[str, float]] = None  # Class -> probability
    uncertainty_score: float = 0.0
    diversity_score: float = 0.0
    combined_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LabeledSample:
    """A labeled sample."""
    id: str
    data: Any
    label: Any
    confidence: float = 1.0
    labeler: str = "unknown"
    labeled_at: float = field(default_factory=time.time)


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning."""
    strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY
    batch_size: int = 10
    uncertainty_weight: float = 0.7
    diversity_weight: float = 0.3
    min_uncertainty_threshold: float = 0.3
    random_seed: Optional[int] = None


class UncertaintySampler:
    """Samples based on model uncertainty."""
    
    def __init__(self, strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY):
        """
        Initialize uncertainty sampler.
        
        Args:
            strategy: Sampling strategy to use
        """
        self._strategy = strategy
    
    def score(self, sample: Sample) -> float:
        """
        Calculate uncertainty score for a sample.
        
        Args:
            sample: Sample with predictions
            
        Returns:
            Uncertainty score (higher = more uncertain)
        """
        if not sample.prediction:
            return 0.0
        
        probs = list(sample.prediction.values())
        
        if self._strategy == SamplingStrategy.UNCERTAINTY:
            return self._least_confident(probs)
        elif self._strategy == SamplingStrategy.MARGIN:
            return self._margin_score(probs)
        elif self._strategy == SamplingStrategy.ENTROPY:
            return self._entropy_score(probs)
        else:
            return self._least_confident(probs)
    
    def _least_confident(self, probs: list[float]) -> float:
        """1 - max probability (higher = less confident)."""
        if not probs:
            return 0.0
        return 1.0 - max(probs)
    
    def _margin_score(self, probs: list[float]) -> float:
        """1 - margin between top 2 predictions."""
        if len(probs) < 2:
            return 0.0
        
        sorted_probs = sorted(probs, reverse=True)
        margin = sorted_probs[0] - sorted_probs[1]
        return 1.0 - margin
    
    def _entropy_score(self, probs: list[float]) -> float:
        """Normalized entropy of predictions."""
        if not probs:
            return 0.0
        
        # Calculate entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Normalize by max entropy (uniform distribution)
        max_entropy = math.log2(len(probs))
        if max_entropy > 0:
            return entropy / max_entropy
        return 0.0


class DiversitySampler:
    """Samples to maximize diversity/coverage."""
    
    def __init__(self):
        """Initialize diversity sampler."""
        self._selected_features: list[list[float]] = []
    
    def reset(self):
        """Reset selected samples."""
        self._selected_features = []
    
    def score(self, sample: Sample) -> float:
        """
        Calculate diversity score for a sample.
        
        Args:
            sample: Sample with features
            
        Returns:
            Diversity score (higher = more diverse from selected)
        """
        if not sample.features:
            return random.random()  # Random if no features
        
        if not self._selected_features:
            return 1.0  # First sample is always diverse
        
        # Minimum distance to any selected sample
        min_distance = float('inf')
        for selected in self._selected_features:
            distance = self._euclidean_distance(sample.features, selected)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _euclidean_distance(self, a: list[float], b: list[float]) -> float:
        """Calculate Euclidean distance between two feature vectors."""
        if len(a) != len(b):
            return 0.0
        
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    def add_selected(self, sample: Sample):
        """Add a sample to the selected set."""
        if sample.features:
            self._selected_features.append(sample.features)


class ActiveLearner:
    """Main active learning controller."""
    
    def __init__(self, config: Optional[ActiveLearningConfig] = None):
        """
        Initialize active learner.
        
        Args:
            config: Active learning configuration
        """
        self._config = config or ActiveLearningConfig()
        
        self._uncertainty_sampler = UncertaintySampler(self._config.strategy)
        self._diversity_sampler = DiversitySampler()
        
        self._unlabeled_pool: dict[str, Sample] = {}
        self._labeled_samples: dict[str, LabeledSample] = {}
        self._selection_history: list[list[str]] = []
        self._max_selection_history = 100  # Prevent unbounded growth
        
        if self._config.random_seed is not None:
            random.seed(self._config.random_seed)
    
    def add_unlabeled(self, samples: list[Sample]):
        """
        Add unlabeled samples to the pool.
        
        Args:
            samples: List of samples
        """
        for sample in samples:
            self._unlabeled_pool[sample.id] = sample
        
        logger.info(f"Added {len(samples)} samples to unlabeled pool")
    
    def set_predictions(self, predictions: dict[str, dict[str, float]]):
        """
        Set model predictions for samples.
        
        Args:
            predictions: Dict mapping sample_id to class probabilities
        """
        for sample_id, probs in predictions.items():
            if sample_id in self._unlabeled_pool:
                self._unlabeled_pool[sample_id].prediction = probs
    
    def select_batch(self, batch_size: Optional[int] = None) -> list[Sample]:
        """
        Select a batch of samples for labeling.
        
        Args:
            batch_size: Number of samples (uses config default if None)
            
        Returns:
            List of selected samples
        """
        batch_size = batch_size or self._config.batch_size
        
        if not self._unlabeled_pool:
            logger.warning("No unlabeled samples in pool")
            return []
        
        # Score all samples
        scored_samples = []
        for sample in self._unlabeled_pool.values():
            self._score_sample(sample)
            scored_samples.append(sample)
        
        # Select based on strategy
        if self._config.strategy == SamplingStrategy.RANDOM:
            selected = self._random_select(scored_samples, batch_size)
        elif self._config.strategy == SamplingStrategy.DIVERSITY:
            selected = self._diversity_select(scored_samples, batch_size)
        elif self._config.strategy == SamplingStrategy.HYBRID:
            selected = self._hybrid_select(scored_samples, batch_size)
        else:
            selected = self._uncertainty_select(scored_samples, batch_size)
        
        # Record selection
        selected_ids = [s.id for s in selected]
        self._selection_history.append(selected_ids)
        
        # Trim history if too long
        if len(self._selection_history) > self._max_selection_history:
            self._selection_history = self._selection_history[-self._max_selection_history:]
        
        return selected
    
    def _score_sample(self, sample: Sample):
        """Score a sample with uncertainty and diversity."""
        sample.uncertainty_score = self._uncertainty_sampler.score(sample)
        sample.diversity_score = self._diversity_sampler.score(sample)
        
        # Combined score
        sample.combined_score = (
            self._config.uncertainty_weight * sample.uncertainty_score +
            self._config.diversity_weight * sample.diversity_score
        )
    
    def _random_select(self, samples: list[Sample], n: int) -> list[Sample]:
        """Random selection."""
        return random.sample(samples, min(n, len(samples)))
    
    def _uncertainty_select(self, samples: list[Sample], n: int) -> list[Sample]:
        """Select most uncertain samples."""
        sorted_samples = sorted(samples, key=lambda s: s.uncertainty_score, reverse=True)
        return sorted_samples[:n]
    
    def _diversity_select(self, samples: list[Sample], n: int) -> list[Sample]:
        """Select diverse samples using greedy farthest-first."""
        self._diversity_sampler.reset()
        selected = []
        remaining = list(samples)
        
        while len(selected) < n and remaining:
            # Score remaining samples
            for sample in remaining:
                sample.diversity_score = self._diversity_sampler.score(sample)
            
            # Select most diverse
            best = max(remaining, key=lambda s: s.diversity_score)
            selected.append(best)
            remaining.remove(best)
            self._diversity_sampler.add_selected(best)
        
        return selected
    
    def _hybrid_select(self, samples: list[Sample], n: int) -> list[Sample]:
        """Hybrid selection combining uncertainty and diversity."""
        self._diversity_sampler.reset()
        selected = []
        remaining = list(samples)
        
        while len(selected) < n and remaining:
            # Re-score for diversity
            for sample in remaining:
                sample.diversity_score = self._diversity_sampler.score(sample)
                sample.combined_score = (
                    self._config.uncertainty_weight * sample.uncertainty_score +
                    self._config.diversity_weight * sample.diversity_score
                )
            
            # Select best combined
            best = max(remaining, key=lambda s: s.combined_score)
            selected.append(best)
            remaining.remove(best)
            self._diversity_sampler.add_selected(best)
        
        return selected
    
    def label_sample(self, sample_id: str, label: Any, labeler: str = "user"):
        """
        Label a sample and move it from unlabeled to labeled pool.
        
        Args:
            sample_id: Sample ID
            label: The label
            labeler: Who labeled it
        """
        if sample_id not in self._unlabeled_pool:
            logger.warning(f"Sample {sample_id} not in unlabeled pool")
            return
        
        sample = self._unlabeled_pool.pop(sample_id)
        
        labeled = LabeledSample(
            id=sample_id,
            data=sample.data,
            label=label,
            labeler=labeler
        )
        
        self._labeled_samples[sample_id] = labeled
        logger.info(f"Labeled sample {sample_id}")
    
    def get_labeled_samples(self) -> list[LabeledSample]:
        """Get all labeled samples."""
        return list(self._labeled_samples.values())
    
    def get_statistics(self) -> dict[str, Any]:
        """Get active learning statistics."""
        return {
            "unlabeled_count": len(self._unlabeled_pool),
            "labeled_count": len(self._labeled_samples),
            "batches_selected": len(self._selection_history),
            "strategy": self._config.strategy.value,
            "uncertainty_scores": self._get_score_distribution("uncertainty"),
            "diversity_scores": self._get_score_distribution("diversity")
        }
    
    def _get_score_distribution(self, score_type: str) -> dict[str, float]:
        """Get score distribution statistics."""
        if not self._unlabeled_pool:
            return {"min": 0, "max": 0, "mean": 0}
        
        if score_type == "uncertainty":
            scores = [s.uncertainty_score for s in self._unlabeled_pool.values()]
        else:
            scores = [s.diversity_score for s in self._unlabeled_pool.values()]
        
        return {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores)
        }
    
    def export_for_training(self) -> list[dict[str, Any]]:
        """
        Export labeled samples for training.
        
        Returns:
            List of training samples
        """
        return [
            {
                "id": sample.id,
                "data": sample.data,
                "label": sample.label,
                "metadata": {
                    "labeler": sample.labeler,
                    "labeled_at": sample.labeled_at
                }
            }
            for sample in self._labeled_samples.values()
        ]


def create_active_learner(strategy: str = "uncertainty", **kwargs) -> ActiveLearner:
    """
    Create an active learner with specified strategy.
    
    Args:
        strategy: Sampling strategy name
        **kwargs: Additional config parameters
        
    Returns:
        Configured ActiveLearner
    """
    strategy_map = {
        "random": SamplingStrategy.RANDOM,
        "uncertainty": SamplingStrategy.UNCERTAINTY,
        "margin": SamplingStrategy.MARGIN,
        "entropy": SamplingStrategy.ENTROPY,
        "diversity": SamplingStrategy.DIVERSITY,
        "hybrid": SamplingStrategy.HYBRID
    }
    
    config = ActiveLearningConfig(
        strategy=strategy_map.get(strategy, SamplingStrategy.UNCERTAINTY),
        **kwargs
    )
    
    return ActiveLearner(config)


__all__ = [
    'ActiveLearner',
    'ActiveLearningConfig',
    'UncertaintySampler',
    'DiversitySampler',
    'Sample',
    'LabeledSample',
    'SamplingStrategy',
    'create_active_learner'
]
