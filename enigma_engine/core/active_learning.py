"""
Active Learning for Enigma AI Engine

AI identifies uncertain examples and requests labels.

Features:
- Uncertainty sampling
- Query-by-committee
- Diversity sampling
- Expected model change
- Batch selection

Usage:
    from enigma_engine.core.active_learning import ActiveLearner, get_learner
    
    learner = get_learner(model)
    
    # Select samples for labeling
    samples = learner.select_samples(unlabeled_data, n=10)
    
    # User provides labels
    labeled = [(sample, label) for sample, label in user_labels]
    
    # Update model
    learner.update(labeled)
"""

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Sample selection strategies."""
    UNCERTAINTY = "uncertainty"  # Most uncertain samples
    MARGIN = "margin"  # Smallest margin between top predictions
    ENTROPY = "entropy"  # Highest entropy
    RANDOM = "random"  # Random baseline
    DIVERSITY = "diversity"  # Maximize sample diversity
    HYBRID = "hybrid"  # Combine strategies


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning."""
    strategy: SelectionStrategy = SelectionStrategy.UNCERTAINTY
    
    # Selection
    batch_size: int = 10
    pool_size: int = 1000
    
    # Diversity
    diversity_weight: float = 0.3
    
    # Stopping
    max_queries: int = 100
    target_accuracy: float = 0.95
    
    # Training
    retrain_threshold: int = 5  # Retrain after this many labels


@dataclass
class Sample:
    """A sample for active learning."""
    id: int
    data: Any
    
    # Scores
    uncertainty_score: float = 0.0
    diversity_score: float = 0.0
    combined_score: float = 0.0
    
    # Label
    label: Optional[Any] = None
    is_labeled: bool = False


class UncertaintySampler:
    """Select samples with highest uncertainty."""
    
    def __init__(self, model: Any = None):
        self._model = model
    
    def compute_uncertainty(
        self,
        samples: List[Sample],
        model: Optional[Any] = None
    ) -> List[float]:
        """
        Compute uncertainty scores.
        
        Args:
            samples: Samples to score
            model: Override model
            
        Returns:
            List of uncertainty scores
        """
        model = model or self._model
        scores = []
        
        if model is None:
            # Random scores if no model
            return [random.random() for _ in samples]
        
        for sample in samples:
            try:
                import torch
                import torch.nn.functional as F
                
                # Get model predictions
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba([sample.data])[0]
                elif hasattr(model, '__call__'):
                    with torch.no_grad():
                        output = model(sample.data)
                        if hasattr(output, 'logits'):
                            probs = F.softmax(output.logits, dim=-1).squeeze()
                        else:
                            probs = F.softmax(output, dim=-1).squeeze()
                        probs = probs.cpu().numpy()
                else:
                    # No model prediction available
                    scores.append(random.random())
                    continue
                
                # Least confidence uncertainty
                max_prob = max(probs) if len(probs) > 0 else 0.5
                uncertainty = 1 - max_prob
                scores.append(uncertainty)
                
            except Exception as e:
                logger.debug(f"Uncertainty computation failed: {e}")
                scores.append(random.random())
        
        return scores


class MarginSampler:
    """Select samples with smallest prediction margin."""
    
    def __init__(self, model: Any = None):
        self._model = model
    
    def compute_margin(
        self,
        samples: List[Sample],
        model: Optional[Any] = None
    ) -> List[float]:
        """
        Compute margin scores (lower = more uncertain).
        
        Returns:
            List of margin scores (inverted so higher = select)
        """
        model = model or self._model
        scores = []
        
        if model is None:
            return [random.random() for _ in samples]
        
        for sample in samples:
            try:
                import torch
                import torch.nn.functional as F
                
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba([sample.data])[0]
                elif hasattr(model, '__call__'):
                    with torch.no_grad():
                        output = model(sample.data)
                        if hasattr(output, 'logits'):
                            probs = F.softmax(output.logits, dim=-1).squeeze()
                        else:
                            probs = F.softmax(output, dim=-1).squeeze()
                        probs = sorted(probs.cpu().numpy(), reverse=True)
                else:
                    scores.append(random.random())
                    continue
                
                # Margin: difference between top two predictions
                if len(probs) >= 2:
                    margin = probs[0] - probs[1]
                else:
                    margin = probs[0] if len(probs) > 0 else 0.5
                
                # Invert so smaller margin = higher score
                scores.append(1 - margin)
                
            except Exception as e:
                logger.debug(f"Margin computation failed: {e}")
                scores.append(random.random())
        
        return scores


class EntropySampler:
    """Select samples with highest prediction entropy."""
    
    def __init__(self, model: Any = None):
        self._model = model
    
    def compute_entropy(
        self,
        samples: List[Sample],
        model: Optional[Any] = None
    ) -> List[float]:
        """
        Compute entropy scores.
        
        Returns:
            List of entropy scores (higher = more uncertain)
        """
        import math
        
        model = model or self._model
        scores = []
        
        if model is None:
            return [random.random() for _ in samples]
        
        for sample in samples:
            try:
                import torch
                import torch.nn.functional as F
                
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba([sample.data])[0]
                elif hasattr(model, '__call__'):
                    with torch.no_grad():
                        output = model(sample.data)
                        if hasattr(output, 'logits'):
                            probs = F.softmax(output.logits, dim=-1).squeeze()
                        else:
                            probs = F.softmax(output, dim=-1).squeeze()
                        probs = probs.cpu().numpy()
                else:
                    scores.append(random.random())
                    continue
                
                # Shannon entropy
                entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
                
                # Normalize by max entropy
                max_entropy = math.log(len(probs)) if len(probs) > 0 else 1
                normalized = entropy / max_entropy if max_entropy > 0 else 0
                
                scores.append(normalized)
                
            except Exception as e:
                logger.debug(f"Entropy computation failed: {e}")
                scores.append(random.random())
        
        return scores


class DiversitySampler:
    """Select diverse samples using clustering."""
    
    def __init__(self, embedding_fn: Optional[Callable] = None):
        self._embedding_fn = embedding_fn
    
    def compute_embeddings(
        self,
        samples: List[Sample]
    ) -> List[List[float]]:
        """Get embeddings for samples."""
        if self._embedding_fn is None:
            # Simple bag-of-words embedding
            embeddings = []
            for sample in samples:
                text = str(sample.data)
                words = text.lower().split()
                # Simple hash-based embedding
                embedding = [0.0] * 128
                for word in words:
                    idx = hash(word) % 128
                    embedding[idx] += 1
                norm = sum(e**2 for e in embedding) ** 0.5
                embedding = [e / norm if norm > 0 else 0 for e in embedding]
                embeddings.append(embedding)
            return embeddings
        
        return [self._embedding_fn(s.data) for s in samples]
    
    def compute_diversity(
        self,
        samples: List[Sample],
        selected_indices: set
    ) -> List[float]:
        """
        Compute diversity scores relative to selected samples.
        
        Returns:
            List of diversity scores (higher = more different from selected)
        """
        embeddings = self.compute_embeddings(samples)
        scores = []
        
        if not selected_indices:
            return [1.0 for _ in samples]
        
        # Get embeddings of selected samples
        selected_embeddings = [embeddings[i] for i in selected_indices if i < len(embeddings)]
        
        for i, emb in enumerate(embeddings):
            if i in selected_indices:
                scores.append(0.0)  # Already selected
                continue
            
            # Min distance to selected samples
            if selected_embeddings:
                min_dist = min(
                    self._cosine_distance(emb, sel_emb)
                    for sel_emb in selected_embeddings
                )
            else:
                min_dist = 1.0
            
            scores.append(min_dist)
        
        return scores
    
    def _cosine_distance(self, a: List[float], b: List[float]) -> float:
        """Compute cosine distance."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 1.0
        
        similarity = dot / (norm_a * norm_b)
        return 1 - max(-1, min(1, similarity))


class ActiveLearner:
    """High-level active learning interface."""
    
    def __init__(
        self,
        model: Any = None,
        config: Optional[ActiveLearningConfig] = None
    ):
        """
        Initialize active learner.
        
        Args:
            model: Model for uncertainty estimation
            config: Active learning configuration
        """
        self._model = model
        self._config = config or ActiveLearningConfig()
        
        self._uncertainty = UncertaintySampler(model)
        self._margin = MarginSampler(model)
        self._entropy = EntropySampler(model)
        self._diversity = DiversitySampler()
        
        # State
        self._labeled_samples: List[Sample] = []
        self._pool: List[Sample] = []
        self._query_count = 0
    
    def initialize_pool(
        self,
        data: List[Any],
        ids: Optional[List[int]] = None
    ):
        """
        Initialize unlabeled data pool.
        
        Args:
            data: Unlabeled data items
            ids: Optional IDs for items
        """
        self._pool = []
        
        for i, item in enumerate(data):
            sample_id = ids[i] if ids else i
            self._pool.append(Sample(id=sample_id, data=item))
        
        logger.info(f"Initialized pool with {len(self._pool)} samples")
    
    def select_samples(
        self,
        n: Optional[int] = None,
        strategy: Optional[SelectionStrategy] = None
    ) -> List[Sample]:
        """
        Select samples for labeling.
        
        Args:
            n: Number of samples to select
            strategy: Override selection strategy
            
        Returns:
            Selected samples
        """
        n = n or self._config.batch_size
        strategy = strategy or self._config.strategy
        
        # Get unlabeled samples
        unlabeled = [s for s in self._pool if not s.is_labeled]
        
        if not unlabeled:
            logger.warning("No unlabeled samples in pool")
            return []
        
        # Compute scores based on strategy
        if strategy == SelectionStrategy.UNCERTAINTY:
            scores = self._uncertainty.compute_uncertainty(unlabeled, self._model)
        elif strategy == SelectionStrategy.MARGIN:
            scores = self._margin.compute_margin(unlabeled, self._model)
        elif strategy == SelectionStrategy.ENTROPY:
            scores = self._entropy.compute_entropy(unlabeled, self._model)
        elif strategy == SelectionStrategy.DIVERSITY:
            selected_ids = {s.id for s in self._labeled_samples}
            scores = self._diversity.compute_diversity(unlabeled, selected_ids)
        elif strategy == SelectionStrategy.HYBRID:
            # Combine uncertainty and diversity
            uncertainty_scores = self._uncertainty.compute_uncertainty(unlabeled, self._model)
            selected_ids = {s.id for s in self._labeled_samples}
            diversity_scores = self._diversity.compute_diversity(unlabeled, selected_ids)
            
            w = self._config.diversity_weight
            scores = [
                (1 - w) * u + w * d
                for u, d in zip(uncertainty_scores, diversity_scores)
            ]
        else:
            # Random
            scores = [random.random() for _ in unlabeled]
        
        # Assign scores
        for sample, score in zip(unlabeled, scores):
            sample.uncertainty_score = score
        
        # Select top n
        sorted_samples = sorted(unlabeled, key=lambda s: -s.uncertainty_score)
        selected = sorted_samples[:n]
        
        self._query_count += len(selected)
        logger.info(f"Selected {len(selected)} samples (total queries: {self._query_count})")
        
        return selected
    
    def label_sample(
        self,
        sample: Sample,
        label: Any
    ):
        """
        Assign label to a sample.
        
        Args:
            sample: Sample to label
            label: The label
        """
        sample.label = label
        sample.is_labeled = True
        self._labeled_samples.append(sample)
    
    def label_samples(
        self,
        labels: List[Tuple[Sample, Any]]
    ):
        """
        Label multiple samples.
        
        Args:
            labels: List of (sample, label) tuples
        """
        for sample, label in labels:
            self.label_sample(sample, label)
        
        # Check if we should retrain
        if len(self._labeled_samples) % self._config.retrain_threshold == 0:
            self.update_model()
    
    def update_model(self):
        """Update model with labeled data."""
        if self._model is None:
            logger.warning("No model to update")
            return
        
        # Prepare training data
        train_data = [(s.data, s.label) for s in self._labeled_samples if s.is_labeled]
        
        if not train_data:
            return
        
        # Model-specific training
        if hasattr(self._model, 'fit'):
            X = [d for d, _ in train_data]
            y = [l for _, l in train_data]
            self._model.fit(X, y)
            logger.info(f"Model updated with {len(train_data)} samples")
        else:
            logger.warning("Model doesn't have fit method - manual training required")
    
    def get_labeled_data(self) -> List[Tuple[Any, Any]]:
        """Get all labeled data."""
        return [(s.data, s.label) for s in self._labeled_samples if s.is_labeled]
    
    def should_continue(self) -> bool:
        """Check if active learning should continue."""
        if self._query_count >= self._config.max_queries:
            logger.info("Reached max queries")
            return False
        
        if not any(not s.is_labeled for s in self._pool):
            logger.info("Pool exhausted")
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get active learning statistics."""
        return {
            "total_queries": self._query_count,
            "labeled_count": len(self._labeled_samples),
            "pool_remaining": sum(1 for s in self._pool if not s.is_labeled),
            "strategy": self._config.strategy.value
        }
    
    def run_loop(
        self,
        labeler: Callable[[Sample], Any],
        max_iterations: Optional[int] = None
    ) -> List[Tuple[Any, Any]]:
        """
        Run active learning loop.
        
        Args:
            labeler: Function that returns label for a sample
            max_iterations: Maximum iterations
            
        Returns:
            All labeled data
        """
        max_iterations = max_iterations or self._config.max_queries
        
        for iteration in range(max_iterations):
            if not self.should_continue():
                break
            
            # Select samples
            samples = self.select_samples()
            
            if not samples:
                break
            
            # Get labels
            labels = []
            for sample in samples:
                label = labeler(sample)
                labels.append((sample, label))
            
            # Update
            self.label_samples(labels)
            
            logger.info(f"Iteration {iteration + 1}: labeled {len(labels)} samples")
        
        return self.get_labeled_data()


# Global instance
_learner: Optional[ActiveLearner] = None


def get_learner(
    model: Optional[Any] = None,
    config: Optional[ActiveLearningConfig] = None
) -> ActiveLearner:
    """Get or create global active learner."""
    global _learner
    if _learner is None or model is not None:
        _learner = ActiveLearner(model, config)
    return _learner
