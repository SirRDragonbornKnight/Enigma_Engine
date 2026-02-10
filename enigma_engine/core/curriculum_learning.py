"""
Curriculum Learning for Enigma AI Engine

Train models from easy to hard examples.

Features:
- Difficulty scoring
- Automatic curriculum
- Competence-based progression
- Self-paced learning
- Anti-curriculum option

Usage:
    from enigma_engine.core.curriculum_learning import CurriculumTrainer, DifficultyScorer
    
    scorer = DifficultyScorer()
    trainer = CurriculumTrainer(model, scorer)
    
    # Train with curriculum
    trainer.train(dataset, epochs=10)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger(__name__)


class CurriculumStrategy(Enum):
    """Curriculum learning strategies."""
    FIXED = "fixed"  # Fixed schedule
    SELF_PACED = "self_paced"  # Model decides pace
    COMPETENCE = "competence"  # Based on model competence
    ANTI = "anti"  # Hard to easy (for fine-tuning)
    BABY_STEP = "baby_step"  # Very gradual increase


@dataclass
class DifficultyConfig:
    """Configuration for difficulty scoring."""
    use_length: bool = True  # Text length
    use_vocab: bool = True  # Vocabulary complexity
    use_loss: bool = False  # Model loss as difficulty
    length_weight: float = 0.3
    vocab_weight: float = 0.3
    loss_weight: float = 0.4


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    strategy: CurriculumStrategy = CurriculumStrategy.COMPETENCE
    initial_fraction: float = 0.2  # Start with easiest 20%
    growth_rate: float = 0.1  # Grow by 10% per epoch
    competence_threshold: float = 0.7  # Min accuracy to advance
    min_epochs_per_stage: int = 1  # Min epochs before advancing
    num_stages: int = 5  # Number of difficulty stages


class DifficultyScorer(ABC):
    """Abstract base class for difficulty scoring."""
    
    @abstractmethod
    def score(self, sample: Any) -> float:
        """
        Score difficulty of a sample.
        
        Returns:
            Difficulty score (0-1, higher = harder)
        """
    
    @abstractmethod
    def score_batch(self, samples: List[Any]) -> List[float]:
        """Score a batch of samples."""


class TextDifficultyScorer(DifficultyScorer):
    """Score text difficulty based on various metrics."""
    
    def __init__(
        self,
        tokenizer=None,
        config: Optional[DifficultyConfig] = None
    ):
        self.tokenizer = tokenizer
        self.config = config or DifficultyConfig()
        
        # Vocabulary frequency (for complexity scoring)
        self._vocab_freq: Dict[str, int] = {}
        
        # Statistics for normalization
        self._max_length = 1000
        self._mean_loss = 1.0
        
        logger.info("TextDifficultyScorer initialized")
    
    def fit_vocabulary(self, texts: List[str]):
        """Build vocabulary frequency from corpus."""
        for text in texts:
            tokens = self._tokenize(text)
            for token in tokens:
                self._vocab_freq[token] = self._vocab_freq.get(token, 0) + 1
        
        # Calculate max length
        lengths = [len(self._tokenize(t)) for t in texts]
        self._max_length = max(lengths) if lengths else 1000
    
    def score(self, sample: Any) -> float:
        """Score difficulty of a text sample."""
        if isinstance(sample, dict):
            text = sample.get('text', sample.get('input', str(sample)))
        else:
            text = str(sample)
        
        difficulty = 0.0
        weights_sum = 0.0
        
        if self.config.use_length:
            length_score = self._score_length(text)
            difficulty += length_score * self.config.length_weight
            weights_sum += self.config.length_weight
        
        if self.config.use_vocab:
            vocab_score = self._score_vocabulary(text)
            difficulty += vocab_score * self.config.vocab_weight
            weights_sum += self.config.vocab_weight
        
        if weights_sum > 0:
            difficulty /= weights_sum
        
        return min(1.0, max(0.0, difficulty))
    
    def score_batch(self, samples: List[Any]) -> List[float]:
        """Score a batch of samples."""
        return [self.score(s) for s in samples]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        if self.tokenizer:
            return self.tokenizer.encode(text)
        return text.split()
    
    def _score_length(self, text: str) -> float:
        """Score based on text length."""
        tokens = self._tokenize(text)
        return len(tokens) / self._max_length
    
    def _score_vocabulary(self, text: str) -> float:
        """Score based on vocabulary complexity."""
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0
        
        if not self._vocab_freq:
            return 0.5  # No frequency data
        
        # Calculate average rarity
        total_freq = sum(self._vocab_freq.values())
        rarities = []
        
        for token in tokens:
            freq = self._vocab_freq.get(token, 1)
            rarity = 1.0 - (freq / total_freq)
            rarities.append(rarity)
        
        return sum(rarities) / len(rarities) if rarities else 0.5


class LossDifficultyScorer(DifficultyScorer):
    """Score difficulty based on model loss."""
    
    def __init__(self, model: nn.Module, criterion=None):
        self.model = model
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = next(model.parameters()).device
        
        # Cache for scores
        self._score_cache: Dict[int, float] = {}
    
    def score(self, sample: Any) -> float:
        """Score based on model loss."""
        sample_hash = hash(str(sample))
        
        if sample_hash in self._score_cache:
            return self._score_cache[sample_hash]
        
        # Compute loss
        self.model.eval()
        with torch.no_grad():
            try:
                if isinstance(sample, dict):
                    input_ids = torch.tensor([sample['input_ids']], device=self.device)
                    labels = torch.tensor([sample.get('labels', sample['input_ids'])], device=self.device)
                else:
                    input_ids = torch.tensor([[sample]], device=self.device)
                    labels = input_ids.clone()
                
                outputs = self.model(input_ids)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                ).item()
                
                # Normalize loss to 0-1
                score = min(1.0, loss / 10.0)  # Assume max loss ~10
                
            except Exception:
                score = 0.5  # Default
        
        self._score_cache[sample_hash] = score
        return score
    
    def score_batch(self, samples: List[Any]) -> List[float]:
        """Score a batch of samples."""
        return [self.score(s) for s in samples]
    
    def clear_cache(self):
        """Clear score cache."""
        self._score_cache.clear()


class CurriculumSampler(Sampler):
    """Sampler that respects curriculum difficulty ordering."""
    
    def __init__(
        self,
        dataset: Dataset,
        difficulties: List[float],
        fraction: float = 1.0,
        shuffle_within_stage: bool = True
    ):
        self.dataset = dataset
        self.difficulties = difficulties
        self.fraction = fraction
        self.shuffle_within_stage = shuffle_within_stage
        
        # Sort indices by difficulty
        self._sorted_indices = sorted(
            range(len(difficulties)),
            key=lambda i: difficulties[i]
        )
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over indices."""
        # Select fraction of easiest samples
        n_samples = int(len(self._sorted_indices) * self.fraction)
        indices = self._sorted_indices[:n_samples]
        
        if self.shuffle_within_stage:
            import random
            indices = list(indices)
            random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self) -> int:
        return int(len(self._sorted_indices) * self.fraction)
    
    def set_fraction(self, fraction: float):
        """Update the fraction of data to use."""
        self.fraction = min(1.0, max(0.0, fraction))


class CurriculumTrainer:
    """Trainer with curriculum learning."""
    
    def __init__(
        self,
        model: nn.Module,
        difficulty_scorer: DifficultyScorer,
        config: Optional[CurriculumConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        Initialize curriculum trainer.
        
        Args:
            model: Model to train
            difficulty_scorer: Scorer for sample difficulty
            config: Curriculum configuration
            optimizer: Optimizer (created if not provided)
        """
        self.model = model
        self.scorer = difficulty_scorer
        self.config = config or CurriculumConfig()
        
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self._current_fraction = self.config.initial_fraction
        self._current_stage = 0
        self._epochs_at_stage = 0
        self._competence = 0.0
        
        # Metrics
        self._history: List[Dict] = []
        
        logger.info(
            f"CurriculumTrainer initialized: strategy={self.config.strategy.value}"
        )
    
    def train(
        self,
        dataset: Dataset,
        epochs: int = 10,
        batch_size: int = 32,
        eval_dataset: Optional[Dataset] = None,
        callback: Optional[Callable] = None
    ):
        """
        Train with curriculum learning.
        
        Args:
            dataset: Training dataset
            epochs: Number of epochs
            batch_size: Batch size
            eval_dataset: Optional evaluation dataset
            callback: Callback function(epoch, metrics)
        """
        device = next(self.model.parameters()).device
        
        # Score all samples
        logger.info("Scoring dataset difficulty...")
        difficulties = []
        for i in range(len(dataset)):
            sample = dataset[i]
            difficulties.append(self.scorer.score(sample))
        
        # Create curriculum sampler
        sampler = CurriculumSampler(
            dataset, difficulties,
            fraction=self._current_fraction
        )
        
        for epoch in range(epochs):
            # Update curriculum
            self._update_curriculum(epoch, sampler)
            
            # Create dataloader with current curriculum
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler
            )
            
            # Train epoch
            metrics = self._train_epoch(dataloader, device, epoch)
            
            # Evaluate
            if eval_dataset:
                eval_metrics = self._evaluate(eval_dataset, batch_size, device)
                metrics.update(eval_metrics)
                self._competence = eval_metrics.get('accuracy', 0.0)
            
            # Record history
            metrics['epoch'] = epoch
            metrics['fraction'] = self._current_fraction
            metrics['stage'] = self._current_stage
            self._history.append(metrics)
            
            logger.info(
                f"Epoch {epoch}: loss={metrics.get('loss', 0):.4f}, "
                f"fraction={self._current_fraction:.2f}, stage={self._current_stage}"
            )
            
            if callback:
                callback(epoch, metrics)
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        device: torch.device,
        epoch: int
    ) -> Dict:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            # Handle different batch formats
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                labels = batch.get('labels', input_ids).to(device)
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
                labels = batch[1].to(device) if len(batch) > 1 else input_ids.clone()
            else:
                input_ids = batch.to(device)
                labels = input_ids.clone()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Calculate loss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return {
            'loss': total_loss / max(1, n_batches),
            'n_batches': n_batches
        }
    
    def _evaluate(
        self,
        dataset: Dataset,
        batch_size: int,
        device: torch.device
    ) -> Dict:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(device)
                    labels = batch.get('labels', input_ids).to(device)
                else:
                    input_ids = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                    labels = input_ids.clone()
                
                outputs = self.model(input_ids)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                
                total_loss += loss.item()
                
                # Calculate accuracy
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        
        return {
            'eval_loss': total_loss / len(dataloader),
            'accuracy': correct / max(1, total)
        }
    
    def _update_curriculum(self, epoch: int, sampler: CurriculumSampler):
        """Update curriculum based on strategy."""
        self._epochs_at_stage += 1
        
        if self.config.strategy == CurriculumStrategy.FIXED:
            # Linear growth
            self._current_fraction = min(
                1.0,
                self.config.initial_fraction + epoch * self.config.growth_rate
            )
        
        elif self.config.strategy == CurriculumStrategy.COMPETENCE:
            # Advance when competence threshold reached
            if self._competence >= self.config.competence_threshold:
                if self._epochs_at_stage >= self.config.min_epochs_per_stage:
                    self._advance_stage()
        
        elif self.config.strategy == CurriculumStrategy.SELF_PACED:
            # Use loss trend to decide
            if len(self._history) >= 2:
                recent_loss = self._history[-1].get('loss', float('inf'))
                prev_loss = self._history[-2].get('loss', float('inf'))
                
                if recent_loss < prev_loss * 0.95:  # 5% improvement
                    self._advance_stage()
        
        elif self.config.strategy == CurriculumStrategy.ANTI:
            # Start hard, get easier (for fine-tuning)
            self._current_fraction = max(
                self.config.initial_fraction,
                1.0 - epoch * self.config.growth_rate
            )
        
        elif self.config.strategy == CurriculumStrategy.BABY_STEP:
            # Very gradual increase
            stage_fraction = 1.0 / self.config.num_stages
            self._current_fraction = min(
                1.0,
                (self._current_stage + 1) * stage_fraction
            )
        
        sampler.set_fraction(self._current_fraction)
    
    def _advance_stage(self):
        """Advance to next curriculum stage."""
        if self._current_stage < self.config.num_stages - 1:
            self._current_stage += 1
            self._epochs_at_stage = 0
            
            stage_fraction = 1.0 / self.config.num_stages
            self._current_fraction = min(
                1.0,
                (self._current_stage + 1) * stage_fraction
            )
            
            logger.info(f"Advanced to stage {self._current_stage}")
    
    def get_history(self) -> List[Dict]:
        """Get training history."""
        return list(self._history)
