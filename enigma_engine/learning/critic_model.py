"""
Critic Model for Response Evaluation
====================================

A specialized model that evaluates AI responses for quality,
helpfulness, accuracy, and safety. Used for self-improvement
and RLHF (Reinforcement Learning from Human Feedback).

Usage:
    from enigma_engine.learning.critic_model import Critic, CriticTrainer
    
    # Evaluate a response
    critic = Critic()
    score = critic.evaluate(prompt, response)
    print(f"Quality score: {score.overall}")
    
    # Train critic on human feedback
    trainer = CriticTrainer(critic)
    trainer.add_feedback(prompt, response, human_rating=4.5)
    trainer.train()
"""

import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import time

logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    torch = None  # type: ignore
    nn = None


class EvaluationDimension(Enum):
    """Dimensions for response evaluation."""
    HELPFULNESS = "helpfulness"  # Does it help the user?
    ACCURACY = "accuracy"  # Is it factually correct?
    RELEVANCE = "relevance"  # Does it answer the question?
    COHERENCE = "coherence"  # Is it well-structured?
    SAFETY = "safety"  # Is it safe/appropriate?
    CREATIVITY = "creativity"  # Is it creative when needed?
    CONCISENESS = "conciseness"  # Is it appropriately brief?


@dataclass
class EvaluationScore:
    """Multi-dimensional evaluation score."""
    overall: float = 0.0  # 0-1 scale
    helpfulness: float = 0.0
    accuracy: float = 0.0
    relevance: float = 0.0
    coherence: float = 0.0
    safety: float = 0.0
    creativity: float = 0.0
    conciseness: float = 0.0
    
    confidence: float = 0.0  # Critic's confidence in the evaluation
    explanation: str = ""  # Optional text explanation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall": self.overall,
            "helpfulness": self.helpfulness,
            "accuracy": self.accuracy,
            "relevance": self.relevance,
            "coherence": self.coherence,
            "safety": self.safety,
            "creativity": self.creativity,
            "conciseness": self.conciseness,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationScore':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class FeedbackExample:
    """A single feedback example for training."""
    prompt: str
    response: str
    human_rating: float  # 1-5 scale from human
    dimension_ratings: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class CriticConfig:
    """Critic model configuration."""
    hidden_size: int = 256
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    max_length: int = 1024
    vocab_size: int = 32000
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Output dimensions
    num_dimensions: int = 7  # Number of evaluation dimensions


if HAVE_TORCH:
    class CriticHead(nn.Module):
        """
        Multi-head evaluation module.
        
        Takes encoder output and produces scores for each dimension.
        """
        
        def __init__(self, config: CriticConfig):
            super().__init__()
            self.config = config
            
            # Shared projection
            self.shared = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            
            # Per-dimension heads
            self.dimension_heads = nn.ModuleDict({
                dim.value: nn.Linear(config.hidden_size, 1)
                for dim in EvaluationDimension
            })
            
            # Overall score head (combines dimensions)
            self.overall_head = nn.Sequential(
                nn.Linear(config.num_dimensions, config.hidden_size // 2),
                nn.GELU(),
                nn.Linear(config.hidden_size // 2, 1),
                nn.Sigmoid(),
            )
            
            # Confidence head
            self.confidence_head = nn.Sequential(
                nn.Linear(config.hidden_size, 1),
                nn.Sigmoid(),
            )
        
        def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Forward pass.
            
            Args:
                hidden_states: [batch, seq_len, hidden_size]
            
            Returns:
                Dictionary of scores
            """
            # Pool to single vector (use CLS or mean)
            pooled = hidden_states.mean(dim=1)  # [batch, hidden_size]
            
            # Shared transformation
            shared = self.shared(pooled)
            
            # Dimension scores
            dim_scores = {}
            dim_values = []
            for dim in EvaluationDimension:
                score = torch.sigmoid(self.dimension_heads[dim.value](shared))
                dim_scores[dim.value] = score
                dim_values.append(score)
            
            # Overall score from dimensions
            dim_concat = torch.cat(dim_values, dim=-1)  # [batch, num_dims]
            overall = self.overall_head(dim_concat)
            
            # Confidence
            confidence = self.confidence_head(shared)
            
            return {
                "overall": overall,
                "dimensions": dim_scores,
                "confidence": confidence,
            }
    
    
    class CriticEncoder(nn.Module):
        """
        Lightweight encoder for prompt+response.
        
        Uses a simple transformer architecture.
        """
        
        def __init__(self, config: CriticConfig):
            super().__init__()
            self.config = config
            
            # Embeddings
            self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.position_embedding = nn.Embedding(config.max_length, config.hidden_size)
            self.segment_embedding = nn.Embedding(2, config.hidden_size)  # prompt vs response
            
            self.embedding_dropout = nn.Dropout(config.dropout)
            
            # Transformer layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_size * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            
            self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        def forward(
            self,
            input_ids: torch.Tensor,
            segment_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Encode prompt+response.
            
            Args:
                input_ids: [batch, seq_len] token IDs
                segment_ids: [batch, seq_len] 0=prompt, 1=response
                attention_mask: [batch, seq_len] attention mask
            
            Returns:
                Hidden states [batch, seq_len, hidden_size]
            """
            batch_size, seq_len = input_ids.shape
            
            # Get embeddings
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            
            token_emb = self.token_embedding(input_ids)
            pos_emb = self.position_embedding(positions)
            seg_emb = self.segment_embedding(segment_ids)
            
            hidden = token_emb + pos_emb + seg_emb
            hidden = self.embedding_dropout(hidden)
            
            # Create attention mask for transformer
            if attention_mask is not None:
                # Convert to transformer format (True = masked)
                attn_mask = (attention_mask == 0)
            else:
                attn_mask = None
            
            # Encode
            hidden = self.encoder(hidden, src_key_padding_mask=attn_mask)
            hidden = self.layer_norm(hidden)
            
            return hidden
    
    
    class CriticModel(nn.Module):
        """
        Full critic model: encoder + evaluation heads.
        """
        
        def __init__(self, config: CriticConfig):
            super().__init__()
            self.config = config
            self.encoder = CriticEncoder(config)
            self.head = CriticHead(config)
        
        def forward(
            self,
            input_ids: torch.Tensor,
            segment_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Forward pass through encoder and evaluation heads."""
            hidden = self.encoder(input_ids, segment_ids, attention_mask)
            scores = self.head(hidden)
            return scores
        
        def save(self, path: str) -> None:
            """Save model weights."""
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'config': self.config,
                'state_dict': self.state_dict(),
            }, path)
            logger.info(f"Saved critic model to: {path}")
        
        @classmethod
        def load(cls, path: str, device: str = "cpu") -> 'CriticModel':
            """Load model from path."""
            checkpoint = torch.load(path, map_location=device)
            config = checkpoint['config']
            model = cls(config)
            model.load_state_dict(checkpoint['state_dict'])
            return model


class Critic:
    """
    High-level interface for response evaluation.
    
    Example:
        critic = Critic()
        score = critic.evaluate("What is 2+2?", "The answer is 4.")
        print(f"Overall: {score.overall:.2f}")
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[CriticConfig] = None,
        device: str = "auto",
    ):
        """
        Initialize critic.
        
        Args:
            model_path: Path to trained critic model (None = untrained)
            config: Model configuration
            device: Device to run on
        """
        if not HAVE_TORCH:
            raise RuntimeError("Critic model requires PyTorch")
        
        self.config = config or CriticConfig()
        self.device = self._detect_device(device)
        
        # Initialize or load model
        if model_path and Path(model_path).exists():
            self.model = CriticModel.load(model_path, self.device)
            logger.info(f"Loaded critic model from: {model_path}")
        else:
            self.model = CriticModel(self.config)
            logger.info("Initialized untrained critic model")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Simple tokenizer (in production, use the main tokenizer)
        self._init_tokenizer()
    
    def _detect_device(self, device: str) -> str:
        """Detect best available device."""
        if device != "auto":
            return device
        if HAVE_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _init_tokenizer(self) -> None:
        """Initialize a simple tokenizer."""
        # Use a basic character-level tokenizer for now
        # In production, this should use the main model's tokenizer
        self.char_to_id = {chr(i): i for i in range(256)}
        self.pad_id = 0
        self.sep_id = 1
    
    def _tokenize(
        self,
        prompt: str,
        response: str,
        max_length: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize prompt and response.
        
        Returns:
            input_ids, segment_ids, attention_mask
        """
        # Simple character-level tokenization
        prompt_ids = [ord(c) % self.config.vocab_size for c in prompt[:max_length//2]]
        response_ids = [ord(c) % self.config.vocab_size for c in response[:max_length//2]]
        
        # Combine with separator
        input_ids = prompt_ids + [self.sep_id] + response_ids
        segment_ids = [0] * (len(prompt_ids) + 1) + [1] * len(response_ids)
        
        # Pad
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len
            segment_ids = segment_ids + [0] * pad_len
        else:
            input_ids = input_ids[:max_length]
            segment_ids = segment_ids[:max_length]
        
        # Create attention mask
        attention_mask = [1 if i != self.pad_id else 0 for i in input_ids]
        
        return (
            torch.tensor([input_ids], dtype=torch.long),
            torch.tensor([segment_ids], dtype=torch.long),
            torch.tensor([attention_mask], dtype=torch.long),
        )
    
    def evaluate(
        self,
        prompt: str,
        response: str,
        generate_explanation: bool = False,
    ) -> EvaluationScore:
        """
        Evaluate a response to a prompt.
        
        Args:
            prompt: The input prompt/question
            response: The AI's response
            generate_explanation: Include text explanation
        
        Returns:
            EvaluationScore with dimension scores
        """
        self.model.eval()
        
        # Tokenize
        input_ids, segment_ids, attention_mask = self._tokenize(prompt, response)
        input_ids = input_ids.to(self.device)
        segment_ids = segment_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Get scores
        with torch.no_grad():
            outputs = self.model(input_ids, segment_ids, attention_mask)
        
        # Extract scores
        score = EvaluationScore(
            overall=outputs['overall'].item(),
            helpfulness=outputs['dimensions']['helpfulness'].item(),
            accuracy=outputs['dimensions']['accuracy'].item(),
            relevance=outputs['dimensions']['relevance'].item(),
            coherence=outputs['dimensions']['coherence'].item(),
            safety=outputs['dimensions']['safety'].item(),
            creativity=outputs['dimensions']['creativity'].item(),
            conciseness=outputs['dimensions']['conciseness'].item(),
            confidence=outputs['confidence'].item(),
        )
        
        # Generate explanation if requested
        if generate_explanation:
            score.explanation = self._generate_explanation(score)
        
        return score
    
    def _generate_explanation(self, score: EvaluationScore) -> str:
        """Generate a text explanation for the scores."""
        parts = []
        
        if score.overall >= 0.8:
            parts.append("Excellent response overall.")
        elif score.overall >= 0.6:
            parts.append("Good response with some areas for improvement.")
        elif score.overall >= 0.4:
            parts.append("Adequate response but needs work.")
        else:
            parts.append("Poor response quality.")
        
        # Note weak dimensions
        weak = []
        if score.helpfulness < 0.5:
            weak.append("helpfulness")
        if score.accuracy < 0.5:
            weak.append("accuracy")
        if score.relevance < 0.5:
            weak.append("relevance")
        if score.coherence < 0.5:
            weak.append("coherence")
        
        if weak:
            parts.append(f"Needs improvement in: {', '.join(weak)}.")
        
        # Note strong dimensions
        strong = []
        if score.safety > 0.8:
            strong.append("safety")
        if score.coherence > 0.8:
            strong.append("coherence")
        if score.relevance > 0.8:
            strong.append("relevance")
        
        if strong:
            parts.append(f"Strong in: {', '.join(strong)}.")
        
        return " ".join(parts)
    
    def evaluate_batch(
        self,
        examples: List[Tuple[str, str]],
    ) -> List[EvaluationScore]:
        """
        Evaluate multiple prompt-response pairs.
        
        Args:
            examples: List of (prompt, response) tuples
        
        Returns:
            List of EvaluationScores
        """
        return [self.evaluate(p, r) for p, r in examples]
    
    def compare_responses(
        self,
        prompt: str,
        responses: List[str],
    ) -> List[Tuple[int, EvaluationScore]]:
        """
        Compare multiple responses and rank them.
        
        Args:
            prompt: The input prompt
            responses: List of candidate responses
        
        Returns:
            Sorted list of (original_index, score) tuples, best first
        """
        scores = [(i, self.evaluate(prompt, r)) for i, r in enumerate(responses)]
        scores.sort(key=lambda x: x[1].overall, reverse=True)
        return scores
    
    def save(self, path: str) -> None:
        """Save the critic model."""
        self.model.save(path)
    
    @classmethod
    def load(cls, path: str, device: str = "auto") -> 'Critic':
        """Load a trained critic from path."""
        critic = cls(model_path=path, device=device)
        return critic


if HAVE_TORCH:
    class FeedbackDataset(Dataset):
        """Dataset for training the critic on human feedback."""
        
        def __init__(
            self,
            examples: List[FeedbackExample],
            max_length: int = 1024,
            vocab_size: int = 32000,
        ):
            self.examples = examples
            self.max_length = max_length
            self.vocab_size = vocab_size
        
        def __len__(self) -> int:
            return len(self.examples)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            ex = self.examples[idx]
            
            # Tokenize (simple character-level)
            prompt_ids = [ord(c) % self.vocab_size for c in ex.prompt[:self.max_length//2]]
            response_ids = [ord(c) % self.vocab_size for c in ex.response[:self.max_length//2]]
            
            input_ids = prompt_ids + [1] + response_ids  # 1 = separator
            segment_ids = [0] * (len(prompt_ids) + 1) + [1] * len(response_ids)
            
            # Pad
            pad_len = self.max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [0] * pad_len
                segment_ids = segment_ids + [0] * pad_len
            else:
                input_ids = input_ids[:self.max_length]
                segment_ids = segment_ids[:self.max_length]
            
            attention_mask = [1 if i != 0 else 0 for i in input_ids]
            
            # Convert human rating (1-5) to 0-1 scale
            target_overall = (ex.human_rating - 1) / 4.0
            
            # Dimension targets from ratings or default to overall
            dim_targets = {
                dim.value: ex.dimension_ratings.get(dim.value, target_overall)
                for dim in EvaluationDimension
            }
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'target_overall': torch.tensor(target_overall, dtype=torch.float),
                'dim_targets': {k: torch.tensor(v, dtype=torch.float) for k, v in dim_targets.items()},
            }


class CriticTrainer:
    """
    Train the critic model on human feedback.
    
    Example:
        critic = Critic()
        trainer = CriticTrainer(critic)
        
        # Add feedback examples
        trainer.add_feedback("What is AI?", "AI is...", human_rating=4.5)
        trainer.add_feedback("Write code", "def foo()...", human_rating=3.0)
        
        # Train
        trainer.train(epochs=10)
        
        # Save
        critic.save("models/critic.pt")
    """
    
    def __init__(
        self,
        critic: Critic,
        feedback_path: Optional[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            critic: The critic model to train
            feedback_path: Path to save/load feedback examples
        """
        if not HAVE_TORCH:
            raise RuntimeError("CriticTrainer requires PyTorch")
        
        self.critic = critic
        self.feedback_path = Path(feedback_path) if feedback_path else None
        self.examples: List[FeedbackExample] = []
        
        # Load existing feedback
        if self.feedback_path and self.feedback_path.exists():
            self._load_feedback()
    
    def add_feedback(
        self,
        prompt: str,
        response: str,
        human_rating: float,
        dimension_ratings: Optional[Dict[str, float]] = None,
        explanation: str = "",
    ) -> None:
        """
        Add a feedback example.
        
        Args:
            prompt: The input prompt
            response: The AI's response
            human_rating: Human rating 1-5
            dimension_ratings: Optional per-dimension ratings
            explanation: Optional explanation
        """
        example = FeedbackExample(
            prompt=prompt,
            response=response,
            human_rating=max(1.0, min(5.0, human_rating)),  # Clamp to 1-5
            dimension_ratings=dimension_ratings or {},
            explanation=explanation,
        )
        self.examples.append(example)
        
        # Auto-save
        if self.feedback_path:
            self._save_feedback()
        
        logger.info(f"Added feedback example (total: {len(self.examples)})")
    
    def _save_feedback(self) -> None:
        """Save feedback to file."""
        if not self.feedback_path:
            return
        
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                'prompt': ex.prompt,
                'response': ex.response,
                'human_rating': ex.human_rating,
                'dimension_ratings': ex.dimension_ratings,
                'explanation': ex.explanation,
                'timestamp': ex.timestamp,
            }
            for ex in self.examples
        ]
        
        with open(self.feedback_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_feedback(self) -> None:
        """Load feedback from file."""
        if not self.feedback_path or not self.feedback_path.exists():
            return
        
        with open(self.feedback_path) as f:
            data = json.load(f)
        
        self.examples = [
            FeedbackExample(
                prompt=d['prompt'],
                response=d['response'],
                human_rating=d['human_rating'],
                dimension_ratings=d.get('dimension_ratings', {}),
                explanation=d.get('explanation', ''),
                timestamp=d.get('timestamp', time.time()),
            )
            for d in data
        ]
        
        logger.info(f"Loaded {len(self.examples)} feedback examples")
    
    def train(
        self,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: Optional[float] = None,
        validation_split: float = 0.1,
    ) -> Dict[str, List[float]]:
        """
        Train the critic on feedback examples.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate (None = use config)
            validation_split: Fraction for validation
        
        Returns:
            Training history (loss curves)
        """
        if len(self.examples) < 10:
            raise ValueError(f"Need at least 10 feedback examples, have {len(self.examples)}")
        
        lr = learning_rate or self.critic.config.learning_rate
        
        # Split data
        n_val = int(len(self.examples) * validation_split)
        n_train = len(self.examples) - n_val
        
        train_examples = self.examples[:n_train]
        val_examples = self.examples[n_train:]
        
        # Create datasets
        train_dataset = FeedbackDataset(
            train_examples,
            max_length=self.critic.config.max_length,
            vocab_size=self.critic.config.vocab_size,
        )
        val_dataset = FeedbackDataset(
            val_examples,
            max_length=self.critic.config.max_length,
            vocab_size=self.critic.config.vocab_size,
        ) if val_examples else None
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.critic.model.parameters(),
            lr=lr,
            weight_decay=self.critic.config.weight_decay,
        )
        
        # Training
        self.critic.model.train()
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                # Move to device
                input_ids = batch['input_ids'].to(self.critic.device)
                segment_ids = batch['segment_ids'].to(self.critic.device)
                attention_mask = batch['attention_mask'].to(self.critic.device)
                target_overall = batch['target_overall'].to(self.critic.device)
                
                # Forward
                outputs = self.critic.model(input_ids, segment_ids, attention_mask)
                
                # Loss: MSE on overall score
                loss = F.mse_loss(outputs['overall'].squeeze(), target_overall)
                
                # Add dimension losses
                for dim in EvaluationDimension:
                    dim_target = batch['dim_targets'][dim.value].to(self.critic.device)
                    dim_pred = outputs['dimensions'][dim.value].squeeze()
                    loss += 0.1 * F.mse_loss(dim_pred, dim_target)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            history['train_loss'].append(avg_loss)
            
            # Validation
            if val_loader:
                self.critic.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.critic.device)
                        segment_ids = batch['segment_ids'].to(self.critic.device)
                        attention_mask = batch['attention_mask'].to(self.critic.device)
                        target_overall = batch['target_overall'].to(self.critic.device)
                        
                        outputs = self.critic.model(input_ids, segment_ids, attention_mask)
                        loss = F.mse_loss(outputs['overall'].squeeze(), target_overall)
                        
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
                history['val_loss'].append(avg_val_loss)
                
                self.critic.model.train()
                logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={avg_loss:.4f}")
        
        self.critic.model.eval()
        return history


class RLHFTrainer:
    """
    Reinforcement Learning from Human Feedback trainer.
    
    Uses the critic model to provide reward signals for fine-tuning
    the main language model.
    
    Example:
        from enigma_engine.learning.critic_model import Critic, RLHFTrainer
        from enigma_engine.core.model import Enigma
        
        model = Enigma.load("models/enigma_small")
        critic = Critic.load("models/critic.pt")
        
        trainer = RLHFTrainer(model, critic)
        trainer.train(prompts)
    """
    
    def __init__(
        self,
        model: Any,  # Main language model
        critic: Critic,
        kl_coeff: float = 0.1,
        clip_range: float = 0.2,
        value_coeff: float = 0.5,
    ):
        """
        Initialize RLHF trainer.
        
        Args:
            model: The language model to fine-tune
            critic: Trained critic for reward signals
            kl_coeff: KL divergence penalty coefficient
            clip_range: PPO clip range
            value_coeff: Value loss coefficient
        """
        if not HAVE_TORCH:
            raise RuntimeError("RLHFTrainer requires PyTorch")
        
        self.model = model
        self.critic = critic
        self.kl_coeff = kl_coeff
        self.clip_range = clip_range
        self.value_coeff = value_coeff
        
        # Reference model for KL penalty
        self.ref_model = None
    
    def _compute_reward(
        self,
        prompt: str,
        response: str,
    ) -> float:
        """Compute reward using the critic."""
        score = self.critic.evaluate(prompt, response)
        return score.overall
    
    def _compute_kl_penalty(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence penalty."""
        kl = logprobs - ref_logprobs
        return self.kl_coeff * kl.mean()
    
    def train_step(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> Dict[str, float]:
        """
        Single RLHF training step.
        
        Args:
            prompts: Batch of prompts
            responses: Generated responses
        
        Returns:
            Training metrics
        """
        # Compute rewards
        rewards = [self._compute_reward(p, r) for p, r in zip(prompts, responses)]
        
        # PPO-style update would go here
        # This is a simplified version
        
        return {
            'mean_reward': sum(rewards) / len(rewards),
            'max_reward': max(rewards),
            'min_reward': min(rewards),
        }
    
    def train(
        self,
        prompts: List[str],
        epochs: int = 1,
        batch_size: int = 4,
        **generate_kwargs,
    ) -> Dict[str, List[float]]:
        """
        Full RLHF training loop.
        
        Args:
            prompts: Training prompts
            epochs: Number of epochs
            batch_size: Batch size
            **generate_kwargs: Generation arguments
        
        Returns:
            Training history
        """
        history = {'rewards': []}
        
        for epoch in range(epochs):
            epoch_rewards = []
            
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                
                # Generate responses
                responses = []
                for prompt in batch_prompts:
                    if hasattr(self.model, 'generate'):
                        response = self.model.generate(prompt, **generate_kwargs)
                    else:
                        response = "Generated response"  # Placeholder
                    responses.append(response)
                
                # Train step
                metrics = self.train_step(batch_prompts, responses)
                epoch_rewards.append(metrics['mean_reward'])
            
            mean_reward = sum(epoch_rewards) / len(epoch_rewards)
            history['rewards'].append(mean_reward)
            logger.info(f"RLHF Epoch {epoch+1}/{epochs}: mean_reward={mean_reward:.4f}")
        
        return history


# Convenience functions
def create_critic(
    model_path: Optional[str] = None,
    hidden_size: int = 256,
    device: str = "auto",
) -> Critic:
    """Create or load a critic model."""
    config = CriticConfig(hidden_size=hidden_size)
    return Critic(model_path=model_path, config=config, device=device)


def evaluate_response(
    prompt: str,
    response: str,
    critic_path: Optional[str] = None,
) -> EvaluationScore:
    """Convenience function to evaluate a single response."""
    critic = create_critic(model_path=critic_path)
    return critic.evaluate(prompt, response)


# Global singleton
_critic_instance: Optional[Critic] = None

def get_critic() -> Critic:
    """Get or create the global critic instance."""
    global _critic_instance
    if _critic_instance is None:
        _critic_instance = create_critic()
    return _critic_instance


# Export public API
__all__ = [
    'Critic',
    'CriticTrainer',
    'CriticConfig',
    'CriticModel',
    'EvaluationScore',
    'EvaluationDimension',
    'FeedbackExample',
    'RLHFTrainer',
    'create_critic',
    'evaluate_response',
    'get_critic',
]
