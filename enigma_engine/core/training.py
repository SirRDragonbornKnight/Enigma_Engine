"""
Training Module for Enigma AI Engine

Provides:
- TrainingConfig: Configuration for training
- Trainer: Basic fine-tuning with progress callbacks
- best_of_n: Generate N responses, return best one
- collect_training_data: Run best-of-N on tasks, collect winners
- evolutionary_training: Self-play training loop

Usage:
    from enigma_engine.core.training import Trainer, TrainingConfig
    
    config = TrainingConfig(epochs=10, batch_size=4, learning_rate=1e-4)
    trainer = Trainer(model, tokenizer, config)
    trainer.train(data)
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..config import CONFIG

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training.
    
    Attributes:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        warmup_steps: Steps for learning rate warmup
        gradient_clip: Max gradient norm (0 to disable)
        save_every: Save checkpoint every N epochs
        checkpoint_dir: Directory for saving checkpoints
        eval_every: Evaluate every N steps (0 to disable)
        log_every: Log metrics every N steps
        use_amp: Use automatic mixed precision (fp16)
        max_grad_accumulation: Gradient accumulation steps
    """
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    save_every: int = 1
    checkpoint_dir: str = "models/checkpoints"
    eval_every: int = 0
    log_every: int = 10
    use_amp: bool = True
    max_grad_accumulation: int = 1
    
    # Early-stopping / safety guardrails
    early_stopping_patience: int = 0  # 0 = disabled
    max_loss: float = 100.0  # abort if loss exceeds this
    max_training_seconds: float = 0  # 0 = unlimited

    def validate(self) -> None:
        """Raise *ValueError* if any field is nonsensical."""
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.gradient_clip < 0:
            raise ValueError(f"gradient_clip must be >= 0, got {self.gradient_clip}")
        if self.max_grad_accumulation < 1:
            raise ValueError(
                f"max_grad_accumulation must be >= 1, got {self.max_grad_accumulation}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "gradient_clip": self.gradient_clip,
            "save_every": self.save_every,
            "checkpoint_dir": self.checkpoint_dir,
            "eval_every": self.eval_every,
            "log_every": self.log_every,
            "use_amp": self.use_amp,
            "max_grad_accumulation": self.max_grad_accumulation,
            "early_stopping_patience": self.early_stopping_patience,
            "max_loss": self.max_loss,
            "max_training_seconds": self.max_training_seconds,
        }


@dataclass
class TrainingState:
    """Tracks training state for checkpointing and resume."""
    epoch: int = 0
    step: int = 0
    best_loss: float = float('inf')
    total_tokens: int = 0
    training_losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)


# =============================================================================
# TRAINER CLASS
# =============================================================================

class Trainer:
    """
    Trainer for fine-tuning Enigma models.
    
    Supports:
    - Basic fine-tuning on text data
    - Q&A pair training
    - JSONL training data
    - Progress callbacks for GUI integration
    - Checkpoint saving/loading
    - Gradient accumulation
    - Mixed precision training
    
    Usage:
        trainer = Trainer(model, tokenizer, config)
        trainer.on_progress = lambda pct, msg: print(f"{pct}% - {msg}")
        trainer.train(data)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: TrainingConfig | None = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: The Enigma model to train
            tokenizer: Tokenizer for encoding text
            config: Training configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TrainingConfig()
        self.config.validate()  # fail-fast on bad config
        self.state = TrainingState()
        
        # Device
        self.device = next(model.parameters()).device
        
        # Callbacks for progress updates
        self.on_progress: Callable[[int, str], None] | None = None
        self.on_loss: Callable[[float], None] | None = None
        self.on_epoch_complete: Callable[[int, float], None] | None = None
        
        # Training control
        self._stop_requested = False
        self._lock = threading.Lock()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if self.config.use_amp and torch.cuda.is_available() else None
        
        logger.info(f"Trainer initialized: device={self.device}, config={self.config.to_dict()}")
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        # Separate weight decay for different parameter types
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        self.optimizer = AdamW([
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.config.learning_rate)
        
        # Cosine annealing scheduler
        self.scheduler = None  # Will be set when we know total steps
    
    def _emit_progress(self, percent: int, message: str) -> None:
        """Emit progress update via callback."""
        if self.on_progress:
            try:
                self.on_progress(percent, message)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")
    
    def _emit_loss(self, loss: float) -> None:
        """Emit loss update via callback."""
        if self.on_loss:
            try:
                self.on_loss(loss)
            except Exception as e:
                logger.debug(f"Loss callback error: {e}")
    
    def request_stop(self) -> None:
        """Request graceful stop of training."""
        with self._lock:
            self._stop_requested = True
        logger.info("Training stop requested")
    
    def _should_stop(self) -> bool:
        """Check if stop was requested."""
        with self._lock:
            return self._stop_requested
    
    def _parse_training_data(self, data: str | list[dict]) -> list[str]:
        """
        Parse training data into sequences.
        
        Supports:
        - Raw text (split by newlines or double newlines)
        - Q&A format: "Q: question\\nA: answer"
        - JSONL format: {"prompt": "...", "completion": "..."}
        
        Args:
            data: Raw text or list of dicts
            
        Returns:
            List of training sequences
        """
        sequences = []
        
        if isinstance(data, list):
            # Already parsed list of dicts
            for item in data:
                if isinstance(item, dict):
                    prompt = item.get("prompt", item.get("question", ""))
                    completion = item.get("completion", item.get("answer", ""))
                    if prompt and completion:
                        sequences.append(f"Q: {prompt}\nA: {completion}")
                elif isinstance(item, str):
                    sequences.append(item)
            return sequences
        
        # Raw text - detect format
        data = data.strip()
        
        # Try JSONL first
        if data.startswith('{'):
            for line in data.split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    prompt = item.get("prompt", item.get("question", ""))
                    completion = item.get("completion", item.get("answer", ""))
                    if prompt and completion:
                        sequences.append(f"Q: {prompt}\nA: {completion}")
                except json.JSONDecodeError:
                    continue
            if sequences:
                return sequences
        
        # Try Q&A format
        qa_pattern = re.compile(r'Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)', re.DOTALL)
        matches = qa_pattern.findall(data)
        if matches:
            for q, a in matches:
                sequences.append(f"Q: {q.strip()}\nA: {a.strip()}")
            return sequences
        
        # Fall back to paragraph splitting
        paragraphs = data.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if len(para) > 20:  # Skip very short paragraphs
                sequences.append(para)
        
        if not sequences:
            # Last resort: split by lines
            sequences = [line.strip() for line in data.split('\n') if line.strip()]
        
        return sequences
    
    def _create_batches(
        self, 
        sequences: list[str],
        max_length: int = 512
    ) -> list[torch.Tensor]:
        """
        Create batches from sequences.
        
        Args:
            sequences: List of text sequences
            max_length: Maximum sequence length
            
        Returns:
            List of batched tensors
        """
        # Encode all sequences
        encoded = []
        for seq in sequences:
            tokens = self.tokenizer.encode(seq)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            if len(tokens) > 1:  # Need at least 2 tokens for next-token prediction
                encoded.append(tokens)
        
        if not encoded:
            raise ValueError("No valid sequences after encoding")
        
        # Sort by length for efficient batching
        encoded.sort(key=len, reverse=True)
        
        # Create batches
        batches = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(encoded), batch_size):
            batch_tokens = encoded[i:i + batch_size]
            
            # Pad to max length in batch
            max_len = max(len(t) for t in batch_tokens)
            padded = []
            for tokens in batch_tokens:
                padding = [0] * (max_len - len(tokens))  # 0 = pad token
                padded.append(tokens + padding)
            
            batch_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)
            batches.append(batch_tensor)
        
        return batches
    
    def train(self, data: str | list[dict]) -> TrainingState:
        """
        Train the model on data.
        
        Args:
            data: Training data (text, Q&A pairs, or JSONL)
            
        Returns:
            Final training state
        """
        self._stop_requested = False
        self.model.train()
        self._training_start_time = time.monotonic()
        self._epochs_without_improvement = 0
        
        self._emit_progress(0, "Preparing training data...")
        logger.info("Starting training")
        
        # Parse data
        try:
            sequences = self._parse_training_data(data)
            logger.info(f"Parsed {len(sequences)} training sequences")
        except Exception as e:
            logger.error(f"Failed to parse training data: {e}")
            raise
        
        if not sequences:
            raise ValueError("No training sequences found in data")
        
        self._emit_progress(5, f"Creating batches from {len(sequences)} sequences...")
        
        # Create batches
        try:
            batches = self._create_batches(sequences)
            logger.info(f"Created {len(batches)} batches")
        except Exception as e:
            logger.error(f"Failed to create batches: {e}")
            raise
        
        # Setup scheduler
        total_steps = len(batches) * self.config.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
        # Warmup lambda
        def warmup_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            return 1.0
        
        # Training loop
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.config.epochs):
            if self._should_stop():
                logger.info("Training stopped by user")
                break

            # Time-limit guard
            if (
                self.config.max_training_seconds > 0
                and time.monotonic() - self._training_start_time
                > self.config.max_training_seconds
            ):
                logger.info("Training stopped: time limit reached")
                break
            
            epoch_loss = 0.0
            epoch_tokens = 0
            
            progress_base = int(5 + (epoch / self.config.epochs) * 90)
            self._emit_progress(progress_base, f"Epoch {epoch + 1}/{self.config.epochs}")
            
            # Shuffle batches each epoch
            random.shuffle(batches)
            
            for batch_idx, batch in enumerate(batches):
                if self._should_stop():
                    break
                
                # Forward pass
                with torch.amp.autocast('cuda', enabled=self.config.use_amp and torch.cuda.is_available()):
                    # Input is all but last token, target is all but first token
                    input_ids = batch[:, :-1]
                    targets = batch[:, 1:]
                    
                    logits, loss = self.model(input_ids, targets=targets)
                    
                    # Add MoE auxiliary loss if available
                    if hasattr(self.model, 'get_moe_aux_loss'):
                        aux_loss = self.model.get_moe_aux_loss()
                        loss = loss + aux_loss * 0.01
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.max_grad_accumulation
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.max_grad_accumulation == 0:
                    # Gradient clipping
                    if self.config.gradient_clip > 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.gradient_clip
                        )
                    
                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    # Warmup + scheduler step
                    warmup_factor = warmup_lambda(self.state.step)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.config.learning_rate * warmup_factor
                    if self.state.step >= self.config.warmup_steps:
                        self.scheduler.step()
                
                # Track metrics
                batch_loss = loss.item() * self.config.max_grad_accumulation

                # Loss explosion guard
                if math.isnan(batch_loss) or math.isinf(batch_loss):
                    logger.error("Training aborted: NaN/Inf loss detected")
                    self.model.eval()
                    return self.state
                if batch_loss > self.config.max_loss:
                    logger.error(
                        f"Training aborted: loss {batch_loss:.4f} exceeded "
                        f"max_loss {self.config.max_loss}"
                    )
                    self.model.eval()
                    return self.state

                batch_tokens = batch.numel()
                epoch_loss += batch_loss * batch_tokens
                epoch_tokens += batch_tokens
                self.state.step += 1
                self.state.total_tokens += batch_tokens
                
                # Log periodically
                if self.state.step % self.config.log_every == 0:
                    avg_loss = epoch_loss / max(1, epoch_tokens)
                    logger.debug(f"Step {self.state.step}: loss={avg_loss:.4f}")
                    self._emit_loss(avg_loss)
                
                # Update progress
                batch_progress = int(progress_base + (batch_idx / len(batches)) * (90 / self.config.epochs))
                self._emit_progress(batch_progress, f"Epoch {epoch + 1}: Batch {batch_idx + 1}/{len(batches)}")
            
            # Epoch complete
            avg_epoch_loss = epoch_loss / max(1, epoch_tokens)
            self.state.training_losses.append(avg_epoch_loss)
            self.state.epoch = epoch + 1
            
            logger.info(f"Epoch {epoch + 1} complete: loss={avg_epoch_loss:.4f}")
            
            if self.on_epoch_complete:
                self.on_epoch_complete(epoch + 1, avg_epoch_loss)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt")
            
            # Track best loss + early stopping
            if avg_epoch_loss < self.state.best_loss:
                self.state.best_loss = avg_epoch_loss
                self._epochs_without_improvement = 0
                self._save_checkpoint(checkpoint_dir / "best_model.pt")
            else:
                self._epochs_without_improvement += 1
                if (
                    self.config.early_stopping_patience > 0
                    and self._epochs_without_improvement
                    >= self.config.early_stopping_patience
                ):
                    logger.info(
                        f"Early stopping: no improvement for "
                        f"{self._epochs_without_improvement} epochs"
                    )
                    break
        
        # Final save
        self._emit_progress(95, "Saving final model...")
        self._save_checkpoint(checkpoint_dir / "final_model.pt")
        
        # Restore model to eval mode for inference
        self.model.eval()
        
        self._emit_progress(100, "Training complete!")
        logger.info(f"Training complete: {self.state.epoch} epochs, best_loss={self.state.best_loss:.4f}")
        
        return self.state
    
    def _save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_state': {
                    'epoch': self.state.epoch,
                    'step': self.state.step,
                    'best_loss': self.state.best_loss,
                    'total_tokens': self.state.total_tokens,
                },
                'config': self.config.to_dict(),
            }
            if hasattr(self.model, 'config'):
                checkpoint['model_config'] = self.model.config.__dict__
            
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint: {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            state = checkpoint.get('training_state', {})
            self.state.epoch = state.get('epoch', 0)
            self.state.step = state.get('step', 0)
            self.state.best_loss = state.get('best_loss', float('inf'))
            self.state.total_tokens = state.get('total_tokens', 0)
            
            logger.info(f"Loaded checkpoint: {path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise


# =============================================================================
# EVOLUTIONARY TRAINING (BEST-OF-N SAMPLING)
# =============================================================================

def score_response(
    prompt: str, 
    response: str,
    method: str = "length_quality"
) -> float:
    """
    Score a response for quality.
    
    Scoring methods:
    - "length_quality": Balance between length and coherence
    - "consistency": Compare with other responses
    - "perplexity": Model's own confidence
    - "rule_based": Check for patterns
    
    Args:
        prompt: Original prompt
        response: Generated response
        method: Scoring method
        
    Returns:
        Score from 0-100
    """
    score = 50.0  # Base score
    
    if method == "length_quality":
        # Score based on response characteristics
        
        # Length scoring (prefer medium length)
        length = len(response)
        if length < 10:
            score -= 20
        elif length < 50:
            score -= 10
        elif length < 200:
            score += 10
        elif length < 500:
            score += 20
        elif length < 1000:
            score += 15
        else:
            score += 5  # Very long responses slightly less preferred
        
        # Coherence checks
        words = response.split()
        if len(words) > 5:
            # Check for repetition
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                score -= 30  # Heavy repetition
            elif unique_ratio < 0.5:
                score -= 15
            elif unique_ratio > 0.7:
                score += 10
        
        # Punctuation check
        if any(p in response for p in '.!?'):
            score += 5  # Has sentence endings
        
        # Check for code patterns (if prompt asks for code)
        if 'code' in prompt.lower() or 'function' in prompt.lower():
            if 'def ' in response or 'function' in response or '()' in response:
                score += 15
        
        # Check for explanation pattern
        if 'explain' in prompt.lower() or 'what is' in prompt.lower():
            if len(response) > 100:
                score += 10
    
    elif method == "rule_based":
        # Simple rule-based scoring
        
        # Positive patterns
        if response.strip():
            score += 10
        if len(response) > 50:
            score += 10
        if '\n' in response:  # Has structure
            score += 5
        
        # Negative patterns
        if response.count(response.split()[0] if response.split() else '') > 5:
            score -= 20  # Repetitive
        if 'error' in response.lower() or 'sorry' in response.lower():
            score -= 10
    
    return max(0, min(100, score))


def best_of_n(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    n: int = 5,
    temperature_range: tuple[float, float] = (0.5, 1.0),
    max_tokens: int = 256,
    scoring_method: str = "length_quality"
) -> tuple[str, float]:
    """
    Generate N responses and return the best one.
    
    This is the core of evolutionary training:
    - Generate multiple responses with varied settings
    - Score each response
    - Return the winner
    
    Args:
        model: The model to generate from
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt
        n: Number of responses to generate
        temperature_range: Range for random temperature
        max_tokens: Maximum tokens to generate
        scoring_method: Method for scoring responses
        
    Returns:
        Tuple of (best_response, score)
    """
    model.eval()
    device = next(model.parameters()).device
    
    responses = []
    
    for i in range(n):
        # Vary temperature for diversity
        temperature = random.uniform(*temperature_range)
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
            )
        
        # Get only the new tokens (skip the input)
        if hasattr(output_ids, 'squeeze'):
            output_ids = output_ids.squeeze(0)
        new_tokens = output_ids[len(input_ids):].tolist()
        response = tokenizer.decode(new_tokens)
        
        # Score response
        score = score_response(prompt, response, method=scoring_method)
        responses.append((response, score, temperature))
        
        logger.debug(f"Response {i+1}/{n}: score={score:.1f}, temp={temperature:.2f}")
    
    # Return best
    best = max(responses, key=lambda x: x[1])
    logger.info(f"Best of {n}: score={best[1]:.1f}, temp={best[2]:.2f}")
    
    return best[0], best[1]


# =============================================================================
# MULTI-INSTANCE (PARALLEL GENERATION)
# =============================================================================

def _generate_single_response(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    temperature: float,
    max_tokens: int,
    device: torch.device,
    scoring_method: str
) -> tuple[str, float, float]:
    """
    Generate a single response (for use in parallel execution).
    
    Args:
        model: Model to generate from
        tokenizer: Tokenizer
        prompt: Input prompt
        temperature: Generation temperature
        max_tokens: Max tokens to generate
        device: Device to run on
        scoring_method: Scoring method
        
    Returns:
        Tuple of (response, score, temperature)
    """
    try:
        # Encode prompt
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
            )
        
        # Decode
        if hasattr(output_ids, 'squeeze'):
            output_ids = output_ids.squeeze(0)
        new_tokens = output_ids[len(input_ids):].tolist()
        response = tokenizer.decode(new_tokens)
        
        # Score
        score = score_response(prompt, response, method=scoring_method)
        
        return (response, score, temperature)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return ("", 0.0, temperature)


def parallel_best_of_n(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    n: int = 5,
    max_workers: int = 4,
    temperature_range: tuple[float, float] = (0.5, 1.0),
    max_tokens: int = 256,
    scoring_method: str = "length_quality"
) -> tuple[str, float]:
    """
    Generate N responses in parallel and return the best one.
    
    Uses ThreadPoolExecutor for parallel generation. Note that due to
    Python's GIL, this provides speedup mainly for I/O-bound operations.
    For GPU inference, the actual speedup may be limited, but this
    enables running multiple inference processes.
    
    Args:
        model: Model to generate from
        tokenizer: Tokenizer
        prompt: Input prompt
        n: Number of responses to generate
        max_workers: Maximum parallel workers
        temperature_range: Range for random temperature
        max_tokens: Max tokens to generate
        scoring_method: Scoring method
        
    Returns:
        Tuple of (best_response, score)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    model.eval()
    device = next(model.parameters()).device
    
    # Create varied temperatures for each instance
    temperatures = [random.uniform(*temperature_range) for _ in range(n)]
    
    responses = []
    
    # Use thread pool for parallel generation
    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as executor:
        futures = {}
        for i, temp in enumerate(temperatures):
            future = executor.submit(
                _generate_single_response,
                model, tokenizer, prompt, temp, max_tokens, device, scoring_method
            )
            futures[future] = i
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                responses.append(result)
                logger.debug(f"Instance {idx+1}/{n}: score={result[1]:.1f}, temp={result[2]:.2f}")
            except Exception as e:
                logger.error(f"Instance {idx+1} failed: {e}")
    
    if not responses:
        return "", 0.0
    
    # Return best
    best = max(responses, key=lambda x: x[1])
    logger.info(f"Parallel best of {n}: score={best[1]:.1f}, temp={best[2]:.2f}")
    
    return best[0], best[1]


def multi_instance_collect(
    model: nn.Module,
    tokenizer: Any,
    tasks: list[str],
    n_per_task: int = 5,
    max_workers: int = 4,
    score_threshold: float = 60.0,
    on_progress: Callable[[int, str], None] | None = None
) -> list[dict]:
    """
    Collect training data using parallel generation.
    
    Args:
        model: Model to generate from
        tokenizer: Tokenizer
        tasks: List of prompts/tasks
        n_per_task: Responses per task
        max_workers: Parallel workers
        score_threshold: Minimum score to keep
        on_progress: Progress callback
        
    Returns:
        List of training examples
    """
    training_data = []
    
    for i, task in enumerate(tasks):
        if on_progress:
            on_progress(int(i / len(tasks) * 100), f"Task {i+1}/{len(tasks)}")
        
        try:
            best_response, score = parallel_best_of_n(
                model, tokenizer, task,
                n=n_per_task,
                max_workers=max_workers
            )
            
            if score >= score_threshold:
                training_data.append({
                    "prompt": task,
                    "completion": best_response,
                    "score": score
                })
                logger.info(f"Task {i+1}: score={score:.1f} - KEPT")
            else:
                logger.info(f"Task {i+1}: score={score:.1f} - BELOW THRESHOLD")
        
        except Exception as e:
            logger.error(f"Task {i+1} failed: {e}")
    
    logger.info(f"Collected {len(training_data)} examples from {len(tasks)} tasks (parallel)")
    return training_data


def collect_training_data(
    model: nn.Module,
    tokenizer: Any,
    tasks: list[str],
    n_per_task: int = 5,
    score_threshold: float = 60.0,
    on_progress: Callable[[int, str], None] | None = None
) -> list[dict]:
    """
    Generate training data by running best-of-N on tasks.
    
    Args:
        model: Model to generate from
        tokenizer: Tokenizer
        tasks: List of prompts/tasks
        n_per_task: Number of generations per task
        score_threshold: Minimum score to include
        on_progress: Progress callback
        
    Returns:
        List of training examples: [{"prompt": str, "completion": str, "score": float}]
    """
    training_data = []
    
    for i, task in enumerate(tasks):
        if on_progress:
            on_progress(int(i / len(tasks) * 100), f"Task {i+1}/{len(tasks)}")
        
        try:
            best_response, score = best_of_n(
                model, tokenizer, task, 
                n=n_per_task
            )
            
            if score >= score_threshold:
                training_data.append({
                    "prompt": task,
                    "completion": best_response,
                    "score": score
                })
                logger.info(f"Task {i+1}: score={score:.1f} - KEPT")
            else:
                logger.info(f"Task {i+1}: score={score:.1f} - BELOW THRESHOLD")
        
        except Exception as e:
            logger.error(f"Task {i+1} failed: {e}")
    
    logger.info(f"Collected {len(training_data)} training examples from {len(tasks)} tasks")
    return training_data


def evolutionary_training(
    model: nn.Module,
    tokenizer: Any,
    tasks: list[str],
    generations: int = 10,
    n_per_task: int = 5,
    training_config: TrainingConfig | None = None,
    checkpoint_dir: str = "models/evolutionary",
    on_progress: Callable[[int, str], None] | None = None
) -> nn.Module:
    """
    Train model through evolutionary selection (self-play).
    
    The loop:
    1. Generate N responses per task
    2. Score responses
    3. Keep only the best response per task
    4. Fine-tune model on winning outputs
    5. Repeat with improved model
    
    Args:
        model: Initial model
        tokenizer: Tokenizer
        tasks: Training tasks/prompts
        generations: Number of evolutionary generations
        n_per_task: Responses to generate per task
        training_config: Config for fine-tuning step
        checkpoint_dir: Where to save generation checkpoints
        on_progress: Progress callback
        
    Returns:
        Trained model
    """
    config = training_config or TrainingConfig(epochs=1, save_every=1)
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting evolutionary training: {generations} generations, {len(tasks)} tasks")
    
    for gen in range(generations):
        gen_progress = int(gen / generations * 100)
        if on_progress:
            on_progress(gen_progress, f"Generation {gen + 1}/{generations}")
        
        logger.info(f"=== Generation {gen + 1} ===")
        
        # Collect training data from best-of-N
        training_data = collect_training_data(
            model, tokenizer, tasks,
            n_per_task=n_per_task,
            on_progress=lambda p, m: on_progress(gen_progress + int(p * 0.5 / generations), m) if on_progress else None
        )
        
        if not training_data:
            logger.warning(f"Generation {gen + 1}: No training data collected, skipping")
            continue
        
        # Prepare training text
        train_text = "\n\n".join([
            f"Q: {ex['prompt']}\nA: {ex['completion']}"
            for ex in training_data
        ])
        
        # Fine-tune on winning outputs
        trainer = Trainer(model, tokenizer, config)
        trainer.config.checkpoint_dir = str(checkpoint_path / f"gen_{gen + 1}")
        
        if on_progress:
            trainer.on_progress = lambda p, m: on_progress(
                gen_progress + 50 + int(p * 0.5 / generations), 
                f"Gen {gen + 1}: {m}"
            )
        
        trainer.train(train_text)
        
        # Save generation checkpoint
        gen_checkpoint = checkpoint_path / f"generation_{gen + 1}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'generation': gen + 1,
            'training_data_count': len(training_data),
            'avg_score': sum(ex['score'] for ex in training_data) / len(training_data)
        }, gen_checkpoint)
        
        logger.info(f"Generation {gen + 1} complete: {len(training_data)} examples, saved to {gen_checkpoint}")
    
    # Save final model
    final_path = checkpoint_path / "final_evolved_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'total_generations': generations,
    }, final_path)
    
    logger.info(f"Evolutionary training complete: {generations} generations")
    if on_progress:
        on_progress(100, "Evolutionary training complete!")
    
    return model


def save_training_data(
    data: list[dict],
    path: str | Path,
    format: str = "jsonl"
) -> None:
    """
    Save collected training data to file.
    
    Args:
        data: List of training examples
        path: Output file path
        format: "jsonl" or "txt"
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(f"Q: {item['prompt']}\n")
                f.write(f"A: {item['completion']}\n\n")
    
    logger.info(f"Saved {len(data)} training examples to {path}")


def load_training_data(path: str | Path) -> list[dict]:
    """
    Load training data from file.
    
    Args:
        path: Input file path (jsonl or txt)
        
    Returns:
        List of training examples
    """
    path = Path(path)
    data = []
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try JSONL
    if path.suffix == '.jsonl' or content.strip().startswith('{'):
        for line in content.split('\n'):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if data:
            return data
    
    # Try Q&A format
    qa_pattern = re.compile(r'Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)', re.DOTALL)
    matches = qa_pattern.findall(content)
    for q, a in matches:
        data.append({
            "prompt": q.strip(),
            "completion": a.strip()
        })
    
    logger.info(f"Loaded {len(data)} training examples from {path}")
    return data
