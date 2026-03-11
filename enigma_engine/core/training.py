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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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
        save_every: Save checkpoint every N epochs (0 = disabled)
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
    save_every: int = 0
    checkpoint_dir: str = "models/checkpoints"
    eval_every: int = 0
    log_every: int = 10
    use_amp: bool = True
    max_grad_accumulation: int = 1
    use_gradient_checkpointing: bool = False

    # Reasoning-weighted loss (CoT-E)
    # Weight multiplier for tokens inside <think>...</think> blocks.
    # 1.0 = normal (no extra weight), 2.0 = double weight on reasoning tokens.
    reasoning_loss_weight: float = 1.0

    # Rolling best checkpoints (CK-C)
    rolling_best_k: int = 0  # 0 = disabled, N = keep N best by loss

    # Early-stopping / safety guardrails
    early_stopping_patience: int = 0  # 0 = disabled
    max_loss: float = 100.0  # abort if loss exceeds this
    max_training_seconds: float = 0  # 0 = unlimited

    # Before/after evaluation (EV-C)
    run_evaluation: bool = False  # Evaluate before and after training
    eval_test_prompts: list[str] = None  # Custom test prompts (None = use defaults)

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
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "rolling_best_k": self.rolling_best_k,
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
# DATA VALIDATION (DQ-B)
# =============================================================================

@dataclass
class DataValidationResult:
    """Result of validating training data before use.

    Attributes:
        is_valid: True if data is usable (may still have warnings).
        total_sequences: Total number of sequences found.
        warnings: Non-fatal issues (short sequences, duplicates, etc.).
        errors: Fatal issues that block training.
        stats: Summary statistics about the data.
    """
    is_valid: bool = True
    total_sequences: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def validate_training_data(
    data: str,
    min_length: int = 5,
    max_length: int = 50000,
    warn_duplicate_ratio: float = 0.3,
) -> DataValidationResult:
    """Validate training data before training begins (DQ-B).

    Checks for:
    - Empty or whitespace-only data
    - Encoding issues (null bytes, control characters)
    - Very short sequences (< min_length chars)
    - Very long sequences (> max_length chars)
    - Duplicate sequences
    - Empty lines that produce no training signal

    Args:
        data: Raw training data text.
        min_length: Minimum chars per sequence to be useful.
        max_length: Maximum chars per sequence before truncation warning.
        warn_duplicate_ratio: Warn if duplicates exceed this fraction.

    Returns:
        DataValidationResult with is_valid, warnings, errors, and stats.
    """
    result = DataValidationResult()

    # Check for empty/whitespace data
    if not data or not data.strip():
        result.is_valid = False
        result.errors.append("Training data is empty or whitespace-only.")
        return result

    # Check for encoding issues (null bytes, stray control chars)
    null_count = data.count("\x00")
    if null_count > 0:
        result.warnings.append(
            f"Data contains {null_count} null bytes — "
            f"may indicate encoding corruption.")

    # Count non-printable control characters (exclude newline, tab, CR)
    ctrl_count = sum(
        1 for ch in data
        if ord(ch) < 32 and ch not in ("\n", "\r", "\t"))
    if ctrl_count > 0:
        result.warnings.append(
            f"Data contains {ctrl_count} control characters.")

    # Split into sequences (paragraphs or double-newline separated)
    # Use the same logic as _parse_training_data for consistency
    lines = [ln.strip() for ln in data.split("\n") if ln.strip()]

    if not lines:
        result.is_valid = False
        result.errors.append("No non-empty lines found in data.")
        return result

    result.total_sequences = len(lines)

    # Check individual sequence quality
    short_count = 0
    long_count = 0
    empty_count = data.count("\n\n\n")  # triple+ newlines = wasted space
    lengths: list[int] = []

    for line in lines:
        line_len = len(line)
        lengths.append(line_len)
        if line_len < min_length:
            short_count += 1
        if line_len > max_length:
            long_count += 1

    if short_count > 0:
        pct = short_count / len(lines) * 100
        result.warnings.append(
            f"{short_count} sequences ({pct:.0f}%) shorter than "
            f"{min_length} chars — may not provide useful signal.")

    if long_count > 0:
        result.warnings.append(
            f"{long_count} sequences exceed {max_length:,} chars — "
            f"will be truncated to model max_seq_len.")

    if empty_count > 5:
        result.warnings.append(
            f"{empty_count} runs of 3+ empty lines — "
            f"consider cleaning whitespace.")

    # Check for duplicates
    unique_lines = set(lines)
    dup_count = len(lines) - len(unique_lines)
    if dup_count > 0:
        dup_ratio = dup_count / len(lines)
        msg = (f"{dup_count} duplicate sequences "
               f"({dup_ratio:.0%} of data).")
        if dup_ratio >= warn_duplicate_ratio:
            result.warnings.append(msg + " High duplication may "
                                   "cause over-fitting.")
        else:
            result.warnings.append(msg)

    # Compute stats
    result.stats = {
        "total_chars": len(data),
        "total_lines": len(lines),
        "unique_lines": len(unique_lines),
        "duplicates": dup_count,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "short_sequences": short_count,
        "long_sequences": long_count,
    }

    # Data is valid if no errors (warnings are informational)
    result.is_valid = len(result.errors) == 0

    return result


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

        # Gradient checkpointing — trades compute for VRAM savings
        if self.config.use_gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            else:
                # Manual: enable on each transformer layer
                for layer in getattr(self.model, "layers", []):
                    if hasattr(layer, "gradient_checkpointing"):
                        layer.gradient_checkpointing = True

        # Resolve pad token ID from the tokenizer — used for
        # padding batches and as ignore_index in cross-entropy loss.
        # Built-in tokenizer uses 0, tiktoken uses base+0.
        self.pad_token_id: int = getattr(
            self.tokenizer, "pad_token_id", 0)

        # Rolling best checkpoint tracking (CK-C)
        # List of (loss, path) tuples sorted by loss ascending
        self._rolling_checkpoints: list[tuple[float, Path]] = []

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
                    thinking = item.get("thinking", item.get("reasoning", ""))
                    if prompt and completion:
                        # Wrap thinking in <think> tags if provided
                        if thinking:
                            from .reasoning import wrap_reasoning
                            completion = wrap_reasoning(thinking, completion)
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
                    thinking = item.get("thinking", item.get("reasoning", ""))
                    if prompt and completion:
                        if thinking:
                            from .reasoning import wrap_reasoning
                            completion = wrap_reasoning(thinking, completion)
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

        # Try User/AI dialogue format
        dialogue_pattern = re.compile(
            r'(?:User|Human):\s*(.+?)\s*(?:AI|Assistant):\s*(.+?)(?=(?:User|Human):|$)',
            re.DOTALL | re.IGNORECASE)
        d_matches = dialogue_pattern.findall(data)
        if d_matches:
            for user_msg, ai_msg in d_matches:
                sequences.append(
                    f"User: {user_msg.strip()}\nAI: {ai_msg.strip()}")
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

    def _apply_reasoning_weight(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        base_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Re-compute loss with higher weight on ``<think>`` tokens (CoT-E).

        Identifies tokens that fall inside ``<think>...</think>`` spans
        and multiplies their per-token loss by ``config.reasoning_loss_weight``.
        This forces the model to learn the reasoning *process*, not just
        memorise final answers.

        Falls back to *base_loss* if the reasoning token IDs can't be
        resolved or no think tokens exist in the batch.
        """
        try:
            # Resolve token IDs for <think> and </think>
            think_start_ids = self._get_token_ids("<think>")
            think_end_ids = self._get_token_ids("</think>")

            if not think_start_ids or not think_end_ids:
                return base_loss

            # Per-token cross-entropy (not reduced)
            per_token_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.pad_token_id,
                reduction="none",
            ).reshape(targets.shape)

            # Build weight mask: 1.0 everywhere, reasoning_loss_weight
            # inside <think>...</think> spans
            weight = torch.ones_like(per_token_loss)
            w = self.config.reasoning_loss_weight

            for b in range(targets.size(0)):
                in_think = False
                for t in range(targets.size(1)):
                    tok = targets[b, t].item()
                    if tok in think_start_ids:
                        in_think = True
                    if in_think:
                        weight[b, t] = w
                    if tok in think_end_ids:
                        in_think = False

            # Weighted mean (ignoring padding)
            valid = (targets != self.pad_token_id).float()
            weighted = per_token_loss * weight * valid
            denom = (weight * valid).sum()
            if denom > 0:
                return weighted.sum() / denom
            return base_loss

        except Exception:
            logger.debug("Reasoning-weighted loss failed, using base loss",
                         exc_info=True)
            return base_loss

    def _get_token_ids(self, text: str) -> set[int]:
        """Encode *text* and return the set of resulting token IDs."""
        try:
            if hasattr(self.tokenizer, "encode"):
                ids = self.tokenizer.encode(text)
                if isinstance(ids, list):
                    return set(ids)
            return set()
        except Exception:
            return set()

    def _create_batches(
        self,
        sequences: list[str],
        max_length: int | None = None,
    ) -> list[torch.Tensor]:
        """
        Create batches from sequences.

        Args:
            sequences: List of text sequences
            max_length: Maximum sequence length.  Defaults to the
                model's ``max_seq_len`` if available, otherwise 512.

        Returns:
            List of batched tensors
        """
        # Use model's configured context length when available
        if max_length is None:
            cfg = getattr(self.model, "config", None)
            max_length = getattr(cfg, "max_seq_len", 512)

        pad_token_id = self.pad_token_id

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
                padding = [pad_token_id] * (max_len - len(tokens))
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

        # Deduplicate while preserving order — prevents the model
        # from over-fitting on repeated examples.
        pre_dedup = len(sequences)
        sequences = list(dict.fromkeys(sequences))
        if len(sequences) < pre_dedup:
            logger.info(
                f"Removed {pre_dedup - len(sequences)} duplicate "
                f"sequences ({len(sequences)} unique remain)")

        # Run before-training evaluation (EV-C)
        before_eval = None
        if self.config.run_evaluation:
            self._emit_progress(3, "Evaluating model (before training)...")
            try:
                from enigma_engine.core.training_evaluation import (
                    evaluate_model, DEFAULT_TEST_PROMPTS,
                )
                test_prompts = (
                    self.config.eval_test_prompts or DEFAULT_TEST_PROMPTS
                )
                device = next(self.model.parameters()).device
                before_eval = evaluate_model(
                    self.model, self.tokenizer, test_prompts, str(device)
                )
                logger.info(
                    f"Before training: perplexity={before_eval['perplexity']:.2f}, "
                    f"loss={before_eval['loss']:.4f}"
                )
            except Exception as exc:
                logger.warning(f"Before-training evaluation failed: {exc}")

        self._emit_progress(5, f"Creating batches from {len(sequences)} sequences...")

        # Use model's max_seq_len for batch creation
        cfg = getattr(self.model, "config", None)
        max_seq_len = getattr(cfg, "max_seq_len", 512)

        # Create batches
        try:
            batches = self._create_batches(
                sequences, max_length=max_seq_len)
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

                try:
                    batch_loss = self._train_one_batch(batch, batch_idx)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    # OOM recovery: clear cache, enable gradient
                    # checkpointing, and retry the batch once
                    self._handle_oom(exc)
                    try:
                        batch_loss = self._train_one_batch(
                            batch, batch_idx)
                    except RuntimeError as exc2:
                        if "out of memory" not in str(exc2).lower():
                            raise
                        logger.error(
                            "Training aborted: OOM persists after "
                            "enabling gradient checkpointing. "
                            "Reduce batch_size or max_seq_len."
                        )
                        self._emit_progress(
                            0, "OOM: reduce batch size or sequence length")
                        self.model.eval()
                        return self.state

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

            # Save periodic checkpoint (0 = disabled)
            if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
                ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                self._save_checkpoint(ckpt_path)
                self._cleanup_periodic_checkpoints(
                    checkpoint_dir, "checkpoint_epoch_", keep=3)

            # Track best loss + early stopping
            if avg_epoch_loss < self.state.best_loss:
                self.state.best_loss = avg_epoch_loss
                self._epochs_without_improvement = 0
                self._save_checkpoint(checkpoint_dir / "best_model.pt")
                # Rolling best checkpoints (CK-C)
                if self.config.rolling_best_k > 0:
                    self._save_rolling_checkpoint(
                        checkpoint_dir, avg_epoch_loss, epoch + 1)
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

        # Run after-training evaluation (EV-C)
        after_eval = None
        if self.config.run_evaluation:
            self._emit_progress(97, "Evaluating model (after training)...")
            try:
                from enigma_engine.core.training_evaluation import (
                    evaluate_model, DEFAULT_TEST_PROMPTS,
                )
                test_prompts = (
                    self.config.eval_test_prompts or DEFAULT_TEST_PROMPTS
                )
                device = next(self.model.parameters()).device
                after_eval = evaluate_model(
                    self.model, self.tokenizer, test_prompts, str(device)
                )
                logger.info(
                    f"After training: perplexity={after_eval['perplexity']:.2f}, "
                    f"loss={after_eval['loss']:.4f}"
                )
                if before_eval:
                    ppl_improvement = before_eval["perplexity"] - after_eval["perplexity"]
                    logger.info(
                        f"Perplexity improvement: {ppl_improvement:.2f} "
                        f"({ppl_improvement / before_eval['perplexity'] * 100:.1f}%)"
                    )
            except Exception as exc:
                logger.warning(f"After-training evaluation failed: {exc}")

        # Store evaluation results in state for later retrieval
        if before_eval:
            self.state.before_eval = before_eval  # type: ignore
        if after_eval:
            self.state.after_eval = after_eval  # type: ignore

        # Restore model to eval mode for inference
        self.model.eval()

        self._emit_progress(100, "Training complete!")
        logger.info(f"Training complete: {self.state.epoch} epochs, best_loss={self.state.best_loss:.4f}")

        return self.state

    def _train_one_batch(self, batch: torch.Tensor, batch_idx: int) -> float:
        """Run forward + backward on a single batch.

        Returns the unscaled batch loss (float).  Raises RuntimeError
        on CUDA OOM so the caller can handle recovery.
        """
        # Forward pass
        with torch.amp.autocast(
            'cuda',
            enabled=self.config.use_amp and torch.cuda.is_available(),
        ):
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            logits, loss = self.model(
                input_ids, targets=targets,
                pad_token_id=self.pad_token_id)

            # Reasoning-weighted loss (CoT-E)
            if (
                self.config.reasoning_loss_weight > 1.0
                and loss is not None
            ):
                loss = self._apply_reasoning_weight(
                    logits, targets, loss)

            # MoE auxiliary loss
            if hasattr(self.model, 'get_moe_aux_loss'):
                aux_loss = self.model.get_moe_aux_loss()
                loss = loss + aux_loss * 0.01

            # Scale for gradient accumulation
            loss = loss / self.config.max_grad_accumulation

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation step
        if (batch_idx + 1) % self.config.max_grad_accumulation == 0:
            if self.config.gradient_clip > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip)

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            # Warmup + scheduler step
            warmup_factor = (
                self.state.step / max(1, self.config.warmup_steps)
                if self.state.step < self.config.warmup_steps
                else 1.0
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = (
                    self.config.learning_rate * warmup_factor)
            if self.state.step >= self.config.warmup_steps:
                self.scheduler.step()

        return loss.item() * self.config.max_grad_accumulation

    def _handle_oom(self, exc: RuntimeError) -> None:
        """Handle CUDA OOM: clear cache, enable gradient checkpointing.

        Called on the first OOM per training run.  Enables gradient
        checkpointing (trades compute for VRAM) and clears the CUDA
        cache so the retry has the best chance of succeeding.
        """
        logger.warning(
            "CUDA out of memory — clearing cache and enabling "
            "gradient checkpointing for retry: %s", exc)
        self.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Enable gradient checkpointing if not already on
        if not self.config.use_gradient_checkpointing:
            self.config.use_gradient_checkpointing = True
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            else:
                for layer in getattr(self.model, "layers", []):
                    if hasattr(layer, "gradient_checkpointing"):
                        layer.gradient_checkpointing = True
            logger.info(
                "Gradient checkpointing enabled to reduce VRAM usage")

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
                'training_config': self.config.to_dict(),
            }
            if hasattr(self.model, 'config'):
                checkpoint['model_config'] = self.model.config.__dict__
                # Also save as 'config' for compatibility with gui_forge loader
                checkpoint['config'] = self.model.config.__dict__

            from enigma_engine.core.safe_save import atomic_torch_save
            atomic_torch_save(checkpoint, path)
            logger.info(f"Saved checkpoint: {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _save_rolling_checkpoint(
        self, checkpoint_dir: Path, loss: float, epoch: int
    ) -> None:
        """Save a rolling best checkpoint (CK-C).

        Keeps only the K best checkpoints by loss.  When a new
        checkpoint is better than the worst in the rolling set,
        the worst is deleted.  This prevents disk bloat during
        long training runs.

        Args:
            checkpoint_dir: Directory where checkpoints are saved.
            loss: The loss value for this checkpoint.
            epoch: The epoch number for filename.
        """
        k = self.config.rolling_best_k
        if k <= 0:
            return

        path = checkpoint_dir / f"rolling_best_e{epoch}_loss{loss:.4f}.pt"

        # If we haven't filled K slots yet, just save
        if len(self._rolling_checkpoints) < k:
            self._save_checkpoint(path)
            self._rolling_checkpoints.append((loss, path))
            self._rolling_checkpoints.sort(key=lambda x: x[0])
            logger.info(
                "Rolling checkpoint saved: %s (%d/%d slots)",
                path.name, len(self._rolling_checkpoints), k)
            return

        # Check if this is better than the worst
        worst_loss, worst_path = self._rolling_checkpoints[-1]
        if loss < worst_loss:
            # Delete the worst checkpoint
            try:
                if worst_path.exists():
                    worst_path.unlink()
                    logger.info(
                        "Rolling checkpoint deleted (worst): %s",
                        worst_path.name)
            except OSError as exc:
                logger.debug(
                    "Could not delete old rolling checkpoint: %s", exc)

            # Remove worst from tracking and add new
            self._rolling_checkpoints.pop()
            self._save_checkpoint(path)
            self._rolling_checkpoints.append((loss, path))
            self._rolling_checkpoints.sort(key=lambda x: x[0])
            logger.info(
                "Rolling checkpoint replaced: %s (loss=%.4f)",
                path.name, loss)

    @staticmethod
    def _cleanup_periodic_checkpoints(
        checkpoint_dir: Path, prefix: str, keep: int = 3
    ) -> None:
        """Delete old periodic checkpoints, keeping only the most recent *keep*.

        Matches files like ``checkpoint_epoch_5.pt`` or ``vision_epoch_10.pt``
        inside *checkpoint_dir*.  Files are sorted by the epoch number embedded
        in the filename; only the highest *keep* are retained.
        """
        import re
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.pt$")
        found: list[tuple[int, Path]] = []
        try:
            for f in checkpoint_dir.iterdir():
                m = pattern.match(f.name)
                if m:
                    found.append((int(m.group(1)), f))
        except OSError:
            return

        if len(found) <= keep:
            return

        # Sort by epoch number descending — keep the newest
        found.sort(key=lambda x: x[0], reverse=True)
        for _epoch_num, old_path in found[keep:]:
            try:
                old_path.unlink()
                logger.info("Deleted old periodic checkpoint: %s", old_path.name)
            except OSError as exc:
                logger.debug("Could not delete old checkpoint %s: %s", old_path.name, exc)

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        try:
            from enigma_engine.core.model_registry import safe_load_weights
            checkpoint = safe_load_weights(path, map_location=self.device)

            # Unwrap state dict — handles both flat and wrapped formats
            state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint.get('model')
            if state_dict is None:
                # Assume the whole checkpoint is a bare state dict
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)

            opt_state = checkpoint.get('optimizer_state_dict')
            if opt_state:
                self.optimizer.load_state_dict(opt_state)

            state = checkpoint.get('training_state', {})
            self.state.epoch = state.get('epoch', 0)
            self.state.step = state.get('step', 0)
            self.state.best_loss = state.get('best_loss', float('inf'))
            self.state.total_tokens = state.get('total_tokens', 0)

            logger.info(f"Loaded checkpoint: {path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    # -----------------------------------------------------------------
    # DPO — Direct Preference Optimization
    # -----------------------------------------------------------------

    @staticmethod
    def _dpo_loss(
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        ref_chosen_logps: "torch.Tensor",
        ref_rejected_logps: "torch.Tensor",
        beta: float = 0.1,
    ) -> "torch.Tensor":
        """Compute the DPO loss (Rafailov et al., 2023).

        loss = -log(sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x)
                                - log pi(y_l|x)/pi_ref(y_l|x))))
        """
        import torch.nn.functional as F  # noqa: N812

        chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
        return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    def _get_sequence_logps(
        self, model: "nn.Module", input_ids: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute per-sample average log-probability of *labels*.

        Args:
            model: The model to evaluate.
            input_ids: (B, L) token ids.
            labels: (B, L) target token ids (-100 for ignored positions).

        Returns:
            (B,) tensor of average log-probabilities.
        """
        import torch.nn.functional as F  # noqa: N812

        logits, _ = model(input_ids[:, :-1], targets=None)
        # logits: (B, L-1, V)
        log_probs = F.log_softmax(logits, dim=-1)
        targets = labels[:, 1:]  # shift right

        # Gather log-probs for target tokens
        per_token = log_probs.gather(
            2, targets.unsqueeze(-1)).squeeze(-1)

        # Mask out padding — labels use -100 for ignored positions,
        # 0 is a valid token ID (pad_token) that should still be masked.
        mask = (targets != -100).float()
        return (per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

    def train_dpo(
        self,
        preference_data: list[dict],
        beta: float = 0.1,
    ) -> TrainingState:
        """Train with Direct Preference Optimization.

        Each item in *preference_data* must contain:
        - ``prompt``: the user prompt
        - ``chosen``: the preferred response
        - ``rejected``: the dis-preferred response

        A frozen copy of the model serves as the reference policy.

        Args:
            preference_data: List of preference pairs.
            beta: DPO temperature (lower = stronger preference signal).

        Returns:
            Final :class:`TrainingState`.
        """
        import copy

        if not preference_data:
            raise ValueError("No preference data provided")

        self._stop_requested = False
        self.model.train()
        self._training_start_time = time.monotonic()
        self._epochs_without_improvement = 0

        self._emit_progress(0, "Preparing DPO training...")

        # Build a frozen reference model (same architecture, same weights)
        ref_model = copy.deepcopy(self.model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # Encode preference pairs
        pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for item in preference_data:
            prompt = item.get("prompt", "")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            if not (prompt and chosen and rejected):
                continue

            prompt_ids = self.tokenizer.encode(prompt)
            chosen_ids = self.tokenizer.encode(f"Q: {prompt}\nA: {chosen}")
            rejected_ids = self.tokenizer.encode(f"Q: {prompt}\nA: {rejected}")

            # Truncate to max 512
            chosen_ids = chosen_ids[:512]
            rejected_ids = rejected_ids[:512]

            # Create label tensors (-100 for prompt tokens)
            prompt_len = len(prompt_ids)
            chosen_labels = (
                [-100] * min(prompt_len, len(chosen_ids))
                + chosen_ids[prompt_len:]
            )
            rejected_labels = (
                [-100] * min(prompt_len, len(rejected_ids))
                + rejected_ids[prompt_len:]
            )

            # Pad to same length
            max_len = max(len(chosen_ids), len(rejected_ids))
            chosen_ids += [0] * (max_len - len(chosen_ids))
            rejected_ids += [0] * (max_len - len(rejected_ids))
            chosen_labels += [-100] * (max_len - len(chosen_labels))
            rejected_labels += [-100] * (max_len - len(rejected_labels))

            pairs.append((
                torch.tensor([chosen_ids, rejected_ids],
                             dtype=torch.long, device=self.device),
                torch.tensor([chosen_labels],
                             dtype=torch.long, device=self.device),
                torch.tensor([rejected_labels],
                             dtype=torch.long, device=self.device),
            ))

        if not pairs:
            raise ValueError("No valid preference pairs after encoding")

        self._emit_progress(5, f"DPO training: {len(pairs)} pairs")
        logger.info(f"DPO training: {len(pairs)} preference pairs, "
                    f"beta={beta}")

        # Setup scheduler
        total_steps = len(pairs) * self.config.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1),
            eta_min=self.config.learning_rate * 0.1)

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            if self._should_stop():
                break
            if (self.config.max_training_seconds > 0
                    and time.monotonic() - self._training_start_time
                    > self.config.max_training_seconds):
                break

            epoch_loss = 0.0
            random.shuffle(pairs)

            progress_base = int(5 + (epoch / self.config.epochs) * 90)
            self._emit_progress(
                progress_base, f"DPO Epoch {epoch + 1}/{self.config.epochs}")

            for i, (input_ids, chosen_labels, rejected_labels) in enumerate(
                    pairs):
                if self._should_stop():
                    break

                self.optimizer.zero_grad()

                # Policy log-probs
                policy_chosen = self._get_sequence_logps(
                    self.model, input_ids[:1], chosen_labels)
                policy_rejected = self._get_sequence_logps(
                    self.model, input_ids[1:], rejected_labels)

                # Reference log-probs (no grad)
                with torch.no_grad():
                    ref_chosen = self._get_sequence_logps(
                        ref_model, input_ids[:1], chosen_labels)
                    ref_rejected = self._get_sequence_logps(
                        ref_model, input_ids[1:], rejected_labels)

                loss = self._dpo_loss(
                    policy_chosen, policy_rejected,
                    ref_chosen, ref_rejected, beta=beta)

                loss.backward()

                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip)

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                batch_loss = loss.item()
                if math.isnan(batch_loss) or math.isinf(batch_loss):
                    logger.error("DPO aborted: NaN/Inf loss")
                    self.model.eval()
                    del ref_model
                    return self.state

                epoch_loss += batch_loss
                self.state.step += 1

            avg_loss = epoch_loss / max(len(pairs), 1)
            self.state.training_losses.append(avg_loss)
            self.state.epoch = epoch + 1
            self._emit_loss(avg_loss)
            logger.info(f"DPO Epoch {epoch + 1}: loss={avg_loss:.4f}")

            if self.on_epoch_complete:
                self.on_epoch_complete(epoch + 1, avg_loss)

            if avg_loss < self.state.best_loss:
                self.state.best_loss = avg_loss
                self._epochs_without_improvement = 0
                self._save_checkpoint(checkpoint_dir / "best_model.pt")
            else:
                self._epochs_without_improvement += 1
                if (self.config.early_stopping_patience > 0
                        and self._epochs_without_improvement
                        >= self.config.early_stopping_patience):
                    logger.info("DPO early stopping triggered")
                    break

        self._emit_progress(100, "DPO training complete")
        self.model.eval()
        del ref_model
        return self.state

    # ─────────────────────────────────────────────────────────────────────
    # VISION TRAINING — image-text pair training
    # ─────────────────────────────────────────────────────────────────────

    def train_vision(
        self,
        vision_encoder: nn.Module,
        data: list[dict[str, Any]],
        unfreeze_text_layers: int = 0,
    ) -> "TrainingState":
        """
        Train the vision encoder and projection layer on image-text pairs.

        Both the vision encoder and the model's vision_projection layer are
        trained together. The text transformer is frozen by default (set
        unfreeze_text_layers > 0 to fine-tune the last N text layers too).

        Data format: list of dicts with:
            - "image": PIL Image object, file path string, or Path
            - "text": caption/description string

        Args:
            vision_encoder: VisionEncoder instance to train.
            data: List of image-text pair dicts.
            unfreeze_text_layers: Number of last text transformer layers to
                unfreeze (0 = freeze all text layers, only train encoder +
                projection).

        Returns:
            TrainingState with loss history.

        Raises:
            ValueError: If model lacks vision_projection or data is empty.
        """
        from .vision_encoder import preprocess_image

        # ── Validation ──────────────────────────────────────────────────
        if not hasattr(self.model, "vision_projection") or self.model.vision_projection is None:
            raise ValueError(
                "Model does not have a vision projection layer. "
                "Set vision_hidden_size in ForgeConfig to enable vision."
            )
        if not data:
            raise ValueError("No training data provided for vision training.")

        self._stop_requested = False
        self.state = TrainingState()
        self._epochs_without_improvement = 0
        start_time = time.time()

        self._emit_progress(0, "Preparing vision training data...")

        # ── Freeze / unfreeze layers ────────────────────────────────────
        # Freeze all text model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze vision projection (always trainable)
        for param in self.model.vision_projection.parameters():
            param.requires_grad = True

        # Unfreeze output/embedding (needed for text loss)
        for param in self.model.tok_embeddings.parameters():
            param.requires_grad = True
        for param in self.model.output.parameters():
            param.requires_grad = True
        for param in self.model.norm.parameters():
            param.requires_grad = True

        # Optionally unfreeze last N text transformer layers
        if unfreeze_text_layers > 0:
            n_layers = len(self.model.layers)
            for layer in self.model.layers[max(0, n_layers - unfreeze_text_layers):]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Vision encoder is fully trainable
        for param in vision_encoder.parameters():
            param.requires_grad = True

        # ── Build optimizer over all trainable parameters ───────────────
        trainable_params = (
            list(filter(lambda p: p.requires_grad, self.model.parameters()))
            + list(vision_encoder.parameters())
        )
        optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scaler = torch.amp.GradScaler("cuda", enabled=(
            self.config.use_amp and self.device.type == "cuda"
        ))

        # ── Preprocess data ─────────────────────────────────────────────
        image_size = vision_encoder.config.image_size
        pairs: list[tuple[torch.Tensor, list[int]]] = []

        for i, item in enumerate(data):
            image = item.get("image")
            text = item.get("text", "")
            if image is None or not text:
                logger.warning(f"Skipping vision data item {i}: missing image or text")
                continue

            # Preprocess image to tensor
            try:
                img_tensor = preprocess_image(image, image_size=image_size)
                img_tensor = img_tensor.to(self.device)
            except Exception as exc:
                logger.warning(f"Skipping vision data item {i}: {exc}")
                continue

            # Encode text to token IDs
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) < 1:
                continue
            pairs.append((img_tensor, token_ids))

        if not pairs:
            raise ValueError("No valid image-text pairs found in training data.")

        self._emit_progress(5, f"Prepared {len(pairs)} image-text pairs")
        logger.info(f"Vision training: {len(pairs)} pairs, {self.config.epochs} epochs")

        # ── Training loop ───────────────────────────────────────────────
        vision_encoder.train()
        self.model.train()

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            if self._should_stop():
                logger.info("Vision training stopped by user request")
                break

            # Time limit check
            if (self.config.max_training_seconds > 0
                    and (time.time() - start_time) > self.config.max_training_seconds):
                logger.info("Vision training: time limit reached")
                break

            epoch_loss = 0.0
            epoch_steps = 0

            # Shuffle training pairs each epoch
            random.shuffle(pairs)

            for step, (img_tensor, token_ids) in enumerate(pairs):
                if self._should_stop():
                    break

                optimizer.zero_grad()

                # Encode image through vision encoder
                with torch.amp.autocast("cuda", enabled=(
                    self.config.use_amp and self.device.type == "cuda"
                )):
                    vision_features = vision_encoder(img_tensor)  # [1, patches, v_dim]

                    # Build text input: token IDs as target
                    text_tensor = torch.tensor(
                        [token_ids], dtype=torch.long, device=self.device
                    )

                    # Forward through model with vision features
                    # The model concatenates [vision_patches, text_tokens] internally
                    logits = self.model.forward_multimodal(
                        input_ids=text_tensor,
                        vision_features=vision_features,
                    )
                    # logits shape: [1, vision_patches + text_len, vocab_size]

                    # We only compute loss on the text portion
                    # The text tokens start after the vision patches
                    n_patches = vision_features.shape[1]
                    text_logits = logits[:, n_patches:-1, :]  # predict next token
                    text_targets = text_tensor[:, 1:]          # shift targets

                    # Align lengths (in case of truncation)
                    min_len = min(text_logits.shape[1], text_targets.shape[1])
                    if min_len < 1:
                        continue

                    text_logits = text_logits[:, :min_len, :]
                    text_targets = text_targets[:, :min_len]

                    loss = nn.functional.cross_entropy(
                        text_logits.reshape(-1, text_logits.size(-1)),
                        text_targets.reshape(-1),
                    )

                # Backward and optimize
                if self.config.use_amp and self.device.type == "cuda":
                    scaler.scale(loss).backward()
                    if self.config.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            trainable_params, self.config.gradient_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if self.config.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            trainable_params, self.config.gradient_clip
                        )
                    optimizer.step()

                loss_val = loss.item()

                # NaN guard
                if math.isnan(loss_val) or math.isinf(loss_val):
                    logger.error(f"Vision training: NaN/Inf loss at epoch {epoch + 1}, step {step}")
                    self._emit_progress(100, "Training aborted: NaN loss")
                    self.model.eval()
                    return self.state

                # Safety guard
                if loss_val > self.config.max_loss:
                    logger.error(f"Vision training: loss {loss_val:.2f} exceeds max {self.config.max_loss}")
                    self._emit_progress(100, "Training aborted: loss too high")
                    self.model.eval()
                    return self.state

                epoch_loss += loss_val
                epoch_steps += 1
                self.state.step += 1

                # Log every N steps
                if self.config.log_every > 0 and self.state.step % self.config.log_every == 0:
                    self._emit_loss(loss_val)

                # Progress within epoch
                total_steps = len(pairs) * self.config.epochs
                done_steps = epoch * len(pairs) + step + 1
                pct = min(int(done_steps / max(total_steps, 1) * 95) + 5, 99)
                self._emit_progress(pct, f"Epoch {epoch + 1}/{self.config.epochs} — loss: {loss_val:.4f}")

            # Epoch summary
            avg_loss = epoch_loss / max(epoch_steps, 1)
            self.state.training_losses.append(avg_loss)
            self.state.epoch = epoch + 1
            self._emit_loss(avg_loss)
            logger.info(f"Vision Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

            if self.on_epoch_complete:
                try:
                    self.on_epoch_complete(epoch + 1, avg_loss)
                except Exception:
                    logger.debug("on_epoch_complete callback error", exc_info=True)

            # Best model tracking and early stopping
            if avg_loss < self.state.best_loss:
                self.state.best_loss = avg_loss
                self._epochs_without_improvement = 0
                self._save_checkpoint(checkpoint_dir / "best_vision_model.pt")
            else:
                self._epochs_without_improvement += 1
                if (self.config.early_stopping_patience > 0
                        and self._epochs_without_improvement
                        >= self.config.early_stopping_patience):
                    logger.info("Vision training: early stopping triggered")
                    break

            # Periodic checkpoint (0 = disabled)
            if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(checkpoint_dir / f"vision_epoch_{epoch + 1}.pt")
                self._cleanup_periodic_checkpoints(
                    checkpoint_dir, "vision_epoch_", keep=3)

        # ── Cleanup ─────────────────────────────────────────────────────
        self._emit_progress(100, "Vision training complete!")
        self.model.eval()
        vision_encoder.eval()

        # Re-enable all model parameters for subsequent use
        for param in self.model.parameters():
            param.requires_grad = True

        return self.state


# =============================================================================
# EVOLUTIONARY TRAINING (BEST-OF-N SAMPLING)
# =============================================================================

def best_of_n(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    n: int = 5,
    temperature_range: tuple[float, float] = (0.5, 1.0),
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Generate N responses and return one at random.

    Generates multiple responses with varied temperatures for
    diversity, then picks a random non-empty result.

    Args:
        model: The model to generate from
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt
        n: Number of responses to generate
        temperature_range: Range for random temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Tuple of (response, 1.0)
    """
    model.eval()
    device = next(model.parameters()).device

    responses: list[str] = []

    for i in range(n):
        temperature = random.uniform(*temperature_range)

        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
            )

        if hasattr(output_ids, 'squeeze'):
            output_ids = output_ids.squeeze(0)
        new_tokens = output_ids[len(input_ids):].tolist()
        response = tokenizer.decode(new_tokens)

        if response.strip():
            responses.append(response)

        logger.debug(f"Response {i+1}/{n}: len={len(response)}, temp={temperature:.2f}")

    if not responses:
        return "", 1.0

    chosen = random.choice(responses)
    logger.info(f"Best of {n}: picked 1 of {len(responses)} non-empty responses")
    return chosen, 1.0


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
) -> tuple[str, float]:
    """
    Generate a single response (for use in parallel execution).

    Args:
        model: Model to generate from
        tokenizer: Tokenizer
        prompt: Input prompt
        temperature: Generation temperature
        max_tokens: Max tokens to generate
        device: Device to run on

    Returns:
        Tuple of (response, temperature)
    """
    try:
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
            )

        if hasattr(output_ids, 'squeeze'):
            output_ids = output_ids.squeeze(0)
        new_tokens = output_ids[len(input_ids):].tolist()
        response = tokenizer.decode(new_tokens)

        return (response, temperature)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return ("", temperature)


def parallel_best_of_n(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    n: int = 5,
    max_workers: int = 4,
    temperature_range: tuple[float, float] = (0.5, 1.0),
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Generate N responses in parallel and return one at random.

    Uses ThreadPoolExecutor for parallel generation with varied
    temperatures. Picks a random non-empty response.

    Args:
        model: Model to generate from
        tokenizer: Tokenizer
        prompt: Input prompt
        n: Number of responses to generate
        max_workers: Maximum parallel workers
        temperature_range: Range for random temperature
        max_tokens: Max tokens to generate

    Returns:
        Tuple of (response, 1.0)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    model.eval()
    device = next(model.parameters()).device

    temperatures = [random.uniform(*temperature_range) for _ in range(n)]

    responses: list[str] = []

    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as executor:
        futures = {}
        for i, temp in enumerate(temperatures):
            future = executor.submit(
                _generate_single_response,
                model, tokenizer, prompt, temp, max_tokens, device
            )
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            try:
                resp, temp = future.result()
                if resp.strip():
                    responses.append(resp)
                logger.debug(f"Instance {idx+1}/{n}: len={len(resp)}, temp={temp:.2f}")
            except Exception as e:
                logger.error(f"Instance {idx+1} failed: {e}")

    if not responses:
        return "", 1.0

    chosen = random.choice(responses)
    logger.info(f"Parallel best of {n}: picked 1 of {len(responses)} non-empty responses")
    return chosen, 1.0


def multi_instance_collect(
    model: nn.Module,
    tokenizer: Any,
    tasks: list[str],
    n_per_task: int = 5,
    max_workers: int = 4,
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
        on_progress: Progress callback

    Returns:
        List of training examples
    """
    training_data = []

    for i, task in enumerate(tasks):
        if on_progress:
            on_progress(int(i / len(tasks) * 100), f"Task {i+1}/{len(tasks)}")

        try:
            best_response, _ = parallel_best_of_n(
                model, tokenizer, task,
                n=n_per_task,
                max_workers=max_workers
            )

            if best_response.strip():
                training_data.append({
                    "prompt": task,
                    "completion": best_response,
                })
                logger.info(f"Task {i+1}: KEPT")

        except Exception as e:
            logger.error(f"Task {i+1} failed: {e}")

    logger.info(f"Collected {len(training_data)} examples from {len(tasks)} tasks (parallel)")
    return training_data


def collect_training_data(
    model: nn.Module,
    tokenizer: Any,
    tasks: list[str],
    n_per_task: int = 5,
    on_progress: Callable[[int, str], None] | None = None
) -> list[dict]:
    """
    Generate training data by running best-of-N on tasks.

    Args:
        model: Model to generate from
        tokenizer: Tokenizer
        tasks: List of prompts/tasks
        n_per_task: Number of generations per task
        on_progress: Progress callback

    Returns:
        List of training examples: [{"prompt": str, "completion": str}]
    """
    training_data = []

    for i, task in enumerate(tasks):
        if on_progress:
            on_progress(int(i / len(tasks) * 100), f"Task {i+1}/{len(tasks)}")

        try:
            best_response, _ = best_of_n(
                model, tokenizer, task,
                n=n_per_task
            )

            if best_response.strip():
                training_data.append({
                    "prompt": task,
                    "completion": best_response,
                })
                logger.info(f"Task {i+1}: KEPT")

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
    2. Pick a response for each task
    3. Fine-tune model on the outputs
    4. Repeat with improved model

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
            on_progress=lambda p, m, _gp=gen_progress: on_progress(_gp + int(p * 0.5 / generations), m) if on_progress else None
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
            trainer.on_progress = lambda p, m, _gp=gen_progress, _gen=gen: on_progress(
                _gp + 50 + int(p * 0.5 / generations),
                f"Gen {_gen + 1}: {m}"
            )

        trainer.train(train_text)

        # Save generation checkpoint
        gen_checkpoint = checkpoint_path / f"generation_{gen + 1}.pt"
        gen_save = {
            'model_state_dict': model.state_dict(),
            'generation': gen + 1,
            'training_data_count': len(training_data),
            'avg_score': sum(ex['score'] for ex in training_data) / len(training_data)
        }
        if hasattr(model, 'config'):
            gen_save['model_config'] = model.config.__dict__
            gen_save['config'] = model.config.__dict__
        from enigma_engine.core.safe_save import atomic_torch_save
        atomic_torch_save(gen_save, gen_checkpoint)

        logger.info(f"Generation {gen + 1} complete: {len(training_data)} examples, saved to {gen_checkpoint}")

    # Save final model
    final_path = checkpoint_path / "final_evolved_model.pt"
    final_save = {
        'model_state_dict': model.state_dict(),
        'total_generations': generations,
    }
    if hasattr(model, 'config'):
        final_save['model_config'] = model.config.__dict__
        final_save['config'] = model.config.__dict__
    atomic_torch_save(final_save, final_path)

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
