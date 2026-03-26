"""
Training Module for Enigma AI Engine

Provides:
- TrainingConfig: Configuration for training
- Trainer: Basic fine-tuning with progress callbacks

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
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

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
    amp_dtype: str = "auto"  # "auto", "float16", "bfloat16"
    max_grad_accumulation: int = 1
    use_gradient_checkpointing: bool = False

    # Optimizer betas (LM-friendly defaults)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Reasoning-weighted loss (CoT-E)
    # Weight multiplier for tokens inside <think>...</think> blocks.
    # 1.0 = normal (no extra weight), 2.0 = double weight on reasoning tokens.
    reasoning_loss_weight: float = 1.0

    # Rolling best checkpoints (CK-C)
    rolling_best_k: int = 0  # 0 = disabled, N = keep N best by loss

    # Early-stopping / safety guardrails
    early_stopping_patience: int = 5  # Stop if no improvement for 5 epochs
    max_loss: float = 100.0  # abort if loss exceeds this
    max_training_seconds: float = 0  # 0 = unlimited

    # Before/after evaluation (EV-C)
    run_evaluation: bool = False  # Evaluate before and after training
    eval_test_prompts: list[str] = None  # Custom test prompts (None = use defaults)

    # Label smoothing (fairseq pattern â€” reduces overconfidence)
    label_smoothing: float = 0.05  # Reduces overconfidence, preserves generality

    # Validation split
    val_split: float = 0.1  # 10% held out to detect overfitting early

    # EMA weight averaging (smooths training noise, use EMA for eval)
    ema_decay: float = 0.0  # 0.0 = disabled, 0.999 or 0.9999 typical

    # torch.compile (10-20% throughput gain on supported hardware)
    use_compile: bool = False

    # Sequence packing: pack short sequences into max_seq_len rows
    # separated by EOS tokens with block-diagonal attention masks.
    # 30-50% throughput gain by eliminating padding waste.
    use_sequence_packing: bool = False

    # General data mixing — prevents catastrophic forgetting.
    # When focused training data is provided alongside general data,
    # this ratio controls how much general data is mixed into each epoch.
    # 0.0 = no mixing (only focused), 1.0 = only general.
    # 0.2 = 20% general + 80% focused (good default).
    general_mix_ratio: float = 0.2
    general_data: str = ""  # Path or text of general knowledge data

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
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError(
                f"val_split must be in [0.0, 1.0), got {self.val_split}"
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
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_eps": self.adam_eps,
            "rolling_best_k": self.rolling_best_k,
            "early_stopping_patience": self.early_stopping_patience,
            "max_loss": self.max_loss,
            "max_training_seconds": self.max_training_seconds,
            "label_smoothing": self.label_smoothing,
            "val_split": self.val_split,
            "ema_decay": self.ema_decay,
            "use_compile": self.use_compile,
            "use_sequence_packing": self.use_sequence_packing,
            "reasoning_loss_weight": self.reasoning_loss_weight,
            "general_mix_ratio": self.general_mix_ratio,
            "general_data": self.general_data,
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


class EMAWeightAverager:
    """Exponential Moving Average of model weights.

    Maintains shadow copies of every parameter, updated each step:
        shadow = decay * shadow + (1 - decay) * current_param

    Use ``apply()`` before eval/save, ``restore()`` after to swap
    EMA weights in and live weights back.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: list[torch.Tensor] = [
            p.clone().detach() for p in model.parameters()
        ]
        self._backup: list[torch.Tensor] = []

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights toward current model parameters."""
        for shadow, param in zip(self.shadow, model.parameters()):
            shadow.lerp_(param.detach(), 1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        """Swap EMA weights into the model (back up live weights)."""
        self._backup = [p.clone() for p in model.parameters()]
        for param, shadow in zip(model.parameters(), self.shadow):
            param.data.copy_(shadow)

    def restore(self, model: nn.Module) -> None:
        """Restore live weights from backup (undo ``apply()``)."""
        for param, backup in zip(model.parameters(), self._backup):
            param.data.copy_(backup)
        self._backup = []

    def state_dict(self) -> dict[str, list[torch.Tensor]]:
        """Serialize EMA state for checkpointing."""
        return {"shadow": self.shadow}

    def load_state_dict(self, state: dict[str, list[torch.Tensor]]) -> None:
        """Restore EMA state from checkpoint."""
        self.shadow = [s.clone() for s in state["shadow"]]


# =============================================================================
# SEQUENCE PACKING
# =============================================================================

def pack_sequences(
    encoded_seqs: list[list[int]],
    max_length: int,
    eos_id: int,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack multiple short sequences into max_length rows with EOS separators.

    Each row is filled greedily: sequences are appended with an EOS token
    after each until the row is full.  A 4D block-diagonal causal mask
    prevents tokens in different documents from attending to each other.

    Args:
        encoded_seqs: List of token-id lists (already encoded).
        max_length: Target row length (model's max_seq_len).
        eos_id: End-of-sequence token used as a separator.
        pad_id: Padding token for remaining space.

    Returns:
        (packed_tensor, attention_mask_2d):
        - packed_tensor: ``(num_rows, max_length)`` long tensor.
        - attention_mask_2d: ``(num_rows, 1, max_length, max_length)``
          float tensor. 0 = attend, -inf = block.
    """
    rows: list[list[int]] = []
    # boundaries[i] is a list of (start, end) spans for each document in row i
    boundaries: list[list[tuple[int, int]]] = []

    current_row: list[int] = []
    current_bounds: list[tuple[int, int]] = []

    for seq in encoded_seqs:
        # Each packed document = seq tokens + 1 EOS separator
        needed = len(seq) + 1
        if needed > max_length:
            # Truncate long sequences to fit a full row
            seq = seq[:max_length - 1]
            needed = max_length

        if len(current_row) + needed > max_length:
            # Current row is full — flush it
            if current_row:
                rows.append(current_row)
                boundaries.append(current_bounds)
            current_row = []
            current_bounds = []

        start = len(current_row)
        current_row.extend(seq)
        current_row.append(eos_id)
        end = len(current_row)
        current_bounds.append((start, end))

    # Flush the last row
    if current_row:
        rows.append(current_row)
        boundaries.append(current_bounds)

    # Pad each row to max_length and build 4D masks
    neg_inf = float('-inf')
    packed = []
    masks = []

    for row, bounds in zip(rows, boundaries):
        pad_len = max_length - len(row)
        padded_row = row + [pad_id] * pad_len
        packed.append(padded_row)

        # Build the 2D mask for this row
        # Start with everything blocked
        mask = [[neg_inf] * max_length for _ in range(max_length)]

        # For each document span, allow causal attention within that span
        for start, end in bounds:
            for i in range(start, end):
                for j in range(start, i + 1):  # causal: attend to self and past within doc
                    mask[i][j] = 0.0

        masks.append(mask)

    packed_tensor = torch.tensor(packed, dtype=torch.long)
    mask_tensor = torch.tensor(masks, dtype=torch.float32).unsqueeze(1)  # (B, 1, T, T)

    return packed_tensor, mask_tensor


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
            f"Data contains {null_count} null bytes â€” "
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
            f"{min_length} chars â€” may not provide useful signal.")

    if long_count > 0:
        result.warnings.append(
            f"{long_count} sequences exceed {max_length:,} chars â€” "
            f"will be truncated to model max_seq_len.")

    if empty_count > 5:
        result.warnings.append(
            f"{empty_count} runs of 3+ empty lines â€” "
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

        # Resolve AMP dtype (BF16 on Blackwell / Ampere+, FP16 otherwise)
        self._amp_dtype = self._resolve_amp_dtype()

        # Mixed precision scaler — disabled for BF16 (no loss scaling needed)
        self.scaler = None
        if self.config.use_amp and torch.cuda.is_available():
            if self._amp_dtype != torch.bfloat16:
                self.scaler = torch.amp.GradScaler('cuda')

        # Gradient checkpointing â€” trades compute for VRAM savings
        if self.config.use_gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            else:
                # Manual: enable on each transformer layer
                for layer in getattr(self.model, "layers", []):
                    if hasattr(layer, "gradient_checkpointing"):
                        layer.gradient_checkpointing = True

        # Resolve pad token ID from the tokenizer â€” used for
        # padding batches and as ignore_index in cross-entropy loss.
        # Built-in tokenizer uses 0, tiktoken uses base+0.
        self.pad_token_id: int = getattr(
            self.tokenizer, "pad_token_id", 0)

        # Rolling best checkpoint tracking (CK-C)
        # List of (loss, path) tuples sorted by loss ascending
        self._rolling_checkpoints: list[tuple[float, Path]] = []

        # EMA weight averaging
        self.ema: EMAWeightAverager | None = None
        if self.config.ema_decay > 0:
            self.ema = EMAWeightAverager(self.model, decay=self.config.ema_decay)
            logger.info(f"EMA enabled with decay={self.config.ema_decay}")

        # torch.compile for throughput gains (PyTorch 2.0+)
        self._compiled = False
        if self.config.use_compile:
            try:
                self.model = torch.compile(self.model)
                self._compiled = True
                logger.info("torch.compile enabled")
            except Exception as e:
                logger.warning(f"torch.compile not available: {e}")

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

        # Use fused AdamW on CUDA when available (5-10% faster optimizer step)
        adamw_kwargs: dict[str, Any] = {}
        if torch.cuda.is_available():
            import inspect
            if 'fused' in inspect.signature(AdamW).parameters:
                adamw_kwargs['fused'] = True

        self.optimizer = AdamW([
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.config.learning_rate,
           betas=(self.config.adam_beta1, self.config.adam_beta2),
           eps=self.config.adam_eps,
           **adamw_kwargs)

        # Scheduler set when we know total steps (see _create_scheduler)
        self.scheduler = None

    def _resolve_amp_dtype(self) -> torch.dtype:
        """Resolve the AMP autocast dtype from config.

        ``"auto"`` picks BF16 when the GPU supports it (Ampere+, Blackwell)
        and falls back to FP16 otherwise.  BF16 has better numeric range
        which avoids many loss-scaling headaches.
        """
        from .rl_training import _resolve_amp_dtype
        return _resolve_amp_dtype(self.config.amp_dtype)

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
                        sequences.append(
                            f"User: {prompt}\nAssistant: {completion}")
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
                        sequences.append(
                            f"User: {prompt}\nAssistant: {completion}")
                except json.JSONDecodeError:
                    continue
            if sequences:
                return sequences

        # Try Q&A format — normalise to User/Assistant to match
        # the chat inference prompt format.
        qa_pattern = re.compile(r'Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)', re.DOTALL)
        matches = qa_pattern.findall(data)
        if matches:
            for q, a in matches:
                sequences.append(
                    f"User: {q.strip()}\nAssistant: {a.strip()}")
            return sequences

        # Try User/AI dialogue format — normalise role labels
        dialogue_pattern = re.compile(
            r'(?:User|Human):\s*(.+?)\s*(?:AI|Assistant):\s*(.+?)(?=(?:User|Human):|$)',
            re.DOTALL | re.IGNORECASE)
        d_matches = dialogue_pattern.findall(data)
        if d_matches:
            for user_msg, ai_msg in d_matches:
                sequences.append(
                    f"User: {user_msg.strip()}\nAssistant: {ai_msg.strip()}")
            return sequences

        # Fall back to paragraph splitting
        paragraphs = data.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:  # Skip short/noisy paragraphs
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
            # Use named attributes (available on all 4 tokenizer classes)
            start_id = getattr(self.tokenizer, 'think_start_id', None)
            end_id = getattr(self.tokenizer, 'think_end_id', None)

            if start_id is None or end_id is None:
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
                    if tok == start_id:
                        in_think = True
                    if in_think:
                        weight[b, t] = w
                    if tok == end_id:
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

    def _pack_sequences(
        self,
        encoded_seqs: list[list[int]],
        max_length: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Pack encoded sequences into dense rows with 4D masks.

        Wrapper around the standalone ``pack_sequences()`` that uses
        the trainer's tokenizer EOS/PAD IDs and moves tensors to the
        correct device.

        Returns:
            List of (packed_tensor, mask_4d) tuples.  ``mask_4d`` has
            shape ``(B, 1, T, T)`` with 0 = attend and -inf = block.
        """
        eos_id = getattr(self.tokenizer, "eos_token_id", 2)
        packed, masks = pack_sequences(
            encoded_seqs,
            max_length=max_length,
            eos_id=eos_id,
            pad_id=self.pad_token_id,
        )
        packed = packed.to(self.device)
        masks = masks.to(self.device)

        # Split into batch-sized chunks
        batches = []
        batch_size = self.config.batch_size
        for i in range(0, packed.shape[0], batch_size):
            batches.append((
                packed[i:i + batch_size],
                masks[i:i + batch_size],
            ))
        return batches

    def _create_batches(
        self,
        sequences: list[str],
        max_length: int | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Create batches from sequences.

        Args:
            sequences: List of text sequences
            max_length: Maximum sequence length.  Defaults to the
                model's ``max_seq_len`` if available, otherwise 512.

        Returns:
            List of (batch_tensor, attention_mask) tuples.
            attention_mask has 1 for real tokens, 0 for padding.
            When sequence packing is enabled, attention_mask is a 4D
            tensor ``(B, 1, T, T)`` instead of 2D ``(B, T)``.
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
            if len(tokens) >= 5:  # Need enough tokens for meaningful training
                encoded.append(tokens)

        if not encoded:
            raise ValueError("No valid sequences after encoding")

        # ---- Sequence packing path ----
        if self.config.use_sequence_packing:
            return self._pack_sequences(encoded, max_length)

        # ---- Standard padding path ----
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
            masks = []
            for tokens in batch_tokens:
                pad_len = max_len - len(tokens)
                masks.append([1] * len(tokens) + [0] * pad_len)
                padded.append(tokens + [pad_token_id] * pad_len)

            batch_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device)
            batches.append((batch_tensor, attention_mask))

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


        # Mix general knowledge data to prevent catastrophic forgetting.
        # When focused data is provided, keep the model general by mixing
        # a portion of general/diverse examples into each training batch.
        if (self.config.general_data
                and self.config.general_mix_ratio > 0):
            try:
                general_text = self.config.general_data
                # If it looks like a file path, load it
                gp = Path(general_text)
                if gp.exists() and gp.is_file():
                    general_text = gp.read_text(encoding="utf-8")
                general_seqs = self._parse_training_data(general_text)
                if general_seqs:
                    ratio = min(self.config.general_mix_ratio, 0.9)
                    n_focused = len(sequences)
                    n_general = max(1, int(
                        n_focused * ratio / max(0.01, 1.0 - ratio)))
                    if len(general_seqs) >= n_general:
                        mixed = random.sample(general_seqs, n_general)
                    else:
                        mixed = (general_seqs
                                 * (n_general // len(general_seqs) + 1)
                                 )[:n_general]
                    sequences.extend(mixed)
                    random.shuffle(sequences)
                    logger.info(
                        f"Mixed {n_general} general sequences "
                        f"with {n_focused} focused sequences "
                        f"(ratio {ratio:.0%} general)")
            except Exception as exc:
                logger.warning(
                    f"Could not mix general data: {exc}")

        # Deduplicate while preserving order â€” prevents the model
        # from over-fitting on repeated examples.
        pre_dedup = len(sequences)
        sequences = list(dict.fromkeys(sequences))
        if len(sequences) < pre_dedup:
            logger.info(
                f"Removed {pre_dedup - len(sequences)} duplicate "
                f"sequences ({len(sequences)} unique remain)")

        # Split train/validation when val_split > 0
        val_sequences: list[str] = []
        if self.config.val_split > 0 and len(sequences) > 1:
            n_val = max(1, int(len(sequences) * self.config.val_split))
            # Random stratified split — avoids bias from data ordering
            indices = list(range(len(sequences)))
            random.Random(42).shuffle(indices)  # deterministic seed for reproducibility
            val_indices = set(indices[:n_val])
            val_sequences = [sequences[i] for i in sorted(val_indices)]
            sequences = [sequences[i] for i in range(len(sequences)) if i not in val_indices]
            logger.info(
                f"Validation split: {len(sequences)} train, "
                f"{len(val_sequences)} val "
                f"({self.config.val_split:.0%})")

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
            logger.info(f"Created {len(batches)} training batches")
        except Exception as e:
            logger.error(f"Failed to create batches: {e}")
            raise

        # Create validation batches (if split is active)
        val_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        if val_sequences:
            try:
                val_batches = self._create_batches(
                    val_sequences, max_length=max_seq_len)
                logger.info(f"Created {len(val_batches)} validation batches")
            except Exception as exc:
                logger.warning(f"Failed to create val batches: {exc}")

        # Setup scheduler: SequentialLR(warmup â†’ cosine decay)
        total_steps = len(batches) * self.config.epochs
        warmup = max(1, self.config.warmup_steps)
        decay_steps = max(1, total_steps - warmup)

        warmup_scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup))
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=decay_steps,
            eta_min=self.config.learning_rate * 0.1)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup])

        # Restore scheduler/scaler state from checkpoint if available
        self._restore_pending_state()

        # Training loop
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot one weight tensor for sanity checking after warmup.
        # Pick a dense weight (not embeddings, which are sparse).
        # We delay the check until after warmup steps so the learning
        # rate is fully ramped and weight updates are meaningful.
        _weight_check_done = False
        _weight_check_step = min(warmup, len(batches))  # check after warmup
        _weight_snapshot: torch.Tensor | None = None
        _weight_ref_name = ""
        for _name, _p in self.model.named_parameters():
            if (_p.requires_grad and _p.ndim >= 2
                    and "embed" not in _name and "output" not in _name
                    and "head" not in _name):
                _weight_snapshot = _p.data.clone()
                _weight_ref_name = _name
                break

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

                batch_tokens = batch[0].numel()
                epoch_loss += batch_loss * batch_tokens
                epoch_tokens += batch_tokens
                self.state.step += 1
                self.state.total_tokens += batch_tokens

                # One-time sanity check: verify weights actually changed
                # after warmup completes (LR is near-zero during early
                # warmup, so checking sooner would produce false alarms).
                if (
                    not _weight_check_done
                    and _weight_snapshot is not None
                    and self.state.step >= _weight_check_step
                    and (batch_idx + 1) % self.config.max_grad_accumulation == 0
                ):
                    _weight_check_done = True
                    for _name, _p in self.model.named_parameters():
                        if _name == _weight_ref_name:
                            delta = (_weight_snapshot - _p.data).abs().max().item()
                            if delta == 0:
                                logger.error(
                                    "WARNING: Model weights unchanged after "
                                    "%d optimizer steps (%s). Training may "
                                    "not be effective.",
                                    self.state.step, _name)
                            else:
                                logger.info(
                                    "Weight update verified: %s "
                                    "max_delta=%.2e", _name, delta)
                            break
                    del _weight_snapshot
                    _weight_snapshot = None

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

            # Run validation pass (if we have val data)
            val_loss = None
            if val_batches:
                val_loss = self._validate(val_batches)
                self.state.validation_losses.append(val_loss)
                logger.info(
                    f"Epoch {epoch + 1} val_loss={val_loss:.4f}")

            if self.on_epoch_complete:
                self.on_epoch_complete(epoch + 1, avg_epoch_loss)

            # Save periodic checkpoint (0 = disabled)
            if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
                ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                self._save_checkpoint(ckpt_path)
                self._cleanup_periodic_checkpoints(
                    checkpoint_dir, "checkpoint_epoch_", keep=3)

            # Track best loss + early stopping
            # Prefer val_loss when available, fall back to train loss
            tracked_loss = val_loss if val_loss is not None else avg_epoch_loss
            if tracked_loss < self.state.best_loss:
                self.state.best_loss = tracked_loss
                self._epochs_without_improvement = 0
                self._save_checkpoint(checkpoint_dir / "best_model.pt")
                # Rolling best checkpoints (CK-C)
                if self.config.rolling_best_k > 0:
                    self._save_rolling_checkpoint(
                        checkpoint_dir, tracked_loss, epoch + 1)
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

    def _train_one_batch(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        """Run forward + backward on a single batch.

        Args:
            batch: Tuple of (token_ids, attention_mask).
            batch_idx: Index within the epoch (for grad accumulation gating).

        Returns the unscaled batch loss (float).  Raises RuntimeError
        on CUDA OOM so the caller can handle recovery.
        """
        batch_tensor, attention_mask = batch
        is_packed = attention_mask.ndim == 4  # 4D = sequence packing

        # Forward pass
        with torch.amp.autocast(
            'cuda',
            dtype=self._amp_dtype,
            enabled=self.config.use_amp and torch.cuda.is_available(),
        ):
            input_ids = batch_tensor[:, :-1]
            targets = batch_tensor[:, 1:]

            if is_packed:
                # 4D block-diagonal mask: slice both spatial dims
                attn_mask_2d = attention_mask[:, :, :-1, :-1]
                logits, loss = self.model(
                    input_ids, targets=targets,
                    pad_token_id=self.pad_token_id,
                    label_smoothing=self.config.label_smoothing,
                    attention_mask_2d=attn_mask_2d)
            else:
                # Standard 2D pad mask: slice last position
                attn_mask = attention_mask[:, :-1]
                logits, loss = self.model(
                    input_ids, targets=targets,
                    pad_token_id=self.pad_token_id,
                    label_smoothing=self.config.label_smoothing,
                    attention_mask=attn_mask)

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

            # Scheduler step (SequentialLR handles warmup â†’ cosine internally)
            if self.scheduler is not None:
                self.scheduler.step()

            # EMA: update shadow weights after each optimizer step
            if self.ema is not None:
                self.ema.update(self.model)

        return loss.item() * self.config.max_grad_accumulation

    @torch.no_grad()
    def _validate(self, val_batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Run a forward-only pass on validation batches.

        Returns the average validation loss (float).
        Uses EMA weights when available for more stable evaluation.
        """
        # Swap in EMA weights for evaluation if available
        if self.ema is not None:
            self.ema.apply(self.model)

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch_tensor, attention_mask in val_batches:
            is_packed = attention_mask.ndim == 4
            with torch.amp.autocast(
                'cuda',
                dtype=self._amp_dtype,
                enabled=self.config.use_amp and torch.cuda.is_available(),
            ):
                input_ids = batch_tensor[:, :-1]
                targets = batch_tensor[:, 1:]
                if is_packed:
                    attn_mask_2d = attention_mask[:, :, :-1, :-1]
                    _logits, loss = self.model(
                        input_ids, targets=targets,
                        pad_token_id=self.pad_token_id,
                        attention_mask_2d=attn_mask_2d)
                else:
                    attn_mask = attention_mask[:, :-1]
                    _logits, loss = self.model(
                        input_ids, targets=targets,
                        pad_token_id=self.pad_token_id,
                        attention_mask=attn_mask)

            if loss is not None:
                n_tokens = batch_tensor.numel()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens

        self.model.train()

        # Restore live training weights
        if self.ema is not None:
            self.ema.restore(self.model)

        return total_loss / max(1, total_tokens)

    def _handle_oom(self, exc: RuntimeError) -> None:
        """Handle CUDA OOM: clear cache, enable gradient checkpointing.

        Called on the first OOM per training run.  Enables gradient
        checkpointing (trades compute for VRAM) and clears the CUDA
        cache so the retry has the best chance of succeeding.
        """
        logger.warning(
            "CUDA out of memory â€” clearing cache and enabling "
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

    def _restore_pending_state(self) -> None:
        """Restore scheduler and scaler state stashed by load_checkpoint."""
        sched_state = getattr(self, '_pending_scheduler_state', None)
        if sched_state is not None and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(sched_state)
                logger.info("Restored scheduler state from checkpoint")
            except Exception as exc:
                logger.warning("Could not restore scheduler state: %s", exc)
            self._pending_scheduler_state = None

        scaler_state = getattr(self, '_pending_scaler_state', None)
        if scaler_state is not None and self.scaler is not None:
            try:
                self.scaler.load_state_dict(scaler_state)
                logger.info("Restored scaler state from checkpoint")
            except Exception as exc:
                logger.warning("Could not restore scaler state: %s", exc)
            self._pending_scaler_state = None

    def _save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint with full training state.

        Saves model weights, optimizer, scheduler, scaler, and step
        counters so training can resume exactly where it left off.
        """
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_state': {
                    'epoch': self.state.epoch,
                    'step': self.state.step,
                    'best_loss': self.state.best_loss,
                    'total_tokens': self.state.total_tokens,
                    'training_losses': self.state.training_losses,
                    'validation_losses': self.state.validation_losses,
                },
                'training_config': self.config.to_dict(),
            }
            # Save scheduler state for exact resume
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            # Save AMP scaler state
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()

            # Save EMA state for resume
            if self.ema is not None:
                checkpoint['ema_state_dict'] = self.ema.state_dict()

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

        # Sort by epoch number descending â€” keep the newest
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

            # Unwrap state dict â€” handles both flat and wrapped formats
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
            self.state.training_losses = state.get('training_losses', [])
            self.state.validation_losses = state.get('validation_losses', [])

            # Stash scheduler/scaler state for deferred restore
            self._pending_scheduler_state = checkpoint.get('scheduler_state_dict')
            self._pending_scaler_state = checkpoint.get('scaler_state_dict')

            # Restore EMA state if saved and EMA is active
            ema_state = checkpoint.get('ema_state_dict')
            if ema_state is not None and self.ema is not None:
                self.ema.load_state_dict(ema_state)

            logger.info(f"Loaded checkpoint: {path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    # -----------------------------------------------------------------
    # DPO â€” Direct Preference Optimization
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
        attention_mask: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Compute per-sample average log-probability of *labels*.

        Args:
            model: The model to evaluate.
            input_ids: (B, L) token ids.
            labels: (B, L) target token ids (-100 for ignored positions).
            attention_mask: (B, L) mask where 1=real, 0=pad.

        Returns:
            (B,) tensor of average log-probabilities.
        """
        import torch.nn.functional as F  # noqa: N812

        attn_mask = attention_mask[:, :-1] if attention_mask is not None else None
        logits, _ = model(input_ids[:, :-1], targets=None, attention_mask=attn_mask)
        # logits: (B, L-1, V)
        log_probs = F.log_softmax(logits, dim=-1)
        targets = labels[:, 1:]  # shift right

        # Gather log-probs for target tokens
        per_token = log_probs.gather(
            2, targets.unsqueeze(-1)).squeeze(-1)

        # Mask out padding â€” labels use -100 for ignored positions,
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

        # Build a frozen reference model.
        # Preferred: if the model has LoRA adapters, disable them to get
        # reference logps from the frozen base (zero extra VRAM).
        # Fallback: deepcopy for non-LoRA models.
        use_lora_ref = False
        ref_model = None
        if hasattr(self.model, 'disable_adapter_layers'):
            use_lora_ref = True
            logger.info("DPO: using LoRA disable — frozen base weights "
                        "serve as reference policy (no extra VRAM)")
        else:
            ref_model = copy.deepcopy(self.model)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False
            logger.info("DPO: using deepcopy reference model")

        # Encode preference pairs
        pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for item in preference_data:
            prompt = item.get("prompt", "")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            if not (prompt and chosen and rejected):
                continue

            prompt_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: ")
            chosen_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: {chosen}")
            rejected_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: {rejected}")

            # Truncate to model's max sequence length
            max_len_dpo = getattr(self.model, 'config', None)
            max_len_dpo = getattr(max_len_dpo, 'max_seq_len', 512) if max_len_dpo else 512
            chosen_ids = chosen_ids[:max_len_dpo]
            rejected_ids = rejected_ids[:max_len_dpo]

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

            # Build attention masks before padding (1=real, 0=pad)
            chosen_mask = [1] * len(chosen_ids)
            rejected_mask = [1] * len(rejected_ids)

            # Pad to same length
            max_len = max(len(chosen_ids), len(rejected_ids))
            chosen_mask += [0] * (max_len - len(chosen_mask))
            rejected_mask += [0] * (max_len - len(rejected_mask))
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
                torch.tensor([chosen_mask, rejected_mask],
                             dtype=torch.long, device=self.device),
            ))

        if not pairs:
            raise ValueError("No valid preference pairs after encoding")

        self._emit_progress(5, f"DPO training: {len(pairs)} pairs")
        logger.info(f"DPO training: {len(pairs)} preference pairs, "
                    f"beta={beta}")

        # Gradient accumulation: batch N pairs before optimizer step
        accum_steps = max(1, self.config.max_grad_accumulation)

        # Setup scheduler: SequentialLR(warmup â†’ cosine decay)
        steps_per_epoch = max(1, len(pairs) // accum_steps)
        total_steps = steps_per_epoch * self.config.epochs
        warmup = max(1, self.config.warmup_steps)
        decay_steps = max(1, total_steps - warmup)

        warmup_sched = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup))
        cosine_sched = CosineAnnealingLR(
            self.optimizer,
            T_max=decay_steps,
            eta_min=self.config.learning_rate * 0.1)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup])

        # Restore scheduler/scaler state from checkpoint if available
        self._restore_pending_state()

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
            self.optimizer.zero_grad()

            progress_base = int(5 + (epoch / self.config.epochs) * 90)
            self._emit_progress(
                progress_base, f"DPO Epoch {epoch + 1}/{self.config.epochs}")

            for i, (input_ids, chosen_labels, rejected_labels, attention_mask) in enumerate(
                    pairs):
                if self._should_stop():
                    break

                # Policy log-probs
                policy_chosen = self._get_sequence_logps(
                    self.model, input_ids[:1], chosen_labels,
                    attention_mask=attention_mask[:1])
                policy_rejected = self._get_sequence_logps(
                    self.model, input_ids[1:], rejected_labels,
                    attention_mask=attention_mask[1:])

                # Reference log-probs (no grad)
                with torch.no_grad():
                    if use_lora_ref:
                        self.model.eval()
                        self.model.disable_adapter_layers()
                        try:
                            ref_chosen = self._get_sequence_logps(
                                self.model, input_ids[:1], chosen_labels,
                                attention_mask=attention_mask[:1])
                            ref_rejected = self._get_sequence_logps(
                                self.model, input_ids[1:], rejected_labels,
                                attention_mask=attention_mask[1:])
                        finally:
                            self.model.enable_adapter_layers()
                        self.model.train()
                    else:
                        ref_chosen = self._get_sequence_logps(
                            ref_model, input_ids[:1], chosen_labels,
                            attention_mask=attention_mask[:1])
                        ref_rejected = self._get_sequence_logps(
                            ref_model, input_ids[1:], rejected_labels,
                            attention_mask=attention_mask[1:])

                loss = self._dpo_loss(
                    policy_chosen, policy_rejected,
                    ref_chosen, ref_rejected, beta=beta)

                # Scale loss for gradient accumulation
                loss = loss / accum_steps
                loss.backward()

                batch_loss = loss.item() * accum_steps
                if math.isnan(batch_loss) or math.isinf(batch_loss):
                    logger.error("DPO aborted: NaN/Inf loss")
                    self.model.eval()
                    if ref_model is not None:
                        del ref_model
                    return self.state

                epoch_loss += batch_loss

                # Step optimizer every accum_steps pairs
                if (i + 1) % accum_steps == 0:
                    if self.config.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                    self.state.step += 1

            # Flush remaining accumulated gradients at epoch end
            if len(pairs) % accum_steps != 0:
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()
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
        if ref_model is not None:
            del ref_model
        return self.state

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # VISION TRAINING â€” image-text pair training
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        from .vision_encoder import augment_vision_tensor, preprocess_image

        # â”€â”€ Validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

        # â”€â”€ Freeze / unfreeze layers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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


        # Re-freeze pretrained backbone if configured (the blanket unfreeze
        # above would otherwise override freeze_backbone from __init__)
        if getattr(vision_encoder.config, "freeze_backbone", False):
            backbone = getattr(vision_encoder, "backbone", None)
            if backbone is not None:
                for param in backbone.parameters():
                    param.requires_grad = False

        trainable_params = (
            list(filter(lambda p: p.requires_grad, self.model.parameters()))
            + list(filter(lambda p: p.requires_grad, vision_encoder.parameters()))
        )
        optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
        )

        scaler = torch.amp.GradScaler("cuda", enabled=(
            self.config.use_amp and self.device.type == "cuda"
            and self._amp_dtype != torch.bfloat16
        ))

        # Setup scheduler: SequentialLR(warmup + cosine decay)
        total_steps = len(data) * self.config.epochs  # 1 step per pair
        warmup = max(1, self.config.warmup_steps)
        decay_steps = max(1, total_steps - warmup)
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup))
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=decay_steps,
            eta_min=self.config.learning_rate * 0.1)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup])

        image_size = vision_encoder.config.image_size
        use_imagenet = getattr(vision_encoder.config, "use_pretrained", False)
        pairs: list[tuple[torch.Tensor, list[int]]] = []

        for i, item in enumerate(data):
            image = item.get("image")
            text = item.get("text", "")
            if image is None or not text:
                logger.warning(f"Skipping vision data item {i}: missing image or text")
                continue

            # Preprocess image to tensor
            try:
                img_tensor = preprocess_image(
                    image,
                    image_size=image_size,
                    imagenet_normalize=use_imagenet,
                )
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

        # â”€â”€ Training loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

                # Training-time augmentation (random flip, color jitter)
                img_tensor = augment_vision_tensor(img_tensor)

                optimizer.zero_grad()

                # Encode image through vision encoder
                with torch.amp.autocast(
                    "cuda",
                    dtype=self._amp_dtype,
                    enabled=self.config.use_amp and self.device.type == "cuda",
                ):
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

                scheduler.step()

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
                self._emit_progress(pct, f"Epoch {epoch + 1}/{self.config.epochs} â€” loss: {loss_val:.4f}")

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

        # â”€â”€ Cleanup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._emit_progress(100, "Vision training complete!")
        self.model.eval()
        vision_encoder.eval()

        # Re-enable all model parameters for subsequent use
        for param in self.model.parameters():
            param.requires_grad = True

        return self.state

    # -----------------------------------------------------------------------
    # AUDIO MULTIMODAL TRAINING
    # -----------------------------------------------------------------------

    def train_audio(
        self,
        audio_encoder: nn.Module,
        data: list[dict[str, Any]],
        unfreeze_text_layers: int = 0,
    ) -> "TrainingState":
        """
        Train the audio encoder and projection layer on audio-text pairs.

        Both the audio encoder and the model's audio_projection layer are
        trained together. The text transformer is frozen by default (set
        unfreeze_text_layers > 0 to fine-tune the last N text layers too).

        Data format: list of dicts with:
            - "audio": file path string, Path, or 1D waveform tensor
            - "text": transcript/description string

        Args:
            audio_encoder: AudioEncoder instance to train.
            data: List of audio-text pair dicts.
            unfreeze_text_layers: Number of last text transformer layers to
                unfreeze (0 = freeze all text layers, only train encoder +
                projection).

        Returns:
            TrainingState with loss history.

        Raises:
            ValueError: If model lacks audio_projection or data is empty.
        """
        from .audio_encoder import preprocess_audio, spec_augment

        # -- Validation --
        if not hasattr(self.model, "audio_projection") or self.model.audio_projection is None:
            raise ValueError(
                "Model does not have an audio projection layer. "
                "Set audio_hidden_size in ForgeConfig to enable audio."
            )
        if not data:
            raise ValueError("No training data provided for audio training.")

        self._stop_requested = False
        self.state = TrainingState()
        self._epochs_without_improvement = 0
        start_time = time.time()

        self._emit_progress(0, "Preparing audio training data...")

        # -- Freeze / unfreeze layers --
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze audio projection (always trainable)
        for param in self.model.audio_projection.parameters():
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

        # Audio encoder is fully trainable
        for param in audio_encoder.parameters():
            param.requires_grad = True

        trainable_params = (
            list(filter(lambda p: p.requires_grad, self.model.parameters()))
            + list(filter(lambda p: p.requires_grad, audio_encoder.parameters()))
        )
        optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
        )

        scaler = torch.amp.GradScaler("cuda", enabled=(
            self.config.use_amp and self.device.type == "cuda"
            and self._amp_dtype != torch.bfloat16
        ))

        # Setup scheduler: SequentialLR(warmup + cosine decay)
        total_steps = len(data) * self.config.epochs  # 1 step per pair
        warmup = max(1, self.config.warmup_steps)
        decay_steps = max(1, total_steps - warmup)
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup))
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=decay_steps,
            eta_min=self.config.learning_rate * 0.1)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup])

        # -- Preprocess data --
        audio_config = getattr(audio_encoder, "config", None)
        pairs: list[tuple[torch.Tensor, list[int]]] = []

        for i, item in enumerate(data):
            audio = item.get("audio")
            text = item.get("text", "")
            if audio is None or not text:
                logger.warning(f"Skipping audio data item {i}: missing audio or text")
                continue

            try:
                mel_tensor = preprocess_audio(audio, config=audio_config)
                mel_tensor = mel_tensor.to(self.device)
            except Exception as exc:
                logger.warning(f"Skipping audio data item {i}: {exc}")
                continue

            token_ids = self.tokenizer.encode(text)
            if len(token_ids) < 1:
                continue
            pairs.append((mel_tensor, token_ids))

        if not pairs:
            raise ValueError("No valid audio-text pairs found in training data.")

        self._emit_progress(5, f"Prepared {len(pairs)} audio-text pairs")
        logger.info(f"Audio training: {len(pairs)} pairs, {self.config.epochs} epochs")

        # -- Training loop --
        audio_encoder.train()
        self.model.train()

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            if self._should_stop():
                logger.info("Audio training stopped by user request")
                break

            if (self.config.max_training_seconds > 0
                    and (time.time() - start_time) > self.config.max_training_seconds):
                logger.info("Audio training: time limit reached")
                break

            epoch_loss = 0.0
            epoch_steps = 0

            random.shuffle(pairs)

            for step, (mel_tensor, token_ids) in enumerate(pairs):
                if self._should_stop():
                    break

                optimizer.zero_grad()

                with torch.amp.autocast(
                    "cuda",
                    dtype=self._amp_dtype,
                    enabled=self.config.use_amp and self.device.type == "cuda",
                ):
                    # SpecAugment: mask random frequency/time bands for regularization
                    augmented_mel = spec_augment(mel_tensor)

                    # Encode audio through audio encoder
                    audio_features = audio_encoder(augmented_mel)  # [1, T/2, a_dim]

                    text_tensor = torch.tensor(
                        [token_ids], dtype=torch.long, device=self.device
                    )

                    logits = self.model.forward_multimodal(
                        input_ids=text_tensor,
                        audio_features=audio_features,
                    )

                    # Text loss only (skip audio tokens)
                    n_audio_tokens = audio_features.shape[1]
                    text_logits = logits[:, n_audio_tokens:-1, :]
                    text_targets = text_tensor[:, 1:]

                    min_len = min(text_logits.shape[1], text_targets.shape[1])
                    if min_len < 1:
                        continue

                    text_logits = text_logits[:, :min_len, :]
                    text_targets = text_targets[:, :min_len]

                    loss = nn.functional.cross_entropy(
                        text_logits.reshape(-1, text_logits.size(-1)),
                        text_targets.reshape(-1),
                    )

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

                scheduler.step()

                loss_val = loss.item()

                if math.isnan(loss_val) or math.isinf(loss_val):
                    logger.error(f"Audio training: NaN/Inf loss at epoch {epoch + 1}, step {step}")
                    self._emit_progress(100, "Training aborted: NaN loss")
                    self.model.eval()
                    return self.state

                if loss_val > self.config.max_loss:
                    logger.error(f"Audio training: loss {loss_val:.2f} exceeds max {self.config.max_loss}")
                    self._emit_progress(100, "Training aborted: loss too high")
                    self.model.eval()
                    return self.state

                epoch_loss += loss_val
                epoch_steps += 1
                self.state.step += 1

                if self.config.log_every > 0 and self.state.step % self.config.log_every == 0:
                    self._emit_loss(loss_val)

                total_steps = len(pairs) * self.config.epochs
                done_steps = epoch * len(pairs) + step + 1
                pct = min(int(done_steps / max(total_steps, 1) * 95) + 5, 99)
                self._emit_progress(pct, f"Epoch {epoch + 1}/{self.config.epochs} - loss: {loss_val:.4f}")

            avg_loss = epoch_loss / max(epoch_steps, 1)
            self.state.training_losses.append(avg_loss)
            self.state.epoch = epoch + 1
            self._emit_loss(avg_loss)
            logger.info(f"Audio Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

            if self.on_epoch_complete:
                try:
                    self.on_epoch_complete(epoch + 1, avg_loss)
                except Exception:
                    logger.debug("on_epoch_complete callback error", exc_info=True)

            if avg_loss < self.state.best_loss:
                self.state.best_loss = avg_loss
                self._epochs_without_improvement = 0
                self._save_checkpoint(checkpoint_dir / "best_audio_model.pt")
            else:
                self._epochs_without_improvement += 1
                if (self.config.early_stopping_patience > 0
                        and self._epochs_without_improvement
                        >= self.config.early_stopping_patience):
                    logger.info("Audio training: early stopping triggered")
                    break

            if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(checkpoint_dir / f"audio_epoch_{epoch + 1}.pt")
                self._cleanup_periodic_checkpoints(
                    checkpoint_dir, "audio_epoch_", keep=3)

        # -- Cleanup --
        self._emit_progress(100, "Audio training complete!")
        self.model.eval()
        audio_encoder.eval()

        for param in self.model.parameters():
            param.requires_grad = True

        return self.state
