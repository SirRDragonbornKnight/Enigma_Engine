"""
Enigma Training System
======================

A production-grade training system with:
- Mixed precision training (AMP) for faster training on modern GPUs
- Gradient accumulation for larger effective batch sizes
- Cosine annealing with warmup for optimal learning rate scheduling
- Gradient clipping for training stability
- Progress tracking and checkpointing
- Support for multiple model sizes

Usage:
    from enigma.core.training import Trainer, train_model

    # Quick training
    train_model(data_path="data/data.txt", epochs=30, model_size="small")

    # Custom training
    trainer = Trainer(model, tokenizer, device="cuda")
    trainer.train(texts, epochs=30)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass

from .model import create_model, MODEL_PRESETS
from .tokenizer import get_tokenizer, train_tokenizer
from ..config import CONFIG

logger = logging.getLogger(__name__)

# Default paths
MODELS_DIR = Path(CONFIG.get("models_dir", "models"))
DATA_DIR = Path(CONFIG.get("data_dir", "data"))


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Training hyperparameters
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1

    # Learning rate schedule
    warmup_steps: int = 100
    min_lr: float = 1e-5

    # Gradient settings
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 4

    # Mixed precision
    use_amp: bool = True

    # Checkpointing
    save_every: int = 5  # Save every N epochs
    checkpoint_dir: Optional[str] = None

    # Logging
    log_every: int = 10  # Log every N steps
    verbose: bool = True

    # Sequence settings
    max_seq_len: int = 512

    def __post_init__(self):
        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(MODELS_DIR / "checkpoints")


# =============================================================================
# Dataset Classes
# =============================================================================

class TextDataset(Dataset):
    """
    Dataset for language model training.

    Creates sequences of fixed length for causal language modeling.
    Each sequence is used to predict the next token.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: Any,
        max_length: int = 512,
        stride: int = 256
    ):
        """
        Initialize dataset.

        Args:
            texts: List of training texts
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            stride: Step size when creating sequences (for overlap)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.sequences = []

        # Process texts
        for text in texts:
            self._process_text(text)

        logger.info(f"Created {len(self.sequences)} training sequences")

    def _process_text(self, text: str):
        """Process a text into training sequences."""
        # Encode text
        if hasattr(self.tokenizer, 'encode'):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        else:
            enc = self.tokenizer(text, add_special_tokens=False)
            ids = enc['input_ids']
            if hasattr(ids, 'tolist'):
                ids = ids.tolist()
            if isinstance(ids[0], list):
                ids = ids[0]

        # Create sequences with stride
        for i in range(0, max(1, len(ids) - self.max_length), self.stride):
            seq = ids[i:i + self.max_length + 1]  # +1 for target
            if len(seq) > 2:  # Need at least a few tokens
                self.sequences.append(seq)

        # Don't forget the last chunk
        if len(ids) > self.max_length:
            seq = ids[-self.max_length - 1:]
            if len(seq) > 2:
                self.sequences.append(seq)
        elif len(ids) > 2:
            self.sequences.append(ids)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]

        # Pad if needed
        if len(seq) < self.max_length + 1:
            pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
            seq = seq + [pad_id] * (self.max_length + 1 - len(seq))

        seq = seq[:self.max_length + 1]  # Truncate if needed

        # Input is all but last, target is all but first
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': target_ids
        }


class QADataset(Dataset):
    """
    Dataset for Q&A format training.

    Parses Q:/A: format and creates appropriate training examples.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: Any,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Parse Q&A pairs
        for text in texts:
            self._parse_qa(text)

        logger.info(f"Created {len(self.examples)} Q&A training examples")

    def _parse_qa(self, text: str):
        """Parse Q:/A: format into examples."""
        import re

        # Split on Q: markers
        parts = re.split(r'\n?Q:\s*', text, flags=re.IGNORECASE)

        for part in parts:
            if not part.strip():
                continue

            # Split into question and answer
            qa_split = re.split(r'\n?A:\s*', part, maxsplit=1, flags=re.IGNORECASE)

            if len(qa_split) == 2:
                question = qa_split[0].strip()
                answer = qa_split[1].strip()

                if question and answer:
                    # Create full example text
                    full_text = f"Q: {question}\nA: {answer}"
                    self.examples.append(full_text)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.examples[idx]

        # Encode
        if hasattr(self.tokenizer, 'encode'):
            ids = self.tokenizer.encode(text, add_special_tokens=True)
        else:
            enc = self.tokenizer(text, add_special_tokens=True)
            ids = enc['input_ids']
            if hasattr(ids, 'tolist'):
                ids = ids.tolist()
            if isinstance(ids[0], list):
                ids = ids[0]

        # Pad/truncate
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0)

        if len(ids) < self.max_length + 1:
            ids = ids + [pad_id] * (self.max_length + 1 - len(ids))
        ids = ids[:self.max_length + 1]

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': target_ids
        }


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

class CosineWarmupScheduler:
    """
    Cosine annealing with linear warmup.

    Learning rate schedule:
    1. Linear warmup from 0 to max_lr
    2. Cosine decay from max_lr to min_lr
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: float,
        min_lr: float = 1e-5
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Calculate current learning rate."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * self.current_step / self.warmup_steps

        # Cosine decay
        progress = (self.current_step - self.warmup_steps) / \
            max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)

        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """
    Production-grade trainer for Enigma models.

    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Cosine warmup scheduling
    - Gradient clipping
    - Checkpointing
    - Progress tracking
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            tokenizer: Tokenizer instance
            config: Training configuration
            device: Device to train on ("cuda" or "cpu")
        """
        self.config = config or TrainingConfig()

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Model and tokenizer
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        # Mixed precision scaler
        self.scaler = GradScaler() if self.config.use_amp and self.device.type == "cuda" else None

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.best_loss = float('inf')

        # Track losses for reporting
        self.loss_history = []

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  AMP enabled: {self.scaler is not None}")

    def train(
        self,
        texts: List[str],
        epochs: Optional[int] = None,
        dataset_type: str = "auto",
        callback: Optional[Callable[[Dict], None]] = None
    ) -> Dict[str, Any]:
        """
        Train the model on texts.

        Args:
            texts: List of training texts
            epochs: Number of epochs (overrides config)
            dataset_type: "text", "qa", or "auto" (detects from content)
            callback: Optional callback function called after each epoch

        Returns:
            Training metrics
        """
        epochs = epochs or self.config.epochs

        # Detect dataset type
        if dataset_type == "auto":
            sample = "\n".join(texts[:10])
            if "Q:" in sample or "A:" in sample:
                dataset_type = "qa"
            else:
                dataset_type = "text"

        # Create dataset
        if dataset_type == "qa":
            dataset = QADataset(
                texts,
                self.tokenizer,
                max_length=self.config.max_seq_len
            )
        else:
            dataset = TextDataset(
                texts,
                self.tokenizer,
                max_length=self.config.max_seq_len,
                stride=self.config.max_seq_len // 2
            )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Keep simple for compatibility
            pin_memory=self.device.type == "cuda"
        )

        # Calculate total steps
        steps_per_epoch = len(dataloader)
        total_steps = steps_per_epoch * epochs

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )

        # Initialize scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=min(self.config.warmup_steps, total_steps // 10),
            total_steps=total_steps,
            max_lr=self.config.learning_rate,
            min_lr=self.config.min_lr
        )

        # Print training info
        if self.config.verbose:
            print("=" * 60)
            print("ENIGMA TRAINING")
            print("=" * 60)
            print(f"  Device: {self.device}")
            print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"  Dataset size: {len(dataset):,} sequences")
            print(f"  Batch size: {self.config.batch_size}")
            print(f"  Gradient accumulation: {self.config.grad_accumulation_steps}")
            print(
                f"  Effective batch size: {
                    self.config.batch_size *
                    self.config.grad_accumulation_steps}")
            print(f"  Steps per epoch: {steps_per_epoch}")
            print(f"  Total steps: {total_steps}")
            print(f"  Epochs: {epochs}")
            print(f"  Learning rate: {self.config.learning_rate}")
            print(f"  AMP: {self.scaler is not None}")
            print("=" * 60)

        # Training loop
        start_time = time.time()
        self.loss_history = []

        for epoch in range(epochs):
            epoch_loss = self._train_epoch(dataloader, epoch, epochs)
            self.loss_history.append(epoch_loss)

            # Callback
            if callback:
                callback({
                    'epoch': epoch + 1,
                    'loss': epoch_loss,
                    'lr': self.optimizer.param_groups[0]['lr']
                })

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch + 1)

            # Track best loss
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss

        # Training complete
        elapsed = time.time() - start_time

        if self.config.verbose:
            print()
            print("=" * 60)
            print("TRAINING COMPLETE")
            print("=" * 60)
            print(f"  Total time: {elapsed:.1f}s")
            print(f"  Final loss: {self.loss_history[-1]:.4f}")
            print(f"  Best loss: {self.best_loss:.4f}")
            print("=" * 60)

        return {
            'final_loss': self.loss_history[-1],
            'best_loss': self.best_loss,
            'loss_history': self.loss_history,
            'elapsed_time': elapsed,
            'total_steps': self.global_step
        }

    def _train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        epoch_start = time.time()

        for step, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass with AMP
            if self.scaler is not None:
                with autocast():
                    logits = self.model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=getattr(self.tokenizer, 'pad_token_id', 0)
                    )
                    loss = loss / self.config.grad_accumulation_steps

                # Backward with scaling
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=getattr(self.tokenizer, 'pad_token_id', 0)
                )
                loss = loss / self.config.grad_accumulation_steps
                loss.backward()

            total_loss += loss.item() * self.config.grad_accumulation_steps
            num_batches += 1

            # Gradient accumulation step
            if (step + 1) % self.config.grad_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

                # Logging
                if self.config.verbose and self.global_step % self.config.log_every == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"  Step {self.global_step:,} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        # Handle remaining gradients
        if len(dataloader) % self.config.grad_accumulation_steps != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.global_step += 1

        epoch_loss = total_loss / max(1, num_batches)
        epoch_time = time.time() - epoch_start

        if self.config.verbose:
            print(f"Epoch {epoch + 1}/{total_epochs} | Loss: {epoch_loss:.4f} | Time: {epoch_time:.1f}s")

        return epoch_loss

    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_history[-1] if self.loss_history else None,
            'global_step': self.global_step,
            'config': self.config.__dict__
        }

        path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)

        logger.info(f"Saved checkpoint to {path}")

    def save_model(self, path: Union[str, Path]):
        """Save trained model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved model to {path}")


# =============================================================================
# Convenience Functions
# =============================================================================

def train_model(
    data_path: Optional[Union[str, Path]] = None,
    epochs: int = 30,
    model_size: str = "small",
    output_path: Optional[Union[str, Path]] = None,
    train_tokenizer_first: bool = True,
    force: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    High-level training function with comprehensive validation.

    Args:
        data_path: Path to training data file
        epochs: Number of training epochs (must be > 0)
        model_size: Model size preset ("tiny", "small", "medium", "large", "xl", "xxl")
        output_path: Where to save the trained model
        train_tokenizer_first: Whether to train tokenizer on data
        force: Train even if model exists
        **kwargs: Additional TrainingConfig parameters

    Returns:
        Training results dictionary with keys:
            - status: 'success', 'skipped', or 'failed'
            - model_path: Path to saved model
            - final_loss: Final training loss
            - epochs_completed: Number of epochs completed

    Raises:
        ValueError: If parameters are invalid (epochs, paths)
        TypeError: If parameter types are incorrect
        FileNotFoundError: If data file doesn't exist
        RuntimeError: If training, tokenization, or model creation fails
    """
    # Validate inputs
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    if not isinstance(model_size, str):
        raise TypeError(f"model_size must be a string, got {type(model_size).__name__}")

    # Default paths with validation
    if data_path is None:
        data_path = DATA_DIR / "data.txt"
    data_path = Path(data_path)

    if output_path is None:
        output_path = MODELS_DIR / f"{model_size}_enigma.pth"
    output_path = Path(output_path)

    # Validate data file exists and is readable
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            f"Please create a training data file or specify a valid path."
        )

    if not data_path.is_file():
        raise ValueError(f"data_path must be a file, got directory: {data_path}")

    # Check file size
    file_size = data_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Training data file is empty: {data_path}")

    if file_size < 100:
        logger.warning(
            f"Training data file is very small ({file_size} bytes). "
            f"Training may not be effective."
        )

    # Check if model exists
    if output_path.exists() and not force:
        print(f"Model already exists at {output_path}")
        print("Use force=True to retrain")
        return {"status": "skipped", "path": str(output_path)}

    # Ensure output directory exists
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise RuntimeError(f"Cannot create output directory: {e}") from e

    # Load training data with error handling
    try:
        texts = [data_path.read_text(encoding='utf-8')]
        logger.info(f"Loaded {len(texts[0]):,} characters from {data_path}")
    except (UnicodeDecodeError, IOError) as e:
        raise RuntimeError(f"Failed to read training data: {e}") from e

    # Train tokenizer if needed
    if train_tokenizer_first:
        print("Training tokenizer...")
        try:
            tokenizer = train_tokenizer(
                data_paths=[str(data_path)],
                vocab_size=8000,
                tokenizer_type="bpe"
            )
        except Exception as e:
            logger.error(f"Tokenizer training failed: {e}")
            raise RuntimeError(f"Tokenizer training failed: {e}") from e
    else:
        try:
            tokenizer = get_tokenizer()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer: {e}\n"
                f"Try setting train_tokenizer_first=True"
            ) from e

    # Create model with error handling
    print(f"Creating {model_size} model...")
    try:
        model = create_model(model_size, vocab_size=tokenizer.vocab_size)
    except (ValueError, RuntimeError) as e:
        raise RuntimeError(f"Model creation failed: {e}") from e

    # Create training config
    config = TrainingConfig(
        epochs=epochs,
        **{k: v for k, v in kwargs.items() if hasattr(TrainingConfig, k)}
    )

    # Train with error handling
    try:
        trainer = Trainer(model, tokenizer, config)
        results = trainer.train(texts)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise RuntimeError(f"Training failed: {e}") from e

    # Save model with error handling
    try:
        trainer.save_model(output_path)
        logger.info(f"Model saved to {output_path}")
    except (IOError, OSError) as e:
        raise RuntimeError(f"Failed to save model: {e}") from e

    # Save tokenizer alongside model
    tokenizer_path = output_path.parent / f"{output_path.stem}_tokenizer.json"
    if hasattr(tokenizer, 'save'):
        try:
            tokenizer.save(tokenizer_path)
            logger.info(f"Tokenizer saved to {tokenizer_path}")
        except Exception as e:
            logger.warning(f"Failed to save tokenizer: {e}")

    results['model_path'] = str(output_path)
    results['tokenizer_path'] = str(tokenizer_path)

    return results


def load_trained_model(
    model_path: Union[str, Path],
    device: Optional[str] = None
) -> tuple:
    """
    Load a trained model and tokenizer.

    Args:
        model_path: Path to saved model
        device: Device to load to

    Returns:
        (model, tokenizer) tuple
    """
    model_path = Path(model_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    state_dict = torch.load(model_path, map_location=device)

    # Infer model size from state dict
    # Look at embedding dimension
    embed_key = None
    for key in state_dict.keys():
        if 'embed' in key.lower() or 'token' in key.lower():
            embed_key = key
            break

    if embed_key:
        vocab_size, hidden_dim = state_dict[embed_key].shape
    else:
        # Default
        vocab_size = 8000
        hidden_dim = 512

    # Find matching preset or create custom
    model_size = "small"  # Default
    for name, preset in MODEL_PRESETS.items():
        preset_dim = preset.dim if hasattr(preset, 'dim') else preset.get('hidden_dim', 512)
        if preset_dim == hidden_dim:
            model_size = name
            break

    # Create and load model
    model = create_model(model_size, vocab_size=vocab_size)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer_path = model_path.parent / f"{model_path.stem}_tokenizer.json"
    if tokenizer_path.exists():
        from .advanced_tokenizer import AdvancedBPETokenizer
        tokenizer = AdvancedBPETokenizer(vocab_file=tokenizer_path)
    else:
        tokenizer = get_tokenizer()

    return model, tokenizer


# =============================================================================
# Backwards Compatibility
# =============================================================================

# Old MODEL_PATH constant
MODEL_PATH = MODELS_DIR / "tiny_enigma.pth"


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Main classes
    "Trainer",
    "TrainingConfig",
    "TextDataset",
    "QADataset",
    "CosineWarmupScheduler",

    # Functions
    "train_model",
    "load_trained_model",

    # Constants
    "MODELS_DIR",
    "DATA_DIR",
    "MODEL_PATH",
]
