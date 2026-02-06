"""
Advanced Training System - Forge v2
====================================

A production-grade training system with:
  - Mixed precision training (AMP)
  - Gradient accumulation
  - Learning rate scheduling (cosine with warmup)
  - Gradient clipping
  - Early stopping
  - Model checkpointing
  - Training metrics tracking
  - Automatic batch sizing
"""
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

from .advanced_model import ForgeModel


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Learning rate
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 100

    # Batch settings
    batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # Training duration
    max_epochs: int = 100
    max_steps: Optional[int] = None

    # Regularization
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Mixed precision
    use_amp: bool = True

    # Checkpointing
    checkpoint_interval: int = 500
    eval_interval: int = 100

    # Early stopping
    patience: int = 10
    min_delta: float = 0.001

    # Logging
    log_interval: int = 10


class TextDataset(Dataset):
    """
    Dataset for language model training.

    Handles tokenization and creates input/target pairs
    for next-token prediction.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: Any,
        max_length: int = 512,
        stride: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # Tokenize all texts and create chunks
        for text in texts:
            tokens = tokenizer.encode(text)

            # Create overlapping chunks for better context
            for i in range(0, len(tokens) - max_length, stride):
                chunk = tokens[i:i + max_length + 1]  # +1 for target
                if len(chunk) > max_length // 2:  # Skip very short chunks
                    self.samples.append(chunk)

            # Handle last chunk
            if len(tokens) > max_length // 2:
                last_chunk = tokens[-max_length - 1:]
                if len(last_chunk) >= max_length // 2:
                    self.samples.append(last_chunk)

        print(f"Created dataset with {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.samples[idx]

        # Pad if necessary
        if len(tokens) < self.max_length + 1:
            tokens = tokens + [0] * (self.max_length + 1 - len(tokens))

        tokens = tokens[:self.max_length + 1]

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)

        return {'input_ids': input_ids, 'targets': targets}


class QADataset(Dataset):
    """
    Dataset for Q&A format training.

    Handles Q:/A: formatted data specifically.
    """

    def __init__(
        self,
        qa_pairs: list[dict[str, str]],
        tokenizer: Any,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for pair in qa_pairs:
            q = pair.get('question', pair.get('Q', ''))
            a = pair.get('answer', pair.get('A', ''))

            # Format as Q&A
            text = f"Q: {q}\nA: {a}"
            tokens = tokenizer.encode(text)

            if len(tokens) <= max_length:
                self.samples.append(tokens)

        print(f"Created Q&A dataset with {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.samples[idx]

        # Pad if necessary
        if len(tokens) < self.max_length + 1:
            tokens = tokens + [0] * (self.max_length + 1 - len(tokens))

        tokens = tokens[:self.max_length + 1]

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)

        return {'input_ids': input_ids, 'targets': targets}


class CosineWarmupScheduler:
    """
    Learning rate scheduler with linear warmup and cosine decay.

    This is the standard scheduler used by most modern LLMs.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
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
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / \
                max(1, self.max_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                (1 + math.cos(math.pi * progress))


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, loss: float) -> bool:
        """Check if training should stop."""
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


class Trainer:
    """
    Advanced trainer for Forge models.
    """

    def __init__(
        self,
        model: ForgeModel,
        config: TrainingConfig,
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model.to(self.device)

        # Setup optimizer with weight decay
        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

        # Don't apply weight decay to norms and biases
        decay_params = [p for n, p in param_dict.items() if 'norm' not in n and 'bias' not in n]
        nodecay_params = [p for n, p in param_dict.items() if 'norm' in n or 'bias' in n]

        optim_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.train_history = []

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        checkpoint_dir: Optional[Path] = None,
        callback: Optional[Callable] = None,
    ) -> dict[str, Any]:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            checkpoint_dir: Directory for saving checkpoints
            callback: Callback function called after each step

        Returns:
            Training history dictionary
        """
        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        # Calculate total steps
        steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        total_steps = self.config.max_steps or (self.config.max_epochs * steps_per_epoch)

        # Setup scheduler
        scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            max_steps=total_steps,
            min_lr=self.config.min_lr,
        )

        # Setup early stopping
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
        )

        # Create checkpoint directory
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Starting Training")
        print(f"{'=' * 60}")
        print(f"Device: {self.device}")
        print(f"Total steps: {total_steps:,}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        effective_batch = self.config.batch_size * self.config.gradient_accumulation_steps
        print(f"Effective batch size: {effective_batch}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"{'=' * 60}\n")

        # Training loop
        self.model.train()
        running_loss = 0.0
        step_times = []

        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0

            for batch_idx, batch in enumerate(train_loader):
                step_start = time.time()

                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['targets'].to(self.device)

                # Forward pass with mixed precision
                if self.config.use_amp:
                    with autocast(device_type=self.device.type):
                        logits, loss = self.model(input_ids, targets)
                        loss = loss / self.config.gradient_accumulation_steps

                    # Backward pass with scaling
                    self.scaler.scale(loss).backward()
                else:
                    logits, loss = self.model(input_ids, targets)
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()

                running_loss += loss.item()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    # Optimizer step
                    if self.config.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    scheduler.step()

                    self.global_step += 1

                    # Calculate step time
                    step_time = time.time() - step_start
                    step_times.append(step_time)

                    # Calculate average loss
                    avg_loss = running_loss
                    epoch_loss += avg_loss
                    epoch_steps += 1
                    running_loss = 0.0

                    # Record history
                    self.train_history.append({
                        'step': self.global_step,
                        'loss': avg_loss,
                        'lr': scheduler.get_lr(),
                        'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    })

                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        avg_time = sum(step_times[-10:]) / len(step_times[-10:])
                        eta = avg_time * (total_steps - self.global_step)

                        print(
                            f"Step {self.global_step:>6}/{total_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {scheduler.get_lr():.2e} | "
                            f"Time: {avg_time:.2f}s | "
                            f"ETA: {eta / 60:.1f}min"
                        )

                    # Callback
                    if callback:
                        callback(self.global_step, avg_loss, scheduler.get_lr())

                    # Checkpoint
                    if checkpoint_dir and self.global_step % self.config.checkpoint_interval == 0:
                        self.save_checkpoint(checkpoint_dir / f'checkpoint_{self.global_step}.pt')

                    # Evaluation
                    if val_dataset and self.global_step % self.config.eval_interval == 0:
                        val_loss = self.evaluate(val_dataset)
                        print(f"  Validation Loss: {val_loss:.4f}")

                        # Save best model
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            if checkpoint_dir:
                                self.save_checkpoint(checkpoint_dir / 'best_model.pt')

                        # Early stopping check
                        if early_stopping(val_loss):
                            print(f"\nEarly stopping triggered at step {self.global_step}")
                            return self._get_history()

                        self.model.train()

                    # Check max steps
                    if self.config.max_steps and self.global_step >= self.config.max_steps:
                        print(f"\nReached max steps ({self.config.max_steps})")
                        return self._get_history()

            # End of epoch
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            print(f"\n{'=' * 40}")
            print(f"Epoch {epoch + 1} complete | Avg Loss: {avg_epoch_loss:.4f}")
            print(f"{'=' * 40}\n")

        # Final checkpoint
        if checkpoint_dir:
            self.save_checkpoint(checkpoint_dir / 'final_model.pt')

        return self._get_history()

    @torch.no_grad()
    def evaluate(self, dataset: Dataset) -> float:
        """Evaluate model on dataset."""
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        total_loss = 0.0
        total_batches = 0

        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            targets = batch['targets'].to(self.device)

            _, loss = self.model(input_ids, targets)
            total_loss += loss.item()
            total_batches += 1

        return total_loss / max(total_batches, 1)

    def save_checkpoint(self, path: Path):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.model.config),
            'training_config': asdict(self.config),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'train_history': self.train_history[-1000:],  # Keep last 1000 entries
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        from .model_registry import safe_load_weights
        checkpoint = safe_load_weights(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.train_history = checkpoint.get('train_history', [])

        print(f"Loaded checkpoint from {path}")
        print(f"Resuming from step {self.global_step}, epoch {self.epoch}")

    def _get_history(self) -> dict[str, Any]:
        """Get training history."""
        return {
            'steps': [h['step'] for h in self.train_history],
            'losses': [h['loss'] for h in self.train_history],
            'learning_rates': [h['lr'] for h in self.train_history],
            'final_loss': self.train_history[-1]['loss'] if self.train_history else None,
            'best_loss': self.best_loss,
            'total_steps': self.global_step,
            'epochs': self.epoch + 1,
        }


def load_training_data(data_path: Path) -> list[str]:
    """Load training data from file."""
    texts = []

    if data_path.is_dir():
        # Load all text files in directory
        for file in data_path.glob('*.txt'):
            texts.append(file.read_text(encoding='utf-8'))
    else:
        # Load single file
        texts.append(data_path.read_text(encoding='utf-8'))

    return texts


def load_qa_data(data_path: Path) -> list[dict[str, str]]:
    """Load Q&A formatted data."""
    qa_pairs = []

    text = data_path.read_text(encoding='utf-8')

    # Parse Q: A: format
    lines = text.strip().split('\n')
    current_q = None
    current_a = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('Q:'):
            # Save previous pair
            if current_q and current_a:
                qa_pairs.append({
                    'question': current_q,
                    'answer': ' '.join(current_a)
                })

            current_q = line[2:].strip()
            current_a = []

        elif line.startswith('A:'):
            current_a.append(line[2:].strip())

        elif current_a:  # Continuation of answer
            current_a.append(line)

    # Don't forget last pair
    if current_q and current_a:
        qa_pairs.append({
            'question': current_q,
            'answer': ' '.join(current_a)
        })

    return qa_pairs
