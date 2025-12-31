"""
Advanced training system for Enigma models.

Features:
  - Train any named model from the registry
  - Multi-GPU support (DataParallel / DistributedDataParallel)
  - Mixed precision training (AMP)
  - Gradient accumulation
  - Learning rate scheduling (cosine with warmup)
  - Checkpointing and resume
  - Training history tracking
  - Early stopping
  - Configurable everything

USAGE:
    from enigma.core.trainer import EnigmaTrainer
    from enigma.core.model_registry import ModelRegistry

    registry = ModelRegistry()

    # Create a new model
    model = registry.create_model("artemis", size="small")

    # Set up trainer
    trainer = EnigmaTrainer(
        model=model,
        model_name="artemis",
        registry=registry,
        data_path="data/my_training_data.txt",
        use_multi_gpu=True,  # Use all available GPUs
        use_amp=True,        # Mixed precision training
    )

    # Train
    trainer.train(epochs=100, save_every=10)
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Callable, Dict, Any, Union
import json
import os

from .model import Enigma, TinyEnigma  # TinyEnigma is backwards compat alias
from .model_registry import ModelRegistry
from .tokenizer import load_tokenizer
from ..config import CONFIG


class TextDataset(Dataset):
    """
    Text dataset for language model training with sliding window chunking.

    Handles tokenization, chunking, and proper input/label pairs for causal LM training.
    """

    def __init__(
        self,
        text: str,
        tokenizer,
        max_len: int = 512,
        stride: Optional[int] = None,
        min_chunk_len: int = 32,
    ):
        """
        Args:
            text: Raw training text
            tokenizer: Tokenizer to encode text
            max_len: Maximum sequence length
            stride: Step size for sliding window (default: max_len // 2)
            min_chunk_len: Minimum chunk length to include
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        stride = stride or max_len // 2

        # Tokenize entire text
        enc = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
        if isinstance(enc["input_ids"], list):
            all_ids = torch.tensor(enc["input_ids"], dtype=torch.long).squeeze()
        else:
            all_ids = enc["input_ids"].squeeze().long()

        # Handle 0-dim tensor
        if all_ids.dim() == 0:
            all_ids = all_ids.unsqueeze(0)

        # Create sliding window chunks
        self.chunks = []
        for i in range(0, max(1, len(all_ids) - max_len + 1), stride):
            chunk = all_ids[i:i + max_len]
            if len(chunk) >= min_chunk_len:
                # Pad if necessary
                if len(chunk) < max_len:
                    padded = torch.zeros(max_len, dtype=torch.long)
                    padded[:len(chunk)] = chunk
                    chunk = padded
                self.chunks.append(chunk)

        # Handle very short text
        if len(self.chunks) == 0 and len(all_ids) > 0:
            padded = torch.zeros(max_len, dtype=torch.long)
            padded[:min(len(all_ids), max_len)] = all_ids[:max_len]
            self.chunks.append(padded)

        print(f"Created dataset with {len(self.chunks)} chunks of {max_len} tokens")
        print(f"Total tokens: {len(all_ids):,}")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        # For language modeling: input is all but last token, labels are all but first
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
            "attention_mask": (chunk[:-1] != 0).long(),  # Mask padding
        }


class EnigmaTrainer:
    """
    Full-featured trainer for Enigma models.

    Features:
    - Multi-GPU training
    - Mixed precision (AMP)
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpointing
    - Early stopping
    - Comprehensive logging
    """

    def __init__(
        self,
        model: Union[Enigma, TinyEnigma],
        model_name: str,
        registry: ModelRegistry,
        data_path: Optional[str] = None,
        data_text: Optional[str] = None,
        use_multi_gpu: bool = False,
        use_amp: bool = True,
        device: Optional[str] = None,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_len: int = 512,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
    ):
        """
        Args:
            model: The model to train
            model_name: Name in the registry
            registry: ModelRegistry instance
            data_path: Path to training text file
            data_text: Or provide text directly
            use_multi_gpu: Use all available GPUs
            use_amp: Use automatic mixed precision
            device: Force specific device
            batch_size: Training batch size
            learning_rate: Initial learning rate
            weight_decay: Weight decay for AdamW
            max_len: Max sequence length
            gradient_accumulation_steps: Accumulate gradients over N steps
            warmup_steps: Warmup steps for LR scheduler
            max_grad_norm: Max gradient norm for clipping
        """
        self.model_name = model_name
        self.registry = registry
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_len = max_len
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm

        # Set up device(s)
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            # Apply GPU memory limit from resource settings
            gpu_fraction = CONFIG.get("gpu_memory_fraction", 0.9)
            try:
                torch.cuda.set_per_process_memory_fraction(gpu_fraction)
            except BaseException:
                pass  # Older PyTorch versions may not support this
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Apply CPU thread limit from resource settings
        cpu_threads = CONFIG.get("cpu_threads", 0)
        if cpu_threads > 0:
            torch.set_num_threads(cpu_threads)

        # Print device info
        print(f"[Training] Device: {self.device}")
        if self.device.type == "cuda":
            print(f"[Training] GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"[Training] GPU Memory: {
                    torch.cuda.get_device_properties(0).total_memory //
                    1024**2} MB")
        print(f"[Training] CPU Threads: {torch.get_num_threads()}")

        # Mixed precision setup
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Multi-GPU setup
        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1
        if self.use_multi_gpu:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(model)
        else:
            self.model = model

        self.model.to(self.device)

        # Load tokenizer
        self.tokenizer = load_tokenizer()

        # Load data
        if data_path:
            data_file = Path(data_path)
            if data_file.exists():
                with open(data_file, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                raise FileNotFoundError(f"Training data not found: {data_path}")
        elif data_text:
            text = data_text
        else:
            # Default dataset
            text = "Hello world. This is Enigma. I am learning to think and respond helpfully.\n" * 100
            print("Warning: No training data provided. Using default dataset.")

        self.dataset = TextDataset(text, self.tokenizer, max_len=max_len)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set >0 for faster loading on PC
            pin_memory=torch.cuda.is_available(),
            drop_last=True,  # Avoid issues with small batches
        )

        # Training state
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.current_epoch = 0
        self.global_step = 0
        self.training_history: List[Dict[str, Any]] = []
        self.best_loss = float('inf')

    def _get_scheduler(self, num_training_steps: int):
        """Create cosine scheduler with warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            progress = float(step - self.warmup_steps) / \
                float(max(1, num_training_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(
        self,
        epochs: int = 10,
        save_every: int = 10,
        log_every: int = 10,
        eval_every: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        """
        Train the model.

        Args:
            epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            log_every: Print loss every N steps
            eval_every: Evaluate every N epochs (optional)
            early_stopping_patience: Stop if no improvement for N epochs
            callbacks: Optional list of callback functions
        """
        # Get model parameter count
        model_ref = self.model.module if self.use_multi_gpu else self.model
        param_count = sum(p.numel() for p in model_ref.parameters())

        print(f"\n{'=' * 60}")
        print(f"TRAINING: {self.model_name}")
        print(f"{'=' * 60}")
        print(f"Device: {self.device}")
        print(f"Multi-GPU: {self.use_multi_gpu}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Model Parameters: {param_count:,}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Dataset size: {len(self.dataset)} chunks")
        print(f"{'=' * 60}\n")

        # Calculate total steps and create scheduler
        steps_per_epoch = len(self.dataloader) // self.gradient_accumulation_steps
        total_steps = steps_per_epoch * epochs
        scheduler = self._get_scheduler(total_steps)

        start_time = datetime.now()
        no_improvement_count = 0

        for epoch in range(epochs):
            self.current_epoch += 1
            epoch_loss = 0.0
            num_batches = 0

            self.model.train()
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(self.dataloader):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass with AMP
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(input_ids)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        loss = self.criterion(
                            outputs.reshape(-1, outputs.size(-1)),
                            labels.reshape(-1)
                        )
                        loss = loss / self.gradient_accumulation_steps

                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.model(input_ids)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self.criterion(
                        outputs.reshape(-1, outputs.size(-1)),
                        labels.reshape(-1)
                    )
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()

                # Gradient accumulation step
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()

                    scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % log_every == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        print(
                            f"  Step {
                                self.global_step} | Loss: {
                                loss.item() *
                                self.gradient_accumulation_steps:.4f} | LR: {
                                current_lr:.2e}")

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)

            # Log epoch
            self.training_history.append({
                "epoch": self.current_epoch,
                "loss": avg_loss,
                "lr": scheduler.get_last_lr()[0],
                "timestamp": datetime.now().isoformat(),
            })

            print(f"Epoch {self.current_epoch}/{epochs} | Avg Loss: {avg_loss:.4f}")

            # Check for improvement
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                no_improvement_count = 0
                self._save_best()
            else:
                no_improvement_count += 1

            # Save checkpoint
            if self.current_epoch % save_every == 0:
                self._save_checkpoint()

            # Early stopping
            if early_stopping_patience and no_improvement_count >= early_stopping_patience:
                print(
                    f"\nEarly stopping triggered after {no_improvement_count} epochs without improvement")
                break

            # Run callbacks
            if callbacks:
                for cb in callbacks:
                    cb(self)

        # Final save
        self._save_checkpoint()
        self._save_final()

        elapsed = datetime.now() - start_time
        print(f"\n{'=' * 60}")
        print(f"TRAINING COMPLETE")
        print(f"Total time: {elapsed}")
        print(f"Final loss: {self.training_history[-1]['loss']:.4f}")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"{'=' * 60}\n")

    def _save_checkpoint(self):
        """Save a training checkpoint."""
        model_to_save = self.model.module if self.use_multi_gpu else self.model

        self.registry.save_model(
            self.model_name,
            model_to_save,
            epoch=self.current_epoch,
            save_checkpoint=True
        )
        print(f"  Checkpoint saved at epoch {self.current_epoch}")

    def _save_best(self):
        """Save the best model so far."""
        model_to_save = self.model.module if self.use_multi_gpu else self.model

        self.registry.save_model(
            self.model_name,
            model_to_save,
            epoch=self.current_epoch,
            save_checkpoint=True,
            checkpoint_name="best"
        )
        print(f"  New best model saved (loss: {self.best_loss:.4f})")

    def _save_final(self):
        """Save final model and update metadata."""
        model_to_save = self.model.module if self.use_multi_gpu else self.model

        self.registry.save_model(self.model_name, model_to_save)
        self.registry.update_metadata(
            self.model_name,
            total_epochs=self.current_epoch,
            total_steps=self.global_step,
            best_loss=self.best_loss,
            training_history=self.training_history,
            last_trained=datetime.now().isoformat(),
        )

    def resume_from_checkpoint(self, checkpoint_name: str = "best"):
        """Resume training from a checkpoint."""
        model, config = self.registry.load_model(
            self.model_name,
            device=str(self.device),
            checkpoint=checkpoint_name
        )

        if self.use_multi_gpu:
            self.model = nn.DataParallel(model)
        else:
            self.model = model

        # Try to restore epoch from checkpoint name
        if checkpoint_name.startswith("epoch_"):
            self.current_epoch = int(checkpoint_name.split("_")[1])

        print(f"Resumed from checkpoint: {checkpoint_name}")
        return self


def train_model_by_name(
    name: str,
    data_path: str,
    epochs: int = 100,
    size: str = "small",
    use_multi_gpu: bool = False,
    use_amp: bool = True,
    **kwargs
) -> Union[Enigma, TinyEnigma]:
    """
    Convenience function to create and train a new model.

    Example:
        train_model_by_name(
            "artemis",
            data_path="data/conversations.txt",
            epochs=100,
            size="medium",
            use_multi_gpu=True,
            use_amp=True,
        )
    """
    registry = ModelRegistry()

    # Create model if it doesn't exist
    if name not in registry.registry["models"]:
        model = registry.create_model(name, size=size)
    else:
        model, _ = registry.load_model(name)

    # Train
    trainer = EnigmaTrainer(
        model=model,
        model_name=name,
        registry=registry,
        data_path=data_path,
        use_multi_gpu=use_multi_gpu,
        use_amp=use_amp,
        **kwargs
    )

    trainer.train(epochs=epochs)

    return model


if __name__ == "__main__":
    # Example usage
    print("EnigmaTrainer - Advanced Training System")
    print("Use train_model_by_name() or create an EnigmaTrainer instance")
