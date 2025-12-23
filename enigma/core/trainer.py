"""
Advanced training system for Enigma models.

Features:
  - Train any named model from the registry
  - Multi-GPU support (DataParallel)
  - Checkpointing and resume
  - Training history tracking
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
    )
    
    # Train
    trainer.train(epochs=100, save_every=10)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Callable
import json
import os

from .model import TinyEnigma
from .model_registry import ModelRegistry
from .tokenizer import load_tokenizer
from ..config import CONFIG


class TextDataset(Dataset):
    """Simple text dataset for language model training."""
    
    def __init__(self, text: str, tokenizer, max_len: int = 512, stride: int = 256):
        """
        Args:
            text: Raw training text
            tokenizer: Tokenizer to encode text
            max_len: Maximum sequence length
            stride: Step size for sliding window (overlap = max_len - stride)
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Tokenize entire text
        enc = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
        if isinstance(enc["input_ids"], list):
            all_ids = torch.tensor(enc["input_ids"], dtype=torch.long).squeeze()
        else:
            all_ids = enc["input_ids"].squeeze().long()
        
        # Create sliding window chunks
        self.chunks = []
        for i in range(0, len(all_ids) - max_len, stride):
            self.chunks.append(all_ids[i:i + max_len])
        
        # Add final chunk if there's remaining data
        if len(all_ids) >= max_len:
            self.chunks.append(all_ids[-max_len:])
        elif len(all_ids) > 0:
            # Pad short text
            padded = torch.zeros(max_len, dtype=torch.long)
            padded[:len(all_ids)] = all_ids
            self.chunks.append(padded)
        
        pass  # Dataset info available via len()
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        # For language modeling: input and target are same, shifted by 1
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


class EnigmaTrainer:
    """
    Full-featured trainer for Enigma models.
    """
    
    def __init__(
        self,
        model: TinyEnigma,
        model_name: str,
        registry: ModelRegistry,
        data_path: Optional[str] = None,
        data_text: Optional[str] = None,
        use_multi_gpu: bool = False,
        device: Optional[str] = None,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        max_len: int = 512,
    ):
        """
        Args:
            model: The model to train
            model_name: Name in the registry
            registry: ModelRegistry instance
            data_path: Path to training text file
            data_text: Or provide text directly
            use_multi_gpu: Use all available GPUs
            device: Force specific device
            batch_size: Training batch size
            learning_rate: Initial learning rate
            max_len: Max sequence length
        """
        self.model_name = model_name
        self.registry = registry
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_len = max_len
        
        # Set up device(s)
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Multi-GPU setup
        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1
        if self.use_multi_gpu:
            self.model = nn.DataParallel(model)
        else:
            self.model = model
        
        self.model.to(self.device)
        
        # Load tokenizer
        self.tokenizer = load_tokenizer()
        
        # Load data
        if data_path:
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif data_text:
            text = data_text
        else:
            # Default tiny dataset
            text = "Hello world. This is Enigma. I am learning to think."
        
        self.dataset = TextDataset(text, self.tokenizer, max_len=max_len)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Set >0 for faster loading on PC
            pin_memory=torch.cuda.is_available(),
        )
        
        # Training state
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.current_epoch = 0
        self.global_step = 0
        self.training_history = []
    
    def train(
        self,
        epochs: int = 10,
        save_every: int = 10,
        log_every: int = 10,
        callbacks: Optional[List[Callable]] = None,
    ):
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            log_every: Print loss every N steps
            callbacks: Optional list of callback functions
        """
        print(f"\n[SYSTEM] {'='*50}")
        print(f"[SYSTEM] TRAINING: {self.model_name}")
        print(f"[SYSTEM] {'='*50}")
        print(f"[SYSTEM] Device: {self.device}")
        print(f"[SYSTEM] Multi-GPU: {self.use_multi_gpu}")
        print(f"[SYSTEM] Epochs: {epochs}")
        print(f"[SYSTEM] Batch size: {self.batch_size}")
        print(f"[SYSTEM] Learning rate: {self.learning_rate}")
        print(f"[SYSTEM] Dataset size: {len(self.dataset)} chunks")
        print(f"[SYSTEM] {'='*50}\n")
        
        start_time = datetime.now()
        
        for epoch in range(epochs):
            self.current_epoch += 1
            epoch_loss = 0.0
            num_batches = 0
            
            self.model.train()
            
            for batch_idx, batch in enumerate(self.dataloader):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids)
                
                # Compute loss
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1)
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (prevents exploding gradients)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                if self.global_step % log_every == 0:
                    print(f"[SYSTEM]   Step {self.global_step} | Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / max(num_batches, 1)
            
            # Log epoch
            self.training_history.append({
                "epoch": self.current_epoch,
                "loss": avg_loss,
                "timestamp": datetime.now().isoformat(),
            })
            
            print(f"[SYSTEM] Epoch {self.current_epoch}/{self.current_epoch + epochs - epoch - 1} | Avg Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if self.current_epoch % save_every == 0:
                self._save_checkpoint()
            
            # Run callbacks
            if callbacks:
                for cb in callbacks:
                    cb(self)
        
        # Final save
        self._save_checkpoint()
        self._save_final()
        
        elapsed = datetime.now() - start_time
        print(f"\n[SYSTEM] {'='*50}")
        print(f"[SYSTEM] TRAINING COMPLETE")
        print(f"[SYSTEM] Total time: {elapsed}")
        print(f"[SYSTEM] Final loss: {self.training_history[-1]['loss']:.4f}")
        print(f"[SYSTEM] {'='*50}\n")
    
    def _save_checkpoint(self):
        """Save a training checkpoint."""
        # Get the actual model (unwrap DataParallel if needed)
        model_to_save = self.model.module if self.use_multi_gpu else self.model
        
        self.registry.save_model(
            self.model_name,
            model_to_save,
            epoch=self.current_epoch,
            save_checkpoint=True
        )
    
    def _save_final(self):
        """Save final model and update metadata."""
        model_to_save = self.model.module if self.use_multi_gpu else self.model
        
        self.registry.save_model(self.model_name, model_to_save)
        self.registry.update_metadata(
            self.model_name,
            total_epochs=self.current_epoch,
            total_steps=self.global_step,
            training_history=self.training_history,
        )
    
    def resume_from_checkpoint(self, checkpoint_name: str):
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
        
        print(f"[SYSTEM] Resumed from checkpoint: {checkpoint_name}")


def train_model_by_name(
    name: str,
    data_path: str,
    epochs: int = 100,
    size: str = "small",
    use_multi_gpu: bool = False,
    **kwargs
):
    """
    Convenience function to create and train a new model.
    
    Example:
        train_model_by_name(
            "artemis",
            data_path="data/conversations.txt",
            epochs=100,
            size="medium",
            use_multi_gpu=True
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
        **kwargs
    )
    
    trainer.train(epochs=epochs)
    
    return model


if __name__ == "__main__":
    # Example usage
    print("[SYSTEM] EnigmaTrainer - Advanced Training System")
    print("[SYSTEM] Use train_model_by_name() or create an EnigmaTrainer instance")
