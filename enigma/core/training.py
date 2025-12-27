"""
Training utilities for Enigma Language Models

Features:
  - Mixed precision training (FP16/BF16)
  - Gradient accumulation for large effective batch sizes
  - Learning rate scheduling (cosine, linear warmup)
  - Gradient checkpointing for memory efficiency
  - Automatic mixed precision (AMP)
  - Checkpoint saving and resumption
  - Training metrics logging

USAGE:
    from enigma.core.training import train_model
    
    train_model(
        data_path="data/training.txt",
        epochs=100,
        lr=1e-4,
        batch_size=8,
    )
"""
import torch
import torch.nn as nn
import math
from pathlib import Path
from typing import Optional, Callable, List
from datetime import datetime

from .model import Enigma, TinyEnigma  # TinyEnigma is backwards compat alias
from .tokenizer import load_tokenizer
from ..config import CONFIG

MODEL_PATH = Path(CONFIG["models_dir"]) / "enigma.pth"
LEGACY_PATH = Path(CONFIG["models_dir"]) / "tiny_enigma.pth"


def get_cosine_schedule(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Create a cosine learning rate scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(
    data_path: Optional[str] = None,
    data_text: Optional[str] = None,
    force: bool = False,
    num_epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 4,
    max_len: int = 512,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    use_amp: bool = True,
    save_every: int = 10,
    log_every: int = 10,
    callback: Optional[Callable] = None,
):
    """
    Train an Enigma model from scratch or continue training.
    
    Args:
        data_path: Path to training text file
        data_text: Raw training text (alternative to data_path)
        force: Force training even if model exists
        num_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        max_len: Maximum sequence length
        gradient_accumulation_steps: Accumulate gradients over N steps
        warmup_steps: Number of warmup steps for LR scheduler
        weight_decay: Weight decay for AdamW optimizer
        use_amp: Use automatic mixed precision
        save_every: Save checkpoint every N epochs
        log_every: Log metrics every N steps
        callback: Optional callback function called each epoch
        
    Returns:
        Trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"ENIGMA TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")
    print(f"Max sequence length: {max_len}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Mixed precision: {use_amp and device.type == 'cuda'}")
    print(f"{'='*60}\n")
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    vocab_size = getattr(tokenizer, "vocab_size", 32000)
    
    # Load training data
    if data_path:
        data_file = Path(data_path)
    else:
        data_file = Path(CONFIG["data_dir"]) / "data.txt"
    
    if data_file.exists():
        raw = data_file.read_text(encoding="utf-8")
    elif data_text:
        raw = data_text
    else:
        print("Warning: No training data found. Creating default dataset.")
        raw = "Hello world.\nThis is Enigma.\nI am learning to think and respond helpfully.\n" * 100
        data_file.parent.mkdir(parents=True, exist_ok=True)
        data_file.write_text(raw)
    
    # Tokenize data
    enc = tokenizer(raw, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    if isinstance(enc["input_ids"], list):
        all_ids = torch.tensor(enc["input_ids"], dtype=torch.long).squeeze()
    else:
        all_ids = enc["input_ids"].squeeze().long()
    
    # Create chunks for training
    chunks = []
    stride = max_len // 2  # 50% overlap
    for i in range(0, len(all_ids) - max_len, stride):
        chunks.append(all_ids[i:i + max_len])
    if len(all_ids) >= max_len:
        chunks.append(all_ids[-max_len:])
    elif len(all_ids) > 0:
        padded = torch.zeros(max_len, dtype=torch.long)
        padded[:len(all_ids)] = all_ids
        chunks.append(padded)
    
    chunks = torch.stack(chunks)
    print(f"Created {len(chunks)} training chunks of {max_len} tokens each")
    
    # Initialize model
    model = Enigma(
        vocab_size=vocab_size,
        dim=CONFIG.get("embed_dim", 256),
        depth=CONFIG.get("depth", 6),
        heads=CONFIG.get("heads", 8),
        max_len=max_len,
    ).to(device)
    
    # Load existing weights if available and not forcing
    if not force:
        for path in [MODEL_PATH, LEGACY_PATH]:
            if path.exists():
                try:
                    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=False)
                    print(f"Loaded existing weights from {path}")
                    break
                except Exception as e:
                    print(f"Could not load weights from {path}: {e}")
    
    print(f"Model parameters: {model.num_parameters:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    
    total_steps = (len(chunks) // batch_size) * num_epochs // gradient_accumulation_steps
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == "cuda" else None
    
    # Training loop
    model.train()
    global_step = 0
    best_loss = float('inf')
    training_history = []
    
    start_time = datetime.now()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        perm = torch.randperm(len(chunks))
        shuffled = chunks[perm]
        
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i:i + batch_size].to(device)
            
            # Input is all but last token, labels are all but first
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]
            
            # Forward pass with AMP
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    output = model(input_ids)
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    loss = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (i // batch_size + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
            else:
                output = model(input_ids)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                loss = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))
                loss = loss / gradient_accumulation_steps
                
                loss.backward()
                
                if (i // batch_size + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            
            if global_step % log_every == 0 and global_step > 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Step {global_step} | Loss: {loss.item() * gradient_accumulation_steps:.4f} | LR: {current_lr:.2e}")
        
        avg_loss = epoch_loss / max(num_batches, 1)
        training_history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "lr": scheduler.get_last_lr()[0],
            "timestamp": datetime.now().isoformat(),
        })
        
        print(f"Epoch {epoch + 1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0 or avg_loss < best_loss:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), MODEL_PATH.with_stem("enigma_best"))
                print(f"  New best model saved (loss: {best_loss:.4f})")
        
        # Callback
        if callback:
            callback(epoch + 1, avg_loss, model)
    
    # Final save
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Total time: {elapsed}")
    print(f"Final loss: {training_history[-1]['loss']:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"{'='*60}\n")
    
    return model


if __name__ == "__main__":
    train_model(num_epochs=10)
