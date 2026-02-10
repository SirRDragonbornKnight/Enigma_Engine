"""
LoRA Fine-tuning Utilities
===========================

Utilities for training and managing LoRA (Low-Rank Adaptation) adapters.
Enables efficient fine-tuning with minimal parameters.

Usage:
    from enigma_engine.core.lora_utils import LoRATrainer, LoRAConfig
    
    config = LoRAConfig(rank=8, alpha=16, target_modules=['q_proj', 'v_proj'])
    trainer = LoRATrainer(model, config)
    
    trainer.train(dataset, epochs=10)
    trainer.save_adapter("my_lora.pth")
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA training."""
    rank: int = 8                           # LoRA rank (smaller = fewer params)
    alpha: float = 16.0                     # LoRA scaling factor
    dropout: float = 0.1                    # Dropout probability
    target_modules: list[str] = None        # Modules to apply LoRA to
    
    # Training
    learning_rate: float = 3e-4
    batch_size: int = 4
    epochs: int = 10
    warmup_steps: int = 100
    
    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default: apply to attention query and value projections
            self.target_modules = ['q_proj', 'v_proj']


class LoRATrainer:
    """
    LoRA fine-tuning trainer.
    
    Freezes base model and only trains LoRA adapter weights.
    """
    
    def __init__(self, model: nn.Module, config: LoRAConfig):
        """
        Initialize LoRA trainer.
        
        Args:
            model: Base model to fine-tune
            config: LoRA configuration
        """
        self.model = model
        self.config = config
        self.lora_modules = {}
        
        # Apply LoRA to target modules
        self._apply_lora()
        
        # Freeze base model
        self._freeze_base_model()
        
        logger.info(f"LoRA trainer initialized with rank={config.rank}")
        self._log_trainable_params()
    
    def _apply_lora(self):
        """Apply LoRA layers to target modules."""
        from enigma_engine.core.nn.experts import LoRALayer
        
        count = 0
        for name, module in self.model.named_modules():
            # Check if this module should have LoRA
            should_apply = any(target in name for target in self.config.target_modules)
            
            if should_apply and isinstance(module, nn.Linear):
                # Create LoRA layer
                lora = LoRALayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=self.config.rank,
                    alpha=self.config.alpha
                )
                
                # Register as a parameter of the model
                # We'll add the LoRA output to the original linear layer
                self.lora_modules[name] = lora
                
                # Add dropout if specified
                if self.config.dropout > 0:
                    lora = nn.Sequential(
                        lora,
                        nn.Dropout(self.config.dropout)
                    )
                
                count += 1
        
        logger.info(f"Applied LoRA to {count} modules")
    
    def _freeze_base_model(self):
        """Freeze all parameters except LoRA."""
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze LoRA parameters
        for lora in self.lora_modules.values():
            for param in lora.parameters():
                param.requires_grad = True
    
    def _log_trainable_params(self):
        """Log number of trainable parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        percentage = 100 * trainable_params / total_params if total_params > 0 else 0
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable (LoRA): {trainable_params:,} ({percentage:.2f}%)")
    
    def forward_with_lora(self, input_ids, attention_mask=None, **kwargs):
        """
        Forward pass with LoRA adapters applied.
        
        This hooks into the model's forward pass and adds LoRA contributions
        to the targeted linear layers (q_proj, v_proj, etc.).
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            **kwargs: Additional arguments passed to the model
            
        Returns:
            Model output with LoRA adaptations applied
        """
        # Store original forward methods
        original_forwards = {}
        
        # Temporarily replace forward methods of target modules to include LoRA
        for name, module in self.model.named_modules():
            if name in self.lora_modules and isinstance(module, nn.Linear):
                lora = self.lora_modules[name]
                original_forwards[name] = module.forward
                
                # Create a closure that applies LoRA
                def make_lora_forward(original_forward, lora_layer):
                    def lora_forward(x):
                        base_output = original_forward(x)
                        # Handle Sequential wrapper (dropout case)
                        if isinstance(lora_layer, nn.Sequential):
                            lora_output = lora_layer(x)
                        else:
                            lora_output = lora_layer(x)
                        return base_output + lora_output
                    return lora_forward
                
                module.forward = make_lora_forward(module.forward, lora)
        
        try:
            # Run the model forward pass
            outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        finally:
            # Restore original forward methods
            for name, module in self.model.named_modules():
                if name in original_forwards:
                    module.forward = original_forwards[name]
        
        return outputs
    
    def train(
        self,
        train_dataset,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        **kwargs
    ):
        """
        Train LoRA adapters.
        
        Args:
            train_dataset: Training dataset
            epochs: Number of epochs (default from config)
            learning_rate: Learning rate (default from config)
            **kwargs: Additional training arguments
        """
        epochs = epochs or self.config.epochs
        learning_rate = learning_rate or self.config.learning_rate
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Training loop
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in train_dataset:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.forward_with_lora(batch['input_ids'])
                
                # Calculate loss
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                elif isinstance(outputs, torch.Tensor):
                    # Compute cross-entropy loss manually if outputs are logits
                    if 'labels' in batch:
                        loss = F.cross_entropy(
                            outputs.view(-1, outputs.size(-1)),
                            batch['labels'].view(-1)
                        )
                    else:
                        # Use input_ids shifted as labels (language modeling)
                        labels = batch['input_ids'][:, 1:].contiguous()
                        logits = outputs[:, :-1, :].contiguous()
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1)
                        )
                else:
                    logger.warning("Could not compute loss - skipping batch")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.max_grad_norm
                    )
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def save_adapter(self, path: str):
        """
        Save only the LoRA adapter weights.
        
        Args:
            path: Path to save adapter
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save only LoRA parameters
        lora_state = {
            name: lora.state_dict()
            for name, lora in self.lora_modules.items()
        }
        
        save_dict = {
            'lora_state': lora_state,
            'config': self.config.__dict__,
        }
        
        torch.save(save_dict, path)
        logger.info(f"LoRA adapter saved to {path}")
    
    def load_adapter(self, path: str):
        """
        Load LoRA adapter weights.
        
        Args:
            path: Path to adapter file
        """
        from .model_registry import safe_load_weights
        checkpoint = safe_load_weights(path, map_location='cpu')
        
        lora_state = checkpoint['lora_state']
        for name, lora in self.lora_modules.items():
            if name in lora_state:
                lora.load_state_dict(lora_state[name])
        
        logger.info(f"LoRA adapter loaded from {path}")
    
    def merge_and_save(self, path: str):
        """
        Merge LoRA weights into base model and save.
        
        Args:
            path: Path to save merged model
        """
        from enigma_engine.core.nn.experts import LoRALayer

        # Merge LoRA weights into base model
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.lora_modules:
                lora = self.lora_modules[name]
                if isinstance(lora, LoRALayer):
                    # Merge LoRA weights
                    module.weight.data = lora.merge_weights(module.weight.data)
        
        # Save merged model
        torch.save(self.model.state_dict(), path)
        logger.info(f"Merged model saved to {path}")


def prepare_lora_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 512
):
    """
    Prepare dataset for LoRA fine-tuning.
    
    Args:
        data_path: Path to training data file
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Prepared dataset
    """
    # Load data
    with open(data_path, encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse Q&A pairs or conversation format
    samples = []
    current_q = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('Q:') or line.startswith('User:'):
            current_q = line.split(':', 1)[1].strip()
        elif (line.startswith('A:') or line.startswith('AI:')) and current_q:
            answer = line.split(':', 1)[1].strip()
            samples.append((current_q, answer))
            current_q = None
    
    logger.info(f"Prepared {len(samples)} training samples")
    return samples


def create_lora_config(
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: list[str] = None
) -> LoRAConfig:
    """
    Create a LoRA configuration with sensible defaults.
    
    Args:
        rank: LoRA rank (4-64, smaller = faster/less memory)
        alpha: Scaling factor (typically 2x rank)
        target_modules: Modules to apply LoRA to
        
    Returns:
        LoRAConfig instance
    """
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
    
    return LoRAConfig(
        rank=rank,
        alpha=alpha,
        target_modules=target_modules
    )


if __name__ == "__main__":
    # Example usage
    print("LoRA Fine-tuning Utilities")
    print("===========================")
    print()
    print("To use LoRA fine-tuning:")
    print("1. Load your base model")
    print("2. Create LoRAConfig with desired settings")
    print("3. Initialize LoRATrainer with model and config")
    print("4. Train with your dataset")
    print("5. Save adapter or merge with base model")
    print()
    print("Benefits of LoRA:")
    print("- 10-100x fewer trainable parameters")
    print("- Faster training")
    print("- Lower memory usage")
    print("- Multiple adapters for different tasks")
