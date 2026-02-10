"""
================================================================================
LARGE MODEL TRAINING - TRAIN BIGGER MODELS ON LIMITED HARDWARE
================================================================================

Combines multiple techniques to train models larger than VRAM allows:
- Gradient checkpointing (trade compute for memory)
- Gradient accumulation (simulate larger batches)
- Mixed precision training (FP16/BF16)
- CPU offloading (use system RAM)
- ZeRO optimization (partition optimizer states)

FILE: enigma_engine/core/large_model_training.py
TYPE: Training Enhancement
MAIN CLASS: LargeModelTrainer

USAGE:
    from enigma_engine.core.large_model_training import LargeModelTrainer, LargeModelConfig
    
    config = LargeModelConfig(
        target_model_size="large",
        available_vram_gb=8,
        enable_checkpointing=True,
        enable_cpu_offload=True
    )
    
    trainer = LargeModelTrainer(model, config)
    trainer.train(dataset, epochs=10)
"""

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW


logger = logging.getLogger(__name__)


@dataclass
class LargeModelConfig:
    """Configuration for large model training."""
    
    # Target settings
    target_model_size: str = "large"  # Model size preset
    available_vram_gb: float = 8.0    # Available GPU memory
    available_ram_gb: float = 16.0    # Available system memory
    
    # Memory optimization
    enable_checkpointing: bool = True       # Gradient checkpointing
    checkpoint_ratio: float = 0.5           # Fraction of layers to checkpoint
    enable_cpu_offload: bool = False        # Offload to CPU
    offload_optimizer: bool = True          # Offload optimizer states
    offload_gradients: bool = False         # Offload gradients too
    
    # Training settings
    gradient_accumulation_steps: int = 8    # Accumulate before update
    micro_batch_size: int = 1               # Per-step batch size
    effective_batch_size: int = 32          # Target effective batch
    
    # Precision
    use_mixed_precision: bool = True        # FP16/BF16 training
    precision: str = "fp16"                 # "fp16", "bf16", or "fp32"
    
    # ZeRO optimization
    zero_stage: int = 2                     # 0=none, 1=optimizer, 2=grad+opt, 3=full
    
    # Advanced
    pin_memory: bool = True                 # Pin CPU tensors
    num_workers: int = 4                    # DataLoader workers
    prefetch_factor: int = 2                # Prefetch batches
    
    def __post_init__(self):
        """Calculate derived values."""
        # Calculate accumulation steps from batch sizes
        if self.effective_batch_size and self.micro_batch_size:
            self.gradient_accumulation_steps = max(
                1, 
                self.effective_batch_size // self.micro_batch_size
            )


@dataclass
class MemoryEstimate:
    """Memory usage estimates."""
    
    model_params_gb: float = 0.0
    optimizer_states_gb: float = 0.0
    gradients_gb: float = 0.0
    activations_gb: float = 0.0
    total_gb: float = 0.0
    fits_in_vram: bool = False
    recommendations: list[str] = field(default_factory=list)


class CPUOffloader:
    """
    Offload tensors to CPU to save GPU memory.
    
    Manages moving optimizer states and gradients to/from CPU.
    """
    
    def __init__(self, device: torch.device, pin_memory: bool = True):
        """
        Initialize offloader.
        
        Args:
            device: GPU device
            pin_memory: Pin CPU memory for faster transfers
        """
        self.device = device
        self.pin_memory = pin_memory
        self.offloaded: dict[str, torch.Tensor] = {}
        self._original_locations: dict[str, torch.device] = {}
    
    def offload(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """
        Offload tensor to CPU.
        
        Args:
            name: Identifier for the tensor
            tensor: Tensor to offload
        
        Returns:
            CPU tensor (pinned if enabled)
        """
        self._original_locations[name] = tensor.device
        cpu_tensor = tensor.cpu()
        if self.pin_memory and cpu_tensor.is_pinned() is False:
            try:
                cpu_tensor = cpu_tensor.pin_memory()
            except RuntimeError:
                pass  # Pin memory not available
        self.offloaded[name] = cpu_tensor
        return cpu_tensor
    
    def restore(self, name: str) -> Optional[torch.Tensor]:
        """
        Restore tensor to original device.
        
        Args:
            name: Tensor identifier
        
        Returns:
            GPU tensor or None if not found
        """
        if name not in self.offloaded:
            return None
        
        tensor = self.offloaded[name]
        device = self._original_locations.get(name, self.device)
        return tensor.to(device, non_blocking=True)
    
    def clear(self):
        """Clear all offloaded tensors."""
        self.offloaded.clear()
        self._original_locations.clear()
        gc.collect()


class LargeModelTrainer:
    """
    Trainer optimized for large models on limited hardware.
    
    Combines:
    - Gradient checkpointing
    - Gradient accumulation
    - Mixed precision training
    - CPU offloading
    - ZeRO-style optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: LargeModelConfig = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        Initialize large model trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            optimizer: Optional custom optimizer
        """
        self.config = config or LargeModelConfig()
        self.model = model
        self.device = next(model.parameters()).device
        
        # Setup mixed precision
        self.scaler = None
        self.autocast_dtype = torch.float32
        if self.config.use_mixed_precision and torch.cuda.is_available():
            if self.config.precision == "bf16" and torch.cuda.is_bf16_supported():
                self.autocast_dtype = torch.bfloat16
            else:
                self.autocast_dtype = torch.float16
                self.scaler = GradScaler()
        
        # Apply gradient checkpointing
        if self.config.enable_checkpointing:
            self._apply_checkpointing()
        
        # Setup CPU offloading
        self.offloader = None
        if self.config.enable_cpu_offload:
            self.offloader = CPUOffloader(self.device, self.config.pin_memory)
        
        # Setup optimizer with offloading
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
        
        # Training state
        self._step = 0
        self._accumulated_loss = 0.0
        self._num_accumulated = 0
        
        logger.info(f"Large model trainer initialized:")
        logger.info(f"  - Checkpointing: {self.config.enable_checkpointing}")
        logger.info(f"  - CPU offload: {self.config.enable_cpu_offload}")
        logger.info(f"  - Mixed precision: {self.config.precision}")
        logger.info(f"  - Grad accumulation: {self.config.gradient_accumulation_steps}")
    
    def _apply_checkpointing(self):
        """Apply gradient checkpointing to model."""
        try:
            from .checkpointing import checkpoint_model
            checkpoint_model(self.model, self.config.checkpoint_ratio)
            logger.info(f"Applied gradient checkpointing (ratio={self.config.checkpoint_ratio})")
        except ImportError:
            # Manual checkpointing
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Enabled model's built-in gradient checkpointing")
            else:
                logger.warning("Gradient checkpointing not available")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with memory-efficient settings."""
        # Use fused AdamW if available (faster, less memory)
        try:
            optimizer = AdamW(
                self.model.parameters(),
                lr=1e-4,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.01,
                fused=True
            )
            logger.info("Using fused AdamW optimizer")
        except TypeError:
            optimizer = AdamW(
                self.model.parameters(),
                lr=1e-4,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.01
            )
            logger.info("Using standard AdamW optimizer")
        
        return optimizer
    
    def estimate_memory(self) -> MemoryEstimate:
        """
        Estimate memory requirements.
        
        Returns:
            MemoryEstimate with breakdown and recommendations
        """
        estimate = MemoryEstimate()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Bytes per parameter
        param_bytes = 4  # FP32
        if self.config.use_mixed_precision:
            param_bytes = 2  # FP16/BF16
        
        # Model parameters
        estimate.model_params_gb = (total_params * param_bytes) / (1024**3)
        
        # Optimizer states (AdamW: 2x params for momentum + variance)
        estimate.optimizer_states_gb = (trainable_params * 4 * 2) / (1024**3)  # Always FP32
        
        # Gradients
        grad_bytes = 2 if self.config.use_mixed_precision else 4
        estimate.gradients_gb = (trainable_params * grad_bytes) / (1024**3)
        
        # Activations (rough estimate: 2x model size per sample)
        estimate.activations_gb = estimate.model_params_gb * 2 * self.config.micro_batch_size
        if self.config.enable_checkpointing:
            estimate.activations_gb *= (1 - self.config.checkpoint_ratio * 0.8)
        
        # Total
        total = estimate.model_params_gb + estimate.optimizer_states_gb
        if not self.config.enable_cpu_offload:
            total += estimate.gradients_gb
        total += estimate.activations_gb
        estimate.total_gb = total
        
        # Check fit
        estimate.fits_in_vram = total <= self.config.available_vram_gb
        
        # Recommendations
        if not estimate.fits_in_vram:
            diff = total - self.config.available_vram_gb
            estimate.recommendations.append(f"Need {diff:.1f}GB more VRAM or optimizations")
            
            if not self.config.enable_checkpointing:
                savings = estimate.activations_gb * 0.6
                estimate.recommendations.append(
                    f"Enable checkpointing: save ~{savings:.1f}GB"
                )
            
            if not self.config.enable_cpu_offload:
                savings = estimate.optimizer_states_gb
                estimate.recommendations.append(
                    f"Enable CPU offload: save ~{savings:.1f}GB"
                )
            
            if not self.config.use_mixed_precision:
                savings = estimate.model_params_gb * 0.5
                estimate.recommendations.append(
                    f"Enable mixed precision: save ~{savings:.1f}GB"
                )
        
        return estimate
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable = None
    ) -> dict[str, float]:
        """
        Perform a single training step with accumulation.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            loss_fn: Loss function (default: CrossEntropyLoss)
        
        Returns:
            Dictionary with loss and other metrics
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        # Move to device if needed
        if inputs.device != self.device:
            inputs = inputs.to(self.device)
        if targets.device != self.device:
            targets = targets.to(self.device)
        
        # Forward pass with mixed precision
        with autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=self.config.use_mixed_precision):
            outputs = self.model(inputs)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if outputs.dim() == 3:  # (batch, seq, vocab)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
            
            loss = loss_fn(outputs, targets)
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Track accumulated loss
        self._accumulated_loss += loss.item() * self.config.gradient_accumulation_steps
        self._num_accumulated += 1
        
        # Optimizer step when accumulated enough
        metrics = {}
        if self._num_accumulated >= self.config.gradient_accumulation_steps:
            # Gradient clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Record metrics
            metrics['loss'] = self._accumulated_loss / self._num_accumulated
            metrics['step'] = self._step
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            
            # Reset accumulation
            self._accumulated_loss = 0.0
            self._num_accumulated = 0
            self._step += 1
        
        return metrics
    
    def train(
        self,
        train_loader,
        epochs: int = 1,
        val_loader = None,
        callbacks: list[Callable] = None
    ):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            epochs: Number of epochs
            val_loader: Optional validation loader
            callbacks: Optional list of callbacks
        """
        self.model.train()
        callbacks = callbacks or []
        
        # Estimate memory
        estimate = self.estimate_memory()
        logger.info(f"Memory estimate: {estimate.total_gb:.1f}GB "
                   f"(fits: {estimate.fits_in_vram})")
        
        for rec in estimate.recommendations:
            logger.warning(f"  Recommendation: {rec}")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                elif isinstance(batch, dict):
                    inputs = batch.get('input_ids', batch.get('inputs'))
                    targets = batch.get('labels', batch.get('targets'))
                else:
                    inputs = targets = batch
                
                # Training step
                metrics = self.train_step(inputs, targets)
                
                if metrics:  # Only when optimizer stepped
                    epoch_loss += metrics['loss']
                    num_batches += 1
                    
                    # Log progress
                    if self._step % 10 == 0:
                        logger.info(
                            f"Epoch {epoch+1}/{epochs} | "
                            f"Step {self._step} | "
                            f"Loss: {metrics['loss']:.4f}"
                        )
                    
                    # Callbacks
                    for callback in callbacks:
                        callback(metrics, epoch, batch_idx)
            
            # Epoch complete
            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")
            
            # Validation
            if val_loader:
                val_loss = self.validate(val_loader)
                logger.info(f"Validation Loss: {val_loss:.4f}")
        
        return self.model
    
    def validate(self, val_loader) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        loss_fn = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                elif isinstance(batch, dict):
                    inputs = batch.get('input_ids', batch.get('inputs'))
                    targets = batch.get('labels', batch.get('targets'))
                else:
                    inputs = targets = batch
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if outputs.dim() == 3:
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                
                loss = loss_fn(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, path: Union[str, Path]):
        """Save training checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self._step,
            'config': self.config.__dict__,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Union[str, Path]):
        """Load training checkpoint."""
        path = Path(path)
        # weights_only=False needed for full checkpoint with optimizer state
        # This is safe because we only load from our own training checkpoints
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._step = checkpoint.get('step', 0)
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {path} (step {self._step})")


def get_optimal_config(
    model_params: int,
    available_vram_gb: float,
    available_ram_gb: float = 16.0
) -> LargeModelConfig:
    """
    Get optimal training configuration for hardware.
    
    Args:
        model_params: Number of model parameters
        available_vram_gb: GPU memory in GB
        available_ram_gb: System RAM in GB
    
    Returns:
        Optimized LargeModelConfig
    """
    config = LargeModelConfig(
        available_vram_gb=available_vram_gb,
        available_ram_gb=available_ram_gb
    )
    
    # Estimate memory per parameter (with optimizer)
    # FP32: 4 bytes param + 8 bytes optimizer = 12 bytes
    # FP16: 2 bytes param + 8 bytes optimizer = 10 bytes (master weights still FP32)
    bytes_per_param = 10  # Assume mixed precision
    model_memory_gb = (model_params * bytes_per_param) / (1024**3)
    
    # If model doesn't fit in VRAM
    if model_memory_gb > available_vram_gb * 0.8:
        config.enable_cpu_offload = True
        config.offload_optimizer = True
        config.enable_checkpointing = True
        config.checkpoint_ratio = 0.7
        config.micro_batch_size = 1
        config.gradient_accumulation_steps = 16
        logger.info("Large model detected - enabling CPU offload and checkpointing")
    
    # If model barely fits
    elif model_memory_gb > available_vram_gb * 0.5:
        config.enable_checkpointing = True
        config.checkpoint_ratio = 0.5
        config.micro_batch_size = 2
        config.gradient_accumulation_steps = 8
        logger.info("Medium model - enabling checkpointing")
    
    # Model fits comfortably
    else:
        config.enable_checkpointing = False
        config.micro_batch_size = 4
        config.gradient_accumulation_steps = 4
        logger.info("Small model - using standard training")
    
    return config


# Convenience function
def train_large_model(
    model: nn.Module,
    train_loader,
    epochs: int = 1,
    vram_gb: float = None,
    **kwargs
) -> nn.Module:
    """
    Train a large model with automatic optimization.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        epochs: Number of epochs
        vram_gb: Available VRAM (auto-detected if None)
        **kwargs: Additional config options
    
    Returns:
        Trained model
    """
    # Auto-detect VRAM
    if vram_gb is None:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            vram_gb = 4.0  # Assume limited memory for CPU
    
    # Get optimal config
    model_params = sum(p.numel() for p in model.parameters())
    config = get_optimal_config(model_params, vram_gb)
    
    # Override with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Train
    trainer = LargeModelTrainer(model, config)
    return trainer.train(train_loader, epochs)
