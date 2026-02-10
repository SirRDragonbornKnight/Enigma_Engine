"""
DeepSpeed Integration

ZeRO optimization for distributed training with memory efficiency.
Supports ZeRO stages 1-3 and mixed precision training.

FILE: enigma_engine/core/deepspeed_integration.py
TYPE: Training
MAIN CLASSES: DeepSpeedConfig, DeepSpeedTrainer
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ZeROStage(Enum):
    """ZeRO optimization stages."""
    DISABLED = 0
    OPTIMIZER_STATES = 1  # Partition optimizer states
    GRADIENTS = 2  # + partition gradients
    PARAMETERS = 3  # + partition parameters


class FP16Mode(Enum):
    """Mixed precision modes."""
    DISABLED = "disabled"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class ZeROConfig:
    """ZeRO optimization configuration."""
    stage: ZeROStage = ZeROStage.GRADIENTS
    
    # Stage 3 specific
    offload_optimizer: bool = False  # CPU offload
    offload_param: bool = False  # Parameter CPU offload
    
    # Memory optimization
    reduce_bucket_size: int = 500_000_000
    allgather_bucket_size: int = 500_000_000
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    
    # Stage 3 settings
    sub_group_size: int = 1_000_000_000
    stage3_prefetch_bucket_size: int = 50_000_000
    stage3_param_persistence_threshold: int = 100_000
    stage3_max_live_parameters: int = 1_000_000_000
    stage3_max_reuse_distance: int = 1_000_000_000


@dataclass
class FP16Config:
    """Mixed precision configuration."""
    enabled: bool = True
    mode: FP16Mode = FP16Mode.FP16
    loss_scale: float = 0.0  # 0 = dynamic
    initial_scale_power: int = 16
    loss_scale_window: int = 1000
    hysteresis: int = 2
    min_loss_scale: float = 1.0


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str = "AdamW"
    params: dict[str, Any] = field(default_factory=lambda: {
        "lr": 3e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01
    })


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "WarmupLR"
    params: dict[str, Any] = field(default_factory=lambda: {
        "warmup_min_lr": 0,
        "warmup_max_lr": 3e-4,
        "warmup_num_steps": 100
    })


@dataclass
class DeepSpeedConfig:
    """Complete DeepSpeed configuration."""
    # Training
    train_batch_size: int = 32
    train_micro_batch_size_per_gpu: int = 4
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    
    # ZeRO
    zero: ZeROConfig = field(default_factory=ZeROConfig)
    
    # Mixed precision
    fp16: FP16Config = field(default_factory=FP16Config)
    
    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    # Scheduler
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Communication
    communication_data_type: str = "fp16"
    
    # Activation checkpointing
    activation_checkpointing: bool = False
    checkpoint_num_layers: int = 1
    
    # Misc
    steps_per_print: int = 100
    wall_clock_breakdown: bool = False
    dump_state: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to DeepSpeed config dictionary."""
        config = {
            "train_batch_size": self.train_batch_size,
            "train_micro_batch_size_per_gpu": self.train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.gradient_clipping,
            "steps_per_print": self.steps_per_print,
            "wall_clock_breakdown": self.wall_clock_breakdown,
        }
        
        # ZeRO config
        config["zero_optimization"] = {
            "stage": self.zero.stage.value,
            "reduce_bucket_size": self.zero.reduce_bucket_size,
            "allgather_bucket_size": self.zero.allgather_bucket_size,
            "overlap_comm": self.zero.overlap_comm,
            "contiguous_gradients": self.zero.contiguous_gradients
        }
        
        if self.zero.stage == ZeROStage.PARAMETERS:
            config["zero_optimization"].update({
                "stage3_prefetch_bucket_size": self.zero.stage3_prefetch_bucket_size,
                "stage3_param_persistence_threshold": self.zero.stage3_param_persistence_threshold,
                "stage3_max_live_parameters": self.zero.stage3_max_live_parameters,
                "stage3_max_reuse_distance": self.zero.stage3_max_reuse_distance
            })
            
            if self.zero.offload_optimizer:
                config["zero_optimization"]["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True
                }
            
            if self.zero.offload_param:
                config["zero_optimization"]["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True
                }
        
        # FP16/BF16 config
        if self.fp16.enabled:
            if self.fp16.mode == FP16Mode.FP16:
                config["fp16"] = {
                    "enabled": True,
                    "loss_scale": self.fp16.loss_scale,
                    "initial_scale_power": self.fp16.initial_scale_power,
                    "loss_scale_window": self.fp16.loss_scale_window,
                    "hysteresis": self.fp16.hysteresis,
                    "min_loss_scale": self.fp16.min_loss_scale
                }
            elif self.fp16.mode == FP16Mode.BF16:
                config["bf16"] = {"enabled": True}
                config["fp16"] = {"enabled": False}
        else:
            config["fp16"] = {"enabled": False}
        
        # Optimizer
        config["optimizer"] = {
            "type": self.optimizer.type,
            "params": self.optimizer.params
        }
        
        # Scheduler
        config["scheduler"] = {
            "type": self.scheduler.type,
            "params": self.scheduler.params
        }
        
        # Activation checkpointing
        if self.activation_checkpointing:
            config["activation_checkpointing"] = {
                "partition_activations": True,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": False,
                "number_checkpoints": self.checkpoint_num_layers
            }
        
        return config
    
    def save(self, path: Path):
        """Save config to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved DeepSpeed config to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'DeepSpeedConfig':
        """Load config from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        
        # Parse into config object
        config = cls()
        config.train_batch_size = data.get("train_batch_size", 32)
        config.train_micro_batch_size_per_gpu = data.get("train_micro_batch_size_per_gpu", 4)
        config.gradient_accumulation_steps = data.get("gradient_accumulation_steps", 1)
        
        zero_data = data.get("zero_optimization", {})
        config.zero.stage = ZeROStage(zero_data.get("stage", 2))
        
        return config


class DeepSpeedTrainer:
    """Trainer wrapper for DeepSpeed."""
    
    def __init__(self,
                 model: nn.Module,
                 config: DeepSpeedConfig,
                 training_data: Any = None):
        """
        Initialize DeepSpeed trainer.
        
        Args:
            model: Model to train
            config: DeepSpeed configuration
            training_data: Training dataset
        """
        self.config = config
        self._model = model
        self._training_data = training_data
        
        self._engine = None
        self._optimizer = None
        self._scheduler = None
        self._initialized = False
    
    def initialize(self):
        """Initialize DeepSpeed engine."""
        try:
            import deepspeed

            # Convert config to dict
            ds_config = self.config.to_dict()
            
            # Initialize DeepSpeed
            self._engine, self._optimizer, _, self._scheduler = deepspeed.initialize(
                model=self._model,
                config=ds_config,
                model_parameters=[p for p in self._model.parameters() if p.requires_grad]
            )
            
            self._initialized = True
            logger.info(f"DeepSpeed initialized with ZeRO stage {self.config.zero.stage.value}")
            
        except ImportError:
            logger.error("DeepSpeed not installed. Install with: pip install deepspeed")
            raise
    
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Perform a training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Step metrics
        """
        if not self._initialized:
            self.initialize()
        
        # Move batch to device
        batch = {k: v.to(self._engine.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self._engine(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # Backward pass
        self._engine.backward(loss)
        
        # Optimizer step
        self._engine.step()
        
        return {
            "loss": loss.item(),
            "lr": self._scheduler.get_last_lr()[0] if self._scheduler else 0
        }
    
    def save_checkpoint(self, path: str, tag: str = "latest"):
        """Save training checkpoint."""
        if self._engine:
            self._engine.save_checkpoint(path, tag)
            logger.info(f"Saved checkpoint to {path}/{tag}")
    
    def load_checkpoint(self, path: str, tag: str = "latest"):
        """Load training checkpoint."""
        if self._engine:
            self._engine.load_checkpoint(path, tag)
            logger.info(f"Loaded checkpoint from {path}/{tag}")
    
    def get_memory_stats(self) -> dict[str, float]:
        """Get memory usage statistics."""
        stats = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                stats[f"gpu_{i}_allocated_mb"] = torch.cuda.memory_allocated(i) / 1024**2
                stats[f"gpu_{i}_reserved_mb"] = torch.cuda.memory_reserved(i) / 1024**2
        
        return stats


def create_deepspeed_config(zero_stage: int = 2,
                            batch_size: int = 32,
                            learning_rate: float = 3e-4,
                            fp16: bool = True,
                            cpu_offload: bool = False) -> DeepSpeedConfig:
    """
    Create a DeepSpeed config with common settings.
    
    Args:
        zero_stage: ZeRO optimization stage (0-3)
        batch_size: Total batch size
        learning_rate: Learning rate
        fp16: Enable FP16 training
        cpu_offload: Enable CPU offloading (Stage 3)
        
    Returns:
        DeepSpeed configuration
    """
    config = DeepSpeedConfig(
        train_batch_size=batch_size,
        zero=ZeROConfig(
            stage=ZeROStage(zero_stage),
            offload_optimizer=cpu_offload and zero_stage == 3,
            offload_param=cpu_offload and zero_stage == 3
        ),
        fp16=FP16Config(enabled=fp16),
        optimizer=OptimizerConfig(
            params={"lr": learning_rate, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01}
        )
    )
    
    return config


def estimate_memory_usage(model: nn.Module,
                          config: DeepSpeedConfig,
                          num_gpus: int = 1) -> dict[str, float]:
    """
    Estimate memory usage with DeepSpeed.
    
    Args:
        model: Model to analyze
        config: DeepSpeed config
        num_gpus: Number of GPUs
        
    Returns:
        Memory estimates in GB
    """
    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = param_count * 4  # FP32
    
    # Base memory per GPU
    if config.fp16.enabled:
        model_memory = param_bytes / 2  # FP16
    else:
        model_memory = param_bytes
    
    # Optimizer states (Adam: 2x params for momentum and variance)
    optimizer_memory = param_bytes * 2
    
    # Gradients
    gradient_memory = param_bytes
    
    # ZeRO partitioning
    stage = config.zero.stage.value
    
    if stage >= 1:
        optimizer_memory /= num_gpus
    if stage >= 2:
        gradient_memory /= num_gpus
    if stage >= 3:
        model_memory /= num_gpus
    
    # CPU offload
    cpu_memory = 0
    if config.zero.offload_optimizer:
        cpu_memory += optimizer_memory
        optimizer_memory = 0
    if config.zero.offload_param:
        cpu_memory += model_memory
        model_memory = model_memory * 0.1  # Keep small buffer
    
    return {
        "model_gb": model_memory / 1024**3,
        "optimizer_gb": optimizer_memory / 1024**3,
        "gradients_gb": gradient_memory / 1024**3,
        "total_gpu_gb": (model_memory + optimizer_memory + gradient_memory) / 1024**3,
        "cpu_offload_gb": cpu_memory / 1024**3,
        "param_count_millions": param_count / 1e6
    }


__all__ = [
    'DeepSpeedConfig',
    'DeepSpeedTrainer',
    'ZeROConfig',
    'ZeROStage',
    'FP16Config',
    'FP16Mode',
    'OptimizerConfig',
    'SchedulerConfig',
    'create_deepspeed_config',
    'estimate_memory_usage'
]
