"""
QLoRA Support

Quantized Low-Rank Adaptation for efficient fine-tuning of large models.
Enables training larger models with limited GPU memory.

FILE: enigma_engine/core/qlora.py
TYPE: Training
MAIN CLASSES: QLoRAConfig, QLoRAModel, QLoRATrainer
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Quantization types for base model."""
    INT4 = "int4"
    INT8 = "int8"
    FP16 = "fp16"
    BF16 = "bf16"


class LoRATarget(Enum):
    """Layers to target with LoRA."""
    QUERY = "q_proj"
    KEY = "k_proj"
    VALUE = "v_proj"
    OUTPUT = "o_proj"
    GATE = "gate_proj"
    UP = "up_proj"
    DOWN = "down_proj"


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA training."""
    # Quantization
    quantization: QuantizationType = QuantizationType.INT4
    double_quantization: bool = True
    quantization_dtype: str = "nf4"  # nf4 or fp4
    
    # LoRA parameters
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list[LoRATarget] = field(
        default_factory=lambda: [LoRATarget.QUERY, LoRATarget.VALUE]
    )
    
    # Training
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation: int = 4
    warmup_ratio: float = 0.03
    max_steps: int = 1000
    
    # Memory optimization
    gradient_checkpointing: bool = True
    max_memory_mb: int = 0  # 0 = auto-detect
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 rank: int = 8,
                 alpha: int = 16,
                 dropout: float = 0.0):
        """
        Initialize LoRA linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank (r)
            alpha: LoRA scaling factor
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Frozen base weight (loaded from quantized model)
        self.register_buffer(
            'weight',
            torch.zeros(out_features, in_features)
        )
        
        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA."""
        # Base forward (frozen)
        base_out = torch.nn.functional.linear(x, self.weight)
        
        # LoRA forward
        lora_out = self.dropout(x)
        lora_out = lora_out @ self.lora_A.T  # B x r
        lora_out = lora_out @ self.lora_B.T  # B x out
        lora_out = lora_out * self.scaling
        
        return base_out + lora_out
    
    def merge_weights(self) -> None:
        """Merge LoRA weights into base weight."""
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
    
    @property
    def trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return self.lora_A.numel() + self.lora_B.numel()


class QuantizedLinear(nn.Module):
    """4-bit quantized linear layer."""
    
    def __init__(self, in_features: int, out_features: int, bits: int = 4):
        """
        Initialize quantized linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bits: Quantization bits (4 or 8)
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Quantized weights stored as packed integers
        self.register_buffer(
            'quantized_weight',
            torch.zeros(out_features, in_features // (8 // bits), dtype=torch.uint8)
        )
        
        # Quantization scales and zeros
        self.register_buffer('scales', torch.ones(out_features))
        self.register_buffer('zeros', torch.zeros(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization."""
        # Dequantize weights
        weight = self._dequantize()
        return torch.nn.functional.linear(x, weight)
    
    def _dequantize(self) -> torch.Tensor:
        """Dequantize weights to float."""
        # Unpack quantized values
        if self.bits == 4:
            # Unpack 4-bit values
            weight_low = self.quantized_weight & 0x0F
            weight_high = (self.quantized_weight >> 4) & 0x0F
            weight = torch.stack([weight_low, weight_high], dim=-1)
            weight = weight.view(self.out_features, self.in_features)
        else:
            weight = self.quantized_weight.float()
        
        # Apply scales and zeros
        weight = (weight.float() - self.zeros.unsqueeze(1)) * self.scales.unsqueeze(1)
        
        return weight
    
    def quantize(self, weight: torch.Tensor) -> None:
        """Quantize a weight tensor."""
        # Simple symmetric quantization
        max_val = 2 ** (self.bits - 1) - 1
        min_val = -2 ** (self.bits - 1)
        
        # Per-output-channel quantization
        scale = weight.abs().max(dim=1).values / max_val
        scale = torch.clamp(scale, min=1e-8)
        
        # Quantize
        quantized = torch.round(weight / scale.unsqueeze(1))
        quantized = torch.clamp(quantized, min_val, max_val)
        
        # Store
        self.scales.copy_(scale)
        
        if self.bits == 4:
            # Pack 4-bit values
            q_int = (quantized - min_val).to(torch.uint8)
            q_low = q_int[:, 0::2]
            q_high = q_int[:, 1::2]
            self.quantized_weight.copy_(q_low | (q_high << 4))
        else:
            self.quantized_weight.copy_(quantized.to(torch.uint8))


class QLoRAModel(nn.Module):
    """Model wrapper with QLoRA adapters."""
    
    def __init__(self, base_model: nn.Module, config: QLoRAConfig):
        """
        Initialize QLoRA model.
        
        Args:
            base_model: Base model to adapt
            config: QLoRA configuration
        """
        super().__init__()
        
        self.config = config
        self.base_model = base_model
        self.lora_layers: dict[str, LoRALinear] = {}
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Add LoRA adapters
        self._add_lora_adapters()
    
    def _add_lora_adapters(self) -> None:
        """Add LoRA adapters to target modules."""
        target_names = [t.value for t in self.config.target_modules]
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this layer should have LoRA
                module_name = name.split('.')[-1]
                if module_name in target_names or self._should_adapt(name):
                    self._replace_with_lora(name, module)
    
    def _should_adapt(self, name: str) -> bool:
        """Check if a module should be adapted."""
        target_names = [t.value for t in self.config.target_modules]
        return any(t in name for t in target_names)
    
    def _replace_with_lora(self, name: str, module: nn.Linear) -> None:
        """Replace a linear layer with LoRA version."""
        lora = LoRALinear(
            module.in_features,
            module.out_features,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout
        )
        
        # Copy frozen weights
        lora.weight.copy_(module.weight.data)
        
        # Store reference
        self.lora_layers[name] = lora
        
        # Replace in model
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        if parent_name:
            parent = dict(self.base_model.named_modules())[parent_name]
            setattr(parent, child_name, lora)
        else:
            setattr(self.base_model, child_name, lora)
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through adapted model."""
        return self.base_model(*args, **kwargs)
    
    def get_trainable_params(self) -> int:
        """Get total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Get total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def merge_and_save(self, path: str) -> None:
        """Merge LoRA weights and save."""
        for lora in self.lora_layers.values():
            lora.merge_weights()
        
        torch.save(self.base_model.state_dict(), path)
        logger.info(f"Saved merged model to {path}")
    
    def save_lora_only(self, path: str) -> None:
        """Save only LoRA weights."""
        lora_state = {}
        for name, lora in self.lora_layers.items():
            lora_state[f"{name}.lora_A"] = lora.lora_A
            lora_state[f"{name}.lora_B"] = lora.lora_B
        
        torch.save(lora_state, path)
        logger.info(f"Saved LoRA weights to {path}")
    
    def load_lora(self, path: str) -> None:
        """Load LoRA weights."""
        lora_state = torch.load(path)
        
        for name, lora in self.lora_layers.items():
            if f"{name}.lora_A" in lora_state:
                lora.lora_A.copy_(lora_state[f"{name}.lora_A"])
            if f"{name}.lora_B" in lora_state:
                lora.lora_B.copy_(lora_state[f"{name}.lora_B"])
        
        logger.info(f"Loaded LoRA weights from {path}")


class QLoRATrainer:
    """Trainer for QLoRA fine-tuning."""
    
    def __init__(self, model: QLoRAModel, config: QLoRAConfig):
        """
        Initialize QLoRA trainer.
        
        Args:
            model: QLoRA model
            config: Training configuration
        """
        self.model = model
        self.config = config
        
        # Optimizer (only trainable params)
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate
        )
        
        self._step = 0
        self._accumulated = 0
    
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Perform a training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Step metrics
        """
        self.model.train()
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation
        
        # Backward pass
        loss.backward()
        
        self._accumulated += 1
        
        # Optimizer step
        if self._accumulated >= self.config.gradient_accumulation:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._accumulated = 0
            self._step += 1
        
        return {
            "loss": loss.item() * self.config.gradient_accumulation,
            "step": self._step
        }
    
    def get_memory_usage(self) -> dict[str, float]:
        """Get GPU memory usage."""
        if torch.cuda.is_available():
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved() / 1024**2
            }
        return {"allocated_mb": 0, "reserved_mb": 0}


def create_qlora_model(base_model: nn.Module,
                       rank: int = 64,
                       alpha: int = 16,
                       **kwargs) -> QLoRAModel:
    """
    Create a QLoRA-adapted model.
    
    Args:
        base_model: Base model to adapt
        rank: LoRA rank
        alpha: LoRA alpha
        **kwargs: Additional config parameters
        
    Returns:
        QLoRA-wrapped model
    """
    config = QLoRAConfig(
        lora_rank=rank,
        lora_alpha=alpha,
        **kwargs
    )
    
    return QLoRAModel(base_model, config)


__all__ = [
    'QLoRAConfig',
    'QLoRAModel',
    'QLoRATrainer',
    'LoRALinear',
    'QuantizedLinear',
    'QuantizationType',
    'LoRATarget',
    'create_qlora_model'
]
