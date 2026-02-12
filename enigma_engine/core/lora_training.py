"""
LoRA and QLoRA Training for enigma_engine

Low-Rank Adaptation (LoRA) enables efficient fine-tuning by:
- Freezing pre-trained weights
- Training small rank decomposition matrices
- 10-100x fewer trainable parameters
- No inference latency overhead (weights can be merged)

QLoRA adds:
- 4-bit quantized base model
- Double quantization
- Paged optimizers
- Fine-tune 65B models on single 48GB GPU

References:
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al.)
- "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al.)
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA training."""
    r: int = 8  # Rank of decomposition
    alpha: int = 16  # Scaling factor (effective lr = alpha/r * lr)
    dropout: float = 0.05  # Dropout on LoRA layers
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"  # "none", "all", or "lora_only"
    modules_to_save: list[str] = field(default_factory=list)  # Additional modules to train
    
    # QLoRA specific
    use_qlora: bool = False
    bits: int = 4  # Quantization bits (4 or 8)
    double_quant: bool = True  # Double quantization for QLoRA


class LoRALinear(nn.Module):
    """
    LoRA-adapted linear layer.
    
    Implements: output = W*x + (alpha/r) * B*A*x
    Where A is r x in_features, B is out_features x r
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = False
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Frozen original weight (will be set from pretrained)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize LoRA weights
        self.reset_lora_parameters()
        
        # For merged weights
        self._merged = False
    
    def reset_lora_parameters(self) -> None:
        """Initialize LoRA matrices."""
        # A uses Kaiming uniform, B is zero-initialized
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        """Create LoRA layer from existing linear layer."""
        lora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None
        )
        
        # Copy frozen weights
        lora.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            lora.bias.data.copy_(linear.bias.data)
        
        return lora
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._merged:
            return F.linear(x, self.weight, self.bias)
        
        # Original forward
        result = F.linear(x, self.weight, self.bias)
        
        # LoRA forward: x @ A^T @ B^T * scaling
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        
        return result + lora_output
    
    def merge_weights(self) -> None:
        """Merge LoRA weights into main weights for inference."""
        if not self._merged:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self._merged = True
    
    def unmerge_weights(self) -> None:
        """Unmerge LoRA weights for continued training."""
        if self._merged:
            self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self._merged = False
    
    def get_lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Get only the LoRA parameters."""
        state = {
            'lora_A': self.lora_A.data,
            'lora_B': self.lora_B.data,
        }
        if self.bias is not None:
            state['bias'] = self.bias.data
        return state
    
    def load_lora_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load LoRA parameters."""
        self.lora_A.data.copy_(state_dict['lora_A'])
        self.lora_B.data.copy_(state_dict['lora_B'])
        if 'bias' in state_dict and self.bias is not None:
            self.bias.data.copy_(state_dict['bias'])


class QLoRALinear(LoRALinear):
    """
    QLoRA-adapted linear layer with quantized base weights.
    
    Base weights are stored in 4-bit, LoRA adapters in full precision.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = False,
        bits: int = 4
    ) -> None:
        super().__init__(in_features, out_features, r, alpha, dropout, bias)
        self.bits = bits
        
        # Replace weight with quantized version
        del self.weight
        
        # Quantized weight storage
        self.register_buffer('qweight', torch.zeros(
            out_features, in_features // 2 if bits == 4 else in_features,
            dtype=torch.uint8
        ))
        self.register_buffer('scales', torch.ones(out_features, dtype=torch.float16))
        self.register_buffer('zeros', torch.zeros(out_features, dtype=torch.float16))
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int = 8, alpha: int = 16, 
                    dropout: float = 0.0, bits: int = 4):
        """Create QLoRA layer from existing linear layer."""
        qlora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
            bits=bits
        )
        
        # Quantize weights
        weight = linear.weight.data.float()
        
        # Per-channel quantization
        w_min = weight.min(dim=1).values
        w_max = weight.max(dim=1).values
        
        levels = 2 ** bits
        scales = (w_max - w_min) / (levels - 1)
        scales = scales.clamp(min=1e-8)
        
        # Quantize
        qweight = torch.round((weight - w_min.unsqueeze(1)) / scales.unsqueeze(1))
        qweight = qweight.clamp(0, levels - 1).to(torch.uint8)
        
        # Pack 4-bit weights
        if bits == 4:
            packed = torch.zeros(
                qweight.shape[0], qweight.shape[1] // 2,
                dtype=torch.uint8, device=qweight.device
            )
            packed = (qweight[:, 0::2] << 4) | qweight[:, 1::2]
            qlora.qweight.copy_(packed)
        else:
            qlora.qweight.copy_(qweight)
        
        qlora.scales.copy_(scales.half())
        qlora.zeros.copy_(w_min.half())
        
        if linear.bias is not None:
            qlora.bias.data.copy_(linear.bias.data)
        
        return qlora
    
    def _dequantize_weight(self) -> torch.Tensor:
        """Dequantize weights for computation."""
        if self.bits == 4:
            # Unpack 4-bit weights
            high = (self.qweight >> 4).to(torch.float32)
            low = (self.qweight & 0x0F).to(torch.float32)
            weight = torch.zeros(
                self.qweight.shape[0], self.qweight.shape[1] * 2,
                dtype=torch.float32, device=self.qweight.device
            )
            weight[:, 0::2] = high
            weight[:, 1::2] = low
        else:
            weight = self.qweight.float()
        
        # Dequantize
        weight = weight * self.scales.unsqueeze(1).float() + self.zeros.unsqueeze(1).float()
        return weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize base weights
        weight = self._dequantize_weight()
        
        # Original forward with dequantized weights
        result = F.linear(x, weight, self.bias)
        
        # LoRA forward
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        
        return result + lora_output


class LoRAModel(nn.Module):
    """
    Wrapper that adds LoRA adapters to a model.
    
    Usage:
        base_model = load_model("forge-large")
        lora_model = LoRAModel(base_model, config=LoRAConfig(r=8))
        
        # Train
        for batch in dataloader:
            loss = lora_model(batch)
            loss.backward()
            optimizer.step()
        
        # Save adapters only
        lora_model.save_adapters("lora_weights/")
        
        # Merge for inference
        lora_model.merge_adapters()
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[LoRAConfig] = None
    ) -> None:
        super().__init__()
        self.config = config or LoRAConfig()
        self.base_model = base_model
        self._lora_layers: dict[str, Union[LoRALinear, QLoRALinear]] = {}
        
        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Add LoRA adapters
        self._add_lora_adapters()
        
        logger.info(f"Created LoRA model with {self.num_trainable_params:,} trainable parameters")
    
    def _add_lora_adapters(self) -> None:
        """Replace target modules with LoRA versions."""
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this module should be adapted
                should_adapt = any(
                    target in name for target in self.config.target_modules
                )
                
                if should_adapt:
                    # Get parent module
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent = dict(self.base_model.named_modules())[parent_name]
                    else:
                        parent = self.base_model
                    
                    # Create LoRA layer
                    if self.config.use_qlora:
                        lora_layer = QLoRALinear.from_linear(
                            module,
                            r=self.config.r,
                            alpha=self.config.alpha,
                            dropout=self.config.dropout,
                            bits=self.config.bits
                        )
                    else:
                        lora_layer = LoRALinear.from_linear(
                            module,
                            r=self.config.r,
                            alpha=self.config.alpha,
                            dropout=self.config.dropout
                        )
                    
                    # Replace module
                    setattr(parent, child_name, lora_layer)
                    self._lora_layers[name] = lora_layer
                    
                    logger.debug(f"Added LoRA adapter to {name}")
        
        # Unfreeze additional modules if specified
        for module_name in self.config.modules_to_save:
            for name, param in self.base_model.named_parameters():
                if module_name in name:
                    param.requires_grad = True
    
    @property
    def num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property
    def num_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, *args, **kwargs):
        """Forward pass through the adapted model."""
        return self.base_model(*args, **kwargs)
    
    def merge_adapters(self) -> None:
        """Merge LoRA weights into base weights for inference."""
        for name, layer in self._lora_layers.items():
            if isinstance(layer, LoRALinear) and not isinstance(layer, QLoRALinear):
                layer.merge_weights()
                logger.debug(f"Merged LoRA weights for {name}")
    
    def unmerge_adapters(self) -> None:
        """Unmerge LoRA weights for continued training."""
        for name, layer in self._lora_layers.items():
            if isinstance(layer, LoRALinear) and not isinstance(layer, QLoRALinear):
                layer.unmerge_weights()
    
    def save_adapters(self, path: Union[str, Path]) -> None:
        """Save only the LoRA adapter weights."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        lora_state = {}
        for name, layer in self._lora_layers.items():
            layer_state = layer.get_lora_state_dict()
            for key, value in layer_state.items():
                lora_state[f"{name}.{key}"] = value
        
        torch.save(lora_state, path / "adapter_model.bin")
        
        # Save config
        config_dict = {
            'r': self.config.r,
            'alpha': self.config.alpha,
            'dropout': self.config.dropout,
            'target_modules': self.config.target_modules,
            'bias': self.config.bias,
            'use_qlora': self.config.use_qlora,
            'bits': self.config.bits
        }
        
        with open(path / "adapter_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved LoRA adapters to {path}")
    
    def load_adapters(self, path: Union[str, Path]) -> None:
        """Load LoRA adapter weights."""
        path = Path(path)
        
        # Load weights (weights_only=True for security against pickle attacks)
        lora_state = torch.load(path / "adapter_model.bin", map_location='cpu', weights_only=True)
        
        for name, layer in self._lora_layers.items():
            layer_state = {}
            for key in ['lora_A', 'lora_B', 'bias']:
                full_key = f"{name}.{key}"
                if full_key in lora_state:
                    layer_state[key] = lora_state[full_key]
            
            if layer_state:
                layer.load_lora_state_dict(layer_state)
        
        logger.info(f"Loaded LoRA adapters from {path}")
    
    def get_trainable_params(self):
        """Get only trainable parameters (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]


class LoRATrainer:
    """
    Trainer for LoRA fine-tuning.
    
    Usage:
        trainer = LoRATrainer(lora_model, tokenizer)
        trainer.train(dataset, epochs=3)
        trainer.save("output/lora_model")
    """
    
    def __init__(
        self,
        model: LoRAModel,
        tokenizer: Any,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 4
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Optimizer only for trainable params
        self.optimizer = torch.optim.AdamW(
            model.get_trainable_params(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    def train(
        self,
        dataset: list[str],
        epochs: int = 3,
        batch_size: int = 4,
        max_length: int = 512,
        log_interval: int = 10
    ) -> dict[str, list[float]]:
        """Train with LoRA."""
        device = next(self.model.parameters()).device
        self.model.train()
        
        history = {'loss': [], 'lr': []}
        global_step = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Simple batching
            for i in range(0, len(dataset), batch_size):
                batch_texts = dataset[i:i + batch_size]
                
                # Tokenize
                tokens = [self.tokenizer.encode(text)[:max_length] for text in batch_texts]
                max_len = max(len(t) for t in tokens)
                
                # Pad
                input_ids = torch.zeros(len(tokens), max_len, dtype=torch.long, device=device)
                for j, t in enumerate(tokens):
                    input_ids[j, :len(t)] = torch.tensor(t)
                
                # Forward
                outputs = self.model(input_ids[:, :-1])
                
                # Compute loss
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    input_ids[:, 1:].reshape(-1),
                    ignore_index=0
                )
                
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1
                
                # Update weights
                if num_batches % self.gradient_accumulation_steps == 0:
                    # Warmup
                    if global_step < self.warmup_steps:
                        lr_scale = (global_step + 1) / self.warmup_steps
                        for pg in self.optimizer.param_groups:
                            pg['lr'] = self.lr * lr_scale
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    if global_step % log_interval == 0:
                        avg_loss = epoch_loss / num_batches
                        logger.info(f"Epoch {epoch+1}, Step {global_step}, Loss: {avg_loss:.4f}")
                        history['loss'].append(avg_loss)
                        history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
        
        return history
    
    def save(self, path: str) -> None:
        """Save the LoRA model."""
        self.model.save_adapters(path)


def create_lora_model(
    base_model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    target_modules: Optional[list[str]] = None,
    use_qlora: bool = False,
    bits: int = 4
) -> LoRAModel:
    """
    Create a LoRA-adapted model.
    
    Args:
        base_model: Pre-trained model to adapt
        r: LoRA rank (higher = more capacity, more params)
        alpha: Scaling factor
        target_modules: Module names to adapt (default: q_proj, v_proj)
        use_qlora: Use 4-bit quantization for base model
        bits: Quantization bits for QLoRA
    
    Returns:
        LoRAModel ready for fine-tuning
    
    Example:
        model = load_model("forge-7b")
        lora_model = create_lora_model(model, r=16, use_qlora=True)
        
        # Only 0.1% of parameters are trainable
        print(f"Trainable: {lora_model.num_trainable_params:,}")
        print(f"Total: {lora_model.num_total_params:,}")
    """
    config = LoRAConfig(
        r=r,
        alpha=alpha,
        target_modules=target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
        use_qlora=use_qlora,
        bits=bits
    )
    
    return LoRAModel(base_model, config)
