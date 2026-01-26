"""
Model Quantization - Run larger models on smaller hardware.

Supports:
- Dynamic INT8 quantization (PyTorch native) - ~2x speedup
- 4-bit weight compression (simulated) - ~4x memory savings
- Mixed precision (FP16/BF16)
- Automatic device-aware quantization

Usage:
    from forge_ai.core.quantization import quantize_model, QuantConfig
    
    # Quick INT8 quantization
    model = quantize_model(model, bits=8)
    
    # Full configuration
    config = QuantConfig(bits=4, exclude_layers=['embed'])
    model = quantize_model(model, config=config)
    
    # Auto-detect best settings
    model = auto_quantize(model)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import warnings

logger = logging.getLogger(__name__)


@dataclass
class QuantConfig:
    """Configuration for model quantization."""
    
    # Bit width: 4, 8, or 16 (FP16)
    bits: int = 8
    
    # Layers to exclude from quantization
    exclude_layers: List[str] = field(default_factory=lambda: ['embed', 'ln_f', 'norm'])
    
    # Whether to quantize activations too
    quantize_activations: bool = False
    
    # Group size for grouped quantization (0 = per-tensor)
    group_size: int = 0
    
    @classmethod
    def for_device(cls, device_class: str) -> 'QuantConfig':
        """Get recommended config for device class."""
        configs = {
            'embedded': cls(bits=4, quantize_activations=True),
            'mobile': cls(bits=8),
            'laptop': cls(bits=8, exclude_layers=['embed', 'ln_f']),
            'desktop': cls(bits=16),
            'server': cls(bits=16),
        }
        return configs.get(device_class.lower(), cls())


class QuantizedLinear(nn.Module):
    """Linear layer with quantized weights."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        self.register_buffer('weight_int', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('scales', torch.ones(out_features, 1))
        self.register_buffer('zeros', torch.zeros(out_features, 1))
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 8) -> 'QuantizedLinear':
        """Create from existing Linear layer."""
        ql = cls(linear.in_features, linear.out_features, linear.bias is not None, bits)
        
        with torch.no_grad():
            weight = linear.weight.float()
            
            qmin, qmax = (-8, 7) if bits == 4 else (-128, 127)
            
            w_min = weight.min(dim=1, keepdim=True)[0]
            w_max = weight.max(dim=1, keepdim=True)[0]
            scale = (w_max - w_min) / (qmax - qmin)
            scale = scale.clamp(min=1e-8)
            zero = -w_min / scale + qmin
            
            ql.scales.copy_(scale)
            ql.zeros.copy_(zero)
            
            q_weight = torch.clamp(torch.round(weight / scale + zero), qmin, qmax).to(torch.int8)
            ql.weight_int.copy_(q_weight)
            
            if linear.bias is not None:
                ql.bias.copy_(linear.bias)
        
        return ql
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = (self.weight_int.float() - self.zeros) * self.scales
        weight = weight.to(x.dtype)
        return nn.functional.linear(x, weight, self.bias)
    
    def extra_repr(self) -> str:
        return f'in={self.in_features}, out={self.out_features}, bits={self.bits}'


def quantize_model(
    model: nn.Module,
    bits: int = 8,
    config: Optional[QuantConfig] = None,
    inplace: bool = False,
) -> nn.Module:
    """
    Quantize model to reduce memory usage.
    
    Args:
        model: Model to quantize
        bits: Bit width (4, 8, or 16)
        config: Full configuration
        inplace: Modify in place
        
    Returns:
        Quantized model
    """
    if config is None:
        config = QuantConfig(bits=bits)
    
    bits = config.bits
    
    if bits == 16:
        logger.info("Converting to FP16")
        return model.half() if inplace else model.clone().half()
    
    if bits not in (4, 8):
        raise ValueError(f"Unsupported bits: {bits}")
    
    if not inplace:
        import copy
        model = copy.deepcopy(model)
    
    logger.info(f"Quantizing to INT{bits}")
    
    n_quantized = 0
    
    def should_quantize(name: str) -> bool:
        return not any(ex.lower() in name.lower() for ex in config.exclude_layers)
    
    def replace_layers(module: nn.Module, name: str = ""):
        nonlocal n_quantized
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and should_quantize(full_name):
                setattr(module, child_name, QuantizedLinear.from_linear(child, bits))
                n_quantized += 1
            else:
                replace_layers(child, full_name)
    
    replace_layers(model)
    logger.info(f"Quantized {n_quantized} layers")
    
    return model


def dynamic_quantize(model: nn.Module) -> nn.Module:
    """Apply PyTorch dynamic INT8 quantization (CPU only)."""
    logger.info("Applying dynamic INT8 quantization")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def estimate_model_size(model: nn.Module, bits: int = 32) -> Dict[str, float]:
    """Estimate model memory usage."""
    params = sum(p.numel() for p in model.parameters())
    size_mb = (params * 4) / (1024 * 1024)
    quant_mb = (params * bits / 8) / (1024 * 1024)
    
    return {
        'params': params,
        'size_mb': size_mb,
        'quantized_mb': quant_mb,
        'compression': size_mb / max(quant_mb, 0.001),
    }


def auto_quantize(model: nn.Module, device_class: str = None) -> nn.Module:
    """Auto-select and apply best quantization."""
    if device_class is None:
        try:
            from .device_profiles import get_device_profiler
            device_class = get_device_profiler().classify().name.lower()
        except ImportError:
            device_class = 'desktop'
    
    config = QuantConfig.for_device(device_class)
    
    if config.bits == 16:
        return model.half()
    
    return quantize_model(model, config=config)


def load_quantized(path: str, dtype: str = "int8"):
    """Load and quantize a model."""
    from .model import create_model
    from .model_registry import safe_load_weights
    
    state_dict = safe_load_weights(path, map_location="cpu")
    model = create_model("auto")
    model.load_state_dict(state_dict, strict=False)
    
    bits = 8 if dtype == "int8" else 4 if dtype == "int4" else 16
    return quantize_model(model, bits=bits)


__all__ = [
    'QuantConfig',
    'QuantizedLinear',
    'quantize_model',
    'dynamic_quantize',
    'estimate_model_size',
    'auto_quantize',
    'load_quantized',
]
