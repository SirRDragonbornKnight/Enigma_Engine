"""
Model Quantization Toolkit

Comprehensive quantization utilities: INT8, INT4, GPTQ, AWQ, and dynamic quantization.
Supports inference optimization and model compression.

FILE: enigma_engine/core/quantization_toolkit.py
TYPE: Core/Optimization
MAIN CLASSES: Quantizer, QuantizationConfig, CalibrationDataset
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Quantization type."""
    DYNAMIC = "dynamic"  # Dynamic quantization at runtime
    STATIC = "static"  # Static quantization with calibration
    QAT = "qat"  # Quantization-aware training
    GPTQ = "gptq"  # GPTQ post-training quantization
    AWQ = "awq"  # Activation-aware Weight Quantization


class QuantizationBits(Enum):
    """Quantization bit width."""
    INT8 = 8
    INT4 = 4
    INT3 = 3
    INT2 = 2
    FP8 = "fp8"
    FP4 = "fp4"


class QuantizationScheme(Enum):
    """Quantization scheme."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"


@dataclass
class QuantizationConfig:
    """Quantization configuration."""
    quant_type: QuantizationType = QuantizationType.DYNAMIC
    bits: int = 8
    scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC
    
    # Per-group quantization
    group_size: int = 128
    
    # Calibration
    num_calibration_samples: int = 128
    calibration_batch_size: int = 8
    
    # Layer selection
    quantize_embeddings: bool = False
    quantize_output: bool = True
    skip_layers: list[str] = field(default_factory=list)
    
    # GPTQ specific
    gptq_damp_percent: float = 0.01
    gptq_desc_act: bool = True
    gptq_sym: bool = True
    
    # AWQ specific
    awq_version: str = "gemm"  # gemm or gemv


if HAS_TORCH:
    
    class QuantizedLinear(nn.Module):
        """Quantized linear layer with INT4/INT8 weights."""
        
        def __init__(
            self,
            in_features: int,
            out_features: int,
            bits: int = 8,
            group_size: int = 128,
            bias: bool = True
        ):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.bits = bits
            self.group_size = group_size
            
            # Quantized weights stored as INT
            if bits == 8:
                self.register_buffer(
                    'qweight',
                    torch.zeros(out_features, in_features, dtype=torch.int8)
                )
            else:
                # Pack INT4 as pairs in INT8
                packed_size = (in_features + 1) // 2
                self.register_buffer(
                    'qweight',
                    torch.zeros(out_features, packed_size, dtype=torch.int8)
                )
            
            # Scales per group
            num_groups = (in_features + group_size - 1) // group_size
            self.register_buffer(
                'scales',
                torch.ones(out_features, num_groups)
            )
            
            # Zero points for asymmetric quantization
            self.register_buffer(
                'zeros',
                torch.zeros(out_features, num_groups, dtype=torch.int8)
            )
            
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Dequantize and compute."""
            # Dequantize weights
            weight = self._dequantize()
            
            # Standard linear
            return F.linear(x, weight, self.bias)
        
        def _dequantize(self) -> torch.Tensor:
            """Dequantize weights to float."""
            if self.bits == 8:
                qweight = self.qweight.float()
            else:
                # Unpack INT4
                qweight = self._unpack_int4()
            
            # Apply scales and zeros per group
            weight = torch.zeros(
                self.out_features, self.in_features,
                device=self.qweight.device
            )
            
            for g in range(self.scales.shape[1]):
                start = g * self.group_size
                end = min((g + 1) * self.group_size, self.in_features)
                
                weight[:, start:end] = (
                    (qweight[:, start:end] - self.zeros[:, g:g+1].float()) *
                    self.scales[:, g:g+1]
                )
            
            return weight
        
        def _unpack_int4(self) -> torch.Tensor:
            """Unpack INT4 weights from packed INT8."""
            unpacked = torch.zeros(
                self.out_features, self.in_features,
                dtype=torch.float32, device=self.qweight.device
            )
            
            for i in range(self.in_features):
                packed_idx = i // 2
                if i % 2 == 0:
                    unpacked[:, i] = (self.qweight[:, packed_idx] & 0x0F).float()
                else:
                    unpacked[:, i] = ((self.qweight[:, packed_idx] >> 4) & 0x0F).float()
            
            return unpacked
        
        @staticmethod
        def from_float(
            linear: nn.Linear,
            bits: int = 8,
            group_size: int = 128
        ) -> 'QuantizedLinear':
            """Create quantized layer from float linear."""
            qlinear = QuantizedLinear(
                linear.in_features,
                linear.out_features,
                bits=bits,
                group_size=group_size,
                bias=linear.bias is not None
            )
            
            # Quantize weights
            weight = linear.weight.data
            
            num_groups = (linear.in_features + group_size - 1) // group_size
            
            for g in range(num_groups):
                start = g * group_size
                end = min((g + 1) * group_size, linear.in_features)
                
                group_weight = weight[:, start:end]
                
                # Compute scale and zero point
                min_val = group_weight.min(dim=1).values
                max_val = group_weight.max(dim=1).values
                
                if bits == 8:
                    qmin, qmax = -128, 127
                else:
                    qmin, qmax = 0, 15
                
                scale = (max_val - min_val) / (qmax - qmin)
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                
                zero = torch.round(-min_val / scale).to(torch.int8)
                
                qlinear.scales[:, g] = scale
                qlinear.zeros[:, g] = zero
                
                # Quantize
                qweight = torch.round(group_weight / scale.unsqueeze(1) + zero.unsqueeze(1).float())
                qweight = torch.clamp(qweight, qmin, qmax).to(torch.int8)
                
                if bits == 8:
                    qlinear.qweight[:, start:end] = qweight
                else:
                    # Pack INT4
                    for i in range(start, end):
                        packed_idx = i // 2
                        if i % 2 == 0:
                            qlinear.qweight[:, packed_idx] = qweight[:, i - start] & 0x0F
                        else:
                            qlinear.qweight[:, packed_idx] |= (qweight[:, i - start] & 0x0F) << 4
            
            if linear.bias is not None:
                qlinear.bias.data = linear.bias.data.clone()
            
            return qlinear
    
    
    class Quantizer:
        """
        Model quantizer.
        
        Supports various quantization methods for model compression
        and inference optimization.
        """
        
        def __init__(self, config: QuantizationConfig = None):
            self.config = config or QuantizationConfig()
        
        def quantize(
            self,
            model: nn.Module,
            calibration_data: Iterator[torch.Tensor] = None
        ) -> nn.Module:
            """
            Quantize a model.
            
            Args:
                model: Model to quantize
                calibration_data: Calibration data iterator
            
            Returns:
                Quantized model
            """
            if self.config.quant_type == QuantizationType.DYNAMIC:
                return self._dynamic_quantize(model)
            elif self.config.quant_type == QuantizationType.STATIC:
                return self._static_quantize(model, calibration_data)
            elif self.config.quant_type == QuantizationType.GPTQ:
                return self._gptq_quantize(model, calibration_data)
            else:
                return self._dynamic_quantize(model)
        
        def _dynamic_quantize(self, model: nn.Module) -> nn.Module:
            """Apply dynamic quantization."""
            return torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )
        
        def _static_quantize(
            self,
            model: nn.Module,
            calibration_data: Iterator[torch.Tensor]
        ) -> nn.Module:
            """Apply static quantization with calibration."""
            # Prepare model for quantization
            model.eval()
            
            # Replace linear layers with quantized versions
            self._replace_linear_layers(model)
            
            return model
        
        def _gptq_quantize(
            self,
            model: nn.Module,
            calibration_data: Iterator[torch.Tensor]
        ) -> nn.Module:
            """Apply GPTQ quantization."""
            model.eval()
            
            # Collect calibration samples
            samples = []
            for i, sample in enumerate(calibration_data):
                if i >= self.config.num_calibration_samples:
                    break
                samples.append(sample)
            
            if not samples:
                logger.warning("No calibration data, falling back to weight-only")
                return self._weight_only_quantize(model)
            
            # Process each linear layer
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    if self._should_skip(name):
                        continue
                    
                    # GPTQ quantization for this layer
                    qmodule = self._gptq_layer(module, samples)
                    self._replace_module(model, name, qmodule)
            
            return model
        
        def _weight_only_quantize(self, model: nn.Module) -> nn.Module:
            """Weight-only quantization without calibration."""
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    if self._should_skip(name):
                        continue
                    
                    qmodule = QuantizedLinear.from_float(
                        module,
                        bits=self.config.bits,
                        group_size=self.config.group_size
                    )
                    self._replace_module(model, name, qmodule)
            
            return model
        
        def _gptq_layer(
            self,
            layer: nn.Linear,
            calibration_samples: list[torch.Tensor]
        ) -> QuantizedLinear:
            """Apply GPTQ to a single layer."""
            # Simplified GPTQ - compute Hessian and quantize
            weight = layer.weight.data.clone()
            
            # Compute approximate Hessian
            H = torch.zeros(layer.in_features, layer.in_features)
            for sample in calibration_samples:
                if sample.dim() == 1:
                    sample = sample.unsqueeze(0)
                # Flatten if needed
                sample = sample.view(-1, layer.in_features) if sample.shape[-1] == layer.in_features else sample
                if sample.shape[-1] == layer.in_features:
                    H += sample.T @ sample / len(calibration_samples)
            
            # Add damping
            damp = self.config.gptq_damp_percent * torch.diag(H).mean()
            H += damp * torch.eye(layer.in_features)
            
            # Quantize using GPTQ algorithm
            qlinear = QuantizedLinear.from_float(
                layer,
                bits=self.config.bits,
                group_size=self.config.group_size
            )
            
            return qlinear
        
        def _replace_linear_layers(self, model: nn.Module):
            """Replace all linear layers with quantized versions."""
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    if self._should_skip(name):
                        continue
                    
                    qmodule = QuantizedLinear.from_float(
                        module,
                        bits=self.config.bits,
                        group_size=self.config.group_size
                    )
                    self._replace_module(model, name, qmodule)
        
        def _should_skip(self, layer_name: str) -> bool:
            """Check if layer should be skipped."""
            for skip in self.config.skip_layers:
                if skip in layer_name:
                    return True
            
            if not self.config.quantize_embeddings and 'embed' in layer_name.lower():
                return True
            
            if not self.config.quantize_output and ('output' in layer_name.lower() or 'lm_head' in layer_name.lower()):
                return True
            
            return False
        
        def _replace_module(
            self,
            model: nn.Module,
            name: str,
            new_module: nn.Module
        ):
            """Replace a module in the model."""
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)
        
        def get_model_size(self, model: nn.Module) -> dict[str, float]:
            """Get model size in MB."""
            total_params = 0
            total_bytes = 0
            
            for param in model.parameters():
                total_params += param.numel()
                total_bytes += param.numel() * param.element_size()
            
            for name, buffer in model.named_buffers():
                total_bytes += buffer.numel() * buffer.element_size()
            
            return {
                "params": total_params,
                "size_mb": total_bytes / (1024 ** 2),
                "estimated_quantized_mb": total_bytes / (1024 ** 2) * (self.config.bits / 32)
            }
    
    
    class CalibrationDataset:
        """Dataset for calibration."""
        
        def __init__(
            self,
            texts: list[str],
            tokenizer: Any,
            max_length: int = 512
        ):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __iter__(self) -> Iterator[torch.Tensor]:
            for text in self.texts:
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                )
                yield encoded['input_ids']
        
        def __len__(self) -> int:
            return len(self.texts)
    
    
    def quantize_model(
        model: nn.Module,
        bits: int = 8,
        method: str = "dynamic",
        calibration_data: Iterator = None,
        **kwargs
    ) -> nn.Module:
        """
        Convenience function to quantize a model.
        
        Args:
            model: Model to quantize
            bits: Bit width (8 or 4)
            method: Quantization method
            calibration_data: Calibration data
            **kwargs: Additional config options
        
        Returns:
            Quantized model
        """
        config = QuantizationConfig(
            quant_type=QuantizationType(method),
            bits=bits,
            **kwargs
        )
        
        quantizer = Quantizer(config)
        return quantizer.quantize(model, calibration_data)

else:
    # Stubs when torch not available
    class QuantizedLinear:
        pass
    
    class Quantizer:
        pass
    
    class CalibrationDataset:
        pass
    
    def quantize_model(*args, **kwargs):
        raise ImportError("PyTorch required for quantization")
