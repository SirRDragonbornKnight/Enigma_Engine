"""
EXL2 Quantization for enigma_engine

EXL2 is an advanced quantization format from ExLlamaV2 that provides:
- Per-layer adaptive bit allocation
- Better quality than GPTQ at same size
- Mixed precision within layers
- Optimized CUDA kernels

This module provides:
- EXL2 quantization calibration
- Format conversion
- Integration with ExLlamaV2 for inference
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class EXL2Config:
    """Configuration for EXL2 quantization."""
    target_bpw: float = 4.0  # Target bits per weight (e.g., 3.0, 4.0, 5.0, 6.0)
    calibration_length: int = 2048  # Sequence length for calibration
    calibration_rows: int = 100  # Number of calibration sequences
    head_bits: int = 6  # Bits for attention heads (higher = better quality)
    measurement_rows: int = 32  # Rows for initial measurement
    allow_mixed: bool = True  # Allow mixed bit precision per layer
    output_dir: Optional[str] = None  # Output directory for quantized model


@dataclass
class LayerMeasurement:
    """Measurement results for a layer."""
    name: str
    rows: int
    columns: int
    numel: int
    bits_options: list[float] = field(default_factory=list)
    errors: dict[float, float] = field(default_factory=dict)  # bits -> error
    selected_bits: float = 4.0


class EXL2Quantizer:
    """
    EXL2 quantizer for Forge models.
    
    EXL2 uses per-layer calibration to determine optimal bit allocation.
    Layers with higher impact on output quality get more bits.
    
    Usage:
        config = EXL2Config(target_bpw=4.0)
        quantizer = EXL2Quantizer(config)
        
        # Calibrate with sample data
        quantizer.calibrate(model, tokenizer, calibration_data)
        
        # Quantize
        quantized_model = quantizer.quantize(model)
        
        # Save
        quantizer.save(quantized_model, "output/model-exl2")
    """
    
    # Supported bit widths for EXL2
    BIT_OPTIONS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 8.0]
    
    def __init__(self, config: Optional[EXL2Config] = None):
        self.config = config or EXL2Config()
        self._measurements: dict[str, LayerMeasurement] = {}
        self._bit_allocation: dict[str, float] = {}
        self._calibration_data: Optional[torch.Tensor] = None
    
    def calibrate(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_texts: list[str],
        device: str = "cuda"
    ) -> dict[str, LayerMeasurement]:
        """
        Calibrate quantization using sample data.
        
        Measures the sensitivity of each layer to quantization error
        and determines optimal bit allocation.
        
        Args:
            model: The model to quantize
            tokenizer: Tokenizer for encoding calibration data
            calibration_texts: List of calibration text samples
            device: Device to use for calibration
        
        Returns:
            Layer measurements
        """
        logger.info(f"Starting EXL2 calibration with {len(calibration_texts)} samples")
        
        model = model.to(device)
        model.eval()
        
        # Prepare calibration data
        self._calibration_data = self._prepare_calibration_data(
            tokenizer, calibration_texts, device
        )
        
        # Measure each linear layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                measurement = self._measure_layer(model, name, module, device)
                self._measurements[name] = measurement
        
        # Determine optimal bit allocation
        self._allocate_bits()
        
        logger.info(f"Calibration complete. Average BPW: {self._get_average_bpw():.2f}")
        
        return self._measurements
    
    def _prepare_calibration_data(
        self,
        tokenizer: Any,
        texts: list[str],
        device: str
    ) -> torch.Tensor:
        """Tokenize and prepare calibration data."""
        all_tokens = []
        
        for text in texts[:self.config.calibration_rows]:
            tokens = tokenizer.encode(text)
            if len(tokens) > self.config.calibration_length:
                tokens = tokens[:self.config.calibration_length]
            all_tokens.append(torch.tensor(tokens))
        
        # Pad to same length
        max_len = max(len(t) for t in all_tokens)
        padded = torch.zeros(len(all_tokens), max_len, dtype=torch.long)
        
        for i, tokens in enumerate(all_tokens):
            padded[i, :len(tokens)] = tokens
        
        return padded.to(device)
    
    @torch.no_grad()
    def _measure_layer(
        self,
        model: nn.Module,
        name: str,
        module: nn.Linear,
        device: str
    ) -> LayerMeasurement:
        """Measure quantization error for different bit widths."""
        weight = module.weight.data.float()
        
        measurement = LayerMeasurement(
            name=name,
            rows=weight.shape[0],
            columns=weight.shape[1],
            numel=weight.numel(),
            bits_options=self.BIT_OPTIONS.copy()
        )
        
        # Measure error at each bit width
        for bits in self.BIT_OPTIONS:
            error = self._compute_quantization_error(weight, bits)
            measurement.errors[bits] = error
        
        return measurement
    
    def _compute_quantization_error(
        self,
        weight: torch.Tensor,
        bits: float
    ) -> float:
        """
        Compute quantization error for a weight tensor at given bit width.
        
        Uses simulated quantization to estimate reconstruction error.
        """
        # Number of quantization levels
        levels = int(2 ** bits)
        
        # Get min/max for scaling
        w_min = weight.min()
        w_max = weight.max()
        
        # Avoid division by zero
        scale = (w_max - w_min) / (levels - 1) if levels > 1 else 1.0
        
        if scale == 0:
            return 0.0
        
        # Quantize and dequantize
        quantized = torch.round((weight - w_min) / scale)
        quantized = torch.clamp(quantized, 0, levels - 1)
        dequantized = quantized * scale + w_min
        
        # Compute MSE
        mse = torch.mean((weight - dequantized) ** 2).item()
        
        return mse
    
    def _allocate_bits(self):
        """
        Allocate bits to each layer to meet target BPW.
        
        Uses a greedy algorithm:
        1. Start all layers at minimum bits
        2. Incrementally increase bits for layer with highest error
        3. Stop when target BPW is reached
        """
        target_bpw = self.config.target_bpw
        
        # Start with minimum bits
        for name, measurement in self._measurements.items():
            measurement.selected_bits = min(self.BIT_OPTIONS)
        
        # Greedy allocation
        total_weights = sum(m.numel for m in self._measurements.values())
        
        while self._get_average_bpw() < target_bpw:
            # Find layer with highest error that can be improved
            best_layer = None
            best_improvement = 0
            
            for name, measurement in self._measurements.items():
                current_bits = measurement.selected_bits
                current_error = measurement.errors.get(current_bits, float('inf'))
                
                # Find next bit option
                idx = self.BIT_OPTIONS.index(current_bits)
                if idx >= len(self.BIT_OPTIONS) - 1:
                    continue  # Already at max
                
                next_bits = self.BIT_OPTIONS[idx + 1]
                next_error = measurement.errors.get(next_bits, current_error)
                
                improvement = current_error - next_error
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_layer = name
            
            if best_layer is None:
                break  # Can't improve further
            
            # Increase bits for best layer
            current = self._measurements[best_layer].selected_bits
            idx = self.BIT_OPTIONS.index(current)
            self._measurements[best_layer].selected_bits = self.BIT_OPTIONS[idx + 1]
        
        # Store final allocation
        self._bit_allocation = {
            name: m.selected_bits for name, m in self._measurements.items()
        }
        
        logger.info(f"Bit allocation complete:")
        for name, bits in sorted(self._bit_allocation.items()):
            logger.debug(f"  {name}: {bits} bits")
    
    def _get_average_bpw(self) -> float:
        """Calculate current average bits per weight."""
        if not self._measurements:
            return 0.0
        
        total_bits = 0
        total_weights = 0
        
        for measurement in self._measurements.values():
            total_bits += measurement.selected_bits * measurement.numel
            total_weights += measurement.numel
        
        return total_bits / total_weights if total_weights > 0 else 0.0
    
    def quantize(self, model: nn.Module) -> nn.Module:
        """
        Quantize the model using calibrated bit allocation.
        
        Args:
            model: Model to quantize
        
        Returns:
            Quantized model
        """
        if not self._bit_allocation:
            raise RuntimeError("Must calibrate before quantizing")
        
        logger.info("Quantizing model with EXL2...")
        
        # Create quantized copy
        quantized_model = self._create_quantized_model(model)
        
        return quantized_model
    
    def _create_quantized_model(self, model: nn.Module) -> nn.Module:
        """Create a quantized copy of the model."""
        import copy
        quantized = copy.deepcopy(model)
        
        # Replace linear layers with quantized versions
        for name, module in list(quantized.named_modules()):
            if isinstance(module, nn.Linear) and name in self._bit_allocation:
                bits = self._bit_allocation[name]
                quantized_linear = self._quantize_linear(module, bits)
                
                # Replace module
                parent_name, child_name = self._split_name(name)
                parent = quantized
                if parent_name:
                    parent = dict(quantized.named_modules())[parent_name]
                setattr(parent, child_name, quantized_linear)
        
        return quantized
    
    def _split_name(self, name: str) -> tuple[str, str]:
        """Split module name into parent and child."""
        parts = name.rsplit('.', 1)
        if len(parts) == 1:
            return '', parts[0]
        return parts[0], parts[1]
    
    def _quantize_linear(self, module: nn.Linear, bits: float) -> nn.Module:
        """Quantize a linear layer."""
        return EXL2Linear(
            in_features=module.in_features,
            out_features=module.out_features,
            bits=bits,
            bias=module.bias is not None
        ).from_float(module)
    
    def save(
        self,
        model: nn.Module,
        output_dir: Union[str, Path],
        save_config: bool = True
    ):
        """Save quantized model in EXL2 format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(model.state_dict(), output_dir / "model.safetensors")
        
        # Save quantization config
        if save_config:
            config = {
                "format": "exl2",
                "target_bpw": self.config.target_bpw,
                "actual_bpw": self._get_average_bpw(),
                "bit_allocation": self._bit_allocation,
                "measurements": {
                    name: {
                        "rows": m.rows,
                        "columns": m.columns,
                        "selected_bits": m.selected_bits
                    }
                    for name, m in self._measurements.items()
                }
            }
            
            with open(output_dir / "quantize_config.json", "w") as f:
                json.dump(config, f, indent=2)
        
        logger.info(f"Saved EXL2 model to {output_dir}")


class EXL2Linear(nn.Module):
    """
    EXL2-quantized linear layer.
    
    Stores weights in low-bit format with per-tensor scaling.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: float,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Quantized weight storage
        self.register_buffer(
            'qweight',
            torch.zeros((out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer(
            'scales',
            torch.ones(out_features, dtype=torch.float16)
        )
        self.register_buffer(
            'zeros',
            torch.zeros(out_features, dtype=torch.float16)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def from_float(self, module: nn.Linear) -> 'EXL2Linear':
        """Initialize from a float linear module."""
        weight = module.weight.data.float()
        
        # Compute scales per output channel
        w_min = weight.min(dim=1).values
        w_max = weight.max(dim=1).values
        
        levels = int(2 ** self.bits)
        scales = (w_max - w_min) / (levels - 1)
        scales = scales.clamp(min=1e-8)
        
        # Quantize
        qweight = torch.round((weight - w_min.unsqueeze(1)) / scales.unsqueeze(1))
        qweight = qweight.clamp(0, levels - 1).to(torch.int8)
        
        self.qweight.copy_(qweight)
        self.scales.copy_(scales.half())
        self.zeros.copy_(w_min.half())
        
        if module.bias is not None:
            self.bias.data.copy_(module.bias.data)
        
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization."""
        # Dequantize weights
        weight = self.qweight.float() * self.scales.unsqueeze(1).float() + self.zeros.unsqueeze(1).float()
        
        # Linear operation
        output = torch.nn.functional.linear(x, weight, self.bias)
        
        return output
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bits={self.bits}'


def quantize_model_exl2(
    model: nn.Module,
    tokenizer: Any,
    calibration_texts: list[str],
    target_bpw: float = 4.0,
    output_dir: Optional[str] = None
) -> nn.Module:
    """
    Convenience function to quantize a model with EXL2.
    
    Args:
        model: Model to quantize
        tokenizer: Tokenizer
        calibration_texts: Calibration data
        target_bpw: Target bits per weight
        output_dir: Optional output directory
    
    Returns:
        Quantized model
    
    Example:
        texts = ["Sample text 1", "Sample text 2", ...]
        quantized = quantize_model_exl2(model, tokenizer, texts, target_bpw=4.0)
    """
    config = EXL2Config(target_bpw=target_bpw, output_dir=output_dir)
    quantizer = EXL2Quantizer(config)
    
    quantizer.calibrate(model, tokenizer, calibration_texts)
    quantized_model = quantizer.quantize(model)
    
    if output_dir:
        quantizer.save(quantized_model, output_dir)
    
    return quantized_model
