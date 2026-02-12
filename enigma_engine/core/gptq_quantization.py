"""
GPTQ Quantization for enigma_engine

Post-training quantization using GPTQ algorithm:
- Calibration-based quantization
- Per-column Hessian estimation
- Lazy batch updates for accuracy
- 4-bit, 3-bit, 2-bit support

Usage:
    from enigma_engine.core.gptq_quantization import GPTQQuantizer
    
    quantizer = GPTQQuantizer(model, tokenizer)
    quantized_model = quantizer.quantize(calibration_data, bits=4)
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GPTQConfig:
    """Configuration for GPTQ quantization."""
    bits: int = 4
    group_size: int = 128  # -1 for per-column
    damp_percent: float = 0.01
    desc_act: bool = False  # Descending activation order
    sym: bool = True  # Symmetric quantization
    true_sequential: bool = True
    batch_size: int = 1
    use_cuda_fp16: bool = True
    
    # Advanced
    act_order: bool = False  # Activation reordering
    static_groups: bool = False


class Quantizer:
    """Basic quantizer for weights."""
    
    def __init__(
        self,
        shape: tuple[int, ...],
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True
    ):
        self.bits = bits
        self.group_size = group_size if group_size > 0 else shape[-1]
        self.sym = sym
        
        self.maxq = 2 ** bits - 1
        
        # Quantization parameters
        self.scale = None
        self.zero = None
    
    def find_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Find optimal scale and zero point."""
        if self.group_size < x.shape[-1]:
            # Reshape for group quantization
            x = x.reshape(-1, self.group_size)
        
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_min = x.min(dim=-1, keepdim=True)[0]
        
        if self.sym:
            x_absmax = torch.max(x_max.abs(), x_min.abs())
            scale = x_absmax / (self.maxq // 2)
            zero = torch.zeros_like(scale)
        else:
            scale = (x_max - x_min) / self.maxq
            zero = torch.round(-x_min / scale)
        
        scale = scale.clamp(min=1e-10)
        
        self.scale = scale
        self.zero = zero
        
        return scale, zero
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize tensor."""
        if self.scale is None:
            self.find_params(x)
        
        original_shape = x.shape
        
        if self.group_size < x.shape[-1]:
            x = x.reshape(-1, self.group_size)
        
        if self.sym:
            q = torch.clamp(
                torch.round(x / self.scale) + self.maxq // 2,
                0, self.maxq
            )
        else:
            q = torch.clamp(
                torch.round(x / self.scale) + self.zero,
                0, self.maxq
            )
        
        return q.reshape(original_shape)
    
    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        """Dequantize tensor."""
        original_shape = q.shape
        
        if self.group_size < q.shape[-1]:
            q = q.reshape(-1, self.group_size)
        
        if self.sym:
            x = (q - self.maxq // 2) * self.scale
        else:
            x = (q - self.zero) * self.scale
        
        return x.reshape(original_shape)


class GPTQ:
    """
    GPTQ algorithm implementation.
    
    Quantizes a single linear layer using the GPTQ algorithm
    which minimizes squared error with Hessian-based updates.
    """
    
    def __init__(
        self,
        layer: nn.Linear,
        config: GPTQConfig
    ):
        self.layer = layer
        self.config = config
        self.device = layer.weight.device
        
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        
        # Hessian matrix (column correlations)
        self.H = torch.zeros(
            (self.columns, self.columns),
            device=self.device,
            dtype=torch.float32
        )
        self.nsamples = 0
    
    def add_batch(self, inp: torch.Tensor) -> None:
        """
        Add a batch of inputs to accumulate Hessian.
        
        Args:
            inp: Input activations (batch, ..., columns)
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        
        batch_size = inp.shape[0]
        
        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        
        inp = inp.float()
        
        # Accumulate H = X^T @ X (Hessian approximation)
        self.H += inp.T @ inp
        self.nsamples += inp.shape[0]
    
    def quantize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize the layer weights using GPTQ.
        
        Returns:
            quantized_weights, scales, zeros
        """
        W = self.layer.weight.data.clone().float()
        
        # Average Hessian
        H = self.H / self.nsamples
        
        # Add damping for numerical stability
        damp = self.config.damp_percent * torch.diag(H).mean()
        H += torch.eye(self.columns, device=self.device) * damp
        
        # Cholesky decomposition for efficient inverse
        try:
            H_inv = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(H_inv)
        except RuntimeError:
            logger.warning("Cholesky failed, using pseudo-inverse")
            H_inv = torch.linalg.pinv(H)
        
        # Initialize quantizer
        quantizer = Quantizer(
            W.shape,
            bits=self.config.bits,
            group_size=self.config.group_size,
            sym=self.config.sym
        )
        
        # Find initial quantization parameters
        quantizer.find_params(W)
        
        # Quantized weights
        Q = torch.zeros_like(W)
        
        # Error accumulator
        Losses = torch.zeros(self.rows, device=self.device)
        
        # Process in blocks
        block_size = 128
        
        for i1 in range(0, self.columns, block_size):
            i2 = min(i1 + block_size, self.columns)
            count = i2 - i1
            
            W_block = W[:, i1:i2].clone()
            Q_block = torch.zeros_like(W_block)
            Err_block = torch.zeros_like(W_block)
            
            H_inv_block = H_inv[i1:i2, i1:i2]
            
            for i in range(count):
                col_idx = i1 + i
                w = W_block[:, i]
                d = H_inv_block[i, i]
                
                # Quantize column
                if self.config.group_size > 0:
                    group_idx = col_idx // self.config.group_size
                    if hasattr(quantizer, 'scale') and quantizer.scale is not None:
                        if len(quantizer.scale.shape) > 0:
                            scale = quantizer.scale[..., group_idx:group_idx+1] if quantizer.scale.numel() > 1 else quantizer.scale
                            zero = quantizer.zero[..., group_idx:group_idx+1] if quantizer.zero.numel() > 1 else quantizer.zero
                        else:
                            scale, zero = quantizer.scale, quantizer.zero
                    else:
                        scale, zero = quantizer.find_params(w.unsqueeze(1))
                else:
                    scale, zero = quantizer.scale, quantizer.zero
                
                # Quantize and dequantize
                if quantizer.sym:
                    q = torch.clamp(
                        torch.round(w / scale.flatten()) + quantizer.maxq // 2,
                        0, quantizer.maxq
                    )
                    q_dequant = (q - quantizer.maxq // 2) * scale.flatten()
                else:
                    q = torch.clamp(
                        torch.round(w / scale.flatten()) + zero.flatten(),
                        0, quantizer.maxq
                    )
                    q_dequant = (q - zero.flatten()) * scale.flatten()
                
                Q_block[:, i] = q
                
                # Compute error
                err = (w - q_dequant) / d
                Err_block[:, i] = err
                
                # Update remaining weights (lazy batch update)
                W_block[:, i:] -= err.unsqueeze(1) @ H_inv_block[i, i:].unsqueeze(0)
                
                Losses += (w - q_dequant) ** 2 / d
            
            Q[:, i1:i2] = Q_block
            
            # Update remaining columns
            W[:, i2:] -= Err_block @ H_inv[i1:i2, i2:]
        
        logger.info(f"GPTQ quantization loss: {Losses.sum().item():.4f}")
        
        return Q.to(self.layer.weight.dtype), quantizer.scale, quantizer.zero


class QuantizedLinear(nn.Module):
    """Quantized linear layer with efficient storage."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 4,
        group_size: int = 128
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size if group_size > 0 else in_features
        
        # Number of groups
        self.num_groups = math.ceil(in_features / self.group_size)
        
        # Quantized weights (packed into int32)
        elements_per_int = 32 // bits
        packed_size = math.ceil(in_features / elements_per_int)
        
        self.register_buffer(
            'qweight',
            torch.zeros((out_features, packed_size), dtype=torch.int32)
        )
        
        # Scales and zeros per group
        self.register_buffer(
            'scales',
            torch.zeros((out_features, self.num_groups), dtype=torch.float16)
        )
        self.register_buffer(
            'zeros',
            torch.zeros((out_features, self.num_groups), dtype=torch.float16)
        )
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None
    
    def pack_weights(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor
    ):
        """Pack quantized weights into int32."""
        elements_per_int = 32 // self.bits
        
        # Ensure qweight is integer
        qweight = qweight.to(torch.int32)
        
        packed = torch.zeros(
            (self.out_features, math.ceil(self.in_features / elements_per_int)),
            dtype=torch.int32,
            device=qweight.device
        )
        
        for i in range(elements_per_int):
            if i * (self.in_features // elements_per_int) < self.in_features:
                start = i
                packed |= (qweight[:, start::elements_per_int] << (i * self.bits))
        
        self.qweight = packed
        self.scales = scales.to(torch.float16)
        self.zeros = zeros.to(torch.float16)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize and compute linear transformation."""
        # Unpack weights
        weight = self._dequantize()
        
        output = F.linear(x, weight, self.bias)
        return output
    
    def _dequantize(self) -> torch.Tensor:
        """Dequantize weights to float."""
        elements_per_int = 32 // self.bits
        mask = (1 << self.bits) - 1
        
        weight = torch.zeros(
            (self.out_features, self.in_features),
            dtype=torch.float16,
            device=self.qweight.device
        )
        
        for i in range(elements_per_int):
            extracted = (self.qweight >> (i * self.bits)) & mask
            
            col_start = i
            col_end = self.in_features
            step = elements_per_int
            
            for col_idx, ext_col in enumerate(range(col_start, col_end, step)):
                if ext_col >= self.in_features:
                    break
                
                group_idx = ext_col // self.group_size
                scale = self.scales[:, group_idx]
                zero = self.zeros[:, group_idx]
                
                weight[:, ext_col] = (extracted[:, col_idx].float() - zero) * scale
        
        return weight


class GPTQQuantizer:
    """
    Main quantizer class for GPTQ quantization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[GPTQConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GPTQConfig()
        self.device = next(model.parameters()).device
    
    def quantize(
        self,
        calibration_data: list[str],
        bits: Optional[int] = None
    ) -> nn.Module:
        """
        Quantize the model using GPTQ.
        
        Args:
            calibration_data: List of calibration texts
            bits: Override bits from config
        
        Returns:
            Quantized model
        """
        if bits is not None:
            self.config.bits = bits
        
        logger.info(f"Starting GPTQ quantization with {self.config.bits} bits")
        
        # Find all Linear layers
        layers_to_quantize = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layers_to_quantize.append((name, module))
        
        logger.info(f"Found {len(layers_to_quantize)} linear layers to quantize")
        
        # Prepare calibration data
        calibration_inputs = self._prepare_calibration(calibration_data)
        
        # Quantize each layer
        for name, layer in layers_to_quantize:
            logger.info(f"Quantizing layer: {name}")
            self._quantize_layer(name, layer, calibration_inputs)
        
        return self.model
    
    def _prepare_calibration(
        self,
        texts: list[str]
    ) -> list[torch.Tensor]:
        """Prepare calibration inputs."""
        inputs = []
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            tokens = tokens[:512]  # Limit length
            inputs.append(
                torch.tensor([tokens], device=self.device)
            )
        
        return inputs
    
    def _quantize_layer(
        self,
        name: str,
        layer: nn.Linear,
        calibration_inputs: list[torch.Tensor]
    ):
        """Quantize a single layer."""
        # Create GPTQ quantizer for this layer
        gptq = GPTQ(layer, self.config)
        
        # Collect activations
        hooks = []
        activations = []
        
        def hook_fn(module: Any, inp: tuple, out: Any) -> None:
            activations.append(inp[0].detach())
        
        # Register hook on the layer
        handle = layer.register_forward_hook(hook_fn)
        
        # Run calibration
        self.model.eval()
        with torch.no_grad():
            for inp in calibration_inputs:
                try:
                    self.model(inp)
                except Exception:
                    pass  # Some models may error, that's ok
        
        handle.remove()
        
        # Add activations to GPTQ
        for act in activations:
            gptq.add_batch(act)
        
        # Quantize
        Q, scales, zeros = gptq.quantize()
        
        # Replace layer with quantized version
        quantized_layer = QuantizedLinear(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            bits=self.config.bits,
            group_size=self.config.group_size
        )
        
        # Pack weights
        quantized_layer.pack_weights(Q, scales, zeros)
        
        if layer.bias is not None:
            quantized_layer.bias = layer.bias.clone()
        
        # Replace in model
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        if parent_name:
            parent = self.model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            setattr(parent, child_name, quantized_layer)
        else:
            setattr(self.model, child_name, quantized_layer)


def quantize_model_gptq(
    model: nn.Module,
    tokenizer: Any,
    calibration_data: list[str],
    bits: int = 4,
    group_size: int = 128
) -> nn.Module:
    """
    Convenience function to quantize a model with GPTQ.
    
    Args:
        model: Model to quantize
        tokenizer: Tokenizer
        calibration_data: List of calibration texts
        bits: Quantization bits (2, 3, 4, 8)
        group_size: Group size for quantization
    
    Returns:
        Quantized model
    """
    config = GPTQConfig(bits=bits, group_size=group_size)
    quantizer = GPTQQuantizer(model, tokenizer, config)
    return quantizer.quantize(calibration_data)
