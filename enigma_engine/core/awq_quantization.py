"""
AWQ (Activation-aware Weight Quantization) for enigma_engine

Advanced quantization that preserves salient weights:
- Activation-aware scaling
- Per-channel quantization
- Better accuracy than naive quantization
- 4-bit with near-FP16 quality

Usage:
    from enigma_engine.core.awq_quantization import AWQQuantizer
    
    quantizer = AWQQuantizer(model, tokenizer)
    quantized_model = quantizer.quantize(calibration_data)
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
class AWQConfig:
    """Configuration for AWQ quantization."""
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True  # Use asymmetric quantization
    
    # AWQ specific
    w_bit: int = 4
    q_group_size: int = 128
    
    # Calibration
    n_samples: int = 128
    seq_len: int = 512
    
    # Search
    search_scale: bool = True
    search_clip: bool = True
    n_grid: int = 20
    max_shrink: float = 0.8
    

class AWQLinear(nn.Module):
    """AWQ quantized linear layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        w_bit: int = 4,
        group_size: int = 128
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size > 0 else in_features
        
        # Number of groups
        self.n_groups = math.ceil(in_features / self.group_size)
        
        # Packed weights (int32)
        elements_per_int = 32 // w_bit
        self.register_buffer(
            'qweight',
            torch.zeros(
                (out_features, math.ceil(in_features / elements_per_int)),
                dtype=torch.int32
            )
        )
        
        # Scales per group
        self.register_buffer(
            'scales',
            torch.zeros((out_features, self.n_groups), dtype=torch.float16)
        )
        
        # Zero points per group
        self.register_buffer(
            'qzeros',
            torch.zeros(
                (out_features, math.ceil(self.n_groups / (32 // w_bit))),
                dtype=torch.int32
            )
        )
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization."""
        weight = self._dequantize()
        return F.linear(x, weight, self.bias)
    
    def _dequantize(self) -> torch.Tensor:
        """Dequantize weights."""
        elements_per_int = 32 // self.w_bit
        mask = (1 << self.w_bit) - 1
        
        weight = torch.zeros(
            (self.out_features, self.in_features),
            dtype=torch.float16,
            device=self.qweight.device
        )
        
        # Unpack weights
        for i in range(elements_per_int):
            col_start = i
            for w_col, col in enumerate(range(col_start, self.in_features, elements_per_int)):
                if col >= self.in_features:
                    break
                
                # Extract weight
                q = (self.qweight[:, w_col] >> (i * self.w_bit)) & mask
                
                # Get group index and scale/zero
                group_idx = col // self.group_size
                scale = self.scales[:, group_idx]
                
                # Extract zero point
                z_col = group_idx // elements_per_int
                z_shift = (group_idx % elements_per_int) * self.w_bit
                zero = (self.qzeros[:, z_col] >> z_shift) & mask
                
                # Dequantize
                weight[:, col] = (q.float() - zero.float()) * scale
        
        return weight


class AWQQuantizer:
    """
    AWQ quantization implementation.
    
    Uses activation statistics to find optimal per-channel scales
    that minimize quantization error for salient weights.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[AWQConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or AWQConfig()
        self.device = next(model.parameters()).device
    
    def quantize(
        self,
        calibration_data: list[str]
    ) -> nn.Module:
        """
        Quantize model using AWQ.
        
        Args:
            calibration_data: List of calibration texts
        
        Returns:
            Quantized model
        """
        logger.info("Starting AWQ quantization")
        
        # Collect activation statistics
        act_scales = self._collect_activation_stats(calibration_data)
        
        # Quantize each linear layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                logger.info(f"Quantizing: {name}")
                self._quantize_layer(name, module, act_scales.get(name, None))
        
        logger.info("AWQ quantization complete")
        return self.model
    
    def _collect_activation_stats(
        self,
        calibration_data: list[str]
    ) -> dict[str, torch.Tensor]:
        """Collect activation statistics for all linear layers."""
        act_scales = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, inp, out):
                if name not in act_scales:
                    act_scales[name] = []
                
                x = inp[0].detach()
                if len(x.shape) == 3:
                    x = x.reshape(-1, x.shape[-1])
                
                # Track max absolute activation per channel
                scale = x.abs().max(dim=0)[0]
                act_scales[name].append(scale)
            
            return hook
        
        # Register hooks
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(make_hook(name))
                hooks.append(handle)
        
        # Run calibration
        self.model.eval()
        with torch.no_grad():
            for text in calibration_data[:self.config.n_samples]:
                tokens = self.tokenizer.encode(text)[:self.config.seq_len]
                input_ids = torch.tensor([tokens], device=self.device)
                
                try:
                    self.model(input_ids)
                except Exception:
                    pass
        
        # Remove hooks
        for handle in hooks:
            handle.remove()
        
        # Average scales
        for name in act_scales:
            scales = torch.stack(act_scales[name])
            act_scales[name] = scales.mean(dim=0)
        
        return act_scales
    
    def _quantize_layer(
        self,
        name: str,
        layer: nn.Linear,
        act_scale: Optional[torch.Tensor]
    ):
        """Quantize a single layer using AWQ."""
        W = layer.weight.data.clone().float()
        
        # Find optimal scales
        if self.config.search_scale and act_scale is not None:
            scales = self._search_scales(W, act_scale)
        else:
            scales = torch.ones(W.shape[1], device=W.device)
        
        # Apply scales to weights
        W_scaled = W * scales.unsqueeze(0)
        
        # Quantize
        Q, qscales, qzeros = self._quantize_weight(W_scaled)
        
        # Create AWQ layer
        awq_layer = AWQLinear(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            w_bit=self.config.w_bit,
            group_size=self.config.q_group_size
        )
        
        # Pack weights
        self._pack_weights(awq_layer, Q, qscales, qzeros)
        
        if layer.bias is not None:
            awq_layer.bias = layer.bias.clone()
        
        # Replace layer
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        if parent_name:
            parent = self.model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            setattr(parent, child_name, awq_layer)
        else:
            setattr(self.model, child_name, awq_layer)
    
    def _search_scales(
        self,
        W: torch.Tensor,
        act_scale: torch.Tensor
    ) -> torch.Tensor:
        """Search for optimal per-channel scales."""
        best_scales = torch.ones(W.shape[1], device=W.device)
        best_error = float('inf')
        
        # Weight importance (magnitude * activation)
        w_scale = W.abs().mean(dim=0)
        importance = act_scale * w_scale
        
        # Normalize importance
        importance = importance / importance.max()
        
        # Grid search for optimal scaling
        for ratio in torch.linspace(0, 1, self.config.n_grid):
            # Scale based on importance
            scales = importance.pow(ratio).clamp(min=1e-4)
            
            # Normalize to preserve magnitude
            scales = scales / scales.mean()
            
            # Quantize with these scales
            W_scaled = W * scales.unsqueeze(0)
            Q, _, _ = self._quantize_weight(W_scaled)
            
            # Dequantize
            W_dequant = self._dequantize_weight(Q, W_scaled)
            
            # Compute error
            error = (W - W_dequant / scales.unsqueeze(0)).pow(2).mean()
            
            if error < best_error:
                best_error = error
                best_scales = scales.clone()
        
        return best_scales
    
    def _quantize_weight(
        self,
        W: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize weights with grouping."""
        out_features, in_features = W.shape
        group_size = self.config.q_group_size
        n_groups = math.ceil(in_features / group_size)
        
        maxq = 2 ** self.config.w_bit - 1
        
        Q = torch.zeros_like(W, dtype=torch.int32)
        scales = torch.zeros((out_features, n_groups), device=W.device)
        zeros = torch.zeros((out_features, n_groups), device=W.device)
        
        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, in_features)
            
            W_group = W[:, start:end]
            
            # Find scale and zero point
            w_min = W_group.min(dim=1)[0]
            w_max = W_group.max(dim=1)[0]
            
            scale = (w_max - w_min) / maxq
            scale = scale.clamp(min=1e-10)
            zero = torch.round(-w_min / scale)
            
            scales[:, g] = scale
            zeros[:, g] = zero
            
            # Quantize
            Q[:, start:end] = torch.clamp(
                torch.round(W_group / scale.unsqueeze(1)) + zero.unsqueeze(1),
                0, maxq
            ).to(torch.int32)
        
        return Q, scales, zeros
    
    def _dequantize_weight(
        self,
        Q: torch.Tensor,
        W_scaled: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize weights for error computation."""
        out_features, in_features = Q.shape
        group_size = self.config.q_group_size
        n_groups = math.ceil(in_features / group_size)
        
        maxq = 2 ** self.config.w_bit - 1
        
        W_dequant = torch.zeros_like(W_scaled)
        
        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, in_features)
            
            Q_group = Q[:, start:end].float()
            W_group = W_scaled[:, start:end]
            
            # Find scale and zero
            w_min = W_group.min(dim=1, keepdim=True)[0]
            w_max = W_group.max(dim=1, keepdim=True)[0]
            
            scale = (w_max - w_min) / maxq
            scale = scale.clamp(min=1e-10)
            zero = torch.round(-w_min / scale)
            
            # Dequantize
            W_dequant[:, start:end] = (Q_group - zero) * scale
        
        return W_dequant
    
    def _pack_weights(
        self,
        layer: AWQLinear,
        Q: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor
    ):
        """Pack quantized weights into AWQ format."""
        elements_per_int = 32 // self.config.w_bit
        
        # Pack weights
        packed = torch.zeros_like(layer.qweight)
        for i in range(elements_per_int):
            col_start = i
            for w_col, col in enumerate(range(col_start, layer.in_features, elements_per_int)):
                if col >= layer.in_features:
                    break
                packed[:, w_col] |= (Q[:, col] << (i * self.config.w_bit))
        
        layer.qweight = packed
        layer.scales = scales.to(torch.float16)
        
        # Pack zeros
        n_groups = zeros.shape[1]
        packed_zeros = torch.zeros_like(layer.qzeros)
        for g in range(n_groups):
            z_col = g // elements_per_int
            z_shift = (g % elements_per_int) * self.config.w_bit
            packed_zeros[:, z_col] |= (zeros[:, g].to(torch.int32) << z_shift)
        
        layer.qzeros = packed_zeros


def quantize_model_awq(
    model: nn.Module,
    tokenizer: Any,
    calibration_data: list[str],
    w_bit: int = 4,
    group_size: int = 128
) -> nn.Module:
    """
    Convenience function to quantize with AWQ.
    
    Args:
        model: Model to quantize
        tokenizer: Tokenizer
        calibration_data: Calibration texts
        w_bit: Quantization bits
        group_size: Group size
    
    Returns:
        Quantized model
    """
    config = AWQConfig(w_bit=w_bit, q_group_size=group_size)
    quantizer = AWQQuantizer(model, tokenizer, config)
    return quantizer.quantize(calibration_data)
