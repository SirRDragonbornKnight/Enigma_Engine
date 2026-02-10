"""
Context Length Extender

Techniques for extending context window beyond training length
including RoPE scaling, ALiBi, and hybrid approaches.

FILE: enigma_engine/core/context_extender.py
TYPE: Core/Inference
MAIN CLASSES: RoPEScaler, ALiBiEncoder, ContextExtender
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScalingMethod(Enum):
    """RoPE scaling methods."""
    LINEAR = auto()      # Simple linear interpolation
    NTK = auto()         # NTK-aware scaling
    DYNAMIC_NTK = auto() # Dynamic NTK
    YARN = auto()        # YaRN (Yet another RoPE extensioN)
    PI = auto()          # Position Interpolation


@dataclass
class RoPEConfig:
    """Configuration for RoPE scaling."""
    base_context: int = 2048  # Original training context length
    target_context: int = 8192  # Desired context length
    method: ScalingMethod = ScalingMethod.DYNAMIC_NTK
    scaling_factor: float = None  # Auto-computed if None
    ntk_alpha: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float = 1.0


if HAS_TORCH:
    
    class RoPEScaler:
        """
        Scale Rotary Position Embeddings for longer contexts.
        
        Implements various scaling methods:
        - LINEAR: Simple linear interpolation
        - NTK: Neural Tangent Kernel aware scaling
        - DYNAMIC_NTK: Context-length aware NTK
        - YARN: Yet another RoPE extensioN
        - PI: Position Interpolation
        """
        
        def __init__(
            self,
            dim: int,
            config: RoPEConfig = None
        ):
            self.dim = dim
            self.config = config or RoPEConfig()
            
            # Compute scaling factor
            if self.config.scaling_factor is None:
                self.config.scaling_factor = (
                    self.config.target_context / self.config.base_context
                )
            
            # Base frequencies
            self.base_freq = 10000.0
            self._cache = {}
        
        def _compute_freqs(
            self,
            seq_len: int,
            device: torch.device = None
        ) -> torch.Tensor:
            """Compute frequency tensor."""
            cache_key = (seq_len, str(device), self.config.method)
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            if self.config.method == ScalingMethod.LINEAR:
                freqs = self._linear_scale(seq_len, device)
            elif self.config.method == ScalingMethod.NTK:
                freqs = self._ntk_scale(seq_len, device)
            elif self.config.method == ScalingMethod.DYNAMIC_NTK:
                freqs = self._dynamic_ntk_scale(seq_len, device)
            elif self.config.method == ScalingMethod.YARN:
                freqs = self._yarn_scale(seq_len, device)
            elif self.config.method == ScalingMethod.PI:
                freqs = self._pi_scale(seq_len, device)
            else:
                freqs = self._base_rope(seq_len, device)
            
            self._cache[cache_key] = freqs
            return freqs
        
        def _base_rope(
            self,
            seq_len: int,
            device: torch.device = None
        ) -> torch.Tensor:
            """Standard RoPE frequencies."""
            inv_freq = 1.0 / (
                self.base_freq ** (
                    torch.arange(0, self.dim, 2, device=device).float() / self.dim
                )
            )
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            return torch.cat([freqs, freqs], dim=-1)
        
        def _linear_scale(
            self,
            seq_len: int,
            device: torch.device = None
        ) -> torch.Tensor:
            """Linear position interpolation."""
            inv_freq = 1.0 / (
                self.base_freq ** (
                    torch.arange(0, self.dim, 2, device=device).float() / self.dim
                )
            )
            # Scale positions
            t = torch.arange(seq_len, device=device).float()
            t = t / self.config.scaling_factor
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            return torch.cat([freqs, freqs], dim=-1)
        
        def _ntk_scale(
            self,
            seq_len: int,
            device: torch.device = None
        ) -> torch.Tensor:
            """NTK-aware scaling - modifies base frequency."""
            # Scale the base frequency
            scaled_base = self.base_freq * (
                (self.config.scaling_factor * self.config.ntk_alpha) ** 
                (self.dim / (self.dim - 2))
            )
            
            inv_freq = 1.0 / (
                scaled_base ** (
                    torch.arange(0, self.dim, 2, device=device).float() / self.dim
                )
            )
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            return torch.cat([freqs, freqs], dim=-1)
        
        def _dynamic_ntk_scale(
            self,
            seq_len: int,
            device: torch.device = None
        ) -> torch.Tensor:
            """Dynamic NTK - adjusts based on actual sequence length."""
            if seq_len <= self.config.base_context:
                return self._base_rope(seq_len, device)
            
            # Dynamic scaling based on actual length
            dynamic_factor = seq_len / self.config.base_context
            scaled_base = self.base_freq * (
                dynamic_factor ** (self.dim / (self.dim - 2))
            )
            
            inv_freq = 1.0 / (
                scaled_base ** (
                    torch.arange(0, self.dim, 2, device=device).float() / self.dim
                )
            )
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            return torch.cat([freqs, freqs], dim=-1)
        
        def _yarn_scale(
            self,
            seq_len: int,
            device: torch.device = None
        ) -> torch.Tensor:
            """
            YaRN scaling - combines interpolation and NTK.
            
            High frequencies (local patterns) use interpolation.
            Low frequencies (global patterns) use NTK.
            """
            pos_freqs = self.base_freq ** (
                torch.arange(0, self.dim, 2, device=device).float() / self.dim
            )
            
            # Wavelengths for determining scaling
            wavelengths = 2 * math.pi * pos_freqs
            
            # Interpolation ratio per dimension
            low = self.config.yarn_beta_slow
            high = self.config.yarn_beta_fast
            
            ratio = (
                (wavelengths / (2 * math.pi * self.config.base_context) - low) /
                (high - low)
            ).clamp(0, 1)
            
            # Blend between interpolation and NTK
            ntk_factor = (
                (self.config.scaling_factor) ** 
                (self.dim / (self.dim - 2))
            )
            interpolated_inv_freq = 1.0 / (pos_freqs * self.config.scaling_factor)
            ntk_inv_freq = 1.0 / (pos_freqs * ntk_factor)
            
            inv_freq = (1 - ratio) * interpolated_inv_freq + ratio * ntk_inv_freq
            
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            
            # Apply mscale for attention score adjustment
            mscale = self.config.yarn_mscale * math.sqrt(
                1 + math.log(self.config.scaling_factor) / math.log(self.config.base_context)
            )
            
            return torch.cat([freqs, freqs], dim=-1) * mscale
        
        def _pi_scale(
            self,
            seq_len: int,
            device: torch.device = None
        ) -> torch.Tensor:
            """Position Interpolation - fine-tuning friendly."""
            inv_freq = 1.0 / (
                self.base_freq ** (
                    torch.arange(0, self.dim, 2, device=device).float() / self.dim
                )
            )
            # Interpolate positions to fit within original context
            t = torch.arange(seq_len, device=device).float()
            if seq_len > self.config.base_context:
                t = t * (self.config.base_context / seq_len)
            
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            return torch.cat([freqs, freqs], dim=-1)
        
        def apply_rotary(
            self,
            x: torch.Tensor,
            position_ids: torch.Tensor = None
        ) -> torch.Tensor:
            """
            Apply rotary embeddings to input.
            
            Args:
                x: Input tensor [batch, seq_len, dim] or [batch, heads, seq_len, dim]
                position_ids: Optional explicit positions
            
            Returns:
                Tensor with rotary embeddings applied
            """
            if len(x.shape) == 3:
                batch, seq_len, dim = x.shape
                x = x.unsqueeze(1)  # Add head dim
                squeeze_back = True
            else:
                batch, heads, seq_len, dim = x.shape
                squeeze_back = False
            
            if position_ids is None:
                freqs = self._compute_freqs(seq_len, x.device)
            else:
                # Gather frequencies for specific positions
                max_pos = position_ids.max().item() + 1
                all_freqs = self._compute_freqs(int(max_pos), x.device)
                freqs = all_freqs[position_ids]
            
            # Apply rotation
            cos = freqs.cos().unsqueeze(0).unsqueeze(0)
            sin = freqs.sin().unsqueeze(0).unsqueeze(0)
            
            x_rotated = self._rotate_half(x)
            output = x * cos + x_rotated * sin
            
            if squeeze_back:
                output = output.squeeze(1)
            
            return output
        
        @staticmethod
        def _rotate_half(x: torch.Tensor) -> torch.Tensor:
            """Rotate half the hidden dims of x."""
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)
    
    
    class ALiBiEncoder(nn.Module):
        """
        Attention with Linear Biases (ALiBi).
        
        No learned position embeddings - adds position-dependent
        bias directly to attention scores. Extrapolates naturally
        to longer sequences.
        """
        
        def __init__(
            self,
            num_heads: int,
            slope_multiplier: float = 1.0
        ):
            super().__init__()
            self.num_heads = num_heads
            self.slope_multiplier = slope_multiplier
            
            # Compute head slopes: geometric sequence
            self.slopes = self._compute_slopes(num_heads, slope_multiplier)
            self.register_buffer('_alibi_slopes', self.slopes)
        
        @staticmethod
        def _compute_slopes(
            num_heads: int,
            multiplier: float = 1.0
        ) -> torch.Tensor:
            """Compute ALiBi slopes for each head."""
            # Start with closest power of 2
            closest_power = 2 ** math.floor(math.log2(num_heads))
            base = 2 ** (-(2 ** -(math.log2(closest_power) - 3)))
            
            slopes = base ** torch.arange(1, closest_power + 1)
            
            # Handle non-power-of-2 heads
            if num_heads != closest_power:
                extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power) - 3)))
                extra_slopes = extra_base ** (2 * torch.arange(1, num_heads - closest_power + 1) + 1)
                slopes = torch.cat([slopes, extra_slopes])
            
            return slopes * multiplier
        
        def get_bias(
            self,
            seq_len: int,
            device: torch.device = None
        ) -> torch.Tensor:
            """
            Get ALiBi bias matrix.
            
            Args:
                seq_len: Sequence length
                device: Target device
            
            Returns:
                Bias tensor [1, num_heads, seq_len, seq_len]
            """
            # Create position difference matrix
            positions = torch.arange(seq_len, device=device)
            relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
            
            # Apply slopes per head
            slopes = self._alibi_slopes.to(device)
            bias = slopes.unsqueeze(1).unsqueeze(1) * relative_positions.unsqueeze(0)
            
            return bias.unsqueeze(0)
        
        def forward(
            self,
            attention_scores: torch.Tensor
        ) -> torch.Tensor:
            """
            Add ALiBi bias to attention scores.
            
            Args:
                attention_scores: [batch, heads, seq_len, seq_len]
            
            Returns:
                Biased attention scores
            """
            seq_len = attention_scores.shape[-1]
            bias = self.get_bias(seq_len, attention_scores.device)
            return attention_scores + bias
    
    
    class ContextExtender:
        """
        Unified interface for context extension.
        
        Combines RoPE scaling and ALiBi for flexible
        context length extension.
        """
        
        def __init__(
            self,
            model: nn.Module,
            base_context: int = 2048,
            target_context: int = 8192,
            method: str = "dynamic_ntk"
        ):
            self.model = model
            self.base_context = base_context
            self.target_context = target_context
            
            # Determine method
            method_map = {
                "linear": ScalingMethod.LINEAR,
                "ntk": ScalingMethod.NTK,
                "dynamic_ntk": ScalingMethod.DYNAMIC_NTK,
                "yarn": ScalingMethod.YARN,
                "pi": ScalingMethod.PI
            }
            self.method = method_map.get(method.lower(), ScalingMethod.DYNAMIC_NTK)
            
            self._modified_layers = []
        
        def extend_rope(self, dim: int = None) -> "ContextExtender":
            """
            Apply RoPE scaling to model.
            
            Args:
                dim: Head dimension (auto-detected if None)
            
            Returns:
                Self for chaining
            """
            # Find RoPE layers and replace
            for name, module in self.model.named_modules():
                # Look for rope/rotary layers
                if any(x in name.lower() for x in ['rope', 'rotary']):
                    if hasattr(module, 'dim'):
                        dim = module.dim
                    
                    config = RoPEConfig(
                        base_context=self.base_context,
                        target_context=self.target_context,
                        method=self.method
                    )
                    
                    scaler = RoPEScaler(dim or 64, config)
                    
                    # Replace forward method
                    original_forward = module.forward
                    
                    def new_forward(x, pos_ids=None, _scaler=scaler):
                        return _scaler.apply_rotary(x, pos_ids)
                    
                    module.forward = new_forward
                    self._modified_layers.append(name)
            
            logger.info(f"Extended RoPE in {len(self._modified_layers)} layers")
            return self
        
        def add_alibi(
            self,
            num_heads: int = None
        ) -> "ContextExtender":
            """
            Add ALiBi bias to attention layers.
            
            Args:
                num_heads: Number of attention heads
            
            Returns:
                Self for chaining
            """
            for name, module in self.model.named_modules():
                if 'attention' in name.lower():
                    # Try to detect num_heads
                    if num_heads is None:
                        if hasattr(module, 'num_heads'):
                            num_heads = module.num_heads
                        elif hasattr(module, 'n_heads'):
                            num_heads = module.n_heads
                        else:
                            continue
                    
                    alibi = ALiBiEncoder(num_heads)
                    
                    # Wrap attention computation
                    if hasattr(module, 'compute_attention'):
                        original = module.compute_attention
                        
                        def new_compute(q, k, v, mask=None, _alibi=alibi, _orig=original):
                            scores = _orig(q, k, v, mask)
                            return _alibi(scores)
                        
                        module.compute_attention = new_compute
            
            return self
        
        def get_config(self) -> dict[str, Any]:
            """Get current extension configuration."""
            return {
                "base_context": self.base_context,
                "target_context": self.target_context,
                "method": self.method.name,
                "scaling_factor": self.target_context / self.base_context,
                "modified_layers": self._modified_layers
            }


    def extend_context(
        model: nn.Module,
        target_length: int,
        method: str = "dynamic_ntk"
    ) -> nn.Module:
        """
        Extend model context length.
        
        Args:
            model: Model to extend
            target_length: Desired context length
            method: Scaling method
        
        Returns:
            Modified model
        """
        # Auto-detect base context
        base_context = 2048
        if hasattr(model, 'config'):
            if hasattr(model.config, 'max_position_embeddings'):
                base_context = model.config.max_position_embeddings
            elif hasattr(model.config, 'max_seq_len'):
                base_context = model.config.max_seq_len
        
        extender = ContextExtender(
            model,
            base_context=base_context,
            target_context=target_length,
            method=method
        )
        
        extender.extend_rope()
        
        return model

else:
    class RoPEScaler:
        pass
    
    class ALiBiEncoder:
        pass
    
    class ContextExtender:
        pass
    
    def extend_context(*args, **kwargs):
        raise ImportError("PyTorch required for context extension")
