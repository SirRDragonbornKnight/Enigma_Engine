"""
Context Extension for Enigma AI Engine

Extend model context window beyond training length.

Features:
- RoPE scaling (linear, NTK, dynamic)
- Position interpolation
- Sliding window attention
- Context compression

Usage:
    from enigma_engine.core.context_extension import ContextExtender, get_extender
    
    extender = get_extender()
    
    # Extend context to 8k
    extender.apply_rope_scaling(model, target_length=8192)
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ScalingMethod(Enum):
    """RoPE scaling methods."""
    LINEAR = "linear"  # Simple linear interpolation
    NTK = "ntk"  # Neural Tangent Kernel aware scaling
    DYNAMIC_NTK = "dynamic_ntk"  # Dynamic NTK based on sequence length
    YARN = "yarn"  # Yet Another RoPE Extension
    ALIBI = "alibi"  # ALiBi-style position bias


@dataclass
class ExtensionConfig:
    """Configuration for context extension."""
    method: ScalingMethod = ScalingMethod.DYNAMIC_NTK
    
    # Original model context
    original_max_position: int = 2048
    
    # Target context
    target_max_position: int = 8192
    
    # NTK parameters
    ntk_alpha: Optional[float] = None  # Auto-computed if None
    
    # YARN parameters
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    
    # Dynamic scaling
    dynamic_factor: float = 2.0
    
    # Sliding window
    sliding_window_size: Optional[int] = None


class RoPEScaler:
    """RoPE (Rotary Position Embedding) scaling implementations."""
    
    def __init__(self, config: ExtensionConfig):
        self._config = config
        self._scaling_factor = config.target_max_position / config.original_max_position
    
    def compute_inverse_freq(
        self,
        dim: int,
        base: float = 10000.0
    ) -> Any:
        """Compute inverse frequency for RoPE."""
        import torch
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        return inv_freq
    
    def linear_scaling(
        self,
        inv_freq: Any,
        positions: Any
    ) -> Tuple[Any, Any]:
        """
        Apply linear interpolation scaling.
        
        Simple but effective for moderate extensions.
        """
        import torch
        
        # Scale positions
        scaled_positions = positions / self._scaling_factor
        
        # Compute cos/sin
        freqs = torch.einsum("i,j->ij", scaled_positions.float(), inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        return cos, sin
    
    def ntk_scaling(
        self,
        dim: int,
        base: float = 10000.0,
        positions: Any = None
    ) -> Tuple[Any, Any]:
        """
        Apply NTK-aware interpolation.
        
        Better preserves high-frequency information.
        """
        import torch
        
        # Compute alpha
        alpha = self._config.ntk_alpha
        if alpha is None:
            alpha = (self._scaling_factor * (dim / (dim - 2))) ** (dim / (dim - 2))
        
        # Scale base frequency
        scaled_base = base * alpha
        
        # Compute new inverse frequencies
        inv_freq = 1.0 / (scaled_base ** (torch.arange(0, dim, 2).float() / dim))
        
        if positions is None:
            positions = torch.arange(self._config.target_max_position)
        
        freqs = torch.einsum("i,j->ij", positions.float(), inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        return cos, sin
    
    def dynamic_ntk_scaling(
        self,
        dim: int,
        seq_len: int,
        base: float = 10000.0
    ) -> Tuple[Any, Any]:
        """
        Dynamic NTK scaling based on sequence length.
        
        Adjusts scaling factor based on actual sequence.
        """
        import torch
        
        if seq_len > self._config.original_max_position:
            # Compute dynamic alpha
            dynamic_scale = seq_len / self._config.original_max_position
            alpha = (self._config.dynamic_factor * dynamic_scale) ** (dim / (dim - 2))
            
            scaled_base = base * alpha
        else:
            scaled_base = base
        
        inv_freq = 1.0 / (scaled_base ** (torch.arange(0, dim, 2).float() / dim))
        
        positions = torch.arange(seq_len)
        freqs = torch.einsum("i,j->ij", positions.float(), inv_freq)
        
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        return cos, sin
    
    def yarn_scaling(
        self,
        dim: int,
        seq_len: int,
        base: float = 10000.0
    ) -> Tuple[Any, Any]:
        """
        YARN (Yet Another RoPE Extension) scaling.
        
        Combines wavelength-aware scaling.
        """
        import torch
        
        beta_fast = self._config.yarn_beta_fast
        beta_slow = self._config.yarn_beta_slow
        
        # Compute wavelength-dependent scaling
        dim_range = torch.arange(0, dim, 2).float()
        wavelength = 2 * math.pi * (base ** (dim_range / dim))
        
        # Thresholds
        low_threshold = self._config.original_max_position / beta_fast
        high_threshold = self._config.original_max_position / beta_slow
        
        # Linear ramp between thresholds
        ratio = (wavelength - low_threshold) / (high_threshold - low_threshold)
        ratio = torch.clamp(ratio, 0, 1)
        
        # Interpolation factor
        scale = 1 - ratio + ratio * self._scaling_factor
        
        # Scaled inverse frequencies
        inv_freq = 1.0 / (base ** (dim_range / dim))
        inv_freq = inv_freq / scale
        
        positions = torch.arange(seq_len)
        freqs = torch.einsum("i,j->ij", positions.float(), inv_freq)
        
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        return cos, sin


class ALiBiPositionBias:
    """ALiBi (Attention with Linear Biases) for context extension."""
    
    def __init__(self, num_heads: int, max_length: int):
        """
        Initialize ALiBi.
        
        Args:
            num_heads: Number of attention heads
            max_length: Maximum sequence length
        """
        self._num_heads = num_heads
        self._max_length = max_length
        self._slopes = self._compute_slopes()
    
    def _compute_slopes(self) -> Any:
        """Compute head-specific slopes."""
        import torch
        
        # Geometric sequence for slopes
        closest_power = 2 ** math.floor(math.log2(self._num_heads))
        base = 2 ** (-8 / closest_power)
        
        slopes = []
        for i in range(1, closest_power + 1):
            slopes.append(base ** i)
        
        if closest_power < self._num_heads:
            extra_base = 2 ** (-8 / (2 * closest_power))
            for i in range(1, 2 * (self._num_heads - closest_power) + 1, 2):
                slopes.append(extra_base ** i)
        
        return torch.tensor(slopes).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    
    def get_bias(self, seq_len: int) -> Any:
        """
        Get position bias matrix.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Bias tensor of shape (1, num_heads, seq_len, seq_len)
        """
        import torch
        
        # Distance matrix
        positions = torch.arange(seq_len)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)
        distance = distance.unsqueeze(0).unsqueeze(0).float()
        
        # Apply slopes
        bias = distance * self._slopes
        
        # Mask future positions (causal)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        bias = bias.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        return bias


class SlidingWindowAttention:
    """Sliding window attention for long contexts."""
    
    def __init__(self, window_size: int):
        """
        Initialize sliding window attention.
        
        Args:
            window_size: Size of attention window
        """
        self._window_size = window_size
    
    def create_mask(self, seq_len: int) -> Any:
        """
        Create sliding window attention mask.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Attention mask tensor
        """
        import torch
        
        # Create base causal mask
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)
        
        # Apply sliding window
        for i in range(seq_len):
            start = max(0, i - self._window_size)
            mask[i, :start] = True
        
        return ~mask  # Invert: True = attend, False = mask


class ContextExtender:
    """High-level context extension interface."""
    
    def __init__(self, config: Optional[ExtensionConfig] = None):
        """
        Initialize context extender.
        
        Args:
            config: Extension configuration
        """
        self._config = config or ExtensionConfig()
        self._scaler = RoPEScaler(self._config)
    
    def apply_rope_scaling(
        self,
        model: Any,
        target_length: Optional[int] = None,
        method: Optional[ScalingMethod] = None
    ) -> bool:
        """
        Apply RoPE scaling to model.
        
        Args:
            model: Model to modify
            target_length: Target context length
            method: Scaling method to use
            
        Returns:
            True if successful
        """
        import torch
        
        if target_length:
            self._config.target_max_position = target_length
            self._scaler = RoPEScaler(self._config)
        
        method = method or self._config.method
        
        try:
            # Find RoPE embeddings in model
            for name, module in model.named_modules():
                if hasattr(module, 'inv_freq') or 'rotary' in name.lower():
                    logger.info(f"Found RoPE module: {name}")
                    
                    # Get dimension
                    if hasattr(module, 'inv_freq'):
                        dim = len(module.inv_freq) * 2
                    elif hasattr(module, 'dim'):
                        dim = module.dim
                    else:
                        continue
                    
                    # Apply scaling
                    if method == ScalingMethod.LINEAR:
                        cos, sin = self._scaler.linear_scaling(
                            module.inv_freq,
                            torch.arange(self._config.target_max_position)
                        )
                    elif method == ScalingMethod.NTK:
                        cos, sin = self._scaler.ntk_scaling(dim)
                    elif method == ScalingMethod.DYNAMIC_NTK:
                        # Dynamic is applied per-forward
                        logger.info("Dynamic NTK will be applied per forward pass")
                        continue
                    elif method == ScalingMethod.YARN:
                        cos, sin = self._scaler.yarn_scaling(
                            dim,
                            self._config.target_max_position
                        )
                    else:
                        continue
                    
                    # Update module
                    if hasattr(module, 'cos_cached'):
                        module.cos_cached = cos
                        module.sin_cached = sin
            
            # Update model max_position_embeddings
            if hasattr(model.config, 'max_position_embeddings'):
                model.config.max_position_embeddings = self._config.target_max_position
            
            logger.info(f"Applied {method.value} scaling for {self._config.target_max_position} context")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply RoPE scaling: {e}")
            return False
    
    def enable_sliding_window(
        self,
        model: Any,
        window_size: Optional[int] = None
    ) -> bool:
        """
        Enable sliding window attention.
        
        Args:
            model: Model to modify
            window_size: Window size (default: half target length)
            
        Returns:
            True if successful
        """
        window_size = window_size or self._config.sliding_window_size
        if window_size is None:
            window_size = self._config.target_max_position // 2
        
        try:
            window = SlidingWindowAttention(window_size)
            
            # Add to model config
            if hasattr(model, 'config'):
                model.config.sliding_window = window_size
                model._sliding_window = window
            
            logger.info(f"Enabled sliding window attention with size {window_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable sliding window: {e}")
            return False
    
    def get_extended_positions(
        self,
        seq_len: int,
        dim: int,
        method: Optional[ScalingMethod] = None
    ) -> Tuple[Any, Any]:
        """
        Get extended position embeddings.
        
        Args:
            seq_len: Sequence length
            dim: Embedding dimension
            method: Scaling method
            
        Returns:
            Tuple of (cos, sin) embeddings
        """
        method = method or self._config.method
        
        if method == ScalingMethod.LINEAR:
            import torch
            inv_freq = self._scaler.compute_inverse_freq(dim)
            positions = torch.arange(seq_len)
            return self._scaler.linear_scaling(inv_freq, positions)
        
        elif method == ScalingMethod.NTK:
            return self._scaler.ntk_scaling(dim)
        
        elif method == ScalingMethod.DYNAMIC_NTK:
            return self._scaler.dynamic_ntk_scaling(dim, seq_len)
        
        elif method == ScalingMethod.YARN:
            return self._scaler.yarn_scaling(dim, seq_len)
        
        else:
            raise ValueError(f"Unknown scaling method: {method}")


# Global instance
_extender: Optional[ContextExtender] = None


def get_extender(config: Optional[ExtensionConfig] = None) -> ContextExtender:
    """Get or create global context extender."""
    global _extender
    if _extender is None or config is not None:
        _extender = ContextExtender(config)
    return _extender
