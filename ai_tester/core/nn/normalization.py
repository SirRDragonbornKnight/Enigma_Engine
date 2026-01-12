"""
Normalization Layers for Enigma

Contains:
- RMSNorm: Root Mean Square Layer Normalization
"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More stable and efficient than LayerNorm for transformer models.
    Normalizes by the RMS of activations without centering.
    
    Args:
        dim: Dimension to normalize over
        eps: Small constant for numerical stability
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Normalized tensor of same shape
        """
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class AdaptiveRMSNorm(nn.Module):
    """
    Adaptive RMS Normalization with learned scale.
    
    Similar to RMSNorm but with an additional learned bias.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight + self.bias
