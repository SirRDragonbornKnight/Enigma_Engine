"""
Activation Functions for Enigma

Contains:
- SwiGLU: Gated Linear Unit with SiLU activation
- GeGLU: Gated Linear Unit with GELU activation
- ReGLU: Gated Linear Unit with ReLU activation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    
    A gated linear unit that uses SiLU (Swish) as the activation.
    Used in modern transformers like LLaMA for better performance.
    
    SwiGLU(x) = SiLU(xW) ⊙ (xV) where ⊙ is element-wise multiplication
    
    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (typically 4 * dim)
        bias: Whether to use bias in linear layers
    """
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GeGLU(nn.Module):
    """
    GEGLU activation function.
    
    A gated linear unit that uses GELU as the activation.
    GeGLU(x) = GELU(xW) ⊙ (xV)
    
    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension
        bias: Whether to use bias
    """
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)) * self.w3(x))


class ReGLU(nn.Module):
    """
    ReGLU activation function.
    
    A gated linear unit that uses ReLU as the activation.
    ReGLU(x) = ReLU(xW) ⊙ (xV)
    
    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension
        bias: Whether to use bias
    """
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)) * self.w3(x))


class FeedForward(nn.Module):
    """
    Standard feed-forward network.
    
    FFN(x) = GELU(xW1)W2
    
    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension
        dropout: Dropout rate
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))
