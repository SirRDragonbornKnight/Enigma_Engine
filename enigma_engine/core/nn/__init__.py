"""
Forge Core - Neural Network Components

This subpackage contains modular neural network components:
- attention.py: Attention mechanisms (MHA, GQA, Sliding Window)
- activations.py: Activation functions (SwiGLU, GeGLU, ReGLU)
- normalization.py: Normalization layers (RMSNorm)
- embeddings.py: Positional embeddings (RoPE, Sinusoidal)
- experts.py: MoE, LoRA, and advanced components
"""

from .activations import FeedForward, GeGLU, ReGLU, SwiGLU
from .attention import GroupedQueryAttention, MultiHeadAttention, SlidingWindowAttention
from .embeddings import LearnedEmbedding, RotaryEmbedding, SinusoidalEmbedding
from .experts import AdaptiveLayerNorm, LoRALayer, MixtureOfExperts, MultiQueryAttention
from .normalization import AdaptiveRMSNorm, RMSNorm

__all__ = [
    # Attention
    "MultiHeadAttention",
    "GroupedQueryAttention", 
    "SlidingWindowAttention",
    "MultiQueryAttention",
    # Activations
    "SwiGLU",
    "GeGLU",
    "ReGLU",
    "FeedForward",
    # Normalization
    "RMSNorm",
    "AdaptiveRMSNorm",
    "AdaptiveLayerNorm",
    # Embeddings
    "RotaryEmbedding",
    "SinusoidalEmbedding",
    "LearnedEmbedding",
    # Advanced
    "LoRALayer",
    "MixtureOfExperts",
]
