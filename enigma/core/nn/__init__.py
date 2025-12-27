"""
Enigma Core - Neural Network Components

This subpackage contains modular neural network components:
- attention.py: Attention mechanisms (MHA, GQA, Sliding Window)
- activations.py: Activation functions (SwiGLU, GeGLU, ReGLU)
- normalization.py: Normalization layers (RMSNorm)
- embeddings.py: Positional embeddings (RoPE, Sinusoidal)
- experts.py: MoE, LoRA, and advanced components
"""

from .attention import MultiHeadAttention, GroupedQueryAttention, SlidingWindowAttention
from .activations import SwiGLU, GeGLU, ReGLU, FeedForward
from .normalization import RMSNorm, AdaptiveRMSNorm
from .embeddings import RotaryEmbedding, SinusoidalEmbedding, LearnedEmbedding
from .experts import LoRALayer, MixtureOfExperts, AdaptiveLayerNorm, MultiQueryAttention

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
