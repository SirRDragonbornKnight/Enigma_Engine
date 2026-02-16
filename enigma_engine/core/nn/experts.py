"""
LoRA (Low-Rank Adaptation) for efficient fine-tuning.
"""
import math

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer for efficient fine-tuning.
    
    Adds trainable low-rank matrices to frozen pretrained weights.
    Instead of fine-tuning all weights, LoRA trains small adapter matrices
    that can be merged back into the base weights.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        rank: Rank of the low-rank matrices (smaller = fewer params)
        alpha: Scaling factor for the LoRA update
    
    Usage:
        # Wrap existing linear layer
        base_layer = nn.Linear(512, 512)
        lora = LoRALayer(512, 512, rank=8)
        
        # During forward:
        output = base_layer(x) + lora(x)
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices: W = A @ B where A is (in, rank) and B is (rank, out)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A @ self.lora_B) * self.scaling
    
    def merge_weights(self, base_weight: torch.Tensor) -> torch.Tensor:
        """Merge LoRA weights into base weight matrix."""
        return base_weight + (self.lora_A @ self.lora_B).T * self.scaling


__all__ = ["LoRALayer"]
