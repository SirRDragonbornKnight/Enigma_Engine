"""
Model Scaling Utilities

Allows you to:
  1. Grow a model (small -> medium -> large) while preserving learning
  2. Distill a large model into a smaller one
  3. Export/import core knowledge between model sizes

This is EXPERIMENTAL but allows your AI to "grow up" over time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, Tuple
import json

from .model import TinyEnigma
from .model_config import MODEL_PRESETS, get_model_config
from .model_registry import ModelRegistry


def grow_model(
    source_model: TinyEnigma,
    target_size: str,
    vocab_size: int,
    copy_weights: bool = True
) -> TinyEnigma:
    """
    Grow a model to a larger size while preserving learned weights.
    
    The new model will have:
    - Existing weights copied to matching positions
    - New weights initialized randomly
    - Same learned patterns, but more capacity to learn
    
    Args:
        source_model: The trained smaller model
        target_size: Target size preset ("medium", "large", etc.)
        vocab_size: Vocabulary size (must match source)
        copy_weights: Whether to copy existing weights
        
    Returns:
        New larger model with transferred knowledge
    """
    source_config = {
        "dim": source_model.dim,
        "depth": len(source_model.layers),
    }
    target_config = get_model_config(target_size)
    target_config["vocab_size"] = vocab_size
    
    print(f"Growing model: dim {source_config['dim']} -> {target_config['dim']}, "
          f"depth {source_config['depth']} -> {target_config['depth']}")
    
    # Create new model
    new_model = TinyEnigma(**target_config)
    
    if copy_weights:
        with torch.no_grad():
            # Copy token embeddings (expand dimensions)
            src_embed = source_model.token_embed.weight
            src_dim = src_embed.shape[1]
            new_dim = new_model.token_embed.weight.shape[1]
            
            # Copy what fits, rest stays random initialized
            min_dim = min(src_dim, new_dim)
            new_model.token_embed.weight[:, :min_dim] = src_embed[:, :min_dim]
            
            # Copy output head
            src_head = source_model.head.weight
            new_model.head.weight[:, :min_dim] = src_head[:, :min_dim]
            new_model.head.bias[:] = source_model.head.bias[:]
            
            # Copy positional embeddings
            src_pos = source_model.pos
            min_len = min(src_pos.shape[1], new_model.pos.shape[1])
            new_model.pos[:, :min_len, :min_dim] = src_pos[:, :min_len, :min_dim]
            
            # Copy transformer layers (as many as we have)
            min_layers = min(len(source_model.layers), len(new_model.layers))
            for i in range(min_layers):
                src_layer = source_model.layers[i]
                tgt_layer = new_model.layers[i]
                
                # Copy self-attention weights (partial)
                _copy_partial_linear(src_layer.self_attn.in_proj_weight, 
                                    tgt_layer.self_attn.in_proj_weight)
                _copy_partial_linear(src_layer.self_attn.out_proj.weight,
                                    tgt_layer.self_attn.out_proj.weight)
                
                # Copy feedforward weights (partial)
                _copy_partial_linear(src_layer.linear1.weight, tgt_layer.linear1.weight)
                _copy_partial_linear(src_layer.linear2.weight, tgt_layer.linear2.weight)
    
    print(f"[OK] Model grown successfully. New parameters: {sum(p.numel() for p in new_model.parameters()):,}")
    return new_model


def _copy_partial_linear(src: torch.Tensor, tgt: torch.Tensor):
    """Copy weights from smaller to larger tensor."""
    min_0 = min(src.shape[0], tgt.shape[0])
    min_1 = min(src.shape[1], tgt.shape[1]) if len(src.shape) > 1 else 1
    
    if len(src.shape) == 1:
        tgt[:min_0] = src[:min_0]
    else:
        tgt[:min_0, :min_1] = src[:min_0, :min_1]


def shrink_model(
    source_model: TinyEnigma,
    target_size: str,
    vocab_size: int,
) -> TinyEnigma:
    """
    Shrink a model to a smaller size (loses some capacity).
    
    Useful for deploying on weaker hardware.
    Note: This is lossy - some knowledge will be lost.
    """
    target_config = get_model_config(target_size)
    target_config["vocab_size"] = vocab_size
    
    new_model = TinyEnigma(**target_config)
    
    with torch.no_grad():
        # Copy what fits
        new_dim = new_model.dim
        
        new_model.token_embed.weight[:] = source_model.token_embed.weight[:, :new_dim]
        new_model.head.weight[:] = source_model.head.weight[:, :new_dim]
        new_model.head.bias[:] = source_model.head.bias[:]
        
        min_len = min(source_model.pos.shape[1], new_model.pos.shape[1])
        new_model.pos[:, :min_len, :] = source_model.pos[:, :min_len, :new_dim]
        
        # Copy layers
        for i in range(len(new_model.layers)):
            if i < len(source_model.layers):
                src_layer = source_model.layers[i]
                tgt_layer = new_model.layers[i]
                _copy_partial_linear(src_layer.self_attn.in_proj_weight,
                                    tgt_layer.self_attn.in_proj_weight)
                _copy_partial_linear(src_layer.self_attn.out_proj.weight,
                                    tgt_layer.self_attn.out_proj.weight)
                _copy_partial_linear(src_layer.linear1.weight, tgt_layer.linear1.weight)
                _copy_partial_linear(src_layer.linear2.weight, tgt_layer.linear2.weight)
    
    print(f"[OK] Model shrunk. New parameters: {sum(p.numel() for p in new_model.parameters()):,}")
    return new_model


def grow_registered_model(
    registry: ModelRegistry,
    source_name: str,
    target_name: str,
    target_size: str,
    description: str = ""
) -> TinyEnigma:
    """
    Grow a model from the registry and save as a new model.
    
    Example:
        # Start with small model
        registry.create_model("enigma_v1", size="small")
        # Train it...
        
        # Grow it to medium
        grow_registered_model(registry, "enigma_v1", "enigma_v2", "medium")
        # Continue training the larger model...
    """
    # Load source
    source_model, source_config = registry.load_model(source_name)
    vocab_size = source_config["vocab_size"]
    
    # Grow
    new_model = grow_model(source_model, target_size, vocab_size)
    
    # Register new model
    target_config = get_model_config(target_size)
    target_config["vocab_size"] = vocab_size
    
    # Create in registry
    registry.create_model(
        name=target_name,
        size=target_size,
        vocab_size=vocab_size,
        description=description or f"Grown from {source_name}",
        custom_config=target_config
    )
    
    # Save weights
    registry.save_model(target_name, new_model)
    
    # Update metadata
    registry.update_metadata(
        target_name,
        grown_from=source_name,
        growth_note=f"Grew from {source_name} ({registry.registry['models'][source_name]['size']} -> {target_size})"
    )
    
    print(f"[OK] Created '{target_name}' by growing '{source_name}'")
    return new_model


class KnowledgeDistiller:
    """
    Train a smaller "student" model to mimic a larger "teacher" model.
    
    This lets you:
    - Train a large model on your PC
    - Distill it to a small model for your Pi
    
    The small model learns to produce similar outputs to the large model.
    """
    
    def __init__(
        self,
        teacher: TinyEnigma,
        student: TinyEnigma,
        temperature: float = 2.0,
        alpha: float = 0.5
    ):
        """
        Args:
            teacher: Large trained model
            student: Small model to train
            temperature: Softmax temperature for distillation
            alpha: Weight between distillation loss and regular loss
        """
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        
        self.teacher.eval()  # Teacher doesn't train
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined distillation and classification loss.
        """
        T = self.temperature
        
        # Soft targets from teacher
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        
        # KL divergence (distillation loss)
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
        
        # Regular cross-entropy loss
        ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        
        # Combined loss
        return self.alpha * distill_loss + (1 - self.alpha) * ce_loss
    
    def distill_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
        """Single distillation training step."""
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids)
        
        student_logits = self.student(input_ids)
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        
        return loss


if __name__ == "__main__":
    print("Model Scaling Utilities")
    print("Use grow_model() to expand a trained model")
    print("Use shrink_model() to compress for deployment")
    print("Use KnowledgeDistiller to transfer knowledge to smaller models")
