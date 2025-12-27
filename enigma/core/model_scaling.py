"""
Model Scaling Utilities

Allows you to:
  1. Grow a model (small → medium → large) while preserving learning
  2. Shrink a model for deployment on weaker hardware
  3. Distill a large model into a smaller one (knowledge transfer)
  4. Export/import core knowledge between model sizes

This is EXPERIMENTAL but allows your AI to "grow up" over time.

USAGE:
    from enigma.core.model_scaling import grow_model, shrink_model, KnowledgeDistiller
    
    # Grow a trained model
    larger_model = grow_model(small_model, "medium", vocab_size=32000)
    
    # Shrink for deployment
    tiny_model = shrink_model(large_model, "tiny", vocab_size=32000)
    
    # Distill knowledge
    distiller = KnowledgeDistiller(teacher=large_model, student=small_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
import json

from .model import Enigma, TinyEnigma  # TinyEnigma is backwards compat alias
from .model_config import MODEL_PRESETS, get_model_config
from .model_registry import ModelRegistry


def grow_model(
    source_model: Union[Enigma, TinyEnigma],
    target_size: str,
    vocab_size: int,
    copy_weights: bool = True
) -> Enigma:
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
        New larger Enigma model with transferred knowledge
    """
    source_config = {
        "dim": source_model.dim,
        "depth": len(source_model.layers),
        "heads": source_model.heads if hasattr(source_model, 'heads') else source_model.dim // 64,
    }
    target_config = get_model_config(target_size)
    target_config["vocab_size"] = vocab_size
    
    print(f"Growing model: dim {source_config['dim']} → {target_config['dim']}, "
          f"depth {source_config['depth']} → {target_config['depth']}")
    
    # Create new model
    new_model = Enigma(**target_config)
    
    if copy_weights:
        with torch.no_grad():
            # Copy token embeddings (expand dimensions)
            src_embed = source_model.token_embed.weight
            src_dim = src_embed.shape[1]
            new_dim = new_model.token_embed.weight.shape[1]
            
            # Copy what fits, rest stays random initialized
            min_dim = min(src_dim, new_dim)
            min_vocab = min(src_embed.shape[0], new_model.token_embed.weight.shape[0])
            new_model.token_embed.weight[:min_vocab, :min_dim] = src_embed[:min_vocab, :min_dim]
            
            # Copy output head (weight tied to embedding in Enigma, so this is redundant but safe)
            if hasattr(source_model, 'head') and source_model.head.weight is not source_model.token_embed.weight:
                src_head = source_model.head.weight
                new_model.head.weight[:min_vocab, :min_dim] = src_head[:min_vocab, :min_dim]
            
            # Copy transformer layers (as many as we have)
            min_layers = min(len(source_model.layers), len(new_model.layers))
            for i in range(min_layers):
                src_layer = source_model.layers[i]
                tgt_layer = new_model.layers[i]
                
                # Copy attention weights
                if hasattr(src_layer, 'attention') and hasattr(tgt_layer, 'attention'):
                    _copy_attention_weights(src_layer.attention, tgt_layer.attention, min_dim, target_config['dim'])
                
                # Copy FFN weights
                if hasattr(src_layer, 'ffn') and hasattr(tgt_layer, 'ffn'):
                    _copy_ffn_weights(src_layer.ffn, tgt_layer.ffn, min_dim, target_config['dim'])
                
                # Copy norm weights
                if hasattr(src_layer, 'attention_norm') and hasattr(tgt_layer, 'attention_norm'):
                    tgt_layer.attention_norm.weight[:min_dim] = src_layer.attention_norm.weight[:min_dim]
                if hasattr(src_layer, 'ffn_norm') and hasattr(tgt_layer, 'ffn_norm'):
                    tgt_layer.ffn_norm.weight[:min_dim] = src_layer.ffn_norm.weight[:min_dim]
    
    print(f"✓ Model grown successfully. New parameters: {sum(p.numel() for p in new_model.parameters()):,}")
    return new_model


def _copy_attention_weights(src_attn, tgt_attn, src_dim: int, tgt_dim: int):
    """Copy attention weights between different sized models."""
    min_dim = min(src_dim, tgt_dim)
    
    # Copy projection weights
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        if hasattr(src_attn, proj_name) and hasattr(tgt_attn, proj_name):
            src_proj = getattr(src_attn, proj_name)
            tgt_proj = getattr(tgt_attn, proj_name)
            
            out_min = min(src_proj.weight.shape[0], tgt_proj.weight.shape[0])
            in_min = min(src_proj.weight.shape[1], tgt_proj.weight.shape[1])
            
            tgt_proj.weight[:out_min, :in_min] = src_proj.weight[:out_min, :in_min]
            if src_proj.bias is not None and tgt_proj.bias is not None:
                tgt_proj.bias[:out_min] = src_proj.bias[:out_min]


def _copy_ffn_weights(src_ffn, tgt_ffn, src_dim: int, tgt_dim: int):
    """Copy FFN weights between different sized models."""
    # SwiGLU style FFN
    for weight_name in ['w1', 'w2', 'w3']:
        if hasattr(src_ffn, weight_name) and hasattr(tgt_ffn, weight_name):
            src_w = getattr(src_ffn, weight_name)
            tgt_w = getattr(tgt_ffn, weight_name)
            
            out_min = min(src_w.weight.shape[0], tgt_w.weight.shape[0])
            in_min = min(src_w.weight.shape[1], tgt_w.weight.shape[1])
            
            tgt_w.weight[:out_min, :in_min] = src_w.weight[:out_min, :in_min]


def _copy_partial_linear(src: torch.Tensor, tgt: torch.Tensor):
    """Copy weights from smaller to larger tensor."""
    min_0 = min(src.shape[0], tgt.shape[0])
    min_1 = min(src.shape[1], tgt.shape[1]) if len(src.shape) > 1 else 1
    
    if len(src.shape) == 1:
        tgt[:min_0] = src[:min_0]
    else:
        tgt[:min_0, :min_1] = src[:min_0, :min_1]


def shrink_model(
    source_model: Union[Enigma, TinyEnigma],
    target_size: str,
    vocab_size: int,
) -> Enigma:
    """
    Shrink a model to a smaller size (loses some capacity).
    
    Useful for deploying on weaker hardware.
    Note: This is lossy - some knowledge will be lost.
    
    Args:
        source_model: The trained larger model
        target_size: Target size preset ("tiny", "small", etc.)
        vocab_size: Vocabulary size
        
    Returns:
        Smaller Enigma model
    """
    target_config = get_model_config(target_size)
    target_config["vocab_size"] = vocab_size
    
    new_model = Enigma(**target_config)
    new_dim = target_config["dim"]
    
    with torch.no_grad():
        # Copy what fits
        min_vocab = min(source_model.token_embed.weight.shape[0], new_model.token_embed.weight.shape[0])
        new_model.token_embed.weight[:min_vocab, :] = source_model.token_embed.weight[:min_vocab, :new_dim]
        
        # Copy layers
        for i in range(len(new_model.layers)):
            if i < len(source_model.layers):
                src_layer = source_model.layers[i]
                tgt_layer = new_model.layers[i]
                
                # Copy attention
                if hasattr(src_layer, 'attention') and hasattr(tgt_layer, 'attention'):
                    _copy_attention_weights(src_layer.attention, tgt_layer.attention, source_model.dim, new_dim)
                
                # Copy FFN
                if hasattr(src_layer, 'ffn') and hasattr(tgt_layer, 'ffn'):
                    _copy_ffn_weights(src_layer.ffn, tgt_layer.ffn, source_model.dim, new_dim)
                
                # Copy norms
                if hasattr(src_layer, 'attention_norm') and hasattr(tgt_layer, 'attention_norm'):
                    tgt_layer.attention_norm.weight[:] = src_layer.attention_norm.weight[:new_dim]
                if hasattr(src_layer, 'ffn_norm') and hasattr(tgt_layer, 'ffn_norm'):
                    tgt_layer.ffn_norm.weight[:] = src_layer.ffn_norm.weight[:new_dim]
    
    print(f"✓ Model shrunk. New parameters: {sum(p.numel() for p in new_model.parameters()):,}")
    return new_model


def grow_registered_model(
    registry: ModelRegistry,
    source_name: str,
    target_name: str,
    target_size: str,
    description: str = ""
) -> Enigma:
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
        growth_note=f"Grew from {source_name} ({registry.registry['models'][source_name]['size']} → {target_size})"
    )
    
    print(f"✓ Created '{target_name}' by growing '{source_name}'")
    return new_model


class KnowledgeDistiller:
    """
    Train a smaller "student" model to mimic a larger "teacher" model.
    
    This lets you:
    - Train a large model on your PC
    - Distill it to a small model for your Pi/mobile
    
    The small model learns to produce similar outputs to the large model.
    
    USAGE:
        distiller = KnowledgeDistiller(
            teacher=large_model,
            student=small_model,
            temperature=2.0,
            alpha=0.5
        )
        
        for batch in dataloader:
            loss = distiller.distill_step(batch["input_ids"], batch["labels"])
            loss.backward()
            optimizer.step()
    """
    
    def __init__(
        self,
        teacher: Union[Enigma, TinyEnigma],
        student: Union[Enigma, TinyEnigma],
        temperature: float = 2.0,
        alpha: float = 0.5
    ):
        """
        Args:
            teacher: Large trained model
            student: Small model to train
            temperature: Softmax temperature for distillation (higher = softer targets)
            alpha: Weight between distillation loss and regular loss
                   (0.5 = equal weight, 1.0 = only distillation)
        """
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        
        self.teacher.eval()  # Teacher doesn't train
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined distillation and classification loss.
        
        Args:
            student_logits: Output from student model
            teacher_logits: Output from teacher model
            labels: Ground truth labels
            
        Returns:
            Combined loss tensor
        """
        T = self.temperature
        
        # Soft targets from teacher
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        
        # KL divergence (distillation loss)
        # Scale by T^2 as per Hinton et al.
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
        
        # Regular cross-entropy loss
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=0  # Ignore padding
        )
        
        # Combined loss
        return self.alpha * distill_loss + (1 - self.alpha) * ce_loss
    
    def distill_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Single distillation training step.
        
        Args:
            input_ids: Input token IDs
            labels: Target labels
            
        Returns:
            Loss tensor (call .backward() on this)
        """
        with torch.no_grad():
            teacher_output = self.teacher(input_ids)
            if isinstance(teacher_output, tuple):
                teacher_logits = teacher_output[0]
            else:
                teacher_logits = teacher_output
        
        student_output = self.student(input_ids)
        if isinstance(student_output, tuple):
            student_logits = student_output[0]
        else:
            student_logits = student_output
            
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        
        return loss
    
    def distill(
        self,
        dataloader,
        optimizer,
        epochs: int = 10,
        log_every: int = 100,
    ):
        """
        Full distillation training loop.
        
        Args:
            dataloader: PyTorch DataLoader with training data
            optimizer: Optimizer for student model
            epochs: Number of training epochs
            log_every: Log progress every N steps
        """
        device = next(self.student.parameters()).device
        self.student.train()
        global_step = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                optimizer.zero_grad()
                loss = self.distill_step(input_ids, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                if global_step % log_every == 0:
                    print(f"  Step {global_step} | Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    print("Model Scaling Utilities")
    print("Use grow_model() to expand a trained model")
    print("Use shrink_model() to compress for deployment")
    print("Use KnowledgeDistiller to transfer knowledge to smaller models")
