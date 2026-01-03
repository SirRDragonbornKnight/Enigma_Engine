"""
Model Scaling Utilities

Allows you to:
  1. Grow a model (small -> medium -> large) while preserving learning
  2. Distill a large model into a smaller one
  3. Export/import core knowledge between model sizes

This is EXPERIMENTAL but allows your AI to "grow up" over time.
"""

import torch
import torch.nn.functional as F

from .model import Enigma, EnigmaConfig, MODEL_PRESETS
from .model_registry import ModelRegistry


def get_model_config(size: str) -> dict:
    """Get model config dict from MODEL_PRESETS."""
    if size not in MODEL_PRESETS:
        raise ValueError(f"Unknown model size: {size}. Available: {list(MODEL_PRESETS.keys())}")
    preset = MODEL_PRESETS[size]
    return preset.to_dict()


def grow_model(
    source_model: Enigma,
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
        New larger model with transferred knowledge
    """
    source_config = {
        "dim": source_model.config.dim,
        "depth": source_model.config.n_layers,
    }
    target_config_dict = get_model_config(target_size)
    target_config_dict["vocab_size"] = vocab_size
    target_config = EnigmaConfig.from_dict(target_config_dict)

    print(f"Growing model: dim {source_config['dim']} -> {target_config.dim}, "
          f"depth {source_config['depth']} -> {target_config.n_layers}")

    # Create new model
    new_model = Enigma(config=target_config)

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
            src_head = source_model.output.weight
            new_model.output.weight[:, :min_dim] = src_head[:, :min_dim]

            # Copy transformer layers (as many as we have)
            min_layers = min(len(source_model.layers), len(new_model.layers))
            for i in range(min_layers):
                src_layer = source_model.layers[i]
                tgt_layer = new_model.layers[i]

                # Copy attention weights (partial) - these use MultiHeadAttention
                _copy_partial_tensor(src_layer.attention.wq.weight, tgt_layer.attention.wq.weight)
                _copy_partial_tensor(src_layer.attention.wk.weight, tgt_layer.attention.wk.weight)
                _copy_partial_tensor(src_layer.attention.wv.weight, tgt_layer.attention.wv.weight)
                _copy_partial_tensor(src_layer.attention.wo.weight, tgt_layer.attention.wo.weight)

                # Copy feedforward weights (partial) - SwiGLU has w1, w2, w3
                if hasattr(src_layer.ffn, 'w1'):
                    _copy_partial_tensor(src_layer.ffn.w1.weight, tgt_layer.ffn.w1.weight)
                    _copy_partial_tensor(src_layer.ffn.w2.weight, tgt_layer.ffn.w2.weight)
                    _copy_partial_tensor(src_layer.ffn.w3.weight, tgt_layer.ffn.w3.weight)

    print(
        f"[OK] Model grown successfully. New parameters: {sum(p.numel() for p in new_model.parameters()):,}")
    return new_model


def _copy_partial_tensor(src: torch.Tensor, tgt: torch.Tensor):
    """Copy weights from smaller to larger tensor."""
    min_0 = min(src.shape[0], tgt.shape[0])
    min_1 = min(src.shape[1], tgt.shape[1]) if len(src.shape) > 1 else 1

    if len(src.shape) == 1:
        tgt[:min_0] = src[:min_0]
    else:
        tgt[:min_0, :min_1] = src[:min_0, :min_1]


def shrink_model(
    source_model: Enigma,
    target_size: str,
    vocab_size: int,
) -> Enigma:
    """
    Shrink a model to a smaller size (loses some capacity).

    Useful for deploying on weaker hardware.
    Note: This is lossy - some knowledge will be lost.
    """
    target_config_dict = get_model_config(target_size)
    target_config_dict["vocab_size"] = vocab_size
    target_config = EnigmaConfig.from_dict(target_config_dict)

    new_model = Enigma(config=target_config)

    with torch.no_grad():
        # Copy what fits
        new_dim = target_config.dim

        new_model.token_embed.weight[:] = source_model.token_embed.weight[:, :new_dim]
        new_model.output.weight[:] = source_model.output.weight[:, :new_dim]

        # Copy layers
        for i in range(len(new_model.layers)):
            if i < len(source_model.layers):
                src_layer = source_model.layers[i]
                tgt_layer = new_model.layers[i]
                
                # Copy attention weights
                _copy_partial_tensor(src_layer.attention.wq.weight, tgt_layer.attention.wq.weight)
                _copy_partial_tensor(src_layer.attention.wk.weight, tgt_layer.attention.wk.weight)
                _copy_partial_tensor(src_layer.attention.wv.weight, tgt_layer.attention.wv.weight)
                _copy_partial_tensor(src_layer.attention.wo.weight, tgt_layer.attention.wo.weight)
                
                # Copy FFN weights (SwiGLU)
                if hasattr(src_layer.ffn, 'w1'):
                    _copy_partial_tensor(src_layer.ffn.w1.weight, tgt_layer.ffn.w1.weight)
                    _copy_partial_tensor(src_layer.ffn.w2.weight, tgt_layer.ffn.w2.weight)
                    _copy_partial_tensor(src_layer.ffn.w3.weight, tgt_layer.ffn.w3.weight)

    print(f"[OK] Model shrunk. New parameters: {sum(p.numel() for p in new_model.parameters()):,}")
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
    source_size = registry.registry['models'][source_name]['size']
    registry.update_metadata(
        target_name,
        grown_from=source_name,
        growth_note=f"Grew from {source_name} ({source_size} -> {target_size})")

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
        teacher: Enigma,
        student: Enigma,
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
