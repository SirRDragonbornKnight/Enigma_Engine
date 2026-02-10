"""
Knowledge Distillation

Train smaller models from larger teacher models.
Supports various distillation strategies and loss functions.

FILE: enigma_engine/core/distillation.py
TYPE: Core/Training
MAIN CLASSES: Distiller, DistillationLoss, DistillationTrainer
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None


class DistillationType(Enum):
    """Types of knowledge distillation."""
    LOGIT = "logit"  # Classic soft label distillation
    FEATURE = "feature"  # Intermediate layer distillation
    ATTENTION = "attention"  # Attention map distillation
    RELATION = "relation"  # Relational knowledge distillation
    SELF = "self"  # Self-distillation (same model architecture)


class TemperatureSchedule(Enum):
    """Temperature scheduling strategies."""
    CONSTANT = "constant"
    LINEAR_DECAY = "linear_decay"
    COSINE_DECAY = "cosine_decay"
    ADAPTIVE = "adaptive"


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    # Distillation type
    distillation_type: DistillationType = DistillationType.LOGIT
    
    # Temperature for softening distributions
    temperature: float = 4.0
    temperature_schedule: TemperatureSchedule = TemperatureSchedule.CONSTANT
    final_temperature: float = 1.0
    
    # Loss weights
    alpha: float = 0.5  # Weight for distillation loss
    beta: float = 0.5  # Weight for task loss
    
    # Feature distillation
    feature_layers: list[str] = field(default_factory=list)
    feature_loss_type: str = "mse"  # mse, cosine, l1
    
    # Attention distillation
    attention_layers: list[str] = field(default_factory=list)
    
    # Training
    epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 32
    warmup_steps: int = 100
    
    # Regularization
    use_label_smoothing: bool = False
    label_smoothing: float = 0.1


class DistillationLoss(nn.Module):
    """Loss functions for knowledge distillation."""
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature
        self.alpha = config.alpha
        self.beta = config.beta
    
    def soft_cross_entropy(
        self,
        student_logits: "torch.Tensor",
        teacher_logits: "torch.Tensor",
        temperature: float = None
    ) -> "torch.Tensor":
        """
        Soft cross entropy loss between student and teacher logits.
        
        Args:
            student_logits: Student model output logits
            teacher_logits: Teacher model output logits
            temperature: Softening temperature
        
        Returns:
            Distillation loss
        """
        T = temperature or self.temperature
        
        # Soften distributions
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        
        # KL divergence
        loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction="batchmean"
        ) * (T * T)  # Scale by T^2
        
        return loss
    
    def feature_loss(
        self,
        student_features: "torch.Tensor",
        teacher_features: "torch.Tensor",
        loss_type: str = None
    ) -> "torch.Tensor":
        """
        Feature distillation loss.
        
        Args:
            student_features: Intermediate student features
            teacher_features: Intermediate teacher features
            loss_type: Type of feature loss
        
        Returns:
            Feature distillation loss
        """
        loss_type = loss_type or self.config.feature_loss_type
        
        # Handle dimension mismatch
        if student_features.shape != teacher_features.shape:
            # Project student features to match teacher dimensions
            if student_features.shape[-1] != teacher_features.shape[-1]:
                student_features = F.linear(
                    student_features,
                    torch.randn(
                        teacher_features.shape[-1],
                        student_features.shape[-1],
                        device=student_features.device
                    )
                )
        
        if loss_type == "mse":
            return F.mse_loss(student_features, teacher_features)
        elif loss_type == "cosine":
            return 1 - F.cosine_similarity(
                student_features.flatten(1),
                teacher_features.flatten(1)
            ).mean()
        elif loss_type == "l1":
            return F.l1_loss(student_features, teacher_features)
        
        return F.mse_loss(student_features, teacher_features)
    
    def attention_loss(
        self,
        student_attention: "torch.Tensor",
        teacher_attention: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Attention map distillation loss.
        
        Args:
            student_attention: Student attention weights [batch, heads, seq, seq]
            teacher_attention: Teacher attention weights
        
        Returns:
            Attention distillation loss
        """
        # Average over heads if dimension mismatch
        if student_attention.shape[1] != teacher_attention.shape[1]:
            # Mean pool attention heads
            student_attention = student_attention.mean(dim=1, keepdim=True)
            teacher_attention = teacher_attention.mean(dim=1, keepdim=True)
        
        # MSE on attention weights
        return F.mse_loss(student_attention, teacher_attention)
    
    def relation_loss(
        self,
        student_features: "torch.Tensor",
        teacher_features: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Relational knowledge distillation loss.
        
        Transfers structural relationships between samples.
        """
        # Compute pairwise distances
        def pairwise_distance(x):
            x_flat = x.flatten(1)
            gram = torch.mm(x_flat, x_flat.t())
            diag = gram.diag().unsqueeze(1)
            return diag + diag.t() - 2 * gram
        
        student_dist = pairwise_distance(student_features)
        teacher_dist = pairwise_distance(teacher_features)
        
        # Normalize
        student_dist = F.normalize(student_dist, p=2, dim=1)
        teacher_dist = F.normalize(teacher_dist, p=2, dim=1)
        
        return F.mse_loss(student_dist, teacher_dist)
    
    def forward(
        self,
        student_outputs: dict[str, "torch.Tensor"],
        teacher_outputs: dict[str, "torch.Tensor"],
        labels: "torch.Tensor" = None
    ) -> dict[str, "torch.Tensor"]:
        """
        Compute combined distillation loss.
        
        Args:
            student_outputs: Dict with 'logits', 'features', 'attentions'
            teacher_outputs: Dict with 'logits', 'features', 'attentions'
            labels: Ground truth labels
        
        Returns:
            Dict with individual losses and total
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=student_outputs["logits"].device)
        
        # Logit distillation
        if self.config.distillation_type in (DistillationType.LOGIT, DistillationType.SELF):
            distill_loss = self.soft_cross_entropy(
                student_outputs["logits"],
                teacher_outputs["logits"]
            )
            losses["distillation"] = distill_loss
            total_loss = total_loss + self.alpha * distill_loss
        
        # Feature distillation
        if self.config.distillation_type == DistillationType.FEATURE:
            if "features" in student_outputs and "features" in teacher_outputs:
                feat_loss = torch.tensor(0.0, device=total_loss.device)
                for s_feat, t_feat in zip(
                    student_outputs["features"],
                    teacher_outputs["features"]
                ):
                    feat_loss = feat_loss + self.feature_loss(s_feat, t_feat)
                losses["feature"] = feat_loss
                total_loss = total_loss + self.alpha * feat_loss
        
        # Attention distillation
        if self.config.distillation_type == DistillationType.ATTENTION:
            if "attentions" in student_outputs and "attentions" in teacher_outputs:
                attn_loss = torch.tensor(0.0, device=total_loss.device)
                for s_attn, t_attn in zip(
                    student_outputs["attentions"],
                    teacher_outputs["attentions"]
                ):
                    attn_loss = attn_loss + self.attention_loss(s_attn, t_attn)
                losses["attention"] = attn_loss
                total_loss = total_loss + self.alpha * attn_loss
        
        # Relation distillation
        if self.config.distillation_type == DistillationType.RELATION:
            if "features" in student_outputs and "features" in teacher_outputs:
                rel_loss = self.relation_loss(
                    student_outputs["features"][-1],
                    teacher_outputs["features"][-1]
                )
                losses["relation"] = rel_loss
                total_loss = total_loss + self.alpha * rel_loss
        
        # Task loss (cross entropy with true labels)
        if labels is not None:
            task_loss = F.cross_entropy(
                student_outputs["logits"].view(-1, student_outputs["logits"].size(-1)),
                labels.view(-1)
            )
            losses["task"] = task_loss
            total_loss = total_loss + self.beta * task_loss
        
        losses["total"] = total_loss
        return losses


class FeatureExtractor:
    """Extract intermediate features from models."""
    
    def __init__(self, model: nn.Module, layer_names: list[str]):
        self.model = model
        self.layer_names = layer_names
        self.features: dict[str, "torch.Tensor"] = {}
        self.attentions: list["torch.Tensor"] = []
        self._hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture features."""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(
                    lambda mod, inp, out, n=name: self._save_feature(n, out)
                )
                self._hooks.append(hook)
    
    def _save_feature(self, name: str, output):
        """Save feature from hook."""
        if isinstance(output, tuple):
            self.features[name] = output[0].detach()
            if len(output) > 1 and output[1] is not None:
                self.attentions.append(output[1].detach())
        else:
            self.features[name] = output.detach()
    
    def get_features(self) -> dict[str, "torch.Tensor"]:
        """Get extracted features."""
        return self.features
    
    def get_attentions(self) -> list["torch.Tensor"]:
        """Get extracted attention maps."""
        return self.attentions
    
    def clear(self):
        """Clear stored features."""
        self.features.clear()
        self.attentions.clear()
    
    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class DistillationTrainer:
    """Trainer for knowledge distillation."""
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: DistillationConfig = None
    ):
        self.teacher = teacher
        self.student = student
        self.config = config or DistillationConfig()
        
        self.loss_fn = DistillationLoss(self.config)
        
        # Feature extractors
        self.teacher_extractor = None
        self.student_extractor = None
        
        if self.config.feature_layers:
            self.teacher_extractor = FeatureExtractor(
                teacher, self.config.feature_layers
            )
            self.student_extractor = FeatureExtractor(
                student, self.config.feature_layers
            )
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Optimizer for student
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.learning_rate
        )
        
        self.current_step = 0
    
    def get_temperature(self) -> float:
        """Get current temperature based on schedule."""
        if self.config.temperature_schedule == TemperatureSchedule.CONSTANT:
            return self.config.temperature
        
        total_steps = self.config.epochs * 1000  # Approximate
        progress = min(self.current_step / total_steps, 1.0)
        
        if self.config.temperature_schedule == TemperatureSchedule.LINEAR_DECAY:
            return self.config.temperature * (1 - progress) + \
                   self.config.final_temperature * progress
        
        elif self.config.temperature_schedule == TemperatureSchedule.COSINE_DECAY:
            import math
            cosine = (1 + math.cos(math.pi * progress)) / 2
            return self.config.final_temperature + \
                   (self.config.temperature - self.config.final_temperature) * cosine
        
        return self.config.temperature
    
    def train_step(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor" = None,
        labels: "torch.Tensor" = None
    ) -> dict[str, float]:
        """
        Single training step.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels
        
        Returns:
            Dict with loss values
        """
        self.student.train()
        
        # Clear feature extractors
        if self.teacher_extractor:
            self.teacher_extractor.clear()
            self.student_extractor.clear()
        
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_output = self.teacher(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=self.config.distillation_type == DistillationType.ATTENTION,
                output_hidden_states=self.config.distillation_type == DistillationType.FEATURE
            )
        
        # Student forward
        student_output = self.student(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=self.config.distillation_type == DistillationType.ATTENTION,
            output_hidden_states=self.config.distillation_type == DistillationType.FEATURE
        )
        
        # Prepare outputs dict
        student_outputs = {"logits": student_output.logits if hasattr(student_output, "logits") else student_output}
        teacher_outputs = {"logits": teacher_output.logits if hasattr(teacher_output, "logits") else teacher_output}
        
        if hasattr(student_output, "hidden_states") and student_output.hidden_states:
            student_outputs["features"] = list(student_output.hidden_states)
            teacher_outputs["features"] = list(teacher_output.hidden_states)
        
        if hasattr(student_output, "attentions") and student_output.attentions:
            student_outputs["attentions"] = list(student_output.attentions)
            teacher_outputs["attentions"] = list(teacher_output.attentions)
        
        # Compute loss
        losses = self.loss_fn(student_outputs, teacher_outputs, labels)
        
        # Backward
        self.optimizer.zero_grad()
        losses["total"].backward()
        self.optimizer.step()
        
        self.current_step += 1
        
        return {k: v.item() for k, v in losses.items()}
    
    def train(
        self,
        train_dataloader,
        eval_dataloader = None,
        progress_callback: Callable[[int, int, dict], None] = None
    ) -> dict[str, list[float]]:
        """
        Full training loop.
        
        Args:
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            progress_callback: Callback for progress updates
        
        Returns:
            Training history
        """
        history = {"train_loss": [], "eval_loss": []}
        
        for epoch in range(self.config.epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Move to device
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
                labels = batch.get("labels")
                
                losses = self.train_step(input_ids, attention_mask, labels)
                epoch_losses.append(losses["total"])
                
                if progress_callback:
                    progress_callback(epoch, batch_idx, losses)
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            history["train_loss"].append(avg_loss)
            
            # Evaluation
            if eval_dataloader:
                eval_loss = self.evaluate(eval_dataloader)
                history["eval_loss"].append(eval_loss)
                logger.info(f"Epoch {epoch+1}/{self.config.epochs} - "
                           f"Train Loss: {avg_loss:.4f}, Eval Loss: {eval_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{self.config.epochs} - "
                           f"Train Loss: {avg_loss:.4f}")
        
        return history
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> float:
        """Evaluate distillation loss on a dataset."""
        self.student.eval()
        total_loss = 0.0
        count = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
            labels = batch.get("labels")
            
            teacher_output = self.teacher(input_ids, attention_mask=attention_mask)
            student_output = self.student(input_ids, attention_mask=attention_mask)
            
            # Simple logit distillation loss for eval
            loss = self.loss_fn.soft_cross_entropy(
                student_output.logits if hasattr(student_output, "logits") else student_output,
                teacher_output.logits if hasattr(teacher_output, "logits") else teacher_output
            )
            
            total_loss += loss.item()
            count += 1
        
        return total_loss / count if count > 0 else 0.0
    
    def save_student(self, path: str):
        """Save distilled student model."""
        torch.save({
            "model_state_dict": self.student.state_dict(),
            "config": self.config.__dict__
        }, path)
        logger.info(f"Saved student model to {path}")


class Distiller:
    """
    Main distillation coordinator.
    
    High-level API for knowledge distillation.
    """
    
    def __init__(self, config: DistillationConfig = None):
        self.config = config or DistillationConfig()
    
    def distill(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_data,
        eval_data = None
    ) -> nn.Module:
        """
        Distill knowledge from teacher to student.
        
        Args:
            teacher: Teacher model
            student: Student model
            train_data: Training data loader
            eval_data: Evaluation data loader
        
        Returns:
            Trained student model
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for distillation")
        
        trainer = DistillationTrainer(teacher, student, self.config)
        trainer.train(train_data, eval_data)
        
        return student
    
    @staticmethod
    def create_student_from_teacher(
        teacher: nn.Module,
        reduction_factor: float = 0.5
    ) -> nn.Module:
        """
        Create a smaller student model based on teacher architecture.
        
        Args:
            teacher: Teacher model
            reduction_factor: Size reduction factor (0.5 = half size)
        
        Returns:
            Student model with reduced dimensions
        """
        # This is architecture-specific
        # For Transformer models, reduce:
        # - n_layers
        # - d_model
        # - n_heads
        # - d_ff
        
        raise NotImplementedError(
            "create_student_from_teacher requires architecture-specific implementation"
        )


def distill_model(
    teacher: nn.Module,
    student: nn.Module,
    train_data,
    temperature: float = 4.0,
    alpha: float = 0.5,
    epochs: int = 3
) -> nn.Module:
    """
    Convenience function for knowledge distillation.
    
    Args:
        teacher: Teacher model
        student: Student model
        train_data: Training data loader
        temperature: Distillation temperature
        alpha: Distillation loss weight
        epochs: Training epochs
    
    Returns:
        Distilled student model
    """
    config = DistillationConfig(
        temperature=temperature,
        alpha=alpha,
        beta=1 - alpha,
        epochs=epochs
    )
    
    distiller = Distiller(config)
    return distiller.distill(teacher, student, train_data)
