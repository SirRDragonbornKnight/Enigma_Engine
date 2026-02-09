"""
================================================================================
DPO Training - Direct Preference Optimization
================================================================================

Simpler alternative to RLHF. Aligns model with human preferences using
just supervised learning - no reward model or PPO needed.

BENEFITS:
    - Same results as RLHF with 10x less complexity
    - Just need preference pairs (chosen vs rejected)
    - Works with any transformer model

📍 FILE: enigma_engine/core/dpo_training.py
🏷️ TYPE: Training Algorithm

USAGE:
    from enigma_engine.core.dpo_training import DPOTrainer, DPOConfig
    
    # Prepare preference data
    preferences = [
        {
            "prompt": "What is 2+2?",
            "chosen": "2+2 equals 4.",
            "rejected": "2+2 equals 5."
        },
        ...
    ]
    
    # Train
    trainer = DPOTrainer(model, tokenizer, ref_model)
    trainer.train(preferences, epochs=3)

DATA FORMAT:
    Each preference pair has:
    - prompt: The input question/context
    - chosen: The preferred response
    - rejected: The less preferred response
"""

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# DPO Configuration
# =============================================================================

@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    
    # Core DPO parameters
    beta: float = 0.1  # Temperature parameter (higher = more conservative)
    
    # Training parameters
    learning_rate: float = 1e-6  # Lower LR for fine-tuning
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Sequence parameters
    max_length: int = 512
    max_prompt_length: int = 256
    
    # Regularization
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    
    # Reference model
    use_reference_free: bool = False  # If True, don't use reference model
    
    # Logging
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 500


# =============================================================================
# Preference Dataset
# =============================================================================

class PreferenceDataset(Dataset):
    """Dataset for preference pairs."""
    
    def __init__(
        self,
        data: list[dict[str, str]],
        tokenizer: Any,
        max_length: int = 512,
        max_prompt_length: int = 256
    ):
        """
        Initialize dataset.
        
        Args:
            data: List of dicts with 'prompt', 'chosen', 'rejected'
            tokenizer: Tokenizer for encoding
            max_length: Max total sequence length
            max_prompt_length: Max prompt length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt)[:self.max_prompt_length]
        
        # Encode chosen response (prompt + chosen)
        chosen_tokens = self.tokenizer.encode(prompt + chosen)[:self.max_length]
        
        # Encode rejected response (prompt + rejected)
        rejected_tokens = self.tokenizer.encode(prompt + rejected)[:self.max_length]
        
        # Pad to same length
        max_len = max(len(chosen_tokens), len(rejected_tokens))
        
        chosen_padded = self._pad(chosen_tokens, max_len)
        rejected_padded = self._pad(rejected_tokens, max_len)
        
        # Create attention masks
        chosen_mask = torch.ones(max_len)
        chosen_mask[len(chosen_tokens):] = 0
        
        rejected_mask = torch.ones(max_len)
        rejected_mask[len(rejected_tokens):] = 0
        
        return {
            "chosen_ids": chosen_padded,
            "rejected_ids": rejected_padded,
            "chosen_mask": chosen_mask,
            "rejected_mask": rejected_mask,
            "prompt_length": len(prompt_tokens),
        }
    
    def _pad(self, tokens: list[int], length: int) -> torch.Tensor:
        """Pad token list to length."""
        padded = tokens + [0] * (length - len(tokens))
        return torch.tensor(padded, dtype=torch.long)


# =============================================================================
# DPO Loss Functions
# =============================================================================

def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute log probabilities of labels given the model.
    
    Args:
        model: The language model
        input_ids: Input token IDs [batch, seq]
        attention_mask: Attention mask [batch, seq]
        labels: Target labels [batch, seq]
        
    Returns:
        Log probabilities per sequence [batch]
    """
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        
        # Handle different output formats
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        
        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for target tokens
        target_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding and sum over sequence
        target_log_probs = target_log_probs * shift_mask
        sequence_log_probs = target_log_probs.sum(dim=-1)
        
        return sequence_log_probs


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute DPO loss.
    
    The DPO objective is:
    L_DPO = -E[log sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x))))]
    
    Where:
    - pi = policy model
    - pi_ref = reference model
    - y_w = chosen response
    - y_l = rejected response
    
    Args:
        policy_chosen_logps: Policy log probs for chosen
        policy_rejected_logps: Policy log probs for rejected
        reference_chosen_logps: Reference log probs for chosen
        reference_rejected_logps: Reference log probs for rejected
        beta: Temperature parameter
        label_smoothing: Label smoothing factor
        
    Returns:
        (loss, metrics_dict)
    """
    # Compute log ratios
    policy_ratio = policy_chosen_logps - policy_rejected_logps
    reference_ratio = reference_chosen_logps - reference_rejected_logps
    
    # DPO implicit reward difference
    logits = beta * (policy_ratio - reference_ratio)
    
    # Binary cross entropy loss
    if label_smoothing > 0:
        # Soft labels
        losses = (
            -label_smoothing * F.logsigmoid(-logits) -
            (1 - label_smoothing) * F.logsigmoid(logits)
        )
    else:
        losses = -F.logsigmoid(logits)
    
    loss = losses.mean()
    
    # Compute metrics
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    
    metrics = {
        "loss": loss.item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        "accuracy": (logits > 0).float().mean().item(),
    }
    
    return loss, metrics


def reference_free_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    beta: float = 0.1
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Reference-free DPO loss (simpler, no reference model needed).
    
    Just maximizes the margin between chosen and rejected.
    """
    logits = beta * (policy_chosen_logps - policy_rejected_logps)
    loss = -F.logsigmoid(logits).mean()
    
    metrics = {
        "loss": loss.item(),
        "logp_chosen": policy_chosen_logps.mean().item(),
        "logp_rejected": policy_rejected_logps.mean().item(),
        "margin": (policy_chosen_logps - policy_rejected_logps).mean().item(),
        "accuracy": (logits > 0).float().mean().item(),
    }
    
    return loss, metrics


# =============================================================================
# DPO Trainer
# =============================================================================

class DPOTrainer:
    """
    Direct Preference Optimization trainer.
    
    Aligns language models with human preferences without
    explicit reward modeling or RL.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        ref_model: Optional[nn.Module] = None,
        config: Optional[DPOConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize DPO trainer.
        
        Args:
            model: The policy model to train
            tokenizer: Tokenizer for encoding text
            ref_model: Reference model (frozen copy of initial model)
            config: Training configuration
            device: Device for training
        """
        self.config = config or DPOConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        
        # Reference model (frozen)
        if ref_model is not None:
            self.ref_model = ref_model.to(self.device)
        elif not self.config.use_reference_free:
            # Create a copy of the model as reference
            logger.info("Creating reference model (copy of initial model)")
            self.ref_model = copy.deepcopy(model).to(self.device)
        else:
            self.ref_model = None
        
        # Freeze reference model
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Stats
        self.global_step = 0
        self.training_history: list[dict[str, float]] = []
    
    def train(
        self,
        train_data: list[dict[str, str]],
        eval_data: Optional[list[dict[str, str]]] = None,
        epochs: Optional[int] = None,
        callback: Optional[Callable[[int, dict[str, float]], None]] = None
    ) -> dict[str, Any]:
        """
        Train the model with DPO.
        
        Args:
            train_data: Training preference pairs
            eval_data: Evaluation preference pairs
            epochs: Number of epochs (overrides config)
            callback: Called after each step with (step, metrics)
            
        Returns:
            Training results dict
        """
        epochs = epochs or self.config.epochs
        
        # Create dataset and dataloader
        train_dataset = PreferenceDataset(
            train_data,
            self.tokenizer,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        logger.info(f"Starting DPO training:")
        logger.info(f"  Samples: {len(train_data)}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Beta: {self.config.beta}")
        logger.info(f"  Reference-free: {self.config.use_reference_free}")
        
        self.model.train()
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            epoch_metrics: list[dict[str, float]] = []
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Compute policy log probs
                policy_chosen_logps = compute_log_probs(
                    self.model,
                    batch["chosen_ids"],
                    batch["chosen_mask"],
                    batch["chosen_ids"]
                )
                
                policy_rejected_logps = compute_log_probs(
                    self.model,
                    batch["rejected_ids"],
                    batch["rejected_mask"],
                    batch["rejected_ids"]
                )
                
                # Compute loss
                if self.config.use_reference_free or self.ref_model is None:
                    loss, metrics = reference_free_dpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        beta=self.config.beta
                    )
                else:
                    # Compute reference log probs
                    with torch.no_grad():
                        ref_chosen_logps = compute_log_probs(
                            self.ref_model,
                            batch["chosen_ids"],
                            batch["chosen_mask"],
                            batch["chosen_ids"]
                        )
                        ref_rejected_logps = compute_log_probs(
                            self.ref_model,
                            batch["rejected_ids"],
                            batch["rejected_mask"],
                            batch["rejected_ids"]
                        )
                    
                    loss, metrics = dpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        ref_chosen_logps,
                        ref_rejected_logps,
                        beta=self.config.beta,
                        label_smoothing=self.config.label_smoothing
                    )
                
                # Backward pass
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                epoch_metrics.append(metrics)
                self.training_history.append(metrics)
                
                # Logging
                if (batch_idx + 1) % self.config.log_every == 0:
                    avg_metrics = self._average_metrics(epoch_metrics[-self.config.log_every:])
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} | Step {batch_idx+1}/{len(train_loader)} | "
                        f"Loss: {avg_metrics['loss']:.4f} | Acc: {avg_metrics['accuracy']:.2%}"
                    )
                
                # Callback
                if callback:
                    callback(self.global_step, metrics)
            
            # Epoch summary
            epoch_avg = self._average_metrics(epoch_metrics)
            logger.info(f"Epoch {epoch+1} complete | "
                       f"Avg Loss: {epoch_avg['loss']:.4f} | "
                       f"Avg Accuracy: {epoch_avg['accuracy']:.2%}")
            
            if epoch_avg['accuracy'] > best_accuracy:
                best_accuracy = epoch_avg['accuracy']
        
        return {
            "final_loss": epoch_avg['loss'],
            "final_accuracy": epoch_avg['accuracy'],
            "best_accuracy": best_accuracy,
            "total_steps": self.global_step,
            "history": self.training_history
        }
    
    def _average_metrics(self, metrics_list: list[dict[str, float]]) -> dict[str, float]:
        """Average a list of metrics dicts."""
        if not metrics_list:
            return {}
        
        avg = {}
        for key in metrics_list[0].keys():
            avg[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
        return avg
    
    def save(self, path: Union[str, Path]):
        """Save the trained model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "global_step": self.global_step,
            "history": self.training_history
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path]):
        """Load a saved model."""
        # Note: weights_only=False needed for optimizer state, only load trusted checkpoints
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.training_history = checkpoint.get("history", [])
        
        logger.info(f"Model loaded from {path}")


# =============================================================================
# Data Utilities
# =============================================================================

def load_preference_data(path: Union[str, Path]) -> list[dict[str, str]]:
    """
    Load preference data from file.
    
    Supports JSON and JSONL formats.
    
    JSON format:
    [
        {"prompt": "...", "chosen": "...", "rejected": "..."},
        ...
    ]
    
    JSONL format:
    {"prompt": "...", "chosen": "...", "rejected": "..."}
    {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    path = Path(path)
    
    if path.suffix == ".jsonl":
        data = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    
    # Validate
    for item in data:
        assert "prompt" in item, "Missing 'prompt' field"
        assert "chosen" in item, "Missing 'chosen' field"
        assert "rejected" in item, "Missing 'rejected' field"
    
    return data


def create_preference_pairs_from_ratings(
    prompts: list[str],
    responses: list[list[str]],
    ratings: list[list[float]]
) -> list[dict[str, str]]:
    """
    Create preference pairs from rated responses.
    
    For each prompt, creates pairs where the higher-rated
    response is 'chosen' and lower-rated is 'rejected'.
    
    Args:
        prompts: List of prompts
        responses: List of response lists (one per prompt)
        ratings: List of rating lists (one per prompt)
        
    Returns:
        List of preference dicts
    """
    preferences = []
    
    for prompt, resps, rates in zip(prompts, responses, ratings):
        # Sort by rating
        sorted_pairs = sorted(zip(rates, resps), reverse=True)
        
        # Create pairs (best vs each other)
        best_rating, best_response = sorted_pairs[0]
        
        for rating, response in sorted_pairs[1:]:
            if rating < best_rating:  # Only if clearly worse
                preferences.append({
                    "prompt": prompt,
                    "chosen": best_response,
                    "rejected": response
                })
    
    return preferences


# =============================================================================
# Quick Training Function
# =============================================================================

def train_dpo(
    model_path: Optional[str] = None,
    data_path: str = "data/preferences.json",
    output_path: str = "models/dpo_model.pth",
    epochs: int = 3,
    beta: float = 0.1,
    **kwargs
) -> dict[str, Any]:
    """
    Quick function to run DPO training.
    
    Args:
        model_path: Path to base model
        data_path: Path to preference data
        output_path: Where to save trained model
        epochs: Training epochs
        beta: DPO beta parameter
        **kwargs: Additional config parameters
        
    Returns:
        Training results
    """
    from .inference import EnigmaEngine
    from .tokenizer import get_tokenizer

    # Load model
    engine = EnigmaEngine(model_path=model_path)
    model = engine.model
    tokenizer = engine.tokenizer
    
    # Load data
    data = load_preference_data(data_path)
    logger.info(f"Loaded {len(data)} preference pairs")
    
    # Create config
    config = DPOConfig(epochs=epochs, beta=beta, **kwargs)
    
    # Train
    trainer = DPOTrainer(model, tokenizer, config=config)
    results = trainer.train(data)
    
    # Save
    trainer.save(output_path)
    
    return results


__all__ = [
    'DPOConfig',
    'DPOTrainer',
    'PreferenceDataset',
    'dpo_loss',
    'reference_free_dpo_loss',
    'load_preference_data',
    'create_preference_pairs_from_ratings',
    'train_dpo',
]
