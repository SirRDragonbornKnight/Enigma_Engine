"""
DPO Training

Direct Preference Optimization for language model alignment.
No reward model needed - trains directly on preferences.

FILE: enigma_engine/core/dpo.py
TYPE: Core/Training
MAIN CLASSES: DPOTrainer, DPOLoss, DPOConfig
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DPOVariant(Enum):
    """DPO loss variants."""
    STANDARD = "standard"  # Original DPO
    IPO = "ipo"  # Identity Preference Optimization
    HINGE = "hinge"  # Hinge loss variant
    SIGMOID = "sigmoid"  # Sigmoid variant
    ROBUST = "robust"  # Conservative DPO


@dataclass
class DPOConfig:
    """DPO training configuration."""
    # Loss
    beta: float = 0.1  # KL penalty coefficient
    loss_variant: DPOVariant = DPOVariant.STANDARD
    label_smoothing: float = 0.0
    
    # Robust DPO
    max_length_diff: int = 0  # Length normalization
    length_penalty: float = 0.0
    
    # IPO
    ipo_tau: float = 0.05  # IPO temperature
    
    # Training
    batch_size: int = 4
    learning_rate: float = 1e-6
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Reference model
    ref_model_update_steps: int = 0  # 0 = frozen
    
    # Logging
    log_every: int = 10


@dataclass
class PreferenceExample:
    """A preference training example."""
    prompt: str
    chosen: str
    rejected: str
    
    # Optional metadata
    margin: float = 0.0  # Expected reward difference
    prompt_tokens: int = 0
    chosen_tokens: int = 0
    rejected_tokens: int = 0


if HAS_TORCH:
    
    class PreferenceDataset(Dataset):
        """Dataset for preference pairs."""
        
        def __init__(
            self,
            examples: list[PreferenceExample],
            tokenizer: Any,
            max_length: int = 512
        ):
            self.examples = examples
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self) -> int:
            return len(self.examples)
        
        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            ex = self.examples[idx]
            
            # Tokenize prompt + chosen
            chosen_text = ex.prompt + ex.chosen
            chosen_enc = self.tokenizer(
                chosen_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Tokenize prompt + rejected
            rejected_text = ex.prompt + ex.rejected
            rejected_enc = self.tokenizer(
                rejected_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Prompt tokens for masking
            prompt_enc = self.tokenizer(
                ex.prompt,
                return_tensors='pt'
            )
            prompt_len = prompt_enc['input_ids'].shape[1]
            
            return {
                'chosen_input_ids': chosen_enc['input_ids'].squeeze(0),
                'chosen_attention_mask': chosen_enc['attention_mask'].squeeze(0),
                'rejected_input_ids': rejected_enc['input_ids'].squeeze(0),
                'rejected_attention_mask': rejected_enc['attention_mask'].squeeze(0),
                'prompt_length': torch.tensor(prompt_len),
                'margin': torch.tensor(ex.margin)
            }
    
    
    class DPOLoss(nn.Module):
        """
        Direct Preference Optimization loss.
        
        Maximizes the margin between chosen and rejected responses
        relative to a reference model.
        """
        
        def __init__(self, config: DPOConfig):
            super().__init__()
            self.config = config
            self.beta = config.beta
        
        def forward(
            self,
            policy_chosen_logps: torch.Tensor,
            policy_rejected_logps: torch.Tensor,
            ref_chosen_logps: torch.Tensor,
            ref_rejected_logps: torch.Tensor,
            margin: torch.Tensor = None
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            """
            Compute DPO loss.
            
            Args:
                policy_chosen_logps: Log probs of chosen under policy
                policy_rejected_logps: Log probs of rejected under policy
                ref_chosen_logps: Log probs of chosen under reference
                ref_rejected_logps: Log probs of rejected under reference
                margin: Optional margin for each pair
            
            Returns:
                (loss, metrics_dict)
            """
            # Compute log ratios
            chosen_logratios = policy_chosen_logps - ref_chosen_logps
            rejected_logratios = policy_rejected_logps - ref_rejected_logps
            
            # Policy advantage
            logits = self.beta * (chosen_logratios - rejected_logratios)
            
            # Apply margin if provided
            if margin is not None and margin.any():
                logits = logits - margin
            
            # Compute loss based on variant
            if self.config.loss_variant == DPOVariant.STANDARD:
                loss = self._standard_loss(logits)
            elif self.config.loss_variant == DPOVariant.IPO:
                loss = self._ipo_loss(logits)
            elif self.config.loss_variant == DPOVariant.HINGE:
                loss = self._hinge_loss(logits)
            elif self.config.loss_variant == DPOVariant.SIGMOID:
                loss = self._sigmoid_loss(logits)
            elif self.config.loss_variant == DPOVariant.ROBUST:
                loss = self._robust_loss(logits, chosen_logratios, rejected_logratios)
            else:
                loss = self._standard_loss(logits)
            
            # Label smoothing
            if self.config.label_smoothing > 0:
                smooth_term = -F.logsigmoid(-logits).mean()
                loss = (1 - self.config.label_smoothing) * loss + \
                       self.config.label_smoothing * smooth_term
            
            # Metrics
            metrics = {
                'chosen_logratios': chosen_logratios.mean(),
                'rejected_logratios': rejected_logratios.mean(),
                'logits': logits.mean(),
                'accuracy': (logits > 0).float().mean()
            }
            
            return loss, metrics
        
        def _standard_loss(self, logits: torch.Tensor) -> torch.Tensor:
            """Standard DPO loss: -log(sigmoid(beta * (r_c - r_r)))"""
            return -F.logsigmoid(logits).mean()
        
        def _ipo_loss(self, logits: torch.Tensor) -> torch.Tensor:
            """IPO loss: (logits - 1/(2*tau))^2"""
            tau = self.config.ipo_tau
            return ((logits - 1.0 / (2 * tau)) ** 2).mean()
        
        def _hinge_loss(self, logits: torch.Tensor) -> torch.Tensor:
            """Hinge loss: max(0, 1 - logits)"""
            return F.relu(1.0 - logits).mean()
        
        def _sigmoid_loss(self, logits: torch.Tensor) -> torch.Tensor:
            """Sigmoid loss with implicit margin."""
            return (1 - torch.sigmoid(logits)).mean()
        
        def _robust_loss(
            self,
            logits: torch.Tensor,
            chosen_logratios: torch.Tensor,
            rejected_logratios: torch.Tensor
        ) -> torch.Tensor:
            """Conservative DPO with length penalty."""
            base_loss = -F.logsigmoid(logits).mean()
            
            # Penalize large deviations from reference
            deviation_penalty = (
                chosen_logratios.abs().mean() +
                rejected_logratios.abs().mean()
            ) * 0.01
            
            return base_loss + deviation_penalty
    
    
    class DPOTrainer:
        """
        DPO Trainer for preference learning.
        
        Trains a policy model to prefer chosen over rejected
        responses without an explicit reward model.
        """
        
        def __init__(
            self,
            policy_model: nn.Module,
            ref_model: nn.Module,
            tokenizer: Any,
            config: DPOConfig = None
        ):
            self.policy = policy_model
            self.ref_model = ref_model
            self.tokenizer = tokenizer
            self.config = config or DPOConfig()
            
            # Freeze reference model
            for param in self.ref_model.parameters():
                param.requires_grad = False
            
            # Loss function
            self.loss_fn = DPOLoss(self.config)
            
            # Optimizer
            self.optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=self.config.learning_rate
            )
            
            # Scheduler
            self.scheduler = None
            
            # Metrics
            self.global_step = 0
            self.metrics_history: list[dict[str, float]] = []
        
        def get_logprobs(
            self,
            model: nn.Module,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            prompt_length: torch.Tensor
        ) -> torch.Tensor:
            """
            Get log probabilities of tokens after prompt.
            
            Args:
                model: Model to use
                input_ids: Input token IDs
                attention_mask: Attention mask
                prompt_length: Length of prompt (to mask)
            
            Returns:
                Sum of log probabilities for response tokens
            """
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:]
            
            # Get log probs
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs, -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Mask prompt tokens and padding
            batch_size = input_ids.shape[0]
            response_mask = torch.zeros_like(shift_mask)
            for i in range(batch_size):
                start = prompt_length[i].item() - 1
                end = shift_mask[i].sum().item()
                if start < end:
                    response_mask[i, start:end] = 1
            
            # Sum log probs for response
            masked_log_probs = token_log_probs * response_mask
            return masked_log_probs.sum(dim=-1)
        
        def train_step(
            self,
            batch: dict[str, torch.Tensor]
        ) -> dict[str, float]:
            """
            Perform single training step.
            
            Args:
                batch: Batch dictionary from dataloader
            
            Returns:
                Metrics dictionary
            """
            self.policy.train()
            
            device = next(self.policy.parameters()).device
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get log probs from policy
            policy_chosen_logps = self.get_logprobs(
                self.policy,
                batch['chosen_input_ids'],
                batch['chosen_attention_mask'],
                batch['prompt_length']
            )
            
            policy_rejected_logps = self.get_logprobs(
                self.policy,
                batch['rejected_input_ids'],
                batch['rejected_attention_mask'],
                batch['prompt_length']
            )
            
            # Get log probs from reference (no grad)
            with torch.no_grad():
                ref_chosen_logps = self.get_logprobs(
                    self.ref_model,
                    batch['chosen_input_ids'],
                    batch['chosen_attention_mask'],
                    batch['prompt_length']
                )
                
                ref_rejected_logps = self.get_logprobs(
                    self.ref_model,
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask'],
                    batch['prompt_length']
                )
            
            # Compute loss
            loss, metrics = self.loss_fn(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                batch.get('margin')
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm
                )
            
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            self.global_step += 1
            
            # Convert metrics
            return {
                'loss': loss.item(),
                **{k: v.item() for k, v in metrics.items()}
            }
        
        def train(
            self,
            examples: list[PreferenceExample],
            epochs: int = 1,
            eval_examples: list[PreferenceExample] = None
        ) -> dict[str, float]:
            """
            Train on preference examples.
            
            Args:
                examples: Training examples
                epochs: Number of epochs
                eval_examples: Validation examples
            
            Returns:
                Final metrics
            """
            # Create dataset and dataloader
            dataset = PreferenceDataset(
                examples, self.tokenizer
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            # Warmup scheduler
            total_steps = len(dataloader) * epochs
            if self.config.warmup_steps > 0:
                from torch.optim.lr_scheduler import LinearLR
                self.scheduler = LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=self.config.warmup_steps
                )
            
            logger.info(f"Training DPO for {epochs} epochs, {total_steps} steps")
            
            for epoch in range(epochs):
                epoch_metrics = []
                
                for batch in dataloader:
                    metrics = self.train_step(batch)
                    epoch_metrics.append(metrics)
                    
                    if self.global_step % self.config.log_every == 0:
                        avg = {
                            k: sum(m[k] for m in epoch_metrics[-self.config.log_every:]) /
                               min(len(epoch_metrics), self.config.log_every)
                            for k in metrics
                        }
                        logger.info(
                            f"Step {self.global_step}: "
                            f"loss={avg['loss']:.4f}, "
                            f"acc={avg['accuracy']:.3f}"
                        )
                
                # Epoch summary
                epoch_avg = {
                    k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
                    for k in epoch_metrics[0]
                }
                logger.info(f"Epoch {epoch + 1}/{epochs}: {epoch_avg}")
                self.metrics_history.append(epoch_avg)
                
                # Evaluation
                if eval_examples:
                    eval_metrics = self.evaluate(eval_examples)
                    logger.info(f"Eval: {eval_metrics}")
            
            return self.metrics_history[-1] if self.metrics_history else {}
        
        def evaluate(
            self,
            examples: list[PreferenceExample]
        ) -> dict[str, float]:
            """Evaluate on preference examples."""
            self.policy.eval()
            
            dataset = PreferenceDataset(examples, self.tokenizer)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size
            )
            
            all_metrics = []
            device = next(self.policy.parameters()).device
            
            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    policy_chosen = self.get_logprobs(
                        self.policy,
                        batch['chosen_input_ids'],
                        batch['chosen_attention_mask'],
                        batch['prompt_length']
                    )
                    
                    policy_rejected = self.get_logprobs(
                        self.policy,
                        batch['rejected_input_ids'],
                        batch['rejected_attention_mask'],
                        batch['prompt_length']
                    )
                    
                    ref_chosen = self.get_logprobs(
                        self.ref_model,
                        batch['chosen_input_ids'],
                        batch['chosen_attention_mask'],
                        batch['prompt_length']
                    )
                    
                    ref_rejected = self.get_logprobs(
                        self.ref_model,
                        batch['rejected_input_ids'],
                        batch['rejected_attention_mask'],
                        batch['prompt_length']
                    )
                    
                    loss, metrics = self.loss_fn(
                        policy_chosen, policy_rejected,
                        ref_chosen, ref_rejected
                    )
                    
                    all_metrics.append({
                        'loss': loss.item(),
                        **{k: v.item() for k, v in metrics.items()}
                    })
            
            return {
                k: sum(m[k] for m in all_metrics) / len(all_metrics)
                for k in all_metrics[0]
            }
        
        def save(self, path: str):
            """Save model and training state."""
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'global_step': self.global_step,
                'config': self.config,
                'metrics_history': self.metrics_history
            }, path)
        
        def load(self, path: str):
            """Load model and training state."""
            checkpoint = torch.load(path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.global_step = checkpoint['global_step']
            self.metrics_history = checkpoint['metrics_history']

else:
    # Stubs when torch not available
    class PreferenceDataset:
        pass
    
    class DPOLoss:
        pass
    
    class DPOTrainer:
        pass
