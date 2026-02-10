"""
DPO and RLHF Training

Direct Preference Optimization and Reinforcement Learning from Human Feedback.
Train models to align with human preferences.

FILE: enigma_engine/core/preference_training.py
TYPE: Core/Training
MAIN CLASSES: DPOTrainer, RLHFTrainer, RewardModel
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class PreferenceData:
    """A single preference comparison."""
    prompt: str
    chosen: str      # Preferred response
    rejected: str    # Non-preferred response
    
    # Optional metadata
    chosen_score: float = 1.0
    rejected_score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    beta: float = 0.1  # Temperature parameter
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # "sigmoid" or "hinge"
    reference_free: bool = False
    
    # Training params
    learning_rate: float = 5e-7
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    num_epochs: int = 1
    warmup_ratio: float = 0.1
    
    # Regularization
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0


@dataclass  
class RLHFConfig:
    """Configuration for RLHF training."""
    # PPO params
    ppo_epochs: int = 4
    mini_batch_size: int = 4
    
    # Advantage estimation
    gamma: float = 1.0
    lam: float = 0.95  # GAE lambda
    
    # Clipping
    clip_range: float = 0.2
    clip_range_vf: float = None  # Value function clip, None = use clip_range
    
    # Coefficients
    vf_coef: float = 0.1
    ent_coef: float = 0.01
    
    # KL control
    target_kl: float = 0.1
    kl_coef: float = 0.2
    
    # Training
    learning_rate: float = 1e-6
    batch_size: int = 16
    max_length: int = 512
    num_epochs: int = 100
    
    # Reward model
    reward_model_path: str = ""
    normalize_reward: bool = True


class PreferenceDataset(Dataset):
    """Dataset for preference training."""
    
    def __init__(
        self,
        data: list[PreferenceData],
        tokenizer: Any,
        max_length: int = 512
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize chosen (prompt + chosen response)
        chosen_text = item.prompt + item.chosen
        rejected_text = item.prompt + item.rejected
        
        chosen_tokens = self.tokenizer.encode(
            chosen_text,
            max_length=self.max_length,
            truncation=True
        )
        
        rejected_tokens = self.tokenizer.encode(
            rejected_text,
            max_length=self.max_length,
            truncation=True
        )
        
        # Get prompt length for masking
        prompt_tokens = self.tokenizer.encode(item.prompt)
        prompt_len = len(prompt_tokens)
        
        return {
            "chosen_ids": torch.tensor(chosen_tokens),
            "rejected_ids": torch.tensor(rejected_tokens),
            "prompt_len": prompt_len,
            "chosen_score": item.chosen_score,
            "rejected_score": item.rejected_score
        }


def collate_preference(batch):
    """Collate function for preference dataset."""
    # Pad sequences
    max_chosen_len = max(len(b["chosen_ids"]) for b in batch)
    max_rejected_len = max(len(b["rejected_ids"]) for b in batch)
    max_len = max(max_chosen_len, max_rejected_len)
    
    chosen_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    rejected_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    chosen_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    rejected_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    prompt_lens = []
    
    for i, b in enumerate(batch):
        c_len = len(b["chosen_ids"])
        r_len = len(b["rejected_ids"])
        
        chosen_ids[i, :c_len] = b["chosen_ids"]
        rejected_ids[i, :r_len] = b["rejected_ids"]
        chosen_mask[i, :c_len] = 1
        rejected_mask[i, :r_len] = 1
        prompt_lens.append(b["prompt_len"])
    
    return {
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "chosen_mask": chosen_mask,
        "rejected_mask": rejected_mask,
        "prompt_lens": torch.tensor(prompt_lens)
    }


class DPOTrainer:
    """
    Direct Preference Optimization trainer.
    
    Trains model directly on preference data without reward model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        tokenizer: Any,
        config: DPOConfig = None
    ):
        """
        Initialize DPO trainer.
        
        Args:
            model: Policy model to train
            ref_model: Reference model (frozen copy of initial policy)
            tokenizer: Tokenizer for encoding
            config: DPO configuration
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config or DPOConfig()
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self._step = 0
        self._epoch = 0
    
    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lens: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities for responses (excluding prompt)."""
        with torch.set_grad_enabled(model == self.model):
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Shift for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask out prompt tokens
        response_mask = torch.zeros_like(shift_labels, dtype=torch.float)
        for i, plen in enumerate(prompt_lens):
            response_mask[i, plen:] = 1.0
        
        # Also mask padding
        response_mask = response_mask * attention_mask[:, 1:]
        
        # Sum log probs for response
        return (token_log_probs * response_mask).sum(dim=-1)
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute DPO loss.
        
        L_DPO = -log(sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x)))))
        """
        # Log ratios
        chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratio = policy_rejected_logps - ref_rejected_logps
        
        # Implicit reward difference
        logits = self.config.beta * (chosen_log_ratio - rejected_log_ratio)
        
        # Loss
        if self.config.loss_type == "sigmoid":
            losses = -F.logsigmoid(logits)
        elif self.config.loss_type == "hinge":
            losses = F.relu(1 - logits)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Label smoothing
        if self.config.label_smoothing > 0:
            smooth_term = -F.logsigmoid(-logits)
            losses = (1 - self.config.label_smoothing) * losses + \
                     self.config.label_smoothing * smooth_term
        
        loss = losses.mean()
        
        # Metrics
        metrics = {
            "loss": loss.item(),
            "chosen_reward": (self.config.beta * chosen_log_ratio).mean().item(),
            "rejected_reward": (self.config.beta * rejected_log_ratio).mean().item(),
            "accuracy": (logits > 0).float().mean().item(),
            "reward_margin": (chosen_log_ratio - rejected_log_ratio).mean().item()
        }
        
        return loss, metrics
    
    def train_step(self, batch: dict) -> dict:
        """Perform single training step."""
        self.model.train()
        
        chosen_ids = batch["chosen_ids"]
        rejected_ids = batch["rejected_ids"]
        chosen_mask = batch["chosen_mask"]
        rejected_mask = batch["rejected_mask"]
        prompt_lens = batch["prompt_lens"]
        
        device = next(self.model.parameters()).device
        chosen_ids = chosen_ids.to(device)
        rejected_ids = rejected_ids.to(device)
        chosen_mask = chosen_mask.to(device)
        rejected_mask = rejected_mask.to(device)
        prompt_lens = prompt_lens.to(device)
        
        # Compute log probs for policy
        policy_chosen_logps = self.compute_log_probs(
            self.model, chosen_ids, chosen_mask, prompt_lens
        )
        policy_rejected_logps = self.compute_log_probs(
            self.model, rejected_ids, rejected_mask, prompt_lens
        )
        
        # Compute log probs for reference (no grad)
        with torch.no_grad():
            ref_chosen_logps = self.compute_log_probs(
                self.ref_model, chosen_ids, chosen_mask, prompt_lens
            )
            ref_rejected_logps = self.compute_log_probs(
                self.ref_model, rejected_ids, rejected_mask, prompt_lens
            )
        
        # DPO loss
        loss, metrics = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps
        )
        
        # Backward
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()
        
        self._step += 1
        
        # Update weights
        if self._step % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return metrics
    
    def train(
        self,
        train_data: list[PreferenceData],
        eval_data: Optional[list[PreferenceData]] = None,
        callback: Callable[[dict], None] = None
    ):
        """Train on preference data."""
        dataset = PreferenceDataset(
            train_data, self.tokenizer, self.config.max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_preference
        )
        
        total_steps = len(dataloader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        logger.info(f"Starting DPO training: {total_steps} steps, {warmup_steps} warmup")
        
        for epoch in range(self.config.num_epochs):
            self._epoch = epoch
            epoch_metrics = []
            
            for batch in dataloader:
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                
                if self._step % 10 == 0:
                    avg_metrics = {
                        k: sum(m[k] for m in epoch_metrics[-10:]) / min(10, len(epoch_metrics))
                        for k in metrics.keys()
                    }
                    logger.info(f"Step {self._step}: {avg_metrics}")
                    
                    if callback:
                        callback({"step": self._step, **avg_metrics})
                
                scheduler.step()
            
            # Epoch summary
            avg_metrics = {
                k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
                for k in epoch_metrics[0].keys()
            }
            logger.info(f"Epoch {epoch + 1}: {avg_metrics}")
        
        logger.info("DPO training complete")


class RewardModel(nn.Module):
    """
    Reward model for RLHF.
    
    Predicts scalar reward for (prompt, response) pairs.
    """
    
    def __init__(self, base_model: nn.Module, hidden_size: int):
        super().__init__()
        self.base = base_model
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute reward for input."""
        outputs = self.base(input_ids, attention_mask=attention_mask)
        
        # Get last token's hidden state
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs
        
        # Find last non-padding token
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1
            bs = hidden.shape[0]
            last_hidden = hidden[torch.arange(bs, device=hidden.device), last_idx]
        else:
            last_hidden = hidden[:, -1]
        
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)


class RewardModelTrainer:
    """Train reward model on preference data."""
    
    def __init__(
        self,
        model: RewardModel,
        tokenizer: Any,
        learning_rate: float = 1e-5,
        batch_size: int = 8
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )
    
    def train_step(self, batch: dict) -> dict:
        """Train step for reward model."""
        self.model.train()
        
        device = next(self.model.parameters()).device
        
        chosen_ids = batch["chosen_ids"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)
        chosen_mask = batch["chosen_mask"].to(device)
        rejected_mask = batch["rejected_mask"].to(device)
        
        # Compute rewards
        chosen_reward = self.model(chosen_ids, chosen_mask)
        rejected_reward = self.model(rejected_ids, rejected_mask)
        
        # Bradley-Terry loss
        loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        accuracy = (chosen_reward > rejected_reward).float().mean()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "chosen_reward": chosen_reward.mean().item(),
            "rejected_reward": rejected_reward.mean().item()
        }
    
    def train(self, data: list[PreferenceData], num_epochs: int = 3):
        """Train reward model."""
        dataset = PreferenceDataset(data, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_preference
        )
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_acc = 0
            n_batches = 0
            
            for batch in dataloader:
                metrics = self.train_step(batch)
                epoch_loss += metrics["loss"]
                epoch_acc += metrics["accuracy"]
                n_batches += 1
            
            logger.info(f"Reward Model Epoch {epoch + 1}: "
                        f"Loss={epoch_loss/n_batches:.4f}, "
                        f"Acc={epoch_acc/n_batches:.4f}")


class RLHFTrainer:
    """
    RLHF trainer using PPO algorithm.
    
    Trains policy to maximize reward while staying close to reference.
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        reward_model: RewardModel,
        tokenizer: Any,
        config: RLHFConfig = None
    ):
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config or RLHFConfig()
        
        # Freeze reference and reward models
        for param in self.ref_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate
        )
        
        self._running_reward_mean = 0.0
        self._running_reward_std = 1.0
    
    def generate_responses(
        self,
        prompts: list[str],
        max_new_tokens: int = 100
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate responses from policy."""
        self.model.eval()
        
        all_ids = []
        all_masks = []
        all_log_probs = []
        
        device = next(self.model.parameters()).device
        
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([input_ids], device=device)
            
            generated_ids = []
            log_probs = []
            
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = self.model(input_ids)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    next_logits = logits[:, -1, :]
                    
                    # Sample
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    
                    # Log prob
                    log_prob = F.log_softmax(next_logits, dim=-1)
                    token_log_prob = log_prob.gather(-1, next_token)
                    
                    generated_ids.append(next_token.item())
                    log_probs.append(token_log_prob.item())
                    
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    
                    # Check for EOS
                    if hasattr(self.tokenizer, 'eos_token_id'):
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break
            
            all_ids.append(input_ids.squeeze())
            all_log_probs.append(torch.tensor(log_probs, device=device))
        
        # Pad sequences
        max_len = max(ids.shape[0] for ids in all_ids)
        padded_ids = torch.zeros(len(prompts), max_len, dtype=torch.long, device=device)
        padded_masks = torch.zeros(len(prompts), max_len, device=device)
        
        for i, ids in enumerate(all_ids):
            padded_ids[i, :len(ids)] = ids
            padded_masks[i, :len(ids)] = 1
        
        # Pad log probs
        max_gen_len = max(lp.shape[0] for lp in all_log_probs)
        padded_log_probs = torch.zeros(len(prompts), max_gen_len, device=device)
        for i, lp in enumerate(all_log_probs):
            padded_log_probs[i, :len(lp)] = lp
        
        return padded_ids, padded_masks, padded_log_probs
    
    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lens: torch.Tensor
    ) -> torch.Tensor:
        """Compute rewards for generated responses."""
        device = input_ids.device
        
        # Get reward model score
        with torch.no_grad():
            rewards = self.reward_model(input_ids, attention_mask)
        
        # Normalize rewards
        if self.config.normalize_reward:
            rewards = (rewards - self._running_reward_mean) / (self._running_reward_std + 1e-8)
            
            # Update running stats
            self._running_reward_mean = 0.9 * self._running_reward_mean + 0.1 * rewards.mean().item()
            self._running_reward_std = 0.9 * self._running_reward_std + 0.1 * rewards.std().item()
        
        return rewards
    
    def compute_kl_penalty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        policy_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL penalty from reference model."""
        device = input_ids.device
        
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits if hasattr(ref_outputs, 'logits') else ref_outputs
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            
            # Gather for actual tokens
            ref_token_log_probs = torch.gather(
                ref_log_probs,
                dim=-1,
                index=input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
        
        # KL = policy_log_prob - ref_log_prob
        # We want to penalize deviation from reference
        kl = policy_log_probs - ref_token_log_probs[:, :policy_log_probs.shape[1]]
        
        return kl.sum(dim=-1)
    
    def ppo_step(
        self,
        prompts: list[str],
        prompt_lens: torch.Tensor
    ) -> dict:
        """Single PPO update step."""
        device = next(self.model.parameters()).device
        
        # Generate responses
        input_ids, attention_mask, old_log_probs = self.generate_responses(prompts)
        prompt_lens = prompt_lens.to(device)
        
        # Compute rewards
        rewards = self.compute_rewards(input_ids, attention_mask, prompt_lens)
        
        # Compute KL penalty
        kl_penalty = self.compute_kl_penalty(input_ids, attention_mask, old_log_probs)
        
        # Adjusted rewards
        adjusted_rewards = rewards - self.config.kl_coef * kl_penalty
        
        # PPO updates
        for _ in range(self.config.ppo_epochs):
            self.model.train()
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            
            # Gather current log probs
            new_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            
            # Truncate to generation length
            gen_len = old_log_probs.shape[1]
            new_log_probs = new_log_probs[:, :gen_len]
            
            # Response mask
            response_mask = torch.zeros_like(old_log_probs)
            for i, plen in enumerate(prompt_lens):
                end = min(plen + gen_len, response_mask.shape[1])
                response_mask[i, max(0, plen-1):end-1] = 1
            
            # Sum log probs for ratio
            old_log_prob_sum = (old_log_probs * response_mask).sum(dim=1)
            new_log_prob_sum = (new_log_probs * response_mask).sum(dim=1)
            
            # Ratio
            ratio = torch.exp(new_log_prob_sum - old_log_prob_sum)
            
            # Clipped ratio
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.clip_range,
                1 + self.config.clip_range
            )
            
            # Policy loss
            policy_loss = -torch.min(
                ratio * adjusted_rewards,
                clipped_ratio * adjusted_rewards
            ).mean()
            
            # Entropy bonus
            entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()
            
            # Total loss
            loss = policy_loss - self.config.ent_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "reward": rewards.mean().item(),
            "kl": kl_penalty.mean().item(),
            "entropy": entropy.item()
        }
    
    def train(
        self,
        prompts: list[str],
        callback: Callable[[dict], None] = None
    ):
        """Train with RLHF."""
        # Get prompt lengths
        prompt_lens = torch.tensor([
            len(self.tokenizer.encode(p)) for p in prompts
        ])
        
        logger.info(f"Starting RLHF training with {len(prompts)} prompts")
        
        for epoch in range(self.config.num_epochs):
            # Shuffle prompts
            indices = torch.randperm(len(prompts))
            
            for i in range(0, len(prompts), self.config.batch_size):
                batch_idx = indices[i:i + self.config.batch_size]
                batch_prompts = [prompts[j] for j in batch_idx]
                batch_lens = prompt_lens[batch_idx]
                
                metrics = self.ppo_step(batch_prompts, batch_lens)
                
                logger.info(f"Epoch {epoch + 1}, Batch {i//self.config.batch_size}: {metrics}")
                
                if callback:
                    callback({"epoch": epoch, "batch": i, **metrics})
                
                # Early stopping on high KL
                if metrics["kl"] > self.config.target_kl * 1.5:
                    logger.warning(f"KL divergence too high ({metrics['kl']:.4f}), stopping epoch")
                    break
        
        logger.info("RLHF training complete")


def load_preference_data(path: str) -> list[PreferenceData]:
    """Load preference data from JSON file."""
    with open(path) as f:
        data = json.load(f)
    
    return [
        PreferenceData(
            prompt=item["prompt"],
            chosen=item["chosen"],
            rejected=item["rejected"],
            chosen_score=item.get("chosen_score", 1.0),
            rejected_score=item.get("rejected_score", 0.0)
        )
        for item in data
    ]


__all__ = [
    'DPOTrainer',
    'DPOConfig',
    'RLHFTrainer',
    'RLHFConfig',
    'RewardModel',
    'RewardModelTrainer',
    'PreferenceData',
    'PreferenceDataset',
    'load_preference_data'
]
