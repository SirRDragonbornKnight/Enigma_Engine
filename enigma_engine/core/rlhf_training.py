"""
RLHF Training (Reinforcement Learning from Human Feedback) for enigma_engine

Full RLHF pipeline:
- Reward model training
- PPO (Proximal Policy Optimization)
- Value head for advantage estimation
- KL penalty for stability

Usage:
    from enigma_engine.core.rlhf_training import RLHFTrainer, RewardModel
    
    reward_model = RewardModel(base_model)
    reward_model.train(comparison_data)
    
    trainer = RLHFTrainer(policy_model, reward_model, tokenizer)
    trainer.train(prompts)
"""

import copy
import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    """Configuration for RLHF training."""
    # PPO hyperparameters
    ppo_epochs: int = 4
    num_rollouts: int = 128
    chunk_size: int = 16
    
    # Learning rates
    policy_lr: float = 1e-5
    value_lr: float = 1e-5
    reward_lr: float = 1e-5
    
    # PPO clipping
    clip_ratio: float = 0.2
    clip_value: float = 0.2
    
    # KL penalty
    init_kl_coef: float = 0.1
    target_kl: float = 0.01
    kl_horizon: int = 10000
    
    # GAE (Generalized Advantage Estimation)
    gamma: float = 1.0
    lam: float = 0.95
    
    # Entropy bonus
    entropy_coef: float = 0.01
    
    # Generation
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50
    
    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Normalization
    whiten_rewards: bool = True
    normalize_advantages: bool = True


class ValueHead(nn.Module):
    """Value head for estimating state values."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.summary = nn.Linear(hidden_size, 1)
        
        # Initialize to output small values
        nn.init.normal_(self.summary.weight, std=0.01)
        nn.init.zeros_(self.summary.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            values: (batch, seq_len)
        """
        output = self.dropout(hidden_states)
        return self.summary(output).squeeze(-1)


class RewardModel(nn.Module):
    """
    Reward model trained on human comparisons.
    
    Given a prompt and two completions, learns to predict which
    completion humans preferred.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: Optional[int] = None
    ):
        super().__init__()
        self.base_model = base_model
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Determine hidden size
        if hidden_size is None:
            if hasattr(base_model, 'config'):
                hidden_size = getattr(base_model.config, 'hidden_size', 768)
            else:
                hidden_size = 768
        
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reward for a sequence.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        
        Returns:
            rewards: (batch,) scalar reward per sequence
        """
        # Get hidden states from base model
        with torch.no_grad():
            outputs = self.base_model(input_ids, attention_mask=attention_mask)
            
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden_states = outputs.hidden_states[-1]
            elif hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                # Assume output is logits, need to get from model differently
                hidden_states = outputs
        
        # Use last token's hidden state (or mean pool)
        if attention_mask is not None:
            # Mean pool over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            mean_hidden = sum_hidden / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            # Use last token
            mean_hidden = hidden_states[:, -1, :]
        
        reward = self.reward_head(mean_hidden).squeeze(-1)
        return reward
    
    def compute_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute Bradley-Terry loss for preference learning.
        
        The model should assign higher reward to chosen over rejected.
        """
        chosen_rewards = self.forward(chosen_ids, chosen_mask)
        rejected_rewards = self.forward(rejected_ids, rejected_mask)
        
        # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        # Accuracy
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'chosen_reward_mean': chosen_rewards.mean().item(),
            'rejected_reward_mean': rejected_rewards.mean().item(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean().item()
        }
        
        return loss, metrics


class RewardModelTrainer:
    """Trainer for the reward model."""
    
    def __init__(
        self,
        model: RewardModel,
        tokenizer: Any,
        learning_rate: float = 1e-5
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(
            model.reward_head.parameters(),
            lr=learning_rate
        )
    
    def train(
        self,
        comparisons: list[dict[str, str]],
        epochs: int = 3,
        batch_size: int = 8
    ) -> dict[str, list[float]]:
        """
        Train reward model on comparison data.
        
        Args:
            comparisons: List of {prompt, chosen, rejected}
            epochs: Number of training epochs
            batch_size: Batch size
        
        Returns:
            Training metrics
        """
        device = next(self.model.parameters()).device
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            
            # Shuffle data
            import random
            random.shuffle(comparisons)
            
            for i in range(0, len(comparisons), batch_size):
                batch = comparisons[i:i + batch_size]
                
                # Tokenize chosen and rejected
                chosen_texts = [c['prompt'] + c['chosen'] for c in batch]
                rejected_texts = [c['prompt'] + c['rejected'] for c in batch]
                
                chosen_enc = self._tokenize(chosen_texts, device)
                rejected_enc = self._tokenize(rejected_texts, device)
                
                # Forward and loss
                loss, metrics = self.model.compute_loss(
                    chosen_enc['input_ids'],
                    rejected_enc['input_ids'],
                    chosen_enc.get('attention_mask'),
                    rejected_enc.get('attention_mask')
                )
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += metrics['loss']
                epoch_acc += metrics['accuracy']
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(avg_acc)
            
            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        
        return history
    
    def _tokenize(self, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        """Tokenize texts."""
        if hasattr(self.tokenizer, 'batch_encode_plus'):
            encoded = self.tokenizer.batch_encode_plus(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            return {k: v.to(device) for k, v in encoded.items()}
        else:
            # Simple tokenizer
            token_ids = [self.tokenizer.encode(t)[:512] for t in texts]
            max_len = max(len(t) for t in token_ids)
            
            padded = []
            masks = []
            for t in token_ids:
                pad_len = max_len - len(t)
                padded.append(t + [0] * pad_len)
                masks.append([1] * len(t) + [0] * pad_len)
            
            return {
                'input_ids': torch.tensor(padded, device=device),
                'attention_mask': torch.tensor(masks, device=device)
            }


class PolicyWithValueHead(nn.Module):
    """Policy model with value head for PPO."""
    
    def __init__(self, policy_model: nn.Module):
        super().__init__()
        self.policy = policy_model
        
        # Determine hidden size
        if hasattr(policy_model, 'config'):
            hidden_size = getattr(policy_model.config, 'hidden_size', 768)
        else:
            hidden_size = 768
        
        self.value_head = ValueHead(hidden_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and values.
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            values: (batch, seq_len)
        """
        outputs = self.policy(input_ids, attention_mask=attention_mask)
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Get hidden states for value head
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]
        else:
            # Approximate from logits (not ideal but works)
            hidden_states = logits
        
        values = self.value_head(hidden_states)
        
        return logits, values


class RLHFTrainer:
    """
    RLHF trainer using PPO.
    
    Trains a policy model to maximize reward from the reward model
    while staying close to the reference policy.
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: RewardModel,
        tokenizer: Any,
        config: Optional[RLHFConfig] = None
    ):
        self.config = config or RLHFConfig()
        self.tokenizer = tokenizer
        
        # Policy with value head
        self.policy = PolicyWithValueHead(policy_model)
        
        # Reference model (frozen copy)
        self.ref_policy = copy.deepcopy(policy_model)
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        
        self.reward_model = reward_model
        
        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.policy.parameters(),
            lr=self.config.policy_lr
        )
        self.value_optimizer = torch.optim.AdamW(
            self.policy.value_head.parameters(),
            lr=self.config.value_lr
        )
        
        # KL coefficient (adaptive)
        self.kl_coef = self.config.init_kl_coef
        
        # Stats
        self.stats = {
            'rewards': [],
            'kl': [],
            'policy_loss': [],
            'value_loss': []
        }
    
    @torch.no_grad()
    def generate_rollouts(
        self,
        prompts: list[str],
        device: torch.device
    ) -> dict[str, torch.Tensor]:
        """
        Generate responses and compute rewards.
        
        Returns rollout data for PPO training.
        """
        self.policy.eval()
        
        all_input_ids = []
        all_response_ids = []
        all_rewards = []
        all_ref_logprobs = []
        all_logprobs = []
        all_values = []
        
        for prompt in prompts:
            # Tokenize prompt
            prompt_ids = torch.tensor(
                [self.tokenizer.encode(prompt)],
                device=device
            )
            
            # Generate response
            response_ids, logprobs, values = self._generate_with_logprobs(
                prompt_ids,
                max_new_tokens=self.config.max_new_tokens
            )
            
            # Get reference logprobs
            full_ids = torch.cat([prompt_ids, response_ids], dim=1)
            ref_logprobs = self._get_logprobs(self.ref_policy, full_ids)
            
            # Compute reward
            reward = self.reward_model(full_ids)
            
            all_input_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_rewards.append(reward)
            all_ref_logprobs.append(ref_logprobs)
            all_logprobs.append(logprobs)
            all_values.append(values)
        
        return {
            'input_ids': all_input_ids,
            'response_ids': all_response_ids,
            'rewards': torch.stack(all_rewards),
            'ref_logprobs': all_ref_logprobs,
            'logprobs': all_logprobs,
            'values': all_values
        }
    
    def _generate_with_logprobs(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate tokens and return logprobs and values."""
        generated = []
        logprobs = []
        values = []
        
        current_ids = input_ids
        
        for _ in range(max_new_tokens):
            logits, value = self.policy(current_ids)
            
            next_token_logits = logits[:, -1, :] / self.config.temperature
            
            # Top-k sampling
            if self.config.top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(
                    next_token_logits, self.config.top_k
                )[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            token_logprob = log_probs.gather(-1, next_token).squeeze(-1)
            
            generated.append(next_token)
            logprobs.append(token_logprob)
            values.append(value[:, -1])
            
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Check for EOS
            if hasattr(self.tokenizer, 'eos_token_id'):
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return (
            torch.cat(generated, dim=1),
            torch.stack(logprobs, dim=1),
            torch.stack(values, dim=1)
        )
    
    def _get_logprobs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Get log probabilities from a model."""
        outputs = model(input_ids)
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_log_probs = log_probs.gather(
            -1,
            input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.
        """
        advantages = torch.zeros_like(values)
        last_gae = 0
        
        for t in reversed(range(values.size(1))):
            if t == values.size(1) - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]
            
            # Only last token gets the reward
            if t == values.size(1) - 1:
                reward = rewards
            else:
                reward = 0
            
            delta = reward + self.config.gamma * next_value - values[:, t]
            advantages[:, t] = last_gae = delta + self.config.gamma * self.config.lam * last_gae
        
        returns = advantages + values
        
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def ppo_step(
        self,
        rollouts: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """
        Perform PPO update.
        """
        self.policy.train()
        
        # Compute advantages
        advantages, returns = self.compute_advantages(
            rollouts['rewards'],
            torch.cat([v.squeeze(0) for v in rollouts['values']], dim=0)
        )
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        num_updates = 0
        
        for _ in range(self.config.ppo_epochs):
            for i in range(len(rollouts['input_ids'])):
                input_ids = rollouts['input_ids'][i]
                response_ids = rollouts['response_ids'][i]
                old_logprobs = rollouts['logprobs'][i]
                ref_logprobs = rollouts['ref_logprobs'][i]
                
                full_ids = torch.cat([input_ids, response_ids], dim=1)
                
                # Forward pass
                logits, values = self.policy(full_ids)
                
                # New log probs
                new_logprobs = self._get_logprobs_from_logits(
                    logits,
                    full_ids
                )
                
                # Only response tokens
                response_start = input_ids.size(1)
                new_logprobs = new_logprobs[:, response_start - 1:]
                
                # Ratio for PPO
                ratio = torch.exp(new_logprobs - old_logprobs)
                
                # Clipped surrogate loss
                adv = advantages[i:i+1, :new_logprobs.size(1)]
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_ratio,
                    1 + self.config.clip_ratio
                ) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # KL penalty
                kl = (old_logprobs - new_logprobs).mean()
                policy_loss = policy_loss + self.kl_coef * kl
                
                # Value loss
                new_values = values[:, response_start:]
                old_values = rollouts['values'][i]
                value_loss = F.mse_loss(
                    new_values[:, :old_values.size(1)],
                    returns[i:i+1, :old_values.size(1)]
                )
                
                # Entropy bonus
                entropy = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss - self.config.entropy_coef * entropy
                
                # Backward
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm
                )
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl += kl.item()
                num_updates += 1
        
        # Update KL coefficient
        avg_kl = total_kl / num_updates
        if avg_kl > self.config.target_kl * 1.5:
            self.kl_coef *= 1.5
        elif avg_kl < self.config.target_kl / 1.5:
            self.kl_coef /= 1.5
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'kl': avg_kl,
            'kl_coef': self.kl_coef
        }
    
    def _get_logprobs_from_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Get log probabilities from logits."""
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_log_probs = log_probs.gather(
            -1,
            input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs
    
    def train(
        self,
        prompts: list[str],
        num_iterations: int = 100
    ) -> dict[str, list[float]]:
        """
        Train policy using PPO.
        
        Args:
            prompts: Training prompts
            num_iterations: Number of PPO iterations
        
        Returns:
            Training statistics
        """
        device = next(self.policy.parameters()).device
        
        for iteration in range(num_iterations):
            # Sample prompts
            import random
            batch_prompts = random.sample(
                prompts,
                min(self.config.num_rollouts, len(prompts))
            )
            
            # Generate rollouts
            rollouts = self.generate_rollouts(batch_prompts, device)
            
            # Whiten rewards if enabled
            if self.config.whiten_rewards:
                rewards = rollouts['rewards']
                rollouts['rewards'] = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            # PPO update
            metrics = self.ppo_step(rollouts)
            
            # Log stats
            self.stats['rewards'].append(rollouts['rewards'].mean().item())
            self.stats['kl'].append(metrics['kl'])
            self.stats['policy_loss'].append(metrics['policy_loss'])
            self.stats['value_loss'].append(metrics['value_loss'])
            
            if iteration % 10 == 0:
                logger.info(
                    f"Iteration {iteration}: "
                    f"reward={self.stats['rewards'][-1]:.4f}, "
                    f"kl={metrics['kl']:.4f}, "
                    f"policy_loss={metrics['policy_loss']:.4f}"
                )
        
        return self.stats
