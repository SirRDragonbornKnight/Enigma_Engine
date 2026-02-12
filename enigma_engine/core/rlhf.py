"""
RLHF Training

Reinforcement Learning from Human Feedback.
Implements reward modeling, PPO training, and preference learning.

FILE: enigma_engine/core/rlhf.py
TYPE: Core/Training
MAIN CLASSES: RewardModel, PPOTrainer, RLHFPipeline
"""

import logging
import random
from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    """RLHF training configuration."""
    # Reward model
    reward_hidden_size: int = 768
    reward_num_layers: int = 1
    
    # PPO
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # KL penalty
    kl_coef: float = 0.1
    kl_target: float = 6.0
    kl_horizon: int = 10000
    
    # Training
    batch_size: int = 32
    mini_batch_size: int = 8
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    
    # Generation
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.95
    
    # Experience buffer
    buffer_size: int = 512
    gamma: float = 1.0
    gae_lambda: float = 0.95


@dataclass
class PreferencePair:
    """A preference pair for reward training."""
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float = 1.0
    rejected_score: float = 0.0


@dataclass
class Experience:
    """Experience tuple for PPO."""
    prompt_ids: Any  # torch.Tensor
    response_ids: Any
    old_log_probs: Any
    advantages: Any
    returns: Any
    rewards: float


if HAS_TORCH:
    
    class RewardModel(nn.Module):
        """
        Reward model for RLHF.
        
        Scores text sequences to predict human preference.
        """
        
        def __init__(
            self,
            base_model: nn.Module,
            config: RLHFConfig = None
        ) -> None:
            super().__init__()
            self.config = config or RLHFConfig()
            
            # Use base model for encoding
            self.encoder = base_model
            
            # Freeze encoder (optional)
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            # Reward head
            hidden_size = self.config.reward_hidden_size
            layers = []
            for i in range(self.config.reward_num_layers):
                layers.extend([
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
            layers.append(nn.Linear(hidden_size, 1))
            
            self.reward_head = nn.Sequential(*layers)
        
        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None
        ) -> torch.Tensor:
            """
            Compute reward for input sequence.
            
            Args:
                input_ids: Input token IDs
                attention_mask: Attention mask
            
            Returns:
                Scalar reward per sequence
            """
            # Get encoder output
            outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Get last hidden state
            if hasattr(outputs, 'last_hidden_state'):
                hidden = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                hidden = outputs.hidden_states[-1]
            else:
                hidden = outputs
            
            # Pool to single vector (use last token like GPT)
            pooled = hidden[:, -1, :]
            
            # Compute reward
            reward = self.reward_head(pooled).squeeze(-1)
            
            return reward
        
        def preference_loss(
            self,
            chosen_ids: torch.Tensor,
            rejected_ids: torch.Tensor,
            chosen_mask: torch.Tensor = None,
            rejected_mask: torch.Tensor = None
        ) -> torch.Tensor:
            """
            Compute preference loss for training.
            
            Uses Bradley-Terry model: P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
            """
            chosen_reward = self.forward(chosen_ids, chosen_mask)
            rejected_reward = self.forward(rejected_ids, rejected_mask)
            
            # Pairwise ranking loss
            loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
            
            return loss
    
    
    class RewardTrainer:
        """Train reward model on preference data."""
        
        def __init__(
            self,
            reward_model: RewardModel,
            config: RLHFConfig = None
        ) -> None:
            self.model = reward_model
            self.config = config or RLHFConfig()
            
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
        
        def train(
            self,
            preferences: list[PreferencePair],
            epochs: int = 3,
            tokenizer: Any = None
        ) -> dict[str, float]:
            """
            Train reward model on preference data.
            
            Args:
                preferences: List of preference pairs
                epochs: Training epochs
                tokenizer: Tokenizer for encoding
            
            Returns:
                Training metrics
            """
            self.model.train()
            device = next(self.model.parameters()).device
            
            total_loss = 0.0
            total_accuracy = 0.0
            n_batches = 0
            
            for epoch in range(epochs):
                random.shuffle(preferences)
                
                for i in range(0, len(preferences), self.config.batch_size):
                    batch = preferences[i:i + self.config.batch_size]
                    
                    # Tokenize
                    chosen_texts = [p.prompt + p.chosen for p in batch]
                    rejected_texts = [p.prompt + p.rejected for p in batch]
                    
                    if tokenizer:
                        chosen_enc = tokenizer(
                            chosen_texts,
                            padding=True,
                            truncation=True,
                            return_tensors='pt'
                        )
                        rejected_enc = tokenizer(
                            rejected_texts,
                            padding=True,
                            truncation=True,
                            return_tensors='pt'
                        )
                        
                        chosen_ids = chosen_enc['input_ids'].to(device)
                        chosen_mask = chosen_enc['attention_mask'].to(device)
                        rejected_ids = rejected_enc['input_ids'].to(device)
                        rejected_mask = rejected_enc['attention_mask'].to(device)
                    else:
                        continue
                    
                    # Compute loss
                    loss = self.model.preference_loss(
                        chosen_ids, rejected_ids,
                        chosen_mask, rejected_mask
                    )
                    
                    # Backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    
                    # Metrics
                    with torch.no_grad():
                        chosen_r = self.model(chosen_ids, chosen_mask)
                        rejected_r = self.model(rejected_ids, rejected_mask)
                        accuracy = (chosen_r > rejected_r).float().mean().item()
                    
                    total_loss += loss.item()
                    total_accuracy += accuracy
                    n_batches += 1
            
            return {
                "loss": total_loss / max(n_batches, 1),
                "accuracy": total_accuracy / max(n_batches, 1)
            }
    
    
    class PPOTrainer:
        """
        Proximal Policy Optimization trainer.
        
        Fine-tunes a language model to maximize reward using PPO.
        """
        
        def __init__(
            self,
            policy_model: nn.Module,
            ref_model: nn.Module,
            reward_model: RewardModel,
            config: RLHFConfig = None
        ) -> None:
            self.policy = policy_model
            self.ref_model = ref_model
            self.reward_model = reward_model
            self.config = config or RLHFConfig()
            
            # Freeze reference model
            for param in self.ref_model.parameters():
                param.requires_grad = False
            
            self.optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=self.config.learning_rate
            )
            
            # KL coefficient (adaptive)
            self.kl_coef = self.config.kl_coef
            
            # Experience buffer
            self.experiences: list[Experience] = []
        
        def generate_responses(
            self,
            prompts: list[str],
            tokenizer: Any
        ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            """
            Generate responses from policy model.
            
            Returns:
                List of (prompt_ids, response_ids, log_probs)
            """
            device = next(self.policy.parameters()).device
            results = []
            
            for prompt in prompts:
                # Encode prompt
                prompt_enc = tokenizer(
                    prompt,
                    return_tensors='pt',
                    padding=True
                )
                prompt_ids = prompt_enc['input_ids'].to(device)
                
                # Generate
                with torch.no_grad():
                    generated = self.policy.generate(
                        prompt_ids,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                response_ids = generated.sequences[:, prompt_ids.shape[1]:]
                
                # Get log probs
                log_probs = self._get_log_probs(
                    prompt_ids, response_ids
                )
                
                results.append((prompt_ids, response_ids, log_probs))
            
            return results
        
        def _get_log_probs(
            self,
            prompt_ids: torch.Tensor,
            response_ids: torch.Tensor
        ) -> torch.Tensor:
            """Get log probabilities of response given prompt."""
            full_ids = torch.cat([prompt_ids, response_ids], dim=1)
            
            with torch.no_grad():
                outputs = self.policy(full_ids)
                logits = outputs.logits[:, prompt_ids.shape[1]-1:-1, :]
                
                log_probs = F.log_softmax(logits, dim=-1)
                response_log_probs = torch.gather(
                    log_probs, -1,
                    response_ids.unsqueeze(-1)
                ).squeeze(-1)
                
                return response_log_probs.sum(dim=-1)
        
        def compute_rewards(
            self,
            prompt_ids: torch.Tensor,
            response_ids: torch.Tensor
        ) -> torch.Tensor:
            """Compute rewards using reward model."""
            full_ids = torch.cat([prompt_ids, response_ids], dim=1)
            
            with torch.no_grad():
                rewards = self.reward_model(full_ids)
            
            return rewards
        
        def compute_kl_penalty(
            self,
            prompt_ids: torch.Tensor,
            response_ids: torch.Tensor
        ) -> torch.Tensor:
            """Compute KL divergence from reference model."""
            full_ids = torch.cat([prompt_ids, response_ids], dim=1)
            
            # Policy log probs
            policy_outputs = self.policy(full_ids)
            policy_logits = policy_outputs.logits[:, prompt_ids.shape[1]-1:-1, :]
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            
            # Reference log probs
            with torch.no_grad():
                ref_outputs = self.ref_model(full_ids)
                ref_logits = ref_outputs.logits[:, prompt_ids.shape[1]-1:-1, :]
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            
            # KL divergence
            kl = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(-1)
            return kl.mean()
        
        def ppo_step(
            self,
            experiences: list[Experience]
        ) -> dict[str, float]:
            """
            Perform PPO update step.
            
            Args:
                experiences: List of experience tuples
            
            Returns:
                Training metrics
            """
            self.policy.train()
            
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            n_updates = 0
            
            for _ in range(self.config.ppo_epochs):
                random.shuffle(experiences)
                
                for i in range(0, len(experiences), self.config.mini_batch_size):
                    batch = experiences[i:i + self.config.mini_batch_size]
                    
                    for exp in batch:
                        # Get new log probs
                        new_log_probs = self._get_log_probs(
                            exp.prompt_ids, exp.response_ids
                        )
                        
                        # PPO ratio
                        ratio = (new_log_probs - exp.old_log_probs).exp()
                        
                        # Clipped surrogate objective
                        surr1 = ratio * exp.advantages
                        surr2 = torch.clamp(
                            ratio,
                            1 - self.config.clip_epsilon,
                            1 + self.config.clip_epsilon
                        ) * exp.advantages
                        
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss (if using value function)
                        value_loss = torch.tensor(0.0)
                        
                        # Entropy bonus
                        entropy = -new_log_probs.mean()
                        
                        # Total loss
                        loss = (
                            policy_loss +
                            self.config.value_loss_coef * value_loss -
                            self.config.entropy_coef * entropy
                        )
                        
                        # Backward
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.policy.parameters(),
                            self.config.max_grad_norm
                        )
                        self.optimizer.step()
                        
                        total_loss += loss.item()
                        total_policy_loss += policy_loss.item()
                        total_entropy += entropy.item()
                        n_updates += 1
            
            return {
                "loss": total_loss / max(n_updates, 1),
                "policy_loss": total_policy_loss / max(n_updates, 1),
                "entropy": total_entropy / max(n_updates, 1)
            }
        
        def update_kl_coef(self, kl: float) -> None:
            """Update KL coefficient adaptively."""
            if kl < self.config.kl_target / 1.5:
                self.kl_coef /= 2
            elif kl > self.config.kl_target * 1.5:
                self.kl_coef *= 2
            
            self.kl_coef = max(0.001, min(self.kl_coef, 1.0))
    
    
    class RLHFPipeline:
        """
        Complete RLHF training pipeline.
        
        Orchestrates reward training and PPO fine-tuning.
        """
        
        def __init__(
            self,
            base_model: nn.Module,
            tokenizer: Any,
            config: RLHFConfig = None
        ) -> None:
            self.tokenizer = tokenizer
            self.config = config or RLHFConfig()
            
            # Create model copies
            self.policy = base_model
            self.ref_model = self._copy_model(base_model)
            
            # Reward model
            self.reward_model = RewardModel(base_model, self.config)
            
            # Trainers
            self.reward_trainer = RewardTrainer(self.reward_model, self.config)
            self.ppo_trainer = PPOTrainer(
                self.policy, self.ref_model,
                self.reward_model, self.config
            )
        
        def _copy_model(self, model: nn.Module) -> nn.Module:
            """Create a frozen copy of the model."""
            import copy
            ref = copy.deepcopy(model)
            for param in ref.parameters():
                param.requires_grad = False
            return ref
        
        def train_reward_model(
            self,
            preferences: list[PreferencePair],
            epochs: int = 3
        ) -> dict[str, float]:
            """Train the reward model."""
            logger.info("Training reward model...")
            return self.reward_trainer.train(
                preferences, epochs, self.tokenizer
            )
        
        def train_policy(
            self,
            prompts: list[str],
            steps: int = 1000
        ) -> dict[str, float]:
            """
            Train policy using PPO.
            
            Args:
                prompts: Training prompts
                steps: Number of PPO steps
            
            Returns:
                Training metrics
            """
            logger.info("Training policy with PPO...")
            
            metrics_history = []
            
            for step in range(steps):
                # Sample prompts
                batch_prompts = random.sample(
                    prompts,
                    min(self.config.batch_size, len(prompts))
                )
                
                # Generate responses
                generations = self.ppo_trainer.generate_responses(
                    batch_prompts, self.tokenizer
                )
                
                # Compute rewards and create experiences
                experiences = []
                for prompt_ids, response_ids, log_probs in generations:
                    rewards = self.ppo_trainer.compute_rewards(
                        prompt_ids, response_ids
                    )
                    
                    # KL penalty
                    kl = self.ppo_trainer.compute_kl_penalty(
                        prompt_ids, response_ids
                    )
                    
                    # Adjusted rewards
                    adjusted_rewards = rewards - self.ppo_trainer.kl_coef * kl
                    
                    experiences.append(Experience(
                        prompt_ids=prompt_ids,
                        response_ids=response_ids,
                        old_log_probs=log_probs,
                        advantages=adjusted_rewards,
                        returns=adjusted_rewards,
                        rewards=rewards.item()
                    ))
                
                # PPO update
                metrics = self.ppo_trainer.ppo_step(experiences)
                metrics_history.append(metrics)
                
                # Update KL coefficient
                avg_kl = sum(
                    self.ppo_trainer.compute_kl_penalty(
                        e.prompt_ids, e.response_ids
                    ).item() for e in experiences
                ) / len(experiences)
                self.ppo_trainer.update_kl_coef(avg_kl)
                
                if step % 100 == 0:
                    avg_reward = sum(e.rewards for e in experiences) / len(experiences)
                    logger.info(
                        f"Step {step}: reward={avg_reward:.3f}, "
                        f"kl={avg_kl:.3f}, kl_coef={self.ppo_trainer.kl_coef:.4f}"
                    )
            
            return metrics_history[-1] if metrics_history else {}

else:
    # Stubs when torch not available
    class RewardModel:
        pass
    
    class RewardTrainer:
        pass
    
    class PPOTrainer:
        pass
    
    class RLHFPipeline:
        pass
