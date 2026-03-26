"""
Reinforcement Learning Training for Enigma Engine
====================================================

Implements:
- RL-B: RLHF with reward model â€” small reward model trained from
  DPO preference data, then PPO-style policy gradient on the main model.
- RL-C: Self-play RL â€” TRAINER model scores STUDENT responses as
  reward signal; policy gradient updates push STUDENT toward higher
  scored outputs.

Both approaches share a RewardModel (small transformer + scalar head)
and a policy gradient training loop with KL penalty against a frozen
reference policy.

Usage:
    from enigma_engine.core.rl_training import (
        RewardModel, RewardTrainer, RLHFTrainer, SelfPlayTrainer,
    )

    # RL-B: Train reward model, then RLHF
    reward_model = RewardModel(base_model)
    reward_trainer = RewardTrainer(reward_model, tokenizer)
    reward_trainer.train(preference_data)

    rlhf = RLHFTrainer(model, tokenizer, reward_model)
    rlhf.train(prompts)

    # RL-C: Self-play with TRAINER as reward
    sp = SelfPlayTrainer(student, tokenizer, trainer_engine)
    sp.train(prompts)
"""
from __future__ import annotations

import copy
import logging
import math
import random
import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _resolve_amp_dtype(amp_dtype: str = "auto") -> torch.dtype:
    """Resolve AMP dtype string to torch.dtype.

    ``"auto"`` picks BF16 on Ampere+ / Blackwell GPUs, FP16 otherwise.
    """
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    if amp_dtype == "float16":
        return torch.float16
    # "auto"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

# =============================================================================
# VALUE HEAD — critic for PPO advantage estimation
# =============================================================================


class ValueHead(nn.Module):
    """MLP critic: hidden_state → scalar value.

    Two-layer MLP that projects transformer hidden states to per-token
    state values for GAE advantage estimation.

    Args:
        dim: Hidden dimension of the transformer model.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, 1)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict per-token values.

        Args:
            hidden_states: (B, T, dim) from transformer.

        Returns:
            (B, T) scalar values.
        """
        x = F.relu(self.fc1(hidden_states))
        return self.fc2(x).squeeze(-1)  # (B, T)


# =============================================================================
# ROLLOUT BUFFER — stores experiences for PPO minibatch updates
# =============================================================================


class RolloutBuffer:
    """Stores per-token rollout data for PPO training.

    Each entry is one generated response with per-token log-probs,
    values, rewards, and a mask indicating response tokens.
    Optionally stores full_ids and prompt_len for recomputing
    log-probs during PPO epoch updates.
    """

    def __init__(self):
        self._log_probs: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._rewards: list[torch.Tensor] = []
        self._masks: list[torch.Tensor] = []
        self._full_ids: list[torch.Tensor | None] = []
        self._prompt_lens: list[int | None] = []

    def store(
        self,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        response_mask: torch.Tensor,
        full_ids: torch.Tensor | None = None,
        prompt_len: int | None = None,
    ) -> None:
        """Store one rollout experience.

        All tensors should be 1D with the same length (response tokens).
        full_ids and prompt_len are optional — when provided, PPO epochs
        can recompute fresh log-probs for proper importance sampling.
        """
        self._log_probs.append(log_probs.detach())
        self._values.append(values.detach())
        self._rewards.append(rewards.detach())
        self._masks.append(response_mask.detach())
        self._full_ids.append(full_ids.detach() if full_ids is not None else None)
        self._prompt_lens.append(prompt_len)

    def __len__(self) -> int:
        return len(self._log_probs)

    def clear(self) -> None:
        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()
        self._masks.clear()
        self._full_ids.clear()
        self._prompt_lens.clear()

    def compute_advantages(
        self,
        gamma: float = 1.0,
        lam: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns across all stored experiences.

        Returns:
            (advantages, returns) — both 1D tensors (concatenated
            across all stored experiences).
        """
        all_advantages = []
        all_returns = []

        for logp, vals, rews, mask in zip(
            self._log_probs, self._values, self._rewards, self._masks,
        ):
            T = vals.shape[0]
            advantages = torch.zeros(T, device=vals.device)
            last_gae = 0.0

            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = 0.0
                else:
                    next_value = vals[t + 1].item()

                delta = rews[t].item() + gamma * next_value - vals[t].item()
                last_gae = delta + gamma * lam * last_gae
                advantages[t] = last_gae

            returns = advantages + vals
            all_advantages.append(advantages)
            all_returns.append(returns)

        return torch.cat(all_advantages), torch.cat(all_returns)

    def get_batched_data(
        self,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Concatenate all stored data for minibatch iteration.

        Returns:
            (log_probs, values, rewards, masks) — all 1D concatenated.
        """
        return (
            torch.cat(self._log_probs),
            torch.cat(self._values),
            torch.cat(self._rewards),
            torch.cat(self._masks),
        )


# =============================================================================
# PRIORITIZED REPLAY BUFFER — retains high-reward experiences across epochs
# =============================================================================


class ReplayBuffer:
    """Prioritized experience replay for RLHF training.

    Retains the best rollout experiences across epochs so high-reward
    responses are trained on more frequently.  Experiences are stored
    as detached CPU tensors to avoid VRAM accumulation.

    Priority is based on absolute reward — high-magnitude experiences
    (both positive and negative) are more informative for the policy.

    Args:
        capacity: Maximum number of experiences to retain.
        priority_alpha: Exponent for priority sampling (0 = uniform,
            1 = fully prioritized).
    """

    def __init__(self, capacity: int = 256,
                 priority_alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = priority_alpha
        self._experiences: list[dict[str, torch.Tensor | float]] = []

    def add(
        self,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        response_mask: torch.Tensor,
        reward_scalar: float,
    ) -> None:
        """Store one rollout experience (moved to CPU)."""
        exp = {
            "log_probs": log_probs.detach().cpu(),
            "values": values.detach().cpu(),
            "rewards": rewards.detach().cpu(),
            "mask": response_mask.detach().cpu(),
            "priority": abs(reward_scalar) + 1e-6,
        }
        self._experiences.append(exp)
        # Evict lowest-priority when over capacity
        if len(self._experiences) > self.capacity:
            self._experiences.sort(key=lambda e: e["priority"])
            self._experiences.pop(0)

    def sample(self, n: int, device: torch.device | str = "cpu",
               ) -> list[dict[str, torch.Tensor]]:
        """Sample *n* experiences weighted by priority.

        Returns list of dicts with tensors moved to *device*.
        """
        if not self._experiences or n <= 0:
            return []

        n = min(n, len(self._experiences))
        priorities = torch.tensor(
            [e["priority"] for e in self._experiences])
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        indices = torch.multinomial(probs, n, replacement=False).tolist()
        result = []
        for idx in indices:
            exp = self._experiences[idx]
            result.append({
                "log_probs": exp["log_probs"].to(device),
                "values": exp["values"].to(device),
                "rewards": exp["rewards"].to(device),
                "mask": exp["mask"].to(device),
            })
        return result

    def state_dict(self) -> dict:
        """Serialize buffer contents for checkpoint persistence."""
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "experiences": [
                {k: v.clone() if isinstance(v, torch.Tensor) else v
                 for k, v in exp.items()}
                for exp in self._experiences
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore buffer from a serialized state dict."""
        self.capacity = state.get("capacity", self.capacity)
        self.alpha = state.get("alpha", self.alpha)
        self._experiences = state.get("experiences", [])

    def __len__(self) -> int:
        return len(self._experiences)


# =============================================================================
# SHARED HELPERS
# =============================================================================


def _get_response_logps(
    model: nn.Module,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Compute per-token log-probs for the response portion only.

    Args:
        model: Language model — forward(input_ids) → logits or (logits, loss).
        full_ids: (1, prompt_len + response_len) token ids.
        prompt_len: Length of the prompt prefix.

    Returns:
        (response_len,) tensor of log-probs.
    """
    logits = model(full_ids[:, :-1])
    if isinstance(logits, tuple):
        logits = logits[0]

    log_probs = F.log_softmax(logits, dim=-1)  # (1, L-1, V)
    targets = full_ids[:, 1:]  # (1, L-1)
    per_token = log_probs.gather(
        2, targets.unsqueeze(-1)).squeeze(-1)  # (1, L-1)

    response_start = max(prompt_len - 1, 0)
    return per_token[0, response_start:]


def _get_response_entropy(
    model: nn.Module,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Compute per-token entropy for the response portion.

    Returns:
        (response_len,) tensor of entropies.
    """
    logits = model(full_ids[:, :-1])
    if isinstance(logits, tuple):
        logits = logits[0]

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # (1, L-1)

    response_start = max(prompt_len - 1, 0)
    return entropy[0, response_start:]


# =============================================================================
# REWARD MODEL
# =============================================================================

class RewardModel(nn.Module):
    """Small transformer + scalar head that outputs a reward score.

    Built on top of an existing Enigma model â€” reuses its embeddings
    and transformer layers but replaces the language model head with
    a single-value linear projection.  The base weights can be frozen
    or fine-tuned.

    Args:
        base_model: An Enigma model whose architecture is reused.
        freeze_base: If True, freeze all base model weights and only
            train the reward head.  Default True.
    """

    def __init__(self, base_model: nn.Module, freeze_base: bool = True):
        super().__init__()

        # Store config for serialization
        self.base_config = getattr(base_model, "config", None)

        # Reuse embeddings + layers from the base model
        self.tok_embeddings = base_model.tok_embeddings
        self.layers = base_model.layers
        self.norm = base_model.norm

        # Copy RoPE frequencies if present
        if hasattr(base_model, "freqs_cis") and base_model.freqs_cis is not None:
            self.register_buffer("freqs_cis", base_model.freqs_cis.clone())
        else:
            self.freqs_cis = None

        # On-demand causal mask (same pattern as Enigma model)
        self._causal_mask: torch.Tensor | None = None
        self._causal_mask_size: int = 0

        # Copy position embeddings if not using RoPE
        config = getattr(base_model, "config", None)
        self._use_rope = getattr(config, "use_rope", True)
        if not self._use_rope and hasattr(base_model, "pos"):
            self.pos = base_model.pos

        # Determine hidden dim
        dim = getattr(config, "dim", 512)

        # Reward head: project last hidden state to scalar reward
        self.reward_head = nn.Linear(dim, 1, bias=False)
        nn.init.normal_(self.reward_head.weight, std=0.02)

        # Optionally freeze the transformer
        if freeze_base:
            for param in self.tok_embeddings.parameters():
                param.requires_grad = False
            for param in self.layers.parameters():
                param.requires_grad = False
            for param in self.norm.parameters():
                param.requires_grad = False

    def _get_causal_mask(self, size: int) -> torch.Tensor:
        """Return a (size, size) upper-triangle causal mask, cached and grown on demand."""
        if self._causal_mask is None or self._causal_mask_size < size:
            new_size = max(size, self._causal_mask_size)
            mask = torch.full((new_size, new_size), float('-inf'))
            self._causal_mask = torch.triu(mask, diagonal=1)
            self._causal_mask_size = new_size
        return self._causal_mask[:size, :size]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute reward scores for input sequences.

        Args:
            input_ids: (B, L) token ids.
            attention_mask: optional (B, L) mask.  1 = real token,
                0 = padding.  Used to pick the last non-pad position.

        Returns:
            (B,) tensor of scalar rewards.
        """
        B, T = input_ids.shape

        h = self.tok_embeddings(input_ids)

        if not self._use_rope and hasattr(self, "pos"):
            h = h + self.pos[:, :T]

        # Causal mask (on-demand, same pattern as Enigma model)
        mask = None
        if T > 1:
            mask = self._get_causal_mask(T).to(device=h.device).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, self.freqs_cis, mask, False, 0)

        h = self.norm(h)  # (B, T, dim)

        # Pick the last non-padding token per sequence
        if attention_mask is not None:
            # Last non-zero position per batch
            lengths = attention_mask.sum(dim=-1).long() - 1
            lengths = lengths.clamp(min=0)
            last_hidden = h[torch.arange(B, device=h.device), lengths]
        else:
            last_hidden = h[:, -1, :]  # (B, dim)

        rewards = self.reward_head(last_hidden).squeeze(-1)  # (B,)
        return rewards


# =============================================================================
# REWARD TRAINER â€” train reward model from preference data
# =============================================================================

@dataclass
class RewardTrainerConfig:
    """Config for reward model training."""
    epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_clip: float = 1.0
    margin: float = 0.0  # optional margin in loss
    max_length: int = 512
    use_amp: bool = True
    amp_dtype: str = "auto"  # "auto", "float16", "bfloat16"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8


class RewardTrainer:
    """Train a RewardModel from preference pairs.

    Preference data format: list of dicts with ``prompt``,
    ``chosen`` (preferred response), and ``rejected``
    (dis-preferred response).

    The loss pushes ``reward(chosen) > reward(rejected)``.
    """

    def __init__(
        self,
        reward_model: RewardModel,
        tokenizer: Any,
        config: RewardTrainerConfig | None = None,
    ):
        self.model = reward_model
        self.tokenizer = tokenizer
        self.config = config or RewardTrainerConfig()
        self.device = next(reward_model.parameters()).device

        self.on_progress: Callable[[int, str], None] | None = None
        self._stop_requested = False
        self._lock = threading.Lock()

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def _should_stop(self) -> bool:
        with self._lock:
            return self._stop_requested

    def train(
        self,
        preference_data: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Train reward model on preference pairs.

        Args:
            preference_data: list of {prompt, chosen, rejected}.

        Returns:
            Dict with final_loss, epochs_completed.
        """
        self._stop_requested = False
        self.model.train()

        # Encode pairs
        pairs = self._encode_pairs(preference_data)
        if not pairs:
            raise ValueError("No valid preference pairs after encoding")

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
        )

        amp_dt = _resolve_amp_dtype(getattr(self.config, "amp_dtype", "auto"))
        scaler = torch.amp.GradScaler("cuda") if (
            self.config.use_amp and torch.cuda.is_available()
            and amp_dt != torch.bfloat16
        ) else None

        final_loss = 0.0
        epochs_done = 0

        for epoch in range(self.config.epochs):
            if self._should_stop():
                break

            epoch_loss = 0.0
            random.shuffle(pairs)

            for chosen_ids, rejected_ids in pairs:
                if self._should_stop():
                    break

                optimizer.zero_grad()

                with torch.amp.autocast(
                    "cuda",
                    dtype=amp_dt,
                    enabled=self.config.use_amp and torch.cuda.is_available(),
                ):
                    chosen_reward = self.model(chosen_ids)
                    rejected_reward = self.model(rejected_ids)

                    # Bradley-Terry loss
                    loss = -F.logsigmoid(
                        chosen_reward - rejected_reward - self.config.margin
                    ).mean()

                if scaler is not None:
                    scaler.scale(loss).backward()
                    if self.config.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if self.config.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip)
                    optimizer.step()

                batch_loss = loss.item()
                if math.isnan(batch_loss) or math.isinf(batch_loss):
                    logger.error("Reward training aborted: NaN/Inf loss")
                    self.model.eval()
                    return {"final_loss": float("inf"), "epochs_completed": epochs_done}

                epoch_loss += batch_loss

            avg = epoch_loss / max(len(pairs), 1)
            final_loss = avg
            epochs_done = epoch + 1
            logger.info("Reward epoch %d: loss=%.4f", epoch + 1, avg)

            if self.on_progress:
                pct = int((epoch + 1) / self.config.epochs * 100)
                self.on_progress(pct, f"Reward epoch {epoch + 1}: loss={avg:.4f}")

        self.model.eval()
        return {"final_loss": final_loss, "epochs_completed": epochs_done}

    def _encode_pairs(
        self,
        data: list[dict[str, str]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Encode preference pairs to tensors."""
        pairs = []
        max_len = self.config.max_length

        for item in data:
            prompt = item.get("prompt", "")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            if not (prompt and chosen and rejected):
                continue

            c_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: {chosen}")[:max_len]
            r_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: {rejected}")[:max_len]

            c_tensor = torch.tensor([c_ids], dtype=torch.long, device=self.device)
            r_tensor = torch.tensor([r_ids], dtype=torch.long, device=self.device)
            pairs.append((c_tensor, r_tensor))

        return pairs


# =============================================================================
# RLHF TRAINER (RL-B) â€” PPO-style policy gradient with reward model
# =============================================================================

@dataclass
class RLHFConfig:
    """Config for RLHF training with PPO."""
    epochs: int = 3
    learning_rate: float = 1e-6
    kl_coeff: float = 0.1  # KL penalty coefficient
    clip_range: float = 0.2  # PPO clipped surrogate ratio
    n_responses: int = 4  # responses per prompt for best-of-N
    max_new_tokens: int = 128
    temperature: float = 0.8
    gradient_clip: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "auto"  # "auto", "float16", "bfloat16"
    max_prompt_length: int = 256
    gamma: float = 1.0  # discount factor
    normalize_rewards: bool = True
    # PPO-specific fields
    value_coeff: float = 0.5  # value loss weight
    entropy_coeff: float = 0.01  # entropy bonus weight
    gae_lambda: float = 0.95  # GAE lambda
    ppo_epochs: int = 4  # minibatch update epochs per rollout
    minibatch_size: int = 4  # rollouts per minibatch
    # Replay buffer settings
    replay_capacity: int = 256  # max stored experiences (0 = disabled)
    replay_ratio: float = 0.25  # fraction of minibatch from replay
    # Optimizer betas (LM-friendly defaults)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    # Checkpoint persistence (empty = disabled)
    checkpoint_dir: str = ""


class RLHFTrainer:
    """RLHF trainer: uses a trained RewardModel to score responses,
    then applies policy gradient updates with KL penalty against
    a frozen reference policy.

    This implements a simplified PPO loop suitable for local training
    with limited compute.

    Args:
        model: The policy model (Enigma) to train.
        tokenizer: Tokenizer for encoding/decoding.
        reward_model: Trained RewardModel.
        config: RLHF training config.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        reward_model: RewardModel,
        config: RLHFConfig | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.config = config or RLHFConfig()
        self.device = next(model.parameters()).device

        self.on_progress: Callable[[int, str], None] | None = None
        self._stop_requested = False
        self._lock = threading.Lock()

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def _should_stop(self) -> bool:
        with self._lock:
            return self._stop_requested

    def _save_checkpoint(
        self,
        path: Path,
        optimizer: Any,
        value_head: nn.Module,
        replay: ReplayBuffer | None,
        epoch: int,
        metrics: dict,
    ) -> None:
        """Save RLHF training checkpoint with replay buffer state."""
        try:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "value_head_state_dict": value_head.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
            }
            if replay is not None:
                checkpoint["replay_buffer"] = replay.state_dict()
            if hasattr(self.model, "config"):
                checkpoint["model_config"] = self.model.config.__dict__
            from enigma_engine.core.safe_save import atomic_torch_save
            atomic_torch_save(checkpoint, path)
            logger.info("RLHF checkpoint saved: %s", path)
        except Exception as exc:
            logger.error("Failed to save RLHF checkpoint: %s", exc)

    def load_checkpoint(self, path: Path) -> None:
        """Load RLHF checkpoint. Pending state is applied in train().

        Restores model weights immediately. Optimizer, value head,
        and replay buffer states are stashed as ``_pending_*``
        attributes and applied after those objects are created
        inside ``train()``.
        """
        from enigma_engine.core.model_registry import safe_load_weights
        checkpoint = safe_load_weights(path, map_location=self.device)

        state_dict = checkpoint.get("model_state_dict")
        if state_dict:
            self.model.load_state_dict(state_dict)

        self._pending_optimizer_state = checkpoint.get(
            "optimizer_state_dict")
        self._pending_value_head_state = checkpoint.get(
            "value_head_state_dict")
        self._pending_replay_state = checkpoint.get("replay_buffer")
        self._start_epoch = checkpoint.get("epoch", 0)
        logger.info("RLHF checkpoint loaded: %s", path)

    def _setup_reference(self) -> None:
        """Set up the reference policy for KL penalty.

        Preferred: wrap model with LoRA so the frozen base *is* the
        reference — zero extra VRAM.  Fallback: CPU-offloaded copy
        (one GPU transfer per KL computation).
        """
        self._use_lora_ref = False
        self._ref_model_cpu = None

        try:
            from enigma_engine.core.lora_utils import (
                create_lora_model, PEFT_AVAILABLE, LoraConfig,
            )
            if PEFT_AVAILABLE:
                lora_cfg = LoraConfig(rank=8, alpha=16, dropout=0.0)
                self.model = create_lora_model(self.model, lora_cfg)
                self._use_lora_ref = True
                logger.info(
                    "RLHF: using LoRA — frozen base weights serve "
                    "as reference policy (no extra VRAM)")
                return
        except Exception as exc:
            logger.debug("LoRA setup failed, using CPU offload: %s", exc)

        # Fallback: CPU-offloaded reference
        ref = copy.deepcopy(self.model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False
        ref = ref.cpu()
        self._ref_model_cpu = ref
        logger.info(
            "RLHF: using CPU-offloaded reference model "
            "(no extra VRAM, slight I/O overhead)")

    def _get_ref_logps(
        self, full_ids: torch.Tensor, prompt_len: int,
    ) -> torch.Tensor:
        """Get reference policy log-probs, using LoRA disable or CPU offload."""
        if self._use_lora_ref:
            self.model.eval()
            with torch.no_grad():
                self.model.disable_adapter_layers()
                try:
                    ref_logps = _get_response_logps(
                        self.model, full_ids, prompt_len)
                finally:
                    self.model.enable_adapter_layers()
            self.model.train()
            return ref_logps

        ref = self._ref_model_cpu.to(self.device)
        try:
            with torch.no_grad():
                ref_logps = _get_response_logps(ref, full_ids, prompt_len)
            return ref_logps
        finally:
            ref.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _get_hidden_states(
        self,
        model: nn.Module,
        full_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Get hidden states for the response portion (for value head).

        Runs a forward pass through the model's embeddings and layers
        to extract hidden states without the LM head.
        """
        # Use model.forward to get logits, then extract from model internals
        # The simplest approach: we already run forward for logps, so we
        # re-run with a hook or use the intermediate hidden state.
        # For efficiency: separate hidden state extraction.
        h = model.tok_embeddings(full_ids[:, :-1])

        config = getattr(model, "config", None)
        use_rope = getattr(config, "use_rope", True)
        if not use_rope and hasattr(model, "pos"):
            T = full_ids.shape[1] - 1
            h = h + model.pos[:, :T]

        freqs = getattr(model, "freqs_cis", None)
        T = h.shape[1]
        mask = None
        if T > 1 and hasattr(model, '_get_causal_mask'):
            mask = model._get_causal_mask(T).to(device=h.device).unsqueeze(0).unsqueeze(0)

        for layer in model.layers:
            h = layer(h, freqs, mask, False, 0)

        h = model.norm(h)  # (1, T, dim)

        response_start = max(prompt_len - 1, 0)
        return h[:, response_start:, :]  # (1, resp_len, dim)

    def train(
        self,
        prompts: list[str],
    ) -> dict[str, Any]:
        """Run PPO training on a list of prompts.

        Two-phase loop per epoch:
          Phase 1 — Rollout collection: for each prompt, generate
          n_responses responses, score with reward model, keep the
          best, collect per-token (logprobs, values, rewards).
          Phase 2 — PPO update: multiple epochs of minibatch updates
          with clipped surrogate loss + value loss + entropy bonus.

        Args:
            prompts: Training prompts.

        Returns:
            Dict with metrics.
        """
        if not prompts:
            raise ValueError("No prompts provided for RLHF training")

        self._stop_requested = False
        cfg = self.config

        # Freeze reward model
        self.reward_model.eval()
        for p in self.reward_model.parameters():
            p.requires_grad = False

        # Set up reference policy (LoRA or CPU offload)
        self._setup_reference()

        # Value head for advantage estimation
        model_config = getattr(self.model, "config", None)
        dim = getattr(model_config, "dim", 512)
        value_head = ValueHead(dim).to(self.device)

        self.model.train()

        # Only train parameters that require grad (LoRA adapters or all)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        all_params = trainable + list(value_head.parameters())
        optimizer = torch.optim.AdamW(
            all_params, lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
        )

        amp_dt = _resolve_amp_dtype(getattr(cfg, "amp_dtype", "auto"))
        scaler = torch.amp.GradScaler("cuda") if (
            cfg.use_amp and torch.cuda.is_available()
            and amp_dt != torch.bfloat16
        ) else None
        use_amp = cfg.use_amp and torch.cuda.is_available()

        all_rewards: list[float] = []
        all_kl: list[float] = []
        epochs_done = 0
        # Bounded reward history for normalization
        reward_history: deque[float] = deque(maxlen=500)

        # Prioritized replay buffer for cross-epoch experience reuse
        replay = (
            ReplayBuffer(capacity=cfg.replay_capacity)
            if cfg.replay_capacity > 0 else None
        )

        # Restore pending state from load_checkpoint()
        start_epoch = getattr(self, "_start_epoch", 0)
        if hasattr(self, "_pending_optimizer_state") and self._pending_optimizer_state:
            optimizer.load_state_dict(self._pending_optimizer_state)
            self._pending_optimizer_state = None
        if hasattr(self, "_pending_value_head_state") and self._pending_value_head_state:
            value_head.load_state_dict(self._pending_value_head_state)
            self._pending_value_head_state = None
        if hasattr(self, "_pending_replay_state") and self._pending_replay_state:
            if replay is not None:
                replay.load_state_dict(self._pending_replay_state)
            self._pending_replay_state = None

        # Checkpoint directory for periodic saves
        ckpt_dir = None
        if cfg.checkpoint_dir:
            from pathlib import Path
            ckpt_dir = Path(cfg.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(start_epoch, cfg.epochs):
            if self._should_stop():
                break

            # ---- Phase 1: Collect rollouts ----
            rollout = RolloutBuffer()
            epoch_reward = 0.0
            epoch_kl = 0.0
            n_collected = 0
            random.shuffle(prompts)

            for prompt in prompts:
                if self._should_stop():
                    break

                prompt_ids = self.tokenizer.encode(prompt)
                prompt_ids = prompt_ids[:cfg.max_prompt_length]
                prompt_tensor = torch.tensor(
                    [prompt_ids], dtype=torch.long, device=self.device)
                prompt_len = len(prompt_ids)

                # Generate n_responses, keep the best
                best_gen = None
                best_reward = float("-inf")

                self.model.eval()
                for _ in range(cfg.n_responses):
                    with torch.no_grad():
                        try:
                            gen_ids = self.model.generate(
                                prompt_tensor,
                                max_new_tokens=cfg.max_new_tokens,
                                temperature=cfg.temperature,
                            )
                        except Exception as exc:
                            logger.debug("Generation failed: %s", exc)
                            continue

                    resp_ids = gen_ids[:, prompt_len:]
                    if resp_ids.shape[1] < 1:
                        continue

                    with torch.no_grad():
                        r = self.reward_model(gen_ids).item()

                    if r > best_reward:
                        best_reward = r
                        best_gen = gen_ids

                self.model.train()

                if best_gen is None:
                    continue

                full_ids = best_gen
                resp_len = full_ids.shape[1] - prompt_len
                reward_history.append(best_reward)

                # Normalized reward → per-token reward (terminal)
                reward_scalar = best_reward
                if cfg.normalize_rewards and len(reward_history) > 1:
                    r_mean = sum(reward_history) / len(reward_history)
                    r_var = sum(
                        (r - r_mean) ** 2 for r in reward_history
                    ) / len(reward_history)
                    r_std = max(r_var ** 0.5, 1e-8)
                    reward_scalar = (best_reward - r_mean) / r_std

                # Per-token reward: 0 everywhere except last response token
                per_token_rewards = torch.zeros(
                    resp_len, device=self.device)
                per_token_rewards[-1] = reward_scalar

                # Collect old log-probs and values (no grad)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                        old_logps = _get_response_logps(
                            self.model, full_ids, prompt_len)
                        hidden = self._get_hidden_states(
                            self.model, full_ids, prompt_len)
                        old_values = value_head(hidden).squeeze(0)  # (resp_len,)

                        ref_logps = self._get_ref_logps(full_ids, prompt_len)

                kl = (old_logps - ref_logps).mean().item()
                response_mask = torch.ones(resp_len, device=self.device)

                rollout.store(
                    log_probs=old_logps,
                    values=old_values,
                    rewards=per_token_rewards,
                    response_mask=response_mask,
                    full_ids=full_ids,
                    prompt_len=prompt_len,
                )

                # Add to replay buffer for cross-epoch reuse
                if replay is not None:
                    replay.add(
                        log_probs=old_logps,
                        values=old_values,
                        rewards=per_token_rewards,
                        response_mask=response_mask,
                        reward_scalar=reward_scalar,
                    )

                epoch_reward += best_reward
                epoch_kl += kl
                n_collected += 1

            if n_collected == 0:
                epochs_done = epoch + 1
                all_rewards.append(0.0)
                all_kl.append(0.0)
                continue

            # ---- Phase 2: PPO minibatch updates ----

            # Mix in replay experiences from prior epochs
            if replay is not None and len(replay) > 0:
                n_replay = max(1, int(n_collected * cfg.replay_ratio))
                replay_samples = replay.sample(
                    n_replay, device=self.device)
                for s in replay_samples:
                    rollout.store(
                        log_probs=s["log_probs"],
                        values=s["values"],
                        rewards=s["rewards"],
                        response_mask=s["mask"],
                    )

            advantages, returns = rollout.compute_advantages(
                gamma=cfg.gamma, lam=cfg.gae_lambda)
            old_logps_all, old_values_all, _, masks_all = (
                rollout.get_batched_data())

            # Normalize advantages
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8)

            # Store full_ids and prompt_lens for recomputation
            # We need to re-forward through the model on each PPO epoch.
            # For simplicity with variable-length rollouts, we iterate
            # over the stored rollout entries.
            n_rollouts = len(rollout)

            for ppo_epoch in range(cfg.ppo_epochs):
                if self._should_stop():
                    break

                # Shuffle rollout indices for minibatch
                indices = list(range(n_rollouts))
                random.shuffle(indices)

                for mb_start in range(0, n_rollouts, cfg.minibatch_size):
                    mb_indices = indices[mb_start:mb_start + cfg.minibatch_size]

                    total_policy_loss = torch.tensor(
                        0.0, device=self.device)
                    total_value_loss = torch.tensor(
                        0.0, device=self.device)
                    total_entropy = torch.tensor(
                        0.0, device=self.device)
                    n_tokens = 0

                    for idx in mb_indices:
                        rollout_len = rollout._log_probs[idx].shape[0]
                        start = sum(
                            rollout._log_probs[j].shape[0]
                            for j in range(idx))
                        end = start + rollout_len

                        mb_advantages_i = advantages[start:end]
                        mb_returns_i = returns[start:end]
                        mb_old_logps_i = old_logps_all[start:end]

                        # Recompute fresh log-probs and values if we
                        # have the original full_ids (not replay items)
                        stored_ids = rollout._full_ids[idx]
                        stored_plen = rollout._prompt_lens[idx]
                        if stored_ids is not None and stored_plen is not None:
                            with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                                new_logps = _get_response_logps(
                                    self.model, stored_ids, stored_plen)
                                hidden = self._get_hidden_states(
                                    self.model, stored_ids, stored_plen)
                                new_values = value_head(hidden).squeeze(0)
                                entropy = _get_response_entropy(
                                    self.model, stored_ids, stored_plen)
                            ratio = torch.exp(new_logps - mb_old_logps_i)
                            entropy_bonus = entropy.mean()
                        else:
                            # Replay items: no full_ids, fall back to
                            # ratio=1 (first PPO epoch is exact)
                            ratio = torch.ones_like(mb_old_logps_i)
                            new_values = old_values_all[start:end]
                            entropy_bonus = -mb_old_logps_i.mean()

                        # Clipped surrogate loss
                        surr1 = ratio * mb_advantages_i
                        surr2 = torch.clamp(
                            ratio,
                            1.0 - cfg.clip_range,
                            1.0 + cfg.clip_range,
                        ) * mb_advantages_i
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Value loss (clipped)
                        value_loss = F.mse_loss(
                            new_values, mb_returns_i)

                        total_policy_loss = (
                            total_policy_loss + policy_loss)
                        total_value_loss = (
                            total_value_loss + value_loss)
                        total_entropy = total_entropy + entropy_bonus
                        n_tokens += rollout_len

                    mb_count = max(len(mb_indices), 1)
                    loss = (
                        total_policy_loss / mb_count
                        + cfg.value_coeff * total_value_loss / mb_count
                        - cfg.entropy_coeff * total_entropy / mb_count
                        + cfg.kl_coeff * abs(epoch_kl / max(n_collected, 1))
                    )

                    optimizer.zero_grad()
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        if cfg.gradient_clip > 0:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(
                                all_params, cfg.gradient_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if cfg.gradient_clip > 0:
                            nn.utils.clip_grad_norm_(
                                all_params, cfg.gradient_clip)
                        optimizer.step()

            avg_reward = epoch_reward / max(n_collected, 1)
            avg_kl = epoch_kl / max(n_collected, 1)
            all_rewards.append(avg_reward)
            all_kl.append(avg_kl)
            epochs_done = epoch + 1

            logger.info(
                "RLHF epoch %d: avg_reward=%.4f, avg_kl=%.4f",
                epoch + 1, avg_reward, avg_kl)

            if self.on_progress:
                pct = int((epoch + 1) / cfg.epochs * 100)
                self.on_progress(
                    pct,
                    f"RLHF epoch {epoch + 1}: reward={avg_reward:.3f}, kl={avg_kl:.4f}")

            if ckpt_dir:
                self._save_checkpoint(
                    ckpt_dir / f"rlhf_epoch{epoch + 1}.pt",
                    optimizer, value_head, replay, epoch + 1,
                    {"avg_rewards": all_rewards, "avg_kl": all_kl})

        self.model.eval()
        if self._ref_model_cpu is not None:
            del self._ref_model_cpu
            self._ref_model_cpu = None

        return {
            "epochs_completed": epochs_done,
            "avg_rewards": all_rewards,
            "avg_kl": all_kl,
            "final_reward": all_rewards[-1] if all_rewards else 0.0,
        }



# =============================================================================
# SELF-PLAY RL (RL-C) â€” TRAINER as reward signal
# =============================================================================

@dataclass
class SelfPlayConfig:
    """Config for self-play RL training."""
    epochs: int = 3
    learning_rate: float = 1e-6
    kl_coeff: float = 0.05
    clip_range: float = 0.2  # PPO clipped surrogate ratio
    n_responses: int = 4
    max_new_tokens: int = 128
    temperature: float = 0.8
    gradient_clip: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "auto"  # "auto", "float16", "bfloat16"
    max_prompt_length: int = 256
    score_prompt: str = (
        "Rate the following response on a scale of 0 to 10 for quality, "
        "helpfulness, and coherence. Respond with ONLY the number.\n\n"
        "Question: {prompt}\n\nResponse: {response}\n\nScore:"
    )
    # PPO-specific fields
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    minibatch_size: int = 4
    # Replay buffer settings
    replay_capacity: int = 256  # max stored experiences (0 = disabled)
    replay_ratio: float = 0.25  # fraction of minibatch from replay
    # Optimizer betas (LM-friendly defaults)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    # Checkpoint persistence (empty = disabled)
    checkpoint_dir: str = ""


class SelfPlayTrainer:
    """Self-play RL: TRAINER model scores STUDENT responses.

    The TRAINER model is used as a reward function â€” it reads the
    STUDENT's response and outputs a numerical score.  Policy gradient
    updates push the STUDENT toward higher-scoring responses, with
    a KL penalty against the initial STUDENT weights.

    Args:
        student: The model being trained (Enigma).
        tokenizer: Tokenizer for the student model.
        trainer_engine: An EnigmaEngine loaded with the TRAINER model,
            used to generate reward scores via chat.
        config: Self-play config.
    """

    def __init__(
        self,
        student: nn.Module,
        tokenizer: Any,
        trainer_engine: Any,
        config: SelfPlayConfig | None = None,
    ):
        self.student = student
        self.tokenizer = tokenizer
        self.trainer_engine = trainer_engine
        self.config = config or SelfPlayConfig()
        self.device = next(student.parameters()).device

        self.on_progress: Callable[[int, str], None] | None = None
        self._stop_requested = False
        self._lock = threading.Lock()

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def _should_stop(self) -> bool:
        with self._lock:
            return self._stop_requested

    def _save_checkpoint(
        self,
        path: Path,
        optimizer: Any,
        value_head: nn.Module,
        replay: ReplayBuffer | None,
        epoch: int,
        metrics: dict,
    ) -> None:
        """Save self-play training checkpoint with replay buffer state."""
        try:
            checkpoint = {
                "model_state_dict": self.student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "value_head_state_dict": value_head.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
            }
            if replay is not None:
                checkpoint["replay_buffer"] = replay.state_dict()
            if hasattr(self.student, "config"):
                checkpoint["model_config"] = self.student.config.__dict__
            from enigma_engine.core.safe_save import atomic_torch_save
            atomic_torch_save(checkpoint, path)
            logger.info("Self-play checkpoint saved: %s", path)
        except Exception as exc:
            logger.error("Failed to save self-play checkpoint: %s", exc)

    def load_checkpoint(self, path: Path) -> None:
        """Load self-play checkpoint. Pending state is applied in train().

        Restores student weights immediately. Optimizer, value head,
        and replay buffer states are stashed as ``_pending_*``
        attributes and applied after those objects are created
        inside ``train()``.
        """
        from enigma_engine.core.model_registry import safe_load_weights
        checkpoint = safe_load_weights(path, map_location=self.device)

        state_dict = checkpoint.get("model_state_dict")
        if state_dict:
            self.student.load_state_dict(state_dict)

        self._pending_optimizer_state = checkpoint.get(
            "optimizer_state_dict")
        self._pending_value_head_state = checkpoint.get(
            "value_head_state_dict")
        self._pending_replay_state = checkpoint.get("replay_buffer")
        self._start_epoch = checkpoint.get("epoch", 0)
        logger.info("Self-play checkpoint loaded: %s", path)

    def _get_trainer_score(self, prompt: str, response: str) -> float:
        """Ask the TRAINER model to score a response.

        Includes a small emotional quality bonus (Phase 6) that rewards
        responses likely to increase user engagement.

        Args:
            prompt: The original question.
            response: The student's response.

        Returns:
            Numeric score 0-10, or 5.0 on failure.
        """
        score_prompt = self.config.score_prompt.format(
            prompt=prompt, response=response)

        base_score = 5.0  # neutral fallback
        try:
            result = self.trainer_engine.chat(score_prompt, max_tokens=16)
            if isinstance(result, dict):
                result = result.get("response", result.get("text", "5"))
            result = str(result).strip()

            # Extract first number from response
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', result)
            if match:
                base_score = float(match.group(1))
                base_score = min(max(base_score, 0.0), 10.0)
        except Exception as exc:
            logger.debug("Trainer scoring failed: %s", exc)

        # Phase 6: emotional quality bonus
        try:
            from enigma_engine.core.sentiment import evaluate_response_quality
            bonus = evaluate_response_quality(prompt, response)
            # Scale bonus to 0-10 range (±0.5 → ±2.5 points)
            base_score = min(max(base_score + bonus * 5.0, 0.0), 10.0)
        except Exception:
            pass

        return base_score

    def _setup_reference(self) -> None:
        """Set up reference policy for KL penalty (same approach as RLHFTrainer)."""
        self._use_lora_ref = False
        self._ref_model_cpu = None

        try:
            from enigma_engine.core.lora_utils import (
                create_lora_model, PEFT_AVAILABLE, LoraConfig,
            )
            if PEFT_AVAILABLE:
                lora_cfg = LoraConfig(rank=8, alpha=16, dropout=0.0)
                self.student = create_lora_model(self.student, lora_cfg)
                self._use_lora_ref = True
                logger.info(
                    "Self-play: using LoRA — frozen base weights serve "
                    "as reference policy (no extra VRAM)")
                return
        except Exception as exc:
            logger.debug("LoRA setup failed, using CPU offload: %s", exc)

        # Fallback: CPU-offloaded reference
        ref = copy.deepcopy(self.student)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False
        ref = ref.cpu()
        self._ref_model_cpu = ref
        logger.info(
            "Self-play: using CPU-offloaded reference model "
            "(no extra VRAM, slight I/O overhead)")

    def _get_ref_logps(
        self, full_ids: torch.Tensor, prompt_len: int,
    ) -> torch.Tensor:
        """Get reference policy log-probs, using LoRA disable or CPU offload."""
        if self._use_lora_ref:
            self.student.eval()
            with torch.no_grad():
                self.student.disable_adapter_layers()
                try:
                    ref_logps = _get_response_logps(
                        self.student, full_ids, prompt_len)
                finally:
                    self.student.enable_adapter_layers()
            self.student.train()
            return ref_logps

        ref = self._ref_model_cpu.to(self.device)
        try:
            with torch.no_grad():
                ref_logps = _get_response_logps(ref, full_ids, prompt_len)
            return ref_logps
        finally:
            ref.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _get_hidden_states(
        self,
        model: nn.Module,
        full_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Get hidden states for the response portion (for value head)."""
        h = model.tok_embeddings(full_ids[:, :-1])

        config = getattr(model, "config", None)
        use_rope = getattr(config, "use_rope", True)
        if not use_rope and hasattr(model, "pos"):
            T = full_ids.shape[1] - 1
            h = h + model.pos[:, :T]

        freqs = getattr(model, "freqs_cis", None)
        T = h.shape[1]
        mask = None
        if T > 1 and hasattr(model, '_get_causal_mask'):
            mask = model._get_causal_mask(T).to(device=h.device).unsqueeze(0).unsqueeze(0)

        for layer in model.layers:
            h = layer(h, freqs, mask, False, 0)

        h = model.norm(h)
        response_start = max(prompt_len - 1, 0)
        return h[:, response_start:, :]

    def train(
        self,
        prompts: list[str],
    ) -> dict[str, Any]:
        """Run self-play PPO training.

        Two-phase loop per epoch:
          Phase 1 — Rollout: for each prompt, generate n_responses,
          have TRAINER score each, keep the best, collect per-token data.
          Phase 2 — PPO update with clipped surrogate + value + entropy.

        Args:
            prompts: Training prompts.

        Returns:
            Dict with metrics.
        """
        if not prompts:
            raise ValueError("No prompts provided for self-play training")

        self._stop_requested = False
        cfg = self.config

        # Set up reference policy (LoRA or CPU offload)
        self._setup_reference()

        # Value head for advantage estimation
        model_config = getattr(self.student, "config", None)
        dim = getattr(model_config, "dim", 512)
        value_head = ValueHead(dim).to(self.device)

        self.student.train()
        trainable = [p for p in self.student.parameters() if p.requires_grad]
        all_params = trainable + list(value_head.parameters())
        optimizer = torch.optim.AdamW(
            all_params, lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
        )

        amp_dt = _resolve_amp_dtype(getattr(cfg, "amp_dtype", "auto"))
        scaler = torch.amp.GradScaler("cuda") if (
            cfg.use_amp and torch.cuda.is_available()
            and amp_dt != torch.bfloat16
        ) else None
        use_amp = cfg.use_amp and torch.cuda.is_available()

        all_scores: list[float] = []
        epochs_done = 0

        # Prioritized replay buffer for cross-epoch experience reuse
        replay = (
            ReplayBuffer(capacity=cfg.replay_capacity)
            if cfg.replay_capacity > 0 else None
        )

        # Restore pending checkpoint state (from load_checkpoint)
        start_epoch = getattr(self, "_start_epoch", 0)
        if hasattr(self, "_pending_optimizer_state") and self._pending_optimizer_state:
            optimizer.load_state_dict(self._pending_optimizer_state)
            self._pending_optimizer_state = None
        if hasattr(self, "_pending_value_head_state") and self._pending_value_head_state:
            value_head.load_state_dict(self._pending_value_head_state)
            self._pending_value_head_state = None
        if hasattr(self, "_pending_replay_state") and self._pending_replay_state:
            if replay is not None:
                replay.load_state_dict(self._pending_replay_state)
            self._pending_replay_state = None

        ckpt_dir = None
        if cfg.checkpoint_dir:
            from pathlib import Path
            ckpt_dir = Path(cfg.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(start_epoch, cfg.epochs):
            if self._should_stop():
                break

            # ---- Phase 1: Collect rollouts ----
            rollout = RolloutBuffer()
            epoch_score = 0.0
            epoch_kl = 0.0
            n_collected = 0
            random.shuffle(prompts)

            for prompt in prompts:
                if self._should_stop():
                    break

                prompt_ids = self.tokenizer.encode(prompt)[:cfg.max_prompt_length]
                prompt_tensor = torch.tensor(
                    [prompt_ids], dtype=torch.long, device=self.device)
                prompt_len = len(prompt_ids)

                # Generate n_responses, score each, keep the best
                best_gen = None
                best_score = float("-inf")

                self.student.eval()
                for _ in range(cfg.n_responses):
                    with torch.no_grad():
                        try:
                            gen_ids = self.student.generate(
                                prompt_tensor,
                                max_new_tokens=cfg.max_new_tokens,
                                temperature=cfg.temperature,
                            )
                        except Exception as exc:
                            logger.debug("Student generation failed: %s", exc)
                            continue

                    resp_ids = gen_ids[0, prompt_len:].tolist()
                    response = self.tokenizer.decode(resp_ids)
                    if not response.strip():
                        continue

                    score = self._get_trainer_score(prompt, response)
                    if score > best_score:
                        best_score = score
                        best_gen = gen_ids

                self.student.train()

                if best_gen is None:
                    continue

                full_ids = best_gen
                resp_len = full_ids.shape[1] - prompt_len

                # Normalize score to [-1, 1] range
                reward_scalar = (best_score - 5.0) / 5.0

                # Per-token reward: terminal only
                per_token_rewards = torch.zeros(
                    resp_len, device=self.device)
                per_token_rewards[-1] = reward_scalar

                # Collect old log-probs and values
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                    old_logps = _get_response_logps(
                        self.student, full_ids, prompt_len)
                    hidden = self._get_hidden_states(
                        self.student, full_ids, prompt_len)
                    old_values = value_head(hidden).squeeze(0)
                    ref_logps = self._get_ref_logps(full_ids, prompt_len)

                kl = (old_logps - ref_logps).mean().item()
                response_mask = torch.ones(resp_len, device=self.device)

                rollout.store(
                    log_probs=old_logps,
                    values=old_values,
                    rewards=per_token_rewards,
                    response_mask=response_mask,
                    full_ids=full_ids,
                    prompt_len=prompt_len,
                )

                # Add to replay buffer for cross-epoch reuse
                if replay is not None:
                    replay.add(
                        log_probs=old_logps,
                        values=old_values,
                        rewards=per_token_rewards,
                        response_mask=response_mask,
                        reward_scalar=reward_scalar,
                    )

                epoch_score += best_score
                epoch_kl += kl
                n_collected += 1

            if n_collected == 0:
                epochs_done = epoch + 1
                all_scores.append(0.0)
                continue

            # ---- Phase 2: PPO minibatch updates ----

            # Mix in replay experiences from prior epochs
            if replay is not None and len(replay) > 0:
                n_replay = max(1, int(n_collected * cfg.replay_ratio))
                replay_samples = replay.sample(
                    n_replay, device=self.device)
                for s in replay_samples:
                    rollout.store(
                        log_probs=s["log_probs"],
                        values=s["values"],
                        rewards=s["rewards"],
                        response_mask=s["mask"],
                    )

            advantages, returns = rollout.compute_advantages(
                gamma=1.0, lam=cfg.gae_lambda)
            old_logps_all, old_values_all, _, masks_all = (
                rollout.get_batched_data())

            # Normalize advantages
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8)

            n_rollouts = len(rollout)

            for ppo_epoch in range(cfg.ppo_epochs):
                if self._should_stop():
                    break

                indices = list(range(n_rollouts))
                random.shuffle(indices)

                for mb_start in range(0, n_rollouts, cfg.minibatch_size):
                    mb_indices = indices[mb_start:mb_start + cfg.minibatch_size]

                    total_policy_loss = torch.tensor(
                        0.0, device=self.device)
                    total_value_loss = torch.tensor(
                        0.0, device=self.device)
                    total_entropy = torch.tensor(
                        0.0, device=self.device)

                    for idx in mb_indices:
                        rollout_len = rollout._log_probs[idx].shape[0]
                        start = sum(
                            rollout._log_probs[j].shape[0]
                            for j in range(idx))
                        end = start + rollout_len

                        mb_advantages_i = advantages[start:end]
                        mb_returns_i = returns[start:end]
                        mb_old_logps_i = old_logps_all[start:end]

                        # Recompute fresh log-probs and values if we
                        # have the original full_ids (not replay items)
                        stored_ids = rollout._full_ids[idx]
                        stored_plen = rollout._prompt_lens[idx]
                        if stored_ids is not None and stored_plen is not None:
                            with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                                new_logps = _get_response_logps(
                                    self.student, stored_ids, stored_plen)
                                hidden = self._get_hidden_states(
                                    self.student, stored_ids, stored_plen)
                                new_values = value_head(hidden).squeeze(0)
                                entropy = _get_response_entropy(
                                    self.student, stored_ids, stored_plen)
                            ratio = torch.exp(new_logps - mb_old_logps_i)
                            entropy_bonus = entropy.mean()
                        else:
                            ratio = torch.ones_like(mb_old_logps_i)
                            new_values = old_values_all[start:end]
                            entropy_bonus = -mb_old_logps_i.mean()

                        surr1 = ratio * mb_advantages_i
                        surr2 = torch.clamp(
                            ratio,
                            1.0 - cfg.clip_range,
                            1.0 + cfg.clip_range,
                        ) * mb_advantages_i
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_loss = F.mse_loss(
                            new_values, mb_returns_i)

                        total_policy_loss = (
                            total_policy_loss + policy_loss)
                        total_value_loss = (
                            total_value_loss + value_loss)
                        total_entropy = total_entropy + entropy_bonus

                    mb_count = max(len(mb_indices), 1)
                    loss = (
                        total_policy_loss / mb_count
                        + cfg.value_coeff * total_value_loss / mb_count
                        - cfg.entropy_coeff * total_entropy / mb_count
                        + cfg.kl_coeff * abs(epoch_kl / max(n_collected, 1))
                    )

                    optimizer.zero_grad()
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        if cfg.gradient_clip > 0:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(
                                all_params, cfg.gradient_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if cfg.gradient_clip > 0:
                            nn.utils.clip_grad_norm_(
                                all_params, cfg.gradient_clip)
                        optimizer.step()

            avg_score = epoch_score / max(n_collected, 1)
            all_scores.append(avg_score)
            epochs_done = epoch + 1

            logger.info("Self-play epoch %d: avg_score=%.2f", epoch + 1, avg_score)

            if self.on_progress:
                pct = int((epoch + 1) / cfg.epochs * 100)
                self.on_progress(pct, f"Self-play epoch {epoch + 1}: score={avg_score:.2f}")

            if ckpt_dir:
                self._save_checkpoint(
                    ckpt_dir / f"selfplay_epoch{epoch + 1}.pt",
                    optimizer, value_head, replay, epoch + 1,
                    {"avg_scores": all_scores})

        self.student.eval()
        if self._ref_model_cpu is not None:
            del self._ref_model_cpu
            self._ref_model_cpu = None

        return {
            "epochs_completed": epochs_done,
            "avg_scores": all_scores,
            "final_score": all_scores[-1] if all_scores else 0.0,
        }

