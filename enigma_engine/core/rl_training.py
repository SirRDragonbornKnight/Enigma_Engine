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
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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

        # Copy causal mask if present
        if hasattr(base_model, "_causal_mask"):
            self.register_buffer(
                "_causal_mask", base_model._causal_mask.clone(),
                persistent=False)
        else:
            self._causal_mask = None

        # Copy position embeddings if not using RoPE
        config = getattr(base_model, "config", None)
        self._use_rope = getattr(config, "use_rope", True)
        if not self._use_rope and hasattr(base_model, "pos"):
            self.pos = base_model.pos

        # Determine hidden dim
        dim = getattr(config, "dim", 512)

        # Reward head: project last hidden state â†’ scalar reward
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

        # Causal mask
        mask = None
        if T > 1 and self._causal_mask is not None:
            mask = self._causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)

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
        )

        scaler = torch.amp.GradScaler("cuda") if (
            self.config.use_amp and torch.cuda.is_available()
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

            c_ids = self.tokenizer.encode(f"Q: {prompt}\nA: {chosen}")[:max_len]
            r_ids = self.tokenizer.encode(f"Q: {prompt}\nA: {rejected}")[:max_len]

            c_tensor = torch.tensor([c_ids], dtype=torch.long, device=self.device)
            r_tensor = torch.tensor([r_ids], dtype=torch.long, device=self.device)
            pairs.append((c_tensor, r_tensor))

        return pairs


# =============================================================================
# RLHF TRAINER (RL-B) â€” PPO-style policy gradient with reward model
# =============================================================================

@dataclass
class RLHFConfig:
    """Config for RLHF training."""
    epochs: int = 3
    learning_rate: float = 1e-6
    kl_coeff: float = 0.1  # KL penalty coefficient
    clip_range: float = 0.2  # PPO clip range
    n_responses: int = 4  # responses per prompt for group scoring
    max_new_tokens: int = 128
    temperature: float = 0.8
    gradient_clip: float = 1.0
    use_amp: bool = True
    max_prompt_length: int = 256
    gamma: float = 1.0  # discount factor
    normalize_rewards: bool = True


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
            # Disable LoRA adapters → forward through frozen base only
            self.model.eval()
            with torch.no_grad():
                self.model.disable_adapter_layers()
                try:
                    ref_logps = self._get_response_logps(
                        self.model, full_ids, prompt_len)
                finally:
                    self.model.enable_adapter_layers()
            self.model.train()
            return ref_logps

        # CPU offload path: move to GPU, compute, move back
        ref = self._ref_model_cpu.to(self.device)
        with torch.no_grad():
            ref_logps = self._get_response_logps(
                ref, full_ids, prompt_len)
        ref.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ref_logps

    def train(
        self,
        prompts: list[str],
    ) -> dict[str, Any]:
        """Run RLHF training on a list of prompts.

        For each prompt:
        1. Generate N responses from current policy
        2. Score each with the reward model
        3. Compute policy gradient with reward + KL penalty

        Uses LoRA when available — the frozen base model weights serve
        as the reference policy with zero extra VRAM.  Falls back to
        CPU-offloaded reference if PEFT is not installed.

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

        self.model.train()

        # Only train parameters that require grad (LoRA adapters or all)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)

        all_rewards: list[float] = []
        all_kl: list[float] = []
        epochs_done = 0

        for epoch in range(cfg.epochs):
            if self._should_stop():
                break

            epoch_reward = 0.0
            epoch_kl = 0.0
            random.shuffle(prompts)

            for prompt in prompts:
                if self._should_stop():
                    break

                # Encode prompt
                prompt_ids = self.tokenizer.encode(prompt)
                prompt_ids = prompt_ids[:cfg.max_prompt_length]
                prompt_tensor = torch.tensor(
                    [prompt_ids], dtype=torch.long, device=self.device)

                # Generate response from current policy
                self.model.eval()
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
                self.model.train()

                # Get the response portion
                response_ids = gen_ids[:, len(prompt_ids):]
                if response_ids.shape[1] < 1:
                    continue

                full_ids = gen_ids  # (1, prompt_len + response_len)

                # Compute reward
                with torch.no_grad():
                    reward = self.reward_model(full_ids)  # (1,)

                reward_val = reward.item()

                # Compute log-probs under current and reference policy
                policy_logps = self._get_response_logps(
                    self.model, full_ids, len(prompt_ids))
                with torch.no_grad():
                    ref_logps = self._get_ref_logps(
                        full_ids, len(prompt_ids))

                # KL divergence estimate: E[log pi - log pi_ref]
                kl = (policy_logps - ref_logps).mean()

                # Combined objective: maximize reward - kl_coeff * KL
                loss = -(reward.detach() * policy_logps.mean()) + cfg.kl_coeff * kl

                optimizer.zero_grad()
                loss.backward()

                if cfg.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(trainable, cfg.gradient_clip)

                optimizer.step()

                epoch_reward += reward_val
                epoch_kl += kl.item()

            n = max(len(prompts), 1)
            avg_reward = epoch_reward / n
            avg_kl = epoch_kl / n
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

        self.model.eval()
        # Clean up CPU reference if used
        if self._ref_model_cpu is not None:
            del self._ref_model_cpu
            self._ref_model_cpu = None

        return {
            "epochs_completed": epochs_done,
            "avg_rewards": all_rewards,
            "avg_kl": all_kl,
            "final_reward": all_rewards[-1] if all_rewards else 0.0,
        }

    def _get_response_logps(
        self,
        model: nn.Module,
        full_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Compute per-token log-probs for the response portion only.

        Args:
            model: Language model.
            full_ids: (1, prompt_len + response_len) token ids.
            prompt_len: Length of the prompt prefix.

        Returns:
            (response_len,) tensor of log-probs.
        """
        logits = model(full_ids[:, :-1])
        if isinstance(logits, tuple):
            logits = logits[0]

        log_probs = F.log_softmax(logits, dim=-1)  # (1, L-1, V)

        # Targets are the shifted full_ids
        targets = full_ids[:, 1:]  # (1, L-1)

        # Gather log-probs for actual tokens
        per_token = log_probs.gather(
            2, targets.unsqueeze(-1)).squeeze(-1)  # (1, L-1)

        # Only keep response tokens (skip prompt tokens)
        # prompt_len tokens in input â†’ first (prompt_len-1) in targets
        response_start = max(prompt_len - 1, 0)
        return per_token[0, response_start:]


# =============================================================================
# SELF-PLAY RL (RL-C) â€” TRAINER as reward signal
# =============================================================================

@dataclass
class SelfPlayConfig:
    """Config for self-play RL training."""
    epochs: int = 3
    learning_rate: float = 1e-6
    kl_coeff: float = 0.05
    n_responses: int = 4
    max_new_tokens: int = 128
    temperature: float = 0.8
    gradient_clip: float = 1.0
    use_amp: bool = True
    max_prompt_length: int = 256
    score_prompt: str = (
        "Rate the following response on a scale of 0 to 10 for quality, "
        "helpfulness, and coherence. Respond with ONLY the number.\n\n"
        "Question: {prompt}\n\nResponse: {response}\n\nScore:"
    )


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

    def _get_trainer_score(self, prompt: str, response: str) -> float:
        """Ask the TRAINER model to score a response.

        Args:
            prompt: The original question.
            response: The student's response.

        Returns:
            Numeric score 0-10, or 5.0 on failure.
        """
        score_prompt = self.config.score_prompt.format(
            prompt=prompt, response=response)

        try:
            result = self.trainer_engine.chat(score_prompt, max_tokens=16)
            if isinstance(result, dict):
                result = result.get("response", result.get("text", "5"))
            result = str(result).strip()

            # Extract first number from response
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', result)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 10.0)
        except Exception as exc:
            logger.debug("Trainer scoring failed: %s", exc)

        return 5.0  # neutral fallback

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
                    ref_logps = self._get_response_logps(
                        self.student, full_ids, prompt_len)
                finally:
                    self.student.enable_adapter_layers()
            self.student.train()
            return ref_logps

        ref = self._ref_model_cpu.to(self.device)
        with torch.no_grad():
            ref_logps = self._get_response_logps(
                ref, full_ids, prompt_len)
        ref.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ref_logps

    def train(
        self,
        prompts: list[str],
    ) -> dict[str, Any]:
        """Run self-play RL training.

        For each prompt:
        1. STUDENT generates a response
        2. TRAINER scores it (0-10)
        3. Score is used as reward for policy gradient

        Uses LoRA when available — the frozen base model weights serve
        as the reference policy with zero extra VRAM.  Falls back to
        CPU-offloaded reference if PEFT is not installed.

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

        self.student.train()
        trainable = [p for p in self.student.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)

        all_scores: list[float] = []
        epochs_done = 0

        for epoch in range(cfg.epochs):
            if self._should_stop():
                break

            epoch_score = 0.0
            random.shuffle(prompts)

            for prompt in prompts:
                if self._should_stop():
                    break

                # Encode
                prompt_ids = self.tokenizer.encode(prompt)[:cfg.max_prompt_length]
                prompt_tensor = torch.tensor(
                    [prompt_ids], dtype=torch.long, device=self.device)

                # Generate from student
                self.student.eval()
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
                self.student.train()

                # Decode response
                resp_ids = gen_ids[0, len(prompt_ids):].tolist()
                response = self.tokenizer.decode(resp_ids)
                if not response.strip():
                    continue

                # Get trainer score
                score = self._get_trainer_score(prompt, response)
                # Normalize to [-1, 1] range
                reward = (score - 5.0) / 5.0

                # Policy gradient
                full_ids = gen_ids
                policy_logps = self._get_response_logps(
                    self.student, full_ids, len(prompt_ids))

                with torch.no_grad():
                    ref_logps = self._get_ref_logps(
                        full_ids, len(prompt_ids))

                kl = (policy_logps - ref_logps).mean()
                reward_tensor = torch.tensor(
                    reward, device=self.device, dtype=torch.float32)

                loss = -(reward_tensor * policy_logps.mean()) + cfg.kl_coeff * kl

                optimizer.zero_grad()
                loss.backward()

                if cfg.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(trainable, cfg.gradient_clip)
                optimizer.step()

                epoch_score += score

            avg_score = epoch_score / max(len(prompts), 1)
            all_scores.append(avg_score)
            epochs_done = epoch + 1

            logger.info("Self-play epoch %d: avg_score=%.2f", epoch + 1, avg_score)

            if self.on_progress:
                pct = int((epoch + 1) / cfg.epochs * 100)
                self.on_progress(pct, f"Self-play epoch {epoch + 1}: score={avg_score:.2f}")

        self.student.eval()
        # Clean up CPU reference if used
        if self._ref_model_cpu is not None:
            del self._ref_model_cpu
            self._ref_model_cpu = None

        return {
            "epochs_completed": epochs_done,
            "avg_scores": all_scores,
            "final_score": all_scores[-1] if all_scores else 0.0,
        }

    def _get_response_logps(
        self,
        model: nn.Module,
        full_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Compute log-probs for response tokens."""
        logits = model(full_ids[:, :-1])
        if isinstance(logits, tuple):
            logits = logits[0]

        log_probs = F.log_softmax(logits, dim=-1)
        targets = full_ids[:, 1:]
        per_token = log_probs.gather(
            2, targets.unsqueeze(-1)).squeeze(-1)

        response_start = max(prompt_len - 1, 0)
        return per_token[0, response_start:]

