"""
Utility functions and global model registry for Enigma Engine.

Contains sampling helpers, hardware detection utilities, and the
thread-safe global model registry.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .model import Enigma

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL MODEL REGISTRY (Thread-Safe)
# ─────────────────────────────────────────────────────────────────────────────
# Registry of all loaded model instances. Uses a lock to ensure thread safety
# when multiple threads access models concurrently (e.g., GUI + API server).

_LOADED_MODELS: dict[str, Enigma] = {}
_MODELS_LOCK = threading.RLock()  # RLock allows re-entrant locking


def get_running_models() -> dict[str, Enigma]:
    """Get a copy of all loaded model instances (thread-safe)."""
    with _MODELS_LOCK:
        return _LOADED_MODELS.copy()


def is_model_loaded(name: str) -> bool:
    """Check if a model is loaded by name (thread-safe)."""
    with _MODELS_LOCK:
        return name in _LOADED_MODELS


def register_model(name: str, model: Enigma) -> None:
    """Register a model instance (thread-safe)."""
    with _MODELS_LOCK:
        _LOADED_MODELS[name] = model
        logger.debug(f"Registered model: {name}")


def unregister_model(name: str) -> Optional[Enigma]:
    """Unregister a model and return it if found (thread-safe)."""
    with _MODELS_LOCK:
        model = _LOADED_MODELS.pop(name, None)
        if model is not None:
            logger.debug(f"Unregistered model: {name}")
        return model


def get_model(name: str) -> Optional[Enigma]:
    """Get a specific model by name (thread-safe)."""
    with _MODELS_LOCK:
        return _LOADED_MODELS.get(name)


# =============================================================================
# 🔄 REPETITION PENALTY HELPER - Efficient penalty application
# =============================================================================

def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float
) -> torch.Tensor:
    """
    Apply repetition penalty to logits based on previously generated tokens.
    
    📖 WHAT THIS DOES:
    Reduces the probability of tokens that have already been generated,
    encouraging the model to produce more diverse output.
    
    📐 HYBRID APPROACH:
    - For short sequences (<1000 tokens): Uses set-based lookup (lower overhead)
    - For longer sequences: Uses bincount (better vectorization)
    
    Args:
        logits: Logits tensor [batch, vocab_size] or [vocab_size]
        generated_tokens: Previously generated token IDs
        penalty: Penalty factor (>1.0 reduces repetition, 1.0 = no penalty)
    
    Returns:
        Modified logits with repetition penalty applied (new tensor, not in-place)
    
    Example:
        logits = apply_repetition_penalty(logits, generated_ids, penalty=1.2)
    
    Note:
        Returns a cloned tensor to avoid in-place mutation issues with
        beam search, speculative decoding, or autograd.
    """
    if penalty == 1.0:
        return logits

    # Clone to avoid in-place mutation (important for beam search, speculative decoding)
    logits = logits.clone()

    vocab_size = logits.shape[-1]
    seq_len = generated_tokens.numel()

    if seq_len < 1000:
        # Set-based for short sequences (lower overhead)
        unique_tokens = set(generated_tokens.view(-1).tolist())
        for token_id in unique_tokens:
            if 0 <= token_id < vocab_size:
                if logits.dim() == 1:
                    logits[token_id] /= penalty
                else:
                    logits[..., token_id] /= penalty
    else:
        # Bincount for longer sequences (better vectorization)
        flat_tokens = generated_tokens.view(-1).clamp(0, vocab_size - 1)
        token_counts = torch.bincount(flat_tokens, minlength=vocab_size)
        appeared_mask = token_counts > 0
        if logits.dim() == 1:
            logits[appeared_mask] /= penalty
        else:
            logits[..., appeared_mask] /= penalty

    return logits


def sample_next_token(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> torch.Tensor:
    """
    Sample one token from logits with repetition penalty, top-k, and top-p.

    Shared helper used by :meth:`Enigma.generate` and
    :meth:`Enigma.generate_stream` so the sampling logic lives in one place.

    Args:
        logits: Raw logits for the last position [batch, vocab_size].
        generated_tokens: All tokens generated so far [batch, seq_len].
        temperature: Sampling temperature (>0).
        top_k: Keep only the top-k highest-probability tokens (0 = disabled).
        top_p: Nucleus sampling threshold (<1.0 to enable).
        repetition_penalty: Penalty for previously seen tokens (1.0 = off).

    Returns:
        Sampled token IDs [batch, 1].
    """
    next_logits = logits / temperature

    # --- Repetition penalty (reuses standalone function) ---
    if repetition_penalty != 1.0:
        next_logits = apply_repetition_penalty(
            next_logits, generated_tokens, repetition_penalty
        )

    # --- Top-k filtering ---
    if top_k > 0:
        v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
        next_logits[next_logits < v[:, [-1]]] = float('-inf')

    # --- Top-p (nucleus) filtering ---
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
        cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumsum > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False
        indices_to_remove = mask.scatter(1, sorted_idx, mask)
        next_logits[indices_to_remove] = float('-inf')

    # --- Sample ---
    probs = F.softmax(next_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# =============================================================================
# 🔧 HARDWARE DETECTION HELPERS
# =============================================================================

def detect_hardware() -> dict[str, Any]:
    """
    Detect hardware capabilities for model configuration.
    
    Returns dict with: total_ram_gb, gpu_vram_gb, is_raspberry_pi, is_arm, etc.
    """
    try:
        import dataclasses
        from .hardware_detection import detect_hardware as _detect
        profile = _detect()
        return dataclasses.asdict(profile)
    except ImportError:
        # Fallback basic detection
        ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3) if hasattr(os, 'sysconf') else 4.0
        return {
            "total_ram_gb": ram_gb,
            "is_raspberry_pi": False,
            "has_cuda": torch.cuda.is_available(),
            "recommended_model_size": "small"
        }


def recommend_model_size(hardware: Optional[dict[str, Any]] = None) -> str:
    """
    Recommend optimal model size based on hardware.
    
    Args:
        hardware: Hardware profile dict (from detect_hardware). If None, auto-detects.
    
    Returns:
        Recommended preset name (e.g., "pi_5", "small", "medium")
    """
    if hardware is None:
        hardware = detect_hardware()

    return hardware.get("recommended_model_size", "small")


def estimate_memory_usage(size: str, quantization: str = "none") -> dict[str, float]:
    """
    Estimate RAM/VRAM requirements for a model configuration.
    
    Args:
        size: Model size preset name
        quantization: Quantization type
    
    Returns:
        Dict with model_size_mb, inference_ram_mb, training_ram_mb
    """
    try:
        from .hardware_detection import estimate_memory_usage as _estimate
        return _estimate(size, quantization)
    except ImportError:
        # Fallback estimation
        param_counts = {
            "pi_zero": 0.5, "pi_4": 3, "pi_5": 8, "nano": 1, "micro": 2,
            "tiny": 5, "mini": 10, "small": 27, "medium": 85, "large": 200
        }
        params_m = param_counts.get(size, 27)
        multiplier = {"none": 4, "dynamic": 1.5, "int8": 1, "int4": 0.5}.get(quantization, 4)
        model_mb = params_m * multiplier
        return {
            "model_size_mb": model_mb,
            "inference_ram_mb": model_mb * 2.5,
            "training_ram_mb": model_mb * 5
        }
