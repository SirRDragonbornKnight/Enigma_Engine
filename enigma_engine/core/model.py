"""
================================================================================
                    CHAPTER 1: THE ENIGMA - WHERE MINDS ARE BORN
================================================================================

    "In the depths of silicon and mathematics, something awakens..."

Welcome, brave explorer. You have reached the most sacred place in all of
Enigma AI Engine - the ENIGMA itself. This is where artificial minds are literally 
CONSTRUCTED, layer by layer, neuron by neuron.

WHY THIS FILE MATTERS:
    Every word your AI speaks, every thought it processes, every creative
    idea it generates - ALL of it flows through this file. The Enigma class
    is the living brain. Without it, Enigma AI Engine is just empty code.

THE JOURNEY AHEAD:
    1. ForgeConfig     - The blueprint (how big? how smart?)
    2. RMSNorm         - The stabilizer (keeps gradients healthy)
    3. Attention       - The memory (what to focus on?)
    4. SwiGLU          - The activation (fire or not fire?)
    5. TransformerBlock - One layer of thinking
    6. Enigma          - The complete brain!

MAIN QUEST: Create an AI brain that can process language and generate text.

DIFFICULTY: EXPERT - This is the most complex file in Enigma AI Engine. Take your time.

                    ┌──────────────────────────────────────┐
                    │  Your words go in...                 │
                    │         ↓                            │
                    │  [Embedding]  "Words become vectors" │
                    │         ↓                            │
                    │  [Transformer x N] "Deep thinking"   │
                    │         ↓                            │
                    │  [Output Head]  "Pick next word"     │
                    │         ↓                            │
                    │  ...AI response comes out!           │
                    └──────────────────────────────────────┘

SIDE QUESTS UNLOCKED (2026 Features):
    + Universal Loading  - Import models from HuggingFace, GGUF, ONNX
    + RoPE Scaling       - Handle longer conversations (up to 8x context!)  
    + LoRA Adapters      - Fine-tune without retraining everything
    + Speculative Decode - Generate 2-4x faster with draft models
    + Flash Attention    - GPU speedup wizardry (requires CUDA)

CHOOSE YOUR FIGHTER (Model Sizes):
    | Size    | Power     | Device          | "Class"          |
    |---------|-----------|-----------------|------------------|
    | pi_zero | ~500K     | Pi Zero         | "The Apprentice" |
    | nano    | ~1M       | Embedded        | "The Scout"      |
    | tiny    | ~5M       | Raspberry Pi    | "The Traveler"   |
    | small   | ~27M      | Desktop         | "The Knight"     |
    | medium  | ~85M      | Gaming PC       | "The Wizard"     |
    | large   | ~200M     | Workstation     | "The Archmage"   |
    │ xl         │ ~600M    │ Multi-GPU                      │
    │ xxl        │ ~1.5B    │ Cloud/Datacenter               │
    │ omega      │ ~70B+    │ Research frontier              │
    └────────────┴──────────┴────────────────────────────────┘

🔗 CONNECTED FILES:
    → USES:      enigma_engine/config/ (CONFIG settings)
    ← USED BY:   enigma_engine/core/inference.py (EnigmaEngine loads this)
    ← USED BY:   enigma_engine/core/training.py (trains this model)
    ← USED BY:   enigma_engine/modules/registry.py (ModelModule wraps this)
    → SEE ALSO:  UNIVERSAL_MODEL_GUIDE.md (detailed feature guide)

📖 BASIC USAGE:
    from enigma_engine.core.model import create_model, Enigma, ForgeConfig
    
    # Simple preset
    model = create_model('small')
    
    # Custom config
    config = ForgeConfig(vocab_size=8000, dim=512, n_layers=8)
    model = Enigma(config=config)

📖 UNIVERSAL FEATURES USAGE:
    # Load from any format
    model = Enigma.from_any("model.gguf")
    model = Enigma.from_huggingface("microsoft/phi-2")
    
    # Extended context with RoPE scaling
    config = ForgeConfig(
        ..., 
        max_seq_len=8192,
        rope_scaling_type="dynamic",
        rope_scaling_factor=4.0
    )
    
    # Multi-modal
    logits = model.forward_multimodal(
        input_ids=text_ids,
        vision_features=vision_output
    )
    
    # LoRA adapters
    model.load_lora("adapter.pth")
    model.merge_lora()
    
    # Speculative decoding
    draft = create_model('tiny')
    model.enable_speculative_decoding(draft)
    output = model.generate_speculative(input_ids)

📖 SEE ALSO:
    • enigma_engine/core/inference.py - To GENERATE text with this model
    • enigma_engine/core/training.py  - To TRAIN this model
    • enigma_engine/core/tokenizer.py - Converts text ↔ numbers
    • UNIVERSAL_MODEL_GUIDE.md   - Comprehensive feature guide
"""
import json
import logging
import math
import os
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CONFIG

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FLASH ATTENTION: Optional high-performance attention (2-4x faster)
# ─────────────────────────────────────────────────────────────────────────────
# Requires: pip install flash-attn (CUDA only, Ampere+ GPU recommended)
# Falls back silently to standard attention if not available.
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
    logger.info("Flash Attention available - will use for fp16/bf16 CUDA tensors")
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_func = None  # type: ignore

MAX_LEN = CONFIG.get("max_len", 1024)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL MODEL REGISTRY (Thread-Safe)
# ─────────────────────────────────────────────────────────────────────────────
# Registry of all loaded model instances. Uses a lock to ensure thread safety
# when multiple threads access models concurrently (e.g., GUI + API server).
import threading

_LOADED_MODELS: dict[str, 'Enigma'] = {}
_MODELS_LOCK = threading.RLock()  # RLock allows re-entrant locking


def get_running_models() -> dict[str, 'Enigma']:
    """Get a copy of all loaded model instances (thread-safe)."""
    with _MODELS_LOCK:
        return _LOADED_MODELS.copy()


def is_model_loaded(name: str) -> bool:
    """Check if a model is loaded by name (thread-safe)."""
    with _MODELS_LOCK:
        return name in _LOADED_MODELS


def register_model(name: str, model: 'Enigma') -> None:
    """Register a model instance (thread-safe)."""
    with _MODELS_LOCK:
        _LOADED_MODELS[name] = model
        logger.debug(f"Registered model: {name}")


def unregister_model(name: str) -> Optional['Enigma']:
    """Unregister a model and return it if found (thread-safe)."""
    with _MODELS_LOCK:
        model = _LOADED_MODELS.pop(name, None)
        if model is not None:
            logger.debug(f"Unregistered model: {name}")
        return model


def get_model(name: str) -> Optional['Enigma']:
    """Get a specific model by name (thread-safe)."""
    with _MODELS_LOCK:
        return _LOADED_MODELS.get(name)


# =============================================================================
# ⚙️ CONFIGURATION - Model Settings
# =============================================================================
# ForgeConfig holds ALL the settings that define a model's architecture.
# Think of it as a blueprint - same settings = same model structure.

@dataclass
class ForgeConfig:
    """
    Model configuration with sensible defaults.
    
    📖 WHAT EACH SETTING DOES:
    
    CORE ARCHITECTURE:
    ┌────────────────────────────────────────────────────────────────────────┐
    │ vocab_size  │ How many unique tokens the model knows (like vocabulary)│
    │ dim         │ Hidden dimension - the "width" of neural pathways       │
    │ n_layers    │ Number of transformer blocks (depth of the network)     │
    │ n_heads     │ Attention heads (parallel attention computations)       │
    │ n_kv_heads  │ Key/Value heads for GQA (memory optimization)          │
    │ hidden_dim  │ FFN hidden size (typically 4x dim for expansion)       │
    └────────────────────────────────────────────────────────────────────────┘
    
    LIMITS & REGULARIZATION:
    ┌────────────────────────────────────────────────────────────────────────┐
    │ max_seq_len │ Maximum tokens in one sequence (context window)         │
    │ dropout     │ Randomly zero neurons during training (prevents overfit)│
    └────────────────────────────────────────────────────────────────────────┘
    
    ARCHITECTURE FLAGS (modern transformer tricks):
    ┌────────────────────────────────────────────────────────────────────────┐
    │ use_rope    │ Rotary Position Embeddings (better position awareness)  │
    │ use_rms_norm│ RMSNorm instead of LayerNorm (faster, equally good)    │
    │ use_swiglu  │ SwiGLU activation (better than ReLU/GELU)              │
    │ use_bias    │ Add bias terms (usually False in modern models)        │
    └────────────────────────────────────────────────────────────────────────┘
    
    UNIVERSAL MODEL ENHANCEMENTS:
    ┌────────────────────────────────────────────────────────────────────────┐
    │ rope_scaling_type   │ RoPE scaling for extended context              │
    │ rope_scaling_factor │ Scaling multiplier for context extension       │
    │ use_moe            │ Enable Mixture of Experts architecture          │
    │ num_experts        │ Number of expert networks (MoE)                 │
    │ num_experts_per_tok│ Experts activated per token (MoE)               │
    │ sliding_window     │ Sliding window attention length                 │
    │ use_paged_attn     │ Enable paged attention for better memory        │
    └────────────────────────────────────────────────────────────────────────┘
    """
    # ─────────────────────────────────────────────────────────────────────────
    # CORE PARAMETERS
    # ─────────────────────────────────────────────────────────────────────────
    vocab_size: int = 8000      # Size of vocabulary (tokenizer determines this)
    dim: int = 512              # Model hidden dimension (larger = smarter but slower)
    n_layers: int = 8           # Number of transformer layers (deeper = more capable)
    n_heads: int = 8            # Attention heads (more = better pattern recognition)
    n_kv_heads: Optional[int] = None  # KV heads for GQA (None = same as n_heads)
    hidden_dim: Optional[int] = None  # FFN dimension (None = auto-calculate)
    max_seq_len: int = 1024     # Maximum sequence length (context window)
    dropout: float = 0.1        # Dropout rate (0.1 = 10% neurons randomly zeroed)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ARCHITECTURE FLAGS - Modern transformer improvements
    # ─────────────────────────────────────────────────────────────────────────
    use_rope: bool = True       # RoPE: Better position encoding than absolute
    use_rms_norm: bool = True   # RMSNorm: Faster normalization, works just as well
    use_swiglu: bool = True     # SwiGLU: Superior activation function
    use_bias: bool = False      # Bias: Usually disabled in modern transformers
    rope_theta: float = 10000.0 # RoPE base frequency (higher = longer context)

    # ─────────────────────────────────────────────────────────────────────────
    # ROPE SCALING - Extended context support
    # ─────────────────────────────────────────────────────────────────────────
    rope_scaling_type: Optional[str] = None  # "linear", "dynamic", "yarn", None
    rope_scaling_factor: float = 1.0  # Context extension multiplier (>1.0 extends)

    # ─────────────────────────────────────────────────────────────────────────
    # MIXTURE OF EXPERTS (MoE)
    # ─────────────────────────────────────────────────────────────────────────
    use_moe: bool = False       # Enable MoE architecture
    num_experts: int = 8        # Number of expert networks
    num_experts_per_token: int = 2  # Top-k experts to activate per token
    moe_load_balancing: float = 0.01  # Load balancing loss weight

    # ─────────────────────────────────────────────────────────────────────────
    # ENHANCED KV-CACHE
    # ─────────────────────────────────────────────────────────────────────────
    sliding_window: Optional[int] = None  # Sliding window attention length
    use_paged_attn: bool = False  # Enable paged attention (better memory)
    kv_cache_dtype: Optional[str] = None  # "int8", "fp16", None (same as model)

    # ─────────────────────────────────────────────────────────────────────────
    # MEMORY OPTIMIZATION
    # ─────────────────────────────────────────────────────────────────────────
    use_gradient_checkpointing: bool = False  # Trade compute for memory during training

    # ─────────────────────────────────────────────────────────────────────────
    # MULTI-MODAL SUPPORT
    # ─────────────────────────────────────────────────────────────────────────
    vision_hidden_size: Optional[int] = None  # Vision encoder dimension
    audio_hidden_size: Optional[int] = None   # Audio encoder dimension
    
    # ─────────────────────────────────────────────────────────────────────────
    # LEGACY ALIASES - For backwards compatibility
    # ─────────────────────────────────────────────────────────────────────────
    depth: Optional[int] = None      # Old name for n_layers
    heads: Optional[int] = None      # Old name for n_heads
    max_len: Optional[int] = None    # Old name for max_seq_len
    embed_dim: Optional[int] = None  # Old name for dim

    # Track if config is frozen (immutable after creation)
    _frozen: bool = False
    
    def __post_init__(self):
        """
        Post-initialization: validate and set computed defaults.
        Called automatically after __init__ (dataclass magic).
        """
        # ─────────────────────────────────────────────────────────────────────
        # MAP LEGACY NAMES: Support old config files
        # ─────────────────────────────────────────────────────────────────────
        if self.depth:
            self.n_layers = self.depth
        if self.heads:
            self.n_heads = self.heads
        if self.max_len:
            self.max_seq_len = self.max_len
        if self.embed_dim:
            self.dim = self.embed_dim

        # ─────────────────────────────────────────────────────────────────────
        # AUTO-CALCULATE KV HEADS: Default to same as n_heads (no GQA)
        # ─────────────────────────────────────────────────────────────────────
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # ─────────────────────────────────────────────────────────────────────
        # AUTO-CALCULATE HIDDEN DIM: The "expansion" in feed-forward layers
        # ─────────────────────────────────────────────────────────────────────
        # Standard: hidden_dim = 4 * dim (4x expansion)
        # SwiGLU: Needs 2/3 of that because it has 3 matrices instead of 2
        # MoE: May need adjustment based on num_experts
        # We also round up to nearest 64 for GPU efficiency
        if self.hidden_dim is None:
            if self.use_swiglu:
                # SwiGLU formula: 2/3 * (4 * dim), rounded to multiple of 64
                self.hidden_dim = int(2 * (4 * self.dim) / 3)
                self.hidden_dim = 64 * ((self.hidden_dim + 63) // 64)
            else:
                self.hidden_dim = 4 * self.dim

        # ─────────────────────────────────────────────────────────────────────
        # VALIDATION: Catch configuration errors early
        # ─────────────────────────────────────────────────────────────────────
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        
        if not (0 <= self.dropout <= 1):
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")
        
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        
        # dim must be divisible by n_heads (each head gets dim/n_heads dimensions)
        if self.dim % self.n_heads != 0:
            # Calculate helpful suggestions
            head_dim = self.dim // self.n_heads
            suggested_dim = self.n_heads * (head_dim + 1)
            suggested_heads = self.dim // (head_dim + 1) if head_dim + 1 > 0 else self.n_heads
            raise ValueError(
                f"n_heads ({self.n_heads}) must divide evenly into dim ({self.dim}). "
                f"Got remainder: {self.dim % self.n_heads}. "
                f"Try: dim={suggested_dim} (with {self.n_heads} heads) or "
                f"n_heads={suggested_heads} (with dim={self.dim})"
            )
        
        # n_heads must be divisible by n_kv_heads (for GQA grouping)
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_kv_heads ({self.n_kv_heads}) must divide evenly into n_heads ({self.n_heads}). "
                f"Got remainder: {self.n_heads % self.n_kv_heads}"
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # VALIDATE NEW FEATURES
        # ─────────────────────────────────────────────────────────────────────
        # RoPE scaling validation
        if self.rope_scaling_type is not None:
            valid_scaling = {"linear", "dynamic", "yarn"}
            if self.rope_scaling_type not in valid_scaling:
                raise ValueError(
                    f"rope_scaling_type must be one of {valid_scaling}, "
                    f"got {self.rope_scaling_type}"
                )
            if self.rope_scaling_factor <= 0:
                raise ValueError(
                    f"rope_scaling_factor must be positive, got {self.rope_scaling_factor}"
                )
        
        # MoE validation
        if self.use_moe:
            if self.num_experts <= 0:
                raise ValueError(f"num_experts must be positive, got {self.num_experts}")
            if self.num_experts_per_token <= 0 or self.num_experts_per_token > self.num_experts:
                raise ValueError(
                    f"num_experts_per_token must be in (0, {self.num_experts}], "
                    f"got {self.num_experts_per_token}"
                )

    def validate(self) -> bool:
        """
        Re-run validation on the config.
        
        Useful after manually modifying config attributes to ensure
        the configuration is still valid.
        
        Returns:
            True if valid
            
        Raises:
            ValueError: If any validation fails
        """
        # Re-run __post_init__ validation logic
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if not (0 <= self.dropout <= 1):
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.dim % self.n_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must divide evenly into dim ({self.dim})")
        if self.n_kv_heads and self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_kv_heads ({self.n_kv_heads}) must divide evenly into n_heads ({self.n_heads})")
        return True
    
    def freeze(self) -> 'ForgeConfig':
        """
        Freeze the config to prevent further modifications.
        
        Once frozen, any attempt to modify attributes will raise an error.
        This is useful for ensuring config immutability after model creation.
        
        Returns:
            self (for chaining)
        """
        object.__setattr__(self, '_frozen', True)
        return self
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to enforce frozen state."""
        if getattr(self, '_frozen', False) and name != '_frozen':
            raise AttributeError(
                f"Cannot modify frozen ForgeConfig. "
                f"Create a new config with the desired changes instead."
            )
        object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            'vocab_size': self.vocab_size,
            'dim': self.dim,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'n_kv_heads': self.n_kv_heads,
            'hidden_dim': self.hidden_dim,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout,
            'use_rope': self.use_rope,
            'use_rms_norm': self.use_rms_norm,
            'use_swiglu': self.use_swiglu,
            'use_bias': self.use_bias,
            'rope_theta': self.rope_theta,
            # New parameters
            'rope_scaling_type': self.rope_scaling_type,
            'rope_scaling_factor': self.rope_scaling_factor,
            'use_moe': self.use_moe,
            'num_experts': self.num_experts,
            'num_experts_per_token': self.num_experts_per_token,
            'moe_load_balancing': self.moe_load_balancing,
            'sliding_window': self.sliding_window,
            'use_paged_attn': self.use_paged_attn,
            'kv_cache_dtype': self.kv_cache_dtype,
            'use_gradient_checkpointing': self.use_gradient_checkpointing,
            'vision_hidden_size': self.vision_hidden_size,
            'audio_hidden_size': self.audio_hidden_size,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'ForgeConfig':
        known = {
            'vocab_size', 'dim', 'n_layers', 'n_heads', 'n_kv_heads',
            'hidden_dim', 'max_seq_len', 'dropout', 'use_rope', 'use_rms_norm',
            'use_swiglu', 'use_bias', 'rope_theta', 'depth', 'heads',
            'max_len', 'embed_dim',
            # New parameters
            'rope_scaling_type', 'rope_scaling_factor', 'use_moe', 'num_experts',
            'num_experts_per_token', 'moe_load_balancing', 'sliding_window',
            'use_paged_attn', 'kv_cache_dtype', 'use_gradient_checkpointing',
            'vision_hidden_size', 'audio_hidden_size'
        }
        return cls(**{k: v for k, v in d.items() if k in known})


# =============================================================================
# ⚡ QUANTIZATION CONFIG - Memory-Efficient Model Deployment
# =============================================================================

@dataclass
class QuantizationConfig:
    """
    Configuration for model quantization.
    
    📖 WHAT THIS DOES:
    Quantization reduces model precision to save memory and speed up inference.
    
    📐 QUANTIZATION TYPES:
    ┌────────────┬────────────────────────────────────────────────────────────┐
    │ Type       │ Description                                                │
    ├────────────┼────────────────────────────────────────────────────────────┤
    │ none       │ Full FP32 precision (default, most accurate)              │
    │ dynamic    │ Dynamic INT8 quantization (good for CPU)                  │
    │ int8       │ Static INT8 quantization (requires calibration)           │
    │ int4       │ 4-bit quantization (smallest, some quality loss)          │
    └────────────┴────────────────────────────────────────────────────────────┘
    
    ⚡ MEMORY SAVINGS:
    - FP32: 4 bytes/param (baseline)
    - FP16: 2 bytes/param (50% savings)
    - INT8: 1 byte/param (75% savings)
    - INT4: 0.5 bytes/param (87.5% savings)
    
    🍓 PI RECOMMENDATIONS:
    - Pi Zero: int4 quantization (fits in 512MB RAM)
    - Pi 4 (4GB): int8 quantization
    - Pi 5 (8GB): dynamic quantization
    """
    mode: str = "none"  # "none", "dynamic", "int8", "int4"
    
    # Static quantization options
    calibration_data: Optional[list[torch.Tensor]] = None
    num_calibration_batches: int = 100
    
    # Dynamic quantization options
    dtype: Optional[torch.dtype] = None  # torch.qint8 for dynamic
    
    # Which layers to quantize
    quantize_linear: bool = True
    quantize_embedding: bool = False  # Usually keep embeddings in FP32
    
    # INT4 specific
    group_size: int = 128  # For grouped quantization
    
    def __post_init__(self):
        valid_modes = {"none", "dynamic", "int8", "int4"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid quantization mode: {self.mode}. Valid: {valid_modes}")


# =============================================================================
# 📊 MODEL PRESETS - From Raspberry Pi to Server Farm!
# =============================================================================
# These presets make it easy to create models of different sizes.
# Just pick a preset name and the config is ready to go!
#
# HOW TO CHOOSE A PRESET:
#   1. What hardware do you have? (RAM, GPU VRAM)
#   2. What quality do you need?
#   3. How fast does it need to be?
#
# ROUGH GUIDELINES:
#   • 4GB RAM/VRAM → tiny or mini
#   • 8GB VRAM → small or medium  
#   • 16GB VRAM → large or xl
#   • 24GB+ VRAM → xxl or larger
#   • Multi-GPU → huge, giant, etc.

MODEL_PRESETS = {
    # ─────────────────────────────────────────────────────────────────────────
    # RASPBERRY PI OPTIMIZED (~500K-8M params) - Specifically tuned for Pi
    # ─────────────────────────────────────────────────────────────────────────
    # Pi Zero 2W (512MB RAM): Ultra-minimal footprint
    'pi_zero': ForgeConfig(dim=64, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=256, dropout=0.0),
    # Pi 4 (4GB RAM): Good balance of capability and memory
    'pi_4': ForgeConfig(dim=192, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=512, dropout=0.05),
    # Pi 5 (8GB RAM): Maximum Pi capability
    'pi_5': ForgeConfig(dim=256, n_layers=6, n_heads=8, n_kv_heads=4, max_seq_len=1024, dropout=0.05),

    # ─────────────────────────────────────────────────────────────────────────
    # EMBEDDED / IoT (~1-2M params) - For microcontrollers and tiny devices
    # ─────────────────────────────────────────────────────────────────────────
    'nano': ForgeConfig(dim=128, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=256),
    'micro': ForgeConfig(dim=192, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=384),

    # ─────────────────────────────────────────────────────────────────────────
    # EDGE / Raspberry Pi (~5-15M params) - For single-board computers
    # ─────────────────────────────────────────────────────────────────────────
    'tiny': ForgeConfig(dim=256, n_layers=6, n_heads=8, n_kv_heads=4, max_seq_len=512),
    'mini': ForgeConfig(dim=384, n_layers=6, n_heads=6, n_kv_heads=3, max_seq_len=512),

    # ─────────────────────────────────────────────────────────────────────────
    # CONSUMER GPU (~27-85M params) - RTX 2080 to RTX 3070
    # ─────────────────────────────────────────────────────────────────────────
    'small': ForgeConfig(dim=512, n_layers=8, n_heads=8, n_kv_heads=4, max_seq_len=1024),
    'medium': ForgeConfig(dim=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=2048),
    'base': ForgeConfig(dim=896, n_layers=14, n_heads=14, n_kv_heads=2, max_seq_len=2048),

    # ─────────────────────────────────────────────────────────────────────────
    # PROSUMER GPU (~200M-600M params) - RTX 3080, RTX 4080, RTX 4090
    # ─────────────────────────────────────────────────────────────────────────
    'large': ForgeConfig(dim=1024, n_layers=16, n_heads=16, n_kv_heads=4, max_seq_len=4096),
    'xl': ForgeConfig(dim=1536, n_layers=24, n_heads=24, n_kv_heads=6, max_seq_len=4096, dropout=0.05),

    # ─────────────────────────────────────────────────────────────────────────
    # MULTI-GPU / SERVER (~1B-3B params) - 2-4x A100, workstation setups
    # ─────────────────────────────────────────────────────────────────────────
    'xxl': ForgeConfig(dim=2048, n_layers=32, n_heads=32, n_kv_heads=8, max_seq_len=8192, dropout=0.05),
    'huge': ForgeConfig(dim=2560, n_layers=40, n_heads=40, n_kv_heads=8, max_seq_len=8192, dropout=0.05),

    # ─────────────────────────────────────────────────────────────────────────
    # DATACENTER / CLOUD (~7B-13B params) - 8x A100, cloud instances
    # ─────────────────────────────────────────────────────────────────────────
    'giant': ForgeConfig(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, max_seq_len=8192, dropout=0.05),
    'colossal': ForgeConfig(dim=4096, n_layers=48, n_heads=32, n_kv_heads=8, max_seq_len=16384, dropout=0.05),

    # ─────────────────────────────────────────────────────────────────────────
    # MAXIMUM SCALE (~30B+ params) - Full datacenter, research frontier
    # ─────────────────────────────────────────────────────────────────────────
    'titan': ForgeConfig(dim=6144, n_layers=48, n_heads=48, n_kv_heads=12, max_seq_len=16384, dropout=0.05),
    'omega': ForgeConfig(dim=8192, n_layers=64, n_heads=64, n_kv_heads=16, max_seq_len=32768, dropout=0.05),
}

# Human-readable descriptions
MODEL_DESCRIPTIONS = {
    # Pi-optimized presets
    'pi_zero': "Pi Zero (~500K) - Raspberry Pi Zero 2W, minimal responses",
    'pi_4': "Pi 4 (~3M) - Raspberry Pi 4 (4GB), balanced performance",
    'pi_5': "Pi 5 (~8M) - Raspberry Pi 5 (8GB), best Pi experience",
    # Standard presets
    'nano': "Minimal (~1M) - Microcontrollers, basic responses",
    'micro': "Tiny (~2M) - IoT devices, simple tasks",
    'tiny': "Small (~5M) - Raspberry Pi, edge devices",
    'mini': "Compact (~10M) - Mobile, low-power devices",
    'small': "Standard (~27M) - Entry GPU, good learning",
    'medium': "Capable (~85M) - Mid-range GPU, solid results",
    'base': "Balanced (~125M) - Good GPU, versatile",
    'large': "Powerful (~200M) - RTX 3080+, high quality",
    'xl': "Advanced (~600M) - RTX 4090, excellent results",
    'xxl': "Massive (~1.5B) - Multi-GPU, near-production",
    'huge': "Enterprise (~3B) - Server GPU, production ready",
    'giant': "Datacenter (~7B) - Multi-node, commercial grade",
    'colossal': "Cloud (~13B) - Distributed, competitive",
    'titan': "Maximum (~30B) - Full datacenter, state-of-art",
    'omega': "Ultimate (~70B+) - Cluster, research frontier",
}


def get_preset(name: str, vocab_size: int = 8000) -> ForgeConfig:
    """Get a preset configuration."""
    if name not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(MODEL_PRESETS.keys())}")

    # Create a copy with vocab_size
    preset = MODEL_PRESETS[name]
    return ForgeConfig(
        vocab_size=vocab_size,
        dim=preset.dim,
        n_layers=preset.n_layers,
        n_heads=preset.n_heads,
        n_kv_heads=preset.n_kv_heads,
        max_seq_len=preset.max_seq_len,
        dropout=preset.dropout,
    )


def estimate_parameters(config: ForgeConfig) -> int:
    """Estimate number of parameters for a config."""
    # Embedding: vocab_size * dim
    embed = config.vocab_size * config.dim

    # Per layer: attention + FFN
    # Attention: 4 * dim * dim (Q, K, V, O)
    # FFN: 3 * dim * hidden_dim (SwiGLU has 3 matrices)
    per_layer = (4 * config.dim * config.dim +
                 3 * config.dim * (config.hidden_dim or 4 * config.dim))

    # Total
    return embed + (per_layer * config.n_layers) + config.dim


def list_presets() -> dict:
    """List all presets with descriptions and estimated parameters."""
    result = {}
    for name, config in MODEL_PRESETS.items():
        # IMPORTANT: Create a copy to avoid mutating the global preset!
        # Without copy, setting vocab_size corrupts the shared config object.
        import copy
        config_copy = copy.deepcopy(config)
        config_copy.vocab_size = 32000  # Standard for estimation
        result[name] = {
            'description': MODEL_DESCRIPTIONS.get(name, ""),
            'estimated_params': estimate_parameters(config_copy),
            'dim': config_copy.dim,
            'layers': config_copy.n_layers,
            'heads': config_copy.n_heads,
            'max_seq_len': config_copy.max_seq_len,
        }
    return result


# =============================================================================
# 🧱 MODEL COMPONENTS - The Building Blocks
# =============================================================================
# These are the LEGO pieces that build the full transformer.
# Each class is a specific neural network layer with a special purpose.

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization - faster than LayerNorm!
    
    📖 WHAT THIS DOES:
    Normalizes the input to have consistent scale. Like adjusting volume
    on speakers so nothing is too loud or too quiet.
    
    📐 THE MATH (simplified):
    1. Calculate RMS: sqrt(mean(x²))
    2. Divide x by RMS (now values are normalized)
    3. Multiply by learned weight (model learns optimal scale)
    
    💡 WHY RMSNorm INSTEAD OF LAYERNORM?
    LayerNorm: Subtracts mean, divides by std (2 stats to compute)
    RMSNorm: Just divides by RMS (1 stat to compute)
    Result: Same quality, ~10% faster!
    
    🔗 USED BY:
      ← TransformerBlock uses this before attention and FFN
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: Dimension of input (should match model dim)
            eps: Small number to prevent division by zero
        """
        super().__init__()
        self.eps = eps
        # Learnable scale parameter - model learns optimal normalization
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor.
        
        x: [batch, sequence, dim] → normalized: [batch, sequence, dim]
        """
        # Calculate Root Mean Square: sqrt(mean(x²) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and apply learned scale
        return x / rms * self.weight


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


# =============================================================================
# 🌀 ROTARY POSITION EMBEDDINGS (RoPE) - How the model knows word order
# =============================================================================
# Without position info, "dog bites man" = "man bites dog" to the model!
# RoPE encodes position by ROTATING the vectors - elegant and effective.

def precompute_rope_frequencies(
    dim: int, 
    max_seq_len: int, 
    theta: float = 10000.0,
    scaling_type: Optional[str] = None,
    scaling_factor: float = 1.0
) -> torch.Tensor:
    """
    Precompute RoPE frequencies for all positions with optional scaling.
    
    📖 WHAT THIS DOES:
    Creates a table of rotation angles for each position and dimension.
    These rotations encode "position 0", "position 1", etc.
    With scaling, extends context length beyond training length.
    
    📐 THE MATH:
    For dimension pair i, frequency = 1 / (theta^(2i/dim))
    For position p, angle = p * frequency
    
    🎯 ROPE SCALING:
    - linear: freqs = freqs / scaling_factor (simple compression)
    - dynamic: Adaptive NTK-aware scaling (better quality)
    - yarn: Yet another RoPE extension (best for very long contexts)
    
    💡 WHY THIS WORKS:
    - Different dimensions get different rotation speeds
    - Position 5 at dim 0 rotates differently than position 5 at dim 10
    - Model can learn to "read" these rotations to understand order
    - Scaling lets model handle longer contexts than it was trained on
    
    Args:
        dim: Dimension per head (must be even)
        max_seq_len: Maximum sequence length
        theta: Base frequency (higher = better long context)
        scaling_type: Type of scaling ("linear", "dynamic", "yarn", None)
        scaling_factor: Scaling multiplier (>1.0 extends context)
    
    Returns:
        Complex tensor of shape [max_seq_len, dim/2] with rotation values
    """
    # Calculate base frequencies: lower dimensions rotate faster
    # freqs[i] = 1 / (theta^(2i/dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    
    # Apply RoPE scaling if specified
    if scaling_type == "linear":
        # Linear scaling: compress frequencies uniformly
        freqs = freqs / scaling_factor
        logger.debug(f"Applied linear RoPE scaling (factor={scaling_factor})")
        
    elif scaling_type == "dynamic":
        # Dynamic NTK-aware scaling: adjust theta based on extension
        # Better quality than linear for moderate extensions
        alpha = scaling_factor
        # Adjust base frequency with NTK-aware interpolation
        adjusted_theta = theta * (alpha ** (dim / (dim - 2)))
        freqs = 1.0 / (adjusted_theta ** (torch.arange(0, dim, 2).float() / dim))
        logger.debug(f"Applied dynamic NTK RoPE scaling (factor={scaling_factor})")
        
    elif scaling_type == "yarn":
        # YaRN (Yet another RoPE extensioN): Best for very long contexts
        # Uses attention-aware scaling with ramp function
        alpha = scaling_factor
        # YaRN applies different scaling to different frequency bands
        beta_fast = 32  # Low frequency threshold
        beta_slow = 1   # High frequency threshold
        
        # Compute frequency-dependent scaling
        dim_indices = torch.arange(0, dim, 2).float()
        # Ramp function: smoothly transition between fast and slow scaling
        ramp = (dim_indices / dim - beta_slow) / (beta_fast / dim - beta_slow)
        ramp = torch.clamp(ramp, 0, 1)
        
        # Apply scaled freqs with ramp
        freqs_scaled = freqs / alpha
        freqs = freqs_scaled * ramp + freqs * (1 - ramp)
        logger.debug(f"Applied YaRN RoPE scaling (factor={scaling_factor})")
    
    # Create position indices: [0, 1, 2, ..., max_seq_len-1]
    positions = torch.arange(max_seq_len)
    
    # Outer product: angles[pos, dim] = pos * freq[dim]
    angles = torch.outer(positions, freqs)
    
    # Convert to complex numbers for rotation: e^(i*angle) = cos(angle) + i*sin(angle)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rotary_embedding(
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0) -> torch.Tensor:
    """
    Apply rotary embeddings to Q and K tensors.
    
    📖 WHAT THIS DOES:
    Rotates the query/key vectors based on their position.
    This lets the model know "this word is at position 5" vs "position 10".
    
    📐 HOW IT WORKS:
    1. Treat pairs of dimensions as complex numbers
    2. Multiply by rotation (complex multiplication = rotation!)
    3. Convert back to real numbers
    
    Args:
        x: Input tensor [batch, seq, heads, dim]
        freqs_cis: Precomputed rotation frequencies
        start_pos: Starting position (for KV-cache continuation)
    
    Returns:
        Rotated tensor, same shape as input
    """
    seq_len = x.shape[1]
    # Get the right slice of frequencies for our positions
    freqs = freqs_cis[start_pos:start_pos + seq_len]
    
    # Reshape x to treat pairs of dims as complex: [batch, seq, heads, dim/2, 2]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # Add batch and head dimensions to freqs for broadcasting
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    
    # Complex multiplication = rotation!
    x_rotated = x_complex * freqs
    
    # Convert back to real numbers and original shape
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


class Attention(nn.Module):
    """
    Multi-Head Attention with Grouped Query Attention (GQA).
    
    📖 WHAT THIS DOES:
    Attention is how the model "looks at" different parts of the input.
    "The cat sat on the mat" - when processing "sat", attention lets
    the model look back at "cat" to know WHO sat.
    
    📐 THE MATH (simplified):
    1. Create Query (Q), Key (K), Value (V) from input
    2. Attention scores = Q @ K.T / sqrt(dim)  (which words to look at?)
    3. Softmax → probabilities (normalize scores)
    4. Output = scores @ V  (weighted combination of values)
    
    ⚡ GROUPED QUERY ATTENTION (GQA):
    Normal: Each head has its own K and V (memory hungry!)
    GQA: Multiple Q heads share the same K,V (saves 2-4x memory!)
    
    Example: 8 Q heads, 2 KV heads → 4 Q heads share each KV head
    
    💾 KV-CACHE:
    During generation, we only add ONE new token at a time.
    Instead of recomputing K,V for all previous tokens, we cache them!
    This makes generation O(n) instead of O(n²) - HUGE speedup!
    
    🔗 CONNECTS TO:
      → Uses RoPE (apply_rotary_embedding) for position encoding
      ← Used by TransformerBlock
    """
    
    # Maximum KV-cache size (sliding window for memory efficiency)
    MAX_CACHE_SEQ_LEN = 4096

    def __init__(self, config: ForgeConfig):
        """
        Initialize attention layer.
        
        Args:
            config: Model configuration with n_heads, n_kv_heads, dim, etc.
        """
        super().__init__()
        self.n_heads = config.n_heads          # Number of query heads
        self.n_kv_heads = config.n_kv_heads    # Number of key/value heads (for GQA)
        self.head_dim = config.dim // config.n_heads  # Dimension per head
        self.n_rep = self.n_heads // self.n_kv_heads  # How many Q heads per KV head
        
        # Cache size limit from config or default
        self.max_cache_len = min(
            config.max_seq_len if hasattr(config, 'max_seq_len') else self.MAX_CACHE_SEQ_LEN,
            self.MAX_CACHE_SEQ_LEN
        )

        # ─────────────────────────────────────────────────────────────────────
        # PROJECTION LAYERS: Transform input into Q, K, V, and output
        # ─────────────────────────────────────────────────────────────────────
        # Wq: Project to queries (one per head)
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=config.use_bias)
        # Wk: Project to keys (fewer for GQA)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=config.use_bias)
        # Wv: Project to values (same as keys)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=config.use_bias)
        # Wo: Project attention output back to model dimension
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=config.use_bias)

        self.dropout = nn.Dropout(config.dropout)
        self.use_rope = config.use_rope

        # ─────────────────────────────────────────────────────────────────────
        # KV-CACHE: Stores past keys and values for fast generation
        # ─────────────────────────────────────────────────────────────────────
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None, use_cache: bool = False, start_pos: int = 0
    ) -> torch.Tensor:
        """
        Forward pass through attention.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            freqs_cis: RoPE frequencies for position encoding
            mask: Attention mask (prevents looking at future tokens)
            use_cache: Whether to use/update KV-cache
            start_pos: Starting position (for cache continuation)
        
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        B, T, _ = x.shape  # Batch, Time (seq_len), _ (dim)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Project input to Q, K, V
        # ─────────────────────────────────────────────────────────────────────
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Apply RoPE position embeddings to Q and K
        # ─────────────────────────────────────────────────────────────────────
        if self.use_rope and freqs_cis is not None:
            q = apply_rotary_embedding(q, freqs_cis, start_pos)
            k = apply_rotary_embedding(k, freqs_cis, start_pos)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Handle KV-cache (for efficient generation)
        # ─────────────────────────────────────────────────────────────────────
        if use_cache:
            # Detach K, V from computation graph to prevent memory explosion
            # if someone accidentally backprops with use_cache=True
            k = k.detach()
            v = v.detach()
            
            if self.cache_k is None:
                # First token - just store K, V
                self.cache_k, self.cache_v = k, v
            else:
                # Append new K, V to cache
                self.cache_k = torch.cat([self.cache_k, k], dim=1)
                self.cache_v = torch.cat([self.cache_v, v], dim=1)
                
                # Sliding window: trim if cache gets too big
                if self.cache_k.shape[1] > self.max_cache_len:
                    trim_amount = self.cache_k.shape[1] - self.max_cache_len
                    self.cache_k = self.cache_k[:, trim_amount:, :, :]
                    self.cache_v = self.cache_v[:, trim_amount:, :, :]
                    
            k, v = self.cache_k, self.cache_v

        # ─────────────────────────────────────────────────────────────────────
        # STEP 4: Repeat K, V for GQA (if using fewer KV heads)
        # ─────────────────────────────────────────────────────────────────────
        if self.n_rep > 1:
            # Each KV head serves multiple Q heads
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 5: Compute attention (Flash or Standard)
        # ─────────────────────────────────────────────────────────────────────
        # Flash Attention conditions (all must be true):
        #   1. flash_attn package is installed (pip install flash-attn)
        #   2. Running on CUDA GPU
        #   3. Using half precision (fp16 or bf16) - Flash doesn't support fp32
        #   4. NOT using KV-cache (Flash doesn't support incremental decoding)
        #   5. Processing full sequence (not continuing from cached K/V)
        #
        # ⚠️ IMPORTANT LIMITATION:
        # Flash Attention is DISABLED during generation (use_cache=True) because:
        #   - Flash computes the full attention matrix efficiently but atomically
        #   - Incremental KV-cache decoding needs to attend to cached K/V
        #   - This is a fundamental limitation, not a bug
        #
        # Flash is used during: Training, prompt encoding, non-cached inference
        # Flash is NOT used during: Token-by-token generation with KV-cache
        #
        # Performance impact: Training gets 2-4x speedup. Generation uses standard
        # attention which is still efficient due to KV-cache (O(1) per token).
        use_flash = (
            HAS_FLASH_ATTN
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and not use_cache  # Flash doesn't support incremental decode
            and (mask is None or T == k.shape[1])  # Full sequence, not cached
        )
        
        if use_flash:
            # ─────────────────────────────────────────────────────────────────
            # FLASH ATTENTION PATH: O(n) memory, 2-4x faster
            # ─────────────────────────────────────────────────────────────────
            # flash_attn expects [batch, seq, heads, dim] - we already have that!
            # It handles causal masking internally with is_causal=True
            output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                causal=True,  # Autoregressive masking
                softmax_scale=1.0 / math.sqrt(self.head_dim)
            )
            # output is [batch, seq, heads, dim], need [batch, seq, dim]
            output = output.view(B, T, -1)
        else:
            # ─────────────────────────────────────────────────────────────────
            # STANDARD ATTENTION PATH: Works everywhere (CPU, MPS, any dtype)
            # ─────────────────────────────────────────────────────────────────
            # Transpose for batched matrix multiply: [batch, heads, seq, dim]
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            
            # scores = Q @ K.T / sqrt(head_dim) - scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # Mask is -inf for blocked positions

            # Softmax and dropout, then weighted sum of values
            attn = self.dropout(F.softmax(scores, dim=-1))
            output = torch.matmul(attn, v)
            
            # Reshape back: [batch, heads, seq, dim] -> [batch, seq, heads*dim]
            output = output.transpose(1, 2).contiguous().view(B, T, -1)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 6: Project back to model dimension
        # ─────────────────────────────────────────────────────────────────────
        return self.wo(output)

    def clear_cache(self):
        """Clear the KV-cache (call between different sequences)."""
        self.cache_k = self.cache_v = None


class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    📖 WHAT THIS DOES:
    After attention decides WHAT to look at, the FFN decides
    WHAT TO DO with that information. It's the "thinking" part!
    
    📐 SWIGLU FORMULA:
    Standard FFN: output = W2(ReLU(W1(x)))
    SwiGLU:       output = W2(Swish(W1(x)) * W3(x))
    
    💡 WHY SWIGLU IS BETTER:
    - Swish activation is smoother than ReLU (no hard corners)
    - Gating mechanism (the W3 multiplication) helps information flow
    - Empirically shown to train faster and achieve lower loss
    
    🔗 CONNECTS TO:
      ← Used by TransformerBlock after attention
    """

    def __init__(self, config: ForgeConfig):
        """
        Args:
            config: Model config with dim, hidden_dim, use_swiglu flag
        """
        super().__init__()
        self.use_swiglu = config.use_swiglu

        if self.use_swiglu:
            # ─────────────────────────────────────────────────────────────────
            # SWIGLU: 3 linear layers
            # ─────────────────────────────────────────────────────────────────
            # W1: Projects to hidden dim (for the gate)
            self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=config.use_bias)
            # W2: Projects back to model dim
            self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=config.use_bias)
            # W3: Projects to hidden dim (for the value)
            self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=config.use_bias)
        else:
            # ─────────────────────────────────────────────────────────────────
            # STANDARD FFN: 2 linear layers with ReLU
            # ─────────────────────────────────────────────────────────────────
            self.up = nn.Linear(config.dim, config.hidden_dim, bias=config.use_bias)
            self.down = nn.Linear(config.hidden_dim, config.dim, bias=config.use_bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.
        
        📐 SwiGLU computation:
        1. gate = swish(W1 @ x)  ← Smooth activation
        2. value = W3 @ x        ← Unactivated projection  
        3. hidden = gate * value ← Gated combination
        4. output = W2 @ hidden  ← Project back
        
        The "gating" (multiplication) is what makes SwiGLU special!
        """
        if self.use_swiglu:
            # SwiGLU: swish(W1(x)) * W3(x), then W2
            # F.silu = swish = x * sigmoid(x)
            return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))
        # Standard FFN: GELU(W1(x)), then W2
        return self.down(self.dropout(F.gelu(self.up(x))))


class MoEFeedForward(nn.Module):
    """
    Mixture of Experts Feed-Forward layer.
    
    📖 WHAT THIS DOES:
    Routes each token to top-k experts for specialized processing.
    Different experts can specialize in different types of content
    (e.g., one for code, one for math, one for creative writing).
    
    📐 MOE ARCHITECTURE:
    ┌────────────────────────────────────────────────────────────────────────┐
    │  Input x                                                               │
    │      ↓                                                                 │
    │  [Router/Gate] → Selects top-k experts                                │
    │      ↓                                                                 │
    │  ┌─────────┬─────────┬─────────┬─────────┐                            │
    │  │Expert 1 │Expert 2 │Expert 3 │Expert N │  (only top-k activated)    │
    │  └─────────┴─────────┴─────────┴─────────┘                            │
    │      ↓                                                                 │
    │  [Weighted Sum] → Combined output                                      │
    └────────────────────────────────────────────────────────────────────────┘
    
    💡 WHY MOE?
    - More parameters without proportional compute increase
    - Experts can specialize in different domains
    - Scales to very large models efficiently (GPT-4, Mixtral)
    
    ⚠️ TRAINING CONSIDERATIONS:
    - Load balancing loss prevents all tokens going to same expert
    - Auxiliary loss weight (moe_load_balancing) controls this
    """

    def __init__(self, config: ForgeConfig):
        """
        Args:
            config: Model configuration with MoE settings
        """
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.aux_loss_weight = config.moe_load_balancing
        
        # Router: determines which experts to use for each token
        self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
        
        # Expert networks (each is a standard FeedForward)
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.num_experts)
        ])
        
        # Track load balancing loss for training
        self.load_balancing_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE layer with vectorized routing.
        
        This implementation groups tokens by expert and processes them in batches,
        avoiding the O(tokens × experts) nested loop that would kill performance.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
        
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        B, T, D = x.shape
        num_tokens = B * T
        
        # Flatten batch and sequence dimensions for routing
        x_flat = x.view(-1, D)  # [num_tokens, D]
        
        # Compute router scores
        router_logits = self.gate(x_flat)  # [num_tokens, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts for each token
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.num_experts_per_token, dim=-1
        )  # Both: [num_tokens, k]
        
        # Normalize selected expert weights
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute load balancing loss for training (vectorized)
        if self.training:
            # Use one_hot + sum for vectorized counting (no loop)
            # Flatten top_k_indices and create one-hot encoding
            flat_indices = top_k_indices.view(-1)  # [num_tokens * k]
            expert_counts = torch.bincount(
                flat_indices, minlength=self.num_experts
            ).float()
            
            # Ideal distribution is uniform
            ideal_count = (num_tokens * self.num_experts_per_token) / self.num_experts
            # Loss is variance from ideal (encourages balance)
            self.load_balancing_loss = ((expert_counts - ideal_count) ** 2).mean()
        
        # ─────────────────────────────────────────────────────────────────────
        # VECTORIZED EXPERT ROUTING: Process each expert once with batched tokens
        # ─────────────────────────────────────────────────────────────────────
        # Instead of nested loops O(k × num_experts × num_tokens), we:
        # 1. Create a flat list of (token_idx, expert_idx, weight) assignments
        # 2. Sort/group by expert
        # 3. Process each expert's batch once
        # 4. Scatter results back
        
        # Expand token indices for all k selections
        token_indices = torch.arange(num_tokens, device=x.device)
        token_indices = token_indices.unsqueeze(1).expand(-1, self.num_experts_per_token)
        # token_indices: [num_tokens, k] - which token each assignment belongs to
        
        # Flatten everything for batched processing
        flat_token_idx = token_indices.reshape(-1)  # [num_tokens * k]
        flat_expert_idx = top_k_indices.reshape(-1)  # [num_tokens * k]
        flat_weights = top_k_probs.reshape(-1)  # [num_tokens * k]
        
        # Initialize output accumulator
        output = torch.zeros_like(x_flat)  # [num_tokens, D]
        
        # Process each expert's tokens in a single batch
        for expert_id in range(self.num_experts):
            # Find all assignments to this expert
            expert_mask = (flat_expert_idx == expert_id)
            
            if not expert_mask.any():
                continue
            
            # Get token indices and weights for this expert
            selected_token_idx = flat_token_idx[expert_mask]
            selected_weights = flat_weights[expert_mask]
            
            # Gather input tokens for this expert (single gather operation)
            expert_input = x_flat[selected_token_idx]  # [num_selected, D]
            
            # Process all tokens through this expert at once
            expert_output = self.experts[expert_id](expert_input)  # [num_selected, D]
            
            # Weight the outputs
            weighted_output = expert_output * selected_weights.unsqueeze(-1)
            
            # Scatter-add back to output (handles duplicate token indices)
            output.index_add_(0, selected_token_idx, weighted_output)
        
        # Reshape back to [B, T, D]
        return output.view(B, T, D)
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get the auxiliary load balancing loss for training."""
        return self.load_balancing_loss * self.aux_loss_weight


class TransformerBlock(nn.Module):
    """
    Single Transformer block with pre-norm architecture.
    
    📖 WHAT THIS DOES:
    One "layer" of the transformer - stack N of these for the full model.
    
    📐 PRE-NORM ARCHITECTURE (better than original post-norm!):
    x → [Norm] → [Attention] → + → [Norm] → [FFN] → + → output
         │                     ↑         │           ↑
         └─────────────────────┘         └───────────┘
              (residual skip)           (residual skip)
    
    💡 WHY PRE-NORM?
    Original transformers: Attention → Norm (post-norm)
    Modern transformers: Norm → Attention (pre-norm)
    Pre-norm is more stable during training, especially for deep models!
    
    ⚡ RESIDUAL CONNECTIONS (the + signs):
    Skip connections let gradients flow directly through the network.
    Without them, deep networks are nearly impossible to train.
    
    ⚡ GRADIENT CHECKPOINTING:
    When enabled, recomputes activations during backward pass instead of
    storing them. Trades ~30% compute for ~50% memory savings - essential
    for training large models on limited hardware.
    """

    def __init__(self, config: ForgeConfig, layer_id: int):
        """
        Args:
            config: Model configuration
            layer_id: Which layer this is (for debugging/logging)
        """
        super().__init__()
        self.layer_id = layer_id
        self.use_checkpoint = getattr(config, 'use_gradient_checkpointing', False)
        self.use_moe = getattr(config, 'use_moe', False)

        # Choose normalization type based on config
        Norm = RMSNorm if config.use_rms_norm else nn.LayerNorm
        
        # Two normalizations: one before attention, one before FFN
        self.attention_norm = Norm(config.dim)
        self.ffn_norm = Norm(config.dim)
        
        # The actual computation modules
        self.attention = Attention(config)
        # Use MoE feed-forward if enabled, otherwise standard feed-forward
        if self.use_moe:
            self.feed_forward = MoEFeedForward(config)
        else:
            self.feed_forward = FeedForward(config)

    def _forward_impl(
        self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None, use_cache: bool = False, start_pos: int = 0
    ) -> torch.Tensor:
        """Internal forward implementation (used by checkpointing)."""
        # Attention sub-layer with residual connection
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, use_cache, start_pos)
        # FFN sub-layer with residual connection
        return h + self.feed_forward(self.ffn_norm(h))

    def forward(
        self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None, use_cache: bool = False, start_pos: int = 0
    ) -> torch.Tensor:
        """
        Forward pass: Norm → Attention → Add → Norm → FFN → Add
        
        Uses gradient checkpointing during training if enabled, which
        recomputes activations during backward pass to save memory.
        
        Args:
            x: Input [batch, seq_len, dim]
            freqs_cis: RoPE frequencies
            mask: Causal attention mask
            use_cache: Whether to use KV-cache
            start_pos: Position for KV-cache
        
        Returns:
            Output tensor, same shape as input
        """
        # Use gradient checkpointing during training if enabled
        # Don't use with KV-cache as it doesn't make sense (inference only)
        if self.use_checkpoint and self.training and not use_cache:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                x, freqs_cis, mask, use_cache, start_pos,
                use_reentrant=False  # Recommended for newer PyTorch
            )
        return self._forward_impl(x, freqs_cis, mask, use_cache, start_pos)

    def clear_cache(self):
        """Clear KV-cache in the attention layer."""
        self.attention.clear_cache()

    def get_moe_aux_loss(self) -> torch.Tensor:
        """Get MoE auxiliary loss for load balancing during training."""
        if self.use_moe and hasattr(self.feed_forward, 'get_aux_loss'):
            return self.feed_forward.get_aux_loss()
        return torch.tensor(0.0)


# =============================================================================
# 🧠 MAIN MODEL - THE FULL TRANSFORMER
# =============================================================================

class Enigma(nn.Module):
    """
    Enigma - Modern Transformer Language Model
    
    📖 THIS IS THE COMPLETE MODEL!
    Stacks together all the components: embeddings, transformer blocks, output.
    
    📐 FULL ARCHITECTURE:
    ┌────────────────────────────────────────────────────────────────────────┐
    │  Input Token IDs [batch, seq_len]                                       │
    │          ↓                                                              │
    │  ┌──────────────────┐                                                  │
    │  │ Token Embedding  │  Converts token IDs to vectors                   │
    │  └──────────────────┘                                                  │
    │          ↓                                                              │
    │  ┌──────────────────┐                                                  │
    │  │ Dropout          │  Regularization during training                  │
    │  └──────────────────┘                                                  │
    │          ↓                                                              │
    │  ╔══════════════════╗                                                  │
    │  ║ TransformerBlock ║ × n_layers                                       │
    │  ║  • Attention     ║  (the "thinking" happens here!)                  │
    │  ║  • Feed-Forward  ║                                                  │
    │  ╚══════════════════╝                                                  │
    │          ↓                                                              │
    │  ┌──────────────────┐                                                  │
    │  │ Final Norm       │  RMSNorm for stable outputs                      │
    │  └──────────────────┘                                                  │
    │          ↓                                                              │
    │  ┌──────────────────┐                                                  │
    │  │ Output Head      │  Projects to vocabulary size                     │
    │  └──────────────────┘                                                  │
    │          ↓                                                              │
    │  Output Logits [batch, seq_len, vocab_size]                            │
    └────────────────────────────────────────────────────────────────────────┘
    
    ⚡ FEATURES:
    - RoPE positional embeddings (knows word order)
    - RMSNorm (fast and stable)
    - SwiGLU activation (better learning)
    - GQA attention (memory efficient)
    - KV cache (fast generation)
    
    🔗 CONNECTS TO:
      → Uses all the components defined above
      ← Used by EnigmaEngine for inference
      ← Used by Trainer for training
    """

    def __init__(
        self, vocab_size: int = 8000, dim: Optional[int] = None,
        depth: Optional[int] = None, heads: Optional[int] = None,
        max_len: Optional[int] = None, config: Optional[ForgeConfig] = None, **kwargs
    ):
        """
        Initialize the Forge model.
        
        Can be initialized two ways:
        1. With a ForgeConfig object
        2. With individual parameters (for backwards compatibility)
        
        Args:
            vocab_size: Size of vocabulary
            dim: Model hidden dimension
            depth: Number of transformer layers
            heads: Number of attention heads
            max_len: Maximum sequence length
            config: ForgeConfig object (overrides other args)
            **kwargs: Additional config parameters
        """
        super().__init__()

        # Build config
        if config is not None:
            self.config = config
        else:
            self.config = ForgeConfig(
                vocab_size=vocab_size,
                dim=dim or CONFIG.get("embed_dim", 512),
                n_layers=depth or CONFIG.get("num_layers", 8),
                n_heads=heads or CONFIG.get("num_heads", 8),
                max_seq_len=max_len or CONFIG.get("max_len", 1024),
                **{k: v for k, v in kwargs.items() if hasattr(ForgeConfig, k)}
            )

        if vocab_size != 8000:
            self.config.vocab_size = vocab_size

        # Legacy attributes for compatibility
        self.vocab_size = self.config.vocab_size
        self.dim = self.config.dim
        self.depth = self.config.n_layers
        self.heads = self.config.n_heads
        self.max_len = self.config.max_seq_len

        # Token embeddings
        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.dim)

        # Legacy alias
        self.token_embed = self.tok_embeddings

        # ─────────────────────────────────────────────────────────────────────
        # MULTI-MODAL PROJECTION LAYERS (Optional)
        # ─────────────────────────────────────────────────────────────────────
        # These allow integrating vision/audio encoders with the text model
        if self.config.vision_hidden_size is not None:
            self.vision_projection = nn.Linear(
                self.config.vision_hidden_size,
                self.config.dim,
                bias=False
            )
            logger.info(f"Added vision projection: {self.config.vision_hidden_size} → {self.config.dim}")
        else:
            self.vision_projection = None
        
        if self.config.audio_hidden_size is not None:
            self.audio_projection = nn.Linear(
                self.config.audio_hidden_size,
                self.config.dim,
                bias=False
            )
            logger.info(f"Added audio projection: {self.config.audio_hidden_size} → {self.config.dim}")
        else:
            self.audio_projection = None

        # Position embeddings (fallback)
        if not self.config.use_rope:
            self.pos = nn.Parameter(torch.randn(1, self.config.max_seq_len, self.config.dim) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(self.config, i) for i in range(self.config.n_layers)
        ])

        # Output
        Norm = RMSNorm if self.config.use_rms_norm else nn.LayerNorm
        self.norm = Norm(self.config.dim)
        self.output = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)
        self.head = self.output  # Legacy alias

        # Weight tying
        self.output.weight = self.tok_embeddings.weight

        # RoPE frequencies with optional scaling
        if self.config.use_rope:
            head_dim = self.config.dim // self.config.n_heads
            # Validate head_dim is even (required for RoPE complex number reshape)
            if head_dim % 2 != 0:
                raise ValueError(
                    f"head_dim must be even for RoPE, got {head_dim} "
                    f"(dim={self.config.dim}, n_heads={self.config.n_heads}). "
                    f"Adjust dim or n_heads so dim/n_heads is even."
                )
            self.register_buffer(
                'freqs_cis',
                precompute_rope_frequencies(
                    head_dim,
                    self.config.max_seq_len * 2,
                    self.config.rope_theta,
                    scaling_type=self.config.rope_scaling_type,
                    scaling_factor=self.config.rope_scaling_factor
                )
            )
        else:
            self.freqs_cis = None

        # ─────────────────────────────────────────────────────────────────────
        # CAUSAL MASK CACHE: Pre-compute and cache the causal attention mask
        # ─────────────────────────────────────────────────────────────────────
        # Instead of rebuilding the mask every forward pass, we cache a max-size
        # mask and slice it as needed. This is especially beneficial for:
        # - Training: Same mask reused if batch sizes are consistent
        # - Inference: Even single-token generation reuses the cached mask
        max_seq = self.config.max_seq_len
        causal_mask = torch.full((max_seq, max_seq), float('-inf'))
        causal_mask = torch.triu(causal_mask, diagonal=1)
        self.register_buffer('_causal_mask', causal_mask, persistent=False)

        # Initialize
        self.apply(self._init_weights)
        self._init_output_weights()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_output_weights(self):
        for name, p in self.named_parameters():
            if name.endswith('wo.weight') or name.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None,
        use_cache: bool = False, start_pos: int = 0, return_loss: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs (B, T)
            targets: Optional target IDs for loss computation
            use_cache: Whether to use KV cache
            start_pos: Starting position for RoPE
            return_loss: If True, always return (logits, loss) tuple

        Returns:
            logits if no targets and return_loss=False, else (logits, loss)
        """
        B, T = input_ids.shape

        h = self.tok_embeddings(input_ids)

        if not self.config.use_rope:
            h = h + self.pos[:, start_pos:start_pos + T]

        # Use cached causal mask (sliced to current sequence length)
        # This avoids rebuilding the mask every forward pass
        mask = None
        if T > 1:
            # Slice the pre-computed mask to current sequence length
            # Shape: (T, T) -> (1, 1, T, T) for broadcasting with attention
            mask = self._causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, self.freqs_cis, mask, use_cache, start_pos)

        logits = self.output(self.norm(h))

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=0)

        # Return format depends on whether loss was computed
        if targets is not None or return_loss:
            return logits, loss

        return logits

    def clear_cache(self):
        for layer in self.layers:
            layer.clear_cache()

    def get_moe_aux_loss(self) -> torch.Tensor:
        """
        Get total MoE auxiliary loss from all layers.
        
        📖 WHAT THIS DOES:
        Collects load balancing losses from all MoE layers to encourage
        even distribution of tokens across experts during training.
        
        ⚠️ TRAINING ONLY:
        This loss should be added to the main training loss:
            total_loss = ce_loss + model.get_moe_aux_loss()
        
        Returns:
            Scalar tensor with combined auxiliary loss from all MoE layers.
            Returns 0.0 if MoE is not enabled.
        """
        if not self.config.use_moe:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            aux_loss = aux_loss + layer.get_moe_aux_loss()
        return aux_loss

    def forward_multimodal(
        self,
        input_ids: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with multi-modal inputs.
        
        📖 WHAT THIS DOES:
        Processes text, vision, and/or audio inputs together. Vision and audio
        features are projected to the text embedding space and concatenated.
        
        Args:
            input_ids: Text token IDs [batch, text_seq_len]
            vision_features: Vision encoder output [batch, vision_seq_len, vision_dim]
            audio_features: Audio encoder output [batch, audio_seq_len, audio_dim]
            **kwargs: Additional forward pass arguments
        
        Returns:
            Model output logits
        
        Example:
            # Text + vision
            logits = model.forward_multimodal(
                input_ids=text_ids,
                vision_features=vision_output
            )
        """
        embeddings_list = []
        
        # Process vision features
        if vision_features is not None:
            if self.vision_projection is None:
                raise ValueError(
                    "Vision features provided but vision_hidden_size not set in config"
                )
            vision_embeds = self.vision_projection(vision_features)
            embeddings_list.append(vision_embeds)
        
        # Process audio features
        if audio_features is not None:
            if self.audio_projection is None:
                raise ValueError(
                    "Audio features provided but audio_hidden_size not set in config"
                )
            audio_embeds = self.audio_projection(audio_features)
            embeddings_list.append(audio_embeds)
        
        # Process text
        if input_ids is not None:
            text_embeds = self.tok_embeddings(input_ids)
            embeddings_list.append(text_embeds)
        
        if not embeddings_list:
            raise ValueError("At least one of input_ids, vision_features, or audio_features must be provided")
        
        # Concatenate all modalities
        combined_embeds = torch.cat(embeddings_list, dim=1)
        
        # Continue with standard forward pass
        B, T, _ = combined_embeds.shape
        h = combined_embeds
        
        # Add positional embeddings if not using RoPE
        if not self.config.use_rope and hasattr(self, 'pos'):
            if T <= self.config.max_seq_len:
                h = h + self.pos[:, :T]
        
        # Build causal mask
        mask = None
        if T > 1:
            mask = self._causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)
        
        # Transform through layers
        for layer in self.layers:
            h = layer(h, self.freqs_cis, mask, kwargs.get('use_cache', False), kwargs.get('start_pos', 0))
        
        # Output projection
        logits = self.output(self.norm(h))
        
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[list[int]] = None,
        *,  # Force keyword-only args after this
        return_logits: bool = False,
        stream: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k tokens for sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            stop_tokens: Token IDs that stop generation
            return_logits: If True, also return the final logits
            stream: If True, use streaming generation (returns generator)
        
        Returns:
            Generated token IDs, or (token_ids, logits) if return_logits=True,
            or generator if stream=True
            
        Raises:
            ValueError: If temperature is not positive
        """
        # Validate temperature
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
            
        # Delegate to streaming generator if requested
        if stream:
            return self.generate_stream(
                input_ids, max_new_tokens, temperature, top_k, top_p,
                repetition_penalty, stop_tokens
            )
        
        self.clear_cache()
        stop_tokens = stop_tokens or [2]

        generated = input_ids
        logits = self.forward(input_ids, use_cache=True)
        final_logits = logits  # Track for return_logits option

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / temperature

            # Repetition penalty - Hybrid approach for optimal performance
            # Uses set-based lookup for short sequences, bincount for longer ones
            if repetition_penalty != 1.0:
                vocab_size = next_logits.shape[-1]
                for i in range(input_ids.shape[0]):
                    seq_len = generated[i].shape[0]
                    if seq_len < 1000:
                        # Set-based for short sequences (lower overhead)
                        unique_tokens = set(generated[i].tolist())
                        for token_id in unique_tokens:
                            if 0 <= token_id < vocab_size:
                                next_logits[i, token_id] /= repetition_penalty
                    else:
                        # Bincount for longer sequences (better vectorization)
                        token_ids = generated[i].clamp(0, vocab_size - 1)
                        token_counts = torch.bincount(token_ids, minlength=vocab_size)
                        appeared_mask = token_counts > 0
                        next_logits[i, appeared_mask] /= repetition_penalty

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')

            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumsum > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False
                indices_to_remove = mask.scatter(1, sorted_idx, mask)
                next_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() in stop_tokens:
                break

            logits = self.forward(next_token, use_cache=True, start_pos=generated.shape[1] - 1)
            final_logits = logits  # Update for return_logits

        # Return based on options
        if return_logits:
            return generated, final_logits
        return generated

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[list[int]] = None
    ) -> Generator[torch.Tensor, None, None]:
        """
        Streaming token generation - yields tokens as they're generated.
        
        📖 WHAT THIS DOES:
        Instead of waiting for all tokens to be generated, this yields each
        token as soon as it's produced. Essential for real-time chat UX!
        
        📐 STREAMING FLOW:
        ┌────────────────────────────────────────────────────────────────────┐
        │  User: "Tell me a story"                                           │
        │                                                                    │
        │  [Model starts generating]                                         │
        │       ↓                                                            │
        │  yield "Once"    ← User sees this immediately                      │
        │       ↓                                                            │
        │  yield " upon"   ← And this...                                     │
        │       ↓                                                            │
        │  yield " a"      ← And this...                                     │
        │       ↓                                                            │
        │  yield " time"   ← Progressive display!                            │
        └────────────────────────────────────────────────────────────────────┘
        
        💡 ADVANTAGES:
        - Better user experience (see output immediately)
        - Can cancel generation early
        - Works great with chat interfaces
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k tokens for sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
            stop_tokens: Token IDs that stop generation
        
        Yields:
            Individual tokens as they're generated [1] tensor
        
        Example:
            for token in model.generate_stream(input_ids):
                print(tokenizer.decode([token.item()]), end='', flush=True)
        """
        self.clear_cache()
        stop_tokens = stop_tokens or [2]
        
        generated = input_ids
        logits = self.forward(input_ids, use_cache=True)
        vocab_size = logits.shape[-1]
        
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / temperature
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                # Efficient penalty for shorter sequences using set lookup
                if generated.shape[1] < 1000:
                    unique_tokens = set(generated[0].tolist())
                    for token_id in unique_tokens:
                        if 0 <= token_id < vocab_size:
                            next_logits[0, token_id] /= repetition_penalty
                else:
                    # Bincount for longer sequences
                    token_ids = generated[0].clamp(0, vocab_size - 1)
                    token_counts = torch.bincount(token_ids, minlength=vocab_size)
                    appeared_mask = token_counts > 0
                    next_logits[0, appeared_mask] /= repetition_penalty
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, vocab_size))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumsum > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False
                indices_to_remove = mask.scatter(1, sorted_idx, mask)
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Yield the token immediately
            yield next_token.squeeze()
            
            # Check for stop tokens
            if next_token.item() in stop_tokens:
                break
            
            # Update generated sequence and get next logits
            generated = torch.cat([generated, next_token], dim=1)
            logits = self.forward(next_token, use_cache=True, start_pos=generated.shape[1] - 1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict[str, Any]:
        return self.config.to_dict()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'Enigma':
        return cls(config=ForgeConfig.from_dict(config))

    @classmethod
    def from_pretrained(cls, path: Path) -> 'Enigma':
        from .model_registry import safe_load_weights
        path = Path(path)
        config_file = path / 'config.json' if path.is_dir() else path.with_suffix('.json')

        if config_file.exists():
            with open(config_file) as f:
                model = cls.from_config(json.load(f))
        else:
            model = cls()

        weights_file = path / 'weights.pth' if path.is_dir() else path
        if weights_file.exists():
            state_dict = safe_load_weights(weights_file, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

        return model

    @classmethod
    def from_pretrained_quantized(cls, path: Path, quantization: str = "dynamic") -> 'Enigma':
        """
        Load a pretrained model with quantization applied.
        
        📖 WHAT THIS DOES:
        Loads a model from disk and immediately applies quantization to reduce
        memory footprint. Ideal for deployment on memory-constrained devices.
        
        📐 QUANTIZATION FLOW:
        1. Load model weights to CPU
        2. Apply specified quantization
        3. Return memory-optimized model
        
        Args:
            path: Path to model weights or directory
            quantization: Type of quantization ("dynamic", "int8", "int4")
        
        Returns:
            Quantized Forge model
        
        Example:
            # Load quantized model for Pi deployment
            model = Forge.from_pretrained_quantized("models/forge.pth", "int8")
        """
        # Load model first
        model = cls.from_pretrained(path)
        
        # Apply quantization
        model.quantize(quantization)
        
        return model

    @classmethod
    def load_mmap(cls, path: Path) -> 'Enigma':
        """
        Load model using memory-mapped file loading.
        
        📖 WHAT THIS DOES:
        Uses mmap to load model weights without loading everything into RAM.
        This is useful for loading large models on memory-constrained devices.
        
        📐 HOW IT WORKS:
        Instead of loading the entire file into RAM, mmap creates a virtual
        mapping that loads pages on-demand from disk. This dramatically
        reduces peak memory usage during loading.
        
        ⚠️ LIMITATIONS:
        - Slightly slower inference (disk reads on cache miss)
        - Model file must be on fast storage (SSD recommended)
        - Not compatible with CUDA (model stays on CPU)
        
        Args:
            path: Path to model weights (.pth file)
        
        Returns:
            Memory-mapped Forge model
        """
        path = Path(path)
        config_file = path / 'config.json' if path.is_dir() else path.with_suffix('.json')
        
        # Load config
        if config_file.exists():
            with open(config_file) as f:
                model = cls.from_config(json.load(f))
        else:
            model = cls()
        
        # Load weights with mmap
        weights_file = path / 'weights.pth' if path.is_dir() else path
        if weights_file.exists():
            # Use mmap_mode for memory-efficient loading
            # weights_only=True for security against pickle attacks
            try:
                state_dict = torch.load(
                    weights_file, 
                    map_location='cpu',
                    mmap=True,  # Memory-mapped loading (PyTorch 2.0+)
                    weights_only=True
                )
            except TypeError:
                # Fallback for older PyTorch versions
                logger.warning("mmap loading not available, using standard load")
                state_dict = torch.load(weights_file, map_location='cpu', weights_only=True)
            
            model.load_state_dict(state_dict, strict=False)
        
        logger.info(f"Loaded model with mmap from: {path}")
        return model

    @classmethod
    def auto_configure(cls, vocab_size: int = 8000) -> 'Enigma':
        """
        Create an optimally-configured model for the current hardware.
        
        📖 WHAT THIS DOES:
        Detects your hardware (GPU, RAM, Pi model) and automatically creates
        a model with the best configuration for your system.
        
        📐 DETECTION FLOW:
        ┌─────────────────────────────────────────────────────────────────────┐
        │  1. Detect hardware (RAM, GPU, is_raspberry_pi)                     │
        │  2. Recommend model size (pi_zero, pi_4, pi_5, small, medium, etc.) │
        │  3. Recommend quantization (none, dynamic, int8, int4)              │
        │  4. Create model with optimal config                                │
        │  5. Apply quantization if recommended                               │
        └─────────────────────────────────────────────────────────────────────┘
        
        Args:
            vocab_size: Vocabulary size for the model
        
        Returns:
            Optimally configured Forge model
        
        Example:
            # Auto-detect and create best model for this device
            model = Forge.auto_configure()
        """
        try:
            from .hardware_detection import detect_hardware, get_optimal_config
            
            profile = detect_hardware()
            config = get_optimal_config(profile)
            
            logger.info(f"Auto-configured for: {profile.cpu_model}")
            logger.info(f"Recommended: {config['size']} with {config.get('quantization', 'none')} quantization")
            
            # Create model with recommended size
            model = create_model(config['size'], vocab_size=vocab_size)
            
            # Apply quantization if recommended
            quant = config.get('quantization', 'none')
            if quant and quant != 'none':
                model.quantize(quant)
            
            return model
            
        except ImportError as e:
            logger.warning(f"Hardware detection not available: {e}, using default config")
            return create_model('small', vocab_size=vocab_size)

    def quantize(self, mode: str = "dynamic") -> 'Enigma':
        """
        Apply quantization to reduce model memory footprint.
        
        📖 WHAT THIS DOES:
        Converts model weights to lower precision to save memory and
        potentially speed up inference on CPU.
        
        📐 QUANTIZATION MODES:
        ┌────────────────────────────────────────────────────────────────────┐
        │ Mode     │ Memory │ Speed  │ Quality │ Best For                   │
        ├──────────┼────────┼────────┼─────────┼────────────────────────────┤
        │ none     │ 100%   │ Fast   │ Best    │ GPU, plenty of RAM         │
        │ dynamic  │ ~50%   │ Fast   │ Good    │ CPU inference, Pi 5        │
        │ int8     │ ~25%   │ Medium │ Good    │ Pi 4, limited RAM          │
        │ int4     │ ~12%   │ Slower │ Fair    │ Pi Zero, extreme limits    │
        └────────────────────────────────────────────────────────────────────┘
        
        ⚠️ NOTES:
        - Quantization is irreversible (on the current model instance)
        - GPU acceleration may not work after quantization
        - Quality degradation is usually minor for dynamic/int8
        
        Args:
            mode: Quantization mode ("none", "dynamic", "int8", "int4")
        
        Returns:
            Self (for method chaining)
        
        Example:
            model = create_model('small')
            model.quantize('dynamic')  # Now uses less memory
        """
        if mode == "none":
            logger.info("No quantization applied")
            return self
        
        # Ensure model is on CPU for quantization
        device = next(self.parameters()).device
        if device.type != 'cpu':
            logger.info("Moving model to CPU for quantization")
            self.cpu()
        
        if mode == "dynamic":
            # Dynamic quantization - fastest, good quality
            try:
                self._apply_dynamic_quantization()
                logger.info("Applied dynamic INT8 quantization")
            except Exception as e:
                logger.warning(f"Dynamic quantization failed: {e}")
                
        elif mode == "int8":
            # Static INT8 quantization
            try:
                self._apply_static_int8_quantization()
                logger.info("Applied static INT8 quantization")
            except Exception as e:
                logger.warning(f"Static INT8 quantization failed: {e}")
                
        elif mode == "int4":
            # 4-bit quantization (most aggressive)
            try:
                self._apply_int4_quantization()
                logger.info("Applied INT4 quantization")
            except Exception as e:
                logger.warning(f"INT4 quantization failed: {e}")
        else:
            logger.warning(f"Unknown quantization mode: {mode}")
        
        return self

    def _apply_dynamic_quantization(self):
        """Apply PyTorch dynamic quantization to linear layers."""
        torch.quantization.quantize_dynamic(
            self,
            {nn.Linear},  # Quantize Linear layers
            dtype=torch.qint8,
            inplace=True
        )

    def _apply_static_int8_quantization(self):
        """Apply static INT8 quantization (requires calibration data)."""
        # For static quantization, we'd need calibration data
        # Fall back to dynamic for now
        logger.info("Static INT8 uses dynamic quantization (no calibration data)")
        self._apply_dynamic_quantization()

    def _apply_int4_quantization(self):
        """Apply 4-bit weight-only quantization."""
        # INT4 quantization is more complex, use dynamic as fallback
        # True INT4 would require specialized libraries like bitsandbytes
        try:
            # Try using bitsandbytes if available
            import bitsandbytes as bnb

            # Convert Linear layers to 4-bit
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    # Replace with 4-bit linear
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = self.get_submodule(parent_name) if parent_name else self
                    
                    new_layer = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None
                    )
                    new_layer.weight = bnb.nn.Params4bit(
                        module.weight.data,
                        requires_grad=False
                    )
                    if module.bias is not None:
                        new_layer.bias = module.bias
                    
                    setattr(parent, child_name, new_layer)
                    
            logger.info("Applied bitsandbytes INT4 quantization")
            
        except ImportError:
            # Fallback to dynamic quantization
            logger.warning("bitsandbytes not available, using dynamic quantization")
            self._apply_dynamic_quantization()

    # =========================================================================
    # 🌐 UNIVERSAL MODEL LOADING - Load from any format
    # =========================================================================
    
    @classmethod
    def from_any(cls, path: Union[str, Path], **kwargs) -> 'Enigma':
        """
        Universal model loader - auto-detects format and loads appropriately.
        
        📖 WHAT THIS DOES:
        Automatically detects the model format and uses the appropriate loader.
        Supports: HuggingFace, Safetensors, GGUF, ONNX, and native Forge format.
        
        📐 FORMAT DETECTION:
        - Directory with config.json + model files → HuggingFace format
        - *.safetensors → Safetensors format
        - *.gguf → GGUF/llama.cpp format
        - *.onnx → ONNX format
        - *.pth, *.pt → Native PyTorch/Forge format
        
        Args:
            path: Path to model file or directory
            **kwargs: Additional arguments passed to specific loader
        
        Returns:
            Loaded Forge model
        
        Example:
            model = Forge.from_any("microsoft/phi-2")  # HuggingFace
            model = Forge.from_any("model.gguf")        # GGUF
            model = Forge.from_any("model.safetensors") # Safetensors
        """
        path = Path(path) if not isinstance(path, Path) else path
        
        # Check if it's a HuggingFace model ID (format: org/model, no file extensions)
        path_str = str(path)
        # HF IDs: don't exist as files, contain exactly one '/', no file extension
        is_hf_id = (
            not os.path.exists(path_str) and 
            '/' in path_str and 
            path_str.count('/') == 1 and  # org/model format
            not path_str.startswith('/') and  # not absolute path
            '.' not in Path(path_str).name  # no file extension
        )
        if is_hf_id:
            logger.info(f"Detected HuggingFace model ID: {path_str}")
            return cls.from_huggingface(path_str, **kwargs)
        
        # Check if path exists
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        
        # Directory - check for HuggingFace format
        if path.is_dir():
            config_file = path / 'config.json'
            if config_file.exists():
                # Could be HuggingFace or Forge format
                with open(config_file) as f:
                    config_data = json.load(f)
                    # HuggingFace configs have 'model_type' or 'architectures'
                    if 'model_type' in config_data or 'architectures' in config_data:
                        logger.info("Detected HuggingFace format (directory)")
                        return cls.from_huggingface(path, **kwargs)
            # Default to Forge format
            logger.info("Using Forge format loader")
            return cls.from_pretrained(path)
        
        # File - check extension
        suffix = path.suffix.lower()
        
        if suffix == '.safetensors':
            logger.info("Detected Safetensors format")
            return cls.from_safetensors(path, **kwargs)
        
        elif suffix == '.gguf':
            logger.info("Detected GGUF format")
            return cls.from_gguf(path, **kwargs)
        
        elif suffix == '.onnx':
            logger.info("Detected ONNX format")
            return cls.from_onnx(path, **kwargs)
        
        elif suffix in ['.pth', '.pt', '.bin']:
            logger.info("Detected PyTorch/Forge format")
            return cls.from_pretrained(path)
        
        else:
            raise ValueError(
                f"Unknown model format: {suffix}. "
                f"Supported: .pth, .pt, .safetensors, .gguf, .onnx, or HuggingFace ID"
            )
    
    @classmethod
    def from_huggingface(cls, model_id: str, **kwargs) -> 'Forge':
        """
        Load a model from HuggingFace Hub or local HuggingFace format.
        
        📖 WHAT THIS DOES:
        Downloads and converts a HuggingFace transformer model to Forge format.
        Supports most decoder-only models (GPT-2, GPT-Neo, LLaMA, etc.).
        
        Args:
            model_id: HuggingFace model ID (e.g., "gpt2") or local path
            **kwargs: Additional arguments (cache_dir, revision, etc.)
        
        Returns:
            Forge model with loaded weights
        
        Example:
            model = Forge.from_huggingface("gpt2")
            model = Forge.from_huggingface("microsoft/phi-2")
        """
        try:
            from .huggingface_loader import convert_huggingface_to_forge
            logger.info(f"Loading HuggingFace model: {model_id}")
            return convert_huggingface_to_forge(model_id, **kwargs)
        except ImportError:
            logger.error(
                "HuggingFace model loading requires transformers library. "
                "Install with: pip install transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise
    
    @classmethod
    def from_safetensors(cls, path: Union[str, Path], **kwargs) -> 'Forge':
        """
        Load a model from Safetensors format.
        
        📖 WHAT THIS DOES:
        Loads model weights from Safetensors format, which is faster and
        safer than pickle-based formats (no arbitrary code execution).
        
        Args:
            path: Path to .safetensors file
            **kwargs: Additional arguments (map_location, etc.)
        
        Returns:
            Forge model with loaded weights
        
        Example:
            model = Forge.from_safetensors("model.safetensors")
        """
        try:
            from safetensors.torch import load_file
        except ImportError:
            logger.error(
                "Safetensors loading requires safetensors library. "
                "Install with: pip install safetensors"
            )
            raise
        
        path = Path(path)
        
        # Load config if available
        config_file = path.with_suffix('.json')
        if config_file.exists():
            with open(config_file) as f:
                model = cls.from_config(json.load(f))
        else:
            logger.warning("No config file found, using default config")
            model = cls()
        
        # Load weights
        logger.info(f"Loading Safetensors from: {path}")
        state_dict = load_file(str(path), device=kwargs.get('map_location', 'cpu'))
        model.load_state_dict(state_dict, strict=False)
        
        return model
    
    @classmethod
    def from_gguf(cls, path: Union[str, Path], **kwargs) -> 'Forge':
        """
        Load a model from GGUF format (llama.cpp compatible).
        
        📖 WHAT THIS DOES:
        Loads quantized models in GGUF format, commonly used by llama.cpp.
        Automatically dequantizes weights to PyTorch tensors.
        
        ⚠️ NOTE:
        GGUF models are often quantized (Q4, Q8, etc.). Loading converts
        them to full precision PyTorch, which may use more memory than
        the original GGUF file.
        
        Args:
            path: Path to .gguf file
            **kwargs: Additional arguments
        
        Returns:
            Forge model with loaded weights
        
        Example:
            model = Forge.from_gguf("llama-2-7b.Q4_K_M.gguf")
        """
        try:
            from .gguf_loader import load_gguf_model
            logger.info(f"Loading GGUF model from: {path}")
            return load_gguf_model(str(path), **kwargs)
        except ImportError:
            logger.error(
                "GGUF model loading requires gguf library. "
                "Install with: pip install gguf"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            raise
    
    @classmethod
    def from_onnx(cls, path: Union[str, Path], **kwargs) -> 'Forge':
        """
        Load a model from ONNX format.
        
        📖 WHAT THIS DOES:
        Loads a model exported to ONNX format and converts it to Forge.
        Useful for cross-platform deployment and inference optimization.
        
        ⚠️ NOTE:
        ONNX models may have optimizations that don't translate perfectly
        to PyTorch. Some features may not work after conversion.
        
        Args:
            path: Path to .onnx file
            **kwargs: Additional arguments
        
        Returns:
            Forge model with loaded weights
        
        Example:
            model = Forge.from_onnx("model.onnx")
        """
        logger.info(f"Loading ONNX model from: {path}")
        
        try:
            from .onnx_loader import load_onnx_model
            return load_onnx_model(str(path), **kwargs)
        except ImportError:
            logger.error(
                "ONNX model loading requires onnx library. "
                "Install with: pip install onnx"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    # =========================================================================
    # 🎯 LORA & ADAPTER SUPPORT
    # =========================================================================
    
    def load_lora(
        self, 
        path: Union[str, Path], 
        adapter_name: str = "default",
        merge: bool = False
    ) -> None:
        """
        Load LoRA (Low-Rank Adaptation) weights.
        
        📖 WHAT THIS DOES:
        Loads LoRA adapter weights that modify the model's behavior without
        full fine-tuning. LoRA is memory-efficient and can be quickly swapped.
        
        📐 HOW LORA WORKS:
        Instead of updating full weight matrix W, LoRA adds:
        W' = W + A × B  (where A, B are small low-rank matrices)
        
        Args:
            path: Path to LoRA weights
            adapter_name: Name for this adapter (for multi-adapter support)
            merge: If True, immediately merge LoRA into base weights
        
        Example:
            model.load_lora("lora_adapters/coding.pth")
            model.load_lora("lora_adapters/creative.pth", "creative")
        """
        try:
            from .lora_utils import apply_lora, load_lora_weights
        except ImportError:
            logger.error("LoRA support requires lora_utils module")
            raise
        
        logger.info(f"Loading LoRA adapter '{adapter_name}' from: {path}")
        
        # Load LoRA weights
        lora_weights = load_lora_weights(path)
        
        # Apply to model
        if merge:
            # Merge into base weights immediately
            apply_lora(self, lora_weights, merge=True)
            logger.info(f"Merged LoRA adapter '{adapter_name}' into base weights")
        else:
            # Keep as separate adapter
            if not hasattr(self, '_lora_adapters'):
                self._lora_adapters = {}
            self._lora_adapters[adapter_name] = lora_weights
            apply_lora(self, lora_weights, adapter_name=adapter_name)
            logger.info(f"Loaded LoRA adapter '{adapter_name}'")
    
    def merge_lora(self, adapter_name: Optional[str] = None) -> None:
        """
        Merge LoRA adapters into base model weights.
        
        📖 WHAT THIS DOES:
        Permanently integrates LoRA adapter weights into the base model.
        After merging, the adapter can be removed to save memory.
        
        Args:
            adapter_name: Specific adapter to merge (None = merge all)
        
        Example:
            model.load_lora("adapter.pth", "my_adapter")
            model.merge_lora("my_adapter")  # Merge into base weights
        """
        if not hasattr(self, '_lora_adapters'):
            logger.warning("No LoRA adapters loaded")
            return
        
        try:
            from .lora_utils import merge_lora_weights
        except ImportError:
            logger.error("LoRA support requires lora_utils module")
            raise
        
        if adapter_name is None:
            # Merge all adapters
            for name in list(self._lora_adapters.keys()):
                merge_lora_weights(self, self._lora_adapters[name])
                del self._lora_adapters[name]
                logger.info(f"Merged LoRA adapter: {name}")
        else:
            # Merge specific adapter
            if adapter_name not in self._lora_adapters:
                raise ValueError(f"LoRA adapter '{adapter_name}' not found")
            merge_lora_weights(self, self._lora_adapters[adapter_name])
            del self._lora_adapters[adapter_name]
            logger.info(f"Merged LoRA adapter: {adapter_name}")
    
    # =========================================================================
    # � MODEL EXPORT METHODS
    # =========================================================================
    
    def export_to_safetensors(self, path: Union[str, Path]) -> None:
        """
        Export model to Safetensors format.
        
        📖 WHAT THIS DOES:
        Saves the model weights in Safetensors format, which is:
        - Faster to load than pickle-based formats
        - Safer (no arbitrary code execution)
        - Compatible with many frameworks (HuggingFace, etc.)
        
        Args:
            path: Output path for .safetensors file
        
        Example:
            model.export_to_safetensors("model.safetensors")
            # Also creates model.json with config
        """
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ImportError(
                "Safetensors export requires safetensors library. "
                "Install with: pip install safetensors"
            )
        
        path = Path(path)
        
        # Save weights
        save_file(self.state_dict(), str(path))
        logger.info(f"Exported weights to: {path}")
        
        # Save config alongside
        config_path = path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Exported config to: {config_path}")
    
    def export_to_onnx(
        self, 
        path: Union[str, Path], 
        opset_version: int = 14,
        input_names: Optional[list[str]] = None,
        output_names: Optional[list[str]] = None,
        dynamic_axes: Optional[dict[str, dict[int, str]]] = None
    ) -> None:
        """
        Export model to ONNX format for deployment.
        
        📖 WHAT THIS DOES:
        Exports the model to ONNX format, enabling:
        - Cross-platform deployment (C++, mobile, web)
        - Hardware-specific optimizations (TensorRT, OpenVINO)
        - Framework-agnostic inference
        
        ⚠️ NOTES:
        - KV-cache based generation is not directly supported in ONNX
        - Export captures the model at a fixed sequence length
        - Dynamic axes allow variable batch/sequence sizes
        
        Args:
            path: Output path for .onnx file
            opset_version: ONNX opset version (14 is widely supported)
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dict specifying variable dimensions
        
        Example:
            model.export_to_onnx("model.onnx")
            # Use with ONNX Runtime for fast inference
        """
        path = Path(path)
        
        # Default configurations
        input_names = input_names or ['input_ids']
        output_names = output_names or ['logits']
        dynamic_axes = dynamic_axes or {
            'input_ids': {0: 'batch', 1: 'sequence'},
            'logits': {0: 'batch', 1: 'sequence'}
        }
        
        # Create dummy input (representative of actual input)
        dummy_input = torch.randint(
            0, self.vocab_size, 
            (1, min(128, self.max_len)),
            device=next(self.parameters()).device
        )
        
        # Export
        logger.info(f"Exporting to ONNX (opset {opset_version})...")
        torch.onnx.export(
            self,
            dummy_input,
            str(path),
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,  # Optimize constants
        )
        logger.info(f"Exported to ONNX: {path}")
        
        # Save config alongside
        config_path = path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Exported config to: {config_path}")
    
    def export_to_pytorch(self, path: Union[str, Path]) -> None:
        """
        Export model to standard PyTorch format (.pth).
        
        📖 WHAT THIS DOES:
        Saves model weights and config in native PyTorch format.
        This is the default format used by Enigma AI Engine.
        
        Args:
            path: Output path for .pth file
        
        Example:
            model.export_to_pytorch("models/my_model.pth")
        """
        path = Path(path)
        
        # Save weights
        torch.save(self.state_dict(), path)
        logger.info(f"Exported weights to: {path}")
        
        # Save config alongside
        config_path = path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Exported config to: {config_path}")
    
    # =========================================================================
    # �🚀 SPECULATIVE DECODING
    # =========================================================================
    
    def enable_speculative_decoding(
        self, 
        draft_model: 'Forge',
        num_speculative_tokens: int = 4
    ) -> None:
        """
        Enable speculative decoding for faster generation.
        
        📖 WHAT THIS DOES:
        Uses a smaller "draft" model to predict multiple tokens quickly,
        then verifies them with the main model in parallel. Can be 2-4x faster!
        
        📐 HOW IT WORKS:
        1. Draft model generates K tokens quickly (small model = fast)
        2. Main model verifies all K tokens in one forward pass (parallel!)
        3. Accept correct tokens, reject and regenerate incorrect ones
        
        Args:
            draft_model: Smaller, faster model for speculation
            num_speculative_tokens: How many tokens to speculate (2-8 typical)
        
        Example:
            small_model = create_model('tiny')
            large_model = create_model('large')
            large_model.enable_speculative_decoding(small_model, num_speculative_tokens=4)
        """
        self._draft_model = draft_model
        self._num_speculative_tokens = num_speculative_tokens
        self._use_speculation = True
        logger.info(
            f"Enabled speculative decoding with {num_speculative_tokens} tokens"
        )
    
    def disable_speculative_decoding(self) -> None:
        """Disable speculative decoding and return to standard generation."""
        self._use_speculation = False
        self._draft_model = None
        logger.info("Disabled speculative decoding")
    
    @torch.no_grad()
    def generate_speculative(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate tokens using speculative decoding.
        
        📖 WHAT THIS DOES:
        Faster generation using a draft model for speculation.
        Falls back to standard generation if no draft model is set.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            **kwargs: Generation parameters (temperature, top_k, etc.)
        
        Returns:
            Generated token IDs [batch, seq_len + new_tokens]
        """
        if not hasattr(self, '_use_speculation') or not self._use_speculation:
            # Fall back to standard generation
            return self.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)
        
        draft_model = self._draft_model
        num_spec = self._num_speculative_tokens
        
        generated = input_ids
        tokens_generated = 0
        
        while tokens_generated < max_new_tokens:
            # Step 1: Draft model generates K tokens
            draft_tokens = draft_model.generate(
                generated,
                max_new_tokens=num_spec,
                **kwargs
            )
            
            # Step 2: Main model verifies all K tokens in one pass
            # Concatenate draft tokens to input
            candidate_ids = torch.cat([generated, draft_tokens[:, generated.shape[1]:]], dim=1)
            
            # Get probabilities from main model
            logits = self.forward(candidate_ids)
            probs = F.softmax(logits[:, -num_spec-1:-1, :], dim=-1)
            
            # Step 3: Accept or reject each token
            accepted = 0
            for i in range(num_spec):
                draft_token = draft_tokens[:, generated.shape[1] + i]
                main_prob = probs[:, i, draft_token]
                
                # Simple acceptance: if probability is high enough, accept
                if main_prob > 0.5:  # Threshold can be tuned
                    accepted += 1
                else:
                    break
            
            # Add accepted tokens
            if accepted > 0:
                generated = candidate_ids[:, :generated.shape[1] + accepted]
                tokens_generated += accepted
            else:
                # No tokens accepted, generate one with main model
                next_token_logits = logits[:, -1, :]
                next_token = torch.multinomial(
                    F.softmax(next_token_logits / kwargs.get('temperature', 0.8), dim=-1),
                    num_samples=1
                )
                generated = torch.cat([generated, next_token], dim=1)
                tokens_generated += 1
        
        return generated


# =============================================================================
# 🔧 HARDWARE DETECTION HELPERS
# =============================================================================

def detect_hardware() -> dict[str, Any]:
    """
    Detect hardware capabilities for model configuration.
    
    Returns dict with: total_ram_gb, gpu_vram_gb, is_raspberry_pi, is_arm, etc.
    """
    try:
        from .hardware_detection import detect_hardware as _detect
        profile = _detect()
        return profile.to_dict()
    except ImportError:
        # Fallback basic detection
        import os
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


@staticmethod
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


# =============================================================================
# Aliases (for backwards compatibility)
# =============================================================================

# Legacy aliases - the main class is 'Enigma', these are for backwards compatibility
# Kept for compatibility with old code that used different names
Forge = Enigma  # Alias: some old code used 'Forge'
TinyForge = Enigma  # Deprecated alias
ForgeModel = Enigma  # Deprecated alias

# =============================================================================
# Factory Functions
# =============================================================================

def create_model(size: str = 'small', vocab_size: Optional[int] = None, **kwargs) -> Enigma:
    """
    Create an Enigma model from a preset configuration.

    Args:
        size: Model size preset (tiny, small, medium, large, xl, etc.)
        vocab_size: Size of vocabulary. If None, auto-detects from default tokenizer.
        **kwargs: Additional config overrides (unknown keys are logged and ignored)

    Returns:
        Configured Enigma model instance

    Raises:
        ValueError: If size is invalid or vocab_size is invalid
        TypeError: If size is not a string or vocab_size is not an integer
        RuntimeError: If model initialization fails

    Example:
        >>> model = create_model('small', vocab_size=8000)
        >>> model = create_model('medium', dropout=0.2)
    """
    # Validate inputs
    if not isinstance(size, str):
        raise TypeError(f"size must be a string, got {type(size).__name__}")

    # Auto-detect vocab size from tokenizer if not specified
    if vocab_size is None:
        try:
            from .tokenizer import get_tokenizer
            tok = get_tokenizer()
            vocab_size = tok.vocab_size
            logger.info(f"Auto-detected vocab_size={vocab_size} from tokenizer")
        except Exception as e:
            logger.warning(f"Could not auto-detect vocab_size: {e}, using default 8000")
            vocab_size = 8000

    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError(f"vocab_size must be a positive integer, got {vocab_size}")

    if vocab_size > 1000000:
        logger.warning(f"Very large vocab_size ({vocab_size:,}). This may use excessive memory.")

    # Get preset configuration (raises ValueError if size is invalid)
    try:
        config = get_preset(size, vocab_size)
    except ValueError as e:
        logger.error(f"Failed to create model: {e}")
        raise

    # Apply kwargs overrides with validation
    for k, v in kwargs.items():
        if not hasattr(config, k):
            logger.warning(f"Unknown config parameter '{k}' - ignoring")
            continue
        setattr(config, k, v)

    # Create model
    try:
        model = Enigma(config=config)
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise RuntimeError(f"Model creation failed: {e}") from e

    logger.info(f"Created Enigma ({size}): {model.num_parameters:,} params, "
          f"{config.dim}d, {config.n_layers}L")
    return model


# Additional aliases (already defined above, these are duplicates - keeping for safety)
# Forge = Enigma  # Already defined above
# TinyForge = Enigma  # Already defined above
# ForgeModel = Enigma  # Already defined above


def create_model_auto(size: str = 'small', vocab_size: Optional[int] = None, **kwargs):
    """
    Create a model with automatic backend selection.
    
    For nano/micro models, may use pure Python backend if PyTorch is unavailable
    or if configured to do so. For larger models, always uses PyTorch.
    
    Args:
        size: Model size preset (nano, micro, tiny, small, medium, large, xl, etc.)
        vocab_size: Size of vocabulary. If None, auto-detects from default tokenizer.
        **kwargs: Additional config overrides
        
    Returns:
        Model instance (Forge for PyTorch, PureTransformer for pure Python)
    """
    from ..config import CONFIG

    # Get backend preference from config
    backend = CONFIG.get("nn_backend", "auto")
    threshold = CONFIG.get("nn_backend_threshold", 5_000_000)
    
    # Estimate param count for this size
    SIZE_PARAMS = {
        "nano": 200_000,
        "micro": 1_000_000,
        "tiny": 5_000_000,
        "small": 25_000_000,
        "medium": 85_000_000,
        "large": 300_000_000,
    }
    estimated_params = SIZE_PARAMS.get(size, 25_000_000)
    
    # Determine which backend to use
    # NEW PRIORITY: Pure Python + Numba is PRIMARY, PyTorch is fallback for large models
    use_pure = True  # Default to pure Python + Numba
    
    if backend == "torch":
        # User explicitly requested PyTorch
        use_pure = False
        print(f"[Backend] PyTorch requested explicitly for {size}")
    elif backend == "pure":
        # User explicitly requested pure Python
        use_pure = True
        print(f"[Backend] Pure Python requested explicitly for {size}")
    elif backend == "auto":
        # Auto mode: Use pure for small/medium, PyTorch for large (GPU-scale)
        if estimated_params > threshold:
            # Large model - try PyTorch if available
            try:
                import torch
                use_pure = False
                print(f"[Backend] FALLBACK to PyTorch - model too large ({estimated_params/1e6:.0f}M > {threshold/1e6:.0f}M threshold)")
                if torch.cuda.is_available():
                    print(f"[Backend] GPU detected: {torch.cuda.get_device_name(0)}")
                else:
                    print(f"[Backend] No GPU - PyTorch will use CPU")
            except ImportError:
                use_pure = True
                print(f"[Backend] PyTorch not installed - using Pure Python + Numba for large model")
        else:
            # Small/medium model - Pure Python + Numba is efficient enough
            use_pure = True
    
    if use_pure:
        # Use pure Python backend
        try:
            from ..builtin.neural_network import get_model_for_size, set_backend
            set_backend("pure", threshold)
            return get_model_for_size(size)
        except Exception as e:
            print(f"[Backend] FALLBACK to PyTorch - Pure Python failed: {e}")
            logger.warning(f"Pure Python backend failed: {e}, falling back to PyTorch")
    
    # Use PyTorch backend (fallback)
    print(f"[Backend] Using PyTorch for {size}")
    return create_model(size, vocab_size, **kwargs)


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

# Keep Forge as an alias for existing code
Forge = Enigma
