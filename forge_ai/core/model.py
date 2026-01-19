"""
================================================================================
🧠 FORGE MODEL - THE BRAIN OF FORGEAI
================================================================================

This is the HEART of ForgeAI - a production-grade transformer neural network!
This is where the actual AI "thinking" happens.

📍 FILE: forge_ai/core/model.py
🏷️ TYPE: Neural Network Architecture
🎯 MAIN CLASSES: Forge, ForgeConfig

┌─────────────────────────────────────────────────────────────────────────────┐
│  ARCHITECTURE DIAGRAM:                                                      │
│                                                                             │
│  Input Text → [Tokenizer] → Numbers                                        │
│       ↓                                                                     │
│  [Embedding Layer] - Converts numbers to vectors                           │
│       ↓                                                                     │
│  [Transformer Blocks] × N layers                                           │
│    ├── RMSNorm (normalization - faster than LayerNorm)                     │
│    ├── Self-Attention with RoPE (understanding context)                    │
│    ├── SwiGLU Activation (better than ReLU!)                               │
│    └── Residual connections                                                │
│       ↓                                                                     │
│  [Output Head] → Next word probabilities                                   │
└─────────────────────────────────────────────────────────────────────────────┘

⚡ KEY FEATURES:
    • RoPE (Rotary Position Embeddings) - Better position awareness
    • RMSNorm - Faster and more stable than LayerNorm  
    • SwiGLU - Superior activation function
    • GQA (Grouped Query Attention) - Memory efficient
    • KV-Cache - Fast autoregressive generation

📊 MODEL SIZES (15 presets!):
    ┌────────────┬──────────┬────────────────────────────────┐
    │ Size       │ Params   │ Best For                       │
    ├────────────┼──────────┼────────────────────────────────┤
    │ nano       │ ~1M      │ Embedded/Testing               │
    │ tiny       │ ~5M      │ Raspberry Pi                   │
    │ small      │ ~27M     │ Desktop default (RTX 2080)     │
    │ medium     │ ~85M     │ Good balance (RTX 3080)        │
    │ large      │ ~200M    │ Quality focus (RTX 4090)       │
    │ xl         │ ~600M    │ Multi-GPU                      │
    │ xxl        │ ~1.5B    │ Cloud/Datacenter               │
    └────────────┴──────────┴────────────────────────────────┘

🔗 CONNECTED FILES:
    → USES:      forge_ai/config/ (CONFIG settings)
    ← USED BY:   forge_ai/core/inference.py (ForgeEngine loads this)
    ← USED BY:   forge_ai/core/training.py (trains this model)
    ← USED BY:   forge_ai/modules/registry.py (ModelModule wraps this)

📖 USAGE:
    from forge_ai.core.model import create_model, Forge, ForgeConfig
    
    model = create_model('small')  # Use preset
    # OR custom:
    config = ForgeConfig(vocab_size=8000, dim=512, n_layers=8)
    model = Forge(config)

📖 SEE ALSO:
    • forge_ai/core/inference.py - To GENERATE text with this model
    • forge_ai/core/training.py  - To TRAIN this model
    • forge_ai/core/tokenizer.py - Converts text ↔ numbers
"""
import math
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
from pathlib import Path

from ..config import CONFIG

logger = logging.getLogger(__name__)

MAX_LEN = CONFIG.get("max_len", 1024)

# Global registry of loaded models
_LOADED_MODELS: Dict[str, 'Forge'] = {}


def get_running_models() -> Dict[str, 'Forge']:
    """Get all loaded model instances."""
    return _LOADED_MODELS.copy()


def is_model_loaded(name: str) -> bool:
    """Check if a model is loaded."""
    return name in _LOADED_MODELS


def register_model(name: str, model: 'Forge'):
    """Register a model instance."""
    _LOADED_MODELS[name] = model


def unregister_model(name: str):
    """Unregister a model."""
    _LOADED_MODELS.pop(name, None)


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
    # LEGACY ALIASES - For backwards compatibility
    # ─────────────────────────────────────────────────────────────────────────
    depth: Optional[int] = None      # Old name for n_layers
    heads: Optional[int] = None      # Old name for n_heads
    max_len: Optional[int] = None    # Old name for max_seq_len
    embed_dim: Optional[int] = None  # Old name for dim

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
            raise ValueError(
                f"n_heads ({self.n_heads}) must divide evenly into dim ({self.dim}). "
                f"Got remainder: {self.dim % self.n_heads}"
            )
        
        # n_heads must be divisible by n_kv_heads (for GQA grouping)
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_kv_heads ({self.n_kv_heads}) must divide evenly into n_heads ({self.n_heads}). "
                f"Got remainder: {self.n_heads % self.n_kv_heads}"
            )

    def to_dict(self) -> Dict[str, Any]:
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
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ForgeConfig':
        known = {
            'vocab_size', 'dim', 'n_layers', 'n_heads', 'n_kv_heads',
            'hidden_dim', 'max_seq_len', 'dropout', 'use_rope', 'use_rms_norm',
            'use_swiglu', 'use_bias', 'rope_theta', 'depth', 'heads',
            'max_len', 'embed_dim'
        }
        return cls(**{k: v for k, v in d.items() if k in known})


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
        config.vocab_size = 32000  # Standard for estimation
        result[name] = {
            'description': MODEL_DESCRIPTIONS.get(name, ""),
            'estimated_params': estimate_parameters(config),
            'dim': config.dim,
            'layers': config.n_layers,
            'heads': config.n_heads,
            'max_seq_len': config.max_seq_len,
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
# 🌀 ROTARY POSITION EMBEDDINGS (RoPE) - How the model knows word order
# =============================================================================
# Without position info, "dog bites man" = "man bites dog" to the model!
# RoPE encodes position by ROTATING the vectors - elegant and effective.

def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute RoPE frequencies for all positions.
    
    📖 WHAT THIS DOES:
    Creates a table of rotation angles for each position and dimension.
    These rotations encode "position 0", "position 1", etc.
    
    📐 THE MATH:
    For dimension pair i, frequency = 1 / (theta^(2i/dim))
    For position p, angle = p * frequency
    
    💡 WHY THIS WORKS:
    - Different dimensions get different rotation speeds
    - Position 5 at dim 0 rotates differently than position 5 at dim 10
    - Model can learn to "read" these rotations to understand order
    
    Returns:
        Complex tensor of shape [max_seq_len, dim/2] with rotation values
    """
    # Calculate frequencies: lower dimensions rotate faster
    # freqs[i] = 1 / (theta^(2i/dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    
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

        # Transpose for batched matrix multiply: [batch, heads, seq, dim]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 5: Compute attention scores and apply mask
        # ─────────────────────────────────────────────────────────────────────
        # scores = Q @ K.T / sqrt(head_dim) - scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # Mask is -inf for blocked positions

        # ─────────────────────────────────────────────────────────────────────
        # STEP 6: Softmax and dropout, then weighted sum of values
        # ─────────────────────────────────────────────────────────────────────
        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, v)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 7: Concatenate heads and project back
        # ─────────────────────────────────────────────────────────────────────
        return self.wo(output.transpose(1, 2).contiguous().view(B, T, -1))

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
    """

    def __init__(self, config: ForgeConfig, layer_id: int):
        """
        Args:
            config: Model configuration
            layer_id: Which layer this is (for debugging/logging)
        """
        super().__init__()
        self.layer_id = layer_id

        # Choose normalization type based on config
        Norm = RMSNorm if config.use_rms_norm else nn.LayerNorm
        
        # Two normalizations: one before attention, one before FFN
        self.attention_norm = Norm(config.dim)
        self.ffn_norm = Norm(config.dim)
        
        # The actual computation modules
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)

    def forward(
        self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None, use_cache: bool = False, start_pos: int = 0
    ) -> torch.Tensor:
        """
        Forward pass: Norm → Attention → Add → Norm → FFN → Add
        
        Args:
            x: Input [batch, seq_len, dim]
            freqs_cis: RoPE frequencies
            mask: Causal attention mask
            use_cache: Whether to use KV-cache
            start_pos: Position for KV-cache
        
        Returns:
            Output tensor, same shape as input
        """
        # Attention sub-layer with residual connection
        # h = x + Attention(Norm(x))
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, use_cache, start_pos)
        # FFN sub-layer with residual connection
        # output = h + FFN(Norm(h))
        return h + self.feed_forward(self.ffn_norm(h))

    def clear_cache(self):
        """Clear KV-cache in the attention layer."""
        self.attention.clear_cache()


# =============================================================================
# 🧠 MAIN MODEL - THE FULL TRANSFORMER
# =============================================================================

class Forge(nn.Module):
    """
    Forge - Modern Transformer Language Model
    
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
      ← Used by ForgeEngine for inference
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

        # RoPE frequencies
        if self.config.use_rope:
            self.register_buffer(
                'freqs_cis',
                precompute_rope_frequencies(
                    self.config.dim // self.config.n_heads,
                    self.config.max_seq_len * 2,
                    self.config.rope_theta
                )
            )
        else:
            self.freqs_cis = None

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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
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

        mask = None
        if T > 1:
            mask = torch.full((T, T), float('-inf'), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

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

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int = 100,
        temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
        repetition_penalty: float = 1.1, stop_tokens: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        self.clear_cache()
        stop_tokens = stop_tokens or [2]

        generated = input_ids
        logits = self.forward(input_ids, use_cache=True)

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    for tok in set(generated[i].tolist()):
                        if 0 <= tok < next_logits.shape[1]:
                            next_logits[i, tok] /= repetition_penalty

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

        return generated

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        return self.config.to_dict()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Forge':
        return cls(config=ForgeConfig.from_dict(config))

    @classmethod
    def from_pretrained(cls, path: Path) -> 'Forge':
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


# =============================================================================
# Aliases (for backwards compatibility)
# =============================================================================

# Primary alias - 'Forge' is the original name
Forge = Forge

# =============================================================================
# Factory Functions
# =============================================================================

def create_model(size: str = 'small', vocab_size: Optional[int] = None, **kwargs) -> Forge:
    """
    Create an Forge model from a preset configuration.

    Args:
        size: Model size preset (tiny, small, medium, large, xl, etc.)
        vocab_size: Size of vocabulary. If None, auto-detects from default tokenizer.
        **kwargs: Additional config overrides (unknown keys are logged and ignored)

    Returns:
        Configured Forge model instance

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
        model = Forge(config=config)
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise RuntimeError(f"Model creation failed: {e}") from e

    print(f"Created Forge ({size}): {model.num_parameters:,} params, "
          f"{config.dim}d, {config.n_layers}L")
    return model


# Additional aliases for backwards compatibility
TinyForge = Forge
ForgeModel = Forge
Enigma = Forge  # Legacy name used in tests and documentation
