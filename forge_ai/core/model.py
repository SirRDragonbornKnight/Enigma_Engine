"""
Forge Model - Unified Transformer Architecture
================================================

A production-grade transformer that scales from tiny to extra-large:
  - tiny:   256 dim,  6 layers -> ~5M params   (Pi 4/5)
  - small:  512 dim,  8 layers -> ~27M params  (RTX 2080)
  - medium: 768 dim, 12 layers -> ~85M params  (RTX 3080)
  - large: 1024 dim, 16 layers -> ~200M params (RTX 4090)
  - xl:    1536 dim, 24 layers -> ~600M params (Multi-GPU)
  - xxl:   2048 dim, 32 layers -> ~1.5B params (Cloud)

Architecture based on LLaMA/Mistral with:
  - Rotary Position Embeddings (RoPE)
  - RMSNorm (faster than LayerNorm)
  - SwiGLU activation
  - Grouped Query Attention (GQA)
  - KV Cache for fast generation

This WILL learn with enough data and training.
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
# Configuration
# =============================================================================

@dataclass
class ForgeConfig:
    """Model configuration with sensible defaults."""
    vocab_size: int = 8000
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    hidden_dim: Optional[int] = None
    max_seq_len: int = 1024
    dropout: float = 0.1
    use_rope: bool = True
    use_rms_norm: bool = True
    use_swiglu: bool = True
    use_bias: bool = False
    rope_theta: float = 10000.0

    # Legacy aliases
    depth: Optional[int] = None
    heads: Optional[int] = None
    max_len: Optional[int] = None
    embed_dim: Optional[int] = None

    def __post_init__(self):
        # Map legacy names
        if self.depth:
            self.n_layers = self.depth
        if self.heads:
            self.n_heads = self.heads
        if self.max_len:
            self.max_seq_len = self.max_len
        if self.embed_dim:
            self.dim = self.embed_dim

        # Defaults
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        if self.hidden_dim is None:
            if self.use_swiglu:
                self.hidden_dim = int(2 * (4 * self.dim) / 3)
                self.hidden_dim = 64 * ((self.hidden_dim + 63) // 64)
            else:
                self.hidden_dim = 4 * self.dim

        # Validate parameters
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
            raise ValueError(
                f"n_heads ({self.n_heads}) must divide evenly into dim ({self.dim}). "
                f"Got remainder: {self.dim % self.n_heads}"
            )
        
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
# Model Presets - From Raspberry Pi to Server Farm
# =============================================================================

MODEL_PRESETS = {
    # Embedded / IoT (~1-2M params)
    'nano': ForgeConfig(dim=128, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=256),
    'micro': ForgeConfig(dim=192, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=384),

    # Edge / Raspberry Pi (~5-15M params)
    'tiny': ForgeConfig(dim=256, n_layers=6, n_heads=8, n_kv_heads=4, max_seq_len=512),
    'mini': ForgeConfig(dim=384, n_layers=6, n_heads=6, n_kv_heads=3, max_seq_len=512),

    # Consumer GPU (~27-85M params)
    'small': ForgeConfig(dim=512, n_layers=8, n_heads=8, n_kv_heads=4, max_seq_len=1024),
    'medium': ForgeConfig(dim=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=2048),
    'base': ForgeConfig(dim=896, n_layers=14, n_heads=14, n_kv_heads=2, max_seq_len=2048),

    # Prosumer GPU (~200M-600M params)
    'large': ForgeConfig(dim=1024, n_layers=16, n_heads=16, n_kv_heads=4, max_seq_len=4096),
    'xl': ForgeConfig(dim=1536, n_layers=24, n_heads=24, n_kv_heads=6, max_seq_len=4096, dropout=0.05),

    # Multi-GPU / Server (~1B-3B params)
    'xxl': ForgeConfig(dim=2048, n_layers=32, n_heads=32, n_kv_heads=8, max_seq_len=8192, dropout=0.05),
    'huge': ForgeConfig(dim=2560, n_layers=40, n_heads=40, n_kv_heads=8, max_seq_len=8192, dropout=0.05),

    # Datacenter / Cloud (~7B-13B params)
    'giant': ForgeConfig(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, max_seq_len=8192, dropout=0.05),
    'colossal': ForgeConfig(dim=4096, n_layers=48, n_heads=32, n_kv_heads=8, max_seq_len=16384, dropout=0.05),

    # Maximum Scale (~30B+ params)
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
# Model Components
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - faster than LayerNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len)
    angles = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rotary_embedding(
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0) -> torch.Tensor:
    """Apply rotary embeddings to Q and K."""
    seq_len = x.shape[1]
    freqs = freqs_cis[start_pos:start_pos + seq_len]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


class Attention(nn.Module):
    """Multi-Head Attention with Grouped Query Attention (GQA)."""
    
    # Maximum KV-cache size (sliding window for memory efficiency)
    MAX_CACHE_SEQ_LEN = 4096

    def __init__(self, config: ForgeConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        
        # Cache size limit from config or default
        self.max_cache_len = min(
            config.max_seq_len if hasattr(config, 'max_seq_len') else self.MAX_CACHE_SEQ_LEN,
            self.MAX_CACHE_SEQ_LEN
        )

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=config.use_bias)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=config.use_bias)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=config.use_bias)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=config.use_bias)

        self.dropout = nn.Dropout(config.dropout)
        self.use_rope = config.use_rope

        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None, use_cache: bool = False, start_pos: int = 0
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        if self.use_rope and freqs_cis is not None:
            q = apply_rotary_embedding(q, freqs_cis, start_pos)
            k = apply_rotary_embedding(k, freqs_cis, start_pos)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=1)
                self.cache_v = torch.cat([self.cache_v, v], dim=1)
                
                # Enforce cache size limit (sliding window)
                if self.cache_k.shape[1] > self.max_cache_len:
                    # Keep most recent tokens
                    trim_amount = self.cache_k.shape[1] - self.max_cache_len
                    self.cache_k = self.cache_k[:, trim_amount:, :, :]
                    self.cache_v = self.cache_v[:, trim_amount:, :, :]
                    
            k, v = self.cache_k, self.cache_v

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, v)

        return self.wo(output.transpose(1, 2).contiguous().view(B, T, -1))

    def clear_cache(self):
        self.cache_k = self.cache_v = None


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: ForgeConfig):
        super().__init__()
        self.use_swiglu = config.use_swiglu

        if self.use_swiglu:
            self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=config.use_bias)
            self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=config.use_bias)
            self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=config.use_bias)
        else:
            self.up = nn.Linear(config.dim, config.hidden_dim, bias=config.use_bias)
            self.down = nn.Linear(config.hidden_dim, config.dim, bias=config.use_bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))
        return self.down(self.dropout(F.gelu(self.up(x))))


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(self, config: ForgeConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id

        Norm = RMSNorm if config.use_rms_norm else nn.LayerNorm
        self.attention_norm = Norm(config.dim)
        self.ffn_norm = Norm(config.dim)
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)

    def forward(
        self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None, use_cache: bool = False, start_pos: int = 0
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, use_cache, start_pos)
        return h + self.feed_forward(self.ffn_norm(h))

    def clear_cache(self):
        self.attention.clear_cache()


# =============================================================================
# Main Model
# =============================================================================

class Forge(nn.Module):
    """
    Forge - Modern Transformer Language Model

    Based on LLaMA/Mistral architecture with:
      - RoPE positional embeddings
      - RMSNorm
      - SwiGLU activation
      - GQA attention
      - KV cache
    """

    def __init__(
        self, vocab_size: int = 8000, dim: Optional[int] = None,
        depth: Optional[int] = None, heads: Optional[int] = None,
        max_len: Optional[int] = None, config: Optional[ForgeConfig] = None, **kwargs
    ):
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

def create_model(size: str = 'small', vocab_size: int = 8000, **kwargs) -> Forge:
    """
    Create an Forge model from a preset configuration.

    Args:
        size: Model size preset (tiny, small, medium, large, xl, etc.)
        vocab_size: Size of vocabulary (must be > 0)
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
