"""
Neural network components for the Enigma transformer.

Contains RMSNorm, RoPE functions, DropPath, Attention, FeedForward,
MoEFeedForward, and TransformerBlock modules.
"""
import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_presets import ForgeConfig

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

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
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
        # Compute in float32 for numerical stability in fp16/bf16 training
        orig_dtype = x.dtype
        x_f32 = x.float()
        rms = torch.sqrt(torch.mean(x_f32 ** 2, dim=-1, keepdim=True) + self.eps)
        return (x_f32 / rms * self.weight.float()).to(orig_dtype)


class DropPath(nn.Module):
    """Stochastic depth — drops entire residual branches during training.

    Each sample in the batch is independently kept (scaled up) or zeroed.
    At eval time this is a no-op.  ``drop_prob`` should increase linearly
    with layer depth (deeper layers → higher drop rate).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        # Random tensor per sample: shape (B, 1, 1) so it blankets the whole sample
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


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
        if dim < 4:
            raise ValueError(
                f"RoPE dynamic scaling requires dim >= 4, got {dim}"
            )
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

    def __init__(self, config: ForgeConfig) -> None:
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
        self.use_qk_norm = getattr(config, 'use_qk_norm', False)

        # Learned QK norms (Qwen3-style RMSNorm per head)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # ─────────────────────────────────────────────────────────────────────
        # KV-CACHE: Pre-allocated for O(1) per-token writes during generation
        # Uses the optimized KVCache from kv_cache.py instead of torch.cat()
        # which caused O(n) reallocation every token.
        # ─────────────────────────────────────────────────────────────────────
        self._kv_cache: Optional[object] = None

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
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Apply RoPE position embeddings to Q and K
        # ─────────────────────────────────────────────────────────────────────
        if self.use_rope and freqs_cis is not None:
            q = apply_rotary_embedding(q, freqs_cis, start_pos)
            k = apply_rotary_embedding(k, freqs_cis, start_pos)

        # QK normalization: prevents fp16 attention overflow on long sequences
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Handle KV-cache (for efficient generation)
        # ─────────────────────────────────────────────────────────────────────
        if use_cache:
            # Detach K, V from computation graph to prevent memory explosion
            # if someone accidentally backprops with use_cache=True
            k = k.detach()
            v = v.detach()

            # Lazy-init pre-allocated cache on first use
            if self._kv_cache is None:
                from enigma_engine.core.kv_cache import KVCache
                self._kv_cache = KVCache(
                    batch_size=B,
                    max_seq_len=self.max_cache_len,
                    n_kv_heads=self.n_kv_heads,
                    head_dim=self.head_dim,
                    device=k.device,
                    dtype=k.dtype,
                )

            # O(1) index write instead of O(n) torch.cat() + realloc
            self._kv_cache.update(k, v)
            k, v = self._kv_cache.get()

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
            output = output.reshape(B, T, -1)
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
            output = output.transpose(1, 2).reshape(B, T, -1)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 6: Project back to model dimension
        # ─────────────────────────────────────────────────────────────────────
        return self.wo(output)

    def clear_cache(self) -> None:
        """Clear the KV-cache (call between different sequences)."""
        if self._kv_cache is not None:
            self._kv_cache.clear()
        self._kv_cache = None


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

    def __init__(self, config: ForgeConfig) -> None:
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

    def __init__(self, config: ForgeConfig) -> None:
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
        x_flat = x.reshape(-1, D)  # [num_tokens, D]

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
            flat_indices = top_k_indices.reshape(-1)  # [num_tokens * k]
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
        return output.reshape(B, T, D)

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

    def __init__(self, config: ForgeConfig, layer_id: int) -> None:
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

        # LayerScale: learnable per-channel scaling of residual outputs
        # Initialized to a tiny value so early training is stable
        self.use_layer_scale = getattr(config, 'use_layer_scale', False)
        if self.use_layer_scale:
            self.ls_attn = nn.Parameter(torch.full((config.dim,), 1e-5))
            self.ls_ffn = nn.Parameter(torch.full((config.dim,), 1e-5))

        # Drop path (stochastic depth): linearly increasing per layer
        drop_rate = getattr(config, 'drop_path_rate', 0.0)
        n_layers = getattr(config, 'n_layers', 1)
        layer_drop = drop_rate * layer_id / max(n_layers - 1, 1) if drop_rate > 0 else 0.0
        self.drop_path_attn = DropPath(layer_drop)
        self.drop_path_ffn = DropPath(layer_drop)

    def _forward_impl(
        self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None, use_cache: bool = False, start_pos: int = 0
    ) -> torch.Tensor:
        """Internal forward implementation (used by checkpointing)."""
        # Attention sub-layer with residual connection
        attn_out = self.attention(self.attention_norm(x), freqs_cis, mask, use_cache, start_pos)
        if self.use_layer_scale:
            attn_out = attn_out * self.ls_attn
        h = x + self.drop_path_attn(attn_out)
        # FFN sub-layer with residual connection
        ffn_out = self.feed_forward(self.ffn_norm(h))
        if self.use_layer_scale:
            ffn_out = ffn_out * self.ls_ffn
        return h + self.drop_path_ffn(ffn_out)

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

    def clear_cache(self) -> None:
        """Clear KV-cache in the attention layer."""
        self.attention.clear_cache()

    def get_moe_aux_loss(self) -> torch.Tensor:
        """Get MoE auxiliary loss for load balancing during training."""
        if self.use_moe and hasattr(self.feed_forward, 'get_aux_loss'):
            return self.feed_forward.get_aux_loss()
        return torch.tensor(0.0)
