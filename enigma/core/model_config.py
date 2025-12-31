"""
Model configuration presets for different sizes.
Choose based on your hardware capabilities.

USAGE:
    from enigma.core.model_config import MODEL_PRESETS, get_model_config

    config = get_model_config("medium")  # or "tiny", "small", "large", "xl"
    model = Enigma(**config)
"""

# Model size presets
# Parameters ≈ vocab_size * dim + dim * dim * depth * 12 (approximate, with weight tying)

MODEL_PRESETS = {
    # ~2M params - Raspberry Pi, mobile, testing
    "tiny": {
        "dim": 128,
        "depth": 4,
        "heads": 4,
        "max_len": 512,
        "ff_mult": 4.0,
        "description": "Tiny model for testing/mobile (~2M params)",
        "min_ram_gb": 1,
        "min_vram_gb": 0,
    },

    # ~15M params - Single GPU, laptop
    "small": {
        "dim": 256,
        "depth": 6,
        "heads": 8,
        "max_len": 1024,
        "ff_mult": 4.0,
        "description": "Small model for learning (~15M params)",
        "min_ram_gb": 4,
        "min_vram_gb": 2,
    },

    # ~50M params - Good GPU needed
    "medium": {
        "dim": 512,
        "depth": 8,
        "heads": 8,
        "max_len": 2048,
        "ff_mult": 4.0,
        "description": "Medium model for real use (~50M params)",
        "min_ram_gb": 8,
        "min_vram_gb": 4,
    },

    # ~125M params - Like GPT-2 small
    "large": {
        "dim": 768,
        "depth": 12,
        "heads": 12,
        "max_len": 2048,
        "ff_mult": 4.0,
        "description": "Large model (~125M params, GPT-2 small equivalent)",
        "min_ram_gb": 16,
        "min_vram_gb": 8,
    },

    # ~350M params - Like GPT-2 medium
    "xl": {
        "dim": 1024,
        "depth": 24,
        "heads": 16,
        "max_len": 2048,
        "ff_mult": 4.0,
        "description": "XL model (~350M params, GPT-2 medium equivalent)",
        "min_ram_gb": 32,
        "min_vram_gb": 12,
    },

    # ~770M params - Like GPT-2 large
    "xxl": {
        "dim": 1280,
        "depth": 36,
        "heads": 20,
        "max_len": 2048,
        "ff_mult": 4.0,
        "description": "XXL model (~770M params, GPT-2 large equivalent)",
        "min_ram_gb": 64,
        "min_vram_gb": 24,
    },

    # ~1.5B params - Like GPT-2 XL
    "xxxl": {
        "dim": 1600,
        "depth": 48,
        "heads": 25,
        "max_len": 2048,
        "ff_mult": 4.0,
        "description": "XXXL model (~1.5B params, GPT-2 XL equivalent)",
        "min_ram_gb": 128,
        "min_vram_gb": 48,
    },
}


def get_model_config(size: str = "tiny") -> dict:
    """
    Get model configuration for a given size preset.

    Args:
        size: One of "tiny", "small", "medium", "large", "xl", "xxl", "xxxl"

    Returns:
        Dict with dim, depth, heads, max_len, ff_mult
    """
    if size not in MODEL_PRESETS:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(MODEL_PRESETS.keys())}")

    config = MODEL_PRESETS[size].copy()
    # Remove non-model keys
    config.pop("description", None)
    config.pop("min_ram_gb", None)
    config.pop("min_vram_gb", None)
    return config


def estimate_parameters(
        vocab_size: int,
        dim: int,
        depth: int,
        ff_mult: float = 4.0,
        **kwargs) -> int:
    """
    Estimate total trainable parameters for an Enigma model.

    Args:
        vocab_size: Vocabulary size
        dim: Hidden dimension
        depth: Number of transformer layers
        ff_mult: FFN hidden dimension multiplier

    Returns:
        Estimated parameter count
    """
    # Embedding (weight-tied with output head)
    embed_params = vocab_size * dim

    # Per-layer params:
    # - Attention: Q, K, V, O projections = 4 * dim * dim
    # - FFN (SwiGLU): w1, w2, w3 = 3 * dim * (dim * ff_mult)
    # - Norms: 2 * dim (attention_norm, ffn_norm)
    attn_params = 4 * dim * dim
    ffn_params = 3 * dim * int(dim * ff_mult)
    norm_params = 2 * dim

    layer_params = (attn_params + ffn_params + norm_params) * depth

    # Final norm
    final_norm = dim

    return embed_params + layer_params + final_norm


def print_model_info():
    """Print all available model presets."""
    print("\n" + "=" * 70)
    print("ENIGMA MODEL SIZE PRESETS")
    print("=" * 70)

    for name, config in MODEL_PRESETS.items():
        params = estimate_parameters(
            vocab_size=32000,
            dim=config['dim'],
            depth=config['depth'],
            ff_mult=config.get('ff_mult', 4.0)
        )
        print(f"\n{name.upper()}")
        print(f"  {config['description']}")
        print(f"  Dimensions: {config['dim']}, Layers: {config['depth']}, Heads: {config['heads']}")
        print(f"  Max Length: {config['max_len']}, FFN Mult: {config.get('ff_mult', 4.0)}")
        print(f"  Est. Parameters: {params:,}")
        print(f"  Min RAM: {config['min_ram_gb']}GB, Min VRAM: {config['min_vram_gb']}GB")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_model_info()
