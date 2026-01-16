"""
Model configuration presets for different sizes.
Choose based on your hardware capabilities.

USAGE:
    from forge_ai.core.model_config import get_model_config

    config = get_model_config("medium")  # or "tiny", "small", "large", "xl"
    
NOTE: This module re-exports from forge_ai.core.model for backward compatibility.
The canonical MODEL_PRESETS are defined in forge_ai.core.model.
"""

# Import from the canonical location
from .model import MODEL_PRESETS, MODEL_DESCRIPTIONS, ForgeConfig


def get_model_config(size: str = "tiny") -> dict:
    """
    Get model configuration for a given size preset.

    Args:
        size: One of the available model sizes (nano, micro, tiny, small, medium, etc.)

    Returns:
        Dict with model configuration parameters
    """
    if size not in MODEL_PRESETS:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(MODEL_PRESETS.keys())}")

    return MODEL_PRESETS[size].to_dict()


def estimate_parameters(
        vocab_size: int,
        dim: int,
        n_layers: int = None,
        depth: int = None,
        ff_mult: float = 4.0,
        **kwargs) -> int:
    """
    Estimate total trainable parameters for an Forge model.

    Args:
        vocab_size: Vocabulary size
        dim: Hidden dimension
        n_layers: Number of transformer layers (preferred)
        depth: Number of transformer layers (legacy alias)
        ff_mult: FFN hidden dimension multiplier

    Returns:
        Estimated parameter count
    """
    # Support both n_layers (new) and depth (legacy)
    layers = n_layers or depth or 8
    
    # Embedding (weight-tied with output head)
    embed_params = vocab_size * dim

    # Per-layer params:
    # - Attention: Q, K, V, O projections = 4 * dim * dim
    # - FFN (SwiGLU): w1, w2, w3 = 3 * dim * (dim * ff_mult)
    # - Norms: 2 * dim (attention_norm, ffn_norm)
    attn_params = 4 * dim * dim
    ffn_params = 3 * dim * int(dim * ff_mult)
    norm_params = 2 * dim

    layer_params = (attn_params + ffn_params + norm_params) * layers

    # Final norm
    final_norm = dim

    return embed_params + layer_params + final_norm


def print_model_info():
    """Print all available model presets."""
    print("\n" + "=" * 70)
    print("FORGE AI MODEL SIZE PRESETS")
    print("=" * 70)

    for name, preset in MODEL_PRESETS.items():
        config = preset.to_dict() if hasattr(preset, 'to_dict') else preset
        params = estimate_parameters(
            vocab_size=32000,
            dim=config.get('dim', 512),
            n_layers=config.get('n_layers', config.get('depth', 8)),
            ff_mult=4.0
        )
        desc = MODEL_DESCRIPTIONS.get(name, "")
        print(f"\n{name.upper()}")
        print(f"  {desc}")
        print(f"  Dimensions: {config.get('dim')}, Layers: {config.get('n_layers')}, Heads: {config.get('n_heads')}")
        print(f"  Max Length: {config.get('max_seq_len')}")
        print(f"  Est. Parameters: {params:,}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_model_info()
