"""
Model configuration presets for different sizes.
Choose based on your hardware capabilities.

USAGE:
    from enigma.core.model_config import MODEL_PRESETS, get_model_config
    
    config = get_model_config("medium")  # or "tiny", "small", "large", "xl"
    model = TinyEnigma(**config)
"""

# Model size presets
# Parameters = vocab_size * dim * 2 + dim * dim * depth * 12 (approximate)

MODEL_PRESETS = {
    # ~500K params - Raspberry Pi, testing
    "tiny": {
        "dim": 128,
        "depth": 4,
        "heads": 4,
        "max_len": 512,
        "description": "Tiny model for testing (~500K params)",
        "min_ram_gb": 1,
        "min_vram_gb": 0,
    },
    
    # ~10M params - Single GPU, laptop
    "small": {
        "dim": 256,
        "depth": 6,
        "heads": 8,
        "max_len": 1024,
        "description": "Small model for learning (~10M params)",
        "min_ram_gb": 4,
        "min_vram_gb": 2,
    },
    
    # ~50M params - Good GPU needed
    "medium": {
        "dim": 512,
        "depth": 8,
        "heads": 8,
        "max_len": 2048,
        "description": "Medium model for real use (~50M params)",
        "min_ram_gb": 8,
        "min_vram_gb": 4,
    },
    
    # ~150M params - Serious GPU needed
    "large": {
        "dim": 768,
        "depth": 12,
        "heads": 12,
        "max_len": 2048,
        "description": "Large model (~150M params, like GPT-2 small)",
        "min_ram_gb": 16,
        "min_vram_gb": 8,
    },
    
    # ~350M params - Multi-GPU territory
    "xl": {
        "dim": 1024,
        "depth": 24,
        "heads": 16,
        "max_len": 2048,
        "description": "XL model (~350M params, like GPT-2 medium)",
        "min_ram_gb": 32,
        "min_vram_gb": 12,
    },
    
    # ~750M params - Multi-GPU required
    "xxl": {
        "dim": 1280,
        "depth": 36,
        "heads": 20,
        "max_len": 2048,
        "description": "XXL model (~750M params, like GPT-2 large)",
        "min_ram_gb": 64,
        "min_vram_gb": 24,
    },
}


def get_model_config(size: str = "tiny") -> dict:
    """
    Get model configuration for a given size preset.
    
    Args:
        size: One of "tiny", "small", "medium", "large", "xl", "xxl"
        
    Returns:
        Dict with dim, depth, heads, max_len
    """
    if size not in MODEL_PRESETS:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(MODEL_PRESETS.keys())}")
    
    config = MODEL_PRESETS[size].copy()
    # Remove non-model keys
    config.pop("description", None)
    config.pop("min_ram_gb", None)
    config.pop("min_vram_gb", None)
    return config


def estimate_parameters(vocab_size: int, dim: int, depth: int, **kwargs) -> int:
    """Estimate total trainable parameters."""
    embed_params = vocab_size * dim * 2  # token embed + output head
    layer_params = dim * dim * 12 * depth  # rough transformer layer estimate
    return embed_params + layer_params


def print_model_info():
    """Print all available model presets."""
    print("\n[SYSTEM] " + "="*60)
    print("[SYSTEM] ENIGMA MODEL SIZE PRESETS")
    print("[SYSTEM] " + "="*60)
    
    for name, config in MODEL_PRESETS.items():
        params = estimate_parameters(vocab_size=32000, **{k:v for k,v in config.items() 
                                                          if k in ['dim','depth','heads']})
        print(f"\n[SYSTEM] {name.upper()}")
        print(f"[SYSTEM]   {config['description']}")
        print(f"[SYSTEM]   Dimensions: {config['dim']}, Layers: {config['depth']}, Heads: {config['heads']}")
        print(f"[SYSTEM]   Est. Parameters: {params:,}")
        print(f"[SYSTEM]   Min RAM: {config['min_ram_gb']}GB, Min VRAM: {config['min_vram_gb']}GB")
    
    print("\n[SYSTEM] " + "="*60)


if __name__ == "__main__":
    print_model_info()
