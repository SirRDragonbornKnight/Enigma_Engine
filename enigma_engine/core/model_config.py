"""
Model configuration presets for different sizes.
Choose based on your hardware capabilities.

USAGE:
    from enigma_engine.core.model_config import get_model_config

    config = get_model_config("medium")  # or "tiny", "small", "large", "xl"
    
NOTE: This module re-exports from enigma_engine.core.model for backward compatibility.
The canonical MODEL_PRESETS are defined in enigma_engine.core.model.
"""

# Import from the canonical location
from .model_presets import MODEL_PRESETS


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
