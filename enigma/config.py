"""
Enigma Engine Configuration

Central configuration for all Enigma components.
Provides sensible defaults, validation, and preset management.
"""
from pathlib import Path
from typing import Dict, Any, List
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "memory" / "memory.db"

# Create directories with error handling
try:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.makedirs(Path(DB_PATH).parent, exist_ok=True)
except (OSError, PermissionError) as e:
    logger.warning(f"Could not create directories: {e}")

# =============================================================================
# MAIN CONFIGURATION
# =============================================================================
# Change these variables to customize your AI setup

CONFIG = {
    # Paths
    "root": str(ROOT),
    "data_dir": str(DATA_DIR),
    "models_dir": str(MODELS_DIR),
    "db_path": str(DB_PATH),

    # === CHANGE THESE TO SWITCH AI ===
    "default_model": "enigma",          # Which AI to load by default (change to your AI name)
    "auto_learn": True,                 # Learn from conversations automatically
    "auto_train_threshold": 10,         # Train after N conversations (0 = never)

    # Model architecture (for Enigma and scalable models)
    "max_len": 512,                     # Maximum sequence length
    "embed_dim": 128,                   # Embedding dimension (bigger = smarter but slower)
    "num_layers": 4,                    # Transformer layers (more = smarter but slower)
    "num_heads": 4,                     # Attention heads
    "vocab_size": 32000,                # Vocabulary size

    # Training defaults
    "default_epochs": 10,
    "default_batch_size": 2,            # Keep small for Pi
    "default_learning_rate": 0.0001,

    # Features
    "enable_voice": True,
    "enable_vision": True,
    "enable_avatar": True,

    # Multi-model support
    "allow_multiple_models": True,      # Allow running multiple AIs at once
    "max_concurrent_models": 4,         # Maximum number of models to run

    # === RESOURCE LIMITING ===
    # Use these to limit CPU/RAM so you can run other apps (like games!)
    "resource_mode": "balanced",        # "minimal", "balanced", "performance", "max"
    "cpu_threads": 0,                   # 0 = auto, or set specific number (1-16)
    "memory_limit_mb": 0,               # 0 = no limit, or set max MB for AI
    "gpu_memory_fraction": 0.5,         # 0.0-1.0, how much GPU memory to use
    "low_priority": False,              # Run AI at lower process priority
}

# =============================================================================
# MODEL PRESETS - Easy scaling
# =============================================================================
# Use these to quickly change model size

MODEL_PRESETS = {
    "tiny": {
        "embed_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "description": "Very small, fast on Pi Zero/3"
    },
    "small": {
        "embed_dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "description": "Default for Pi 4/5"
    },
    "medium": {
        "embed_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "description": "For Pi 5 with 8GB RAM"
    },
    "large": {
        "embed_dim": 512,
        "num_layers": 8,
        "num_heads": 8,
        "description": "Needs GPU or high-end machine"
    }
}


def apply_preset(preset_name: str) -> bool:
    """
    Apply a model preset to CONFIG.

    Args:
        preset_name: Name of preset to apply (tiny, small, medium, large)

    Returns:
        True if preset was applied, False if preset not found

    Example:
        >>> apply_preset("small")
        True
        >>> CONFIG["embed_dim"]
        128
    """
    if preset_name not in MODEL_PRESETS:
        logger.warning(f"Unknown preset '{preset_name}'. Available: {list(MODEL_PRESETS.keys())}")
        return False

    preset = MODEL_PRESETS[preset_name]
    CONFIG["embed_dim"] = preset["embed_dim"]
    CONFIG["num_layers"] = preset["num_layers"]
    CONFIG["num_heads"] = preset["num_heads"]
    logger.info(f"Applied preset '{preset_name}': {preset['description']}")
    return True


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.

    Args:
        model_name: Name of the model to get configuration for

    Returns:
        Dictionary containing model configuration

    Raises:
        ValueError: If model_name is empty or invalid
    """
    if not model_name or not isinstance(model_name, str):
        raise ValueError(f"Invalid model name: {model_name}")

    model_dir = MODELS_DIR / model_name
    config_file = model_dir / "config.json"

    if config_file.exists():
        import json
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load config for {model_name}: {e}")
            # Fall through to return defaults

    # Return defaults
    return {
        "name": model_name,
        "embed_dim": CONFIG["embed_dim"],
        "num_layers": CONFIG["num_layers"],
        "num_heads": CONFIG["num_heads"],
        "vocab_size": CONFIG["vocab_size"],
        "auto_learn": CONFIG["auto_learn"]
    }


def validate_config() -> List[str]:
    """
    Validate configuration values.

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Validate numeric ranges
    if CONFIG["embed_dim"] < 1:
        errors.append("embed_dim must be positive")
    if CONFIG["num_layers"] < 1:
        errors.append("num_layers must be positive")
    if CONFIG["num_heads"] < 1:
        errors.append("num_heads must be positive")
    if CONFIG["max_len"] < 1:
        errors.append("max_len must be positive")
    if CONFIG["vocab_size"] < 1:
        errors.append("vocab_size must be positive")

    # Validate training params
    if CONFIG["default_learning_rate"] <= 0 or CONFIG["default_learning_rate"] > 1:
        errors.append("default_learning_rate must be between 0 and 1")
    if CONFIG["default_batch_size"] < 1:
        errors.append("default_batch_size must be positive")
    if CONFIG["default_epochs"] < 1:
        errors.append("default_epochs must be positive")

    # Validate resource limits
    if CONFIG["gpu_memory_fraction"] < 0 or CONFIG["gpu_memory_fraction"] > 1:
        errors.append("gpu_memory_fraction must be between 0 and 1")

    # Validate resource mode
    valid_modes = ["minimal", "balanced", "performance", "max"]
    if CONFIG["resource_mode"] not in valid_modes:
        errors.append(f"resource_mode must be one of {valid_modes}")

    return errors


# Validate configuration on module load
_validation_errors = validate_config()
if _validation_errors:
    logger.error(f"Configuration validation failed: {_validation_errors}")
