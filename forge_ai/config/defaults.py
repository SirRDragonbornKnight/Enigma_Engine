"""
Default configuration values for ForgeAI.

This is the SINGLE SOURCE OF TRUTH for all configuration.

These can be overridden by:
1. User config file (forge_config.json)
2. Environment variables (FORGE_*)
"""
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Base directory (where forge_ai package is)
BASE_DIR = Path(__file__).parent.parent.parent

# Default configuration - SINGLE SOURCE OF TRUTH
CONFIG = {
    # === Paths ===
    "root": str(BASE_DIR),
    "data_dir": str(BASE_DIR / "data"),              # Training data, icons, themes
    "info_dir": str(BASE_DIR / "information"),       # Runtime settings, tasks, reminders
    "models_dir": str(BASE_DIR / "models"),
    "memory_dir": str(BASE_DIR / "memory"),
    "db_path": str(BASE_DIR / "memory" / "memory.db"),
    "vocab_dir": str(BASE_DIR / "forge_ai" / "vocab_model"),
    "logs_dir": str(BASE_DIR / "logs"),

    # === Model Architecture ===
    "default_model": "forge_ai",
    "embed_dim": 256,
    "depth": 6,            # Alias: num_layers (for compatibility)
    "num_layers": 6,       # Alias: depth
    "heads": 8,            # Alias: num_heads (for compatibility)
    "num_heads": 8,        # Alias: heads
    "max_len": 2048,
    "ff_mult": 4.0,
    "dropout": 0.0,
    "vocab_size": 32000,

    # === Training Defaults ===
    "learning_rate": 1e-4,
    "default_learning_rate": 1e-4,  # Alias for GUI
    "batch_size": 32,
    "default_batch_size": 32,       # Alias for GUI
    "epochs": 10,
    "default_epochs": 10,           # Alias for GUI
    "warmup_steps": 100,
    "gradient_accumulation_steps": 1,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "use_amp": True,
    "auto_learn": True,
    "auto_train_threshold": 10,

    # === Inference Defaults ===
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "max_gen": 100,

    # === Server Defaults ===
    "api_host": "127.0.0.1",
    "api_port": 5000,
    "enable_cors": True,

    # === Hardware ===
    "device": "auto",       # "auto", "cpu", "cuda", "mps"
    "precision": "float32", # "float32", "float16", "bfloat16"

    # === Features ===
    "enable_voice": True,
    "enable_vision": True,
    "enable_avatar": True,

    # === Multi-model Support ===
    "allow_multiple_models": True,
    "max_concurrent_models": 4,

    # === Resource Limiting ===
    "resource_mode": "performance",  # "minimal", "balanced", "performance"
    "cpu_threads": 0,             # 0 = auto
    "memory_limit_mb": 0,         # 0 = no limit
    "gpu_memory_fraction": 0.85,  # Use 85% of GPU VRAM
    "low_priority": False,
    
    # === Device Offloading (CPU+GPU) ===
    "enable_offloading": False,   # Enable CPU+GPU layer offloading
    "offload_folder": None,       # Folder for offloaded weights (None = temp)
    "offload_to_disk": False,     # Also offload to disk for very large models
    "max_gpu_layers": None,       # Max layers on GPU (None = auto)
    
    # === Security - Blocked Files ===
    # Files/folders the AI cannot read, write, or modify
    # This setting CANNOT be changed by the AI
    "blocked_paths": [
        # Add paths here that the AI should never access
        # Example: "C:/Windows/System32",
        # Example: "/etc/passwd",
    ],
    "blocked_patterns": [
        # Glob patterns for blocked files
        "*.exe",
        "*.dll",
        "*.sys",
        "*.pem",
        "*.key",
        "*password*",
        "*secret*",
        "*.env",
        ".git/config",
    ],

    # === Logging ===
    "log_level": "INFO",
    "log_to_file": False,
}


def _load_user_config() -> None:
    """Load user configuration file if it exists."""
    config_paths = [
        Path.cwd() / "forge_config.json",
        Path.home() / ".forge_ai" / "config.json",
        BASE_DIR / "forge_config.json",
        # Legacy paths for backwards compatibility
        Path.cwd() / "enigma_config.json",
        BASE_DIR / "enigma_config.json",
    ]

    for path in config_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                if not isinstance(user_config, dict):
                    print(f"Warning: Config in {path} is not a dictionary, skipping")
                    continue
                CONFIG.update(user_config)
                print(f"Loaded config from {path}")
                return
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in config file {path}: {e}")
            except Exception as e:
                print(f"Warning: Failed to load config from {path}: {e}")


def _load_env_config() -> None:
    """Load configuration from environment variables."""
    env_mappings = {
        # New FORGE_ prefix (preferred)
        "FORGE_DATA_DIR": "data_dir",
        "FORGE_MODELS_DIR": "models_dir",
        "FORGE_MEMORY_DIR": "memory_dir",
        "FORGE_DEVICE": "device",
        "FORGE_API_HOST": "api_host",
        "FORGE_API_PORT": "api_port",
        "FORGE_LOG_LEVEL": "log_level",
        # Legacy ENIGMA_ prefix (still supported)
        "ENIGMA_DATA_DIR": "data_dir",
        "ENIGMA_MODELS_DIR": "models_dir",
        "ENIGMA_MEMORY_DIR": "memory_dir",
        "ENIGMA_DEVICE": "device",
        "ENIGMA_API_HOST": "api_host",
        "ENIGMA_API_PORT": "api_port",
        "ENIGMA_LOG_LEVEL": "log_level",
    }

    for env_var, config_key in env_mappings.items():
        if env_var in os.environ:
            value: Any = os.environ[env_var]
            # Type conversion with validation
            if config_key == "api_port":
                try:
                    value = int(value)
                    if not (1 <= value <= 65535):
                        print(f"Warning: Invalid port {value}, using default")
                        continue
                except ValueError:
                    print(f"Warning: Invalid port value {value}, using default")
                    continue
            CONFIG[config_key] = value


def get_config(key: str, default: Any = None) -> Any:
    """
    Get a configuration value.

    Args:
        key: Configuration key to retrieve
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    return CONFIG.get(key, default)


def update_config(updates: Dict[str, Any]) -> None:
    """
    Update configuration with new values.

    Args:
        updates: Dictionary of configuration updates

    Raises:
        TypeError: If updates is not a dictionary
    """
    if not isinstance(updates, dict):
        raise TypeError(f"updates must be a dict, got {type(updates).__name__}")
    CONFIG.update(updates)


def save_config(path: Optional[str] = None) -> None:
    """
    Save current configuration to file.

    Args:
        path: Path to save config file (default: forge_config.json in base directory)

    Raises:
        IOError: If file cannot be written
    """
    if path is None:
        path = str(BASE_DIR / "forge_config.json")

    try:
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(CONFIG, f, indent=2)
    except Exception as e:
        raise IOError(f"Failed to save config to {path}: {e}") from e


# Create directories on import with error handling
for dir_key in ["data_dir", "models_dir", "memory_dir", "logs_dir"]:
    try:
        Path(CONFIG[dir_key]).mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not create directory {CONFIG[dir_key]}: {e}")

# Load user and environment configuration
_load_user_config()
_load_env_config()
