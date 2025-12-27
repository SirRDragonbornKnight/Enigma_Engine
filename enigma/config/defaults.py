"""
Default configuration values for Enigma Engine.

These can be overridden by:
1. User config file (enigma_config.json)
2. Environment variables (ENIGMA_*)
"""
import os
import json
from pathlib import Path

# Base directory (where enigma package is)
BASE_DIR = Path(__file__).parent.parent.parent

# Default configuration
CONFIG = {
    # === Paths ===
    "root": str(BASE_DIR),
    "data_dir": str(BASE_DIR / "data"),
    "models_dir": str(BASE_DIR / "models"),
    "memory_dir": str(BASE_DIR / "memory"),
    "db_path": str(BASE_DIR / "memory" / "memory.db"),
    "vocab_dir": str(BASE_DIR / "enigma" / "vocab_model"),
    "logs_dir": str(BASE_DIR / "logs"),
    
    # === Model Defaults ===
    "embed_dim": 256,
    "depth": 6,
    "heads": 8,
    "max_len": 2048,
    "ff_mult": 4.0,
    "dropout": 0.0,
    "vocab_size": 32000,
    
    # === Training Defaults ===
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 10,
    "warmup_steps": 100,
    "gradient_accumulation_steps": 1,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "use_amp": True,
    
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
    "device": "auto",  # auto, cpu, cuda, mps
    "precision": "float32",  # float32, float16, bfloat16
    
    # === Logging ===
    "log_level": "INFO",
    "log_to_file": False,
}


def _load_user_config():
    """Load user configuration file if it exists."""
    config_paths = [
        Path.cwd() / "enigma_config.json",
        Path.home() / ".enigma" / "config.json",
        BASE_DIR / "enigma_config.json",
    ]
    
    for path in config_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                CONFIG.update(user_config)
                print(f"Loaded config from {path}")
                return
            except Exception as e:
                print(f"Warning: Failed to load config from {path}: {e}")


def _load_env_config():
    """Load configuration from environment variables."""
    env_mappings = {
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
            value = os.environ[env_var]
            # Type conversion
            if config_key in ["api_port"]:
                value = int(value)
            CONFIG[config_key] = value


def get_config(key: str, default=None):
    """Get a configuration value."""
    return CONFIG.get(key, default)


def update_config(updates: dict):
    """Update configuration with new values."""
    CONFIG.update(updates)


def save_config(path: str = None):
    """Save current configuration to file."""
    if path is None:
        path = BASE_DIR / "enigma_config.json"
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)


# Create directories on import
for dir_key in ["data_dir", "models_dir", "memory_dir", "logs_dir"]:
    Path(CONFIG[dir_key]).mkdir(parents=True, exist_ok=True)

# Load user and environment configuration
_load_user_config()
_load_env_config()
