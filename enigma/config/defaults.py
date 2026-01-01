"""
Default configuration values for Enigma Engine.

These can be overridden by:
1. User config file (enigma_config.json)
2. Environment variables (ENIGMA_*)
"""
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

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


def _load_user_config() -> None:
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
        path: Path to save config file (default: enigma_config.json in base directory)
        
    Raises:
        IOError: If file cannot be written
    """
    if path is None:
        path = str(BASE_DIR / "enigma_config.json")
    
    try:
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(CONFIG, f, indent=2)
    except Exception as e:
        raise IOError(f"Failed to save config to {path}: {e}") from e


# Create directories on import
for dir_key in ["data_dir", "models_dir", "memory_dir", "logs_dir"]:
    Path(CONFIG[dir_key]).mkdir(parents=True, exist_ok=True)

# Load user and environment configuration
_load_user_config()
_load_env_config()
