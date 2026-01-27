"""
================================================================================
THE CHAMBER OF CONFIGURATION - DEFAULT SETTINGS
================================================================================

Deep within the foundations of ForgeAI lies the Chamber of Configuration,
where the sacred scrolls of default settings are inscribed. These ancient
texts define the very nature of the forge - its paths, its powers, and its
boundaries.

FILE: forge_ai/config/defaults.py
TYPE: Configuration Management
MAIN EXPORT: CONFIG dictionary

    THE HIERARCHY OF TRUTH:
    
    When ForgeAI awakens, it reads configuration from three sources,
    each layer overriding the last:
    
        1. DEFAULT VALUES (this file) - The ancient foundation
        2. USER CONFIG FILE (forge_config.json) - Custom modifications
        3. ENVIRONMENT VARIABLES (FORGE_*) - Runtime overrides

    THE SACRED SECTIONS:
    
    PATHS           - Where treasures are stored (data, models, memory)
    ARCHITECTURE    - The shape of the AI's mind (layers, dimensions)
    TRAINING        - How the AI learns (learning rate, epochs)
    INFERENCE       - How the AI speaks (temperature, sampling)
    HARDWARE        - What powers the forge (CPU, GPU, precision)
    SECURITY        - What must never be touched (blocked paths)

USAGE:
    from forge_ai.config import CONFIG, get_config, update_config
    
    # Read a value
    data_dir = CONFIG["data_dir"]
    # OR
    data_dir = get_config("data_dir", default="data")
    
    # Update values (in memory only)
    update_config({"temperature": 0.9})
    
    # Persist changes to file
    save_config()

ENVIRONMENT VARIABLES:
    FORGE_DATA_DIR, FORGE_MODELS_DIR, FORGE_DEVICE, etc.
    Legacy ENIGMA_* variables are also supported.

WARNING: The blocked_paths and blocked_patterns settings are sacred
         protections that the AI cannot modify at runtime.
"""
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# THE FOUNDATION STONE - Base Directory
# =============================================================================
# All paths in ForgeAI are relative to this sacred point.

BASE_DIR = Path(__file__).parent.parent.parent


# =============================================================================
# THE GREAT CODEX - Default Configuration
# =============================================================================
# These are the foundational settings upon which ForgeAI is built.
# Each section governs a different aspect of the forge's operation.

CONFIG = {
    # =========================================================================
    # THE MAP OF REALMS - Path Configuration
    # =========================================================================
    # Every treasure has its place. These paths define where ForgeAI
    # stores its knowledge, memories, and creations.
    
    "root": str(BASE_DIR),
    "data_dir": str(BASE_DIR / "data"),              # Training data, icons, themes
    "info_dir": str(BASE_DIR / "information"),       # Runtime settings, tasks, reminders
    "models_dir": str(BASE_DIR / "models"),
    "memory_dir": str(BASE_DIR / "memory"),
    "db_path": str(BASE_DIR / "memory" / "memory.db"),
    "vocab_dir": str(BASE_DIR / "forge_ai" / "vocab_model"),
    "logs_dir": str(BASE_DIR / "logs"),

    # =========================================================================
    # THE ARCHITECT'S BLUEPRINT - Model Architecture
    # =========================================================================
    # These settings define the structure of the AI's mind - how many
    # layers of thought, how wide its neural pathways, how far it can see.
    
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

    # =========================================================================
    # THE TEACHER'S WISDOM - Training Defaults
    # =========================================================================
    # When the AI learns, these settings guide its education - how quickly
    # it absorbs knowledge, how many times it studies the texts.
    
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

    # =========================================================================
    # THE ORACLE'S VOICE - Inference Defaults
    # =========================================================================
    # When the AI speaks, these settings color its responses - how creative,
    # how focused, how varied its words shall be.
    
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "max_gen": 100,

    # =========================================================================
    # THE MESSENGER'S GATE - Server Defaults
    # =========================================================================
    # The API server allows distant travelers to commune with the AI.
    # These settings control access and security.
    
    "api_host": "127.0.0.1",
    "api_port": 5000,
    "enable_cors": True,
    "require_api_key": True,       # Require authentication for API access
    "forgeai_api_key": None,       # Set via env FORGEAI_API_KEY or forge_config.json

    # =========================================================================
    # THE FORGE'S HEART - Hardware Configuration
    # =========================================================================
    # What powers drive the forge? CPU, GPU, or the mystical MPS of Apple?
    # The precision of calculations affects both speed and quality.
    
    "device": "auto",       # "auto", "cpu", "cuda", "mps"
    "precision": "float32", # "float32", "float16", "bfloat16"

    # =========================================================================
    # THE FEATURES MANIFEST - Capability Toggles
    # =========================================================================
    "enable_voice": True,
    "enable_vision": True,
    "enable_avatar": True,
    
    # =========================================================================
    # THE INTERFACE REALM - GUI Configuration
    # =========================================================================
    "gui_mode": "standard",  # "simple", "standard", "advanced", "gaming"
    "gui_theme": "dark",      # "dark", "light", "shadow", "midnight", "gaming"
    "enable_quick_actions": True,
    "enable_feedback_buttons": True,
    "show_game_mode_indicator": True,

    # =========================================================================
    # THE COUNCIL CHAMBER - Multi-Model Support
    # =========================================================================
    "allow_multiple_models": True,
    "max_concurrent_models": 4,
    
    # =========================================================================
    # THE ORCHESTRATOR - Deep Multi-Model Integration
    # =========================================================================
    "orchestrator": {
        "default_chat_model": "auto",           # Auto-select best available
        "default_code_model": "auto",           # Auto-select best available
        "default_vision_model": "auto",         # Auto-select best available
        "default_image_gen_model": "auto",      # Auto-select best available
        "max_loaded_models": 3,                 # Maximum models loaded at once
        "gpu_memory_limit_mb": 8000,            # GPU memory limit
        "cpu_memory_limit_mb": 16000,           # CPU memory limit
        "enable_collaboration": True,           # Enable model-to-model communication
        "enable_auto_fallback": True,           # Enable automatic fallback chains
        "fallback_to_cpu": True,                # Fallback to CPU if GPU full
        "enable_hot_swap": True,                # Allow hot-swapping models
    },

    # =========================================================================
    # THE RESOURCE WARDEN - Memory and CPU Limits
    # =========================================================================
    "resource_mode": "performance",  # "minimal", "balanced", "performance"
    "cpu_threads": 0,             # 0 = auto
    "memory_limit_mb": 0,         # 0 = no limit
    "gpu_memory_fraction": 0.85,  # Use 85% of GPU VRAM
    "low_priority": False,
    
    # =========================================================================
    # THE BRIDGE BETWEEN WORLDS - Device Offloading
    # =========================================================================
    "enable_offloading": False,   # Enable CPU+GPU layer offloading
    "offload_folder": None,       # Folder for offloaded weights (None = temp)
    "offload_to_disk": False,     # Also offload to disk for very large models
    "max_gpu_layers": None,       # Max layers on GPU (None = auto)
    
    # =========================================================================
    # THE GUARDIAN'S DECREE - Security Settings
    # =========================================================================
    # These sacred protections CANNOT be modified by the AI.
    # They define territories forbidden to artificial minds.
    
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

    # =========================================================================
    # THE CHRONICLER'S SETTINGS - Logging Configuration
    # =========================================================================
    "log_level": "INFO",
    "log_to_file": False,
}


# =============================================================================
# THE RITUAL OF READING - Loading User Configuration
# =============================================================================

def _load_user_config() -> None:
    """
    The Ritual of Reading User Configuration.
    
    Searches the realm for custom configuration scrolls (forge_config.json)
    in several sacred locations. The first scroll found is read, and its
    contents override the default settings.
    
    Search Order:
        1. Current working directory
        2. User's home directory (~/.forge_ai/)
        3. ForgeAI installation directory
        4. Legacy enigma_config.json locations (backwards compatibility)
    """
    config_paths = [
        Path.cwd() / "forge_config.json",
        Path.home() / ".forge_ai" / "config.json",
        BASE_DIR / "forge_config.json",
        # Legacy paths preserved for ancient installations
        Path.cwd() / "enigma_config.json",
        BASE_DIR / "enigma_config.json",
    ]

    for path in config_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                if not isinstance(user_config, dict):
                    logger.warning(f"Config in {path} is not a dictionary, skipping")
                    continue
                CONFIG.update(user_config)
                logger.info(f"Loaded config from {path}")
                return
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in config file {path}: {e}")
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}")


# =============================================================================
# THE RITUAL OF ENVIRONMENT - Loading Environment Variables
# =============================================================================

def _load_env_config() -> None:
    """
    The Ritual of Environment Variable Reading.
    
    Scans the environment for FORGE_* variables that override configuration.
    This allows runtime customization without modifying files - useful for
    containers, CI/CD, and temporary adjustments.
    
    Supported Variables:
        FORGE_DATA_DIR, FORGE_MODELS_DIR, FORGE_MEMORY_DIR,
        FORGE_DEVICE, FORGE_API_HOST, FORGE_API_PORT,
        FORGE_LOG_LEVEL, FORGEAI_API_KEY
        
    Legacy ENIGMA_* variables are still honored for backwards compatibility.
    """
    env_mappings = {
        # Modern FORGE_ prefix (preferred)
        "FORGE_DATA_DIR": "data_dir",
        "FORGE_MODELS_DIR": "models_dir",
        "FORGE_MEMORY_DIR": "memory_dir",
        "FORGE_DEVICE": "device",
        "FORGE_API_HOST": "api_host",
        "FORGE_API_PORT": "api_port",
        "FORGE_LOG_LEVEL": "log_level",
        "FORGEAI_API_KEY": "forgeai_api_key",
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
            # Type conversion with validation for port numbers
            if config_key == "api_port":
                try:
                    value = int(value)
                    if not (1 <= value <= 65535):
                        logger.warning(f"Invalid port {value}, using default")
                        continue
                except ValueError:
                    logger.warning(f"Invalid port value {value}, using default")
                    continue
            CONFIG[config_key] = value


# =============================================================================
# THE PUBLIC INTERFACE - Configuration Access Functions
# =============================================================================

def get_config(key: str, default: Any = None) -> Any:
    """
    Retrieve a value from the configuration.
    
    Args:
        key: The configuration key to retrieve
        default: Value to return if key is not found
        
    Returns:
        The configuration value, or default if not found
    """
    return CONFIG.get(key, default)


def update_config(updates: Dict[str, Any]) -> None:
    """
    Update configuration with new values (in memory only).
    
    Use save_config() to persist changes to disk.
    
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
    Persist current configuration to a JSON file.
    
    Args:
        path: Destination path (default: forge_config.json in base directory)
        
    Raises:
        IOError: If the file cannot be written
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


# =============================================================================
# THE AWAKENING - Module Initialization
# =============================================================================
# When this module is imported, it performs the sacred rituals:
# 1. Creates necessary directories
# 2. Loads user configuration
# 3. Applies environment variable overrides

for dir_key in ["data_dir", "models_dir", "memory_dir", "logs_dir"]:
    try:
        Path(CONFIG[dir_key]).mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not create directory {CONFIG[dir_key]}: {e}")

_load_user_config()
_load_env_config()
