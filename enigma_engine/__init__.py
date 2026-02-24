"""
Enigma AI Engine - Modular AI Framework

Simple imports:
    from enigma_engine.core import EnigmaEngine
"""
from pathlib import Path

# Re-export configuration from central location
from .config import CONFIG, get_config, update_config

# For backwards compatibility, export path constants
ROOT = Path(CONFIG["root"])
DATA_DIR = Path(CONFIG["data_dir"])
MODELS_DIR = Path(CONFIG["models_dir"])
DB_PATH = Path(CONFIG["db_path"])

# Version info
__version__ = "1.1.0"
__author__ = "SirRDragonbornKnight"

__all__ = [
    # Configuration
    'CONFIG',
    'get_config',
    'update_config',
    # Path constants
    'ROOT',
    'DATA_DIR',
    'MODELS_DIR',
    'DB_PATH',
    # Version
    '__version__',
]
