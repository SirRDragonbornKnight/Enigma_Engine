"""
Enigma AI Engine
================

A fully modular AI framework where EVERYTHING is a toggleable module.
Scales from Raspberry Pi to datacenter.

Quick Start:
    >>> from enigma.core import create_model, EnigmaEngine
    >>> model = create_model('small')
    >>> engine = EnigmaEngine()
    >>> response = engine.generate("Hello, how are you?")

Package Structure:
    enigma/
    ├── core/       - Model, training, inference, tokenizers
    ├── modules/    - Module system (load/unload capabilities)
    ├── gui/        - PyQt5 interface with generation tabs (image, code, video, audio, 3D)
    ├── memory/     - Conversation storage, vector search
    ├── comms/      - API server, networking
    ├── voice/      - TTS/STT
    ├── avatar/     - Avatar control
    ├── tools/      - Vision, web, file tools
    ├── utils/      - Common utilities
    └── config/     - Configuration management

For more details, see the README.md or visit:
https://github.com/SirRDragonbornKnight/Enigma_AI_Engine
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
__version__ = "2.0.0"
__author__ = "Enigma AI Team"

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
