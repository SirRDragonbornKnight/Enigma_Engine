"""
AI Tester
=========

A fully modular AI framework where EVERYTHING is a toggleable module.
Scales from Raspberry Pi to datacenter.

Quick Start:
    >>> from ai_tester.core import create_model, AITesterEngine
    >>> model = create_model('small')
    >>> engine = AITesterEngine()
    >>> response = engine.generate("Hello, how are you?")

Package Structure:
    ai_tester/
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
https://github.com/SirRDragonbornKnight/AI_Tester
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
__version__ = "0.1.0"
__author__ = "AI Tester Team"

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
