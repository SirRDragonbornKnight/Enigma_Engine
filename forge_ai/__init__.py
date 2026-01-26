"""
================================================================================
🔥 FORGEAI - THE MODULAR AI FRAMEWORK
================================================================================

A fully modular AI framework where EVERYTHING is a toggleable module.
Scales from Raspberry Pi to datacenter.

📍 FILE: forge_ai/__init__.py
🏷️ TYPE: Package Root

┌─────────────────────────────────────────────────────────────────────────────┐
│  PACKAGE STRUCTURE:                                                         │
│                                                                             │
│  forge_ai/                                                                  │
│  ├── core/       🧠 Model, training, inference, tokenizers                  │
│  ├── modules/    ⚙️  Module system (load/unload capabilities)               │
│  ├── gui/        🖥️  PyQt5 interface with generation tabs                   │
│  ├── memory/     💾 Conversation storage, vector search                    │
│  ├── comms/      🌐 API server, networking                                 │
│  ├── voice/      🔊 TTS/STT                                                │
│  ├── avatar/     🤖 Avatar control                                         │
│  ├── tools/      🔧 Vision, web, file tools                                │
│  ├── utils/      🛠️  Common utilities                                       │
│  └── config/     ⚙️  Configuration management                               │
└─────────────────────────────────────────────────────────────────────────────┘

🚀 QUICK START:
    >>> from forge_ai.core import create_model, ForgeEngine
    >>> model = create_model('small')
    >>> engine = ForgeEngine()
    >>> response = engine.generate("Hello, how are you?")

📚 DOCUMENTATION:
    • README.md              - Getting started
    • CODE_ADVENTURE_TOUR.txt - Interactive code guide
    • QUICK_FILE_LOCATOR.txt  - Find files fast
    • docs/CODE_TOUR.md       - Detailed documentation

🔗 ENTRY POINTS:
    • run.py --gui   → forge_ai/gui/enhanced_window.py
    • run.py --train → forge_ai/core/training.py
    • run.py --run   → forge_ai/core/inference.py
    • run.py --serve → forge_ai/comms/api_server.py

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
__author__ = "ForgeAI Team"

# Cross-device integration (Pi + Phone + Gaming PC)
try:
    from .integration import (
        CrossDeviceSystem, SystemRole, DeviceEndpoint,
        quick_setup_gaming_pc, quick_setup_phone, quick_setup_pi,
    )
    HAS_INTEGRATION = True
except ImportError:
    HAS_INTEGRATION = False

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
    # Cross-device integration
    'CrossDeviceSystem',
    'SystemRole',
    'DeviceEndpoint',
    'quick_setup_gaming_pc',
    'quick_setup_phone',
    'quick_setup_pi',
]
