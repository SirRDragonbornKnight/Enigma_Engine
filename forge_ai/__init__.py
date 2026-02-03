"""
================================================================================
               THE KINGDOM OF FORGEAI - YOUR ADVENTURE BEGINS HERE
================================================================================

    "Every great journey has a starting point. This is yours."

Welcome, adventurer! You have discovered ForgeAI - a fully modular AI
framework where EVERYTHING is a toggleable module. Whether you're running
on a tiny Raspberry Pi or a massive datacenter, ForgeAI adapts to you.

WHAT IS THIS FILE?
    This is the FRONT GATE of the kingdom. When you write:
        from forge_ai import something
    Python reads THIS FILE first to know what's available.

THE KINGDOM MAP:
    ┌─────────────────────────────────────────────────────────────────┐
    │  forge_ai/                                                      │
    │  ├── core/       THE FORGE    - AI brains (model, inference)   │
    │  ├── modules/    THE ARMORY   - Load/unload capabilities       │
    │  ├── gui/        THE CASTLE   - Visual interface (PyQt5)       │
    │  ├── memory/     THE LIBRARY  - Conversation storage           │
    │  ├── comms/      THE NETWORK  - API server, remote access      │
    │  ├── voice/      THE HERALD   - TTS/STT voice features         │
    │  ├── avatar/     THE CHAMPION - Virtual character control      │
    │  ├── tools/      THE WORKSHOP - Vision, web, file tools        │
    │  ├── utils/      THE UTILITY  - Common helpers                 │
    │  └── config/     THE CODEX    - Configuration settings         │
    └─────────────────────────────────────────────────────────────────┘

YOUR FIRST QUEST:
    >>> from forge_ai.core import create_model, ForgeEngine
    >>> model = create_model('small')      # Create an AI brain
    >>> engine = ForgeEngine()             # Create a way to talk to it
    >>> response = engine.generate("Hello, how are you?")
    >>> print(response)                    # Magic!

CHOOSE YOUR PATH (Entry Points):
    | Command               | Destination            | Description          |
    |-----------------------|------------------------|----------------------|
    | python run.py --gui   | gui/enhanced_window.py | Visual interface     |
    | python run.py --train | core/training.py       | Teach your AI        |
    | python run.py --run   | core/inference.py      | Chat in terminal     |
    | python run.py --serve | comms/api_server.py    | REST API server      |

DOCUMENTATION SCROLLS:
    • README.md              - Getting started guide
    • CODE_ADVENTURE_TOUR.md - Interactive code exploration
    • QUICK_FILE_LOCATOR.md  - Find files fast
    • docs/CODE_TOUR.md      - Detailed technical docs

May your training converge and your gradients flow smoothly!
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
