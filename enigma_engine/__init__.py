"""
Enigma AI Engine - Modular AI Framework

Simple imports:
    from enigma_engine.core import EnigmaEngine
"""

# Re-export configuration from central location
from .config import CONFIG

try:
    from .client import EnigmaClient
except ImportError:
    EnigmaClient = None

# Version info
__version__ = "1.1.0"
__author__ = "SirRDragonbornKnight"

__all__ = [
    'CONFIG',
    'EnigmaClient',
    '__version__',
]
