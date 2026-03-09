"""
Enigma AI Engine - Modular AI Framework

Simple imports:
    from enigma_engine.core import EnigmaEngine
"""

# Re-export configuration from central location
from .config import CONFIG

# Version info
__version__ = "1.1.0"
__author__ = "SirRDragonbornKnight"

__all__ = [
    'CONFIG',
    '__version__',
]
