# core package - Enigma Model and Training

from .model import Enigma, TinyEnigma
from .inference import EnigmaEngine
from .training import train_model
from .char_tokenizer import CharacterTokenizer
from .model_config import get_model_config, MODEL_PRESETS
from .model_registry import ModelRegistry
from .hardware import get_hardware, HardwareProfile

__all__ = [
    # Model
    "Enigma",
    "TinyEnigma",  # Backwards compatibility alias
    "EnigmaEngine",
    
    # Training
    "train_model",
    
    # Tokenizer
    "CharacterTokenizer",
    
    # Config
    "get_model_config",
    "MODEL_PRESETS",
    
    # Registry
    "ModelRegistry",
    
    # Hardware
    "get_hardware",
    "HardwareProfile",
]
