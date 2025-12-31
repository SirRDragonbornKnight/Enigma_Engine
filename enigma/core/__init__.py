# core package - Enigma Model and Training
"""
Enigma Core Module
==================

Contains the core components of the Enigma AI framework:
- Model architecture (Transformer with RoPE, RMSNorm, SwiGLU, GQA)
- Training system (AMP, gradient accumulation, cosine warmup)
- Inference engine (KV-cache, streaming, chat)
- Tokenization (BPE, character-level)
"""

# Model
from .model import (
    Enigma,
    TinyEnigma,  # Backwards compatibility alias
    EnigmaConfig,
    create_model,
    MODEL_PRESETS,
)

# Inference
from .inference import EnigmaEngine, generate, load_engine

# Training
from .training import (
    Trainer,
    TrainingConfig,
    train_model,
    load_trained_model,
    TextDataset,
    QADataset,
)

# Tokenizers
from .tokenizer import (
    get_tokenizer,
    load_tokenizer,
    train_tokenizer,
    SimpleTokenizer,
)

# Try to import advanced tokenizer (may fail if dependencies missing)
try:
    from .advanced_tokenizer import AdvancedBPETokenizer
except ImportError:
    AdvancedBPETokenizer = None

# Try to import character tokenizer
try:
    from .char_tokenizer import CharacterTokenizer
except ImportError:
    CharacterTokenizer = None

# Model configuration
try:
    from .model_config import get_model_config
except ImportError:
    get_model_config = None

# Model registry
from .model_registry import ModelRegistry

# Hardware detection
try:
    from .hardware import get_hardware, HardwareProfile
except ImportError:
    get_hardware = None
    HardwareProfile = None


__all__ = [
    # Model
    "Enigma",
    "TinyEnigma",
    "EnigmaConfig",
    "create_model",
    "MODEL_PRESETS",

    # Inference
    "EnigmaEngine",
    "generate",
    "load_engine",

    # Training
    "Trainer",
    "TrainingConfig",
    "train_model",
    "load_trained_model",
    "TextDataset",
    "QADataset",

    # Tokenizers
    "get_tokenizer",
    "load_tokenizer",
    "train_tokenizer",
    "SimpleTokenizer",
    "AdvancedBPETokenizer",
    "CharacterTokenizer",

    # Config & Registry
    "get_model_config",
    "ModelRegistry",

    # Hardware
    "get_hardware",
    "HardwareProfile",
]
