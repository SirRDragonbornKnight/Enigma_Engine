# core package - Forge Model and Training
"""
Forge Core Module
==================

Contains the core components of the ForgeAI AI framework:
- Model architecture (Transformer with RoPE, RMSNorm, SwiGLU, GQA)
- Training system (AMP, gradient accumulation, cosine warmup)
- Inference engine (KV-cache, streaming, chat)
- Tokenization (BPE, character-level)
"""

# Model
from .model import (
    Forge,
    TinyForge,  # Backwards compatibility alias
    Enigma,     # Legacy name for backwards compatibility
    ForgeConfig,
    create_model,
    MODEL_PRESETS,
)

# Inference
from .inference import ForgeEngine, generate, load_engine

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

# Try to import Forge tokenizer (may fail if dependencies missing)
try:
    from .advanced_tokenizer import ForgeTokenizer, AdvancedBPETokenizer  # AdvancedBPETokenizer is alias
except ImportError:
    ForgeTokenizer = None
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

# Quantization (optional)
try:
    from .quantization import quantize_model, load_quantized
except ImportError:
    quantize_model = None
    load_quantized = None

# HuggingFace model loading (optional - lazy load to avoid slow imports)
HuggingFaceModel = None
HuggingFaceEngine = None
load_huggingface_model = None

def _lazy_load_huggingface():
    """Lazy load HuggingFace components on first use."""
    global HuggingFaceModel, HuggingFaceEngine, load_huggingface_model
    if HuggingFaceModel is None:
        try:
            from .huggingface_loader import (
                HuggingFaceModel as _HFM,
                HuggingFaceEngine as _HFE,
                load_huggingface_model as _load,
            )
            HuggingFaceModel = _HFM
            HuggingFaceEngine = _HFE
            load_huggingface_model = _load
        except ImportError:
            pass
    return HuggingFaceModel, HuggingFaceEngine, load_huggingface_model

# GGUF model loading (optional)
try:
    from .gguf_loader import GGUFModel
except ImportError:
    GGUFModel = None

# HuggingFace exporter (for uploading ForgeAI models to HF Hub)
HuggingFaceExporter = None
export_model_to_hub = None
export_model_locally = None

def _lazy_load_hf_exporter():
    """Lazy load HuggingFace exporter on first use."""
    global HuggingFaceExporter, export_model_to_hub, export_model_locally
    if HuggingFaceExporter is None:
        try:
            from .huggingface_exporter import (
                HuggingFaceExporter as _HFE,
                export_model_to_hub as _export_hub,
                export_model_locally as _export_local,
            )
            HuggingFaceExporter = _HFE
            export_model_to_hub = _export_hub
            export_model_locally = _export_local
        except ImportError:
            pass
    return HuggingFaceExporter, export_model_to_hub, export_model_locally

# Multi-platform model export system
ModelExporter = None
export_model = None
list_export_providers = None

def _lazy_load_model_exporter():
    """Lazy load model export system on first use."""
    global ModelExporter, export_model, list_export_providers
    if ModelExporter is None:
        try:
            from .model_export import (
                ModelExporter as _ME,
                export_model as _export,
                list_export_providers as _list_providers,
            )
            ModelExporter = _ME
            export_model = _export
            list_export_providers = _list_providers
        except ImportError:
            pass
    return ModelExporter, export_model, list_export_providers

# AI Wants & Motivation System (optional)
AIWantsSystem = None
get_wants_system = None

def _lazy_load_wants_system():
    """Lazy load AI wants system on first use."""
    global AIWantsSystem, get_wants_system
    if AIWantsSystem is None:
        try:
            from .wants_system import (
                AIWantsSystem as _AWS,
                get_wants_system as _get_wants,
            )
            AIWantsSystem = _AWS
            get_wants_system = _get_wants
        except ImportError:
            pass
    return AIWantsSystem, get_wants_system

# Learned Generator System (optional)
LearnedGenerator = None

def _lazy_load_learned_generator():
    """Lazy load learned generator on first use."""
    global LearnedGenerator
    if LearnedGenerator is None:
        try:
            from .learned_generator import AILearnedGenerator as _LG
            LearnedGenerator = _LG
        except ImportError:
            pass
    return LearnedGenerator


__all__ = [
    # Model
    "Forge",
    "TinyForge",
    "ForgeConfig",
    "create_model",
    "MODEL_PRESETS",

    # Inference
    "ForgeEngine",
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
    "ForgeTokenizer",
    "AdvancedBPETokenizer",  # Backwards compatibility alias
    "CharacterTokenizer",

    # Config & Registry
    "get_model_config",
    "ModelRegistry",

    # Hardware
    "get_hardware",
    "HardwareProfile",

    # Quantization
    "quantize_model",
    "load_quantized",
    
    # External model loading
    "HuggingFaceModel",
    "HuggingFaceEngine",
    "load_huggingface_model",
    "GGUFModel",
    
    # HuggingFace export (upload ForgeAI models to Hub)
    "HuggingFaceExporter",
    "export_model_to_hub",
    "export_model_locally",
    
    # Multi-platform model export
    "ModelExporter",
    "export_model",
    "list_export_providers",
    
    # AI Wants & Motivation System
    "AIWantsSystem",
    "get_wants_system",
    
    # Learned Generator
    "LearnedGenerator",
]
