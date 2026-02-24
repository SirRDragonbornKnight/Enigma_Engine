# core package - Model and Inference
"""
Enigma Core Module
==================

Core components for AI model loading and inference:
- Model architecture (Transformer with RoPE, RMSNorm, SwiGLU, GQA)
- Inference engine (KV-cache, streaming, chat)
- Tokenization (BPE, character-level)
- External model loaders (GGUF, HuggingFace, Ollama, ONNX)
"""

# Hardware detection
try:
    from .hardware_detection import HardwareProfile, get_hardware
except ImportError:
    HardwareProfile = None
    get_hardware = None

# Model configuration
try:
    from .model_config import get_model_config
except ImportError:
    get_model_config = None

# Tokenizers
try:
    from .tokenizer import SimpleTokenizer, get_tokenizer, load_tokenizer, train_tokenizer
except ImportError:
    SimpleTokenizer = None
    get_tokenizer = None
    load_tokenizer = None
    train_tokenizer = None

try:
    from .bpe_tokenizer import BPETokenizer
except ImportError:
    BPETokenizer = None

try:
    from .char_tokenizer import CharacterTokenizer
except ImportError:
    CharacterTokenizer = None

# KV Cache
try:
    from .kv_cache import KVCache
except ImportError:
    KVCache = None

# Streaming
try:
    from .streaming import TokenStreamer
except ImportError:
    TokenStreamer = None

# Command system
try:
    from .commands import CommandRegistry, get_registry, parse_commands, CommandResult
except ImportError:
    CommandRegistry = None
    get_registry = None
    parse_commands = None
    CommandResult = None

# External model loaders
try:
    from .gguf_loader import load_gguf_model
except ImportError:
    load_gguf_model = None

try:
    from .huggingface_loader import load_huggingface_model
except ImportError:
    load_huggingface_model = None

try:
    from .ollama_loader import load_ollama_model
except ImportError:
    load_ollama_model = None

try:
    from .onnx_loader import load_onnx_model
except ImportError:
    load_onnx_model = None

try:
    from .gptq_awq_loader import load_gptq_model, load_awq_model
except ImportError:
    load_gptq_model = None
    load_awq_model = None

# AI Profiles
try:
    from .ai_profile import AIProfile, load_profile, save_profile
except ImportError:
    AIProfile = None
    load_profile = None
    save_profile = None

# Per-model context storage
try:
    from .model_context import (
        ModelContext, model_key_from_path,
        load_model_context, list_model_contexts,
    )
except ImportError:
    ModelContext = None
    model_key_from_path = None
    load_model_context = None
    list_model_contexts = None

# Core model and inference (lazy imports to avoid torch import at startup)
def _lazy_load_model():
    from .model import Enigma, ForgeConfig, create_model, MODEL_PRESETS
    return Enigma, ForgeConfig, create_model, MODEL_PRESETS

def _lazy_load_inference():
    from .inference import EnigmaEngine, ForgeEngine, generate, load_engine
    return EnigmaEngine, ForgeEngine, generate, load_engine

# Expose commonly accessed items through __getattr__ for lazy loading
_lazy_cache = {}

def __getattr__(name):
    """Lazy load torch-dependent modules only when accessed."""
    if name in ('Enigma', 'ForgeConfig', 'create_model', 'MODEL_PRESETS'):
        if 'model' not in _lazy_cache:
            Enigma, ForgeConfig, create_model, MODEL_PRESETS = _lazy_load_model()
            _lazy_cache['model'] = {
                'Enigma': Enigma,
                'ForgeConfig': ForgeConfig,
                'create_model': create_model,
                'MODEL_PRESETS': MODEL_PRESETS,
            }
        return _lazy_cache['model'][name]
    
    if name in ('EnigmaEngine', 'ForgeEngine', 'generate', 'load_engine'):
        if 'inference' not in _lazy_cache:
            EnigmaEngine, ForgeEngine, generate, load_engine = _lazy_load_inference()
            _lazy_cache['inference'] = {
                'EnigmaEngine': EnigmaEngine,
                'ForgeEngine': ForgeEngine,
                'generate': generate,
                'load_engine': load_engine,
            }
        return _lazy_cache['inference'][name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Hardware
    'HardwareProfile',
    'get_hardware',
    # Model config
    'get_model_config',
    # Tokenizers
    'SimpleTokenizer',
    'get_tokenizer',
    'load_tokenizer',
    'train_tokenizer',
    'BPETokenizer',
    'CharacterTokenizer',
    # KV Cache
    'KVCache',
    # Streaming
    'TokenStreamer',
    # Commands
    'CommandRegistry',
    'get_registry',
    'parse_commands',
    'CommandResult',
    # External loaders
    'load_gguf_model',
    'load_huggingface_model',
    'load_ollama_model',
    'load_onnx_model',
    'load_gptq_model',
    'load_awq_model',
    # AI Profiles
    'AIProfile',
    'load_profile',
    'save_profile',
    # Per-model context
    'ModelContext',
    'model_key_from_path',
    'load_model_context',
    'list_model_contexts',
    # Model (lazy)
    'Enigma',
    'ForgeConfig',
    'create_model',
    'MODEL_PRESETS',
    # Inference (lazy)
    'EnigmaEngine',
    'ForgeEngine',
    'generate',
    'load_engine',
]
