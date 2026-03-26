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

# KV Cache — lazy to avoid torch import at startup
# Accessed via __getattr__ below.

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

# External model loaders — lazy-loaded to avoid importing torch/transformers
# at startup. Accessed via __getattr__ below.

# Reasoning / Chain-of-thought
try:
    from .reasoning import (
        THINK_START, THINK_END,
        extract_reasoning, strip_reasoning, has_reasoning,
        wrap_reasoning, build_reasoning_instruction, format_reasoning_example,
        extract_all_reasoning, count_reasoning_steps,
        build_multistep_reasoning_instruction, format_multistep_example,
    )
except ImportError:
    THINK_START = None
    THINK_END = None
    extract_reasoning = None
    strip_reasoning = None
    has_reasoning = None
    wrap_reasoning = None
    build_reasoning_instruction = None
    format_reasoning_example = None
    extract_all_reasoning = None
    count_reasoning_steps = None
    build_multistep_reasoning_instruction = None
    format_multistep_example = None

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

# Vision encoder (lazy to avoid torch import at startup)
# Accessed via __getattr__ below.

# Core model and inference (lazy imports to avoid torch import at startup)
def _lazy_load_model():
    from .model import Enigma, ForgeConfig, create_model, MODEL_PRESETS
    return Enigma, ForgeConfig, create_model, MODEL_PRESETS

def _lazy_load_inference():
    from .inference import EnigmaEngine, ForgeEngine, generate, load_engine
    return EnigmaEngine, ForgeEngine, generate, load_engine

# Expose commonly accessed items through __getattr__ for lazy loading
_lazy_cache = {}

# Lazy loader mappings: attribute name → (module, attr, cache_key)
_LAZY_LOADER_MAP = {
    'load_gguf_model': ('.gguf_loader', 'load_gguf_model', 'gguf_loader'),
    'load_huggingface_model': ('.huggingface_loader', 'load_huggingface_model', 'hf_loader'),
    'load_ollama_model': ('.ollama_loader', 'load_ollama_model', 'ollama_loader'),
    'load_onnx_model': ('.onnx_loader', 'load_onnx_model', 'onnx_loader'),
    'load_gptq_model': ('.gptq_awq_loader', 'load_gptq_model', 'gptq_awq_loader'),
    'load_awq_model': ('.gptq_awq_loader', 'load_awq_model', 'gptq_awq_loader'),
    # Vision encoder
    'VisionEncoder': ('.vision_encoder', 'VisionEncoder', 'vision_encoder'),
    'VisionEncoderConfig': ('.vision_encoder', 'VisionEncoderConfig', 'vision_encoder'),
    'VISION_PRESETS': ('.vision_encoder', 'VISION_PRESETS', 'vision_encoder'),
    'IMAGENET_MEAN': ('.vision_encoder', 'IMAGENET_MEAN', 'vision_encoder'),
    'IMAGENET_STD': ('.vision_encoder', 'IMAGENET_STD', 'vision_encoder'),
    'CNNStem': ('.vision_encoder', 'CNNStem', 'vision_encoder'),
    'encode_image': ('.vision_encoder', 'encode_image', 'vision_encoder'),
    'preprocess_image': ('.vision_encoder', 'preprocess_image', 'vision_encoder'),
    'encode_screen': ('.vision_encoder', 'encode_screen', 'vision_encoder'),
    'encode_camera': ('.vision_encoder', 'encode_camera', 'vision_encoder'),
    'encode_video_frames': ('.vision_encoder', 'encode_video_frames', 'vision_encoder'),
    'TemporalConv1d': ('.vision_encoder', 'TemporalConv1d', 'vision_encoder'),
    # Audio encoder
    'AudioEncoder': ('.audio_encoder', 'AudioEncoder', 'audio_encoder'),
    'AudioEncoderConfig': ('.audio_encoder', 'AudioEncoderConfig', 'audio_encoder'),
    'AUDIO_PRESETS': ('.audio_encoder', 'AUDIO_PRESETS', 'audio_encoder'),
    'load_audio': ('.audio_encoder', 'load_audio', 'audio_encoder'),
    'preprocess_audio': ('.audio_encoder', 'preprocess_audio', 'audio_encoder'),
    'log_mel_spectrogram': ('.audio_encoder', 'log_mel_spectrogram', 'audio_encoder'),
    'mel_filterbank': ('.audio_encoder', 'mel_filterbank', 'audio_encoder'),
    'spec_augment': ('.audio_encoder', 'spec_augment', 'audio_encoder'),
    # Multi-GPU (MG-B / MG-C)
    'get_gpu_count': ('.multi_gpu', 'get_gpu_count', 'multi_gpu'),
    'get_gpu_info': ('.multi_gpu', 'get_gpu_info', 'multi_gpu'),
    'is_multi_gpu': ('.multi_gpu', 'is_multi_gpu', 'multi_gpu'),
    'wrap_data_parallel': ('.multi_gpu', 'wrap_data_parallel', 'multi_gpu'),
    'unwrap_data_parallel': ('.multi_gpu', 'unwrap_data_parallel', 'multi_gpu'),
    'DistributedConfig': ('.multi_gpu', 'DistributedConfig', 'multi_gpu'),
    'DistributedTrainer': ('.multi_gpu', 'DistributedTrainer', 'multi_gpu'),
    # Chat export
    'export_html': ('.chat_export', 'export_html', 'chat_export'),
    'export_pdf': ('.chat_export', 'export_pdf', 'chat_export'),
    'history_to_html': ('.chat_export', 'history_to_html', 'chat_export'),
}

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

    # Lazy-load external model loaders (avoids torch/transformers at import)
    if name in _LAZY_LOADER_MAP:
        mod_path, attr, cache_key = _LAZY_LOADER_MAP[name]
        if cache_key not in _lazy_cache:
            import importlib
            try:
                mod = importlib.import_module(mod_path, __name__)
                _lazy_cache[cache_key] = {
                    a: getattr(mod, a, None)
                    for a in _LAZY_LOADER_MAP
                    if _LAZY_LOADER_MAP[a][2] == cache_key
                }
            except ImportError:
                _lazy_cache[cache_key] = {
                    a: None
                    for a in _LAZY_LOADER_MAP
                    if _LAZY_LOADER_MAP[a][2] == cache_key
                }
        return _lazy_cache[cache_key].get(name)

    # Lazy-load KVCache (imports torch at module level)
    if name == 'KVCache':
        if 'kv_cache' not in _lazy_cache:
            try:
                from .kv_cache import KVCache as _KV
                _lazy_cache['kv_cache'] = _KV
            except ImportError:
                _lazy_cache['kv_cache'] = None
        return _lazy_cache['kv_cache']

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
    # Reasoning
    'THINK_START',
    'THINK_END',
    'extract_reasoning',
    'strip_reasoning',
    'has_reasoning',
    'wrap_reasoning',
    'build_reasoning_instruction',
    'format_reasoning_example',
    'extract_all_reasoning',
    'count_reasoning_steps',
    'build_multistep_reasoning_instruction',
    'format_multistep_example',
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
    # Multi-GPU (lazy)
    'get_gpu_count',
    'get_gpu_info',
    'is_multi_gpu',
    'wrap_data_parallel',
    'unwrap_data_parallel',
    'DistributedConfig',
    'DistributedTrainer',
    # Chat export (lazy)
    'export_html',
    'export_pdf',
    'history_to_html',
]
