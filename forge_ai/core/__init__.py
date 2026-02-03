# core package - Forge Model and Training
"""
Forge Core Module
==================

Contains the core components of the ForgeAI AI framework:
- Model architecture (Transformer with RoPE, RMSNorm, SwiGLU, GQA)
- Training system (AMP, gradient accumulation, cosine warmup)
- Inference engine (KV-cache, streaming, chat)
- Tokenization (BPE, character-level)
- Self-Improvement & Autonomous Learning
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

# Self-Improvement System (NEW)
try:
    from .self_improvement import (
        LearningEngine,
        LearningExample,
        LearningSource,
        Priority,
        PerformanceMetrics,
        AutonomousConfig,
        get_learning_engine,
    )
except ImportError:
    LearningEngine = None
    LearningExample = None
    LearningSource = None
    Priority = None
    PerformanceMetrics = None
    AutonomousConfig = None
    get_learning_engine = None

# Autonomous Mode (UPDATED with real implementations)
try:
    from .autonomous import AutonomousMode, AutonomousManager, AutonomousAction
except ImportError:
    AutonomousMode = None
    AutonomousManager = None
    AutonomousAction = None

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

# Engine Pool (for efficient engine reuse)
try:
    from .engine_pool import (
        EnginePool, EnginePoolConfig, PooledEngine,
        get_engine, release_engine, get_engine_context,
        get_pool, create_fallback_response,
    )
except ImportError:
    EnginePool = None
    EnginePoolConfig = None
    PooledEngine = None
    get_engine = None
    release_engine = None
    get_engine_context = None
    get_pool = None
    create_fallback_response = None

# Prompt Builder (for consistent prompt formatting)
try:
    from .prompt_builder import (
        PromptBuilder, PromptTemplate, get_prompt_builder,
        build_chat_prompt, extract_response,
    )
except ImportError:
    PromptBuilder = None
    PromptTemplate = None
    get_prompt_builder = None
    build_chat_prompt = None
    extract_response = None

# Hardware detection
try:
    from .hardware import get_hardware, HardwareProfile
except ImportError:
    get_hardware = None
    HardwareProfile = None

# Advanced hardware detection (Pi, edge devices)
try:
    from .hardware_detection import (
        detect_hardware,
        recommend_model_size,
        get_optimal_config,
        estimate_memory_usage,
        HardwareProfile as AdvancedHardwareProfile,
    )
except ImportError:
    detect_hardware = None
    recommend_model_size = None
    get_optimal_config = None
    estimate_memory_usage = None
    AdvancedHardwareProfile = None

# Quantization (optional)
try:
    from .quantization import quantize_model, load_quantized, auto_quantize, QuantConfig
except ImportError:
    quantize_model = None
    load_quantized = None
    auto_quantize = None
    QuantConfig = None

# Device Profiles (optional)
try:
    from .device_profiles import (
        DeviceProfiler, DeviceClass, DeviceCapabilities, ProfileSettings,
        get_device_profiler, get_optimal_settings, get_recommended_model_size,
    )
except ImportError:
    DeviceProfiler = None
    DeviceClass = None
    DeviceCapabilities = None
    ProfileSettings = None
    get_device_profiler = None
    get_optimal_settings = None
    get_recommended_model_size = None

# Low Power Inference (optional)
try:
    from .low_power_inference import LowPowerEngine, LowPowerConfig
except ImportError:
    LowPowerEngine = None
    LowPowerConfig = None

# Gaming Mode (optional)
try:
    from .gaming_mode import (
        GamingMode, GamingPriority, GamingProfile, ResourceLimits,
        get_gaming_mode,
    )
except ImportError:
    GamingMode = None
    GamingPriority = None
    GamingProfile = None
    ResourceLimits = None
    get_gaming_mode = None

# Adaptive Engine (optional)
try:
    from .adaptive_engine import (
        AdaptiveEngine, AdaptiveConfig, AdaptiveMode,
        get_adaptive_engine,
    )
except ImportError:
    AdaptiveEngine = None
    AdaptiveConfig = None
    AdaptiveMode = None
    get_adaptive_engine = None

# Universal Tool Router (works with ANY model)
try:
    from .universal_router import (
        UniversalToolRouter,
        get_universal_router,
        chat_with_tools,
    )
except ImportError:
    UniversalToolRouter = None
    get_universal_router = None
    chat_with_tools = None

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

# Orchestration System (optional)
try:
    from .capability_registry import (
        CapabilityRegistry,
        Capability,
        ModelCapabilityEntry,
        get_capability_registry,
    )
except ImportError:
    CapabilityRegistry = None
    Capability = None
    ModelCapabilityEntry = None
    get_capability_registry = None

try:
    from .model_pool import (
        ModelPool,
        ModelPoolConfig,
        ModelEntry,
        get_model_pool,
    )
except ImportError:
    ModelPool = None
    ModelPoolConfig = None
    ModelEntry = None
    get_model_pool = None

try:
    from .collaboration import (
        ModelCollaboration,
        CollaborationType,
        CollaborationRequest,
        CollaborationResponse,
        get_collaboration,
    )
except ImportError:
    ModelCollaboration = None
    CollaborationType = None
    CollaborationRequest = None
    CollaborationResponse = None
    get_collaboration = None

try:
    from .orchestrator import (
        ModelOrchestrator,
        OrchestratorConfig,
        Task,
        TaskResult,
        get_orchestrator,
    )
except ImportError:
    ModelOrchestrator = None
    OrchestratorConfig = None
    Task = None
    TaskResult = None
    get_orchestrator = None

try:
    from .task_offloader import (
        TaskOffloader,
        OffloaderConfig,
        TaskStatus,
        get_offloader,
    )
except ImportError:
    TaskOffloader = None
    OffloaderConfig = None
    TaskStatus = None
    get_offloader = None

try:
    from .standalone_tools import (
        use_tool,
        list_available_tools,
        get_tool_info,
    )
except ImportError:
    use_tool = None
    list_available_tools = None
    get_tool_info = None

# Persona System
try:
    from .persona import (
        AIPersona,
        PersonaManager,
        get_persona_manager,
    )
except ImportError:
    AIPersona = None
    PersonaManager = None
    get_persona_manager = None


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
    
    # Self-Improvement System (NEW)
    "LearningEngine",
    "LearningExample",
    "LearningSource",
    "Priority",
    "PerformanceMetrics",
    "AutonomousConfig",
    "get_learning_engine",
    
    # Autonomous Mode (UPDATED)
    "AutonomousMode",
    "AutonomousManager",
    "AutonomousAction",

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
    
    # Advanced hardware detection (Pi, edge devices)
    "detect_hardware",
    "recommend_model_size",
    "get_optimal_config",
    "estimate_memory_usage",
    "AdvancedHardwareProfile",

    # Quantization
    "quantize_model",
    "load_quantized",
    "auto_quantize",
    "QuantConfig",
    
    # Device Profiles
    "DeviceProfiler",
    "DeviceClass",
    "DeviceCapabilities",
    "ProfileSettings",
    "get_device_profiler",
    "get_optimal_settings",
    "get_recommended_model_size",
    
    # Low Power Inference
    "LowPowerEngine",
    "LowPowerConfig",
    
    # Gaming Mode
    "GamingMode",
    "GamingPriority",
    "GamingProfile",
    "ResourceLimits",
    "get_gaming_mode",
    
    # Adaptive Engine
    "AdaptiveEngine",
    "AdaptiveConfig",
    "AdaptiveMode",
    "get_adaptive_engine",
    
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
    
    # Orchestration System
    "CapabilityRegistry",
    "Capability",
    "ModelCapabilityEntry",
    "get_capability_registry",
    "ModelPool",
    "ModelPoolConfig",
    "ModelEntry",
    "get_model_pool",
    "ModelCollaboration",
    "CollaborationType",
    "CollaborationRequest",
    "CollaborationResponse",
    "get_collaboration",
    "ModelOrchestrator",
    "OrchestratorConfig",
    "Task",
    "TaskResult",
    "get_orchestrator",
    "TaskOffloader",
    "OffloaderConfig",
    "TaskStatus",
    "get_offloader",
    "use_tool",
    "list_available_tools",
    "get_tool_info",
    
    # Persona System
    "AIPersona",
    "PersonaManager",
    "get_persona_manager",
]
