"""
================================================================================
                CHAPTER 2: THE ORACLE - SPEAKING WITH YOUR AI
================================================================================

    "You have built the mind. Now learn to converse with it."

Congratulations, adventurer! If you made it through model.py (Chapter 1),
you now understand HOW the AI thinks. This chapter teaches you how to
actually TALK to it.

WHY THIS FILE MATTERS:
    The Enigma model (model.py) is just a brain in a jar - powerful but silent.
    EnigmaEngine is the VOICE. It takes your questions, feeds them to the
    brain, and brings back answers. Every conversation you have with 
    Enigma AI Engine passes through this file.

THE MAGIC PROCESS:
    ┌─────────────────────────────────────────────────────────────┐
    │  YOU: "What is the meaning of life?"                        │
    │   │                                                         │
    │   ↓  (EnigmaEngine encodes your words into numbers)         │
    │  [15496, 318, 262, 3616, ...]                               │
    │   │                                                         │
    │   ↓  (Sends numbers through the Enigma brain)               │
    │  [Matrix multiplication magic x millions]                   │
    │   │                                                         │
    │   ↓  (Gets probability for each possible next word)         │
    │  "The" 0.3, "It" 0.2, "42" 0.15, ...                       │
    │   │                                                         │
    │   ↓  (Picks one, repeats until done)                        │
    │  AI: "The meaning of life is to find purpose..."           │
    └─────────────────────────────────────────────────────────────┘

SPEAKING STYLES (Sampling Strategies):
    | Style      | Description                | When to Use           |
    |------------|----------------------------|-----------------------|
    | Greedy     | Always pick most likely    | Facts, consistency    |
    | Top-K      | Pick from top K choices    | Creative but coherent |
    | Top-P      | Pick from top P% probable  | Natural conversation  |
    | Temperature| Higher = more wild         | Stories, brainstorming|

YOUR FIRST CONVERSATION:
    >>> from enigma_engine.core.inference import EnigmaEngine
    >>> oracle = EnigmaEngine()
    >>> oracle.chat("Tell me a joke about AI")
    "Why did the AI go to therapy? Too many neural issues!"

CONNECTED PATHS:
    You came from → model.py (Chapter 1: The Brain)
    You can go to → tool_router.py (Chapter 3: The Dispatcher)
                  → chat_tab.py (The GUI interface)
                  → api_server.py (REST API for remote access)
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..config import CONFIG
from ..utils.system_messages import info_msg, system_msg
from .model import MODEL_PRESETS, Forge, create_model
from .tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

# Default model paths
MODELS_DIR = Path(CONFIG.get("models_dir", "models"))
DEFAULT_MODEL = MODELS_DIR / "forge.pth"
LEGACY_MODEL = MODELS_DIR / "tiny_enigma_engine.pth"


# =============================================================================
# ⚡ INFERENCE ENGINE - Talk to Your AI!
# =============================================================================
# This is the main class for generating text with a trained model.
# It handles all the complexity of:
#   - Loading models and tokenizers
#   - Running the neural network
#   - Sampling strategies (how to pick the next word)
#   - KV-cache for fast generation
#   - Tool routing for specialized tasks

class EnigmaEngine:
    """
    High-performance inference engine for Enigma models.
    
    📖 WHAT THIS DOES:
    Takes your text prompt and generates a response using the AI model.
    
    📐 GENERATION LOOP:
    ┌────────────────────────────────────────────────────────────────────────┐
    │  "Hello, how are" ─────────────────────────────────────────────────    │
    │         │                                                              │
    │         ▼                                                              │
    │  ┌─────────────┐                                                       │
    │  │ Tokenizer   │ → [15496, 11, 703, 389]                              │
    │  └─────────────┘                                                       │
    │         │                                                              │
    │         ▼                                                              │
    │  ┌─────────────┐     ┌───────────────────────────────────────────┐   │
    │  │   Model     │ ──▶ │ Probabilities for ALL vocab tokens        │   │
    │  └─────────────┘     │ "you": 0.15, "doing": 0.08, "the": 0.02   │   │
    │         │            └───────────────────────────────────────────┘   │
    │         ▼                                                              │
    │  ┌─────────────┐                                                       │
    │  │  Sampler    │ → Pick "you" (based on temperature, top_k, etc.)    │
    │  └─────────────┘                                                       │
    │         │                                                              │
    │         ▼                                                              │
    │  Add "you" to sequence, REPEAT until done                             │
    │         │                                                              │
    │         ▼                                                              │
    │  "Hello, how are you doing today?"                                    │
    └────────────────────────────────────────────────────────────────────────┘
    
    ⚡ KEY FEATURES:
    - KV-cache: Don't recompute past tokens (10x faster!)
    - Multiple samplers: greedy, top-k, top-p, temperature
    - Streaming: Get tokens as they're generated
    - Tools: Route to specialized models/APIs
    - Chat: Maintains conversation history
    
    🎛️ SAMPLING STRATEGIES:
    ┌────────────────────────────────────────────────────────────────────────┐
    │ GREEDY (temperature=0):                                                │
    │   Always pick highest probability token                                │
    │   Pro: Deterministic, consistent                                       │
    │   Con: Repetitive, boring                                              │
    │                                                                        │
    │ TEMPERATURE (0.1 to 2.0):                                              │
    │   Scales probabilities before sampling                                 │
    │   Low (0.3): More focused, predictable                                │
    │   High (1.5): More random, creative                                   │
    │                                                                        │
    │ TOP-K (e.g., k=50):                                                    │
    │   Only consider top K most likely tokens                              │
    │   Prevents sampling very unlikely tokens                              │
    │                                                                        │
    │ TOP-P / NUCLEUS (e.g., p=0.9):                                        │
    │   Only consider tokens covering P% of probability mass                │
    │   Dynamic cutoff based on confidence                                  │
    └────────────────────────────────────────────────────────────────────────┘
    
    Attributes:
        model: The loaded ``Forge`` transformer model instance.
        tokenizer: Tokenizer used to encode/decode text.
        device: ``torch.device`` the model runs on (``cpu``, ``cuda``,
            ``mps``).
        use_half: Whether FP16 precision is enabled.
        enable_tools: Whether the AI tool execution system is active.
        use_routing: Whether specialised model routing is enabled.
        model_metadata: Dict of metadata loaded from alongside the
            model checkpoint (content rating, training info, etc.).

    Example:
        >>> from enigma_engine.core.inference import EnigmaEngine
        >>> engine = EnigmaEngine()
        >>> response = engine.generate("Tell me about AI", max_gen=50)
        >>> print(response)
        >>> reply = engine.chat("Hello!", system_prompt="Be helpful.")

    See Also:
        ``enigma_engine.core.model``:
            The ``Forge`` transformer model architecture.
        ``enigma_engine.core.tokenizer``:
            Tokenizer utilities.
        ``enigma_engine.core.tool_router``:
            Specialised model routing for vision, code, etc.
    """

    @classmethod
    def from_model(
        cls,
        model: Any,
        tokenizer: Any,
        device: str | None = None,
        use_half: bool = False
    ) -> EnigmaEngine:
        """
        Create engine directly from model and tokenizer objects.
        
        📖 USE THIS WHEN:
        You already have a loaded model and tokenizer, and don't want
        the engine to load them again from disk.
        
        📐 EXAMPLE:
            model = create_model('small')
            model.load_state_dict(torch.load('my_model.pth'))
            tokenizer = get_tokenizer()
            engine = EnigmaEngine.from_model(model, tokenizer)
        
        Args:
            model: An Enigma model instance (already loaded)
            tokenizer: A tokenizer instance
            device: Device to use ("cuda", "cpu", or auto-detected)
            use_half: Use FP16 for faster inference (GPU only)
            
        Returns:
            EnigmaEngine instance ready for generation
        """
        import torch

        # Create instance without calling __init__ (bypass normal initialization)
        engine = object.__new__(cls)
        
        # ─────────────────────────────────────────────────────────────────────
        # INITIALIZE REQUIRED ATTRIBUTES
        # ─────────────────────────────────────────────────────────────────────
        engine._generation_lock = threading.Lock()  # Thread safety for KV-cache
        engine.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        engine.use_half = use_half and engine.device.type == "cuda"
        engine.enable_tools = False
        engine.module_manager = None
        engine.use_routing = False
        engine.use_offloading = False
        engine._tool_executor = None
        engine._tool_router = None
        
        # ─────────────────────────────────────────────────────────────────────
        # SET MODEL AND TOKENIZER DIRECTLY
        # ─────────────────────────────────────────────────────────────────────
        engine.tokenizer = tokenizer
        engine.model = model
        
        # Move model to device and set precision
        engine.model.to(engine.device)
        if engine.use_half:
            engine.model.half()  # Convert to FP16
        engine.model.eval()  # Set to evaluation mode (disable dropout)
        
        return engine

    def __init__(
        self,
        model_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        device: str | None = None,
        use_half: bool = False,
        model_size: str = "auto",
        enable_tools: bool = False,
        module_manager: Any | None = None,
        use_routing: bool = False
    ) -> None:
        """
        Initialize the inference engine.
        
        📖 THIS IS THE MAIN CONSTRUCTOR!
        It loads the model and tokenizer, sets up the device,
        and prepares everything for text generation.

        Args:
            model_path: Path to model weights (.pth file)
                        Auto-detected if None (looks in models/ folder)
            tokenizer_path: Path to tokenizer (auto-detected if None)
            device: Device to use:
                    - "cuda" = NVIDIA GPU (fastest)
                    - "cpu" = CPU (slower but always works)
                    - "mps" = Apple Silicon GPU
                    - None = auto-detect best available
            use_half: Use FP16 precision (half the memory, 2x faster on GPU)
            model_size: Model size hint if creating new model
            enable_tools: Enable AI tool system (web search, code, etc.)
            module_manager: ModuleManager for tool execution
            use_routing: Enable specialized model routing
        """
        # ─────────────────────────────────────────────────────────────────────
        # THREAD SAFETY: Lock for KV-cache operations
        # ─────────────────────────────────────────────────────────────────────
        # The KV-cache is stateful - only one generation can run at a time
        self._generation_lock = threading.Lock()
        
        # ─────────────────────────────────────────────────────────────────────
        # DEVICE SELECTION: Pick the best available hardware
        # ─────────────────────────────────────────────────────────────────────
        self.device = self._select_device(device)
        self.use_half = use_half and self.device.type == "cuda"
        
        # ─────────────────────────────────────────────────────────────────────
        # STORE CONFIGURATION
        # ─────────────────────────────────────────────────────────────────────
        self.enable_tools = enable_tools
        self.module_manager = module_manager
        self.use_routing = use_routing
        
        # Check if CPU/GPU offloading is enabled in config
        self.use_offloading = CONFIG.get("enable_offloading", False)
        
        # ─────────────────────────────────────────────────────────────────────
        # TOOL SYSTEM SETUP (optional)
        # ─────────────────────────────────────────────────────────────────────
        self._tool_executor = None
        if enable_tools:
            from ..tools.tool_executor import ToolExecutor
            self._tool_executor = ToolExecutor(module_manager=module_manager)
        
        # Tool router for specialized models (vision, code, etc.)
        self._tool_router = None
        if use_routing:
            from .tool_router import get_router
            self._tool_router = get_router(use_specialized=True)
            logger.info("Specialized model routing enabled")

        # ─────────────────────────────────────────────────────────────────────
        # LOAD TOKENIZER
        # ─────────────────────────────────────────────────────────────────────
        self.tokenizer = self._load_tokenizer(tokenizer_path, model_path)

        # ─────────────────────────────────────────────────────────────────────
        # LOAD MODEL
        # ─────────────────────────────────────────────────────────────────────
        self.model = self._load_model(model_path, model_size)

        # ─────────────────────────────────────────────────────────────────────
        # APPLY DEVICE PLACEMENT
        # ─────────────────────────────────────────────────────────────────────
        if self.use_offloading:
            # Advanced: Split model across CPU+GPU for large models
            self._apply_offloading()
        else:
            # Standard: Move whole model to device
            self.model.to(self.device)
            if self.use_half:
                self.model.half()
        
        # Set to evaluation mode (disables dropout, etc.)
        self.model.eval()

        # ─────────────────────────────────────────────────────────────────────
        # LOAD MODEL METADATA (including content rating support)
        # ─────────────────────────────────────────────────────────────────────
        self._load_model_metadata(model_path)

        # Log what we loaded
        self._log_init_info()

    def _select_device(self, device: str | None) -> torch.device:
        """Select the best available device."""
        if device is not None:
            return torch.device(device)

        # Check power mode settings
        try:
            from .power_mode import get_power_manager
            power_mgr = get_power_manager()
            if not power_mgr.should_use_gpu():
                # Power mode disabled GPU - use CPU
                cpu_threads = CONFIG.get("cpu_threads", 0)
                if cpu_threads > 0:
                    torch.set_num_threads(cpu_threads)
                return torch.device("cpu")
        except ImportError:
            logger.debug("Power mode module not available, skipping power management")

        if torch.cuda.is_available():
            # Apply GPU memory limit from config
            gpu_fraction = CONFIG.get("gpu_memory_fraction", 0.9)
            try:
                torch.cuda.set_per_process_memory_fraction(gpu_fraction)
            except (RuntimeError, AttributeError) as e:
                logger.debug(f"Could not set GPU memory fraction: {e}")
            return torch.device("cuda")

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")

        # Apply CPU thread limit from config
        cpu_threads = CONFIG.get("cpu_threads", 0)
        if cpu_threads > 0:
            torch.set_num_threads(cpu_threads)

        return torch.device("cpu")

    def _apply_offloading(self) -> None:
        """Apply CPU+GPU offloading to the model."""
        try:
            from .offloading import OffloadingConfig, apply_offloading, get_memory_info

            # Log memory info
            mem_info = get_memory_info()
            logger.info(f"[Forge:Offload] CPU RAM available: {mem_info['cpu_available_gb']:.1f}GB")
            if mem_info["gpus"]:
                for gpu in mem_info["gpus"]:
                    logger.info(f"[Forge:Offload] GPU {gpu['index']}: {gpu['free_gb']:.1f}GB free")
            
            # Get offloading config
            config = OffloadingConfig.from_config()
            
            # Apply offloading
            self.model = apply_offloading(
                self.model,
                device_map="auto",
                offload_folder=config.offload_folder,
                offload_to_disk=config.offload_to_disk
            )
            
            logger.info("[Forge:Offload] Model offloading applied successfully")
            
        except ImportError:
            logger.warning("[Forge:Offload] Could not import offloading module, using standard device")
            self.model.to(self.device)
            if self.use_half:
                self.model.half()
        except Exception as e:
            logger.warning(f"[Forge:Offload] Offloading failed: {e}, using standard device")
            self.model.to(self.device)
            if self.use_half:
                self.model.half()

    def _load_tokenizer(
        self,
        tokenizer_path: str | Path | None,
        model_path: str | Path | None
    ) -> Any:
        """Load the tokenizer."""
        # Try explicit tokenizer path first
        if tokenizer_path:
            try:
                from .advanced_tokenizer import AdvancedBPETokenizer
                return AdvancedBPETokenizer(vocab_file=Path(tokenizer_path))
            except Exception as e:
                logger.warning(f"Could not load tokenizer from {tokenizer_path}: {e}")

        # Try to find tokenizer next to model file
        if model_path:
            model_path = Path(model_path)
            tok_path = model_path.parent / f"{model_path.stem}_tokenizer.json"
            if tok_path.exists():
                try:
                    from .advanced_tokenizer import AdvancedBPETokenizer
                    return AdvancedBPETokenizer(vocab_file=tok_path)
                except Exception as e:
                    logger.warning(f"Could not load tokenizer from {tok_path}: {e}")

        # Auto-detect model file and find tokenizer
        detected_model = None
        if DEFAULT_MODEL.exists():
            detected_model = DEFAULT_MODEL
        elif LEGACY_MODEL.exists():
            detected_model = LEGACY_MODEL
        else:
            for f in MODELS_DIR.glob("*.pth"):
                detected_model = f
                break
        
        if detected_model:
            tok_path = detected_model.parent / f"{detected_model.stem}_tokenizer.json"
            if tok_path.exists():
                try:
                    from .advanced_tokenizer import AdvancedBPETokenizer
                    tok = AdvancedBPETokenizer(vocab_file=tok_path)
                    logger.info(f"Loaded tokenizer from {tok_path}")
                    return tok
                except Exception as e:
                    logger.warning(f"Could not load tokenizer from {tok_path}: {e}")

        # Fall back to default
        return get_tokenizer()

    def _load_model(
        self,
        model_path: str | Path | None,
        model_size: str
    ) -> Forge:
        """
        Load or create the model.
        
        📖 AUTO-DETECTION:
        If model_size="auto", this method will:
        1. Detect hardware capabilities (RAM, GPU, Pi)
        2. Choose the best model size for this device
        3. Apply quantization if memory is tight
        
        This enables seamless deployment from Raspberry Pi to datacenter!
        """
        # ─────────────────────────────────────────────────────────────────────
        # AUTO-DETECT MODEL SIZE FOR HARDWARE
        # ─────────────────────────────────────────────────────────────────────
        auto_quantize = False
        quantization_mode = "none"
        
        if model_size == "auto":
            try:
                from .hardware_detection import detect_hardware, get_optimal_config

                # Detect hardware
                profile = detect_hardware()
                
                # Get optimal configuration
                config = get_optimal_config(profile)
                model_size = config["model_size"]
                auto_quantize = config.get("quantization", "none") != "none"
                quantization_mode = config.get("quantization", "none")
                
                logger.info(f"[Auto-Detect] Hardware: {profile.hardware_type}")
                if profile.is_raspberry_pi:
                    logger.info(f"[Auto-Detect] Raspberry Pi Model: {profile.pi_model}")
                logger.info(f"[Auto-Detect] RAM: {profile.total_ram_gb:.1f}GB, VRAM: {profile.gpu_vram_gb or 0:.1f}GB")
                logger.info(f"[Auto-Detect] Recommended model: {model_size}")
                if auto_quantize:
                    logger.info(f"[Auto-Detect] Quantization: {quantization_mode}")
                
            except ImportError:
                logger.warning("[Auto-Detect] Hardware detection not available, using 'small'")
                model_size = "small"
            except Exception as e:
                logger.warning(f"[Auto-Detect] Detection failed: {e}, using 'small'")
                model_size = "small"
        
        # ─────────────────────────────────────────────────────────────────────
        # FIND MODEL FILE
        # ─────────────────────────────────────────────────────────────────────
        model_file = None
        if model_path:
            model_file = Path(model_path)
            if not model_file.exists():
                raise FileNotFoundError(
                    f"Model file not found at specified path: {model_file}\n"
                    f"Please ensure the path is correct or train a model using:\n"
                    f"  python run.py --train"
                )
        elif DEFAULT_MODEL.exists():
            model_file = DEFAULT_MODEL
        elif LEGACY_MODEL.exists():
            model_file = LEGACY_MODEL
        else:
            # Look for any .pth file in models dir
            for f in MODELS_DIR.glob("*.pth"):
                model_file = f
                break

        vocab_size = getattr(self.tokenizer, "vocab_size", 8000)

        if model_file and model_file.exists():
            # ─────────────────────────────────────────────────────────────────
            # TRY MEMORY-MAPPED LOADING FOR LARGE MODELS / LOW MEMORY
            # ─────────────────────────────────────────────────────────────────
            use_mmap = False
            if auto_quantize or model_size in ("pi_zero", "pi_4"):
                # Use memory-efficient loading for constrained devices
                try:
                    from .hardware_detection import detect_hardware
                    profile = detect_hardware()
                    if profile.total_ram_gb < 4 or profile.is_raspberry_pi:
                        use_mmap = True
                        logger.info("[Memory] Using memory-mapped loading for low-memory device")
                except ImportError:
                    logger.debug("Hardware detection not available for memory-mapped loading check")
            
            # Load state dict to infer model architecture
            try:
                from .model_registry import safe_load_weights
                
                if use_mmap:
                    # Memory-mapped loading for constrained devices
                    state_dict = safe_load_weights(model_file, map_location="cpu")
                else:
                    state_dict = safe_load_weights(model_file, map_location="cpu")
                    
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model weights from {model_file}: {e}\n"
                    f"The model file may be corrupted or incompatible.\n"
                    f"Try one of the following:\n"
                    f"  1. Train a new model: python run.py --train\n"
                    f"  2. Download a pre-trained model to {MODELS_DIR}\n"
                    f"  3. Check if the file is a valid PyTorch checkpoint"
                ) from e

            # Infer model size from state dict
            detected_size = self._infer_model_size(state_dict)

            # Get vocab size from embedding
            for key in state_dict.keys():
                if 'embed' in key.lower() or 'token' in key.lower():
                    vocab_size = state_dict[key].shape[0]
                    break

            # Create model with correct architecture
            try:
                model = create_model(
                    detected_size,
                    vocab_size=vocab_size
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create model with size '{detected_size}' and vocab_size={vocab_size}: {e}\n"
                    f"The model configuration may be invalid.\n"
                    f"Try creating a model with a standard size: 'tiny', 'small', 'medium', or 'large'"
                ) from e

            # Load weights
            try:
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded model from {model_file}")
            except Exception as e:
                logger.error(
                    f"Failed to load model weights from {model_file}: {e}\n"
                    f"Model architecture mismatch or corrupted weights.\n"
                    f"The model will be initialized with random weights."
                )
                logger.warning(f"Could not load weights: {e}")
            
            # ─────────────────────────────────────────────────────────────────
            # APPLY AUTO-QUANTIZATION IF NEEDED
            # ─────────────────────────────────────────────────────────────────
            if auto_quantize and quantization_mode != "none":
                try:
                    logger.info(f"[Quantization] Applying {quantization_mode} quantization...")
                    model = model.quantize(mode=quantization_mode)
                    logger.info(f"[Quantization] Successfully applied {quantization_mode}")
                except AttributeError:
                    # Model doesn't have quantize method - try manual
                    if quantization_mode == "dynamic":
                        try:
                            import torch.quantization as tq
                            model = tq.quantize_dynamic(
                                model, {torch.nn.Linear}, dtype=torch.qint8
                            )
                            logger.info("[Quantization] Applied dynamic quantization")
                        except Exception as qe:
                            logger.warning(f"[Quantization] Failed to apply: {qe}")
                except Exception as e:
                    logger.warning(f"[Quantization] Could not apply {quantization_mode}: {e}")
        else:
            # No model file found - raise error instead of creating untrained model
            raise FileNotFoundError(
                f"No trained model found in {MODELS_DIR}\n"
                f"To use a trained model:\n"
                f"  1. Train a model: python run.py --train\n"
                f"  2. Download a HuggingFace model via the GUI Model Manager\n"
                f"  3. Or specify model_path when creating EnigmaEngine"
            )

        return model

    def _infer_model_size(self, state_dict: dict) -> str:
        """Infer model size from state dict."""
        # Look for hidden dimension from embedding layer
        hidden_dim = None
        for key, tensor in state_dict.items():
            if ('embed' in key.lower() or 'token' in key.lower()) and tensor.dim() == 2:
                hidden_dim = tensor.shape[1]
                break
        
        # Fallback: look for norm weights
        if hidden_dim is None:
            for key, tensor in state_dict.items():
                if ('ln' in key.lower() or 'norm' in key.lower()) and tensor.dim() == 1:
                    hidden_dim = tensor.shape[0]
                    break

        if hidden_dim is None:
            return "small"

        # Match to preset - MODEL_PRESETS values are ForgeConfig with 'dim' attribute
        for name, preset in MODEL_PRESETS.items():
            preset_dim = getattr(preset, 'dim', None)
            if preset_dim == hidden_dim:
                return name

        # Find closest match
        def get_dim(preset):
            return getattr(preset, 'dim', 512)

        diffs = [(name, abs(get_dim(preset) - hidden_dim))
                 for name, preset in MODEL_PRESETS.items()]
        return min(diffs, key=lambda x: x[1])[0]

    def _load_model_metadata(self, model_path: Optional[str] = None) -> None:
        """
        Load model metadata including content rating capabilities.
        
        Looks for metadata in:
        1. model_metadata.json alongside the model file
        2. 'metadata' key inside the checkpoint dict
        """
        import json
        
        self.model_metadata = {
            "supports_nsfw": False,
            "content_rating": "sfw",
            "trained_date": None,
            "training_tasks": [],
        }
        
        try:
            # Try to find metadata file
            if model_path:
                model_dir = Path(model_path).parent if Path(model_path).is_file() else Path(model_path)
                metadata_file = model_dir / "model_metadata.json"
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        loaded_metadata = json.load(f)
                    self.model_metadata.update(loaded_metadata)
                    logger.info(f"Loaded model metadata from {metadata_file}")
            
            # Update content filter with model's NSFW capability
            try:
                from .content_rating import get_content_filter
                content_filter = get_content_filter()
                content_filter.set_model_nsfw_capability(self.model_metadata.get("supports_nsfw", False))
                logger.info(f"Model NSFW capability: {self.model_metadata.get('supports_nsfw', False)}")
            except ImportError:
                logger.debug("Content rating module not available")
            except Exception as e:
                logger.warning(f"Could not update content filter: {e}")
                
        except Exception as e:
            logger.debug(f"Could not load model metadata: {e}")

    def _log_init_info(self) -> None:
        """Log initialization information."""
        num_params = sum(p.numel() for p in self.model.parameters())

        logger.info(f"EnigmaEngine initialized on {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Model parameters: {num_params:,}")
        logger.info(f"Vocab size: {self.tokenizer.vocab_size:,}")
        logger.info(f"Max sequence length: {self.model.config.max_seq_len}")
        logger.info(f"FP16: {self.use_half}")

    # =========================================================================
    # 📝 GENERATION METHODS - The Heart of Text Generation
    # =========================================================================

    def generate(
        self,
        prompt: str,
        max_gen: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_strings: list[str] | None = None,
        use_cache: bool = True,
        execute_tools: bool = None,
        max_tool_iterations: int = 5,
        max_tokens: int | None = None,  # Alias for max_gen (backward compatibility)
        max_new_tokens: int | None = None,  # Alias for max_gen (Forge model compatibility)
        max_length: int | None = None  # Alias for max_gen (common parameter name)
    ) -> str:
        """
        Generate text from a prompt.
        
        📖 WHAT THIS DOES:
        This is the main generation function. Give it text, get more text back!
        
        📐 HOW IT WORKS:
        1. Check if prompt needs special routing (image/code/web)
        2. Acquire thread lock (only one generation at a time)
        3. Tokenize the prompt into numbers
        4. Feed tokens to model, get probability distribution
        5. Sample next token using temperature/top-k/top-p
        6. Repeat until max_gen tokens or stop_string found
        7. If AI tried to use tools, execute them and continue
        
        📐 PARAMETER GUIDE:
        ┌────────────────────────────────────────────────────────────────┐
        │ temperature:  Controls randomness                              │
        │   0.1-0.3:   Very focused, predictable                        │
        │   0.7-0.9:   Good balance (default area)                      │
        │   1.0-1.5:   More creative, less coherent                     │
        │   >1.5:      Very random, may be nonsense                     │
        │                                                                │
        │ top_k:       Only consider top K tokens                       │
        │   10-30:     Very focused                                      │
        │   50:        Good default                                      │
        │   100+:      More variety                                      │
        │                                                                │
        │ top_p:       Nucleus sampling - dynamic cutoff                │
        │   0.5:       Conservative, focused                            │
        │   0.9:       Good default                                      │
        │   0.95-1.0:  More variety                                      │
        │                                                                │
        │ repetition_penalty: Discourage repeating words               │
        │   1.0:       No penalty                                        │
        │   1.1:       Mild (good default)                              │
        │   1.3+:      Strong (may break grammar)                       │
        └────────────────────────────────────────────────────────────────┘

        Args:
            prompt: Input text to continue
            max_gen: Maximum tokens to generate (must be > 0)
            temperature: Sampling temperature (higher = more random, > 0)
            top_k: Top-k sampling (>= 0 to disable)
            top_p: Top-p (nucleus) sampling threshold (0-1)
            repetition_penalty: Penalty for repeating tokens (>= 1.0)
            stop_strings: List of strings to stop generation at
            use_cache: Use KV-cache for faster generation
            execute_tools: Execute AI tool calls (default: self.enable_tools)
            max_tool_iterations: Max times AI can call tools in one generation

        Returns:
            Generated text (including the prompt)

        Raises:
            ValueError: If parameters are out of valid range
            TypeError: If prompt is not a string
        """
        # Handle max_tokens, max_new_tokens, max_length aliases for backward compatibility
        if max_tokens is not None:
            max_gen = max_tokens
        if max_new_tokens is not None:
            max_gen = max_new_tokens
        if max_length is not None:
            max_gen = max_length
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 0: Check game mode and apply resource limits
        # ─────────────────────────────────────────────────────────────────────
        try:
            from .game_mode import get_game_mode
            game_mode = get_game_mode()
            
            if game_mode.is_active():
                limits = game_mode.get_resource_limits()
                
                # Check if inference is allowed
                if not limits.inference_allowed:
                    return "AI is paused during game mode."
                
                # Apply token limit for faster responses
                if limits.max_response_tokens > 0:
                    max_gen = min(max_gen, limits.max_response_tokens)
        except Exception as e:
            logger.debug(f"Could not check game mode: {e}")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Determine if tools should be executed
        # ─────────────────────────────────────────────────────────────────────
        if execute_tools is None:
            execute_tools = self.enable_tools
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Check if specialized routing should handle this
        # Some prompts can bypass the main AI for faster execution
        # e.g., "draw a cat" → directly calls image generator
        # ─────────────────────────────────────────────────────────────────────
        if self.use_routing and self._tool_router:
            # Classify what the user wants (image, code, web, etc.)
            intent = self._tool_router.classify_intent(prompt)
            logger.info(f"Classified intent: {intent}")
            
            # Check if this needs AI creativity (ambiguous/creative requests)
            # "surprise me" → needs AI, "draw a cat" → can route directly
            if self._needs_ai_creativity(prompt):
                logger.info("Prompt requires AI creativity, using main AI")
                # Fall through to standard generation
            else:
                # Try direct routing for speed
                direct_result = self._try_direct_routing(intent, prompt)
                if direct_result is not None:
                    return direct_result
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Thread-safe generation (protects KV-cache state)
        # Only one generation can happen at a time!
        # ─────────────────────────────────────────────────────────────────────
        lock = getattr(self, '_generation_lock', None)
        if lock:
            lock.acquire()
        
        try:
            # ─────────────────────────────────────────────────────────────────
            # STEP 4: Standard text generation
            # This is where the actual model inference happens
            # ─────────────────────────────────────────────────────────────────
            text = self._generate_text(
                prompt, max_gen, temperature, top_k, top_p, 
                repetition_penalty, stop_strings, use_cache
            )
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 5: Tool execution loop
            # If AI generated tool calls, execute them and continue
            # ─────────────────────────────────────────────────────────────────
            if execute_tools and self._tool_executor:
                text = self._execute_tools_in_text(
                    text, max_iterations=max_tool_iterations,
                    max_gen=max_gen, temperature=temperature,
                    top_k=top_k, top_p=top_p, 
                    repetition_penalty=repetition_penalty,
                    stop_strings=stop_strings, use_cache=use_cache
                )
        finally:
            if lock:
                lock.release()
        
        return text

    def stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        **kwargs
    ) -> Generator[str]:
        """Stream generated tokens one at a time.

        Instead of waiting for the entire response, each token is yielded
        as soon as it is produced.  This is ideal for chat interfaces
        where the user should see the AI "typing" in real time.

        Args:
            prompt: Input text to continue from.
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional parameters forwarded to
                ``stream_generate()`` (e.g. ``temperature``, ``top_k``,
                ``top_p``, ``repetition_penalty``).

        Yields:
            Each newly generated token string as it is produced.

        Example:
            >>> for token in engine.stream("Once upon a time"):
            ...     print(token, end="", flush=True)
            ' there was a dragon...'
        """
        return self.stream_generate(prompt, max_gen=max_tokens, **kwargs)
    
    def _needs_ai_creativity(self, prompt: str) -> bool:
        """
        Check if the prompt requires AI creativity/context rather than direct execution.
        
        Returns True for ambiguous or creative requests that need AI interpretation.
        """
        prompt_lower = prompt.lower()
        
        # Phrases that indicate need for AI creativity
        creativity_indicators = [
            "what do you think",
            "surprise me",
            "something cool",
            "something interesting",
            "be creative",
            "your choice",
            "you decide",
            "like before",
            "like last time",
            "similar to",
            "in the style of",
            "mood",
            "feeling",
            "vibe",
            "not sure",
            "maybe",
            "suggest",
            "recommend",
            "what would",
            "how about",
            "can you think of",
        ]
        
        for indicator in creativity_indicators:
            if indicator in prompt_lower:
                return True
        
        # Very short prompts might be ambiguous
        words = prompt.split()
        if len(words) <= 2:
            # Single word commands are usually direct ("draw cat")
            # But single words alone are ambiguous
            if len(words) == 1 and words[0].lower() not in ['draw', 'paint', 'generate', 'create', 'make', 'speak', 'say']:
                return True
        
        return False
    
    def _try_direct_routing(self, intent: str, prompt: str) -> str | None:
        """
        Try to handle the request directly without main AI.
        
        Returns the response string if handled, None if should fall through to AI.
        """
        if intent == "image":
            logger.info("Direct routing to image generation")
            return self._direct_generation(prompt, "image", "generate_image")
        
        elif intent == "video":
            logger.info("Direct routing to video generation")
            return self._direct_generation(prompt, "video", "generate_video")
        
        elif intent == "audio":
            logger.info("Direct routing to audio/speech generation")
            return self._direct_generation(prompt, "audio", "speak_text")
        
        elif intent == "3d":
            logger.info("Direct routing to 3D generation")
            return self._direct_generation(prompt, "3D model", "generate_3d")
        
        elif intent == "gif":
            logger.info("Direct routing to GIF generation")
            return self._direct_generation(prompt, "GIF", "generate_gif")
        
        elif intent == "code" and hasattr(self._tool_router, 'generate_code'):
            logger.info("Using specialized code generation model")
            return self._tool_router.generate_code(prompt)
        
        elif intent == "vision" and hasattr(self._tool_router, 'describe_image'):
            logger.info("Vision routing detected, but no features provided")
            # Fall through - vision needs actual image input
            return None
        
        elif intent == "web":
            logger.info("Direct routing to web search")
            return self._direct_web_search(prompt)
        
        # Unknown intent or no direct handler - let AI handle it
        return None
    
    def _direct_generation(self, prompt: str, content_type: str, tool_name: str) -> str:
        """
        Generic direct generation handler for image/video/audio/3D/gif.
        
        Extracts description and calls the appropriate tool directly.
        """
        import re

        # Common patterns for extracting the actual content description
        description = prompt
        
        patterns = [
            r'(?:draw|paint|create|generate|make|produce)\s+(?:me\s+)?(?:a\s+)?(?:picture|image|photo|illustration|artwork|video|clip|animation|sound|audio|speech|model|mesh|gif)?\s*(?:of\s+)?(.+)',
            r'(?:draw|paint|create|generate|make|produce|speak|say|read)\s+(?:me\s+)?(.+)',
            r'(?:picture|image|photo|video|audio|model)\s+of\s+(.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                description = match.group(1).strip()
                break
        
        # Clean up
        description = description.strip('.,!? ')
        if not description:
            description = prompt
        
        logger.info(f"Direct {content_type} generation: {description}")
        
        if not self._tool_executor:
            return f"To generate {content_type}, please use the {content_type.title()} tab. Direct generation not available."
        
        # Execute the tool (tool executor handles auto-loading)
        result = self._tool_executor.execute_tool(tool_name, {"prompt": description})
        
        if result.get("success"):
            path = result.get("path", result.get("result", {}).get("path", ""))
            duration = result.get("duration", 0)
            return f"I've generated {content_type} of '{description}' for you.\n\nSaved to: {path}\nGeneration time: {duration:.1f}s"
        else:
            error = result.get("error", "Unknown error")
            tab_name = content_type.title().replace("3d", "3D")
            return f"I tried to generate {content_type} but encountered an error: {error}\n\nYou can try using the {tab_name} tab directly."
    
    def _direct_web_search(self, prompt: str) -> str:
        """Direct web search without AI intermediary."""
        import re

        # Extract search query
        query = prompt
        patterns = [
            r'(?:search|google|look up|find|browse)\s+(?:for\s+)?(.+)',
            r'what is\s+(.+)',
            r'who is\s+(.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                break
        
        query = query.strip('?,!. ')
        
        if not self._tool_executor:
            return f"To search for '{query}', please use the web tools. Direct search not available."
        
        result = self._tool_executor.execute_tool("web_search", {"query": query})
        
        if result.get("success"):
            search_results = result.get("result", result.get("results", "No results found"))
            return f"Search results for '{query}':\n\n{search_results}"
        else:
            error = result.get("error", "Unknown error")
            return f"Web search failed: {error}"

    # =========================================================================
    # 🔧 INTERNAL GENERATION - Where the magic happens!
    # =========================================================================

    def _generate_text(
        self,
        prompt: str,
        max_gen: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        stop_strings: list[str] | None,
        use_cache: bool
    ) -> str:
        """
        Internal method for standard text generation.
        
        📖 THIS IS THE CORE GENERATION LOOP!
        
        📐 THE AUTOREGRESSIVE LOOP:
        ┌────────────────────────────────────────────────────────────────────┐
        │  tokens = [15496, 11, 703, 389]  # "Hello, how are"               │
        │                                                                    │
        │  REPEAT max_gen times:                                            │
        │    1. Feed tokens to model → logits                               │
        │    2. Apply repetition penalty to logits                          │
        │    3. Apply temperature scaling                                   │
        │    4. Apply top-k filtering                                       │
        │    5. Apply top-p (nucleus) filtering                             │
        │    6. Sample next token from probabilities                        │
        │    7. Add new token to sequence                                   │
        │    8. Check for stop strings                                      │
        │    9. Check for EOS token                                         │
        │                                                                    │
        │  tokens = [15496, 11, 703, 389, 499, 1804, 2651]                  │
        │                                    └─ newly generated             │
        └────────────────────────────────────────────────────────────────────┘
        
        📐 REPETITION PENALTY:
        Discourages the model from repeating the same tokens.
        For each token that already appeared in the sequence,
        divide its probability by repetition_penalty.
        
        📐 TEMPERATURE SCALING:
        logits = logits / temperature
        - Low temp (0.3): Makes high-prob tokens even more likely → focused
        - High temp (1.5): Flattens distribution → more random
        
        📐 TOP-K FILTERING:
        Keep only the K highest probability tokens, zero out the rest.
        Prevents sampling very unlikely tokens.
        
        📐 TOP-P (NUCLEUS) FILTERING:
        Sort tokens by probability, keep tokens until cumulative prob >= p.
        Dynamic cutoff - keeps more tokens when uncertain, fewer when confident.
        """
        # ─────────────────────────────────────────────────────────────────────
        # INPUT VALIDATION
        # ─────────────────────────────────────────────────────────────────────
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")

        if not prompt.strip():
            logger.warning("Empty prompt provided")
            return ""

        if max_gen <= 0:
            raise ValueError(f"max_gen must be positive, got {max_gen}")

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}")

        if not 0 <= top_p <= 1:
            raise ValueError(f"top_p must be between 0 and 1, got {top_p}")

        if repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {repetition_penalty}")

        # ─────────────────────────────────────────────────────────────────────
        # TOKENIZE: Convert text to numbers the model understands
        # "Hello" → [15496]
        # ─────────────────────────────────────────────────────────────────────
        input_ids = self._encode_prompt(prompt)

        # ─────────────────────────────────────────────────────────────────────
        # GENERATE: Run the autoregressive loop
        # ─────────────────────────────────────────────────────────────────────
        with torch.no_grad():  # Disable gradient computation (inference only)
            if use_cache and hasattr(self.model, 'generate'):
                # Use model's built-in generate (has KV-cache optimization)
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_gen,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )
            else:
                # Manual generation
                output_ids = self._generate_manual(
                    input_ids, max_gen, temperature, top_k, top_p, repetition_penalty
                )

        # Decode
        text = self._decode_output(output_ids)

        # Apply stop strings
        if stop_strings:
            for stop_str in stop_strings:
                if stop_str in text:
                    text = text[:text.find(stop_str)]
                    break

        return text

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a prompt to tensor."""
        if hasattr(self.tokenizer, 'encode'):
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        else:
            enc = self.tokenizer(prompt, return_tensors="pt")
            ids = enc["input_ids"]
            if hasattr(ids, 'tolist'):
                ids = ids.tolist()
            if isinstance(ids[0], list):
                ids = ids[0]

        # Convert to tensor
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        return input_ids

    def _decode_output(self, output_ids: torch.Tensor) -> str:
        """Decode output tensor to text."""
        # Handle case where output is already a string
        if isinstance(output_ids, str):
            return output_ids
        
        # Handle tensor output
        try:
            ids = output_ids[0].cpu().tolist()
        except AttributeError:
            # If output_ids[0] doesn't have .cpu(), try direct conversion
            if hasattr(output_ids, '__iter__'):
                ids = list(output_ids[0]) if hasattr(output_ids[0], '__iter__') else [output_ids[0]]
            else:
                return str(output_ids)

        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(ids, skip_special_tokens=True)

        # Fallback
        return "".join(
            self.tokenizer.id_to_token.get(idx, "?")
            for idx in ids
        )

    def _generate_manual(
        self,
        input_ids: torch.Tensor,
        max_gen: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float
    ) -> torch.Tensor:
        """Manual autoregressive generation."""
        generated = input_ids

        for _ in range(max_gen):
            # Truncate if needed
            curr_input = generated
            max_len = self.model.config.max_seq_len
            if curr_input.shape[1] > max_len:
                curr_input = curr_input[:, -max_len:]

            # Forward pass
            logits = self.model(curr_input)

            # Sample next token
            next_token = self._sample_token(
                logits[:, -1, :],
                generated,
                temperature,
                top_k,
                top_p,
                repetition_penalty
            )

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            eos_id = getattr(self.tokenizer, 'eos_token_id', 2)
            if next_token[0, 0].item() == eos_id:
                break

        return generated

    def _sample_token(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float
    ) -> torch.Tensor:
        """Sample next token with various strategies."""
        # Apply repetition penalty - O(vocabulary) vectorized operation
        if repetition_penalty != 1.0:
            vocab_size = logits.shape[-1]
            # Clamp token IDs to valid vocab range and count occurrences
            token_ids = generated[0].clamp(0, vocab_size - 1)
            token_counts = torch.bincount(token_ids, minlength=vocab_size)
            # Create mask for tokens that have appeared
            appeared_mask = token_counts > 0
            # Apply penalty vectorized (much faster than loop)
            logits[0, appeared_mask] = logits[0, appeared_mask] / repetition_penalty

        # Temperature scaling
        logits = logits / max(temperature, 1e-8)

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            min_value = values[:, -1, None]
            logits = torch.where(logits < min_value, float('-inf'), logits)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # =========================================================================
    # Streaming Generation
    # =========================================================================

    def stream_generate(
        self,
        prompt: str,
        max_gen: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        max_tokens: int | None = None,  # Alias for max_gen (backward compatibility)
        max_new_tokens: int | None = None,  # Alias for max_gen (Forge model compatibility)
        max_length: int | None = None  # Alias for max_gen (common parameter name)
    ) -> Generator[str]:
        """
        Stream generated tokens one at a time.

        Args:
            prompt: Input text to continue
            max_gen: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            max_tokens: Alias for max_gen (backward compatibility)
            max_new_tokens: Alias for max_gen (Forge model compatibility)
            max_length: Alias for max_gen (common parameter name)

        Yields:
            Each newly generated token as it's produced
        """
        # Handle max_tokens, max_new_tokens, max_length aliases for backward compatibility
        if max_tokens is not None:
            max_gen = max_tokens
        if max_new_tokens is not None:
            max_gen = max_new_tokens
        if max_length is not None:
            max_gen = max_length
        
        input_ids = self._encode_prompt(prompt)
        generated = input_ids

        with torch.no_grad():
            for _ in range(max_gen):
                # Truncate if needed
                curr_input = generated
                max_len = self.model.config.max_seq_len
                if curr_input.shape[1] > max_len:
                    curr_input = curr_input[:, -max_len:]

                # Forward pass
                logits = self.model(curr_input)

                # Sample
                next_token = self._sample_token(
                    logits[:, -1, :],
                    generated,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty
                )

                generated = torch.cat([generated, next_token], dim=1)

                # Decode and yield
                token_id = next_token[0, 0].item()

                if hasattr(self.tokenizer, 'decode'):
                    token_str = self.tokenizer.decode([token_id], skip_special_tokens=True)
                else:
                    token_str = self.tokenizer.id_to_token.get(token_id, "")

                yield token_str

                # Check for EOS
                eos_id = getattr(self.tokenizer, 'eos_token_id', 2)
                if token_id == eos_id:
                    break

    # =========================================================================
    # Batch Generation
    # =========================================================================

    def batch_generate(
        self,
        prompts: list[str],
        max_gen: int = 100,
        **kwargs
    ) -> list[str]:
        """
        Generate text for multiple prompts in a single batched forward pass.

        Args:
            prompts: List of input prompts
            max_gen: Maximum tokens to generate per prompt
            **kwargs: Additional generation parameters (temperature, top_k, top_p, repetition_penalty)

        Returns:
            List of generated texts
        """
        if not prompts:
            return []
        
        # If only one prompt, use regular generate
        if len(prompts) == 1:
            return [self.generate(prompts[0], max_gen=max_gen, **kwargs)]
        
        # Extract generation parameters
        temperature = kwargs.get('temperature', 0.8)
        top_k = kwargs.get('top_k', 50)
        top_p = kwargs.get('top_p', 0.9)
        repetition_penalty = kwargs.get('repetition_penalty', 1.1)
        
        # Encode all prompts
        if hasattr(self.tokenizer, 'encode'):
            encoded = [self.tokenizer.encode(p) for p in prompts]
        else:
            encoded = [[self.tokenizer.token_to_id.get(t, 3) for t in p] for p in prompts]
        
        # Pad all sequences to the same length
        max_input_len = max(len(e) for e in encoded)
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
        
        # Create padded batch tensor
        batch_size = len(prompts)
        input_ids = torch.full(
            (batch_size, max_input_len),
            pad_id,
            dtype=torch.long,
            device=self.device
        )
        
        # Fill in the actual tokens
        for i, tokens in enumerate(encoded):
            input_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        
        # Track which sequences are still generating
        eos_id = getattr(self.tokenizer, 'eos_token_id', 2)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Generate tokens autoregressively
        generated = input_ids
        all_finished = False
        for step in range(max_gen):
            # Early exit if all sequences finished (check every 5 steps starting from step 5 to reduce overhead)
            if all_finished or (step >= 5 and step % 5 == 0 and finished.all()):
                all_finished = True
                break
            
            # Truncate if needed
            curr_input = generated
            max_len = self.model.config.max_seq_len
            if curr_input.shape[1] > max_len:
                curr_input = curr_input[:, -max_len:]
            
            # Forward pass for entire batch
            with torch.no_grad():
                logits = self.model(curr_input)
            
            # Sample next token for each sequence
            next_tokens = []
            for i in range(batch_size):
                if finished[i]:
                    # Already finished, use pad token
                    next_tokens.append(pad_id)
                else:
                    # Sample from logits
                    token = self._sample_token(
                        logits[i:i+1, -1:, :],
                        generated[i:i+1],
                        temperature,
                        top_k,
                        top_p,
                        repetition_penalty
                    )
                    next_tokens.append(token.item())
                    
                    # Check for EOS
                    if token.item() == eos_id:
                        finished[i] = True
            
            # Append next tokens to generated
            next_tokens_tensor = torch.tensor(
                [[t] for t in next_tokens],
                dtype=torch.long,
                device=self.device
            )
            generated = torch.cat([generated, next_tokens_tensor], dim=1)
        
        # Decode all outputs
        results = []
        for i in range(batch_size):
            # Handle case where generated[i] might not be a tensor
            try:
                ids = generated[i].cpu().tolist()
            except AttributeError:
                if isinstance(generated[i], str):
                    results.append(generated[i])
                    continue
                ids = list(generated[i]) if hasattr(generated[i], '__iter__') else [generated[i]]
            
            if hasattr(self.tokenizer, 'decode'):
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
            else:
                # Fallback
                text = "".join(
                    self.tokenizer.id_to_token.get(idx, "?")
                    for idx in ids
                )
            
            results.append(text)
        
        return results

    # =========================================================================
    # Chat Interface
    # =========================================================================
    
    def clear_kv_cache(self) -> None:
        """
        Clear the KV-cache to prevent hallucinations from stale context.
        
        Call this when:
        - Starting a new conversation
        - After many messages (context gets confused)
        - When AI starts hallucinating
        """
        if hasattr(self.model, 'clear_kv_cache'):
            self.model.clear_kv_cache()
            logger.debug("Cleared model KV-cache")
        elif hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()
            logger.debug("Reset model cache")
        elif hasattr(self.model, 'kv_cache'):
            self.model.kv_cache = None
            logger.debug("Set kv_cache to None")
        # Also clear any internal cache
        if hasattr(self, '_cache'):
            self._cache = None
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Text to count tokens in
            
        Returns:
            Number of tokens
        """
        if hasattr(self.tokenizer, 'encode'):
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        elif hasattr(self.tokenizer, '__call__'):
            result = self.tokenizer(text, return_tensors=None)
            return len(result.get('input_ids', []))
        else:
            # Rough estimate: ~4 chars per token
            return len(text) // 4
    
    def get_max_context_length(self) -> int:
        """Get the model's maximum context length."""
        if hasattr(self.model, 'config'):
            return getattr(self.model.config, 'max_seq_len', 1024)
        return 1024  # Safe default
    
    def _truncate_history(
        self,
        history: list[dict[str, str]],
        current_message: str,
        system_prompt: str | None = None,
        max_history_tokens: int | None = None,
        reserve_for_response: int = 200
    ) -> list[dict[str, str]]:
        """
        Truncate conversation history to fit within context window.
        
        This prevents hallucinations caused by context overflow!
        
        Args:
            history: Full conversation history
            current_message: Current user message
            system_prompt: Optional system prompt
            max_history_tokens: Max tokens for history (auto-calculated if None)
            reserve_for_response: Tokens to reserve for AI response
            
        Returns:
            Truncated history that fits in context window
        """
        if not history:
            return []
        
        # Calculate available space
        max_context = self.get_max_context_length()
        
        # Reserve space for: system prompt + current message + response
        reserved = reserve_for_response
        if system_prompt:
            reserved += self.count_tokens(f"System: {system_prompt}\n")
        reserved += self.count_tokens(f"User: {current_message}\nAssistant:")
        
        max_history_tokens = max_history_tokens or (max_context - reserved)
        
        # If very limited context, keep only last exchange
        if max_history_tokens < 100:
            logger.warning(f"Very limited context ({max_context} tokens), keeping only last exchange")
            return history[-2:] if len(history) >= 2 else history[-1:]
        
        # Build history from most recent, counting tokens
        truncated = []
        total_tokens = 0
        
        for msg in reversed(history):
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            msg_text = f"{role}: {content}\n"
            msg_tokens = self.count_tokens(msg_text)
            
            if total_tokens + msg_tokens > max_history_tokens:
                # Don't add this message, we're at limit
                break
            
            truncated.insert(0, msg)
            total_tokens += msg_tokens
        
        if len(truncated) < len(history):
            logger.info(f"Truncated history: {len(history)} -> {len(truncated)} messages ({total_tokens} tokens)")
        
        return truncated

    def chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        max_gen: int = 200,
        auto_truncate: bool = True,
        **kwargs
    ) -> str:
        """Chat-style generation with conversation history.

        Builds a structured prompt from the conversation history and the
        current user message, runs it through ``generate()``, and extracts
        only the assistant's reply.

        When ``auto_truncate`` is enabled (default), long conversation
        histories are automatically trimmed so they fit inside the model's
        context window.  This prevents the model from receiving a prompt
        that exceeds ``max_seq_len`` -- a common cause of hallucinations
        and garbled output.

        Args:
            message: The user's current message.
            history: Previous turns as a list of dicts, each with
                ``"role"`` (``"user"`` or ``"assistant"``) and
                ``"content"`` keys.  ``None`` starts a fresh
                conversation.
            system_prompt: An optional system instruction prepended to the
                prompt (e.g. ``"You are a helpful coding assistant."``).
            max_gen: Maximum number of new tokens to generate for the
                assistant reply.
            auto_truncate: If ``True``, older history entries are dropped
                when the prompt would exceed the model's context window.
            **kwargs: Extra keyword arguments forwarded to ``generate()``
                (e.g. ``temperature``, ``top_k``, ``top_p``).

        Returns:
            The assistant's response text (without prompt or history).

        Raises:
            RuntimeError: If the underlying model is not loaded or the
                tokenizer fails to encode the prompt.

        Example:
            >>> engine = EnigmaEngine()
            >>> reply = engine.chat("What is Python?")
            >>> print(reply)
            'Python is a high-level programming language...'
            >>>
            >>> # Multi-turn with history
            >>> history = [
            ...     {"role": "user", "content": "Hi!"},
            ...     {"role": "assistant", "content": "Hello! How can I help?"},
            ... ]
            >>> reply = engine.chat("Tell me a joke", history=history)
        """
        # ─────────────────────────────────────────────────────────────────────
        # TRUNCATE HISTORY TO PREVENT HALLUCINATIONS
        # This is critical! Without this, long conversations overflow the
        # context window and cause the model to hallucinate.
        # ─────────────────────────────────────────────────────────────────────
        if auto_truncate and history:
            history = self._truncate_history(
                history,
                current_message=message,
                system_prompt=system_prompt,
                reserve_for_response=max_gen
            )
        
        # Use centralized prompt builder for consistent formatting
        try:
            from .prompt_builder import get_prompt_builder
            builder = get_prompt_builder()
            full_prompt = builder.build_chat_prompt(
                message=message,
                history=history,
                system_prompt=system_prompt,
                include_generation_prefix=True
            )
            stop_strings = builder.get_stop_sequences()
        except ImportError:
            # Fallback to inline prompt building
            prompt_parts = []

            if system_prompt:
                prompt_parts.append(f"System: {system_prompt}\n")

            if history:
                for msg in history:
                    role = msg.get("role", "user").capitalize()
                    content = msg.get("content", "")
                    prompt_parts.append(f"{role}: {content}")

            prompt_parts.append(f"User: {message}")
            prompt_parts.append("Assistant:")

            full_prompt = "\n".join(prompt_parts)
            stop_strings = ["\nUser:", "\n\n", "User:"]

        # Generate
        response = self.generate(
            full_prompt,
            max_gen=max_gen,
            stop_strings=stop_strings,
            **kwargs
        )

        # Extract assistant's response
        try:
            from .prompt_builder import extract_response
            response = extract_response(response, full_prompt)
        except ImportError:
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()

        return response

    def chat_with_tools(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        max_gen: int = 200,
        fallback_to_chat: bool = True,
        **kwargs
    ) -> str:
        """
        Chat with automatic tool routing based on user intent.
        
        Uses the UniversalToolRouter which detects tool intent from
        keywords in the user message. Works regardless of whether the
        model was trained to use tools.
        
        Args:
            message: User's message
            history: Conversation history
            system_prompt: Optional system prompt
            max_gen: Maximum tokens to generate
            fallback_to_chat: If tool fails, use chat instead
            **kwargs: Additional generation parameters
            
        Returns:
            Response (either from tool execution or chat)
        """
        from .universal_router import chat_with_tools as universal_chat

        # Create a chat function that preserves history/system prompt
        def chat_fn(msg, **kw):
            return self.chat(
                msg, 
                history=history, 
                system_prompt=system_prompt,
                max_gen=max_gen, 
                **kwargs
            )
        
        return universal_chat(
            message, 
            chat_fn, 
            fallback_to_chat=fallback_to_chat
        )

    def stream_chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        max_gen: int = 200,
        **kwargs
    ) -> Generator[str]:
        """
        Stream chat-style generation.

        Args:
            message: User's message
            history: Conversation history
            system_prompt: Optional system prompt
            max_gen: Maximum tokens
            **kwargs: Additional parameters

        Yields:
            Generated tokens one at a time
        """
        # Use centralized prompt builder for consistent formatting
        try:
            from .prompt_builder import get_prompt_builder
            builder = get_prompt_builder()
            full_prompt = builder.build_chat_prompt(
                message=message,
                history=history,
                system_prompt=system_prompt,
                include_generation_prefix=True
            )
            stop_strings = builder.get_stop_sequences()
        except ImportError:
            # Fallback to inline prompt building
            prompt_parts = []

            if system_prompt:
                prompt_parts.append(f"System: {system_prompt}\n")

            if history:
                for msg in history:
                    role = msg.get("role", "user").capitalize()
                    content = msg.get("content", "")
                    prompt_parts.append(f"{role}: {content}")

            prompt_parts.append(f"User: {message}")
            prompt_parts.append("Assistant:")

            full_prompt = "\n".join(prompt_parts)
            stop_strings = ["\nUser:", "\n\n"]

        # Stream generation
        buffer = ""
        for token in self.stream_generate(full_prompt, max_gen=max_gen, **kwargs):
            buffer += token

            # Check for stop conditions
            stopped = False
            for stop in stop_strings:
                if stop in buffer:
                    buffer = buffer[:buffer.find(stop)]
                    stopped = True
                    break
            
            if stopped:
                break

            yield token


    # =========================================================================
    # Tool-Aware Generation
    # =========================================================================
    
    def generate_with_tools(
        self,
        prompt: str,
        module_manager=None,
        max_gen: int = 200,
        max_tool_iterations: int = 5,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        include_system_prompt: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text with tool execution support.
        
        The AI can invoke tools during generation, and the results are fed back
        for continued generation. This enables the AI to:
          - Generate images when asked
          - Control avatar expressions
          - Search the web for information
          - Read/write files
          - And more
        
        Args:
            prompt: Input text or user query
            module_manager: ModuleManager instance for tool access
            max_gen: Maximum tokens per generation step
            max_tool_iterations: Maximum number of tool calls in sequence
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            include_system_prompt: Prepend tool usage instructions
            **kwargs: Additional generation parameters
            
        Returns:
            Complete generated text with tool results
        """
        from .tool_interface import ToolInterface
        from .tool_prompts import get_tool_enabled_system_prompt

        # Create tool interface
        tool_interface = ToolInterface(module_manager)
        
        # Prepend system prompt if requested
        if include_system_prompt:
            system_prompt = get_tool_enabled_system_prompt()
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = prompt
        
        # Generate with tool support
        current_prompt = full_prompt
        full_output = ""
        iterations = 0
        
        while iterations < max_tool_iterations:
            # Generate next chunk
            output = self.generate(
                current_prompt,
                max_gen=max_gen,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_cache=True
            )
            
            # Extract new content (remove the prompt if it's in the output)
            if current_prompt in output:
                new_content = output[len(current_prompt):]
            else:
                new_content = output
            
            # Check for tool calls in the new content
            tool_call = tool_interface.parse_tool_call(new_content)
            
            if tool_call:
                # Execute the tool
                result = tool_interface.execute_tool(tool_call)
                result_str = tool_interface.format_tool_result(result)
                
                # Append tool call and result to output
                full_output += new_content[:tool_call.end_pos - tool_call.start_pos]
                full_output += "\n" + result_str + "\n"
                
                # Update prompt for next iteration
                current_prompt = full_prompt + full_output
                iterations += 1
                
                # Continue generation after tool result
                continue
            else:
                # No tool call found, we're done
                full_output += new_content
                break
        
        return full_output
    
    def stream_generate_with_tools(
        self,
        prompt: str,
        module_manager=None,
        max_gen: int = 200,
        max_tool_iterations: int = 5,
        include_system_prompt: bool = True,
        **kwargs
    ) -> Generator[str]:
        """
        Stream generation with tool execution support.
        
        Yields tokens as they're generated, pausing for tool execution
        when tool calls are detected.
        
        Args:
            prompt: Input text
            module_manager: ModuleManager for tool access
            max_gen: Maximum tokens per step
            max_tool_iterations: Maximum tool calls
            include_system_prompt: Include tool instructions
            **kwargs: Additional parameters
            
        Yields:
            Generated tokens, including tool results
        """
        from .tool_interface import ToolInterface
        from .tool_prompts import get_tool_enabled_system_prompt
        
        tool_interface = ToolInterface(module_manager)
        
        if include_system_prompt:
            system_prompt = get_tool_enabled_system_prompt()
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = prompt
        
        current_prompt = full_prompt
        buffer = ""
        iterations = 0
        
        while iterations < max_tool_iterations:
            # Stream generate
            for token in self.stream_generate(current_prompt, max_gen=max_gen, **kwargs):
                buffer += token
                yield token
                
                # Check if we have a complete tool call
                if '<|tool_end|>' in buffer:
                    tool_call = tool_interface.parse_tool_call(buffer)
                    if tool_call:
                        # Execute tool
                        result = tool_interface.execute_tool(tool_call)
                        result_str = tool_interface.format_tool_result(result)
                        
                        # Yield result
                        yield "\n" + result_str + "\n"
                        
                        # Update prompt
                        current_prompt = full_prompt + buffer + "\n" + result_str + "\n"
                        buffer = ""
                        iterations += 1
                        break
            else:
                # Generation completed without tool call
                break


# =============================================================================
# Convenience Functions
# =============================================================================

def generate(
    prompt: str,
    model_path: str | None = None,
    max_gen: int = 100,
    **kwargs
) -> str:
    """
    Quick generation function.

    Args:
        prompt: Input text
        model_path: Optional model path
        max_gen: Maximum tokens
        **kwargs: Additional parameters

    Returns:
        Generated text
    """
    engine = EnigmaEngine(model_path=model_path)
    return engine.generate(prompt, max_gen=max_gen, **kwargs)


def load_engine(
    model_path: str | None = None,
    device: str | None = None
) -> EnigmaEngine:
    """
    Load an inference engine.

    Args:
        model_path: Path to model
        device: Device to use

    Returns:
        EnigmaEngine instance
    """
    return EnigmaEngine(model_path=model_path, device=device)


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

# Keep ForgeEngine as an alias for existing code
ForgeEngine = EnigmaEngine


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "EnigmaEngine",
    "ForgeEngine",  # Backward compatibility alias
    "generate",
    "load_engine",
]
