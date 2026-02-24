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
    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
    â”‚  YOU: "What is the meaning of life?"                        â”‚
    â”‚   â”‚                                                         â”‚
    â”‚   â†“  (EnigmaEngine encodes your words into numbers)         â”‚
    â”‚  [15496, 318, 262, 3616, ...]                               â”‚
    â”‚   â”‚                                                         â”‚
    â”‚   â†“  (Sends numbers through the Enigma brain)               â”‚
    â”‚  [Matrix multiplication magic x millions]                   â”‚
    â”‚   â”‚                                                         â”‚
    â”‚   â†“  (Gets probability for each possible next word)         â”‚
    â”‚  "The" 0.3, "It" 0.2, "42" 0.15, ...                       â”‚
    â”‚   â”‚                                                         â”‚
    â”‚   â†“  (Picks one, repeats until done)                        â”‚
    â”‚  AI: "The meaning of life is to find purpose..."           â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

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
    You came from â†’ model.py (Chapter 1: The Brain)
    You can go to â†’ tool_router.py (Chapter 3: The Dispatcher)
                  â†’ chat_tab.py (The GUI interface)
                  â†’ api_server.py (REST API for remote access)
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
from .engine_chat import _ChatMixin
from .engine_generation import _GenerationMixin
from .model import MODEL_PRESETS, Forge, create_model
from .tokenizer import get_tokenizer

# Simple message functions (previously from utils.system_messages)
def info_msg(msg: str) -> str:
    return f"[INFO] {msg}"

def system_msg(msg: str) -> str:
    return f"[SYSTEM] {msg}"

logger = logging.getLogger(__name__)

# Default model paths
MODELS_DIR = Path(CONFIG.get("models_dir", "models"))
DEFAULT_MODEL = MODELS_DIR / "forge.pth"
LEGACY_MODEL = MODELS_DIR / "tiny_enigma_engine.pth"


# =============================================================================
# âš¡ INFERENCE ENGINE - Talk to Your AI!
# =============================================================================
# This is the main class for generating text with a trained model.
# It handles all the complexity of:
#   - Loading models and tokenizers
#   - Running the neural network
#   - Sampling strategies (how to pick the next word)
#   - KV-cache for fast generation
#   - Tool routing for specialized tasks

class EnigmaEngine(_GenerationMixin, _ChatMixin):
    """
    High-performance inference engine for Enigma models.
    
    ðŸ“– WHAT THIS DOES:
    Takes your text prompt and generates a response using the AI model.
    
    ðŸ“ GENERATION LOOP:
    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
    â”‚  "Hello, how are" â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€    â”‚
    â”‚         â”‚                                                              â”‚
    â”‚         â–¼                                                              â”‚
    â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                                                       â”‚
    â”‚  â”‚ Tokenizer   â”‚ â†’ [15496, 11, 703, 389]                              â”‚
    â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                                                       â”‚
    â”‚         â”‚                                                              â”‚
    â”‚         â–¼                                                              â”‚
    â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”‚
    â”‚  â”‚   Model     â”‚ â”€â”€â–¶ â”‚ Probabilities for ALL vocab tokens        â”‚   â”‚
    â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â”‚ "you": 0.15, "doing": 0.08, "the": 0.02   â”‚   â”‚
    â”‚         â”‚            â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â”‚
    â”‚         â–¼                                                              â”‚
    â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                                                       â”‚
    â”‚  â”‚  Sampler    â”‚ â†’ Pick "you" (based on temperature, top_k, etc.)    â”‚
    â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                                                       â”‚
    â”‚         â”‚                                                              â”‚
    â”‚         â–¼                                                              â”‚
    â”‚  Add "you" to sequence, REPEAT until done                             â”‚
    â”‚         â”‚                                                              â”‚
    â”‚         â–¼                                                              â”‚
    â”‚  "Hello, how are you doing today?"                                    â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    
    âš¡ KEY FEATURES:
    - KV-cache: Don't recompute past tokens (10x faster!)
    - Multiple samplers: greedy, top-k, top-p, temperature
    - Streaming: Get tokens as they're generated
    - Tools: Route to specialized models/APIs
    - Chat: Maintains conversation history
    
    ðŸŽ›ï¸ SAMPLING STRATEGIES:
    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
    â”‚ GREEDY (temperature=0):                                                â”‚
    â”‚   Always pick highest probability token                                â”‚
    â”‚   Pro: Deterministic, consistent                                       â”‚
    â”‚   Con: Repetitive, boring                                              â”‚
    â”‚                                                                        â”‚
    â”‚ TEMPERATURE (0.1 to 2.0):                                              â”‚
    â”‚   Scales probabilities before sampling                                 â”‚
    â”‚   Low (0.3): More focused, predictable                                â”‚
    â”‚   High (1.5): More random, creative                                   â”‚
    â”‚                                                                        â”‚
    â”‚ TOP-K (e.g., k=50):                                                    â”‚
    â”‚   Only consider top K most likely tokens                              â”‚
    â”‚   Prevents sampling very unlikely tokens                              â”‚
    â”‚                                                                        â”‚
    â”‚ TOP-P / NUCLEUS (e.g., p=0.9):                                        â”‚
    â”‚   Only consider tokens covering P% of probability mass                â”‚
    â”‚   Dynamic cutoff based on confidence                                  â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    
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
        
        ðŸ“– USE THIS WHEN:
        You already have a loaded model and tokenizer, and don't want
        the engine to load them again from disk.
        
        ðŸ“ EXAMPLE:
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
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # INITIALIZE REQUIRED ATTRIBUTES
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        engine._is_gguf = False
        engine._web_enabled = False
        engine.model_metadata = {
            "supports_nsfw": False,
            "content_rating": "sfw",
            "trained_date": None,
            "training_tasks": [],
        }
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # SET MODEL AND TOKENIZER DIRECTLY
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        
        ðŸ“– THIS IS THE MAIN CONSTRUCTOR!
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
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # THREAD SAFETY: Lock for KV-cache operations
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # The KV-cache is stateful - only one generation can run at a time
        self._generation_lock = threading.Lock()
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # DEVICE SELECTION: Pick the best available hardware
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.device = self._select_device(device)
        self.use_half = use_half and self.device.type == "cuda"
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # STORE CONFIGURATION
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.enable_tools = enable_tools
        self.module_manager = module_manager
        self.use_routing = use_routing
        
        # Check if CPU/GPU offloading is enabled in config
        self.use_offloading = CONFIG.get("enable_offloading", False)
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # TOOL SYSTEM SETUP (optional)
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._tool_executor = None
        if enable_tools:
            from ..tools.tool_executor import ToolExecutor
            self._tool_executor = ToolExecutor(module_manager=module_manager)
        
        # Tool router for specialized models (vision, code, etc.)
        # Note: tool_router module not yet implemented
        self._tool_router = None

        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # LOAD TOKENIZER
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.tokenizer = self._load_tokenizer(tokenizer_path, model_path)

        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # LOAD MODEL
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.model = self._load_model(model_path, model_size)

        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # APPLY DEVICE PLACEMENT (PyTorch models only)
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # GGUF models handle their own device placement via n_gpu_layers
        if not getattr(self, '_is_gguf', False):
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

        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # LOAD MODEL METADATA (including content rating support)
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._load_model_metadata(model_path)

        # Log what we loaded
        self._log_init_info()

    def _select_device(self, device: str | None) -> torch.device:
        """Select the best available device."""
        if device is not None:
            return torch.device(device)

        # Check power mode settings (device selection)
        # Power mode not yet implemented - use standard device detection
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
        
        ðŸ“– AUTO-DETECTION:
        If model_size="auto", this method will:
        1. Detect hardware capabilities (RAM, GPU, Pi)
        2. Choose the best model size for this device
        3. Apply quantization if memory is tight
        
        This enables seamless deployment from Raspberry Pi to datacenter!
        """
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # AUTO-DETECT MODEL SIZE FOR HARDWARE
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # FIND MODEL FILE
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # GGUF MODEL HANDLING (llama.cpp)
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if model_file and model_file.suffix.lower() == '.gguf':
            logger.info(f"Detected GGUF model: {model_file}")
            try:
                from .gguf_loader import GGUFModel
                
                # Auto-detect GPU layers based on available VRAM
                n_gpu_layers = 0
                n_ctx = 8192  # Default context size
                try:
                    import torch
                    if torch.cuda.is_available():
                        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        # Rough estimate: more VRAM = more layers on GPU
                        if vram_gb >= 24:
                            n_gpu_layers = 100  # Full GPU offload
                            n_ctx = 16384  # Larger context with plenty of VRAM
                        elif vram_gb >= 16:
                            n_gpu_layers = 50
                            n_ctx = 8192
                        elif vram_gb >= 8:
                            n_gpu_layers = 30
                            n_ctx = 4096
                        elif vram_gb >= 4:
                            n_gpu_layers = 15
                            n_ctx = 2048
                        logger.info(f"Auto-detected {vram_gb:.1f}GB VRAM, using {n_gpu_layers} GPU layers, {n_ctx} context")
                except Exception:
                    pass
                
                model = GGUFModel(
                    str(model_file),
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False
                )
                model.load()
                self._is_gguf = True
                return model
            except ImportError as e:
                raise RuntimeError(
                    f"GGUF model detected but llama-cpp-python not installed.\n"
                    f"Install with: pip install llama-cpp-python\n"
                    f"Error: {e}"
                ) from e
            except Exception as e:
                raise RuntimeError(f"Failed to load GGUF model: {e}") from e

        vocab_size = getattr(self.tokenizer, "vocab_size", 8000)

        if model_file and model_file.exists():
            # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # TRY MEMORY-MAPPED LOADING FOR LARGE MODELS / LOW MEMORY
            # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

            # Infer model config from state dict
            inferred_config = self._infer_model_config(state_dict)
            detected_size = self._infer_model_size(state_dict)
            
            vocab_size = inferred_config.get('vocab_size', 8000)
            max_seq_len = inferred_config.get('max_seq_len', 1024)
            n_layers = inferred_config.get('n_layers')
            n_heads = inferred_config.get('n_heads')
            n_kv_heads = inferred_config.get('n_kv_heads')

            # Create model with correct architecture
            try:
                kwargs = {'vocab_size': vocab_size, 'max_seq_len': max_seq_len}
                if n_layers:
                    kwargs['n_layers'] = n_layers
                if n_heads:
                    kwargs['n_heads'] = n_heads
                if n_kv_heads:
                    kwargs['n_kv_heads'] = n_kv_heads
                model = create_model(detected_size, **kwargs)
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
            
            # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # APPLY AUTO-QUANTIZATION IF NEEDED
            # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
                f"  2. Download a HuggingFace model: model.download <repo_id>\n"
                f"  3. Or specify model_path when creating EnigmaEngine"
            )

        return model

    def _infer_model_config(self, state_dict: dict) -> dict:
        """Infer full model config from state dict weights."""
        config = {}
        
        # Get vocab_size and dim from embedding
        for key, tensor in state_dict.items():
            if ('embed' in key.lower() or 'token' in key.lower()) and tensor.dim() == 2:
                config['vocab_size'] = tensor.shape[0]
                config['dim'] = tensor.shape[1]
                break
        
        # Fallback dim from norm weights
        if 'dim' not in config:
            for key, tensor in state_dict.items():
                if ('norm' in key.lower()) and tensor.dim() == 1:
                    config['dim'] = tensor.shape[0]
                    break
        
        # Get max_seq_len and head_dim from freqs_cis
        # freqs_cis shape = [max_seq_len*2, head_dim/2]
        if 'freqs_cis' in state_dict:
            freqs_shape = state_dict['freqs_cis'].shape
            config['max_seq_len'] = freqs_shape[0] // 2
            config['head_dim'] = freqs_shape[1] * 2  # freqs_cis stores head_dim/2
        
        # Count layers
        layer_nums = set()
        for key in state_dict.keys():
            if 'layers.' in key:
                parts = key.split('.')
                for i, p in enumerate(parts):
                    if p == 'layers' and i + 1 < len(parts):
                        try:
                            layer_nums.add(int(parts[i + 1]))
                        except ValueError:
                            pass
        if layer_nums:
            config['n_layers'] = max(layer_nums) + 1
        
        # Compute n_heads from dim and head_dim
        dim = config.get('dim', 512)
        head_dim = config.get('head_dim', 64)
        config['n_heads'] = dim // head_dim
        
        # Infer n_kv_heads from wk weight shape
        for key, tensor in state_dict.items():
            if 'attention.wk.weight' in key and tensor.dim() == 2:
                kv_dim = tensor.shape[0]
                config['n_kv_heads'] = kv_dim // head_dim
                break
        
        return config

    def _infer_model_size(self, state_dict: dict) -> str:
        """Infer model size preset name from state dict."""
        config = self._infer_model_config(state_dict)
        hidden_dim = config.get('dim', 512)

        # Match to preset
        for name, preset in MODEL_PRESETS.items():
            if getattr(preset, 'dim', None) == hidden_dim:
                return name

        # Find closest match
        diffs = [(name, abs(getattr(preset, 'dim', 512) - hidden_dim))
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
                
        except Exception as e:
            logger.debug(f"Could not load model metadata: {e}")

    def _log_init_info(self) -> None:
        """Log initialization information."""
        # GGUF models don't expose parameters() like PyTorch models
        if getattr(self, '_is_gguf', False):
            logger.info("EnigmaEngine initialized with GGUF model")
            if hasattr(self.model, 'model_path'):
                logger.info(f"Model: {self.model.model_path}")
            return
            
        num_params = sum(p.numel() for p in self.model.parameters())

        logger.info(f"EnigmaEngine initialized on {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Model parameters: {num_params:,}")
        logger.info(f"Vocab size: {self.tokenizer.vocab_size:,}")
        logger.info(f"Max sequence length: {self.model.config.max_seq_len}")
        logger.info(f"FP16: {self.use_half}")

    # =========================================================================
    # ðŸ“ GENERATION METHODS - The Heart of Text Generation
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
        
        ðŸ“– WHAT THIS DOES:
        This is the main generation function. Give it text, get more text back!
        
        ðŸ“ HOW IT WORKS:
        1. Check if prompt needs special routing (image/code/web)
        2. Acquire thread lock (only one generation at a time)
        3. Tokenize the prompt into numbers
        4. Feed tokens to model, get probability distribution
        5. Sample next token using temperature/top-k/top-p
        6. Repeat until max_gen tokens or stop_string found
        7. If AI tried to use tools, execute them and continue
        
        ðŸ“ PARAMETER GUIDE:
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚ temperature:  Controls randomness                              â”‚
        â”‚   0.1-0.3:   Very focused, predictable                        â”‚
        â”‚   0.7-0.9:   Good balance (default area)                      â”‚
        â”‚   1.0-1.5:   More creative, less coherent                     â”‚
        â”‚   >1.5:      Very random, may be nonsense                     â”‚
        â”‚                                                                â”‚
        â”‚ top_k:       Only consider top K tokens                       â”‚
        â”‚   10-30:     Very focused                                      â”‚
        â”‚   50:        Good default                                      â”‚
        â”‚   100+:      More variety                                      â”‚
        â”‚                                                                â”‚
        â”‚ top_p:       Nucleus sampling - dynamic cutoff                â”‚
        â”‚   0.5:       Conservative, focused                            â”‚
        â”‚   0.9:       Good default                                      â”‚
        â”‚   0.95-1.0:  More variety                                      â”‚
        â”‚                                                                â”‚
        â”‚ repetition_penalty: Discourage repeating words               â”‚
        â”‚   1.0:       No penalty                                        â”‚
        â”‚   1.1:       Mild (good default)                              â”‚
        â”‚   1.3+:      Strong (may break grammar)                       â”‚
        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

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
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # STEP 1: Determine if tools should be executed
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if execute_tools is None:
            execute_tools = self.enable_tools
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # STEP 2: Check if specialized routing should handle this
        # Some prompts can bypass the main AI for faster execution
        # e.g., "draw a cat" â†’ directly calls image generator
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.use_routing and self._tool_router:
            # Classify what the user wants (image, code, web, etc.)
            intent = self._tool_router.classify_intent(prompt)
            logger.info(f"Classified intent: {intent}")
            
            # Check if this needs AI creativity (ambiguous/creative requests)
            # "surprise me" â†’ needs AI, "draw a cat" â†’ can route directly
            if self._needs_ai_creativity(prompt):
                logger.info("Prompt requires AI creativity, using main AI")
                # Fall through to standard generation
            else:
                # Try direct routing for speed
                direct_result = self._try_direct_routing(intent, prompt)
                if direct_result is not None:
                    return direct_result
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # STEP 3: Thread-safe generation (protects KV-cache state)
        # Only one generation can happen at a time!
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        lock = getattr(self, '_generation_lock', None)
        if lock:
            lock.acquire()
        
        try:
            # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # STEP 4: Standard text generation
            # This is where the actual model inference happens
            # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            text = self._generate_text(
                prompt, max_gen, temperature, top_k, top_p, 
                repetition_penalty, stop_strings, use_cache
            )
            
            # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # STEP 5: Tool execution loop
            # If AI generated tool calls, execute them and continue
            # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # =========================================================================
    # Encoding / Decoding Helpers
    # =========================================================================

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

    # =========================================================================
    # Cache & Token Utilities
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
