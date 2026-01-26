"""
================================================================================
📦 MODULE REGISTRY - THE CATALOG OF ALL MODULES
================================================================================

Central registry where ALL available modules are defined! This is the "catalog"
that tells ModuleManager what modules exist and how to load them.

📍 FILE: forge_ai/modules/registry.py
🏷️ TYPE: Module Definitions & Registration
🎯 MAIN CLASSES: ModelModule, TokenizerModule, ImageGenLocalModule, etc.

┌─────────────────────────────────────────────────────────────────────────────┐
│  AVAILABLE MODULES:                                                         │
│                                                                             │
│  🧠 CORE:                                                                    │
│     • ModelModule       - Forge transformer (forge_ai/core/model.py)       │
│     • TokenizerModule   - Text tokenizer (forge_ai/core/tokenizer.py)      │
│     • InferenceModule   - Text generation (forge_ai/core/inference.py)     │
│                                                                             │
│  🎨 GENERATION (LOCAL vs API - pick one!):                                  │
│     • ImageGenLocalModule  / ImageGenAPIModule                              │
│     • CodeGenLocalModule   / CodeGenAPIModule                               │
│     • VideoGenLocalModule  / VideoGenAPIModule                              │
│     • AudioGenLocalModule  / AudioGenAPIModule                              │
│     • ThreeDGenLocalModule / ThreeDGenAPIModule                             │
│                                                                             │
│  💾 MEMORY:                                                                  │
│     • MemoryModule      - Conversation storage                              │
│     • EmbeddingModule   - Vector embeddings                                 │
│                                                                             │
│  🌐 NETWORK:                                                                 │
│     • APIServerModule   - REST API                                          │
│     • NetworkModule     - Multi-device                                      │
└─────────────────────────────────────────────────────────────────────────────┘

⚠️ MODULE CONFLICTS:
    ✗ image_gen_local + image_gen_api   (same capability)
    ✗ code_gen_local + code_gen_api     (same capability)
    ✗ video_gen_local + video_gen_api   (same capability)

🔗 CONNECTED FILES:
    → USES:      forge_ai/modules/manager.py (Module base class)
    → WRAPS:     forge_ai/core/*.py (core modules)
    → WRAPS:     forge_ai/gui/tabs/*.py (generation tabs)
    ← USED BY:   forge_ai/modules/manager.py (discovers modules)

📖 SEE ALSO:
    • forge_ai/modules/manager.py       - Loads/unloads modules
    • forge_ai/gui/tabs/modules_tab.py  - GUI for toggling
    • docs/MODULE_GUIDE.md              - Module documentation
"""

from functools import lru_cache
import logging
import os
from typing import Dict, List, Optional, Type, Any

from .manager import Module, ModuleInfo, ModuleCategory, ModuleManager

logger = logging.getLogger(__name__)


# =============================================================================
# 🧠 CORE MODULE DEFINITIONS
# =============================================================================
# These are the essential modules that power the AI:
# - ModelModule: The neural network brain
# - TokenizerModule: Text → numbers converter
# - TrainingModule: Learn from data
# - InferenceModule: Generate responses

class ModelModule(Module):
    """
    Core transformer model module.
    
    📖 WHAT THIS IS:
    The "brain" of ForgeAI - a transformer neural network that understands
    and generates text. This wraps the Forge model from core/model.py.
    
    📐 CONFIGURATION OPTIONS:
    ┌────────────────────────────────────────────────────────────────────┐
    │ size        │ nano, micro, tiny, small, medium, large, xl, titan  │
    │ vocab_size  │ Number of tokens (1000-500000, default 8000)        │
    │ device      │ auto, cuda, cpu, mps                                │
    │ dtype       │ float32, float16, bfloat16                          │
    └────────────────────────────────────────────────────────────────────┘
    
    🔗 WRAPS: forge_ai/core/model.py → Forge class
    """

    INFO = ModuleInfo(
        id="model",
        name="Forge Model",
        description="Core transformer language model with RoPE, RMSNorm, SwiGLU, GQA",
        category=ModuleCategory.CORE,
        version="2.0.0",
        requires=[],  # No dependencies - this is the foundation!
        provides=[
            "language_model",      # Provides language understanding
            "model_embeddings"],   # Provides word vectors
        config_schema={
            "size": {
                "type": "choice",
                "options": [
                    "nano",     # ~1M params - Raspberry Pi
                    "micro",    # ~2M params - Light devices
                    "tiny",     # ~5M params - Entry level
                    "small",    # ~27M params - Desktop default
                    "medium",   # ~85M params - Good balance
                    "large",    # ~300M params - Quality focus
                    "xl",       # ~1B params - Powerful
                    "xxl",      # ~3B params - Very powerful
                    "titan"],   # ~7B+ params - Datacenter
                "default": "small"},
            "vocab_size": {
                "type": "int",
                        "min": 1000,
                        "max": 500000,
                        "default": 8000},
            "device": {
                "type": "choice",
                "options": [
                    "auto",   # Auto-detect best
                    "cuda",   # NVIDIA GPU
                    "cpu",    # CPU (slow but works everywhere)
                    "mps"],   # Apple Silicon GPU
                "default": "auto"},
            "dtype": {
                "type": "choice",
                "options": [
                    "float32",   # Full precision (most compatible)
                    "float16",   # Half precision (faster, less VRAM)
                    "bfloat16"], # Brain float (best for training)
                "default": "float32"},
        },
        min_ram_mb=512,  # Minimum 512MB RAM required
    )

    def load(self) -> bool:
        """
        Load the Forge model into memory.
        
        📖 WHAT HAPPENS:
        1. Import the model factory from core/model.py
        2. Create a model of the specified size
        3. Store it in self._instance for later use
        """
        from forge_ai.core.model import create_model

        size = self.config.get('size', 'small')
        vocab_size = self.config.get('vocab_size', 8000)

        # Create the model - this allocates the neural network weights
        self._instance = create_model(size, vocab_size=vocab_size)
        return self._instance is not None

    def unload(self) -> bool:
        """
        Unload the model and free memory.
        
        📖 WHY THIS MATTERS:
        Models can use gigabytes of memory. When unloading,
        we delete the reference so Python can garbage collect it.
        We also explicitly clear GPU cache if available.
        """
        if self._instance is not None:
            del self._instance
            self._instance = None
            # Force GPU memory cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        return True


class TokenizerModule(Module):
    """
    Tokenizer module - converts text to/from tokens.
    
    📖 WHAT THIS DOES:
    AI models don't understand text directly - they work with numbers.
    The tokenizer converts:
      "Hello world" → [15496, 995]  (encode)
      [15496, 995] → "Hello world"  (decode)
    
    📐 TOKENIZER TYPES:
    - auto: Best available (tries BPE first)
    - bpe: Byte-Pair Encoding (like GPT)
    - character: One token per character
    - simple: Whitespace-based splitting
    
    🔗 WRAPS: forge_ai/core/tokenizer.py
    """

    INFO = ModuleInfo(
        id="tokenizer",
        name="Tokenizer",
        description="Text tokenization with BPE, character, or custom vocabularies",
        category=ModuleCategory.CORE,
        version="2.0.0",
        requires=[],
        provides=[
            "tokenization",  # Can convert text ↔ tokens
            "vocabulary"],   # Has a word list
        config_schema={
            "type": {
                "type": "choice",
                "options": [
                    "auto",       # Auto-detect best
                    "bpe",        # Byte-Pair Encoding
                    "character",  # Character-level
                    "simple"],    # Whitespace split
                "default": "auto"},
            "vocab_size": {
                "type": "int",
                        "min": 100,
                        "max": 500000,
                        "default": 8000},
        },
    )

    def load(self) -> bool:
        """Load the tokenizer."""
        from forge_ai.core.tokenizer import get_tokenizer

        tok_type = self.config.get('type', 'auto')
        self._instance = get_tokenizer(tok_type)
        return self._instance is not None


class TrainingModule(Module):
    """
    Training module - trains models on data.
    
    📖 WHAT THIS DOES:
    Takes text data and teaches the model to predict the next word.
    Over many iterations, the model learns patterns in language.
    
    📐 TRAINING FEATURES:
    - AMP: Automatic Mixed Precision (faster training)
    - Gradient Accumulation: Train larger batches on limited VRAM
    - Distributed: Train across multiple GPUs
    
    📐 KEY PARAMETERS:
    ┌────────────────────────────────────────────────────────────────────┐
    │ learning_rate  │ How fast to learn (1e-6 to 0.1, default 3e-4)    │
    │ batch_size     │ Examples per step (1-256, default 8)              │
    │ epochs         │ Times through data (1-10000, default 30)          │
    │ use_amp        │ Mixed precision? (default True)                   │
    │ gradient_accumulation │ Steps before update (1-64, default 4)     │
    └────────────────────────────────────────────────────────────────────┘
    
    🔗 WRAPS: forge_ai/core/training.py → Trainer class
    """

    INFO = ModuleInfo(
        id="training",
        name="Training System",
        description="Production-grade training with AMP, gradient accumulation, distributed support",
        category=ModuleCategory.CORE,
        version="2.0.0",
        requires=[
            "model",      # Need a model to train!
            "tokenizer"], # Need tokenizer to process text
        provides=[
            "model_training",   # Can train models
            "fine_tuning"],     # Can fine-tune existing models
        supports_distributed=True,  # Can use multiple GPUs
        config_schema={
            "learning_rate": {
                "type": "float",
                "min": 1e-6,
                "max": 1e-1,
                "default": 3e-4},
            "batch_size": {
                "type": "int",
                "min": 1,
                        "max": 256,
                        "default": 8},
            "epochs": {
                "type": "int",
                "min": 1,
                "max": 10000,
                "default": 30},
            "use_amp": {
                "type": "bool",
                "default": True},
            "gradient_accumulation": {
                "type": "int",
                "min": 1,
                "max": 64,
                "default": 4},
        },
    )

    def load(self) -> bool:
        from forge_ai.core.training import Trainer, TrainingConfig
        self._trainer_class = Trainer
        self._config_class = TrainingConfig
        return True

    def get_interface(self):
        return {
            'Trainer': self._trainer_class,
            'TrainingConfig': self._config_class,
        }


class InferenceModule(Module):
    """
    Inference module - generates text from models.
    
    📖 WHAT THIS DOES:
    Takes a trained model and generates text from it.
    This is the "thinking" part - you give it a prompt, it gives you a response.
    
    📐 KEY PARAMETERS:
    ┌────────────────────────────────────────────────────────────────────┐
    │ max_length   │ Maximum tokens to generate (1-32768, default 2048) │
    │ temperature  │ Randomness (0=deterministic, 2=chaos, default 0.8) │
    │ top_k        │ Consider only top K tokens (0-1000, default 50)    │
    │ top_p        │ Nucleus sampling (0-1, default 0.9)                │
    └────────────────────────────────────────────────────────────────────┘
    
    📐 SAMPLING STRATEGY GUIDE:
    - Creative writing: temperature=1.0, top_p=0.95
    - Code generation: temperature=0.3, top_k=40
    - Factual answers: temperature=0.1, top_p=0.8
    
    🔗 WRAPS: forge_ai/core/inference.py → ForgeEngine class
    """

    INFO = ModuleInfo(
        id="inference",
        name="Inference Engine",
        description="High-performance text generation with streaming, batching, chat",
        category=ModuleCategory.CORE,
        version="2.0.0",
        requires=["model", "tokenizer"],  # Needs model to run and tokenizer to decode
        provides=[
            "text_generation",  # Can generate text
            "streaming",        # Can stream output token by token
            "chat"],            # Can have conversations
        config_schema={
            "max_length": {"type": "int", "min": 1, "max": 32768, "default": 2048},
            "temperature": {"type": "float", "min": 0.0, "max": 2.0, "default": 0.8},
            "top_k": {"type": "int", "min": 0, "max": 1000, "default": 50},
            "top_p": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.9},
        },
    )

    def load(self) -> bool:
        """Load the ForgeEngine class for local inference."""
        from forge_ai.core.inference import ForgeEngine
        self._engine_class = ForgeEngine
        return True


# =============================================================================
# ☁️ CLOUD/API MODULES
# =============================================================================
# These modules use external APIs instead of local models.
# Great for:
# - Raspberry Pi (not enough power for local models)
# - Access to powerful models (GPT-4, Claude)
# - No training required

class ChatAPIModule(Module):
    """
    Cloud/API chat via OpenAI, Anthropic, Ollama, or other providers.
    
    📖 WHAT THIS DOES:
    Instead of running a model locally, this sends your prompts to
    cloud APIs and gets responses back. Perfect for:
    - Low-power devices (Raspberry Pi)
    - Access to powerful models (GPT-4, Claude)
    - No training or GPU needed
    
    📐 PROVIDER OPTIONS:
    ┌────────────────────────────────────────────────────────────────────┐
    │ ollama    │ FREE! Run Llama/Mistral locally, no API key needed    │
    │ openai    │ GPT-4, GPT-3.5 (paid, needs API key)                  │
    │ anthropic │ Claude models (paid, needs API key)                   │
    │ google    │ Gemini (free tier available)                          │
    └────────────────────────────────────────────────────────────────────┘
    
    📐 OLLAMA MODELS (FREE):
    - llama3.2:1b - Fastest, works on 4GB RAM
    - llama3.2:3b - Good balance
    - mistral:7b - Quality responses
    - phi3:mini - Microsoft's small model
    - gemma2:2b - Google's efficient model
    
    ⚠️ CONFLICTS: Can't use with local inference module
    """

    INFO = ModuleInfo(
        id="chat_api",
        name="Cloud/API Chat",
        description="Chat via API (Ollama FREE, GPT-4, Claude). No training needed!",
        category=ModuleCategory.CORE,
        version="1.0.0",
        requires=[],
        conflicts=["inference"],  # Can't use both local and cloud inference
        provides=["text_generation", "streaming", "chat", "cloud_chat"],
        is_cloud_service=True,
        config_schema={
            "provider": {
                "type": "choice",
                "options": ["ollama", "openai", "anthropic", "google"],
                "default": "ollama"
            },
            "model": {
                "type": "choice",
                "options": [
                    # Ollama (FREE - local)
                    "llama3.2:1b",
                    "llama3.2:3b",
                    "mistral:7b",
                    "phi3:mini",
                    "gemma2:2b",
                    # OpenAI (paid)
                    "gpt-4o",
                    "gpt-4-turbo",
                    "gpt-3.5-turbo",
                    # Anthropic (paid)
                    "claude-sonnet-4-20250514",
                    "claude-opus-4-20250514",
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307",
                    # Google (free tier available)
                    "gemini-pro"
                ],
                "default": "llama3.2:1b"
            },
            "api_key": {"type": "secret", "default": ""},
            "ollama_url": {"type": "string", "default": "http://localhost:11434"},
            "max_tokens": {"type": "int", "min": 100, "max": 128000, "default": 4096},
            "temperature": {"type": "float", "min": 0.0, "max": 2.0, "default": 0.7},
        },
    )

    def __init__(self, manager: 'ModuleManager' = None, config: Dict[str, Any] = None):
        super().__init__(manager, config)
        self._client = None
        self._provider = None

    def load(self) -> bool:
        provider = self.config.get('provider', 'ollama')
        
        try:
            if provider == 'ollama':
                # Ollama is FREE and runs locally - no API key needed!
                import requests
                ollama_url = self.config.get('ollama_url', 'http://localhost:11434')
                # Test connection
                try:
                    resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
                    if resp.status_code == 200:
                        self._client = ollama_url
                        self._provider = 'ollama'
                        logger.info(f"Connected to Ollama at {ollama_url}")
                        return True
                    else:
                        logger.warning(f"Ollama not responding. Install from: https://ollama.ai")
                        return False
                except requests.exceptions.ConnectionError:
                    logger.warning(
                        f"Could not connect to Ollama at {ollama_url}. "
                        f"Install Ollama from https://ollama.ai and run: ollama run llama3.2:1b"
                    )
                    return False
                    
            elif provider == 'openai':
                api_key = self.config.get('api_key') or os.environ.get('OPENAI_API_KEY')
                if not api_key:
                    logger.warning("No OpenAI API key. Set in Settings > API Keys")
                    return False
                import openai
                self._client = openai.OpenAI(api_key=api_key)
                
            elif provider == 'anthropic':
                api_key = self.config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
                if not api_key:
                    logger.warning("No Anthropic API key. Set in Settings > API Keys")
                    return False
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
                
            elif provider == 'google':
                api_key = self.config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
                if not api_key:
                    logger.warning("No Google API key. Get free key at: https://makersuite.google.com")
                    return False
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._client = genai.GenerativeModel(self.config.get('model', 'gemini-pro'))
            
            self._provider = provider
            logger.info(f"Loaded {provider} chat")
            return True
        except ImportError as e:
            logger.warning(f"Missing library for {provider}: {e}")
            return False
        except Exception as e:
            logger.warning(f"Could not load {provider} client: {e}")
            return False

    def chat(self, message: str, history: list = None) -> str:
        """Send a chat message and get a response."""
        if not self._client:
            return "Chat API not initialized. Check connection/API key."
        
        model = self.config.get('model', 'llama3.2:1b')
        max_tokens = self.config.get('max_tokens', 4096)
        temperature = self.config.get('temperature', 0.7)
        
        try:
            if self._provider == 'ollama':
                import requests
                response = requests.post(
                    f"{self._client}/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": message}],
                        "stream": False,
                        "options": {"temperature": temperature}
                    },
                    timeout=120
                )
                if response.status_code == 200:
                    return response.json().get("message", {}).get("content", "No response")
                else:
                    return f"Ollama error: {response.text}"
                    
            elif self._provider == 'openai':
                messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
                if history:
                    messages.extend(history)
                messages.append({"role": "user", "content": message})
                
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
                
            elif self._provider == 'anthropic':
                messages = []
                if history:
                    messages.extend(history)
                messages.append({"role": "user", "content": message})
                
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages
                )
                return response.content[0].text
                
            elif self._provider == 'google':
                response = self._client.generate_content(message)
                return response.text
                
        except Exception as e:
            logger.error(f"Cloud chat error: {e}")
            return f"Error: {e}"

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text (alias for chat without history)."""
        return self.chat(prompt)

    def unload(self) -> bool:
        self._client = None
        self._provider = None
        return True


class GGUFLoaderModule(Module):
    """
    GGUF model loader module - load llama.cpp compatible models.
    
    📖 WHAT THIS IS:
    GGUF is a format for quantized models that run efficiently on CPUs.
    This lets you run Llama, Mistral, etc. without a GPU!
    
    📐 CONFIGURATION:
    ┌────────────────────────────────────────────────────────────────────┐
    │ model_path    │ Path to .gguf file                                │
    │ n_ctx         │ Context window (512-32768, default 2048)          │
    │ n_gpu_layers  │ GPU layers (0=CPU only, higher=more GPU)          │
    │ n_threads     │ CPU threads (1-32, default 4)                     │
    └────────────────────────────────────────────────────────────────────┘
    
    ⚠️ CONFLICTS: Can't use with model or inference modules
    🔗 WRAPS: forge_ai/core/gguf_loader.py → GGUFModel class
    """

    INFO = ModuleInfo(
        id="gguf_loader",
        name="GGUF Model Loader",
        description="Load and run GGUF format models (llama.cpp compatible)",
        category=ModuleCategory.CORE,
        version="1.0.0",
        requires=[],
        conflicts=["model", "inference"],  # Can't use both GGUF and Forge model
        provides=["text_generation", "completion", "gguf_support"],
        config_schema={
            "model_path": {"type": "string", "default": ""},
            "n_ctx": {"type": "int", "min": 512, "max": 32768, "default": 2048},
            "n_gpu_layers": {"type": "int", "min": 0, "max": 999, "default": 0},
            "n_threads": {"type": "int", "min": 1, "max": 32, "default": 4},
        },
    )

    def load(self) -> bool:
        """
        Load a GGUF model into memory.
        
        📖 HOW IT WORKS:
        1. Get the path to the .gguf file from config
        2. Create a GGUFModel wrapper
        3. Load the model (may take time for large models)
        """
        try:
            from forge_ai.core.gguf_loader import GGUFModel
            model_path = self.config.get('model_path', '')
            if not model_path:
                logger.warning("No GGUF model path specified")
                return False
            
            # Create and load the GGUF model wrapper
            self._instance = GGUFModel(
                model_path=model_path,
                n_ctx=self.config.get('n_ctx', 2048),
                n_gpu_layers=self.config.get('n_gpu_layers', 0),
                n_threads=self.config.get('n_threads', 4)
            )
            return self._instance.load()
        except Exception as e:
            logger.warning(f"Could not load GGUF model: {e}")
            return False

    def unload(self) -> bool:
        """Unload the GGUF model and free memory."""
        if self._instance:
            self._instance.unload()
            self._instance = None
            # Force GPU memory cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        return True


# =============================================================================
# 💾 MEMORY MODULES
# =============================================================================
# These modules handle storing and retrieving information.

class MemoryModule(Module):
    """
    Memory module - conversation and knowledge storage.
    
    📖 WHAT THIS DOES:
    Stores chat history so the AI remembers previous conversations.
    Can use different backends:
    - JSON: Simple file storage (good for small datasets)
    - SQLite: Database (good for large histories)
    - Vector: Semantic search (find similar conversations)
    
    📐 CONFIGURATION:
    ┌────────────────────────────────────────────────────────────────────┐
    │ backend           │ json, sqlite, or vector                       │
    │ max_conversations │ How many to keep (default 1000)               │
    └────────────────────────────────────────────────────────────────────┘
    
    🔗 WRAPS: forge_ai/memory/manager.py → ConversationManager class
    """

    INFO = ModuleInfo(
        id="memory",
        name="Memory System",
        description="Conversation history, vector search, long-term memory",
        category=ModuleCategory.MEMORY,
        version="1.0.0",
        requires=[],
        optional=["model"],  # For generating embeddings
        provides=[
            "conversation_storage",  # Can save/load conversations
            "vector_search",         # Can search by similarity
            "knowledge_base"],       # Can store facts
        config_schema={
            "backend": {"type": "choice", "options": ["json", "sqlite", "vector"], "default": "json"},
            "max_conversations": {"type": "int", "min": 1, "max": 100000, "default": 1000},
        },
    )

    def load(self) -> bool:
        """Load the conversation manager."""
        from forge_ai.memory.manager import ConversationManager
        self._instance = ConversationManager()
        return True


# =============================================================================
# 🎤 PERCEPTION MODULES (Input)
# =============================================================================
# These modules handle input from the real world.

class VoiceInputModule(Module):
    """
    Voice input module - speech to text.
    
    📖 WHAT THIS DOES:
    Listens to your microphone and converts speech to text.
    This lets you talk to the AI instead of typing!
    
    📐 ENGINE OPTIONS:
    ┌────────────────────────────────────────────────────────────────────┐
    │ system  │ Uses your OS's built-in speech recognition              │
    │ whisper │ OpenAI's Whisper (very accurate, needs GPU)             │
    │ vosk    │ Offline recognition (works without internet)            │
    └────────────────────────────────────────────────────────────────────┘
    
    🔗 WRAPS: forge_ai/voice/stt_simple.py
    """

    INFO = ModuleInfo(
        id="voice_input",
        name="Voice Input (STT)",
        description="Speech-to-text for voice commands and dictation",
        category=ModuleCategory.PERCEPTION,
        version="1.0.0",
        requires=[],
        provides=[
            "speech_recognition",  # Can convert speech to text
            "voice_commands"],     # Can understand voice commands
        config_schema={
            "engine": {
                "type": "choice",
                "options": [
                    "system",   # OS built-in
                    "whisper",  # OpenAI Whisper
                    "vosk"],    # Offline
                "default": "system"},
            "language": {
                "type": "string",
                        "default": "en-US"},
        },
    )

    def load(self) -> bool:
        """
        Load the speech-to-text system.
        
        📖 HOW IT WORKS:
        Wraps the STT functions in a simple class that has a listen() method.
        """
        try:
            from forge_ai.voice.stt_simple import transcribe_from_mic, transcribe_from_file
            
            # Wrap functions in a simple class for consistency
            class STTWrapper:
                def __init__(self):
                    self.transcribe_from_mic = transcribe_from_mic
                    self.transcribe_from_file = transcribe_from_file
                
                def listen(self, timeout=8):
                    """Listen for speech and return text."""
                    return transcribe_from_mic(timeout)
            
            self._instance = STTWrapper()
            return True
        except Exception as e:
            logger.warning(f"Could not load voice input: {e}")
            return False


# =============================================================================
# 🔊 OUTPUT MODULES
# =============================================================================

class VoiceOutputModule(Module):
    """
    Voice output module - text to speech.
    
    📖 WHAT THIS DOES:
    Converts text to spoken audio so the AI can talk back to you!
    
    📐 ENGINE OPTIONS:
    ┌────────────────────────────────────────────────────────────────────┐
    │ pyttsx3    │ Offline TTS (works without internet, robotic voice)  │
    │ elevenlabs │ Cloud TTS (very natural, paid API)                   │
    │ coqui      │ Local neural TTS (good quality, needs GPU)           │
    └────────────────────────────────────────────────────────────────────┘
    """

    INFO = ModuleInfo(
        id="voice_output",
        name="Voice Output (TTS)",
        description="Text-to-speech for spoken responses",
        category=ModuleCategory.OUTPUT,
        version="1.0.0",
        requires=[],
        provides=[
            "text_to_speech",
            "spoken_response"],
        config_schema={
            "engine": {
                "type": "choice",
                "options": [
                    "system",
                    "pyttsx3",
                    "elevenlabs"],
                "default": "system"},
            "voice": {
                "type": "string",
                        "default": "default"},
            "rate": {
                "type": "float",
                "min": 0.5,
                "max": 2.0,
                "default": 1.0},
        },
    )

    def load(self) -> bool:
        try:
            from forge_ai.voice.tts_simple import speak
            # Wrap function in a simple class for consistency
            class TTSWrapper:
                def __init__(self):
                    self._speak = speak
                
                def speak(self, text):
                    return self._speak(text)
                
                def say(self, text):
                    return self._speak(text)
            
            self._instance = TTSWrapper()
            return True
        except Exception as e:
            logger.warning(f"Could not load voice output: {e}")
            return False


class VisionModule(Module):
    """
    Vision module - image processing and analysis.
    
    📖 WHAT THIS DOES:
    Gives the AI "eyes" to see and understand images. Can:
    - Capture from webcam/screen
    - Read text in images (OCR)
    - Detect objects in images
    
    📐 CONFIGURATION:
    ┌────────────────────────────────────────────────────────────────────┐
    │ camera_id  │ Which camera to use (0 = default webcam)             │
    │ resolution │ 480p, 720p, or 1080p                                 │
    └────────────────────────────────────────────────────────────────────┘
    
    🔗 WRAPS: forge_ai/tools/vision.py → ScreenVision class
    """

    INFO = ModuleInfo(
        id="vision",
        name="Vision System",
        description="Image capture, OCR, object detection, visual understanding",
        category=ModuleCategory.PERCEPTION,
        version="1.0.0",
        requires=[],
        optional=["model"],  # Model can help with image understanding
        provides=[
            "image_capture",     # Can grab images
            "ocr",               # Can read text in images
            "object_detection"], # Can find objects
        config_schema={
            "camera_id": {
                "type": "int",
                "min": 0,
                "max": 10,
                "default": 0},
            "resolution": {
                "type": "choice",
                "options": [
                        "480p",    # 640x480 - Fast
                        "720p",    # 1280x720 - Balanced
                        "1080p"],  # 1920x1080 - Quality
                "default": "720p"},
        },
    )

    def load(self) -> bool:
        """Load the vision system."""
        try:
            from forge_ai.tools.vision import ScreenVision
            self._instance = ScreenVision()
            return True
        except Exception as e:
            logger.warning(f"Could not load vision: {e}")
            return False


class AvatarModule(Module):
    """
    Avatar module - visual AI representation.
    
    📖 WHAT THIS DOES:
    Gives the AI a face! Can display emotions, lip-sync with speech,
    and show animations. Makes the AI feel more human.
    
    📐 STYLES:
    - 2d: Simple 2D character image
    - 3d: 3D rendered character
    - animated: Full animation with expressions
    
    🔗 WRAPS: forge_ai/avatar/controller.py → AvatarController class
    """

    INFO = ModuleInfo(
        id="avatar",
        name="Avatar System",
        description="Visual representation, expressions, animations",
        category=ModuleCategory.OUTPUT,
        version="1.0.0",
        requires=[],
        optional=["voice_output"],  # Can lip-sync with voice
        provides=[
            "visual_avatar",  # Can show a character
            "expressions",    # Can show emotions
            "lip_sync"],      # Can move lips with speech
        config_schema={
            "style": {"type": "choice", "options": ["2d", "3d", "animated"], "default": "2d"},
            "character": {"type": "string", "default": "default"},
        },
    )

    def load(self) -> bool:
        """Load the avatar controller."""
        try:
            from forge_ai.avatar.controller import AvatarController
            self._instance = AvatarController()
            return True
        except Exception as e:
            logger.warning(f"Could not load avatar: {e}")
            return False


# =============================================================================
# 🔧 TOOL MODULES
# =============================================================================
# These modules let the AI interact with the outside world.

class WebToolsModule(Module):
    """
    Web tools module - internet access.
    
    📖 WHAT THIS DOES:
    Lets the AI search the web, fetch web pages, and call APIs.
    This is how the AI can get up-to-date information!
    
    ⚠️ SECURITY: Has safety limits on URLs and timeout
    
    🔗 WRAPS: forge_ai/tools/web_tools.py → WebTools class
    """

    INFO = ModuleInfo(
        id="web_tools",
        name="Web Tools",
        description="Web search, page fetching, API access",
        category=ModuleCategory.TOOLS,
        version="1.0.0",
        requires=[],
        provides=[
            "web_search",   # Can search the web
            "url_fetch",    # Can download web pages
            "api_access"],  # Can call REST APIs
        config_schema={
            "allow_external": {"type": "bool", "default": True},  # Allow internet access?
            "timeout": {"type": "int", "min": 1, "max": 300, "default": 30},  # Timeout in seconds
        },
    )

    def load(self) -> bool:
        """Load web tools."""
        try:
            from forge_ai.tools.web_tools import WebTools
            self._instance = WebTools()
            return True
        except ImportError as e:
            logger.debug(f"WebTools not available (optional): {e}")
            return True  # Optional module - don't fail if unavailable
        except Exception as e:
            logger.warning(f"WebTools failed to initialize: {e}")
            return False  # Unexpected error - report failure


class FileToolsModule(Module):
    """
    File tools module - file system access.
    
    📖 WHAT THIS DOES:
    Lets the AI read and write files on your computer.
    
    ⚠️ SECURITY: Has a blocked paths list to prevent access to sensitive files
    (see forge_ai/utils/security.py)
    
    🔗 WRAPS: forge_ai/tools/file_tools.py
    """

    INFO = ModuleInfo(
        id="file_tools",
        name="File Tools",
        description="Read, write, search files and directories",
        category=ModuleCategory.TOOLS,
        version="1.0.0",
        requires=[],
        provides=[
            "file_read",    # Can read file contents
            "file_write",   # Can write to files
            "file_search"], # Can search for files
        config_schema={
            "allowed_paths": {"type": "list", "default": []},
            "max_file_size_mb": {"type": "int", "min": 1, "max": 1000, "default": 100},
        },
    )

    def load(self) -> bool:
        try:
            from forge_ai.tools.file_tools import FileTools
            self._instance = FileTools()
            return True
        except ImportError as e:
            logger.debug(f"FileTools not available (optional): {e}")
            return True  # Optional module
        except Exception as e:
            logger.warning(f"FileTools failed to initialize: {e}")
            return False


class ToolRouterModule(Module):
    """
    Tool router module - specialized model routing.
    
    📖 WHAT THIS DOES:
    Routes different types of requests to specialized models:
    - Intent classification: "What does the user want?"
    - Vision: "What's in this image?"
    - Code: "Write me Python code"
    
    📐 HOW ROUTING WORKS:
    ┌────────────────────────────────────────────────────────────────────┐
    │ User Request → Router Model → "code" intent → Code Model          │
    │                            → "chat" intent → Chat Model           │
    │                            → "vision" intent → Vision Model       │
    └────────────────────────────────────────────────────────────────────┘
    
    🔗 WRAPS: forge_ai/core/tool_router.py → ToolRouter class
    """

    INFO = ModuleInfo(
        id="tool_router",
        name="Tool Router",
        description="Route requests to specialized models (intent classification, vision, code)",
        category=ModuleCategory.TOOLS,
        version="1.0.0",
        requires=["tokenizer"],
        optional=["model"],  # Uses model if available
        provides=[
            "intent_classification",   # Can classify what user wants
            "specialized_routing",     # Can route to right model
            "model_routing"],          # Can manage multiple models
        config_schema={
            "use_specialized": {
                "type": "bool",
                "default": True,
                "description": "Use specialized models for routing"
            },
            "enable_router": {
                "type": "bool",
                "default": True,
                "description": "Enable intent router model"
            },
            "enable_vision": {
                "type": "bool",
                "default": True,
                "description": "Enable vision captioning model"
            },
            "enable_code": {
                "type": "bool",
                "default": True,
                "description": "Enable code generation model"
            },
        },
    )

    def load(self) -> bool:
        """Load the tool router with specialized model support."""
        try:
            from forge_ai.core.tool_router import get_router
            use_specialized = self.config.get('use_specialized', True)
            self._instance = get_router(use_specialized=use_specialized)
            logger.info(f"Tool router loaded (specialized: {use_specialized})")
            return True
        except Exception as e:
            logger.warning(f"Could not load tool router: {e}")
            return False

    def unload(self) -> bool:
        """Unload and clear cached models."""
        if self._instance is not None:
            # Clear any cached models to free memory
            if hasattr(self._instance, '_specialized_models'):
                self._instance._specialized_models.clear()
            if hasattr(self._instance, '_model_cache'):
                self._instance._model_cache.clear()
            self._instance = None
            # Force GPU memory cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        return True


# =============================================================================
# 🌐 INTERFACE MODULES
# =============================================================================
# These modules provide ways to interact with the AI.

class APIServerModule(Module):
    """
    API server module - REST API interface.
    
    📖 WHAT THIS DOES:
    Creates a REST API so other programs can talk to ForgeAI.
    Great for building apps, bots, or integrating with other tools.
    
    📐 ENDPOINTS (when running):
    - POST /chat - Send a message, get a response
    - GET /health - Check if server is running
    - POST /generate - Generate text from prompt
    
    🔗 WRAPS: forge_ai/comms/api_server.py → create_app()
    """

    INFO = ModuleInfo(
        id="api_server",
        name="API Server",
        description="REST API for remote access and integrations",
        category=ModuleCategory.INTERFACE,
        version="1.0.0",
        requires=["inference"],  # Need inference to generate responses
        provides=[
            "rest_api",       # Provides REST endpoints
            "remote_access"], # Enables remote use
        config_schema={
            "host": {"type": "string", "default": "127.0.0.1"},  # localhost by default (safe)
            "port": {"type": "int", "min": 1, "max": 65535, "default": 5000},
            "auth_enabled": {"type": "bool", "default": False},  # Enable API auth?
        },
    )

    def load(self) -> bool:
        """Load the API server factory."""
        from forge_ai.comms.api_server import create_app
        self._app_factory = create_app
        return True


class NetworkModule(Module):
    """
    Network module - multi-device communication.
    
    📖 WHAT THIS DOES:
    Lets multiple computers work together! You can:
    - Split inference across devices
    - Share models between machines
    - Run the brain on a server, UI on a tablet
    
    📐 NETWORK ROLES:
    ┌────────────────────────────────────────────────────────────────────┐
    │ standalone │ Solo mode, no networking                             │
    │ server     │ Central hub that coordinates others                  │
    │ client     │ Connects to a server                                 │
    │ peer       │ Equal partner in a mesh network                      │
    └────────────────────────────────────────────────────────────────────┘
    
    🔗 WRAPS: forge_ai/comms/network.py → ForgeNode class
    """

    INFO = ModuleInfo(
        id="network",
        name="Network System",
        description="Multi-device communication, distributed inference, model sharing",
        category=ModuleCategory.NETWORK,
        version="1.0.0",
        requires=[],
        optional=[
            "model",      # Can share models
            "inference"], # Can distribute inference
        provides=[
            "multi_device",           # Multiple computers can work together
            "distributed_inference",  # Split work across machines
            "model_sync"],            # Keep models in sync
        supports_distributed=True,
        config_schema={
            "role": {
                "type": "choice",
                "options": [
                        "standalone",  # No networking
                        "server",      # Central coordinator
                        "client",      # Connect to server
                        "peer"],       # Mesh network
                "default": "standalone"},
            "discovery": {
                "type": "bool",
                "default": True},  # Auto-discover other nodes?
        },
    )

    def load(self) -> bool:
        """Load the network node."""
        try:
            from forge_ai.comms.network import ForgeNode
            self._instance = ForgeNode()
            return True
        except ImportError as e:
            logger.debug(f"ForgeNode not available (optional): {e}")
            return True  # Optional module
        except Exception as e:
            logger.warning(f"ForgeNode failed to initialize: {e}")
            return False


class GUIModule(Module):
    """
    GUI module - graphical interface.
    
    📖 WHAT THIS DOES:
    Provides the main graphical interface using PyQt5.
    Has tabs for chat, training, modules, and all the generation features.
    
    📐 THEMES:
    - dark: Easy on the eyes (default)
    - light: Bright mode
    - system: Match your OS settings
    
    🔗 WRAPS: forge_ai/gui/enhanced_window.py → EnhancedMainWindow
    """

    INFO = ModuleInfo(
        id="gui",
        name="Graphical Interface",
        description="PyQt5-based GUI with chat, training, modules management",
        category=ModuleCategory.INTERFACE,
        version="2.0.0",
        requires=[],
        optional=[
            "model",        # For chat
            "tokenizer",    # For chat
            "inference",    # For text generation
            "training",     # For training tab
            "memory",       # For conversation history
            "voice_input",  # For voice commands
            "voice_output", # For spoken responses
            "vision",       # For vision tab
            "avatar"],      # For avatar display
        provides=[
            "graphical_interface",  # Main window
            "chat_ui",              # Chat interface
            "training_ui",          # Training controls
            "module_management"],   # Module toggle UI
        config_schema={
            "theme": {
                "type": "choice",
                "options": [
                        "dark",    # Dark mode
                        "light",   # Light mode
                        "system"], # Match OS
                "default": "dark"},
            "window_size": {
                "type": "choice",
                "options": [
                    "small",       # 800x600
                    "medium",      # 1200x800
                    "large",       # 1600x900
                    "fullscreen"], # Full screen
                "default": "medium"},
        },
    )

    def load(self) -> bool:
        """GUI is loaded on demand when window is created."""
        return True


# =============================================================================
# 🎨 GENERATION MODULES
# =============================================================================
# These modules generate content (images, code, video, audio, 3D, etc.)
# Each capability has LOCAL and API variants that CONFLICT with each other
# (you can only use one at a time - they both provide the same capability)

class GenerationModule(Module):
    """
    Base class for generation modules.
    
    📖 PATTERN:
    All generation modules follow this pattern:
    1. Wrap an "addon" from the GUI tabs
    2. load() creates and loads the addon
    3. generate() calls the addon's generate method
    4. unload() cleans up the addon
    
    📐 LOCAL vs API:
    ┌────────────────────────────────────────────────────────────────────┐
    │ LOCAL modules: Run on your GPU, no internet needed, free          │
    │ API modules: Run in the cloud, need internet + API key, paid      │
    │                                                                    │
    │ ⚠️ LOCAL and API versions CONFLICT - can only load ONE at a time  │
    └────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, manager, config=None):
        super().__init__(manager, config)
        self._addon = None  # The wrapped addon from GUI tabs

    def unload(self) -> bool:
        """Unload the wrapped addon."""
        if self._addon:
            self._addon.unload()
            self._addon = None
        return True

    def generate(self, prompt: str, **kwargs):
        """
        Generate content using the wrapped addon.
        
        Args:
            prompt: What to generate (text description)
            **kwargs: Additional options passed to the addon
            
        Returns:
            dict with 'success' and either 'result' or 'error'
        """
        if not self._addon or not self._addon.is_loaded:
            raise RuntimeError(f"Module not loaded")
        
        # Ensure prompt is a string (CLIP tokenizer requires str type)
        if prompt is not None:
            prompt = str(prompt).strip()
        if not prompt:
            return {"success": False, "error": "Prompt cannot be empty"}
        
        return self._addon.generate(prompt, **kwargs)

    def get_interface(self):
        """Return the addon for direct access."""
        return self._addon


# -----------------------------------------------------------------------------
# 🖼️ IMAGE GENERATION
# -----------------------------------------------------------------------------

class ImageGenLocalModule(GenerationModule):
    """
    Local image generation with Stable Diffusion.
    
    📖 WHAT THIS DOES:
    Generates images from text prompts using Stable Diffusion running
    locally on your GPU. No internet needed, completely free!
    
    📐 MODELS:
    ┌────────────────────────────────────────────────────────────────────┐
    │ sd-2.1      │ Stable Diffusion 2.1 (6GB VRAM)                     │
    │ sdxl        │ Stable Diffusion XL (8GB VRAM, higher quality)      │
    │ sdxl-turbo  │ Fast version of SDXL (4 steps instead of 30)        │
    └────────────────────────────────────────────────────────────────────┘
    
    ⚠️ CONFLICTS: image_gen_api (can only use one image generator)
    🔗 WRAPS: forge_ai/gui/tabs/image_tab.py → StableDiffusionLocal
    """

    INFO = ModuleInfo(
        id="image_gen_local",
        name="Image Generation (Local)",
        description="Generate images locally with Stable Diffusion. Requires GPU with 8GB+ VRAM.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=[],
        provides=["image_generation"],  # Both local and API provide this
        min_vram_mb=6000,  # Need at least 6GB VRAM
        requires_gpu=True,  # GPU required!
        config_schema={
            "model": {
                "type": "choice",
                "options": [
                    "sd-2.1",      # Smaller, works on 6GB
                    "sdxl",        # Better quality, needs 8GB
                    "sdxl-turbo"], # Fast but lower quality
                "default": "sd-2.1"},
            "steps": {
                "type": "int",
                "min": 1,
                "max": 100,
                "default": 30},  # More steps = better quality but slower
        },
    )

    def load(self) -> bool:
        """Load the Stable Diffusion model."""
        try:
            from forge_ai.gui.tabs.image_tab import StableDiffusionLocal
            self._addon = StableDiffusionLocal()
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load local image gen: {e}")
            return False


class ImageGenAPIModule(GenerationModule):
    """
    Cloud image generation via APIs.
    
    📖 WHAT THIS DOES:
    Generates images using cloud APIs (DALL-E, Replicate).
    Great if you don't have a GPU!
    
    📐 PROVIDERS:
    ┌────────────────────────────────────────────────────────────────────┐
    │ openai    │ DALL-E 3 (best quality, $0.04/image)                  │
    │ replicate │ SDXL, Flux, etc (many models, usage-based pricing)    │
    └────────────────────────────────────────────────────────────────────┘
    
    ⚠️ CONFLICTS: image_gen_local (can only use one image generator)
    🔗 WRAPS: forge_ai/gui/tabs/image_tab.py → OpenAIImage, ReplicateImage
    """

    INFO = ModuleInfo(
        id="image_gen_api",
        name="Image Generation (Cloud)",
        description="Generate images via OpenAI DALL-E or Replicate (SDXL, Flux). Requires API key.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=[],
        provides=["image_generation"],  # Same capability as local
        is_cloud_service=True,  # Uses cloud APIs
        config_schema={
            "provider": {
                "type": "choice",
                "options": [
                    "openai",
                    "replicate"],
                "default": "openai"},
            "model": {
                "type": "string",
                "default": "dall-e-3"},
            "api_key": {
                "type": "secret",
                "default": ""},
        },
    )

    def load(self) -> bool:
        try:
            provider = self.config.get('provider', 'openai')
            if provider == 'openai':
                from forge_ai.gui.tabs.image_tab import OpenAIImage
                self._addon = OpenAIImage(api_key=self.config.get('api_key'))
            else:
                from forge_ai.gui.tabs.image_tab import ReplicateImage
                self._addon = ReplicateImage(api_key=self.config.get('api_key'))
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load cloud image gen: {e}")
            return False


# -----------------------------------------------------------------------------
# 💻 CODE GENERATION
# -----------------------------------------------------------------------------

class CodeGenLocalModule(GenerationModule):
    """
    Local code generation using Forge model.
    
    📖 WHAT THIS DOES:
    Generates code using your locally trained Forge model.
    Completely free and private - your code never leaves your machine!
    
    📐 TIP: Lower temperature (0.1-0.3) gives more predictable code
    
    🔗 WRAPS: forge_ai/gui/tabs/code_tab.py → ForgeCode
    """

    INFO = ModuleInfo(
        id="code_gen_local",
        name="Code Generation (Local)",
        description="Generate code using your trained Forge model. Free and private.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=["model", "tokenizer", "inference"],  # Needs the full local stack
        provides=["code_generation"],
        config_schema={
            "model_name": {"type": "string", "default": "sacrifice"},
            "temperature": {"type": "float", "min": 0.1, "max": 1.5, "default": 0.3},
        },
    )

    def load(self) -> bool:
        """Load the local code generation model."""
        try:
            from forge_ai.gui.tabs.code_tab import ForgeCode
            self._addon = ForgeCode(model_name=self.config.get('model_name', 'sacrifice'))
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load local code gen: {e}")
            return False


class CodeGenAPIModule(GenerationModule):
    """
    Cloud code generation via OpenAI.
    
    📖 WHAT THIS DOES:
    Uses OpenAI's GPT-4 to generate code. Very high quality,
    but requires an API key and costs money per request.
    
    📐 MODELS:
    ┌────────────────────────────────────────────────────────────────────┐
    │ gpt-4        │ Best quality, most expensive                       │
    │ gpt-4-turbo  │ Faster, still very good                            │
    │ gpt-3.5-turbo│ Cheapest, good for simple code                     │
    └────────────────────────────────────────────────────────────────────┘
    
    🔗 WRAPS: forge_ai/gui/tabs/code_tab.py → OpenAICode
    """

    INFO = ModuleInfo(
        id="code_gen_api",
        name="Code Generation (Cloud)",
        description="Generate code via OpenAI GPT-4. High quality, requires API key.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=[],
        provides=["code_generation"],
        is_cloud_service=True,
        config_schema={
            "model": {
                "type": "choice",
                "options": [
                    "gpt-4",         # Best
                    "gpt-4-turbo",   # Fast + good
                    "gpt-3.5-turbo"],# Cheap
                "default": "gpt-4"},
            "api_key": {
                "type": "secret",
                "default": ""},
        },
    )

    def load(self) -> bool:
        """Load the OpenAI code generation client."""
        try:
            from forge_ai.gui.tabs.code_tab import OpenAICode
            self._addon = OpenAICode(
                api_key=self.config.get('api_key'),
                model=self.config.get(
                    'model',
                    'gpt-4'))
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load cloud code gen: {e}")
            return False


# -----------------------------------------------------------------------------
# 🎬 VIDEO GENERATION
# -----------------------------------------------------------------------------

class VideoGenLocalModule(GenerationModule):
    """
    Local video generation with AnimateDiff.
    
    📖 WHAT THIS DOES:
    Generates short video clips from text prompts using AnimateDiff.
    Needs a powerful GPU (12GB+ VRAM recommended).
    
    📐 SETTINGS:
    - fps: Frames per second (4-30)
    - duration: Length in seconds (1-10)
    
    ⚠️ RESOURCE INTENSIVE: Uses lots of VRAM and takes time!
    
    🔗 WRAPS: forge_ai/gui/tabs/video_tab.py → LocalVideo
    """

    INFO = ModuleInfo(
        id="video_gen_local",
        name="Video Generation (Local)",
        description="Generate videos locally with AnimateDiff. Requires powerful GPU.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=[],
        provides=["video_generation"],
        min_vram_mb=12000,  # Need 12GB VRAM minimum
        requires_gpu=True,
        config_schema={
            "fps": {"type": "int", "min": 4, "max": 30, "default": 8},
            "duration": {"type": "float", "min": 1, "max": 10, "default": 4},
        },
    )

    def load(self) -> bool:
        """Load the AnimateDiff video generator."""
        try:
            from forge_ai.gui.tabs.video_tab import LocalVideo
            self._addon = LocalVideo()
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load local video gen: {e}")
            return False


class VideoGenAPIModule(GenerationModule):
    """
    Cloud video generation via Replicate.
    
    📖 WHAT THIS DOES:
    Generates videos using cloud APIs (Zeroscope, etc).
    Good option if you don't have a powerful GPU.
    
    🔗 WRAPS: forge_ai/gui/tabs/video_tab.py → ReplicateVideo
    """

    INFO = ModuleInfo(
        id="video_gen_api",
        name="Video Generation (Cloud)",
        description="Generate videos via Replicate (Zeroscope, etc). Requires API key.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=[],
        provides=["video_generation"],
        is_cloud_service=True,
        config_schema={
            "model": {"type": "string", "default": "zeroscope"},
            "api_key": {"type": "secret", "default": ""},
        },
    )

    def load(self) -> bool:
        """Load the Replicate video client."""
        try:
            from forge_ai.gui.tabs.video_tab import ReplicateVideo
            self._addon = ReplicateVideo(api_key=self.config.get('api_key'))
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load cloud video gen: {e}")
            return False


# -----------------------------------------------------------------------------
# 🔊 AUDIO GENERATION
# -----------------------------------------------------------------------------

class AudioGenLocalModule(GenerationModule):
    """
    Local audio/TTS generation.
    
    📖 WHAT THIS DOES:
    Converts text to spoken audio using local engines.
    Works offline, completely free!
    
    📐 ENGINES:
    ┌────────────────────────────────────────────────────────────────────┐
    │ pyttsx3  │ Offline, robotic voice, very fast                      │
    │ edge-tts │ Microsoft Edge voices (needs internet for first use)   │
    └────────────────────────────────────────────────────────────────────┘
    
    🔗 WRAPS: forge_ai/gui/tabs/audio_tab.py → LocalTTS
    """

    INFO = ModuleInfo(
        id="audio_gen_local",
        name="Audio/TTS (Local)",
        description="Local text-to-speech and audio generation. Free, works offline.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=[],
        provides=["audio_generation", "text_to_speech"],
        config_schema={
            "engine": {"type": "choice", "options": ["pyttsx3", "edge-tts"], "default": "pyttsx3"},
            "voice": {"type": "string", "default": "default"},
        },
    )

    def load(self) -> bool:
        try:
            from forge_ai.gui.tabs.audio_tab import LocalTTS
            self._addon = LocalTTS()
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load local audio gen: {e}")
            return False


class AudioGenAPIModule(GenerationModule):
    """
    Cloud audio generation via ElevenLabs/Replicate.
    
    📖 WHAT THIS DOES:
    Premium text-to-speech with natural voices (ElevenLabs)
    or AI music generation (MusicGen via Replicate).
    
    📐 PROVIDERS:
    ┌────────────────────────────────────────────────────────────────────┐
    │ elevenlabs │ Best TTS voices (natural, expressive)                │
    │ replicate  │ MusicGen and other audio models                      │
    └────────────────────────────────────────────────────────────────────┘
    
    🔗 WRAPS: forge_ai/gui/tabs/audio_tab.py → ElevenLabsTTS, ReplicateAudio
    """

    INFO = ModuleInfo(
        id="audio_gen_api",
        name="Audio/TTS (Cloud)",
        description="Premium TTS via ElevenLabs or music via MusicGen. Requires API key.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=[],
        provides=[
            "audio_generation",    # Can make sounds
            "text_to_speech",      # Can speak text
            "music_generation"],   # Can make music
        is_cloud_service=True,
        config_schema={
            "provider": {
                "type": "choice",
                "options": [
                    "elevenlabs",  # Best TTS
                    "replicate"],  # MusicGen
                "default": "elevenlabs"},
            "api_key": {
                "type": "secret",
                        "default": ""},
            "voice": {
                "type": "string",
                "default": "Rachel"},  # ElevenLabs default voice
        },
    )

    def load(self) -> bool:
        """Load the audio generation provider."""
        try:
            provider = self.config.get('provider', 'elevenlabs')
            if provider == 'elevenlabs':
                from forge_ai.gui.tabs.audio_tab import ElevenLabsTTS
                self._addon = ElevenLabsTTS(api_key=self.config.get('api_key'))
            else:
                from forge_ai.gui.tabs.audio_tab import ReplicateAudio
                self._addon = ReplicateAudio(api_key=self.config.get('api_key'))
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load cloud audio gen: {e}")
            return False


# -----------------------------------------------------------------------------
# 🧠 EMBEDDING GENERATION
# -----------------------------------------------------------------------------

class EmbeddingLocalModule(GenerationModule):
    """
    Local embedding generation with sentence-transformers.
    
    📖 WHAT THIS DOES:
    Converts text into vectors (numbers) that capture meaning.
    This enables semantic search - finding similar text even if
    the exact words are different.
    
    📐 MODELS:
    ┌────────────────────────────────────────────────────────────────────┐
    │ all-MiniLM-L6-v2    │ Fast, good quality, 22M params             │
    │ all-mpnet-base-v2   │ Better quality, 109M params                │
    └────────────────────────────────────────────────────────────────────┘
    
    🔗 WRAPS: forge_ai/gui/tabs/embeddings_tab.py → LocalEmbedding
    """

    INFO = ModuleInfo(
        id="embedding_local",
        name="Embeddings (Local)",
        description="Generate vector embeddings locally for semantic search. Free and private.",
        category=ModuleCategory.MEMORY,
        version="1.0.0",
        requires=[],
        provides=[
            "embeddings",        # Can create vectors from text
            "semantic_search"],  # Can find similar text
        config_schema={
            "model": {
                "type": "choice",
                "options": [
                    "all-MiniLM-L6-v2",     # Fast and small
                    "all-mpnet-base-v2"],   # Better quality
                "default": "all-MiniLM-L6-v2"},
        },
    )

    def load(self) -> bool:
        """Load the sentence-transformers model."""
        try:
            from forge_ai.gui.tabs.embeddings_tab import LocalEmbedding
            self._addon = LocalEmbedding(model_name=self.config.get('model', 'all-MiniLM-L6-v2'))
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load local embeddings: {e}")
            return False


class EmbeddingAPIModule(GenerationModule):
    """
    Cloud embeddings via OpenAI.
    
    📖 WHAT THIS DOES:
    Uses OpenAI's embedding models to convert text to vectors.
    Higher quality than local models, but requires API key.
    
    📐 MODELS:
    ┌────────────────────────────────────────────────────────────────────┐
    │ text-embedding-3-small │ Cheaper, 1536 dimensions                 │
    │ text-embedding-3-large │ Better quality, 3072 dimensions          │
    └────────────────────────────────────────────────────────────────────┘
    
    🔗 WRAPS: forge_ai/gui/tabs/embeddings_tab.py → OpenAIEmbedding
    """

    INFO = ModuleInfo(
        id="embedding_api",
        name="Embeddings (Cloud)",
        description="High-quality embeddings via OpenAI API. Requires API key.",
        category=ModuleCategory.MEMORY,
        version="1.0.0",
        requires=[],
        provides=[
            "embeddings",
            "semantic_search"],
        is_cloud_service=True,
        config_schema={
            "model": {
                "type": "choice",
                "options": [
                    "text-embedding-3-small",  # Cheaper
                    "text-embedding-3-large"], # Better
                "default": "text-embedding-3-small"},
            "api_key": {
                "type": "secret",
                        "default": ""},
        },
    )

    def load(self) -> bool:
        """Load the OpenAI embedding client."""
        try:
            from forge_ai.gui.tabs.embeddings_tab import OpenAIEmbedding
            self._addon = OpenAIEmbedding(
                api_key=self.config.get('api_key'),
                model=self.config.get('model'))
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load cloud embeddings: {e}")
            return False


# -----------------------------------------------------------------------------
# 🎮 3D GENERATION
# -----------------------------------------------------------------------------

class ThreeDGenLocalModule(GenerationModule):
    """
    Local 3D model generation with Shap-E or Point-E.
    
    📖 WHAT THIS DOES:
    Generates 3D models from text descriptions or images.
    You can export to common formats like .obj, .ply, .glb.
    
    📐 MODELS:
    ┌────────────────────────────────────────────────────────────────────┐
    │ shap-e  │ OpenAI's text-to-3D model (better quality)              │
    │ point-e │ OpenAI's point cloud model (faster)                     │
    └────────────────────────────────────────────────────────────────────┘
    
    📐 KEY PARAMETERS:
    - guidance_scale: How closely to follow the prompt (1-20)
    - num_inference_steps: More steps = better quality (10-100)
    
    🔗 WRAPS: forge_ai/gui/tabs/threed_tab.py → Local3DGen
    """

    INFO = ModuleInfo(
        id="threed_gen_local",
        name="3D Generation (Local)",
        description="Generate 3D models from text/images locally. Requires GPU.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=[],
        provides=["3d_generation", "mesh_generation"],
        conflicts=["threed_gen_api"],  # Can only use one 3D generator
        min_vram_mb=4000,  # Need 4GB VRAM
        requires_gpu=True,
        config_schema={
            "model": {
                "type": "choice",
                "options": ["shap-e", "point-e"],
                "default": "shap-e"
            },
            "guidance_scale": {
                "type": "float",
                "min": 1.0,
                "max": 20.0,
                "default": 15.0  # Higher = follows prompt more closely
            },
            "num_inference_steps": {
                "type": "int",
                "min": 10,
                "max": 100,
                "default": 64  # More steps = better but slower
            },
        },
    )

    def load(self) -> bool:
        """Load the 3D generation model."""
        try:
            from forge_ai.gui.tabs.threed_tab import Local3DGen
            self._addon = Local3DGen(
                model=self.config.get('model', 'shap-e')
            )
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load local 3D gen: {e}")
            return False


class ThreeDGenAPIModule(GenerationModule):
    """
    Cloud 3D model generation via API services.
    
    📖 WHAT THIS DOES:
    Generates 3D models using cloud APIs.
    Good option if you don't have a GPU.
    
    🔗 WRAPS: forge_ai/gui/tabs/threed_tab.py → Cloud3DGen
    """

    INFO = ModuleInfo(
        id="threed_gen_api",
        name="3D Generation (Cloud)",
        description="Generate 3D models via cloud APIs (Replicate, etc). Requires API key.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=[],
        provides=["3d_generation", "mesh_generation"],
        conflicts=["threed_gen_local"],  # Can only use one
        is_cloud_service=True,
        config_schema={
            "service": {
                "type": "choice",
                "options": ["replicate"],
                "default": "replicate"
            },
            "api_key": {
                "type": "secret",
                "default": ""
            },
        },
    )

    def load(self) -> bool:
        """Load the cloud 3D generation client."""
        try:
            from forge_ai.gui.tabs.threed_tab import Cloud3DGen
            self._addon = Cloud3DGen(
                api_key=self.config.get('api_key'),
                service=self.config.get('service', 'replicate')
            )
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load cloud 3D gen: {e}")
            return False


# -----------------------------------------------------------------------------
# 🎯 MOTION TRACKING
# -----------------------------------------------------------------------------

class MotionTrackingModule(Module):
    """
    Motion tracking module for user mimicry.
    
    📖 WHAT THIS DOES:
    Uses your webcam to track body pose, hands, and face.
    The avatar can then mimic your movements in real-time!
    
    📐 TRACKING MODES:
    ┌────────────────────────────────────────────────────────────────────┐
    │ pose     │ Body pose only (33 landmarks)                          │
    │ hands    │ Hand tracking (21 landmarks per hand)                  │
    │ face     │ Facial mesh (468 landmarks)                            │
    │ holistic │ Everything combined (recommended)                      │
    └────────────────────────────────────────────────────────────────────┘
    
    📐 MODEL COMPLEXITY:
    - 0: Lite (fastest, least accurate)
    - 1: Full (balanced)
    - 2: Heavy (most accurate, slowest)
    
    🔗 WRAPS: forge_ai/tools/motion_tracking.py → MotionTracker
    """

    INFO = ModuleInfo(
        id="motion_tracking",
        name="Motion Tracking",
        description="Real-time motion tracking with MediaPipe for gesture mimicry",
        category=ModuleCategory.PERCEPTION,
        version="1.0.0",
        requires=[],
        optional=["avatar"],  # Can control avatar with movements
        provides=[
            "motion_tracking",       # Track body movements
            "gesture_recognition",   # Recognize gestures
            "pose_estimation"],      # Estimate body pose
        config_schema={
            "camera_id": {
                "type": "int",
                "min": 0,
                "max": 10,
                "default": 0  # Default webcam
            },
            "tracking_mode": {
                "type": "choice",
                "options": ["pose", "hands", "face", "holistic"],
                "default": "holistic"  # Track everything
            },
            "model_complexity": {
                "type": "int",
                "min": 0,
                "max": 2,
                "default": 1  # Balanced
            },
        },
    )

    def load(self) -> bool:
        """Load the MediaPipe motion tracker."""
        try:
            from forge_ai.tools.motion_tracking import MotionTracker
            self._instance = MotionTracker(
                camera_id=self.config.get('camera_id', 0),
                tracking_mode=self.config.get('tracking_mode', 'holistic')
            )
            return True
        except Exception as e:
            logger.warning(f"Could not load motion tracking: {e}")
            return False


# -----------------------------------------------------------------------------
# CAMERA MODULE
# -----------------------------------------------------------------------------

class CameraModule(Module):
    """
    Camera module for webcam capture and analysis.
    
    Provides webcam access for capturing images, recording video,
    and analyzing camera feed with AI vision.
    """

    INFO = ModuleInfo(
        id="camera",
        name="Camera",
        description="Webcam capture, recording, and AI-powered visual analysis",
        category=ModuleCategory.PERCEPTION,
        version="1.0.0",
        requires=[],
        optional=["vision"],  # For AI analysis
        provides=["camera_capture", "video_recording"],
        config_schema={
            "camera_id": {
                "type": "int",
                "min": 0,
                "max": 10,
                "default": 0
            },
        },
    )

    def load(self) -> bool:
        """Load camera module."""
        return True  # Loaded on demand in the tab


# -----------------------------------------------------------------------------
# GIF GENERATION MODULE
# -----------------------------------------------------------------------------

class GIFGenModule(Module):
    """
    GIF generation module.
    
    Create animated GIFs from images, videos, or AI generation.
    """

    INFO = ModuleInfo(
        id="gif_gen",
        name="GIF Generation",
        description="Create animated GIFs from images, videos, or AI generation",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=[],
        optional=["image_gen_local", "image_gen_api"],
        provides=["gif_generation"],
        config_schema={},
    )

    def load(self) -> bool:
        """Load GIF generation module."""
        return True


# -----------------------------------------------------------------------------
# VOICE CLONE MODULE
# -----------------------------------------------------------------------------

class VoiceCloneModule(Module):
    """
    Voice cloning module.
    
    Clone voices from audio samples for TTS synthesis.
    """

    INFO = ModuleInfo(
        id="voice_clone",
        name="Voice Cloning",
        description="Clone voices from audio samples for personalized TTS",
        category=ModuleCategory.OUTPUT,
        version="1.0.0",
        requires=[],
        optional=["voice_output"],
        provides=["voice_cloning"],
        config_schema={},
    )

    def load(self) -> bool:
        """Load voice cloning module."""
        return True


# -----------------------------------------------------------------------------
# NOTES MODULE
# -----------------------------------------------------------------------------

class NotesModule(Module):
    """
    Notes module for saving and organizing thoughts.
    
    Provides persistent notes that the AI can access and reference.
    """

    INFO = ModuleInfo(
        id="notes",
        name="Notes",
        description="Save and organize notes that persist across sessions",
        category=ModuleCategory.MEMORY,
        version="1.0.0",
        requires=[],
        provides=["notes_storage"],
        config_schema={},
    )

    def load(self) -> bool:
        """Load notes module."""
        return True


# -----------------------------------------------------------------------------
# SESSIONS MODULE
# -----------------------------------------------------------------------------

class SessionsModule(Module):
    """
    Sessions module for conversation management.
    
    Save, load, and manage multiple conversation sessions.
    """

    INFO = ModuleInfo(
        id="sessions",
        name="Sessions",
        description="Manage multiple conversation sessions with save/load functionality",
        category=ModuleCategory.MEMORY,
        version="1.0.0",
        requires=["memory"],
        provides=["session_management"],
        config_schema={},
    )

    def load(self) -> bool:
        """Load sessions module."""
        return True


# -----------------------------------------------------------------------------
# SCHEDULER MODULE
# -----------------------------------------------------------------------------

class SchedulerModule(Module):
    """
    Scheduler module for timed tasks.
    
    Schedule AI tasks, reminders, and automated actions.
    """

    INFO = ModuleInfo(
        id="scheduler",
        name="Scheduler",
        description="Schedule timed tasks, reminders, and automated AI actions",
        category=ModuleCategory.TOOLS,
        version="1.0.0",
        requires=[],
        optional=["inference", "chat_api"],
        provides=["task_scheduling"],
        config_schema={},
    )

    def load(self) -> bool:
        """Load scheduler module."""
        return True


# -----------------------------------------------------------------------------
# PERSONALITY MODULE
# -----------------------------------------------------------------------------

class PersonalityModule(Module):
    """
    Personality module for AI behavior customization.
    
    Configure and customize the AI's personality, tone, and behavior.
    """

    INFO = ModuleInfo(
        id="personality",
        name="Personality",
        description="Customize AI personality, tone, communication style",
        category=ModuleCategory.CORE,
        version="1.0.0",
        requires=[],
        provides=["personality_config"],
        config_schema={},
    )

    def load(self) -> bool:
        """Load personality module."""
        return True


# -----------------------------------------------------------------------------
# TERMINAL MODULE
# -----------------------------------------------------------------------------

class TerminalModule(Module):
    """
    Terminal module for command execution.
    
    Provides terminal access for running system commands (with safety limits).
    """

    INFO = ModuleInfo(
        id="terminal",
        name="Terminal",
        description="Execute system commands with AI assistance and safety limits",
        category=ModuleCategory.TOOLS,
        version="1.0.0",
        requires=[],
        provides=["command_execution"],
        config_schema={},
    )

    def load(self) -> bool:
        """Load terminal module."""
        return True


# -----------------------------------------------------------------------------
# ANALYTICS MODULE
# -----------------------------------------------------------------------------

class AnalyticsModule(Module):
    """
    Analytics module for usage statistics and insights.
    
    Tracks usage patterns, performance metrics, and provides insights.
    """

    INFO = ModuleInfo(
        id="analytics",
        name="Analytics",
        description="Usage statistics, performance metrics, and AI insights",
        category=ModuleCategory.INTERFACE,
        version="1.0.0",
        requires=[],
        provides=["analytics", "usage_stats"],
        config_schema={},
    )

    def load(self) -> bool:
        return True


# -----------------------------------------------------------------------------
# DASHBOARD MODULE
# -----------------------------------------------------------------------------

class DashboardModule(Module):
    """
    Dashboard module for system overview.
    
    Provides a unified view of AI status, resources, and quick actions.
    """

    INFO = ModuleInfo(
        id="dashboard",
        name="Dashboard",
        description="System overview with status, resources, and quick actions",
        category=ModuleCategory.INTERFACE,
        version="1.0.0",
        requires=[],
        provides=["dashboard_ui"],
        config_schema={},
    )

    def load(self) -> bool:
        return True


# -----------------------------------------------------------------------------
# EXAMPLES MODULE
# -----------------------------------------------------------------------------

class ExamplesModule(Module):
    """
    Examples module for prompt templates and tutorials.
    
    Provides example prompts, templates, and usage guides.
    """

    INFO = ModuleInfo(
        id="examples",
        name="Examples",
        description="Prompt templates, examples, and usage tutorials",
        category=ModuleCategory.INTERFACE,
        version="1.0.0",
        requires=[],
        provides=["examples", "templates"],
        config_schema={},
    )

    def load(self) -> bool:
        return True


# -----------------------------------------------------------------------------
# INSTRUCTIONS MODULE
# -----------------------------------------------------------------------------

class InstructionsModule(Module):
    """
    Instructions module for system prompts.
    
    Manage system instructions that define AI behavior.
    """

    INFO = ModuleInfo(
        id="instructions",
        name="Instructions",
        description="System prompts and behavior instructions for the AI",
        category=ModuleCategory.CORE,
        version="1.0.0",
        requires=[],
        provides=["system_prompts"],
        config_schema={},
    )

    def load(self) -> bool:
        return True


# -----------------------------------------------------------------------------
# LOGS MODULE
# -----------------------------------------------------------------------------

class LogsModule(Module):
    """
    Logs module for viewing system logs.
    
    View and search application logs for debugging.
    """

    INFO = ModuleInfo(
        id="logs",
        name="Logs",
        description="View and search system logs for debugging",
        category=ModuleCategory.INTERFACE,
        version="1.0.0",
        requires=[],
        provides=["log_viewer"],
        config_schema={},
    )

    def load(self) -> bool:
        return True


# -----------------------------------------------------------------------------
# MODEL ROUTER MODULE (different from tool_router)
# -----------------------------------------------------------------------------

class ModelRouterModule(Module):
    """
    Model router for specialized model assignment.
    
    Route different tasks to specialized AI models.
    """

    INFO = ModuleInfo(
        id="model_router",
        name="Model Router",
        description="Route tasks to specialized models (code, vision, etc.)",
        category=ModuleCategory.CORE,
        version="1.0.0",
        requires=[],
        optional=["model", "chat_api"],
        provides=["model_routing"],
        config_schema={},
    )

    def load(self) -> bool:
        return True


# -----------------------------------------------------------------------------
# SCALING MODULE
# -----------------------------------------------------------------------------

class ScalingModule(Module):
    """
    Scaling module for model size management.
    
    Grow or shrink models, manage model variants.
    """

    INFO = ModuleInfo(
        id="scaling",
        name="Model Scaling",
        description="Grow/shrink models, manage size variants",
        category=ModuleCategory.CORE,
        version="1.0.0",
        requires=["model"],
        provides=["model_scaling"],
        config_schema={},
    )

    def load(self) -> bool:
        return True


# -----------------------------------------------------------------------------
# GAME AI MODULE
# -----------------------------------------------------------------------------

class GameAIModule(Module):
    """
    Game AI module for gaming assistance.
    
    AI features for gaming: strategy, tactics, overlay assistance.
    """

    INFO = ModuleInfo(
        id="game_ai",
        name="Game AI",
        description="Gaming AI assistant with strategy and tactical advice",
        category=ModuleCategory.TOOLS,
        version="1.0.0",
        requires=[],
        optional=["vision", "voice_output"],
        provides=["game_assistance"],
        config_schema={},
    )

    def load(self) -> bool:
        return True


# -----------------------------------------------------------------------------
# ROBOT CONTROL MODULE
# -----------------------------------------------------------------------------

class RobotControlModule(Module):
    """
    Robot control module for hardware integration.
    
    Control robots, servos, and hardware via AI commands.
    """

    INFO = ModuleInfo(
        id="robot_control",
        name="Robot Control",
        description="Control robots and hardware with AI commands",
        category=ModuleCategory.TOOLS,
        version="1.0.0",
        requires=[],
        provides=["robot_control", "hardware_interface"],
        config_schema={},
    )

    def load(self) -> bool:
        return True


# -----------------------------------------------------------------------------
# WORKSPACE MODULE
# -----------------------------------------------------------------------------

class WorkspaceModule(Module):
    """
    Workspace module for project management.
    
    Manage AI projects, training data, and configurations.
    """

    INFO = ModuleInfo(
        id="workspace",
        name="Workspace",
        description="Project management for training data and configurations",
        category=ModuleCategory.INTERFACE,
        version="1.0.0",
        requires=[],
        provides=["workspace_management"],
        config_schema={},
    )

    def load(self) -> bool:
        return True


# -----------------------------------------------------------------------------
# HUGGINGFACE MODULE
# -----------------------------------------------------------------------------

class HuggingFaceModule(Module):
    """
    HuggingFace integration module.
    
    Load and use models from HuggingFace Hub.
    """

    INFO = ModuleInfo(
        id="huggingface",
        name="HuggingFace",
        description="Load and use models from HuggingFace Hub",
        category=ModuleCategory.CORE,
        version="1.0.0",
        requires=[],
        provides=["hf_models", "model_hub"],
        config_schema={},
    )

    def load(self) -> bool:
        return True


# =============================================================================
# 📚 MODULE REGISTRY
# =============================================================================
# This is the master catalog of all available modules.
# The ModuleManager uses this to know what modules exist.
#
# 📐 STRUCTURE:
# ┌────────────────────────────────────────────────────────────────────────┐
# │  'module_id': ModuleClass,                                             │
# │                                                                        │
# │  Example:                                                              │
# │  'image_gen_local': ImageGenLocalModule,  # Local Stable Diffusion    │
# │  'image_gen_api': ImageGenAPIModule,      # Cloud DALL-E/Replicate    │
# └────────────────────────────────────────────────────────────────────────┘
#
# 📐 CATEGORIES:
# - Core: model, tokenizer, training, inference, chat_api, gguf_loader
# - Memory: memory, embedding_local, embedding_api
# - Perception: voice_input, vision, motion_tracking
# - Output: voice_output, avatar
# - Generation: image, code, video, audio, 3D (local and API variants)
# - Tools: web_tools, file_tools, tool_router
# - Network: api_server, network
# - Interface: gui

MODULE_REGISTRY: Dict[str, Type[Module]] = {
    # ─────────────────────────────────────────────────────────────────────
    # Core modules - The foundation of ForgeAI
    # ─────────────────────────────────────────────────────────────────────
    'model': ModelModule,           # The neural network brain
    'tokenizer': TokenizerModule,   # Text ↔ numbers converter
    'training': TrainingModule,     # Teaches the model
    'inference': InferenceModule,   # Generates text
    'chat_api': ChatAPIModule,      # Cloud chat (Ollama, GPT, Claude)
    'gguf_loader': GGUFLoaderModule,# Load llama.cpp models

    # ─────────────────────────────────────────────────────────────────────
    # Memory modules - Remember things
    # ─────────────────────────────────────────────────────────────────────
    'memory': MemoryModule,                 # Conversation storage
    'embedding_local': EmbeddingLocalModule,# Local vector embeddings
    'embedding_api': EmbeddingAPIModule,    # Cloud embeddings
    'notes': NotesModule,                   # Persistent notes
    'sessions': SessionsModule,             # Conversation sessions

    # ─────────────────────────────────────────────────────────────────────
    # Perception modules - See and hear
    # ─────────────────────────────────────────────────────────────────────
    'voice_input': VoiceInputModule,        # Speech-to-text
    'vision': VisionModule,                 # Image/screen analysis
    'motion_tracking': MotionTrackingModule,# Body/hand/face tracking
    'camera': CameraModule,                 # Webcam capture

    # ─────────────────────────────────────────────────────────────────────
    # Output modules - Speak and show
    # ─────────────────────────────────────────────────────────────────────
    'voice_output': VoiceOutputModule,  # Text-to-speech
    'avatar': AvatarModule,             # Visual character
    'voice_clone': VoiceCloneModule,    # Voice cloning

    # ─────────────────────────────────────────────────────────────────────
    # Generation modules - Create content
    # (LOCAL and API versions conflict - only use one at a time!)
    # ─────────────────────────────────────────────────────────────────────
    'image_gen_local': ImageGenLocalModule,   # Stable Diffusion
    'image_gen_api': ImageGenAPIModule,       # DALL-E / Replicate
    'code_gen_local': CodeGenLocalModule,     # Local Forge code
    'code_gen_api': CodeGenAPIModule,         # GPT-4 code
    'video_gen_local': VideoGenLocalModule,   # AnimateDiff
    'video_gen_api': VideoGenAPIModule,       # Replicate video
    'audio_gen_local': AudioGenLocalModule,   # pyttsx3 / edge-tts
    'audio_gen_api': AudioGenAPIModule,       # ElevenLabs / MusicGen
    'threed_gen_local': ThreeDGenLocalModule, # Shap-E / Point-E
    'threed_gen_api': ThreeDGenAPIModule,     # Cloud 3D
    'gif_gen': GIFGenModule,                  # GIF creation

    # ─────────────────────────────────────────────────────────────────────
    # Tool modules - Interact with the world
    # ─────────────────────────────────────────────────────────────────────
    'web_tools': WebToolsModule,        # Web search and fetch
    'file_tools': FileToolsModule,      # File read/write
    'tool_router': ToolRouterModule,    # Route to specialized models
    'scheduler': SchedulerModule,       # Timed tasks/reminders
    'terminal': TerminalModule,         # Command execution
    'game_ai': GameAIModule,            # Gaming AI assistant
    'robot_control': RobotControlModule,# Robot/hardware control

    # ─────────────────────────────────────────────────────────────────────
    # Network modules - Connect devices
    # ─────────────────────────────────────────────────────────────────────
    'api_server': APIServerModule,  # REST API server
    'network': NetworkModule,       # Multi-device networking

    # ─────────────────────────────────────────────────────────────────────
    # Interface modules - User interaction
    # ─────────────────────────────────────────────────────────────────────
    'gui': GUIModule,               # PyQt5 graphical interface
    'personality': PersonalityModule,  # AI personality customization
    'analytics': AnalyticsModule,   # Usage stats and insights
    'dashboard': DashboardModule,   # System overview
    'examples': ExamplesModule,     # Prompt templates
    'instructions': InstructionsModule,  # System prompts
    'logs': LogsModule,             # Log viewer
    'model_router': ModelRouterModule,   # Specialized model routing
    'scaling': ScalingModule,       # Model grow/shrink
    'workspace': WorkspaceModule,   # Project management
    'huggingface': HuggingFaceModule,    # HuggingFace Hub integration
}


# =============================================================================
# 🔧 HELPER FUNCTIONS
# =============================================================================
# These functions make it easy to work with the module registry.

def register_all(manager: ModuleManager):
    """
    Register all built-in modules with a manager.
    
    📖 USAGE:
    ```python
    manager = ModuleManager()
    register_all(manager)  # Now manager knows about all modules
    ```
    """
    for module_class in MODULE_REGISTRY.values():
        manager.register(module_class)


@lru_cache(maxsize=128)
def get_module(module_id: str) -> Optional[Type[Module]]:
    """
    Get a module class by ID.
    
    📖 EXAMPLE:
    ```python
    ImageClass = get_module('image_gen_local')  # Returns ImageGenLocalModule
    ```
    """
    return MODULE_REGISTRY.get(module_id)


@lru_cache(maxsize=32)
def list_modules() -> List[ModuleInfo]:
    """
    List all available modules.
    
    📖 RETURNS:
    List of ModuleInfo objects with id, name, description, etc.
    """
    return [cls.get_info() for cls in MODULE_REGISTRY.values()]


def list_by_category(category: ModuleCategory) -> List[ModuleInfo]:
    """
    List modules by category.
    
    📖 EXAMPLE:
    ```python
    gen_modules = list_by_category(ModuleCategory.GENERATION)
    ```
    """
    return [
        cls.get_info() for cls in MODULE_REGISTRY.values()
        if cls.get_info().category == category
    ]


def list_local_modules() -> List[ModuleInfo]:
    """
    List modules that run 100% locally (no cloud/internet required).
    Great for privacy-conscious users or offline use!
    """
    return [
        cls.get_info() for cls in MODULE_REGISTRY.values()
        if not cls.get_info().is_cloud_service
    ]


def list_cloud_modules() -> List[ModuleInfo]:
    """List modules that require cloud services and API keys."""
    return [
        cls.get_info() for cls in MODULE_REGISTRY.values()
        if cls.get_info().is_cloud_service
    ]
