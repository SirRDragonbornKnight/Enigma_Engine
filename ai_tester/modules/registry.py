"""
AI Tester Module Registry
======================

Central registry of all available modules.
Auto-discovers and registers built-in modules.
"""

from functools import lru_cache
import logging
import os
from typing import Dict, List, Optional, Type, Any

from .manager import Module, ModuleInfo, ModuleCategory, ModuleManager

logger = logging.getLogger(__name__)


# =============================================================================
# Built-in Module Definitions
# =============================================================================

class ModelModule(Module):
    """Core transformer model module."""

    INFO = ModuleInfo(
        id="model",
        name="Enigma Model",
        description="Core transformer language model with RoPE, RMSNorm, SwiGLU, GQA",
        category=ModuleCategory.CORE,
        version="2.0.0",
        requires=[],
        provides=[
            "language_model",
            "model_embeddings"],
        config_schema={
            "size": {
                "type": "choice",
                "options": [
                    "nano",
                    "micro",
                    "tiny",
                    "small",
                    "medium",
                    "large",
                    "xl",
                    "xxl",
                    "titan"],
                "default": "small"},
            "vocab_size": {
                "type": "int",
                        "min": 1000,
                        "max": 500000,
                        "default": 8000},
            "device": {
                "type": "choice",
                "options": [
                    "auto",
                    "cuda",
                    "cpu",
                    "mps"],
                "default": "auto"},
            "dtype": {
                "type": "choice",
                "options": [
                    "float32",
                    "float16",
                    "bfloat16"],
                "default": "float32"},
        },
        min_ram_mb=512,
    )

    def load(self) -> bool:
        from enigma.core.model import create_model

        size = self.config.get('size', 'small')
        vocab_size = self.config.get('vocab_size', 8000)

        self._instance = create_model(size, vocab_size=vocab_size)
        return self._instance is not None

    def unload(self) -> bool:
        if self._instance is not None:
            del self._instance
            self._instance = None
        return True


class TokenizerModule(Module):
    """Tokenizer module - converts text to/from tokens."""

    INFO = ModuleInfo(
        id="tokenizer",
        name="Tokenizer",
        description="Text tokenization with BPE, character, or custom vocabularies",
        category=ModuleCategory.CORE,
        version="2.0.0",
        requires=[],
        provides=[
            "tokenization",
            "vocabulary"],
        config_schema={
            "type": {
                "type": "choice",
                "options": [
                    "auto",
                    "bpe",
                    "character",
                    "simple"],
                "default": "auto"},
            "vocab_size": {
                "type": "int",
                        "min": 100,
                        "max": 500000,
                        "default": 8000},
        },
    )

    def load(self) -> bool:
        from enigma.core.tokenizer import get_tokenizer

        tok_type = self.config.get('type', 'auto')
        self._instance = get_tokenizer(tok_type)
        return self._instance is not None


class TrainingModule(Module):
    """Training module - trains models on data."""

    INFO = ModuleInfo(
        id="training",
        name="Training System",
        description="Production-grade training with AMP, gradient accumulation, distributed support",
        category=ModuleCategory.CORE,
        version="2.0.0",
        requires=[
            "model",
            "tokenizer"],
        provides=[
            "model_training",
            "fine_tuning"],
        supports_distributed=True,
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
        from enigma.core.training import Trainer, TrainingConfig
        self._trainer_class = Trainer
        self._config_class = TrainingConfig
        return True

    def get_interface(self):
        return {
            'Trainer': self._trainer_class,
            'TrainingConfig': self._config_class,
        }


class InferenceModule(Module):
    """Inference module - generates text from models."""

    INFO = ModuleInfo(
        id="inference",
        name="Inference Engine",
        description="High-performance text generation with streaming, batching, chat",
        category=ModuleCategory.CORE,
        version="2.0.0",
        requires=["model", "tokenizer"],
        provides=["text_generation", "streaming", "chat"],
        config_schema={
            "max_length": {"type": "int", "min": 1, "max": 32768, "default": 2048},
            "temperature": {"type": "float", "min": 0.0, "max": 2.0, "default": 0.8},
            "top_k": {"type": "int", "min": 0, "max": 1000, "default": 50},
            "top_p": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.9},
        },
    )

    def load(self) -> bool:
        from enigma.core.inference import AITesterEngine
        self._engine_class = AITesterEngine
        return True


class ChatAPIModule(Module):
    """Cloud/API chat via OpenAI, Anthropic, Ollama, or other providers.
    
    This module provides cloud or API-based chat, allowing you to use powerful
    AI models (GPT-4, Claude, Llama via Ollama, etc.) without training your own.
    Perfect for Raspberry Pi or low-power devices.
    
    FREE OPTIONS:
    - Ollama: Run Llama, Mistral locally (FREE, no API key needed)
    - Google Gemini: Has a free tier
    - GitHub Copilot: If you have Copilot subscription
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
    """GGUF model loader module - load llama.cpp compatible models."""

    INFO = ModuleInfo(
        id="gguf_loader",
        name="GGUF Model Loader",
        description="Load and run GGUF format models (llama.cpp compatible)",
        category=ModuleCategory.CORE,
        version="1.0.0",
        requires=[],
        conflicts=["model", "inference"],  # Can't use both GGUF and Enigma model
        provides=["text_generation", "completion", "gguf_support"],
        config_schema={
            "model_path": {"type": "string", "default": ""},
            "n_ctx": {"type": "int", "min": 512, "max": 32768, "default": 2048},
            "n_gpu_layers": {"type": "int", "min": 0, "max": 999, "default": 0},
            "n_threads": {"type": "int", "min": 1, "max": 32, "default": 4},
        },
    )

    def load(self) -> bool:
        try:
            from enigma.core.gguf_loader import GGUFModel
            model_path = self.config.get('model_path', '')
            if not model_path:
                logger.warning("No GGUF model path specified")
                return False
            
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
        if self._instance:
            self._instance.unload()
            self._instance = None
        return True


class MemoryModule(Module):
    """Memory module - conversation and knowledge storage."""

    INFO = ModuleInfo(
        id="memory",
        name="Memory System",
        description="Conversation history, vector search, long-term memory",
        category=ModuleCategory.MEMORY,
        version="1.0.0",
        requires=[],
        optional=["model"],  # For embeddings
        provides=["conversation_storage", "vector_search", "knowledge_base"],
        config_schema={
            "backend": {"type": "choice", "options": ["json", "sqlite", "vector"], "default": "json"},
            "max_conversations": {"type": "int", "min": 1, "max": 100000, "default": 1000},
        },
    )

    def load(self) -> bool:
        from enigma.memory.manager import ConversationManager
        self._instance = ConversationManager()
        return True


class VoiceInputModule(Module):
    """Voice input module - speech to text."""

    INFO = ModuleInfo(
        id="voice_input",
        name="Voice Input (STT)",
        description="Speech-to-text for voice commands and dictation",
        category=ModuleCategory.PERCEPTION,
        version="1.0.0",
        requires=[],
        provides=[
            "speech_recognition",
            "voice_commands"],
        config_schema={
            "engine": {
                "type": "choice",
                "options": [
                    "system",
                    "whisper",
                    "vosk"],
                "default": "system"},
            "language": {
                "type": "string",
                        "default": "en-US"},
        },
    )

    def load(self) -> bool:
        try:
            from enigma.voice.stt_simple import transcribe_from_mic, transcribe_from_file
            # Wrap functions in a simple class for consistency
            class STTWrapper:
                def __init__(self):
                    self.transcribe_from_mic = transcribe_from_mic
                    self.transcribe_from_file = transcribe_from_file
                
                def listen(self, timeout=8):
                    return transcribe_from_mic(timeout)
            
            self._instance = STTWrapper()
            return True
        except Exception as e:
            logger.warning(f"Could not load voice input: {e}")
            return False


class VoiceOutputModule(Module):
    """Voice output module - text to speech."""

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
            from enigma.voice.tts_simple import speak
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
    """Vision module - image processing and analysis."""

    INFO = ModuleInfo(
        id="vision",
        name="Vision System",
        description="Image capture, OCR, object detection, visual understanding",
        category=ModuleCategory.PERCEPTION,
        version="1.0.0",
        requires=[],
        optional=["model"],
        provides=[
            "image_capture",
            "ocr",
            "object_detection"],
        config_schema={
            "camera_id": {
                "type": "int",
                "min": 0,
                "max": 10,
                "default": 0},
            "resolution": {
                "type": "choice",
                "options": [
                        "480p",
                        "720p",
                        "1080p"],
                "default": "720p"},
        },
    )

    def load(self) -> bool:
        try:
            from enigma.tools.vision import ScreenVision
            self._instance = ScreenVision()
            return True
        except Exception as e:
            logger.warning(f"Could not load vision: {e}")
            return False


class AvatarModule(Module):
    """Avatar module - visual AI representation."""

    INFO = ModuleInfo(
        id="avatar",
        name="Avatar System",
        description="Visual representation, expressions, animations",
        category=ModuleCategory.OUTPUT,
        version="1.0.0",
        requires=[],
        optional=["voice_output"],
        provides=["visual_avatar", "expressions", "lip_sync"],
        config_schema={
            "style": {"type": "choice", "options": ["2d", "3d", "animated"], "default": "2d"},
            "character": {"type": "string", "default": "default"},
        },
    )

    def load(self) -> bool:
        try:
            from enigma.avatar.controller import AvatarController
            self._instance = AvatarController()
            return True
        except Exception as e:
            logger.warning(f"Could not load avatar: {e}")
            return False


class WebToolsModule(Module):
    """Web tools module - internet access."""

    INFO = ModuleInfo(
        id="web_tools",
        name="Web Tools",
        description="Web search, page fetching, API access",
        category=ModuleCategory.TOOLS,
        version="1.0.0",
        requires=[],
        provides=["web_search", "url_fetch", "api_access"],
        config_schema={
            "allow_external": {"type": "bool", "default": True},
            "timeout": {"type": "int", "min": 1, "max": 300, "default": 30},
        },
    )

    def load(self) -> bool:
        try:
            from enigma.tools.web_tools import WebTools
            self._instance = WebTools()
            return True
        except Exception:
            return True  # Optional, don't fail


class FileToolsModule(Module):
    """File tools module - file system access."""

    INFO = ModuleInfo(
        id="file_tools",
        name="File Tools",
        description="Read, write, search files and directories",
        category=ModuleCategory.TOOLS,
        version="1.0.0",
        requires=[],
        provides=["file_read", "file_write", "file_search"],
        config_schema={
            "allowed_paths": {"type": "list", "default": []},
            "max_file_size_mb": {"type": "int", "min": 1, "max": 1000, "default": 100},
        },
    )

    def load(self) -> bool:
        try:
            from enigma.tools.file_tools import FileTools
            self._instance = FileTools()
            return True
        except Exception:
            return True


class ToolRouterModule(Module):
    """Tool router module - specialized model routing."""

    INFO = ModuleInfo(
        id="tool_router",
        name="Tool Router",
        description="Route requests to specialized models (intent classification, vision, code)",
        category=ModuleCategory.TOOLS,
        version="1.0.0",
        requires=["tokenizer"],
        optional=["model"],
        provides=["intent_classification", "specialized_routing", "model_routing"],
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
        try:
            from enigma.core.tool_router import get_router
            use_specialized = self.config.get('use_specialized', True)
            self._instance = get_router(use_specialized=use_specialized)
            logger.info(f"Tool router loaded (specialized: {use_specialized})")
            return True
        except Exception as e:
            logger.warning(f"Could not load tool router: {e}")
            return False

    def unload(self) -> bool:
        if self._instance is not None:
            # Clear any cached models
            if hasattr(self._instance, '_specialized_models'):
                self._instance._specialized_models.clear()
            if hasattr(self._instance, '_model_cache'):
                self._instance._model_cache.clear()
            self._instance = None
        return True


class APIServerModule(Module):
    """API server module - REST API interface."""

    INFO = ModuleInfo(
        id="api_server",
        name="API Server",
        description="REST API for remote access and integrations",
        category=ModuleCategory.INTERFACE,
        version="1.0.0",
        requires=["inference"],
        provides=["rest_api", "remote_access"],
        config_schema={
            "host": {"type": "string", "default": "127.0.0.1"},
            "port": {"type": "int", "min": 1, "max": 65535, "default": 5000},
            "auth_enabled": {"type": "bool", "default": False},
        },
    )

    def load(self) -> bool:
        from enigma.comms.api_server import create_app
        self._app_factory = create_app
        return True


class NetworkModule(Module):
    """Network module - multi-device communication."""

    INFO = ModuleInfo(
        id="network",
        name="Network System",
        description="Multi-device communication, distributed inference, model sharing",
        category=ModuleCategory.NETWORK,
        version="1.0.0",
        requires=[],
        optional=[
            "model",
            "inference"],
        provides=[
            "multi_device",
            "distributed_inference",
            "model_sync"],
        supports_distributed=True,
        config_schema={
            "role": {
                "type": "choice",
                "options": [
                        "standalone",
                        "server",
                        "client",
                        "peer"],
                "default": "standalone"},
            "discovery": {
                "type": "bool",
                "default": True},
        },
    )

    def load(self) -> bool:
        try:
            from enigma.comms.network import AITesterNode
            self._instance = AITesterNode()
            return True
        except Exception:
            return True


class GUIModule(Module):
    """GUI module - graphical interface."""

    INFO = ModuleInfo(
        id="gui",
        name="Graphical Interface",
        description="PyQt5-based GUI with chat, training, modules management",
        category=ModuleCategory.INTERFACE,
        version="2.0.0",
        requires=[],
        optional=[
            "model",
            "tokenizer",
            "inference",
            "training",
            "memory",
            "voice_input",
            "voice_output",
            "vision",
            "avatar"],
        provides=[
            "graphical_interface",
            "chat_ui",
            "training_ui",
            "module_management"],
        config_schema={
            "theme": {
                "type": "choice",
                "options": [
                        "dark",
                        "light",
                        "system"],
                "default": "dark"},
            "window_size": {
                "type": "choice",
                "options": [
                    "small",
                    "medium",
                    "large",
                    "fullscreen"],
                "default": "medium"},
        },
    )

    def load(self) -> bool:
        # GUI is loaded on demand
        return True


# =============================================================================
# AI Generation Modules (Addons integrated into module system)
# =============================================================================

class GenerationModule(Module):
    """Base class for generation modules that wrap addons."""

    def __init__(self, manager, config=None):
        super().__init__(manager, config)
        self._addon = None

    def unload(self) -> bool:
        if self._addon:
            self._addon.unload()
            self._addon = None
        return True

    def generate(self, prompt: str, **kwargs):
        """Generate content using the wrapped addon."""
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


class ImageGenLocalModule(GenerationModule):
    """Local image generation with Stable Diffusion."""

    INFO = ModuleInfo(
        id="image_gen_local",
        name="Image Generation (Local)",
        description="Generate images locally with Stable Diffusion. Requires GPU with 8GB+ VRAM.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=[],
        provides=["image_generation"],
        min_vram_mb=6000,
        requires_gpu=True,
        config_schema={
            "model": {
                "type": "choice",
                "options": [
                    "sd-2.1",
                    "sdxl",
                    "sdxl-turbo"],
                "default": "sd-2.1"},
            "steps": {
                "type": "int",
                "min": 1,
                "max": 100,
                "default": 30},
        },
    )

    def load(self) -> bool:
        try:
            from enigma.gui.tabs.image_tab import StableDiffusionLocal
            self._addon = StableDiffusionLocal()
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load local image gen: {e}")
            return False


class ImageGenAPIModule(GenerationModule):
    """Cloud image generation via APIs."""

    INFO = ModuleInfo(
        id="image_gen_api",
        name="Image Generation (Cloud)",
        description="Generate images via OpenAI DALL-E or Replicate (SDXL, Flux). Requires API key.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=[],
        provides=["image_generation"],
        is_cloud_service=True,
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
                from enigma.gui.tabs.image_tab import OpenAIImage
                self._addon = OpenAIImage(api_key=self.config.get('api_key'))
            else:
                from enigma.gui.tabs.image_tab import ReplicateImage
                self._addon = ReplicateImage(api_key=self.config.get('api_key'))
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load cloud image gen: {e}")
            return False


class CodeGenLocalModule(GenerationModule):
    """Local code generation using Enigma model."""

    INFO = ModuleInfo(
        id="code_gen_local",
        name="Code Generation (Local)",
        description="Generate code using your trained Enigma model. Free and private.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=["model", "tokenizer", "inference"],
        provides=["code_generation"],
        config_schema={
            "model_name": {"type": "string", "default": "sacrifice"},
            "temperature": {"type": "float", "min": 0.1, "max": 1.5, "default": 0.3},
        },
    )

    def load(self) -> bool:
        try:
            from enigma.gui.tabs.code_tab import AITesterCode
            self._addon = AITesterCode(model_name=self.config.get('model_name', 'sacrifice'))
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load local code gen: {e}")
            return False


class CodeGenAPIModule(GenerationModule):
    """Cloud code generation via OpenAI."""

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
                    "gpt-4",
                    "gpt-4-turbo",
                    "gpt-3.5-turbo"],
                "default": "gpt-4"},
            "api_key": {
                "type": "secret",
                "default": ""},
        },
    )

    def load(self) -> bool:
        try:
            from enigma.gui.tabs.code_tab import OpenAICode
            self._addon = OpenAICode(
                api_key=self.config.get('api_key'),
                model=self.config.get(
                    'model',
                    'gpt-4'))
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load cloud code gen: {e}")
            return False


class VideoGenLocalModule(GenerationModule):
    """Local video generation with AnimateDiff."""

    INFO = ModuleInfo(
        id="video_gen_local",
        name="Video Generation (Local)",
        description="Generate videos locally with AnimateDiff. Requires powerful GPU.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=[],
        provides=["video_generation"],
        min_vram_mb=12000,
        requires_gpu=True,
        config_schema={
            "fps": {"type": "int", "min": 4, "max": 30, "default": 8},
            "duration": {"type": "float", "min": 1, "max": 10, "default": 4},
        },
    )

    def load(self) -> bool:
        try:
            from enigma.gui.tabs.video_tab import LocalVideo
            self._addon = LocalVideo()
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load local video gen: {e}")
            return False


class VideoGenAPIModule(GenerationModule):
    """Cloud video generation via Replicate."""

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
        try:
            from enigma.gui.tabs.video_tab import ReplicateVideo
            self._addon = ReplicateVideo(api_key=self.config.get('api_key'))
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load cloud video gen: {e}")
            return False


class AudioGenLocalModule(GenerationModule):
    """Local audio/TTS generation."""

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
            from enigma.gui.tabs.audio_tab import LocalTTS
            self._addon = LocalTTS()
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load local audio gen: {e}")
            return False


class AudioGenAPIModule(GenerationModule):
    """Cloud audio generation via ElevenLabs/Replicate."""

    INFO = ModuleInfo(
        id="audio_gen_api",
        name="Audio/TTS (Cloud)",
        description="Premium TTS via ElevenLabs or music via MusicGen. Requires API key.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=[],
        provides=[
            "audio_generation",
            "text_to_speech",
            "music_generation"],
        is_cloud_service=True,
        config_schema={
            "provider": {
                "type": "choice",
                "options": [
                    "elevenlabs",
                    "replicate"],
                "default": "elevenlabs"},
            "api_key": {
                "type": "secret",
                        "default": ""},
            "voice": {
                "type": "string",
                "default": "Rachel"},
        },
    )

    def load(self) -> bool:
        try:
            provider = self.config.get('provider', 'elevenlabs')
            if provider == 'elevenlabs':
                from enigma.gui.tabs.audio_tab import ElevenLabsTTS
                self._addon = ElevenLabsTTS(api_key=self.config.get('api_key'))
            else:
                from enigma.gui.tabs.audio_tab import ReplicateAudio
                self._addon = ReplicateAudio(api_key=self.config.get('api_key'))
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load cloud audio gen: {e}")
            return False


class EmbeddingLocalModule(GenerationModule):
    """Local embedding generation with sentence-transformers."""

    INFO = ModuleInfo(
        id="embedding_local",
        name="Embeddings (Local)",
        description="Generate vector embeddings locally for semantic search. Free and private.",
        category=ModuleCategory.MEMORY,
        version="1.0.0",
        requires=[],
        provides=[
            "embeddings",
            "semantic_search"],
        config_schema={
            "model": {
                "type": "choice",
                "options": [
                    "all-MiniLM-L6-v2",
                    "all-mpnet-base-v2"],
                "default": "all-MiniLM-L6-v2"},
        },
    )

    def load(self) -> bool:
        try:
            from enigma.gui.tabs.embeddings_tab import LocalEmbedding
            self._addon = LocalEmbedding(model_name=self.config.get('model', 'all-MiniLM-L6-v2'))
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load local embeddings: {e}")
            return False


class EmbeddingAPIModule(GenerationModule):
    """Cloud embeddings via OpenAI."""

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
                    "text-embedding-3-small",
                    "text-embedding-3-large"],
                "default": "text-embedding-3-small"},
            "api_key": {
                "type": "secret",
                        "default": ""},
        },
    )

    def load(self) -> bool:
        try:
            from enigma.gui.tabs.embeddings_tab import OpenAIEmbedding
            self._addon = OpenAIEmbedding(
                api_key=self.config.get('api_key'),
                model=self.config.get('model'))
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load cloud embeddings: {e}")
            return False


class ThreeDGenLocalModule(GenerationModule):
    """Local 3D model generation with Shap-E or Point-E."""

    INFO = ModuleInfo(
        id="threed_gen_local",
        name="3D Generation (Local)",
        description="Generate 3D models from text/images locally. Requires GPU.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=[],
        provides=["3d_generation", "mesh_generation"],
        conflicts=["threed_gen_api"],
        min_vram_mb=4000,
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
                "default": 15.0
            },
            "num_inference_steps": {
                "type": "int",
                "min": 10,
                "max": 100,
                "default": 64
            },
        },
    )

    def load(self) -> bool:
        try:
            from enigma.gui.tabs.threed_tab import Local3DGen
            self._addon = Local3DGen(
                model=self.config.get('model', 'shap-e')
            )
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load local 3D gen: {e}")
            return False


class ThreeDGenAPIModule(GenerationModule):
    """Cloud 3D model generation via API services."""

    INFO = ModuleInfo(
        id="threed_gen_api",
        name="3D Generation (Cloud)",
        description="Generate 3D models via cloud APIs (Replicate, etc). Requires API key.",
        category=ModuleCategory.GENERATION,
        version="1.0.0",
        requires=[],
        provides=["3d_generation", "mesh_generation"],
        conflicts=["threed_gen_local"],
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
        try:
            from enigma.gui.tabs.threed_tab import Cloud3DGen
            self._addon = Cloud3DGen(
                api_key=self.config.get('api_key'),
                service=self.config.get('service', 'replicate')
            )
            return self._addon.load()
        except Exception as e:
            logger.warning(f"Could not load cloud 3D gen: {e}")
            return False


class MotionTrackingModule(Module):
    """Motion tracking module for user mimicry."""

    INFO = ModuleInfo(
        id="motion_tracking",
        name="Motion Tracking",
        description="Real-time motion tracking with MediaPipe for gesture mimicry",
        category=ModuleCategory.PERCEPTION,
        version="1.0.0",
        requires=[],
        optional=["avatar"],
        provides=["motion_tracking", "gesture_recognition", "pose_estimation"],
        config_schema={
            "camera_id": {
                "type": "int",
                "min": 0,
                "max": 10,
                "default": 0
            },
            "tracking_mode": {
                "type": "choice",
                "options": ["pose", "hands", "face", "holistic"],
                "default": "holistic"
            },
            "model_complexity": {
                "type": "int",
                "min": 0,
                "max": 2,
                "default": 1
            },
        },
    )

    def load(self) -> bool:
        try:
            from enigma.tools.motion_tracking import MotionTracker
            self._instance = MotionTracker(
                camera_id=self.config.get('camera_id', 0),
                tracking_mode=self.config.get('tracking_mode', 'holistic')
            )
            return True
        except Exception as e:
            logger.warning(f"Could not load motion tracking: {e}")
            return False


# =============================================================================
# Module Registry
# =============================================================================

MODULE_REGISTRY: Dict[str, Type[Module]] = {
    # Core
    'model': ModelModule,
    'tokenizer': TokenizerModule,
    'training': TrainingModule,
    'inference': InferenceModule,
    'chat_api': ChatAPIModule,
    'gguf_loader': GGUFLoaderModule,

    # Memory
    'memory': MemoryModule,
    'embedding_local': EmbeddingLocalModule,
    'embedding_api': EmbeddingAPIModule,

    # Perception
    'voice_input': VoiceInputModule,
    'vision': VisionModule,
    'motion_tracking': MotionTrackingModule,

    # Output
    'voice_output': VoiceOutputModule,
    'avatar': AvatarModule,

    # Generation (AI Capabilities)
    'image_gen_local': ImageGenLocalModule,
    'image_gen_api': ImageGenAPIModule,
    'code_gen_local': CodeGenLocalModule,
    'code_gen_api': CodeGenAPIModule,
    'video_gen_local': VideoGenLocalModule,
    'video_gen_api': VideoGenAPIModule,
    'audio_gen_local': AudioGenLocalModule,
    'audio_gen_api': AudioGenAPIModule,
    'threed_gen_local': ThreeDGenLocalModule,
    'threed_gen_api': ThreeDGenAPIModule,

    # Tools
    'web_tools': WebToolsModule,
    'file_tools': FileToolsModule,
    'tool_router': ToolRouterModule,

    # Network
    'api_server': APIServerModule,
    'network': NetworkModule,

    # Interface
    'gui': GUIModule,
}


def register_all(manager: ModuleManager):
    """Register all built-in modules with a manager."""
    for module_class in MODULE_REGISTRY.values():
        manager.register(module_class)


@lru_cache(maxsize=128)
def get_module(module_id: str) -> Optional[Type[Module]]:
    """Get a module class by ID."""
    return MODULE_REGISTRY.get(module_id)


@lru_cache(maxsize=32)
def list_modules() -> List[ModuleInfo]:
    """List all available modules."""
    return [cls.get_info() for cls in MODULE_REGISTRY.values()]


def list_by_category(category: ModuleCategory) -> List[ModuleInfo]:
    """List modules by category."""
    return [
        cls.get_info() for cls in MODULE_REGISTRY.values()
        if cls.get_info().category == category
    ]


def list_local_modules() -> List[ModuleInfo]:
    """List modules that run 100% locally (no cloud/internet required)."""
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
