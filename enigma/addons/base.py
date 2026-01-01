"""
Addon Base Classes
==================

Base classes for all Enigma addons.
Extend these to create your own AI capabilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import time


class AddonType(Enum):
    """Types of AI capabilities an addon can provide."""
    
    # Generation types
    TEXT = auto()           # Text generation (default Enigma)
    IMAGE = auto()          # Image generation
    CODE = auto()           # Code generation/completion
    VIDEO = auto()          # Video generation
    AUDIO = auto()          # Audio/music generation
    SPEECH = auto()         # Text-to-speech
    THREED = auto()         # 3D model generation
    
    # Understanding types
    VISION = auto()         # Image understanding
    TRANSCRIPTION = auto()  # Speech-to-text
    TRANSLATION = auto()    # Language translation
    
    # Specialized types
    EMBEDDING = auto()      # Vector embeddings
    MODERATION = auto()     # Content moderation
    REASONING = auto()      # Chain-of-thought reasoning
    AGENT = auto()          # Autonomous agent capabilities
    
    # Meta types
    ROUTER = auto()         # Routes to other addons
    ENSEMBLE = auto()       # Combines multiple addons
    CUSTOM = auto()         # User-defined type


class AddonProvider(Enum):
    """Where the addon's AI runs."""
    
    LOCAL = auto()          # Runs on local hardware
    OPENAI = auto()         # OpenAI API
    ANTHROPIC = auto()      # Anthropic/Claude API
    STABILITY = auto()      # Stability AI
    REPLICATE = auto()      # Replicate.com
    HUGGINGFACE = auto()    # HuggingFace Inference API
    RUNWAY = auto()         # Runway ML (video)
    ELEVENLABS = auto()     # ElevenLabs (voice)
    CUSTOM_API = auto()     # Custom REST API
    ENIGMA_REMOTE = auto()  # Another Enigma instance
    MOCK = auto()           # Testing/placeholder


@dataclass
class AddonConfig:
    """Configuration for an addon."""
    
    # Identity
    name: str
    addon_type: AddonType
    provider: AddonProvider
    version: str = "1.0.0"
    
    # Connection
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    model_name: Optional[str] = None
    model_path: Optional[Path] = None
    
    # Limits
    max_tokens: int = 4096
    timeout: float = 60.0
    rate_limit: float = 1.0  # requests per second
    
    # Features
    supports_streaming: bool = False
    supports_batch: bool = False
    supports_async: bool = False
    
    # Cost tracking
    cost_per_request: float = 0.0
    cost_per_token: float = 0.0
    
    # Custom settings
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'addon_type': self.addon_type.name,
            'provider': self.provider.name,
            'version': self.version,
            'api_url': self.api_url,
            'model_name': self.model_name,
            'max_tokens': self.max_tokens,
            'timeout': self.timeout,
            'supports_streaming': self.supports_streaming,
            'extra': self.extra,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AddonConfig':
        return cls(
            name=data['name'],
            addon_type=AddonType[data['addon_type']],
            provider=AddonProvider[data['provider']],
            version=data.get('version', '1.0.0'),
            api_url=data.get('api_url'),
            model_name=data.get('model_name'),
            max_tokens=data.get('max_tokens', 4096),
            timeout=data.get('timeout', 60.0),
            supports_streaming=data.get('supports_streaming', False),
            extra=data.get('extra', {}),
        )


@dataclass
class AddonResult:
    """Result from an addon operation."""
    
    success: bool
    data: Any = None
    error: Optional[str] = None
    
    # Metadata
    addon_name: str = ""
    operation: str = ""
    duration: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0
    
    # For streaming
    is_streaming: bool = False
    stream_generator: Optional[Any] = None


class Addon(ABC):
    """
    Base class for all Enigma addons.
    
    Extend this to create new AI capabilities.
    
    Example:
        class MyImageGen(Addon):
            def __init__(self):
                super().__init__(AddonConfig(
                    name="my_image_gen",
                    addon_type=AddonType.IMAGE,
                    provider=AddonProvider.LOCAL
                ))
            
            def generate(self, prompt, **kwargs):
                # Your image generation logic
                return AddonResult(success=True, data=image_bytes)
    """
    
    def __init__(self, config: AddonConfig):
        self.config = config
        self.is_loaded = False
        self.is_available = False
        self._last_request_time = 0.0
        self._total_requests = 0
        self._total_cost = 0.0
        self._callbacks: Dict[str, List[Callable]] = {}
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def addon_type(self) -> AddonType:
        return self.config.addon_type
    
    @property
    def provider(self) -> AddonProvider:
        return self.config.provider
    
    # === Lifecycle ===
    
    @abstractmethod
    def load(self) -> bool:
        """Load the addon (connect to API, load model, etc.)."""
    
    @abstractmethod
    def unload(self) -> bool:
        """Unload the addon (disconnect, free resources)."""
    
    def check_availability(self) -> bool:
        """Check if the addon is currently available."""
        return self.is_loaded and self.is_available
    
    # === Core Operations ===
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> AddonResult:
        """
        Main generation method.
        
        Args:
            prompt: The input prompt/request
            **kwargs: Addon-specific parameters
            
        Returns:
            AddonResult with the generated content
        """
    
    def generate_stream(self, prompt: str, **kwargs):
        """
        Streaming generation (if supported).
        
        Yields chunks of the result as they're generated.
        """
        if not self.config.supports_streaming:
            result = self.generate(prompt, **kwargs)
            yield result
            return
        
        # Override in subclass for real streaming
        result = self.generate(prompt, **kwargs)
        yield result
    
    async def generate_async(self, prompt: str, **kwargs) -> AddonResult:
        """Async generation (if supported)."""
        if not self.config.supports_async:
            return self.generate(prompt, **kwargs)
        
        # Override in subclass for real async
        return self.generate(prompt, **kwargs)
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[AddonResult]:
        """Batch generation (if supported)."""
        if not self.config.supports_batch:
            return [self.generate(p, **kwargs) for p in prompts]
        
        # Override in subclass for optimized batch
        return [self.generate(p, **kwargs) for p in prompts]
    
    # === Rate Limiting ===
    
    def _wait_for_rate_limit(self):
        """Respect rate limits."""
        if self.config.rate_limit > 0:
            elapsed = time.time() - self._last_request_time
            min_interval = 1.0 / self.config.rate_limit
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
    
    def _track_request(self, result: AddonResult):
        """Track request for stats."""
        self._total_requests += 1
        self._total_cost += result.cost
    
    # === Events ===
    
    def on(self, event: str, callback: Callable):
        """Register an event callback."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def emit(self, event: str, data: Any = None):
        """Emit an event to all callbacks."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Addon callback error: {e}")
    
    # === Info ===
    
    def get_info(self) -> dict:
        """Get addon information."""
        return {
            'name': self.name,
            'type': self.addon_type.name,
            'provider': self.provider.name,
            'version': self.config.version,
            'loaded': self.is_loaded,
            'available': self.is_available,
            'total_requests': self._total_requests,
            'total_cost': self._total_cost,
            'supports': {
                'streaming': self.config.supports_streaming,
                'batch': self.config.supports_batch,
                'async': self.config.supports_async,
            }
        }
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this addon provides."""
        caps = ['generate']
        if self.config.supports_streaming:
            caps.append('stream')
        if self.config.supports_batch:
            caps.append('batch')
        if self.config.supports_async:
            caps.append('async')
        return caps
    
    def __repr__(self):
        return f"<{self.__class__.__name__}({self.name}) type={self.addon_type.name} provider={self.provider.name}>"


# === Specialized Base Classes ===

class ImageAddon(Addon):
    """Base class for image generation addons."""
    
    def __init__(self, config: AddonConfig):
        config.addon_type = AddonType.IMAGE
        super().__init__(config)
    
    @abstractmethod
    def generate(self, prompt: str, width: int = 512, height: int = 512, 
                 num_images: int = 1, **kwargs) -> AddonResult:
        """Generate images from prompt."""
    
    def edit(self, image: bytes, prompt: str, **kwargs) -> AddonResult:
        """Edit an existing image (if supported)."""
        return AddonResult(success=False, error="Edit not supported")
    
    def variations(self, image: bytes, num_variations: int = 4, **kwargs) -> AddonResult:
        """Generate variations of an image (if supported)."""
        return AddonResult(success=False, error="Variations not supported")


class CodeAddon(Addon):
    """Base class for code generation addons."""
    
    def __init__(self, config: AddonConfig):
        config.addon_type = AddonType.CODE
        super().__init__(config)
    
    @abstractmethod
    def generate(self, prompt: str, language: str = "python", **kwargs) -> AddonResult:
        """Generate code from prompt."""
    
    def complete(self, code: str, cursor_position: int = -1, **kwargs) -> AddonResult:
        """Complete code at cursor position."""
        return AddonResult(success=False, error="Completion not supported")
    
    def explain(self, code: str, **kwargs) -> AddonResult:
        """Explain what code does."""
        return AddonResult(success=False, error="Explain not supported")
    
    def refactor(self, code: str, instructions: str, **kwargs) -> AddonResult:
        """Refactor code based on instructions."""
        return AddonResult(success=False, error="Refactor not supported")


class VideoAddon(Addon):
    """Base class for video generation addons."""
    
    def __init__(self, config: AddonConfig):
        config.addon_type = AddonType.VIDEO
        super().__init__(config)
    
    @abstractmethod
    def generate(self, prompt: str, duration: float = 4.0, 
                 fps: int = 24, **kwargs) -> AddonResult:
        """Generate video from prompt."""
    
    def extend(self, video: bytes, prompt: str, **kwargs) -> AddonResult:
        """Extend an existing video."""
        return AddonResult(success=False, error="Extend not supported")
    
    def image_to_video(self, image: bytes, prompt: str, **kwargs) -> AddonResult:
        """Animate an image into video."""
        return AddonResult(success=False, error="Image-to-video not supported")


class AudioAddon(Addon):
    """Base class for audio generation addons."""
    
    def __init__(self, config: AddonConfig):
        config.addon_type = AddonType.AUDIO
        super().__init__(config)
    
    @abstractmethod
    def generate(self, prompt: str, duration: float = 10.0, **kwargs) -> AddonResult:
        """Generate audio from prompt."""
    
    def text_to_speech(self, text: str, voice: str = "default", **kwargs) -> AddonResult:
        """Convert text to speech."""
        return AddonResult(success=False, error="TTS not supported")
    
    def music(self, prompt: str, genre: str = None, **kwargs) -> AddonResult:
        """Generate music."""
        return AddonResult(success=False, error="Music not supported")


class EmbeddingAddon(Addon):
    """Base class for embedding/vector addons."""
    
    def __init__(self, config: AddonConfig):
        config.addon_type = AddonType.EMBEDDING
        super().__init__(config)
    
    @abstractmethod
    def generate(self, text: str, **kwargs) -> AddonResult:
        """Generate embedding vector for text."""
    
    def batch_embed(self, texts: List[str], **kwargs) -> AddonResult:
        """Embed multiple texts."""
        vectors = []
        for text in texts:
            result = self.generate(text, **kwargs)
            if result.success:
                vectors.append(result.data)
            else:
                return result
        return AddonResult(success=True, data=vectors)


class ThreeDAddon(Addon):
    """Base class for 3D model generation addons."""
    
    def __init__(self, config: AddonConfig):
        config.addon_type = AddonType.THREED
        super().__init__(config)
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> AddonResult:
        """Generate 3D model from text prompt."""
    
    def text_to_3d(self, prompt: str, format: str = "glb", **kwargs) -> AddonResult:
        """Generate 3D model from text, specify output format."""
        return self.generate(prompt, format=format, **kwargs)
    
    def image_to_3d(self, image: bytes, **kwargs) -> AddonResult:
        """Generate 3D model from image (if supported)."""
        return AddonResult(success=False, error="Image-to-3D not supported")
    
    def refine(self, model: bytes, prompt: str, **kwargs) -> AddonResult:
        """Refine existing 3D model."""
        return AddonResult(success=False, error="Refinement not supported")
                return AddonResult(success=False, error=result.error)
        return AddonResult(success=True, data=vectors)
    
    def similarity(self, text1: str, text2: str, **kwargs) -> AddonResult:
        """Calculate similarity between two texts."""
        import math
        r1 = self.generate(text1, **kwargs)
        r2 = self.generate(text2, **kwargs)
        if not (r1.success and r2.success):
            return AddonResult(success=False, error="Failed to embed")
        
        # Cosine similarity
        v1, v2 = r1.data, r2.data
        dot = sum(a*b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a*a for a in v1))
        mag2 = math.sqrt(sum(b*b for b in v2))
        similarity = dot / (mag1 * mag2) if mag1 * mag2 > 0 else 0
        
        return AddonResult(success=True, data=similarity)
