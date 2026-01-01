"""
HuggingFace Integration for Enigma

Provides access to HuggingFace models:
- Text generation (local or API)
- Image generation
- Embeddings
- Text-to-Speech (TTS)

Usage:
    from enigma.addons.huggingface import HuggingFaceTextGeneration
    
    # Use HuggingFace Inference API
    addon = HuggingFaceTextGeneration(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        api_key="your_hf_token"
    )
    addon.load()
    result = addon.generate("Tell me a story")
    
    # Or use local model
    addon = HuggingFaceTextGeneration(
        model_name="gpt2",
        use_local=True
    )
"""

import os
from typing import Optional, List, Dict, Any
from pathlib import Path

from .base import (
    Addon, ImageAddon, AudioAddon, EmbeddingAddon,
    AddonConfig, AddonResult, AddonType, AddonProvider
)

# Check for HuggingFace dependencies
try:
    from huggingface_hub import InferenceClient, HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        pipeline, AutoModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class HuggingFaceTextGeneration(Addon):
    """
    Text generation using HuggingFace models.
    
    Supports:
    - HuggingFace Inference API (cloud)
    - Local transformers models
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        api_key: Optional[str] = None,
        use_local: bool = False,
        device: str = "auto"
    ):
        """
        Initialize HuggingFace text generation.
        
        Args:
            model_name: HuggingFace model ID (e.g., "gpt2", "meta-llama/Llama-2-7b-chat-hf")
            api_key: HuggingFace API token (for Inference API)
            use_local: If True, download and run model locally
            device: Device to use for local models ("cpu", "cuda", "auto")
        """
        config = AddonConfig(
            name=f"hf_text_{model_name.split('/')[-1]}",
            addon_type=AddonType.TEXT,
            provider=AddonProvider.HUGGINGFACE if not use_local else AddonProvider.LOCAL,
            model_name=model_name,
            api_key=api_key or os.getenv("HUGGINGFACE_TOKEN"),
            max_tokens=2048,
            supports_streaming=True
        )
        super().__init__(config)
        
        self.use_local = use_local
        self.device = device
        self.client = None
        self.model = None
        self.tokenizer = None
        self.pipeline = None
    
    def load(self) -> bool:
        """Load the model or connect to API."""
        if self.use_local:
            return self._load_local()
        else:
            return self._load_api()
    
    def _load_api(self) -> bool:
        """Load using HuggingFace Inference API."""
        if not HF_HUB_AVAILABLE:
            print("HuggingFace Hub not available. Install: pip install huggingface-hub")
            return False
        
        if not self.config.api_key:
            print("HuggingFace API key required. Set HUGGINGFACE_TOKEN environment variable.")
            return False
        
        try:
            self.client = InferenceClient(token=self.config.api_key)
            self.is_loaded = True
            self.is_available = True
            return True
        except Exception as e:
            print(f"Failed to connect to HuggingFace API: {e}")
            return False
    
    def _load_local(self) -> bool:
        """Load model locally using transformers."""
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers not available. Install: pip install transformers")
            return False
        
        try:
            import torch
            
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
            )
            
            self.is_loaded = True
            self.is_available = True
            return True
        except Exception as e:
            print(f"Failed to load local model: {e}")
            return False
    
    def unload(self) -> bool:
        """Unload model and free resources."""
        self.client = None
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_loaded = False
        self.is_available = False
        return True
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> AddonResult:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter
        
        Returns:
            AddonResult with generated text
        """
        if not self.is_loaded:
            return AddonResult(success=False, error="Model not loaded")
        
        max_tokens = max_tokens or self.config.max_tokens
        
        try:
            if self.use_local:
                # Local generation
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    **kwargs
                )
                text = outputs[0]["generated_text"]
                
                # Remove prompt from output
                if text.startswith(prompt):
                    text = text[len(prompt):].strip()
                
                return AddonResult(
                    success=True,
                    data=text,
                    addon_name=self.name,
                    operation="generate",
                    tokens_used=len(text.split())
                )
            else:
                # API generation
                response = self.client.text_generation(
                    prompt,
                    model=self.config.model_name,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs
                )
                
                return AddonResult(
                    success=True,
                    data=response,
                    addon_name=self.name,
                    operation="generate",
                    tokens_used=len(response.split())
                )
        except Exception as e:
            return AddonResult(
                success=False,
                error=str(e),
                addon_name=self.name
            )


class HuggingFaceImageGeneration(ImageAddon):
    """
    Image generation using HuggingFace models.
    
    Supports models like:
    - stabilityai/stable-diffusion-2-1
    - runwayml/stable-diffusion-v1-5
    - CompVis/stable-diffusion-v1-4
    """
    
    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-2-1",
        api_key: Optional[str] = None,
        use_local: bool = False
    ):
        """
        Initialize HuggingFace image generation.
        
        Args:
            model_name: HuggingFace model ID
            api_key: HuggingFace API token
            use_local: If True, use local Diffusers pipeline
        """
        config = AddonConfig(
            name=f"hf_image_{model_name.split('/')[-1]}",
            addon_type=AddonType.IMAGE,
            provider=AddonProvider.HUGGINGFACE if not use_local else AddonProvider.LOCAL,
            model_name=model_name,
            api_key=api_key or os.getenv("HUGGINGFACE_TOKEN")
        )
        super().__init__(config)
        
        self.use_local = use_local
        self.client = None
        self.pipeline = None
    
    def load(self) -> bool:
        """Load the model or connect to API."""
        if self.use_local:
            return self._load_local()
        else:
            return self._load_api()
    
    def _load_api(self) -> bool:
        """Load using HuggingFace Inference API."""
        if not HF_HUB_AVAILABLE:
            return False
        
        try:
            self.client = InferenceClient(token=self.config.api_key)
            self.is_loaded = True
            self.is_available = True
            return True
        except Exception as e:
            print(f"Failed to connect to HuggingFace API: {e}")
            return False
    
    def _load_local(self) -> bool:
        """Load model locally using diffusers."""
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            self.is_loaded = True
            self.is_available = True
            return True
        except ImportError:
            print("Diffusers not available. Install: pip install diffusers")
            return False
        except Exception as e:
            print(f"Failed to load local model: {e}")
            return False
    
    def unload(self) -> bool:
        """Unload model and free resources."""
        self.client = None
        self.pipeline = None
        self.is_loaded = False
        return True
    
    def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_images: int = 1,
        **kwargs
    ) -> AddonResult:
        """Generate images from prompt."""
        if not self.is_loaded:
            return AddonResult(success=False, error="Model not loaded")
        
        try:
            if self.use_local:
                # Local generation
                images = self.pipeline(
                    prompt,
                    width=width,
                    height=height,
                    num_images_per_prompt=num_images,
                    **kwargs
                ).images
                
                # Convert to bytes
                import io
                image_bytes_list = []
                for img in images:
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    image_bytes_list.append(buf.getvalue())
                
                return AddonResult(
                    success=True,
                    data=image_bytes_list[0] if num_images == 1 else image_bytes_list,
                    addon_name=self.name
                )
            else:
                # API generation
                image = self.client.text_to_image(
                    prompt,
                    model=self.config.model_name,
                    width=width,
                    height=height,
                    **kwargs
                )
                
                # Convert PIL image to bytes
                import io
                buf = io.BytesIO()
                image.save(buf, format='PNG')
                
                return AddonResult(
                    success=True,
                    data=buf.getvalue(),
                    addon_name=self.name
                )
        except Exception as e:
            return AddonResult(success=False, error=str(e))


class HuggingFaceEmbeddings(EmbeddingAddon):
    """
    Text embeddings using HuggingFace models.
    
    Supports models like:
    - sentence-transformers/all-MiniLM-L6-v2
    - sentence-transformers/all-mpnet-base-v2
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
        use_local: bool = True
    ):
        config = AddonConfig(
            name=f"hf_embed_{model_name.split('/')[-1]}",
            addon_type=AddonType.EMBEDDING,
            provider=AddonProvider.HUGGINGFACE if not use_local else AddonProvider.LOCAL,
            model_name=model_name,
            api_key=api_key or os.getenv("HUGGINGFACE_TOKEN")
        )
        super().__init__(config)
        
        self.use_local = use_local
        self.client = None
        self.model = None
    
    def load(self) -> bool:
        """Load the model."""
        if self.use_local:
            try:
                from sentence_transformers import SentenceTransformer
                
                self.model = SentenceTransformer(self.config.model_name)
                self.is_loaded = True
                self.is_available = True
                return True
            except ImportError:
                print("sentence-transformers not available. Install: pip install sentence-transformers")
                return False
            except Exception as e:
                print(f"Failed to load model: {e}")
                return False
        else:
            if not HF_HUB_AVAILABLE:
                return False
            
            try:
                self.client = InferenceClient(token=self.config.api_key)
                self.is_loaded = True
                self.is_available = True
                return True
            except Exception as e:
                print(f"Failed to connect: {e}")
                return False
    
    def unload(self) -> bool:
        """Unload model."""
        self.model = None
        self.client = None
        self.is_loaded = False
        return True
    
    def generate(self, text: str, **kwargs) -> AddonResult:
        """Generate embedding vector for text."""
        if not self.is_loaded:
            return AddonResult(success=False, error="Model not loaded")
        
        try:
            if self.use_local:
                embedding = self.model.encode(text).tolist()
            else:
                embedding = self.client.feature_extraction(
                    text,
                    model=self.config.model_name
                )
            
            return AddonResult(
                success=True,
                data=embedding,
                addon_name=self.name
            )
        except Exception as e:
            return AddonResult(success=False, error=str(e))


class HuggingFaceTTS(AudioAddon):
    """
    Text-to-Speech using HuggingFace models.
    
    Supports models like:
    - facebook/mms-tts-eng
    - facebook/fastspeech2-en-ljspeech
    """
    
    def __init__(
        self,
        model_name: str = "facebook/mms-tts-eng",
        api_key: Optional[str] = None,
        use_local: bool = False
    ):
        config = AddonConfig(
            name=f"hf_tts_{model_name.split('/')[-1]}",
            addon_type=AddonType.SPEECH,
            provider=AddonProvider.HUGGINGFACE if not use_local else AddonProvider.LOCAL,
            model_name=model_name,
            api_key=api_key or os.getenv("HUGGINGFACE_TOKEN")
        )
        super().__init__(config)
        
        self.use_local = use_local
        self.client = None
        self.pipeline = None
    
    def load(self) -> bool:
        """Load the model."""
        if self.use_local:
            try:
                from transformers import pipeline
                
                self.pipeline = pipeline("text-to-speech", model=self.config.model_name)
                self.is_loaded = True
                self.is_available = True
                return True
            except Exception as e:
                print(f"Failed to load TTS model: {e}")
                return False
        else:
            if not HF_HUB_AVAILABLE:
                return False
            
            try:
                self.client = InferenceClient(token=self.config.api_key)
                self.is_loaded = True
                self.is_available = True
                return True
            except Exception as e:
                print(f"Failed to connect: {e}")
                return False
    
    def unload(self) -> bool:
        """Unload model."""
        self.pipeline = None
        self.client = None
        self.is_loaded = False
        return True
    
    def generate(self, prompt: str, duration: float = 10.0, **kwargs) -> AddonResult:
        """Generate audio from prompt."""
        # For TTS, use text_to_speech instead
        return self.text_to_speech(prompt, **kwargs)
    
    def text_to_speech(self, text: str, voice: str = "default", **kwargs) -> AddonResult:
        """Convert text to speech."""
        if not self.is_loaded:
            return AddonResult(success=False, error="Model not loaded")
        
        try:
            if self.use_local:
                audio = self.pipeline(text)
                # audio is a dict with 'audio' (numpy array) and 'sampling_rate'
                return AddonResult(
                    success=True,
                    data=audio,
                    addon_name=self.name
                )
            else:
                # API doesn't have direct TTS for all models
                # This would need model-specific implementation
                return AddonResult(
                    success=False,
                    error="API TTS not implemented for this model"
                )
        except Exception as e:
            return AddonResult(success=False, error=str(e))
