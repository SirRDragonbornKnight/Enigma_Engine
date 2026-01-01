"""
Built-in Addon Implementations
==============================

Ready-to-use addons for common AI capabilities.
These can connect to various providers or run locally.
"""

import os
import base64
import time
from typing import Optional, List

from .base import (
    AddonConfig, AddonResult, AddonType, AddonProvider, ImageAddon,
    CodeAddon, VideoAddon, AudioAddon, EmbeddingAddon
)


# =============================================================================
# Image Generation Addons
# =============================================================================

class StableDiffusionLocal(ImageAddon):
    """
    Local Stable Diffusion image generation.
    Requires: diffusers, torch, accelerate
    """
    
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-1"):
        super().__init__(AddonConfig(
            name="stable_diffusion_local",
            addon_type=AddonType.IMAGE,
            provider=AddonProvider.LOCAL,
            model_name=model_id,
            extra={'model_id': model_id}
        ))
        self.pipe = None
    
    def load(self) -> bool:
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.config.extra['model_id'],
                torch_dtype=dtype
            ).to(device)
            
            self.is_loaded = True
            self.is_available = True
            return True
        except ImportError:
            print("Install diffusers: pip install diffusers transformers accelerate")
            return False
        except Exception as e:
            print(f"Failed to load Stable Diffusion: {e}")
            return False
    
    def unload(self) -> bool:
        if self.pipe:
            del self.pipe
            self.pipe = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.is_loaded = False
        self.is_available = False
        return True
    
    def generate(self, prompt: str, width: int = 512, height: int = 512,
                 num_images: int = 1, **kwargs) -> AddonResult:
        if not self.is_loaded:
            return AddonResult(success=False, error="Addon not loaded")
        
        try:
            start = time.time()
            
            images = self.pipe(
                prompt,
                width=width,
                height=height,
                num_images_per_prompt=num_images,
                **kwargs
            ).images
            
            # Convert to bytes
            import io
            results = []
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                results.append(buf.getvalue())
            
            return AddonResult(
                success=True,
                data=results if num_images > 1 else results[0],
                duration=time.time() - start,
                operation='generate_image'
            )
        except Exception as e:
            return AddonResult(success=False, error=str(e))


class OpenAIImage(ImageAddon):
    """
    OpenAI DALL-E image generation.
    
    ⚠️  CLOUD SERVICE: Requires internet connection and OpenAI API key.
    
    This addon connects to OpenAI's external cloud API to generate images.
    All prompts are sent to OpenAI's servers.
    
    Requires: openai
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "dall-e-3"):
        super().__init__(AddonConfig(
            name="openai_dalle",
            addon_type=AddonType.IMAGE,
            provider=AddonProvider.OPENAI,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            model_name=model,
            cost_per_request=0.04,  # Approximate
        ))
        self.client = None
    
    def load(self) -> bool:
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config.api_key)
            self.is_loaded = True
            self.is_available = bool(self.config.api_key)
            return True
        except ImportError:
            print("Install openai: pip install openai")
            return False
        except Exception as e:
            print(f"Failed to initialize OpenAI: {e}")
            return False
    
    def unload(self) -> bool:
        self.client = None
        self.is_loaded = False
        self.is_available = False
        return True
    
    def generate(self, prompt: str, width: int = 1024, height: int = 1024,
                 num_images: int = 1, **kwargs) -> AddonResult:
        if not self.is_loaded or not self.client:
            return AddonResult(success=False, error="Addon not loaded")
        
        try:
            start = time.time()
            
            # DALL-E 3 only supports certain sizes
            size = f"{width}x{height}"
            if size not in ["1024x1024", "1792x1024", "1024x1792"]:
                size = "1024x1024"
            
            response = self.client.images.generate(
                model=self.config.model_name,
                prompt=prompt,
                size=size,
                n=num_images,
                response_format="b64_json",
            )
            
            images = [base64.b64decode(img.b64_json) for img in response.data]
            
            return AddonResult(
                success=True,
                data=images if num_images > 1 else images[0],
                duration=time.time() - start,
                cost=self.config.cost_per_request * num_images,
                operation='generate_image'
            )
        except Exception as e:
            return AddonResult(success=False, error=str(e))


class ReplicateImage(ImageAddon):
    """
    Replicate.com image generation (SDXL, Flux, etc).
    
    ⚠️  CLOUD SERVICE: Requires internet connection and Replicate API token.
    
    This addon connects to Replicate's external cloud API to generate images.
    All prompts are sent to Replicate's servers.
    
    Requires: replicate
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: str = "stability-ai/sdxl:latest"):
        super().__init__(AddonConfig(
            name="replicate_image",
            addon_type=AddonType.IMAGE,
            provider=AddonProvider.REPLICATE,
            api_key=api_key or os.environ.get("REPLICATE_API_TOKEN"),
            model_name=model,
        ))
        self.client = None
    
    def load(self) -> bool:
        try:
            import replicate
            self.client = replicate
            self.is_loaded = True
            self.is_available = bool(self.config.api_key)
            return True
        except ImportError:
            print("Install replicate: pip install replicate")
            return False
    
    def unload(self) -> bool:
        self.client = None
        self.is_loaded = False
        return True
    
    def generate(self, prompt: str, width: int = 1024, height: int = 1024,
                 num_images: int = 1, **kwargs) -> AddonResult:
        if not self.is_loaded:
            return AddonResult(success=False, error="Addon not loaded")
        
        try:
            import requests
            start = time.time()
            
            output = self.client.run(
                self.config.model_name,
                input={
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "num_outputs": num_images,
                    **kwargs
                }
            )
            
            # Download images
            images = []
            for url in output:
                resp = requests.get(url)
                images.append(resp.content)
            
            return AddonResult(
                success=True,
                data=images if num_images > 1 else images[0],
                duration=time.time() - start,
                operation='generate_image'
            )
        except Exception as e:
            return AddonResult(success=False, error=str(e))


# =============================================================================
# Code Generation Addons
# =============================================================================

class EnigmaCode(CodeAddon):
    """
    Use Enigma's own model for code generation.
    Best for when you want everything local.
    """
    
    def __init__(self, model_name: str = "sacrifice"):
        super().__init__(AddonConfig(
            name="enigma_code",
            addon_type=AddonType.CODE,
            provider=AddonProvider.LOCAL,
            model_name=model_name,
        ))
        self.model = None
        self.tokenizer = None
    
    def load(self) -> bool:
        try:
            from enigma.core.inference import InferenceEngine
            self.engine = InferenceEngine(model_name=self.config.model_name)
            self.is_loaded = True
            self.is_available = True
            return True
        except Exception as e:
            print(f"Failed to load Enigma model: {e}")
            return False
    
    def unload(self) -> bool:
        self.engine = None
        self.is_loaded = False
        return True
    
    def generate(self, prompt: str, language: str = "python", **kwargs) -> AddonResult:
        if not self.is_loaded:
            return AddonResult(success=False, error="Addon not loaded")
        
        try:
            start = time.time()
            
            # Format prompt for code generation
            code_prompt = f"Write {language} code:\n{prompt}\n\n```{language}\n"
            
            result = self.engine.generate(
                code_prompt,
                max_tokens=kwargs.get('max_tokens', 500),
                temperature=kwargs.get('temperature', 0.3),
                stop_sequences=["\n```", "```"]
            )
            
            return AddonResult(
                success=True,
                data=result,
                duration=time.time() - start,
                operation='generate_code'
            )
        except Exception as e:
            return AddonResult(success=False, error=str(e))


class OpenAICode(CodeAddon):
    """
    OpenAI GPT for code generation.
    
    ⚠️  CLOUD SERVICE: Requires internet connection and OpenAI API key.
    
    This addon connects to OpenAI's external cloud API (GPT-4) for code generation.
    All prompts and code are sent to OpenAI's servers.
    
    Good for complex code tasks.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        super().__init__(AddonConfig(
            name="openai_code",
            addon_type=AddonType.CODE,
            provider=AddonProvider.OPENAI,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            model_name=model,
            supports_streaming=True,
        ))
        self.client = None
    
    def load(self) -> bool:
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config.api_key)
            self.is_loaded = True
            self.is_available = bool(self.config.api_key)
            return True
        except ImportError:
            print("Install openai: pip install openai")
            return False
    
    def unload(self) -> bool:
        self.client = None
        self.is_loaded = False
        return True
    
    def generate(self, prompt: str, language: str = "python", **kwargs) -> AddonResult:
        if not self.is_loaded or not self.client:
            return AddonResult(success=False, error="Addon not loaded")
        
        try:
            start = time.time()
            
            system_prompt = f"You are an expert {language} programmer. Write clean, efficient, well-documented code."
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get('temperature', 0.3),
                max_tokens=kwargs.get('max_tokens', 2000),
            )
            
            code = response.choices[0].message.content
            
            return AddonResult(
                success=True,
                data=code,
                duration=time.time() - start,
                tokens_used=response.usage.total_tokens,
                operation='generate_code'
            )
        except Exception as e:
            return AddonResult(success=False, error=str(e))
    
    def explain(self, code: str, **kwargs) -> AddonResult:
        """Explain what code does."""
        if not self.is_loaded or not self.client:
            return AddonResult(success=False, error="Addon not loaded")
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "Explain code clearly and concisely."},
                    {"role": "user", "content": f"Explain this code:\n\n{code}"}
                ],
                temperature=0.3,
            )
            
            return AddonResult(
                success=True,
                data=response.choices[0].message.content,
                operation='explain_code'
            )
        except Exception as e:
            return AddonResult(success=False, error=str(e))
    
    def refactor(self, code: str, instructions: str, **kwargs) -> AddonResult:
        """Refactor code based on instructions."""
        if not self.is_loaded or not self.client:
            return AddonResult(success=False, error="Addon not loaded")
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "Refactor code as requested. Return only the refactored code."},
                    {"role": "user", "content": f"Refactor this code:\n\n{code}\n\nInstructions: {instructions}"}
                ],
                temperature=0.3,
            )
            
            return AddonResult(
                success=True,
                data=response.choices[0].message.content,
                operation='refactor_code'
            )
        except Exception as e:
            return AddonResult(success=False, error=str(e))


# =============================================================================
# Video Generation Addons
# =============================================================================

class ReplicateVideo(VideoAddon):
    """
    Replicate.com video generation (Runway, AnimateDiff, etc).
    
    ⚠️  CLOUD SERVICE: Requires internet connection and Replicate API token.
    
    This addon connects to Replicate's external cloud API to generate videos.
    All prompts are sent to Replicate's servers.
    """
    
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "anotherjesse/zeroscope-v2-xl:latest"):
        super().__init__(AddonConfig(
            name="replicate_video",
            addon_type=AddonType.VIDEO,
            provider=AddonProvider.REPLICATE,
            api_key=api_key or os.environ.get("REPLICATE_API_TOKEN"),
            model_name=model,
        ))
        self.client = None
    
    def load(self) -> bool:
        try:
            import replicate
            self.client = replicate
            self.is_loaded = True
            self.is_available = bool(self.config.api_key)
            return True
        except ImportError:
            print("Install replicate: pip install replicate")
            return False
    
    def unload(self) -> bool:
        self.client = None
        self.is_loaded = False
        return True
    
    def generate(self, prompt: str, duration: float = 4.0,
                 fps: int = 24, **kwargs) -> AddonResult:
        if not self.is_loaded:
            return AddonResult(success=False, error="Addon not loaded")
        
        try:
            import requests
            start = time.time()
            
            output = self.client.run(
                self.config.model_name,
                input={
                    "prompt": prompt,
                    "num_frames": int(duration * fps),
                    "fps": fps,
                    **kwargs
                }
            )
            
            # Download video
            video_url = output if isinstance(output, str) else output[0]
            resp = requests.get(video_url)
            
            return AddonResult(
                success=True,
                data=resp.content,
                duration=time.time() - start,
                operation='generate_video'
            )
        except Exception as e:
            return AddonResult(success=False, error=str(e))


class LocalVideo(VideoAddon):
    """
    Local video generation using AnimateDiff or similar.
    Requires: diffusers with video support
    """
    
    def __init__(self, model_id: str = "guoyww/animatediff-motion-adapter-v1-5-2"):
        super().__init__(AddonConfig(
            name="local_video",
            addon_type=AddonType.VIDEO,
            provider=AddonProvider.LOCAL,
            model_name=model_id,
        ))
        self.pipe = None
    
    def load(self) -> bool:
        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
            import torch
            
            adapter = MotionAdapter.from_pretrained(self.config.model_name)
            model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
            
            self.pipe = AnimateDiffPipeline.from_pretrained(
                model_id,
                motion_adapter=adapter,
            )
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config,
                beta_schedule="linear"
            )
            
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
            
            self.is_loaded = True
            self.is_available = True
            return True
        except ImportError:
            print("Install diffusers with video support")
            return False
        except Exception as e:
            print(f"Failed to load video model: {e}")
            return False
    
    def unload(self) -> bool:
        if self.pipe:
            del self.pipe
            self.pipe = None
        self.is_loaded = False
        return True
    
    def generate(self, prompt: str, duration: float = 4.0,
                 fps: int = 8, **kwargs) -> AddonResult:
        if not self.is_loaded:
            return AddonResult(success=False, error="Addon not loaded")
        
        try:
            start = time.time()
            
            output = self.pipe(
                prompt,
                num_frames=int(duration * fps),
                guidance_scale=7.5,
            )
            
            # Export as gif or mp4
            frames = output.frames[0]
            
            import io
            
            buf = io.BytesIO()
            frames[0].save(
                buf,
                format='GIF',
                save_all=True,
                append_images=frames[1:],
                duration=1000 // fps,
                loop=0
            )
            
            return AddonResult(
                success=True,
                data=buf.getvalue(),
                duration=time.time() - start,
                operation='generate_video'
            )
        except Exception as e:
            return AddonResult(success=False, error=str(e))


# =============================================================================
# Audio Generation Addons
# =============================================================================

class ElevenLabsTTS(AudioAddon):
    """
    ElevenLabs text-to-speech.
    
    ⚠️  CLOUD SERVICE: Requires internet connection and ElevenLabs API key.
    
    This addon connects to ElevenLabs' external cloud API for voice synthesis.
    All text is sent to ElevenLabs' servers.
    
    High quality voice synthesis.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(AddonConfig(
            name="elevenlabs_tts",
            addon_type=AddonType.AUDIO,
            provider=AddonProvider.ELEVENLABS,
            api_key=api_key or os.environ.get("ELEVENLABS_API_KEY"),
        ))
        self.client = None
    
    def load(self) -> bool:
        try:
            from elevenlabs import ElevenLabs
            self.client = ElevenLabs(api_key=self.config.api_key)
            self.is_loaded = True
            self.is_available = bool(self.config.api_key)
            return True
        except ImportError:
            print("Install elevenlabs: pip install elevenlabs")
            return False
    
    def unload(self) -> bool:
        self.client = None
        self.is_loaded = False
        return True
    
    def generate(self, prompt: str, duration: float = 10.0, **kwargs) -> AddonResult:
        """Generate speech from text."""
        return self.text_to_speech(prompt, **kwargs)
    
    def text_to_speech(self, text: str, voice: str = "Rachel", **kwargs) -> AddonResult:
        if not self.is_loaded:
            return AddonResult(success=False, error="Addon not loaded")
        
        try:
            start = time.time()
            
            audio = self.client.generate(
                text=text,
                voice=voice,
                model="eleven_monolingual_v1"
            )
            
            # Collect audio bytes
            audio_bytes = b"".join(audio)
            
            return AddonResult(
                success=True,
                data=audio_bytes,
                duration=time.time() - start,
                operation='text_to_speech'
            )
        except Exception as e:
            return AddonResult(success=False, error=str(e))


class LocalTTS(AudioAddon):
    """
    Local text-to-speech using pyttsx3 or edge-tts.
    Free and runs offline.
    """
    
    def __init__(self):
        super().__init__(AddonConfig(
            name="local_tts",
            addon_type=AddonType.AUDIO,
            provider=AddonProvider.LOCAL,
        ))
        self.engine = None
    
    def load(self) -> bool:
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.is_loaded = True
            self.is_available = True
            return True
        except ImportError:
            print("Install pyttsx3: pip install pyttsx3")
            return False
    
    def unload(self) -> bool:
        if self.engine:
            self.engine.stop()
        self.engine = None
        self.is_loaded = False
        return True
    
    def generate(self, prompt: str, duration: float = 10.0, **kwargs) -> AddonResult:
        return self.text_to_speech(prompt, **kwargs)
    
    def text_to_speech(self, text: str, voice: str = "default", **kwargs) -> AddonResult:
        if not self.is_loaded:
            return AddonResult(success=False, error="Addon not loaded")
        
        try:
            import tempfile
            import os
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            self.engine.save_to_file(text, temp_path)
            self.engine.runAndWait()
            
            # Read back
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            os.unlink(temp_path)
            
            return AddonResult(
                success=True,
                data=audio_data,
                operation='text_to_speech'
            )
        except Exception as e:
            return AddonResult(success=False, error=str(e))


class ReplicateAudio(AudioAddon):
    """
    Replicate.com audio/music generation.
    
    ⚠️  CLOUD SERVICE: Requires internet connection and Replicate API token.
    
    This addon connects to Replicate's external cloud API for audio/music generation.
    All prompts are sent to Replicate's servers.
    
    Supports MusicGen, AudioCraft, etc.
    """
    
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "meta/musicgen:latest"):
        super().__init__(AddonConfig(
            name="replicate_audio",
            addon_type=AddonType.AUDIO,
            provider=AddonProvider.REPLICATE,
            api_key=api_key or os.environ.get("REPLICATE_API_TOKEN"),
            model_name=model,
        ))
        self.client = None
    
    def load(self) -> bool:
        try:
            import replicate
            self.client = replicate
            self.is_loaded = True
            self.is_available = bool(self.config.api_key)
            return True
        except ImportError:
            print("Install replicate: pip install replicate")
            return False
    
    def unload(self) -> bool:
        self.client = None
        self.is_loaded = False
        return True
    
    def generate(self, prompt: str, duration: float = 10.0, **kwargs) -> AddonResult:
        if not self.is_loaded:
            return AddonResult(success=False, error="Addon not loaded")
        
        try:
            import requests
            start = time.time()
            
            output = self.client.run(
                self.config.model_name,
                input={
                    "prompt": prompt,
                    "duration": duration,
                    **kwargs
                }
            )
            
            # Download audio
            audio_url = output if isinstance(output, str) else output[0]
            resp = requests.get(audio_url)
            
            return AddonResult(
                success=True,
                data=resp.content,
                duration=time.time() - start,
                operation='generate_audio'
            )
        except Exception as e:
            return AddonResult(success=False, error=str(e))
    
    def music(self, prompt: str, genre: str = None, **kwargs) -> AddonResult:
        """Generate music with optional genre."""
        if genre:
            prompt = f"{genre} music: {prompt}"
        return self.generate(prompt, **kwargs)


# =============================================================================
# Embedding Addons
# =============================================================================

class LocalEmbedding(EmbeddingAddon):
    """
    Local embedding using sentence-transformers.
    Great for semantic search and similarity.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(AddonConfig(
            name="local_embedding",
            addon_type=AddonType.EMBEDDING,
            provider=AddonProvider.LOCAL,
            model_name=model_name,
        ))
        self.model = None
    
    def load(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config.model_name)
            self.is_loaded = True
            self.is_available = True
            return True
        except ImportError:
            print("Install sentence-transformers: pip install sentence-transformers")
            return False
    
    def unload(self) -> bool:
        self.model = None
        self.is_loaded = False
        return True
    
    def generate(self, text: str, **kwargs) -> AddonResult:
        if not self.is_loaded:
            return AddonResult(success=False, error="Addon not loaded")
        
        try:
            embedding = self.model.encode(text).tolist()
            return AddonResult(success=True, data=embedding, operation='embed')
        except Exception as e:
            return AddonResult(success=False, error=str(e))
    
    def batch_embed(self, texts: List[str], **kwargs) -> AddonResult:
        if not self.is_loaded:
            return AddonResult(success=False, error="Addon not loaded")
        
        try:
            embeddings = self.model.encode(texts).tolist()
            return AddonResult(success=True, data=embeddings, operation='batch_embed')
        except Exception as e:
            return AddonResult(success=False, error=str(e))


class OpenAIEmbedding(EmbeddingAddon):
    """
    OpenAI embeddings.
    
    ⚠️  CLOUD SERVICE: Requires internet connection and OpenAI API key.
    
    This addon connects to OpenAI's external cloud API for embeddings.
    All text is sent to OpenAI's servers.
    
    High quality for production use.
    """
    
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "text-embedding-3-small"):
        super().__init__(AddonConfig(
            name="openai_embedding",
            addon_type=AddonType.EMBEDDING,
            provider=AddonProvider.OPENAI,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            model_name=model,
        ))
        self.client = None
    
    def load(self) -> bool:
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config.api_key)
            self.is_loaded = True
            self.is_available = bool(self.config.api_key)
            return True
        except ImportError:
            print("Install openai: pip install openai")
            return False
    
    def unload(self) -> bool:
        self.client = None
        self.is_loaded = False
        return True
    
    def generate(self, text: str, **kwargs) -> AddonResult:
        if not self.is_loaded or not self.client:
            return AddonResult(success=False, error="Addon not loaded")
        
        try:
            response = self.client.embeddings.create(
                model=self.config.model_name,
                input=text
            )
            embedding = response.data[0].embedding
            return AddonResult(success=True, data=embedding, operation='embed')
        except Exception as e:
            return AddonResult(success=False, error=str(e))


# =============================================================================
# Mock Addons (for testing)
# =============================================================================

class MockImageAddon(ImageAddon):
    """Mock image addon for testing."""
    
    def __init__(self):
        super().__init__(AddonConfig(
            name="mock_image",
            addon_type=AddonType.IMAGE,
            provider=AddonProvider.MOCK,
        ))
    
    def load(self) -> bool:
        self.is_loaded = True
        self.is_available = True
        return True
    
    def unload(self) -> bool:
        self.is_loaded = False
        return True
    
    def generate(self, prompt: str, width: int = 512, height: int = 512,
                 num_images: int = 1, **kwargs) -> AddonResult:
        # Generate a simple colored image
        try:
            from PIL import Image
            import io
            import random
            
            images = []
            for _ in range(num_images):
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                img = Image.new('RGB', (width, height), color)
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                images.append(buf.getvalue())
            
            return AddonResult(
                success=True,
                data=images if num_images > 1 else images[0],
                operation='mock_generate_image'
            )
        except ImportError:
            return AddonResult(success=True, data=b"mock_image_data", operation='mock')


class MockCodeAddon(CodeAddon):
    """Mock code addon for testing."""
    
    def __init__(self):
        super().__init__(AddonConfig(
            name="mock_code",
            addon_type=AddonType.CODE,
            provider=AddonProvider.MOCK,
        ))
    
    def load(self) -> bool:
        self.is_loaded = True
        self.is_available = True
        return True
    
    def unload(self) -> bool:
        self.is_loaded = False
        return True
    
    def generate(self, prompt: str, language: str = "python", **kwargs) -> AddonResult:
        code = f'''# Generated {language} code
# Prompt: {prompt}

def solution():
    """Mock implementation"""
    print("Hello from generated code!")
    return True

if __name__ == "__main__":
    solution()
'''
        return AddonResult(success=True, data=code, operation='mock_generate_code')


class MockVideoAddon(VideoAddon):
    """Mock video addon for testing."""
    
    def __init__(self):
        super().__init__(AddonConfig(
            name="mock_video",
            addon_type=AddonType.VIDEO,
            provider=AddonProvider.MOCK,
        ))
    
    def load(self) -> bool:
        self.is_loaded = True
        self.is_available = True
        return True
    
    def unload(self) -> bool:
        self.is_loaded = False
        return True
    
    def generate(self, prompt: str, duration: float = 4.0,
                 fps: int = 24, **kwargs) -> AddonResult:
        return AddonResult(success=True, data=b"mock_video_data", operation='mock_generate_video')


# =============================================================================
# HuggingFace Integration
# =============================================================================

# Import HuggingFace addons if available
try:
    from .huggingface import (
        HuggingFaceTextGeneration,
        HuggingFaceImageGeneration,
        HuggingFaceEmbeddings,
        HuggingFaceTTS
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# =============================================================================
# All Built-in Addons
# =============================================================================

BUILTIN_ADDONS = {
    # Image
    'stable_diffusion_local': StableDiffusionLocal,
    'openai_dalle': OpenAIImage,
    'replicate_image': ReplicateImage,
    
    # Code
    'enigma_code': EnigmaCode,
    'openai_code': OpenAICode,
    
    # Video
    'replicate_video': ReplicateVideo,
    'local_video': LocalVideo,
    
    # Audio
    'elevenlabs_tts': ElevenLabsTTS,
    'local_tts': LocalTTS,
    'replicate_audio': ReplicateAudio,
    
    # Embedding
    'local_embedding': LocalEmbedding,
    'openai_embedding': OpenAIEmbedding,
    
    # Mocks
    'mock_image': MockImageAddon,
    'mock_code': MockCodeAddon,
    'mock_video': MockVideoAddon,
}

# Add HuggingFace addons if available
if HF_AVAILABLE:
    BUILTIN_ADDONS.update({
        'huggingface_text': HuggingFaceTextGeneration,
        'huggingface_image': HuggingFaceImageGeneration,
        'huggingface_embedding': HuggingFaceEmbeddings,
        'huggingface_tts': HuggingFaceTTS,
    })
