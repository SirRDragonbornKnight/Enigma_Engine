"""
Image Generation Example for ForgeAI

This example shows how to generate images with ForgeAI.
Create images from text prompts using local or cloud providers.

SUPPORTED PROVIDERS:
- Local: Stable Diffusion (GPU required)
- OpenAI: DALL-E (API key required)
- Replicate: Various models (API key required)

USAGE:
    python examples/image_gen_example.py
    
Or import in your own code:
    from examples.image_gen_example import generate_image
"""

import os
import json
import time
import base64
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

IMAGE_OUTPUT_DIR = Path.home() / ".forge_ai" / "outputs" / "images"
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# IMAGE GENERATOR INTERFACE
# =============================================================================

class ImageGenerator(ABC):
    """Base class for image generators."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generate image from prompt.
        
        Args:
            prompt: Text description of desired image
            **kwargs: Provider-specific options
        
        Returns:
            Path to generated image, or None if failed
        """


# =============================================================================
# LOCAL STABLE DIFFUSION
# =============================================================================

class LocalStableDiffusion(ImageGenerator):
    """
    Local Stable Diffusion image generation.
    
    Requirements:
        - NVIDIA GPU with 4GB+ VRAM (or CPU with patience)
        - pip install diffusers transformers accelerate torch
    
    First run will download ~4GB model.
    """
    
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-1-base"):
        self.model_id = model_id
        self._pipe = None
        self._available = False
        self._check_available()
    
    def _check_available(self):
        """Check if Stable Diffusion can run."""
        try:
            import torch
            self._has_cuda = torch.cuda.is_available()
            self._available = True
            print(f"[IMAGE] Stable Diffusion ready (CUDA: {self._has_cuda})")
        except ImportError:
            print("[IMAGE] Local image gen requires:")
            print("  pip install diffusers transformers accelerate torch")
    
    def _load_model(self):
        """Load the model (lazy loading)."""
        if self._pipe is not None:
            return
        
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            print(f"[IMAGE] Loading model: {self.model_id}")
            print("[IMAGE] This may take a few minutes on first run...")
            
            dtype = torch.float16 if self._has_cuda else torch.float32
            device = "cuda" if self._has_cuda else "cpu"
            
            self._pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            )
            self._pipe = self._pipe.to(device)
            
            # Enable memory optimizations
            if self._has_cuda:
                self._pipe.enable_attention_slicing()
            
            print("[IMAGE] Model loaded!")
            
        except Exception as e:
            print(f"[IMAGE] Model load failed: {e}")
            self._pipe = None
    
    def generate(self, prompt: str, 
                 negative_prompt: str = "",
                 width: int = 512, 
                 height: int = 512,
                 steps: int = 30,
                 guidance_scale: float = 7.5,
                 seed: int = None,
                 output_path: str = None) -> Optional[str]:
        """
        Generate image from prompt.
        
        Args:
            prompt: What to generate
            negative_prompt: What to avoid
            width: Image width (must be divisible by 8)
            height: Image height (must be divisible by 8)
            steps: Number of inference steps (more = better but slower)
            guidance_scale: How closely to follow prompt (7-12 typical)
            seed: Random seed for reproducibility
            output_path: Where to save (auto-generated if None)
        
        Returns:
            Path to generated image
        """
        if not self._available:
            return None
        
        self._load_model()
        if self._pipe is None:
            return None
        
        try:
            import torch
            
            # Set seed if provided
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self._pipe.device).manual_seed(seed)
            
            # Generate
            print(f"[IMAGE] Generating: {prompt[:50]}...")
            
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            
            image = result.images[0]
            
            # Save
            if output_path is None:
                timestamp = int(time.time())
                output_path = str(IMAGE_OUTPUT_DIR / f"image_{timestamp}.png")
            
            image.save(output_path)
            print(f"[IMAGE] Saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"[IMAGE] Generation failed: {e}")
            return None


# =============================================================================
# OPENAI DALL-E
# =============================================================================

class OpenAIImage(ImageGenerator):
    """
    OpenAI DALL-E image generation.
    
    Requirements:
        - OpenAI API key
        - pip install openai
    
    Models:
        - dall-e-3: Best quality, slower
        - dall-e-2: Good quality, faster
    """
    
    CONFIG_FILE = Path.home() / ".forge_ai" / "openai.json"
    
    def __init__(self, api_key: str = None, model: str = "dall-e-3"):
        self.api_key = api_key or self._load_api_key()
        self.model = model
        self._available = bool(self.api_key)
        
        if not self._available:
            print("[IMAGE] OpenAI needs API key:")
            print("  Set OPENAI_API_KEY environment variable")
            print("  Or: gen = OpenAIImage('your-key')")
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment or config."""
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return key
        
        if self.CONFIG_FILE.exists():
            with open(self.CONFIG_FILE) as f:
                return json.load(f).get("api_key")
        
        return None
    
    def save_api_key(self, api_key: str):
        """Save API key for later use."""
        self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump({"api_key": api_key}, f)
        self.api_key = api_key
        self._available = True
    
    def generate(self, prompt: str,
                 size: str = "1024x1024",
                 quality: str = "standard",
                 style: str = "vivid",
                 output_path: str = None) -> Optional[str]:
        """
        Generate image with DALL-E.
        
        Args:
            prompt: Image description
            size: "1024x1024", "1792x1024", or "1024x1792" (DALL-E 3)
            quality: "standard" or "hd"
            style: "vivid" or "natural"
            output_path: Where to save
        
        Returns:
            Path to generated image
        """
        if not self._available:
            return None
        
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            print(f"[IMAGE] Generating with {self.model}: {prompt[:50]}...")
            
            response = client.images.generate(
                model=self.model,
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=1,
            )
            
            # Get image URL
            image_url = response.data[0].url
            
            # Download image
            if output_path is None:
                timestamp = int(time.time())
                output_path = str(IMAGE_OUTPUT_DIR / f"dalle_{timestamp}.png")
            
            self._download_image(image_url, output_path)
            print(f"[IMAGE] Saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"[IMAGE] DALL-E failed: {e}")
            return None
    
    def _download_image(self, url: str, path: str):
        """Download image from URL."""
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            with open(path, 'wb') as f:
                f.write(response.read())


# =============================================================================
# REPLICATE
# =============================================================================

class ReplicateImage(ImageGenerator):
    """
    Replicate.com image generation.
    
    Access to many models:
        - SDXL
        - Flux
        - Kandinsky
        - And more
    
    Requirements:
        - Replicate API key
        - pip install replicate
    """
    
    CONFIG_FILE = Path.home() / ".forge_ai" / "replicate.json"
    
    DEFAULT_MODEL = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or self._load_api_key()
        self.model = model or self.DEFAULT_MODEL
        self._available = bool(self.api_key)
        
        if not self._available:
            print("[IMAGE] Replicate needs API key:")
            print("  Set REPLICATE_API_TOKEN environment variable")
            print("  Or: gen = ReplicateImage('your-key')")
    
    def _load_api_key(self) -> Optional[str]:
        key = os.environ.get("REPLICATE_API_TOKEN")
        if key:
            return key
        
        if self.CONFIG_FILE.exists():
            with open(self.CONFIG_FILE) as f:
                return json.load(f).get("api_key")
        
        return None
    
    def save_api_key(self, api_key: str):
        """Save API key."""
        self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump({"api_key": api_key}, f)
        self.api_key = api_key
        self._available = True
    
    def generate(self, prompt: str,
                 negative_prompt: str = "",
                 width: int = 1024,
                 height: int = 1024,
                 output_path: str = None) -> Optional[str]:
        """
        Generate image with Replicate.
        
        Args:
            prompt: Image description
            negative_prompt: What to avoid
            width: Image width
            height: Image height
            output_path: Where to save
        
        Returns:
            Path to generated image
        """
        if not self._available:
            return None
        
        try:
            import replicate
            
            os.environ["REPLICATE_API_TOKEN"] = self.api_key
            
            print(f"[IMAGE] Generating with Replicate: {prompt[:50]}...")
            
            output = replicate.run(
                self.model,
                input={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                }
            )
            
            # Get output URL (format varies by model)
            if isinstance(output, list):
                image_url = output[0]
            else:
                image_url = output
            
            # Download
            if output_path is None:
                timestamp = int(time.time())
                output_path = str(IMAGE_OUTPUT_DIR / f"replicate_{timestamp}.png")
            
            req = urllib.request.Request(str(image_url))
            with urllib.request.urlopen(req) as response:
                with open(output_path, 'wb') as f:
                    f.write(response.read())
            
            print(f"[IMAGE] Saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[IMAGE] Replicate failed: {e}")
            return None


# =============================================================================
# IMAGE CONTROLLER
# =============================================================================

class ImageController:
    """
    Unified image generation interface.
    """
    
    def __init__(self, preferred_provider: str = "auto"):
        """
        Args:
            preferred_provider: "local", "openai", "replicate", or "auto"
        """
        self.providers: Dict[str, ImageGenerator] = {}
        self._active = None
        
        self._init_providers(preferred_provider)
    
    def _init_providers(self, preferred: str):
        """Initialize available providers."""
        # Check local
        try:
            local = LocalStableDiffusion()
            if local._available:
                self.providers["local"] = local
        except:
            pass
        
        # Check OpenAI
        try:
            openai = OpenAIImage()
            if openai._available:
                self.providers["openai"] = openai
        except:
            pass
        
        # Check Replicate
        try:
            replicate = ReplicateImage()
            if replicate._available:
                self.providers["replicate"] = replicate
        except:
            pass
        
        # Select active
        if preferred != "auto" and preferred in self.providers:
            self._active = preferred
        elif "openai" in self.providers:
            self._active = "openai"
        elif "replicate" in self.providers:
            self._active = "replicate"
        elif "local" in self.providers:
            self._active = "local"
        
        if self._active:
            print(f"[IMAGE] Using provider: {self._active}")
    
    def generate(self, prompt: str, provider: str = None, **kwargs) -> Optional[str]:
        """Generate image from prompt."""
        provider = provider or self._active
        if provider and provider in self.providers:
            return self.providers[provider].generate(prompt, **kwargs)
        print("[IMAGE] No provider available!")
        return None
    
    def list_providers(self) -> List[str]:
        """List available providers."""
        return list(self.providers.keys())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_image(prompt: str, **kwargs) -> Optional[str]:
    """
    Quick image generation.
    
    Args:
        prompt: Image description
        **kwargs: Provider options
    
    Returns:
        Path to generated image
    """
    controller = ImageController()
    return controller.generate(prompt, **kwargs)


# =============================================================================
# MAIN - Run this file directly to test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ForgeAI Image Generation Example")
    print("=" * 60)
    
    # Initialize
    print("\n[1] Checking available providers...")
    controller = ImageController()
    
    print(f"Available: {controller.list_providers()}")
    print(f"Active: {controller._active}")
    
    if not controller._active:
        print("\nNo image generation providers available!")
        print("\nOptions:")
        print("  1. Local (GPU): pip install diffusers torch")
        print("  2. OpenAI: Set OPENAI_API_KEY")
        print("  3. Replicate: Set REPLICATE_API_TOKEN")
    else:
        # Generate test image
        print("\n[2] Generating test image...")
        
        prompt = "A cute robot cat sitting on a windowsill, digital art"
        
        output = controller.generate(
            prompt,
            width=512,
            height=512,
        )
        
        if output:
            print(f"\nGenerated: {output}")
            print(f"Size: {Path(output).stat().st_size} bytes")
        else:
            print("Generation failed!")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nProvider setup:")
    print("  Local:     pip install diffusers transformers torch")
    print("  OpenAI:    export OPENAI_API_KEY='sk-...'")
    print("  Replicate: export REPLICATE_API_TOKEN='r8_...'")
