"""
Image Generation Mod for Enigma AI Engine

Generate images from text prompts using various backends:
- Stable Diffusion WebUI (AUTOMATIC1111) API
- ComfyUI API
- Local diffusers (requires GPU + dependencies)

Threading model:
    All handlers are synchronous. Each mod runs in its own subprocess,
    so blocking HTTP requests and heavy computation are fine.

Usage:
    python -m mods.imagegen.main

The mod automatically detects available backends and uses the first one found.
"""

import base64
import json
import logging
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

# Import the base class — handles connection, protocol, registration
from mod_base import ModClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not installed - HTTP backends disabled")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("PIL not installed - image saving disabled")


class ImageGenMod(ModClient):
    """Image generation mod with multiple backend support.

    Inherits connection/protocol from ModClient.
    Only adds backend detection and cmd_* handlers.
    """

    def __init__(self, config_path: Path = None):
        """Initialize mod from config."""
        super().__init__(config_path)

        self.settings = self.config.get("settings", {})

        # Output directory
        self.output_dir = Path(self.settings.get("output_dir", "outputs/images"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Detect available backend
        self.backend = None
        self._backend_info: Dict[str, Any] = {}
        self._detect_backend()

        logger.info(f"Active backend: {self.backend or 'none'}")
    
    def _detect_backend(self):
        """Detect which backend is available."""
        backend_pref = self.settings.get("backend", "auto")
        backends = self.settings.get("backends", {})
        
        # If specific backend requested
        if backend_pref != "auto" and backend_pref in backends:
            if self._check_backend(backend_pref, backends[backend_pref]):
                self.backend = backend_pref
                return
        
        # Auto-detect: try each enabled backend
        for name, config in backends.items():
            if config.get("enabled", False):
                if self._check_backend(name, config):
                    self.backend = name
                    return
        
        logger.warning("No image generation backend available")
    
    def _check_backend(self, name: str, config: Dict) -> bool:
        """Check if a backend is available."""
        if not HAS_REQUESTS:
            return False
        
        try:
            if name == "sdwebui":
                url = config.get("url", "http://127.0.0.1:7860")
                r = requests.get(f"{url}/sdapi/v1/sd-models", timeout=2)
                if r.status_code == 200:
                    self._backend_info["models"] = [m["title"] for m in r.json()]
                    self._backend_info["url"] = url
                    logger.info(f"SD WebUI available at {url}")
                    return True
            
            elif name == "comfyui":
                url = config.get("url", "http://127.0.0.1:8188")
                r = requests.get(f"{url}/system_stats", timeout=2)
                if r.status_code == 200:
                    self._backend_info["url"] = url
                    logger.info(f"ComfyUI available at {url}")
                    return True
            
            elif name == "diffusers":
                # Check if diffusers is installed
                try:
                    import torch
                    from diffusers import StableDiffusionPipeline
                    self._backend_info["model"] = config.get("model", "runwayml/stable-diffusion-v1-5")
                    self._backend_info["device"] = config.get("device", "auto")
                    return True
                except ImportError:
                    return False
        
        except Exception as e:
            logger.debug(f"Backend {name} not available: {e}")
        
        return False
    
    def cmd_generate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an image from a text prompt."""
        prompt = args.get("prompt", "")
        if not prompt:
            raise ValueError("Prompt is required")
        
        negative_prompt = args.get("negative_prompt", "")
        width = args.get("width", 512)
        height = args.get("height", 512)
        steps = args.get("steps", 20)
        cfg_scale = args.get("cfg_scale", 7.5)
        seed = args.get("seed", -1)
        
        if not self.backend:
            raise RuntimeError("No image generation backend available. Please start Stable Diffusion WebUI or install diffusers.")
        
        # Generate based on backend
        if self.backend == "sdwebui":
            return self._generate_sdwebui(prompt, negative_prompt, width, height, steps, cfg_scale, seed)
        elif self.backend == "comfyui":
            return self._generate_comfyui(prompt, negative_prompt, width, height, steps, cfg_scale, seed)
        elif self.backend == "diffusers":
            return self._generate_diffusers(prompt, negative_prompt, width, height, steps, cfg_scale, seed)
        
        raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def _generate_sdwebui(self, prompt: str, negative_prompt: str, 
                                 width: int, height: int, steps: int, 
                                 cfg_scale: float, seed: int) -> Dict[str, Any]:
        """Generate using Stable Diffusion WebUI API."""
        url = self._backend_info.get("url", "http://127.0.0.1:7860")
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed
        }
        
        response = requests.post(f"{url}/sdapi/v1/txt2img", json=payload, timeout=120)
        
        if response.status_code != 200:
            raise RuntimeError(f"SD WebUI error: {response.status_code}")
        
        result = response.json()
        images = result.get("images", [])
        
        if not images:
            raise RuntimeError("No images returned")
        
        # Save image
        saved_path = None
        if self.settings.get("save_images", True) and HAS_PIL:
            img_data = base64.b64decode(images[0])
            img = Image.open(BytesIO(img_data))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"img_{timestamp}.png"
            saved_path = str(self.output_dir / filename)
            img.save(saved_path)
            logger.info(f"Saved image: {saved_path}")
        
        # Get actual seed used
        info = json.loads(result.get("info", "{}"))
        actual_seed = info.get("seed", seed)
        
        return {
            "image_base64": images[0],
            "saved_path": saved_path,
            "seed": actual_seed,
            "backend": "sdwebui"
        }
    
    def _generate_comfyui(self, prompt: str, negative_prompt: str,
                                 width: int, height: int, steps: int,
                                 cfg_scale: float, seed: int) -> Dict[str, Any]:
        """Generate using ComfyUI API."""
        url = self._backend_info.get("url", "http://127.0.0.1:8188")
        
        # Basic txt2img workflow
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg_scale,
                    "denoise": 1,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "seed": seed if seed > 0 else 42,
                    "steps": steps
                }
            },
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "height": height, "width": width}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": prompt}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": negative_prompt}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "enigma", "images": ["8", 0]}}
        }
        
        # Queue prompt
        response = requests.post(f"{url}/prompt", json={"prompt": workflow}, timeout=120)
        
        if response.status_code != 200:
            raise RuntimeError(f"ComfyUI error: {response.status_code}")
        
        result = response.json()
        prompt_id = result.get("prompt_id")
        
        # Poll for completion
        for _ in range(60):
            time.sleep(2)
            history = requests.get(f"{url}/history/{prompt_id}", timeout=5)
            if history.status_code == 200:
                hist_data = history.json()
                if prompt_id in hist_data:
                    # Get output image path
                    outputs = hist_data[prompt_id].get("outputs", {})
                    if "9" in outputs and outputs["9"].get("images"):
                        img_info = outputs["9"]["images"][0]
                        return {
                            "saved_path": img_info.get("filename"),
                            "backend": "comfyui",
                            "subfolder": img_info.get("subfolder", "")
                        }
        
        raise RuntimeError("ComfyUI generation timeout")
    
    def _generate_diffusers(self, prompt: str, negative_prompt: str,
                                   width: int, height: int, steps: int,
                                   cfg_scale: float, seed: int) -> Dict[str, Any]:
        """Generate using local diffusers library."""
        import torch
        from diffusers import StableDiffusionPipeline
        
        model = self._backend_info.get("model", "runwayml/stable-diffusion-v1-5")
        device_pref = self._backend_info.get("device", "auto")
        
        # Determine device
        if device_pref == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_pref
        
        # Load pipeline (would cache in real impl)
        pipe = StableDiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)
        
        # Generate
        generator = None
        if seed > 0:
            generator = torch.Generator(device).manual_seed(seed)
        
        image = pipe(
            prompt,
            negative_prompt=negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator
        ).images[0]
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"img_{timestamp}.png"
        saved_path = str(self.output_dir / filename)
        image.save(saved_path)
        
        # Encode to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "image_base64": img_base64,
            "saved_path": saved_path,
            "seed": seed,
            "backend": "diffusers"
        }
    
    def cmd_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return mod status."""
        return {
            "name": self.name,
            "running": self.running,
            "backend": self.backend,
            "backend_info": self._backend_info,
            "output_dir": str(self.output_dir),
        }

    def cmd_stop(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Stop the mod."""
        self.running = False
        return {"stopped": True}

    def cmd_list_models(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List available models."""
        models = []

        if self.backend == "sdwebui" and "models" in self._backend_info:
            models = self._backend_info["models"]
        elif self.backend == "diffusers":
            models = [self._backend_info.get("model", "unknown")]

        return {"models": models, "backend": self.backend}

    def cmd_set_model(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Set the active model."""
        model = args.get("model", "")
        if not model:
            raise ValueError("Model name required")

        if self.backend == "sdwebui":
            url = self._backend_info.get("url", "http://127.0.0.1:7860")

            response = requests.post(
                f"{url}/sdapi/v1/options",
                json={"sd_model_checkpoint": model},
                timeout=60,
            )

            if response.status_code != 200:
                raise RuntimeError(f"Failed to set model: {response.status_code}")

            return {"model": model, "status": "changed"}

        return {"error": "Model switching not supported for this backend"}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mod = ImageGenMod()
    try:
        mod.run()
    except KeyboardInterrupt:
        logger.info("Mod stopped by user")
