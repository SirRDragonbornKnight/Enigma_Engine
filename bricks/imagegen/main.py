"""
Image Generation Brick for Enigma AI Engine

Generate images from text prompts using various backends:
- Stable Diffusion WebUI (AUTOMATIC1111) API
- ComfyUI API
- Local diffusers (requires GPU + dependencies)

Usage:
    python -m bricks.imagegen.main

The brick automatically detects available backends and uses the first one found.
"""

import asyncio
import base64
import json
import logging
import os
import struct
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class ImageGenBrick:
    """Image generation brick with multiple backend support."""
    
    def __init__(self, config_path: Path = None):
        """Initialize brick from config."""
        if config_path is None:
            config_path = Path(__file__).parent / "brick.json"
        
        with open(config_path, encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.name = self.config.get("name", "Image Generation Brick")
        self.port = self.config.get("port", 9902)
        self.settings = self.config.get("settings", {})
        self.running = False
        
        # Router connection
        self.router_host = "127.0.0.1"
        self.router_port = 9900
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        
        # Output directory
        self.output_dir = Path(self.settings.get("output_dir", "outputs/images"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect available backend
        self.backend = None
        self._backend_info = {}
        self._detect_backend()
        
        logger.info(f"Loaded brick: {self.name} on port {self.port}")
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
    
    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming message."""
        msg_type = message.get("type", "")
        msg_id = message.get("id", "")
        data = message.get("data", {})
        
        if msg_type == "command":
            command = data.get("command", "")
            args = data.get("args", {})
            
            handler = getattr(self, f"cmd_{command}", None)
            if handler:
                try:
                    result = await handler(args)
                    return {"id": msg_id, "type": "response", "success": True, "data": result}
                except Exception as e:
                    logger.exception(f"Command {command} failed")
                    return {"id": msg_id, "type": "error", "success": False, "error": str(e)}
            else:
                return {"id": msg_id, "type": "error", "success": False, "error": f"Unknown command: {command}"}
        
        return {"id": msg_id, "type": "error", "success": False, "error": f"Unknown message type: {msg_type}"}
    
    async def cmd_generate(self, args: Dict[str, Any]) -> Dict[str, Any]:
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
            return await self._generate_sdwebui(prompt, negative_prompt, width, height, steps, cfg_scale, seed)
        elif self.backend == "comfyui":
            return await self._generate_comfyui(prompt, negative_prompt, width, height, steps, cfg_scale, seed)
        elif self.backend == "diffusers":
            return await self._generate_diffusers(prompt, negative_prompt, width, height, steps, cfg_scale, seed)
        
        raise RuntimeError(f"Unknown backend: {self.backend}")
    
    async def _generate_sdwebui(self, prompt: str, negative_prompt: str, 
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
        
        # Run in executor to not block
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(f"{url}/sdapi/v1/txt2img", json=payload, timeout=120)
        )
        
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
    
    async def _generate_comfyui(self, prompt: str, negative_prompt: str,
                                 width: int, height: int, steps: int,
                                 cfg_scale: float, seed: int) -> Dict[str, Any]:
        """Generate using ComfyUI API."""
        # ComfyUI requires a workflow - use a simple txt2img workflow
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
        
        loop = asyncio.get_event_loop()
        
        # Queue prompt
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(f"{url}/prompt", json={"prompt": workflow}, timeout=120)
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"ComfyUI error: {response.status_code}")
        
        result = response.json()
        prompt_id = result.get("prompt_id")
        
        # Poll for completion (simplified - real impl would use websocket)
        for _ in range(60):
            await asyncio.sleep(2)
            history = await loop.run_in_executor(
                None,
                lambda: requests.get(f"{url}/history/{prompt_id}", timeout=5)
            )
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
    
    async def _generate_diffusers(self, prompt: str, negative_prompt: str,
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
        
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(
            None,
            lambda: pipe(
                prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator
            ).images[0]
        )
        
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
    
    async def cmd_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return brick status."""
        return {
            "name": self.name,
            "running": self.running,
            "port": self.port,
            "backend": self.backend,
            "backend_info": self._backend_info,
            "output_dir": str(self.output_dir)
        }
    
    async def cmd_stop(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Stop the brick."""
        self.running = False
        return {"stopped": True}
    
    async def cmd_list_models(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List available models."""
        models = []
        
        if self.backend == "sdwebui" and "models" in self._backend_info:
            models = self._backend_info["models"]
        elif self.backend == "diffusers":
            models = [self._backend_info.get("model", "unknown")]
        
        return {"models": models, "backend": self.backend}
    
    async def cmd_set_model(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Set the active model."""
        model = args.get("model", "")
        if not model:
            raise ValueError("Model name required")
        
        if self.backend == "sdwebui":
            url = self._backend_info.get("url", "http://127.0.0.1:7860")
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{url}/sdapi/v1/options",
                    json={"sd_model_checkpoint": model},
                    timeout=60
                )
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Failed to set model: {response.status_code}")
            
            return {"model": model, "status": "changed"}
        
        return {"error": "Model switching not supported for this backend"}
    
    # =========================================================================
    # ROUTER CONNECTION (Client Protocol)
    # =========================================================================
    
    async def connect(self) -> bool:
        """Connect to the router."""
        try:
            self.reader, self.writer = await asyncio.open_connection(
                self.router_host, self.router_port
            )
            logger.info(f"Connected to router at {self.router_host}:{self.router_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to router: {e}")
            return False
    
    async def send_message(self, data: Dict) -> bool:
        """Send a message to the router using length-prefixed protocol."""
        if not self.writer:
            return False
        
        try:
            msg = json.dumps(data).encode('utf-8')
            length = struct.pack('>I', len(msg))  # 4-byte big-endian
            self.writer.write(length + msg)
            await self.writer.drain()
            return True
        except Exception as e:
            logger.error(f"Send error: {e}")
            return False
    
    async def receive_message(self) -> Optional[Dict]:
        """Receive a message from the router using length-prefixed protocol."""
        if not self.reader:
            return None
        
        try:
            # Read 4-byte length prefix
            length_data = await self.reader.readexactly(4)
            length = struct.unpack('>I', length_data)[0]
            
            if length > 10_000_000:  # 10MB max for images
                logger.warning(f"Message too large: {length}")
                return None
            
            # Read message
            data = await self.reader.readexactly(length)
            return json.loads(data.decode('utf-8'))
        
        except asyncio.IncompleteReadError:
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            return None
        except Exception as e:
            logger.debug(f"Receive error: {e}")
            return None
    
    async def register(self) -> bool:
        """Register with the router."""
        register_msg = {
            "type": "register",
            "brick_id": self.config.get("id", "imagegen"),
            "name": self.name,
            "capabilities": [cmd.get("name", "") for cmd in self.config.get("commands", [])]
        }
        
        if not await self.send_message(register_msg):
            return False
        
        # Wait for acknowledgment
        response = await self.receive_message()
        if response and response.get("type") == "registered":
            logger.info(f"Registered with router: {response}")
            return True
        
        logger.error(f"Registration failed: {response}")
        return False
    
    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming message from router."""
        msg_type = message.get("type", "")
        msg_id = message.get("id", "")
        data = message.get("data", {})
        
        if msg_type == "command":
            command = data.get("command", "")
            args = data.get("args", {})
            
            handler = getattr(self, f"cmd_{command}", None)
            if handler:
                try:
                    result = await handler(args)
                    return {"id": msg_id, "type": "response", "success": True, "data": result}
                except Exception as e:
                    logger.exception(f"Command {command} failed")
                    return {"id": msg_id, "type": "error", "success": False, "error": str(e)}
            else:
                return {"id": msg_id, "type": "error", "success": False, "error": f"Unknown command: {command}"}
        
        elif msg_type == "ping":
            return {"id": msg_id, "type": "pong"}
        
        return {"id": msg_id, "type": "error", "success": False, "error": f"Unknown message type: {msg_type}"}
    
    async def run(self):
        """Main loop: connect to router and handle messages."""
        self.running = True
        
        # Connect to router
        if not await self.connect():
            logger.error("Could not connect to router. Is it running?")
            logger.info("Start the router in GUI before running bricks.")
            return
        
        # Register
        if not await self.register():
            logger.error("Registration failed")
            return
        
        logger.info(f"Brick '{self.name}' ready and listening for commands")
        logger.info(f"Active backend: {self.backend or 'none'}")
        
        # Message loop
        try:
            while self.running:
                message = await self.receive_message()
                if message is None:
                    logger.warning("Connection to router lost")
                    break
                
                # Handle message and send response
                response = await self.handle_message(message)
                await self.send_message(response)
        
        except asyncio.CancelledError:
            logger.info("Brick cancelled")
        except Exception as e:
            logger.exception(f"Brick error: {e}")
        finally:
            if self.writer:
                self.writer.close()
                try:
                    await self.writer.wait_closed()
                except Exception:
                    pass
            logger.info("Brick stopped")


async def main():
    """Main entry point."""
    brick = ImageGenBrick()
    await brick.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Brick stopped by user")
