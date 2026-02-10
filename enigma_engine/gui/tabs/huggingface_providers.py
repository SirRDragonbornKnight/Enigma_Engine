"""
HuggingFace Free Inference API - Access free models on HuggingFace.

This provider uses the HuggingFace Inference API which has a FREE tier:
- Free tier: Rate limited but no cost
- Pro tier: Higher limits with HF_TOKEN

Models available for free:
- Image: stabilityai/stable-diffusion-xl-base-1.0, runwayml/stable-diffusion-v1-5
- Audio: facebook/mms-tts-eng, hexgrad/Kokoro-82M
- Text: Many LLMs and chat models
"""

import os
import time
from pathlib import Path
from typing import Any, Optional

# Try to get HF token from environment (optional - free tier works without)
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


class HuggingFaceImage:
    """
    Free HuggingFace Inference API for image generation.
    
    Works without API key (rate limited) or with HF_TOKEN for higher limits.
    """
    
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        self.model_id = model_id
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.is_loaded = False
        self.headers = {}
        if HF_TOKEN:
            self.headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    def load(self) -> bool:
        """No loading needed - API is always available."""
        try:
            self.is_loaded = True
            return True
        except ImportError:
            print("Install: pip install requests")
            return False
    
    def unload(self):
        self.is_loaded = False
    
    def generate(self, prompt: str, width: int = 512, height: int = 512,
                 negative_prompt: str = "", **kwargs) -> dict[str, Any]:
        """Generate image using HuggingFace Inference API."""
        if not self.is_loaded:
            self.load()
        
        try:
            import requests
            start = time.time()
            
            payload = {
                "inputs": str(prompt).strip(),
                "parameters": {
                    "width": min(width, 1024),  # API has limits
                    "height": min(height, 1024),
                }
            }
            
            if negative_prompt:
                payload["parameters"]["negative_prompt"] = str(negative_prompt).strip()
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 503:
                # Model is loading - wait and retry
                try:
                    data = response.json()
                    wait_time = data.get("estimated_time", 20)
                    print(f"Model loading, waiting {wait_time}s...")
                    time.sleep(min(wait_time, 30))
                    response = requests.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload,
                        timeout=120
                    )
                except Exception:
                    pass
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"API error {response.status_code}: {response.text[:200]}"
                }
            
            # Save the image
            from ...config import CONFIG
            output_dir = Path(CONFIG.get("outputs_dir", "outputs")) / "images"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"hf_{timestamp}.png"
            filepath = output_dir / filename
            filepath.write_bytes(response.content)
            
            return {
                "success": True,
                "path": str(filepath),
                "duration": time.time() - start,
                "model": self.model_id
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class HuggingFaceTTS:
    """
    Free HuggingFace Inference API for text-to-speech.
    
    Models:
    - facebook/mms-tts-eng - Multilingual TTS
    - hexgrad/Kokoro-82M - Small, fast TTS
    """
    
    def __init__(self, model_id: str = "facebook/mms-tts-eng"):
        self.model_id = model_id
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.is_loaded = False
        self.headers = {}
        if HF_TOKEN:
            self.headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    def load(self) -> bool:
        try:
            self.is_loaded = True
            return True
        except ImportError:
            print("Install: pip install requests")
            return False
    
    def unload(self):
        self.is_loaded = False
    
    def generate(self, text: str, **kwargs) -> dict[str, Any]:
        """Generate audio using HuggingFace Inference API."""
        if not self.is_loaded:
            self.load()
        
        try:
            import requests
            start = time.time()
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": str(text).strip()},
                timeout=60
            )
            
            if response.status_code == 503:
                # Model loading
                try:
                    data = response.json()
                    wait_time = data.get("estimated_time", 20)
                    print(f"Model loading, waiting {wait_time}s...")
                    time.sleep(min(wait_time, 30))
                    response = requests.post(
                        self.api_url,
                        headers=self.headers,
                        json={"inputs": text},
                        timeout=60
                    )
                except Exception:
                    pass
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"API error {response.status_code}: {response.text[:200]}"
                }
            
            # Save the audio
            from ...config import CONFIG
            output_dir = Path(CONFIG.get("outputs_dir", "outputs")) / "audio"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            # Determine extension based on content type
            content_type = response.headers.get("content-type", "audio/wav")
            ext = ".wav" if "wav" in content_type else ".mp3" if "mp3" in content_type else ".flac"
            
            filename = f"hf_tts_{timestamp}{ext}"
            filepath = output_dir / filename
            filepath.write_bytes(response.content)
            
            return {
                "success": True,
                "path": str(filepath),
                "duration": time.time() - start,
                "model": self.model_id
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class HuggingFaceChat:
    """
    Free HuggingFace Inference API for chat/text generation.
    
    Models:
    - microsoft/DialoGPT-medium
    - Qwen/Qwen2.5-0.5B-Instruct (if available)
    - TinyLlama/TinyLlama-1.1B-Chat-v1.0
    """
    
    def __init__(self, model_id: str = "microsoft/DialoGPT-medium"):
        self.model_id = model_id
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.is_loaded = False
        self.headers = {}
        if HF_TOKEN:
            self.headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    def load(self) -> bool:
        try:
            self.is_loaded = True
            return True
        except ImportError:
            return False
    
    def unload(self):
        self.is_loaded = False
    
    def generate(self, prompt: str, max_tokens: int = 200, 
                 temperature: float = 0.7, **kwargs) -> dict[str, Any]:
        """Generate text using HuggingFace Inference API."""
        if not self.is_loaded:
            self.load()
        
        try:
            import requests
            start = time.time()
            
            payload = {
                "inputs": str(prompt).strip(),
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "return_full_text": False,
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 503:
                # Model loading
                try:
                    data = response.json()
                    wait_time = data.get("estimated_time", 20)
                    print(f"Model loading, waiting {wait_time}s...")
                    time.sleep(min(wait_time, 30))
                    response = requests.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload,
                        timeout=60
                    )
                except Exception:
                    pass
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"API error {response.status_code}: {response.text[:200]}"
                }
            
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, list) and len(data) > 0:
                text = data[0].get("generated_text", "")
            elif isinstance(data, dict):
                text = data.get("generated_text", "")
            else:
                text = str(data)
            
            return {
                "success": True,
                "text": text,
                "duration": time.time() - start,
                "model": self.model_id
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class HuggingFaceEmbedding:
    """
    Free HuggingFace Inference API for embeddings.
    
    Models:
    - sentence-transformers/all-MiniLM-L6-v2
    - BAAI/bge-small-en-v1.5
    """
    
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_id = model_id
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
        self.is_loaded = False
        self.headers = {}
        if HF_TOKEN:
            self.headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    def load(self) -> bool:
        try:
            self.is_loaded = True
            return True
        except ImportError:
            return False
    
    def unload(self):
        self.is_loaded = False
    
    def embed(self, texts: list, **kwargs) -> dict[str, Any]:
        """Generate embeddings using HuggingFace Inference API."""
        if not self.is_loaded:
            self.load()
        
        try:
            import requests
            start = time.time()
            
            if isinstance(texts, str):
                texts = [texts]
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": texts},
                timeout=60
            )
            
            if response.status_code == 503:
                try:
                    data = response.json()
                    wait_time = data.get("estimated_time", 10)
                    time.sleep(min(wait_time, 20))
                    response = requests.post(
                        self.api_url,
                        headers=self.headers,
                        json={"inputs": texts},
                        timeout=60
                    )
                except Exception:
                    pass
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"API error {response.status_code}: {response.text[:200]}"
                }
            
            embeddings = response.json()
            
            return {
                "success": True,
                "embeddings": embeddings,
                "duration": time.time() - start,
                "model": self.model_id
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# Factory function
def get_hf_provider(tool_type: str, model_id: Optional[str] = None):
    """
    Get a HuggingFace provider for a specific tool type.
    
    Args:
        tool_type: "image", "audio", "chat", or "embedding"
        model_id: Optional specific model ID
    
    Returns:
        Provider instance
    """
    defaults = {
        "image": "stabilityai/stable-diffusion-xl-base-1.0",
        "audio": "facebook/mms-tts-eng",
        "chat": "microsoft/DialoGPT-medium",
        "embedding": "sentence-transformers/all-MiniLM-L6-v2",
    }
    
    model = model_id or defaults.get(tool_type, "")
    
    if tool_type == "image":
        return HuggingFaceImage(model)
    elif tool_type == "audio":
        return HuggingFaceTTS(model)
    elif tool_type == "chat":
        return HuggingFaceChat(model)
    elif tool_type == "embedding":
        return HuggingFaceEmbedding(model)
    else:
        raise ValueError(f"Unknown tool type: {tool_type}")
