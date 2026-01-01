"""
GGUF Model Loader
=================

Load and run GGUF format models (llama.cpp compatible).
Enables efficient CPU/GPU inference with quantized models.

Usage:
    from enigma.core.gguf_loader import GGUFModel
    
    model = GGUFModel("path/to/model.gguf")
    model.load()
    
    response = model.generate("Hello, how are you?")
    print(response)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import os

logger = logging.getLogger(__name__)

# Check for llama-cpp-python
HAVE_LLAMA_CPP = False
try:
    from llama_cpp import Llama
    HAVE_LLAMA_CPP = True
except ImportError:
    logger.warning("llama-cpp-python not available - GGUF loading disabled")


class GGUFModel:
    """
    GGUF model loader using llama.cpp bindings.
    
    Supports efficient inference with quantized models on CPU and GPU.
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        n_threads: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize GGUF model.
        
        Args:
            model_path: Path to .gguf model file
            n_ctx: Context window size (max tokens)
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
            n_threads: Number of CPU threads (None = auto)
            verbose: Enable verbose logging
        """
        if not HAVE_LLAMA_CPP:
            raise RuntimeError(
                "GGUF loading requires llama-cpp-python. "
                "Install with: pip install llama-cpp-python"
            )
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.verbose = verbose
        
        self.model = None
        self.is_loaded = False
        
        logger.info(f"GGUF model initialized: {model_path}")
    
    def load(self) -> bool:
        """
        Load the GGUF model into memory.
        
        Returns:
            True if loaded successfully
        """
        if self.is_loaded:
            logger.warning("Model already loaded")
            return True
        
        try:
            logger.info(f"Loading GGUF model from {self.model_path}...")
            
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=self.verbose
            )
            
            self.is_loaded = True
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"  Context size: {self.n_ctx}")
            logger.info(f"  GPU layers: {self.n_gpu_layers}")
            logger.info(f"  Threads: {self.n_threads or 'auto'}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            return False
    
    def unload(self):
        """Unload the model from memory."""
        if self.model:
            del self.model
            self.model = None
            self.is_loaded = False
            logger.info("Model unloaded")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 - 2.0)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repeat_penalty: Penalty for repeating tokens
            stop: Stop sequences
            stream: Enable streaming output
            **kwargs: Additional llama.cpp parameters
            
        Returns:
            Generated text
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            if stream:
                # Streaming generation
                output = ""
                for chunk in self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=stop or [],
                    stream=True,
                    **kwargs
                ):
                    text = chunk['choices'][0]['text']
                    output += text
                    print(text, end='', flush=True)
                print()  # Newline after streaming
                return output
            else:
                # Standard generation
                response = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=stop or [],
                    **kwargs
                )
                return response['choices'][0]['text']
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.8,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Chat completion (if model supports it).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Enable streaming
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                **kwargs
            )
            
            if stream:
                output = ""
                for chunk in response:
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            text = delta['content']
                            output += text
                            print(text, end='', flush=True)
                print()
                return output
            else:
                return response['choices'][0]['message']['content']
        
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to token IDs."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model.tokenize(text.encode('utf-8'))
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model.detokenize(tokens).decode('utf-8')
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_path': str(self.model_path),
            'model_name': self.model_path.name,
            'is_loaded': self.is_loaded,
            'context_size': self.n_ctx,
            'gpu_layers': self.n_gpu_layers,
            'threads': self.n_threads,
            'file_size_mb': self.model_path.stat().st_size / (1024 * 1024) if self.model_path.exists() else 0
        }
    
    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return f"GGUFModel({self.model_path.name}, {status})"
    
    def __del__(self):
        """Cleanup on deletion."""
        self.unload()


def list_gguf_models(models_dir: str = None) -> List[Path]:
    """
    List all GGUF model files in a directory.
    
    Args:
        models_dir: Directory to search (default: models/)
        
    Returns:
        List of Path objects for .gguf files
    """
    if models_dir is None:
        from enigma.config import CONFIG
        models_dir = CONFIG['models_dir']
    
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    
    # Find all .gguf files recursively
    gguf_files = list(models_path.rglob("*.gguf"))
    return sorted(gguf_files)


def recommend_gpu_layers(model_size_gb: float, vram_gb: float) -> int:
    """
    Recommend number of GPU layers based on model size and available VRAM.
    
    Args:
        model_size_gb: Model file size in GB
        vram_gb: Available VRAM in GB
        
    Returns:
        Recommended number of layers to offload
    """
    # Rough heuristic: each layer uses about 5% of model size in VRAM
    # Leave some VRAM for context and other operations
    usable_vram = vram_gb * 0.8  # Use 80% of VRAM
    
    # Estimate layers that fit
    if model_size_gb >= usable_vram:
        # Can't fit entire model
        ratio = usable_vram / model_size_gb
        # Rough estimate: 32 layers for 7B model, scale from there
        estimated_layers = int(32 * ratio)
        return max(0, estimated_layers)
    else:
        # Can fit entire model - use all layers
        return 999  # Use a large number to offload all layers


def test_gguf_loading(model_path: str = None):
    """Test function to verify GGUF loading works."""
    print("Testing GGUF loading...")
    
    if not HAVE_LLAMA_CPP:
        print("❌ llama-cpp-python not available")
        print("Install with: pip install llama-cpp-python")
        return False
    
    if model_path is None:
        # Try to find a GGUF model
        models = list_gguf_models()
        if not models:
            print("❌ No GGUF models found in models/ directory")
            return False
        model_path = str(models[0])
        print(f"Using model: {model_path}")
    
    try:
        model = GGUFModel(model_path, n_ctx=512, verbose=False)
        
        if model.load():
            print("✓ Model loaded successfully")
            
            # Test generation
            response = model.generate("Hello!", max_tokens=20)
            print(f"✓ Generated: {response[:50]}...")
            
            model.unload()
            print("✓ Model unloaded")
            return True
        else:
            print("❌ Failed to load model")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_gguf_loading(sys.argv[1])
    else:
        test_gguf_loading()
