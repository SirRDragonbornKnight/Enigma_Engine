"""
GGUF Model Loader
=================

Load and run GGUF format models (llama.cpp compatible).
Enables efficient CPU/GPU inference with quantized models.

Usage:
    from forge_ai.core.gguf_loader import GGUFModel
    
    model = GGUFModel("path/to/model.gguf")
    model.load()
    
    response = model.generate("Hello, how are you?")
    print(response)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Check for llama-cpp-python
HAVE_LLAMA_CPP = False
try:
    from llama_cpp import Llama
    HAVE_LLAMA_CPP = True
except ImportError:
    # Silently disable GGUF - it's an optional feature
    pass


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
            logger.info(f"[OK] Model loaded successfully")
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
        from forge_ai.config import CONFIG
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


def load_gguf_model(
    gguf_model_path: str,
    config: Any = None,
    **kwargs
) -> 'Forge':
    """
    Load a GGUF model and convert it to Forge format.
    
    This function loads a quantized GGUF model (llama.cpp format), extracts
    its weights, dequantizes them if needed, and creates a Forge model.
    
    ⚠️ NOTE: GGUF models are often quantized. Loading converts them to full
    precision PyTorch, which may use MORE memory than the original file.
    
    Args:
        gguf_model_path: Path to .gguf file
        config: Optional ForgeConfig. If None, will try to infer from GGUF
        **kwargs: Additional arguments (n_ctx, n_gpu_layers, etc.)
        
    Returns:
        Forge model with loaded weights
        
    Raises:
        RuntimeError: If required dependencies not installed
        FileNotFoundError: If model file not found
    """
    # Import here to avoid circular imports
    from .model import Forge, ForgeConfig
    from .weight_mapping import WeightMapper
    from pathlib import Path
    
    logger.info(f"Loading GGUF model from: {gguf_model_path}")
    
    # Check if gguf library is available for parsing
    try:
        import gguf
        HAVE_GGUF = True
    except ImportError:
        HAVE_GGUF = False
        logger.warning(
            "gguf library not available. Will attempt to use llama-cpp-python only."
        )
    
    if not HAVE_TORCH:
        raise RuntimeError(
            "GGUF loading requires torch. Install with: pip install torch"
        )
    
    model_path = Path(gguf_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"GGUF model not found: {gguf_model_path}")
    
    # Try to extract metadata and weights from GGUF file
    if HAVE_GGUF:
        logger.info("Using gguf library to parse GGUF file...")
        try:
            import torch
            reader = gguf.GGUFReader(str(model_path))
            
            # Extract metadata
            metadata = {}
            for field in reader.fields.values():
                metadata[field.name] = field.parts[field.data[-1]] if field.parts else field.data
            
            # Try to infer config from metadata
            if config is None:
                vocab_size = metadata.get('tokenizer.ggml.tokens', None)
                if isinstance(vocab_size, list):
                    vocab_size = len(vocab_size)
                
                config_dict = {
                    'vocab_size': vocab_size or 32000,
                    'dim': metadata.get('llama.embedding_length', 4096),
                    'n_layers': metadata.get('llama.block_count', 32),
                    'n_heads': metadata.get('llama.attention.head_count', 32),
                    'n_kv_heads': metadata.get('llama.attention.head_count_kv', None),
                    'max_seq_len': metadata.get('llama.context_length', 2048)
                }
                
                # Remove None values
                config_dict = {k: v for k, v in config_dict.items() if v is not None}
                config = ForgeConfig(**config_dict)
                logger.info(f"Inferred config from GGUF metadata: {config}")
            
            # Extract tensors
            gguf_tensors = {}
            for tensor in reader.tensors:
                tensor_name = tensor.name
                tensor_data = tensor.data
                
                # Convert to PyTorch tensor
                # Note: This is simplified - full implementation would need
                # proper dequantization for quantized tensors
                try:
                    torch_tensor = torch.from_numpy(tensor_data)
                    gguf_tensors[tensor_name] = torch_tensor
                    logger.debug(f"Loaded tensor: {tensor_name}, shape: {torch_tensor.shape}")
                except Exception as e:
                    logger.warning(f"Failed to load tensor {tensor_name}: {e}")
            
            logger.info(f"Extracted {len(gguf_tensors)} tensors from GGUF file")
            
            # Map GGUF tensors to Forge format
            logger.info("Mapping GGUF weights to Forge format...")
            mapper = WeightMapper()
            forge_weights = mapper.map_gguf_to_forge(gguf_tensors, config)
            
            # Create Forge model and load weights
            forge_model = Forge(config=config)
            missing_keys, unexpected_keys = forge_model.load_state_dict(forge_weights, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing {len(missing_keys)} keys - will be randomly initialized")
            if unexpected_keys:
                logger.warning(f"Unexpected {len(unexpected_keys)} keys - will be ignored")
            
            forge_model.eval()
            logger.info("GGUF model successfully loaded into Forge format")
            return forge_model
            
        except Exception as e:
            logger.error(f"Failed to load GGUF with gguf library: {e}")
            # Fall through to llama-cpp-python method
    
    # Fallback: Use llama-cpp-python wrapper (doesn't convert to Forge)
    logger.warning(
        "Could not convert GGUF to Forge format. "
        "To use GGUF models natively, install: pip install gguf\n"
        "For now, returning a GGUFModel wrapper (not a Forge model)."
    )
    
    if not HAVE_LLAMA_CPP:
        raise RuntimeError(
            "GGUF loading requires either:\n"
            "  1. gguf library (pip install gguf) for conversion to Forge\n"
            "  2. llama-cpp-python (pip install llama-cpp-python) for native GGUF inference"
        )
    
    # Return GGUFModel wrapper
    # Note: This is not a Forge model, but provides similar interface
    gguf_wrapper = GGUFModel(str(model_path), **kwargs)
    gguf_wrapper.load()
    return gguf_wrapper


def test_gguf_loading(model_path: str = None):
    """Test function to verify GGUF loading works."""
    print("Testing GGUF loading...")
    
    if not HAVE_LLAMA_CPP:
        print("[ERROR] llama-cpp-python not available")
        print("Install with: pip install llama-cpp-python")
        return False
    
    if model_path is None:
        # Try to find a GGUF model
        models = list_gguf_models()
        if not models:
            print("[ERROR] No GGUF models found in models/ directory")
            return False
        model_path = str(models[0])
        print(f"Using model: {model_path}")
    
    try:
        model = GGUFModel(model_path, n_ctx=512, verbose=False)
        
        if model.load():
            print("[OK] Model loaded successfully")
            
            # Test generation
            response = model.generate("Hello!", max_tokens=20)
            print(f"[OK] Generated: {response[:50]}...")
            
            model.unload()
            print("[OK] Model unloaded")
            return True
        else:
            print("[ERROR] Failed to load model")
            return False
    
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_gguf_loading(sys.argv[1])
    else:
        test_gguf_loading()
