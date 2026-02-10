"""
Metal Backend for Apple Silicon (M1/M2/M3)

Provides optimized inference on Apple GPUs using MPS (Metal Performance Shaders)
and MLX (Apple's ML framework).

Supports:
- Automatic MPS device selection
- MLX conversion for native Apple Silicon performance
- Memory-efficient inference on unified memory
- Mixed precision (float16/bfloat16) on Apple GPUs
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class MetalConfig:
    """Configuration for Metal/MPS backend."""
    device: str = "mps"  # "mps" for PyTorch MPS, "mlx" for MLX
    dtype: str = "float16"  # float16, bfloat16, float32
    use_mlx: bool = False  # Use MLX instead of PyTorch MPS
    memory_fraction: float = 0.9  # Max fraction of unified memory to use
    enable_mps_fallback: bool = True  # Fall back to CPU for unsupported ops
    compile_model: bool = False  # Use torch.compile (experimental on MPS)


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    import platform
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def is_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available."""
    if not is_apple_silicon():
        return False
    
    try:
        return torch.backends.mps.is_available()
    except AttributeError:
        return False


def get_mps_device() -> torch.device:
    """Get the MPS device."""
    if not is_mps_available():
        raise RuntimeError("MPS is not available on this system")
    return torch.device("mps")


class MetalBackend:
    """
    Metal backend for Apple Silicon GPUs.
    
    Provides optimized inference using MPS or MLX.
    
    Usage:
        backend = MetalBackend()
        model = backend.prepare_model(model)
        output = backend.generate(model, input_ids)
    """
    
    def __init__(self, config: Optional[MetalConfig] = None):
        self.config = config or MetalConfig()
        self._validate_environment()
        self._setup_device()
    
    def _validate_environment(self):
        """Validate that we're on Apple Silicon with proper support."""
        if not is_apple_silicon():
            raise RuntimeError(
                "MetalBackend requires Apple Silicon (M1/M2/M3). "
                "Use CUDA backend for NVIDIA GPUs."
            )
        
        if self.config.use_mlx:
            try:
                import mlx.core as mx
                logger.info(f"MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
            except ImportError:
                raise RuntimeError(
                    "MLX not installed. Install with: pip install mlx"
                )
        elif not is_mps_available():
            raise RuntimeError(
                "MPS not available. Ensure PyTorch >= 1.12 and macOS >= 12.3"
            )
    
    def _setup_device(self):
        """Set up the Metal device."""
        if self.config.use_mlx:
            import mlx.core as mx
            self.device = mx.default_device()
            logger.info(f"Using MLX on {self.device}")
        else:
            self.device = get_mps_device()
            logger.info(f"Using PyTorch MPS backend")
            
            # Configure MPS memory
            if hasattr(torch.mps, 'set_per_process_memory_fraction'):
                torch.mps.set_per_process_memory_fraction(self.config.memory_fraction)
            
            # Enable MPS fallback for unsupported operations
            if self.config.enable_mps_fallback:
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    def get_dtype(self) -> torch.dtype:
        """Get the configured data type."""
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32
        }
        return dtype_map.get(self.config.dtype, torch.float16)
    
    def prepare_model(
        self,
        model: torch.nn.Module,
        compile: bool = False
    ) -> torch.nn.Module:
        """
        Prepare a model for Metal inference.
        
        Args:
            model: PyTorch model
            compile: Whether to use torch.compile (experimental)
        
        Returns:
            Model optimized for Metal
        """
        if self.config.use_mlx:
            return self._convert_to_mlx(model)
        
        # Move to MPS device
        model = model.to(self.device)
        
        # Convert to target dtype
        dtype = self.get_dtype()
        if dtype != torch.float32:
            model = model.to(dtype)
        
        # Optional: compile for potential speedup
        if compile and self.config.compile_model:
            try:
                model = torch.compile(model, backend="aot_eager")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed on MPS: {e}")
        
        model.eval()
        return model
    
    def _convert_to_mlx(self, model: torch.nn.Module):
        """Convert PyTorch model to MLX format."""
        import mlx.core as mx

        # This is a simplified conversion - real implementation would be more complex
        logger.warning(
            "MLX conversion is experimental. "
            "For best results, use native MLX models."
        )
        
        # Save PyTorch weights
        state_dict = model.state_dict()
        
        # Convert each tensor to MLX array
        mlx_weights = {}
        for key, tensor in state_dict.items():
            np_array = tensor.cpu().numpy()
            mlx_weights[key] = mx.array(np_array)
        
        return mlx_weights
    
    @torch.inference_mode()
    def generate(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate tokens using Metal-optimized inference.
        
        Args:
            model: The language model
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            do_sample: Whether to sample (vs greedy)
        
        Returns:
            Generated token IDs [batch_size, seq_len + new_tokens]
        """
        if self.config.use_mlx:
            return self._generate_mlx(
                model, input_ids, max_new_tokens,
                temperature, top_p, top_k, do_sample
            )
        
        # Ensure inputs are on MPS
        input_ids = input_ids.to(self.device)
        generated = input_ids.clone()
        
        # Get dtype
        dtype = self.get_dtype()
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.autocast(device_type="mps", dtype=dtype):
                outputs = model(generated)
            
            # Get logits for last token
            logits = outputs[:, -1, :]
            
            if do_sample:
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Sync MPS to check for errors
            if _ % 10 == 0:
                torch.mps.synchronize()
        
        return generated
    
    def _generate_mlx(
        self,
        model,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool
    ):
        """Generate using MLX backend."""
        import mlx.core as mx

        # Convert input to MLX
        input_array = mx.array(input_ids.cpu().numpy())
        
        # MLX generation would go here
        # This is a placeholder - real implementation depends on model architecture
        raise NotImplementedError(
            "MLX generation requires model-specific implementation. "
            "Use PyTorch MPS backend or native MLX models."
        )
    
    def get_memory_info(self) -> dict[str, Any]:
        """Get Metal memory usage information."""
        if self.config.use_mlx:
            pass

            # MLX doesn't expose detailed memory info yet
            return {"backend": "mlx", "available": True}
        
        # MPS memory info (limited)
        return {
            "backend": "mps",
            "device": str(self.device),
            "available": is_mps_available(),
            "current_allocated": torch.mps.current_allocated_memory() if hasattr(torch.mps, 'current_allocated_memory') else "unknown",
            "driver_allocated": torch.mps.driver_allocated_memory() if hasattr(torch.mps, 'driver_allocated_memory') else "unknown"
        }
    
    def synchronize(self):
        """Synchronize Metal operations."""
        if not self.config.use_mlx:
            torch.mps.synchronize()
    
    def empty_cache(self):
        """Clear Metal memory cache."""
        if not self.config.use_mlx:
            torch.mps.empty_cache()


def create_metal_backend(
    use_mlx: bool = False,
    dtype: str = "float16"
) -> MetalBackend:
    """
    Create a Metal backend for Apple Silicon.
    
    Args:
        use_mlx: Use MLX instead of PyTorch MPS
        dtype: Data type (float16, bfloat16, float32)
    
    Returns:
        MetalBackend instance
    
    Example:
        backend = create_metal_backend()
        model = backend.prepare_model(model)
        output = backend.generate(model, input_ids)
    """
    config = MetalConfig(
        use_mlx=use_mlx,
        dtype=dtype
    )
    return MetalBackend(config)


# Convenience function for automatic device selection
def get_optimal_device() -> str:
    """
    Get the optimal device for the current system.
    
    Returns "mps" on Apple Silicon, "cuda" if NVIDIA GPU available,
    otherwise "cpu".
    """
    if is_mps_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def optimize_for_metal(model: torch.nn.Module) -> torch.nn.Module:
    """
    Quick optimization for Metal inference.
    
    Convenience function that:
    1. Moves model to MPS device
    2. Converts to float16
    3. Sets eval mode
    
    Args:
        model: PyTorch model
    
    Returns:
        Optimized model
    """
    if not is_mps_available():
        logger.warning("MPS not available, returning model unchanged")
        return model
    
    return model.to("mps").half().eval()
