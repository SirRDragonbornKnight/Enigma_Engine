"""
CUDA/ROCm Optimization

GPU-specific optimizations for NVIDIA CUDA and AMD ROCm platforms.
Includes kernel tuning, memory management, and performance profiling.

FILE: enigma_engine/core/gpu_optimizer.py
TYPE: Core/Hardware
MAIN CLASSES: GPUOptimizer, CUDAOptimizer, ROCmOptimizer
"""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUBackend(Enum):
    """GPU backend types."""
    CUDA = auto()   # NVIDIA
    ROCM = auto()   # AMD
    MPS = auto()    # Apple
    CPU = auto()    # Fallback


@dataclass
class GPUConfig:
    """GPU optimization configuration."""
    # Memory
    memory_fraction: float = 0.9
    enable_memory_efficient_attention: bool = True
    use_gradient_checkpointing: bool = False
    
    # Compute
    use_tf32: bool = True  # NVIDIA Ampere+
    use_cudnn_benchmark: bool = True
    use_flash_attention: bool = True
    
    # Precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # float16 or bfloat16
    
    # Kernels
    use_custom_kernels: bool = True
    triton_enabled: bool = True
    
    # Profiling
    profile_enabled: bool = False
    profile_warmup: int = 2
    profile_active: int = 5


@dataclass
class GPUStats:
    """GPU performance statistics."""
    utilization: float
    memory_used: int
    memory_total: int
    temperature: Optional[int]
    power_draw: Optional[float]
    clock_speed: Optional[int]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "utilization_percent": self.utilization,
            "memory_used_gb": self.memory_used / (1024**3),
            "memory_total_gb": self.memory_total / (1024**3),
            "temperature_c": self.temperature,
            "power_w": self.power_draw,
            "clock_mhz": self.clock_speed
        }


if HAS_TORCH:
    
    def detect_backend() -> GPUBackend:
        """Detect available GPU backend."""
        if torch.cuda.is_available():
            # Check for ROCm
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                return GPUBackend.ROCM
            return GPUBackend.CUDA
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return GPUBackend.MPS
        return GPUBackend.CPU
    
    
    class GPUOptimizer:
        """
        Base class for GPU optimizations.
        
        Provides unified interface for CUDA/ROCm optimizations.
        """
        
        def __init__(self, config: GPUConfig = None) -> None:
            self.config = config or GPUConfig()
            self.backend = detect_backend()
            self._original_settings = {}
        
        def apply_optimizations(self) -> None:
            """Apply all configured optimizations."""
            if self.backend == GPUBackend.CUDA:
                self._apply_cuda_optimizations()
            elif self.backend == GPUBackend.ROCM:
                self._apply_rocm_optimizations()
            
            logger.info(f"Applied {self.backend.name} optimizations")
        
        def _apply_cuda_optimizations(self) -> None:
            """Apply NVIDIA CUDA-specific optimizations."""
            # TF32 for Ampere+ GPUs
            if self.config.use_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # cuDNN benchmark mode
            if self.config.use_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
            
            # Memory fraction
            if self.config.memory_fraction < 1.0:
                fraction = self.config.memory_fraction
                torch.cuda.set_per_process_memory_fraction(fraction)
            
            # Environment variables
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            
            # Enable flash attention if available
            if self.config.use_flash_attention:
                self._enable_flash_attention()
        
        def _apply_rocm_optimizations(self) -> None:
            """Apply AMD ROCm-specific optimizations."""
            # ROCm-specific settings
            os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'garbage_collection_threshold:0.6'
            
            # cuDNN equivalent settings (MIOpen)
            torch.backends.cudnn.benchmark = self.config.use_cudnn_benchmark
            
            # Memory management
            if self.config.memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
        
        def _enable_flash_attention(self) -> None:
            """Enable flash attention if available."""
            try:
                # PyTorch 2.0+ scaled_dot_product_attention
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    torch.backends.cuda.enable_flash_sdp(True)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    logger.info("Flash Attention enabled via PyTorch SDPA")
            except Exception as e:
                logger.debug(f"Flash attention not available: {e}")
        
        def optimize_model(self, model: nn.Module) -> nn.Module:
            """
            Apply optimizations to a model.
            
            Args:
                model: Model to optimize
            
            Returns:
                Optimized model
            """
            # Move to GPU
            device = self._get_device()
            model = model.to(device)
            
            # Compile with torch.compile if available (PyTorch 2.0+)
            if self.config.use_custom_kernels and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("Model compiled with torch.compile")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
            
            # Enable gradient checkpointing
            if self.config.use_gradient_checkpointing:
                self._enable_gradient_checkpointing(model)
            
            return model
        
        def _enable_gradient_checkpointing(self, model: nn.Module) -> None:
            """Enable gradient checkpointing for memory efficiency."""
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            elif hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
        
        def _get_device(self) -> torch.device:
            """Get the appropriate device."""
            if self.backend in [GPUBackend.CUDA, GPUBackend.ROCM]:
                return torch.device("cuda")
            elif self.backend == GPUBackend.MPS:
                return torch.device("mps")
            return torch.device("cpu")
        
        @contextmanager
        def autocast_context(self):
            """Context manager for automatic mixed precision."""
            if not self.config.use_amp:
                yield
                return
            
            dtype = torch.float16 if self.config.amp_dtype == "float16" else torch.bfloat16
            
            if self.backend in [GPUBackend.CUDA, GPUBackend.ROCM]:
                with torch.cuda.amp.autocast(dtype=dtype):
                    yield
            else:
                yield
        
        def get_stats(self, device_id: int = 0) -> Optional[GPUStats]:
            """Get current GPU statistics."""
            if self.backend not in [GPUBackend.CUDA, GPUBackend.ROCM]:
                return None
            
            try:
                memory_used = torch.cuda.memory_allocated(device_id)
                memory_total = torch.cuda.get_device_properties(device_id).total_memory
                
                # Try to get additional stats via pynvml/similar
                utilization = 0.0
                temperature = None
                power_draw = None
                clock_speed = None
                
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    clock_speed = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                except ImportError:
                    pass  # Intentionally silent
                
                return GPUStats(
                    utilization=utilization,
                    memory_used=memory_used,
                    memory_total=memory_total,
                    temperature=temperature,
                    power_draw=power_draw,
                    clock_speed=clock_speed
                )
            except Exception as e:
                logger.error(f"Failed to get GPU stats: {e}")
                return None
        
        def clear_cache(self) -> None:
            """Clear GPU memory cache."""
            if self.backend in [GPUBackend.CUDA, GPUBackend.ROCM]:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        def reset_peak_stats(self) -> None:
            """Reset peak memory statistics."""
            if self.backend in [GPUBackend.CUDA, GPUBackend.ROCM]:
                torch.cuda.reset_peak_memory_stats()
    
    
    class CUDAOptimizer(GPUOptimizer):
        """NVIDIA CUDA-specific optimizations."""
        
        def __init__(self, config: GPUConfig = None) -> None:
            super().__init__(config)
            if self.backend != GPUBackend.CUDA:
                logger.warning("CUDAOptimizer created but CUDA not available")
        
        def enable_tensor_cores(self) -> None:
            """Enable Tensor Cores for matrix operations."""
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Tensor Cores enabled (TF32)")
        
        def set_memory_allocator(self, allocator: str = "expandable") -> None:
            """
            Set CUDA memory allocator strategy.
            
            Args:
                allocator: 'native', 'expandable', or 'garbage_collection'
            """
            if allocator == "expandable":
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            elif allocator == "garbage_collection":
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6'
        
        def profile_kernels(
            self,
            model: nn.Module,
            input_data: torch.Tensor,
            num_iterations: int = 10
        ) -> dict[str, float]:
            """
            Profile CUDA kernel execution times.
            
            Args:
                model: Model to profile
                input_data: Sample input
                num_iterations: Number of profiling iterations
            
            Returns:
                Dict of operation names to average times
            """
            model.eval()
            
            # Warmup
            for _ in range(self.config.profile_warmup):
                with torch.no_grad():
                    _ = model(input_data)
            
            torch.cuda.synchronize()
            
            # Profile
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True
            ) as prof:
                for _ in range(num_iterations):
                    with torch.no_grad():
                        _ = model(input_data)
                    torch.cuda.synchronize()
            
            # Aggregate results
            results = {}
            for event in prof.key_averages():
                if event.cuda_time_total > 0:
                    results[event.key] = event.cuda_time_total / num_iterations
            
            return results
        
        def find_optimal_work_size(
            self,
            model: nn.Module,
            input_shape: tuple[int, ...],
            test_sizes: list[int] = None
        ) -> int:
            """
            Find optimal batch size for maximum throughput.
            
            Args:
                model: Model to test
                input_shape: Input shape (without batch dim)
                test_sizes: Batch sizes to test
            
            Returns:
                Optimal batch size
            """
            if test_sizes is None:
                test_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
            
            model.eval()
            best_throughput = 0
            best_size = 1
            
            for batch_size in test_sizes:
                try:
                    torch.cuda.empty_cache()
                    
                    # Create test input
                    test_input = torch.randn(batch_size, *input_shape, device='cuda')
                    
                    # Warmup
                    for _ in range(3):
                        with torch.no_grad():
                            _ = model(test_input)
                    
                    torch.cuda.synchronize()
                    
                    # Time iterations
                    import time
                    start = time.perf_counter()
                    
                    iterations = 10
                    for _ in range(iterations):
                        with torch.no_grad():
                            _ = model(test_input)
                    
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    
                    throughput = (batch_size * iterations) / elapsed
                    
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_size = batch_size
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        break
                    raise
            
            return best_size
    
    
    class ROCmOptimizer(GPUOptimizer):
        """AMD ROCm-specific optimizations."""
        
        def __init__(self, config: GPUConfig = None) -> None:
            super().__init__(config)
            if self.backend != GPUBackend.ROCM:
                logger.warning("ROCmOptimizer created but ROCm not available")
        
        def enable_miopen_optimization(self) -> None:
            """Enable MIOpen optimizations (AMD's cuDNN equivalent)."""
            torch.backends.cudnn.benchmark = True
            os.environ['MIOPEN_FIND_MODE'] = '3'  # Exhaustive search
        
        def set_hip_memory_pool(self) -> None:
            """Configure HIP memory pool for better memory management."""
            os.environ['PYTORCH_HIP_ALLOC_CONF'] = (
                'garbage_collection_threshold:0.6,'
                'max_split_size_mb:512'
            )
        
        def get_rocm_info(self) -> dict[str, Any]:
            """Get ROCm-specific information."""
            info = {
                "hip_version": getattr(torch.version, 'hip', 'unknown'),
                "device_count": torch.cuda.device_count(),
                "devices": []
            }
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_gb": props.total_memory / (1024**3)
                })
            
            return info
    
    
    def auto_optimize(model: nn.Module, config: GPUConfig = None) -> nn.Module:
        """
        Automatically optimize model for available GPU.
        
        Args:
            model: Model to optimize
            config: GPU configuration
        
        Returns:
            Optimized model
        """
        backend = detect_backend()
        
        if backend == GPUBackend.CUDA:
            optimizer = CUDAOptimizer(config)
        elif backend == GPUBackend.ROCM:
            optimizer = ROCmOptimizer(config)
        else:
            optimizer = GPUOptimizer(config)
        
        optimizer.apply_optimizations()
        return optimizer.optimize_model(model)

else:
    def detect_backend():
        return GPUBackend.CPU
    
    class GPUOptimizer:
        pass
    
    class CUDAOptimizer:
        pass
    
    class ROCmOptimizer:
        pass
    
    def auto_optimize(model, config=None):
        return model
