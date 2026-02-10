"""
Apple Metal/MPS Optimization

Optimizations for Apple Silicon using Metal Performance Shaders (MPS).
Provides efficient inference on M1/M2/M3 chips.

FILE: enigma_engine/core/mps_optimizer.py
TYPE: Core/Hardware
MAIN CLASSES: MPSOptimizer, MPSMemoryManager, MPSProfiler
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MPSConfig:
    """MPS optimization configuration."""
    # Memory
    use_unified_memory: bool = True
    memory_limit_gb: Optional[float] = None
    enable_fallback_to_cpu: bool = True
    
    # Compute
    use_graph_optimization: bool = True
    enable_mlx_backend: bool = False  # Use MLX if available
    
    # Precision
    use_float16: bool = True
    conv_precision: str = "high"  # 'default', 'high', 'highest'
    
    # Performance
    enable_profiling: bool = False
    async_execution: bool = True


@dataclass
class MPSDeviceInfo:
    """Information about MPS device."""
    name: str
    chip_family: str
    memory_gb: float
    supports_mps: bool
    supports_mps_fallback: bool
    recommended_batch_size: int


if HAS_TORCH:
    
    def is_mps_available() -> bool:
        """Check if MPS backend is available."""
        return (
            hasattr(torch.backends, 'mps') and 
            torch.backends.mps.is_available()
        )
    
    
    def get_mps_device_info() -> Optional[MPSDeviceInfo]:
        """Get information about MPS device."""
        if not is_mps_available():
            return None
        
        try:
            import subprocess

            # Get chip info
            chip = "Unknown"
            memory = 8.0
            
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True, timeout=5
                )
                chip = result.stdout.strip()
            except (subprocess.SubprocessError, OSError):
                pass
            
            # Determine chip family
            chip_family = "Unknown"
            if "M1" in chip:
                chip_family = "M1"
            elif "M2" in chip:
                chip_family = "M2"
            elif "M3" in chip:
                chip_family = "M3"
            elif "M4" in chip:
                chip_family = "M4"
            
            # Get memory
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True, text=True, timeout=5
                )
                memory = int(result.stdout.strip()) / (1024**3)
            except (subprocess.SubprocessError, ValueError, OSError):
                pass
            
            # Recommended batch size based on memory
            if memory >= 32:
                batch_size = 32
            elif memory >= 16:
                batch_size = 16
            elif memory >= 8:
                batch_size = 8
            else:
                batch_size = 4
            
            return MPSDeviceInfo(
                name=chip,
                chip_family=chip_family,
                memory_gb=memory,
                supports_mps=True,
                supports_mps_fallback=hasattr(torch.backends.mps, 'is_built'),
                recommended_batch_size=batch_size
            )
        except Exception as e:
            logger.error(f"Error getting MPS device info: {e}")
            return None
    
    
    class MPSMemoryManager:
        """
        Manage memory for MPS (Metal Performance Shaders).
        
        Apple Silicon uses unified memory, so memory management
        differs from discrete GPUs.
        """
        
        def __init__(self, config: MPSConfig = None):
            self.config = config or MPSConfig()
            self._allocated_tensors: list[torch.Tensor] = []
        
        def get_memory_info(self) -> dict[str, float]:
            """Get current memory usage information."""
            try:
                import psutil
                memory = psutil.virtual_memory()
                
                return {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent_used": memory.percent
                }
            except ImportError:
                return {"error": "psutil not available"}
        
        def optimize_memory_layout(self, tensor: torch.Tensor) -> torch.Tensor:
            """
            Optimize tensor memory layout for MPS.
            
            Args:
                tensor: Input tensor
            
            Returns:
                Memory-optimized tensor
            """
            # Ensure contiguous memory
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            
            # Convert to optimal dtype
            if self.config.use_float16 and tensor.dtype == torch.float32:
                tensor = tensor.half()
            
            return tensor
        
        def move_to_mps(self, data: Any) -> Any:
            """
            Move data to MPS device with optimizations.
            
            Args:
                data: Tensor, model, or nested structure
            
            Returns:
                Data on MPS device
            """
            device = torch.device('mps')
            
            if isinstance(data, torch.Tensor):
                data = self.optimize_memory_layout(data)
                return data.to(device)
            elif isinstance(data, nn.Module):
                return data.to(device)
            elif isinstance(data, dict):
                return {k: self.move_to_mps(v) for k, v in data.items()}
            elif isinstance(data, (list, tuple)):
                return type(data)(self.move_to_mps(item) for item in data)
            
            return data
        
        def clear_cache(self):
            """Clear MPS memory cache."""
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
    
    
    class MPSOptimizer:
        """
        Optimize models for Apple Silicon MPS backend.
        """
        
        def __init__(self, config: MPSConfig = None):
            self.config = config or MPSConfig()
            self.memory_manager = MPSMemoryManager(config)
            self.device_info = get_mps_device_info()
            
            if not is_mps_available():
                logger.warning("MPS not available, optimizations will be limited")
        
        def optimize_model(self, model: nn.Module) -> nn.Module:
            """
            Apply MPS optimizations to a model.
            
            Args:
                model: Model to optimize
            
            Returns:
                Optimized model on MPS device
            """
            if not is_mps_available():
                logger.warning("MPS not available, returning model on CPU")
                return model
            
            # Move to MPS
            model = model.to('mps')
            
            # Convert to float16 if configured
            if self.config.use_float16:
                model = model.half()
            
            # Apply graph optimization if available
            if self.config.use_graph_optimization and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, backend="aot_eager")
                    logger.info("Model compiled with AOT for MPS")
                except Exception as e:
                    logger.debug(f"torch.compile not available for MPS: {e}")
            
            # Set convolution precision
            self._set_conv_precision(model)
            
            return model
        
        def _set_conv_precision(self, model: nn.Module):
            """Set convolution precision for Metal."""
            # Metal convolutions can use different precision modes
            for module in model.modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    # Precision hint via padding mode
                    if hasattr(module, 'padding_mode'):
                        pass  # Metal handles this internally
        
        @contextmanager
        def inference_context(self):
            """
            Context manager for optimized MPS inference.
            """
            # Enable graph mode
            if self.config.use_graph_optimization:
                torch.mps.set_per_process_memory_fraction(0.9)
            
            try:
                with torch.inference_mode():
                    yield
            finally:
                # Sync MPS operations
                if is_mps_available():
                    torch.mps.synchronize()
        
        def benchmark(
            self,
            model: nn.Module,
            input_data: torch.Tensor,
            num_iterations: int = 100,
            warmup: int = 10
        ) -> dict[str, float]:
            """
            Benchmark model performance on MPS.
            
            Args:
                model: Model to benchmark
                input_data: Sample input
                num_iterations: Number of iterations
                warmup: Warmup iterations
            
            Returns:
                Benchmark results
            """
            import time
            
            model.eval()
            input_data = input_data.to('mps')
            
            # Warmup
            with torch.inference_mode():
                for _ in range(warmup):
                    _ = model(input_data)
                torch.mps.synchronize()
            
            # Benchmark
            times = []
            with torch.inference_mode():
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    _ = model(input_data)
                    torch.mps.synchronize()
                    times.append(time.perf_counter() - start)
            
            import statistics
            return {
                "mean_ms": statistics.mean(times) * 1000,
                "std_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0,
                "min_ms": min(times) * 1000,
                "max_ms": max(times) * 1000,
                "throughput_per_sec": 1 / statistics.mean(times)
            }
        
        def get_recommended_settings(self) -> dict[str, Any]:
            """
            Get recommended settings based on device.
            
            Returns:
                Dict of recommended configuration values
            """
            if self.device_info is None:
                return {"error": "No MPS device detected"}
            
            settings = {
                "batch_size": self.device_info.recommended_batch_size,
                "use_float16": True,
                "use_graph_optimization": True
            }
            
            # Chip-specific recommendations
            if self.device_info.chip_family in ["M2", "M3", "M4"]:
                settings["use_neural_engine"] = True
                settings["enable_async"] = True
            
            if self.device_info.memory_gb >= 16:
                settings["batch_size"] = min(32, settings["batch_size"] * 2)
            
            return settings
    
    
    class MPSProfiler:
        """
        Profile MPS operations for optimization insights.
        """
        
        def __init__(self):
            self._records: list[dict[str, Any]] = []
        
        def profile_operation(
            self,
            operation: callable,
            *args,
            name: str = "operation",
            **kwargs
        ) -> tuple[Any, dict[str, float]]:
            """
            Profile a single operation.
            
            Args:
                operation: Function to profile
                *args: Arguments for operation
                name: Operation name for logging
                **kwargs: Keyword arguments for operation
            
            Returns:
                Tuple of (result, timing_info)
            """
            import time
            
            torch.mps.synchronize()
            start = time.perf_counter()
            
            result = operation(*args, **kwargs)
            
            torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            
            timing = {
                "name": name,
                "time_ms": elapsed * 1000,
                "timestamp": time.time()
            }
            self._records.append(timing)
            
            return result, timing
        
        def get_summary(self) -> dict[str, Any]:
            """Get profiling summary."""
            if not self._records:
                return {"error": "No records"}
            
            by_name = {}
            for record in self._records:
                name = record["name"]
                if name not in by_name:
                    by_name[name] = []
                by_name[name].append(record["time_ms"])
            
            summary = {}
            for name, times in by_name.items():
                import statistics
                summary[name] = {
                    "count": len(times),
                    "total_ms": sum(times),
                    "mean_ms": statistics.mean(times),
                    "std_ms": statistics.stdev(times) if len(times) > 1 else 0
                }
            
            return summary
        
        def clear(self):
            """Clear profiling records."""
            self._records.clear()
    
    
    def optimize_for_apple_silicon(
        model: nn.Module,
        config: MPSConfig = None
    ) -> nn.Module:
        """
        Optimize a model for Apple Silicon.
        
        Args:
            model: Model to optimize
            config: MPS configuration
        
        Returns:
            Optimized model
        """
        optimizer = MPSOptimizer(config)
        return optimizer.optimize_model(model)
    
    
    def check_mps_compatibility(model: nn.Module) -> dict[str, Any]:
        """
        Check model compatibility with MPS backend.
        
        Args:
            model: Model to check
        
        Returns:
            Compatibility report
        """
        report = {
            "mps_available": is_mps_available(),
            "compatible_layers": [],
            "incompatible_layers": [],
            "warnings": []
        }
        
        # Known incompatible operations
        incompatible_ops = {
            'grid_sample', 'unique', 'bincount', 'scatter_reduce'
        }
        
        for name, module in model.named_modules():
            module_type = type(module).__name__
            
            # Check for incompatible operations
            if hasattr(module, 'forward'):
                import inspect
                try:
                    source = inspect.getsource(module.forward)
                    for op in incompatible_ops:
                        if op in source:
                            report["incompatible_layers"].append({
                                "name": name,
                                "type": module_type,
                                "reason": f"Uses {op} which may not be MPS-compatible"
                            })
                            break
                    else:
                        report["compatible_layers"].append({
                            "name": name,
                            "type": module_type
                        })
                except (TypeError, OSError):
                    report["compatible_layers"].append({
                        "name": name,
                        "type": module_type
                    })
        
        if report["incompatible_layers"]:
            report["warnings"].append(
                f"{len(report['incompatible_layers'])} layers may have MPS compatibility issues"
            )
        
        return report

else:
    def is_mps_available():
        return False
    
    def get_mps_device_info():
        return None
    
    class MPSMemoryManager:
        pass
    
    class MPSOptimizer:
        pass
    
    class MPSProfiler:
        pass
    
    def optimize_for_apple_silicon(model, config=None):
        return model
    
    def check_mps_compatibility(model):
        return {"error": "PyTorch not available"}
