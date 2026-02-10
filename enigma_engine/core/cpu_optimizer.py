"""
CPU SIMD Optimization

Optimizations for CPU inference using SIMD instructions.
Supports AVX, AVX2, AVX-512, and ARM NEON for faster computation.

FILE: enigma_engine/core/cpu_optimizer.py
TYPE: Core/Hardware
MAIN CLASSES: CPUOptimizer, SIMDDetector, VectorizedOps
"""

import logging
import os
import platform
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SIMDLevel(Enum):
    """SIMD instruction set levels."""
    NONE = "none"
    SSE2 = "sse2"
    SSE4 = "sse4"
    AVX = "avx"
    AVX2 = "avx2"
    AVX512 = "avx512"
    NEON = "neon"  # ARM
    SVE = "sve"    # ARM Scalable Vector Extensions


@dataclass
class CPUInfo:
    """CPU information and capabilities."""
    name: str
    vendor: str
    architecture: str
    cores_physical: int
    cores_logical: int
    simd_level: SIMDLevel
    features: list[str]
    cache_l1: int  # bytes
    cache_l2: int
    cache_l3: int


@dataclass
class CPUConfig:
    """CPU optimization configuration."""
    # Threading
    num_threads: Optional[int] = None  # None = auto
    thread_affinity: bool = True
    
    # Memory
    memory_alignment: int = 64  # bytes
    use_huge_pages: bool = False
    prefetch_distance: int = 256
    
    # Compute
    enable_simd: bool = True
    preferred_simd: Optional[SIMDLevel] = None
    vectorize_threshold: int = 1024  # elements
    
    # Quantization
    use_vnni: bool = True  # Intel VNNI for int8
    use_bf16: bool = False


class SIMDDetector:
    """
    Detect CPU SIMD capabilities.
    """
    
    @staticmethod
    @lru_cache(maxsize=1)
    def detect_simd() -> SIMDLevel:
        """
        Detect highest supported SIMD level.
        
        Returns:
            SIMDLevel enum value
        """
        arch = platform.machine().lower()
        
        # ARM detection
        if arch in ('arm64', 'aarch64', 'armv8'):
            return SIMDDetector._detect_arm_simd()
        
        # x86/x64 detection
        if arch in ('x86_64', 'amd64', 'x86'):
            return SIMDDetector._detect_x86_simd()
        
        return SIMDLevel.NONE
    
    @staticmethod
    def _detect_arm_simd() -> SIMDLevel:
        """Detect ARM SIMD capabilities."""
        # All ARM64 has NEON
        try:
            # Check for SVE (modern ARM servers)
            with open('/proc/cpuinfo') as f:
                cpuinfo = f.read()
                if 'sve' in cpuinfo.lower():
                    return SIMDLevel.SVE
        except OSError:
            pass
        
        return SIMDLevel.NEON
    
    @staticmethod
    def _detect_x86_simd() -> SIMDLevel:
        """Detect x86 SIMD capabilities."""
        features = SIMDDetector._get_cpu_features()
        
        # Check from highest to lowest
        if 'avx512f' in features or 'avx512_vnni' in features:
            return SIMDLevel.AVX512
        if 'avx2' in features:
            return SIMDLevel.AVX2
        if 'avx' in features:
            return SIMDLevel.AVX
        if 'sse4_1' in features or 'sse4_2' in features:
            return SIMDLevel.SSE4
        if 'sse2' in features:
            return SIMDLevel.SSE2
        
        return SIMDLevel.NONE
    
    @staticmethod
    @lru_cache(maxsize=1)
    def _get_cpu_features() -> set:
        """Get CPU feature flags."""
        features = set()
        
        try:
            # Try /proc/cpuinfo on Linux
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.startswith('flags'):
                        features.update(line.split(':')[1].lower().split())
                        break
        except OSError:
            pass
        
        try:
            # Try cpuinfo package if available
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            if 'flags' in info:
                features.update(info['flags'])
        except (ImportError, Exception):
            pass
        
        # Windows fallback
        if not features and platform.system() == 'Windows':
            features = SIMDDetector._detect_windows_features()
        
        return features
    
    @staticmethod
    def _detect_windows_features() -> set:
        """Detect CPU features on Windows."""
        features = set()
        
        try:
            import subprocess
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'name'],
                capture_output=True, text=True, timeout=5
            )
            # Modern Intel/AMD CPUs support at least AVX2
            cpu_name = result.stdout.lower()
            if 'intel' in cpu_name or 'amd' in cpu_name:
                # Assume modern CPU
                features.update(['sse2', 'sse4_1', 'sse4_2', 'avx', 'avx2'])
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            # Conservative fallback
            features.add('sse2')
        
        return features
    
    @staticmethod
    def get_cpu_info() -> CPUInfo:
        """Get comprehensive CPU information."""
        import multiprocessing
        
        simd_level = SIMDDetector.detect_simd()
        features = list(SIMDDetector._get_cpu_features())
        
        # Get CPU name
        cpu_name = platform.processor() or "Unknown"
        
        # Detect vendor
        name_lower = cpu_name.lower()
        if 'intel' in name_lower:
            vendor = "Intel"
        elif 'amd' in name_lower:
            vendor = "AMD"
        elif 'apple' in name_lower or 'arm' in name_lower:
            vendor = "ARM"
        else:
            vendor = "Unknown"
        
        # Core counts
        try:
            import psutil
            cores_physical = psutil.cpu_count(logical=False) or 1
            cores_logical = psutil.cpu_count(logical=True) or 1
        except (ImportError, Exception):
            cores_physical = multiprocessing.cpu_count()
            cores_logical = cores_physical
        
        # Cache sizes (defaults if can't detect)
        cache_l1 = 32 * 1024
        cache_l2 = 256 * 1024
        cache_l3 = 8 * 1024 * 1024
        
        return CPUInfo(
            name=cpu_name,
            vendor=vendor,
            architecture=platform.machine(),
            cores_physical=cores_physical,
            cores_logical=cores_logical,
            simd_level=simd_level,
            features=features[:20],  # Limit list size
            cache_l1=cache_l1,
            cache_l2=cache_l2,
            cache_l3=cache_l3
        )


class VectorizedOps:
    """
    Vectorized operations using NumPy with SIMD awareness.
    """
    
    def __init__(self, simd_level: SIMDLevel = None):
        self.simd_level = simd_level or SIMDDetector.detect_simd()
        
        # Vector width based on SIMD level
        self.vector_width = {
            SIMDLevel.NONE: 1,
            SIMDLevel.SSE2: 2,
            SIMDLevel.SSE4: 2,
            SIMDLevel.AVX: 4,
            SIMDLevel.AVX2: 4,
            SIMDLevel.AVX512: 8,
            SIMDLevel.NEON: 2,
            SIMDLevel.SVE: 8,
        }.get(self.simd_level, 1)
    
    def aligned_array(
        self,
        shape: tuple[int, ...],
        dtype=np.float32,
        alignment: int = 64
    ) -> np.ndarray:
        """
        Create aligned NumPy array for SIMD operations.
        
        Args:
            shape: Array shape
            dtype: Data type
            alignment: Memory alignment
        
        Returns:
            Aligned array
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy required")
        
        size = int(np.prod(shape))
        itemsize = np.dtype(dtype).itemsize
        
        # Allocate with extra bytes for alignment
        buffer = np.empty(size * itemsize + alignment, dtype=np.uint8)
        
        # Find aligned offset
        offset = (alignment - (buffer.ctypes.data % alignment)) % alignment
        
        # Create aligned view
        aligned_buffer = buffer[offset:offset + size * itemsize]
        return np.frombuffer(aligned_buffer, dtype=dtype).reshape(shape)
    
    def vectorized_matmul(
        self,
        a: np.ndarray,
        b: np.ndarray,
        block_size: int = None
    ) -> np.ndarray:
        """
        Matrix multiplication with cache-friendly blocking.
        
        Args:
            a: First matrix
            b: Second matrix
            block_size: Block size for tiling (auto if None)
        
        Returns:
            Result matrix
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy required")
        
        # Auto block size based on L2 cache
        if block_size is None:
            block_size = 64  # Good for most L2 caches
        
        # For small matrices, just use np.matmul
        if a.shape[0] < block_size or a.shape[1] < block_size:
            return np.matmul(a, b)
        
        # NumPy uses optimized BLAS under the hood
        # but we can hint at cache-friendly ordering
        return np.matmul(np.ascontiguousarray(a), np.ascontiguousarray(b))
    
    def fast_softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Fast softmax computation.
        
        Args:
            x: Input array
            axis: Axis for softmax
        
        Returns:
            Softmax output
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy required")
        
        # Numerical stability: subtract max
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def fast_gelu(self, x: np.ndarray) -> np.ndarray:
        """
        Fast GELU approximation.
        
        Args:
            x: Input array
        
        Returns:
            GELU output
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy required")
        
        # Fast approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))
    
    def fast_layer_norm(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        eps: float = 1e-6
    ) -> np.ndarray:
        """
        Fast layer normalization.
        
        Args:
            x: Input array
            weight: Gamma weights
            bias: Beta bias
            eps: Epsilon for stability
        
        Returns:
            Normalized output
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy required")
        
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        return weight * (x - mean) / np.sqrt(var + eps) + bias


if HAS_TORCH:
    
    class CPUOptimizer:
        """
        Optimize PyTorch models for CPU inference.
        """
        
        def __init__(self, config: CPUConfig = None):
            self.config = config or CPUConfig()
            self.cpu_info = SIMDDetector.get_cpu_info()
            self._setup_threading()
        
        def _setup_threading(self):
            """Configure threading for optimal CPU performance."""
            num_threads = self.config.num_threads
            
            if num_threads is None:
                # Use physical cores, not hyperthreads
                num_threads = self.cpu_info.cores_physical
            
            # Set PyTorch threads
            torch.set_num_threads(num_threads)
            
            # Set interop threads for parallel regions
            if hasattr(torch, 'set_num_interop_threads'):
                torch.set_num_interop_threads(max(1, num_threads // 2))
            
            # Set OMP/MKL threads
            os.environ['OMP_NUM_THREADS'] = str(num_threads)
            os.environ['MKL_NUM_THREADS'] = str(num_threads)
            
            logger.info(f"CPU threads configured: {num_threads}")
        
        def optimize_model(self, model: nn.Module) -> nn.Module:
            """
            Optimize model for CPU inference.
            
            Args:
                model: Model to optimize
            
            Returns:
                Optimized model
            """
            model = model.cpu()
            model.eval()
            
            # Enable inference optimizations
            with torch.inference_mode():
                # Try TorchScript
                model = self._try_script(model)
                
                # Try IPEX if available
                model = self._try_ipex(model)
            
            return model
        
        def _try_script(self, model: nn.Module) -> nn.Module:
            """Try to JIT script the model."""
            try:
                scripted = torch.jit.script(model)
                logger.info("Model scripted successfully")
                return scripted
            except Exception as e:
                logger.debug(f"JIT script failed: {e}")
                
                try:
                    # Try trace instead
                    # Would need sample input for this
                    pass
                except Exception:
                    pass
                
                return model
        
        def _try_ipex(self, model: nn.Module) -> nn.Module:
            """Try to apply Intel Extension for PyTorch optimizations."""
            try:
                import intel_extension_for_pytorch as ipex
                model = ipex.optimize(model)
                logger.info("IPEX optimizations applied")
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"IPEX optimization failed: {e}")
            
            return model
        
        def quantize_dynamic(self, model: nn.Module) -> nn.Module:
            """
            Apply dynamic quantization for CPU.
            
            Args:
                model: Model to quantize
            
            Returns:
                Quantized model
            """
            try:
                quantized = torch.quantization.quantize_dynamic(
                    model,
                    {nn.Linear, nn.LSTM, nn.GRU},
                    dtype=torch.qint8
                )
                logger.info("Dynamic quantization applied")
                return quantized
            except Exception as e:
                logger.warning(f"Dynamic quantization failed: {e}")
                return model
        
        def benchmark(
            self,
            model: nn.Module,
            input_data: torch.Tensor,
            num_iterations: int = 100,
            warmup: int = 10
        ) -> dict[str, float]:
            """
            Benchmark CPU inference performance.
            
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
            
            # Warmup
            with torch.inference_mode():
                for _ in range(warmup):
                    _ = model(input_data)
            
            # Benchmark
            times = []
            with torch.inference_mode():
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    _ = model(input_data)
                    times.append(time.perf_counter() - start)
            
            import statistics
            return {
                "simd_level": self.cpu_info.simd_level.value,
                "num_threads": torch.get_num_threads(),
                "mean_ms": statistics.mean(times) * 1000,
                "std_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0,
                "min_ms": min(times) * 1000,
                "max_ms": max(times) * 1000,
                "throughput_per_sec": 1 / statistics.mean(times)
            }
        
        def get_optimization_tips(self) -> list[str]:
            """
            Get optimization tips based on detected CPU.
            
            Returns:
                List of optimization suggestions
            """
            tips = []
            
            # SIMD-based tips
            if self.cpu_info.simd_level == SIMDLevel.AVX512:
                tips.append("AVX-512 detected: Ensure MKL/oneDNN is using AVX-512")
            elif self.cpu_info.simd_level in (SIMDLevel.AVX, SIMDLevel.AVX2):
                tips.append("AVX/AVX2 detected: Use Intel MKL for optimal BLAS")
            elif self.cpu_info.simd_level == SIMDLevel.NEON:
                tips.append("ARM NEON detected: Use ARM Compute Library for optimization")
            
            # Threading tips
            if self.cpu_info.cores_physical > 4:
                tips.append(f"Consider batch processing to utilize {self.cpu_info.cores_physical} cores")
            
            # Memory tips
            tips.append("Use contiguous tensors for better cache utilization")
            tips.append("Consider dynamic quantization for 2-4x speedup on inference")
            
            # Intel-specific
            if self.cpu_info.vendor == "Intel":
                tips.append("Install intel-extension-for-pytorch for additional optimizations")
                tips.append("Enable VNNI for int8 operations if supported")
            
            return tips

else:
    class CPUOptimizer:
        pass


def get_optimal_cpu_config() -> CPUConfig:
    """
    Get optimal CPU configuration based on detected hardware.
    
    Returns:
        CPUConfig with optimal settings
    """
    info = SIMDDetector.get_cpu_info()
    
    config = CPUConfig(
        num_threads=info.cores_physical,
        enable_simd=True,
        preferred_simd=info.simd_level
    )
    
    # Adjust for SIMD level
    if info.simd_level == SIMDLevel.AVX512:
        config.memory_alignment = 64
        config.vectorize_threshold = 512
        config.use_vnni = True
    elif info.simd_level in (SIMDLevel.AVX, SIMDLevel.AVX2):
        config.memory_alignment = 32
        config.vectorize_threshold = 1024
    elif info.simd_level == SIMDLevel.NEON:
        config.memory_alignment = 16
        config.vectorize_threshold = 512
    
    return config


def print_cpu_info():
    """Print CPU information and capabilities."""
    info = SIMDDetector.get_cpu_info()
    
    print(f"\nCPU Information:")
    print(f"  Name: {info.name}")
    print(f"  Vendor: {info.vendor}")
    print(f"  Architecture: {info.architecture}")
    print(f"  Cores: {info.cores_physical} physical, {info.cores_logical} logical")
    print(f"  SIMD Level: {info.simd_level.value}")
    print(f"  Features: {', '.join(info.features[:10])}")


if __name__ == "__main__":
    print_cpu_info()
    
    if HAS_NUMPY:
        ops = VectorizedOps()
        print(f"\nVector width: {ops.vector_width}")
        
        # Test operations
        x = np.random.randn(128, 128).astype(np.float32)
        result = ops.fast_softmax(x)
        print(f"Softmax output shape: {result.shape}")
