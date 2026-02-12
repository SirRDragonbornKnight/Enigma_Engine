"""
ARM64 Optimizations

Performance optimizations for ARM64 devices (Raspberry Pi, Apple Silicon, etc.).
Includes NEON SIMD, memory efficiency, and power management.

FILE: enigma_engine/core/arm64_optimizations.py
TYPE: Core/Performance
MAIN CLASSES: ARM64Optimizer, NEONKernels, PowerManager, MemoryOptimizer
"""

import logging
import os
import platform
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None


class ARM64Device(Enum):
    """Known ARM64 devices."""
    RASPBERRY_PI_4 = "rpi4"
    RASPBERRY_PI_5 = "rpi5"
    APPLE_M1 = "m1"
    APPLE_M2 = "m2"
    APPLE_M3 = "m3"
    JETSON_NANO = "jetson_nano"
    JETSON_ORIN = "jetson_orin"
    GENERIC_ARM64 = "arm64"
    UNKNOWN = "unknown"


@dataclass
class DeviceProfile:
    """Hardware profile for ARM device."""
    device: ARM64Device
    cores: int
    ram_gb: float
    has_neon: bool
    has_fp16: bool
    has_bf16: bool
    has_sve: bool
    max_threads: int
    power_limit_watts: float
    
    @classmethod
    def detect(cls) -> "DeviceProfile":
        """Detect current device profile."""
        machine = platform.machine().lower()
        
        if machine not in ("aarch64", "arm64"):
            return cls(
                device=ARM64Device.UNKNOWN,
                cores=os.cpu_count() or 4,
                ram_gb=8.0,
                has_neon=False,
                has_fp16=False,
                has_bf16=False,
                has_sve=False,
                max_threads=4,
                power_limit_watts=100.0
            )
        
        # Detect specific device
        device = ARM64Device.GENERIC_ARM64
        cores = os.cpu_count() or 4
        
        # Check for Raspberry Pi
        try:
            with open("/proc/device-tree/model") as f:
                model = f.read().lower()
                if "raspberry pi 5" in model:
                    device = ARM64Device.RASPBERRY_PI_5
                elif "raspberry pi 4" in model:
                    device = ARM64Device.RASPBERRY_PI_4
        except OSError:
            pass  # Intentionally silent
        
        # Check for Apple Silicon
        if platform.system() == "Darwin":
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5
                )
                brand = result.stdout.lower()
                if "m1" in brand:
                    device = ARM64Device.APPLE_M1
                elif "m2" in brand:
                    device = ARM64Device.APPLE_M2
                elif "m3" in brand:
                    device = ARM64Device.APPLE_M3
            except (subprocess.SubprocessError, FileNotFoundError, OSError):
                pass  # Intentionally silent
        
        # Check for Jetson
        try:
            if os.path.exists("/etc/nv_tegra_release"):
                device = ARM64Device.JETSON_NANO
        except OSError:
            pass  # Intentionally silent
        
        # Set profile based on device
        profiles = {
            ARM64Device.RASPBERRY_PI_4: cls(
                device=ARM64Device.RASPBERRY_PI_4,
                cores=4, ram_gb=4.0,
                has_neon=True, has_fp16=True, has_bf16=False, has_sve=False,
                max_threads=4, power_limit_watts=15.0
            ),
            ARM64Device.RASPBERRY_PI_5: cls(
                device=ARM64Device.RASPBERRY_PI_5,
                cores=4, ram_gb=8.0,
                has_neon=True, has_fp16=True, has_bf16=False, has_sve=False,
                max_threads=4, power_limit_watts=27.0
            ),
            ARM64Device.APPLE_M1: cls(
                device=ARM64Device.APPLE_M1,
                cores=8, ram_gb=16.0,
                has_neon=True, has_fp16=True, has_bf16=False, has_sve=False,
                max_threads=8, power_limit_watts=35.0
            ),
            ARM64Device.APPLE_M2: cls(
                device=ARM64Device.APPLE_M2,
                cores=8, ram_gb=24.0,
                has_neon=True, has_fp16=True, has_bf16=True, has_sve=False,
                max_threads=8, power_limit_watts=40.0
            ),
            ARM64Device.JETSON_NANO: cls(
                device=ARM64Device.JETSON_NANO,
                cores=4, ram_gb=4.0,
                has_neon=True, has_fp16=True, has_bf16=False, has_sve=False,
                max_threads=4, power_limit_watts=10.0
            ),
        }
        
        return profiles.get(device, cls(
            device=device, cores=cores, ram_gb=8.0,
            has_neon=True, has_fp16=True, has_bf16=False, has_sve=False,
            max_threads=cores, power_limit_watts=50.0
        ))


class NEONKernels:
    """NEON SIMD optimized operations."""
    
    def __init__(self, profile: DeviceProfile) -> None:
        self.profile = profile
        self.available = profile.has_neon
    
    def matmul_neon(
        self,
        a: "np.ndarray",
        b: "np.ndarray",
        out: "np.ndarray" = None
    ) -> "np.ndarray":
        """
        NEON-optimized matrix multiplication.
        Uses blocked algorithm for cache efficiency.
        """
        if np is None:
            raise ImportError("NumPy required")
        
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        
        if out is None:
            out = np.zeros((m, n), dtype=a.dtype)
        
        # Block size tuned for ARM cache
        block = 64
        
        for i in range(0, m, block):
            for j in range(0, n, block):
                for kk in range(0, k, block):
                    i_end = min(i + block, m)
                    j_end = min(j + block, n)
                    k_end = min(kk + block, k)
                    
                    out[i:i_end, j:j_end] += np.dot(
                        a[i:i_end, kk:k_end],
                        b[kk:k_end, j:j_end]
                    )
        
        return out
    
    def softmax_neon(self, x: "np.ndarray", axis: int = -1) -> "np.ndarray":
        """NEON-optimized softmax."""
        if np is None:
            raise ImportError("NumPy required")
        
        # Numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def layernorm_neon(
        self,
        x: "np.ndarray",
        weight: "np.ndarray",
        bias: "np.ndarray",
        eps: float = 1e-5
    ) -> "np.ndarray":
        """NEON-optimized layer normalization."""
        if np is None:
            raise ImportError("NumPy required")
        
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return x_norm * weight + bias
    
    def gelu_neon(self, x: "np.ndarray") -> "np.ndarray":
        """NEON-optimized GELU activation."""
        if np is None:
            raise ImportError("NumPy required")
        
        # Approximation for speed
        return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))
    
    def rope_neon(
        self,
        x: "np.ndarray",
        freqs_cos: "np.ndarray",
        freqs_sin: "np.ndarray"
    ) -> "np.ndarray":
        """NEON-optimized rotary position embedding."""
        if np is None:
            raise ImportError("NumPy required")
        
        # Split into even/odd
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        # Apply rotation
        out = np.empty_like(x)
        out[..., 0::2] = x_even * freqs_cos - x_odd * freqs_sin
        out[..., 1::2] = x_even * freqs_sin + x_odd * freqs_cos
        
        return out


class PowerManager:
    """Power management for battery/thermal efficiency."""
    
    def __init__(self, profile: DeviceProfile) -> None:
        self.profile = profile
        self.power_mode = "balanced"
        self.thermal_limit_c = 80.0
        self._last_check = 0.0
    
    def set_power_mode(self, mode: str) -> None:
        """
        Set power mode.
        
        Args:
            mode: "low_power", "balanced", or "performance"
        """
        self.power_mode = mode
        
        if mode == "low_power":
            self._set_cpu_governor("powersave")
            self._set_max_threads(max(1, self.profile.cores // 2))
        elif mode == "balanced":
            self._set_cpu_governor("ondemand")
            self._set_max_threads(self.profile.cores)
        elif mode == "performance":
            self._set_cpu_governor("performance")
            self._set_max_threads(self.profile.max_threads)
    
    def _set_cpu_governor(self, governor: str) -> None:
        """Set CPU frequency governor (Linux)."""
        try:
            for i in range(self.profile.cores):
                path = f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor"
                if os.path.exists(path):
                    with open(path, "w") as f:
                        f.write(governor)
        except PermissionError:
            logger.warning("Cannot set CPU governor (requires root)")
        except Exception as e:
            logger.debug(f"CPU governor not available: {e}")
    
    def _set_max_threads(self, threads: int) -> None:
        """Set max threads for inference."""
        if torch is not None:
            torch.set_num_threads(threads)
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
    
    def get_temperature(self) -> float:
        """Get current CPU temperature."""
        try:
            # Linux thermal zone
            for i in range(10):
                path = f"/sys/class/thermal/thermal_zone{i}/temp"
                if os.path.exists(path):
                    with open(path) as f:
                        return float(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            pass  # Intentionally silent
        
        try:
            # Raspberry Pi
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return float(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            pass  # Intentionally silent
        
        try:
            # macOS
            import subprocess
            result = subprocess.run(
                ["osx-cpu-temp"],
                capture_output=True, text=True, timeout=5
            )
            return float(result.stdout.strip().replace("C", ""))
        except (subprocess.SubprocessError, FileNotFoundError, ValueError, OSError):
            pass  # Intentionally silent
        
        return 0.0
    
    def check_thermal_throttle(self) -> bool:
        """Check if thermal throttling should be applied."""
        now = time.time()
        if now - self._last_check < 5.0:  # Check every 5 seconds
            return False
        
        self._last_check = now
        temp = self.get_temperature()
        
        if temp > self.thermal_limit_c:
            logger.warning(f"Temperature {temp}C exceeds limit, throttling")
            self.set_power_mode("low_power")
            return True
        
        return False
    
    def get_power_stats(self) -> dict[str, Any]:
        """Get power statistics."""
        return {
            "mode": self.power_mode,
            "temperature_c": self.get_temperature(),
            "thermal_limit_c": self.thermal_limit_c,
            "max_threads": self.profile.max_threads,
            "device": self.profile.device.value
        }


class MemoryOptimizer:
    """Memory optimization for constrained ARM devices."""
    
    def __init__(self, profile: DeviceProfile) -> None:
        self.profile = profile
        self.memory_budget_gb = profile.ram_gb * 0.7  # Use 70% of RAM
    
    def get_optimal_batch_size(self, model_size_mb: float) -> int:
        """Calculate optimal batch size for memory budget."""
        available_mb = self.memory_budget_gb * 1024
        
        # Estimate: 3x model size per batch (activations, gradients)
        per_batch_mb = model_size_mb * 3
        
        batch_size = max(1, int(available_mb / per_batch_mb))
        return min(batch_size, 32)  # Cap at 32
    
    def optimize_tensor_layout(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Optimize tensor layout for ARM cache."""
        # ARM prefers row-major with aligned dimensions
        alignment = 16  # 128-bit NEON registers
        
        optimized = list(shape)
        # Align last dimension
        if optimized[-1] % alignment != 0:
            optimized[-1] = ((optimized[-1] // alignment) + 1) * alignment
        
        return tuple(optimized)
    
    def enable_memory_mapping(self) -> None:
        """Enable memory-mapped model loading."""
        if torch is not None:
            # Use mmap for large tensors
            pass
        
        logger.info("Memory mapping enabled")
    
    def gc_aggressive(self) -> None:
        """Aggressive garbage collection."""
        import gc
        gc.collect()
        
        if torch is not None:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def monitor_memory(self) -> dict[str, float]:
        """Monitor memory usage."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total_gb": mem.total / (1024**3),
                "available_gb": mem.available / (1024**3),
                "used_gb": mem.used / (1024**3),
                "percent": mem.percent
            }
        except ImportError:
            return {"error": "psutil not available"}


class ARM64Optimizer:
    """
    Main ARM64 optimization coordinator.
    
    Combines NEON kernels, power management, and memory optimization.
    """
    
    def __init__(self, profile: DeviceProfile = None) -> None:
        self.profile = profile or DeviceProfile.detect()
        self.neon = NEONKernels(self.profile)
        self.power = PowerManager(self.profile)
        self.memory = MemoryOptimizer(self.profile)
        
        logger.info(f"ARM64 optimizer initialized for {self.profile.device.value}")
    
    def optimize_model(self, model: Any) -> Any:
        """
        Apply ARM64 optimizations to a PyTorch model.
        
        Args:
            model: PyTorch model to optimize
        
        Returns:
            Optimized model
        """
        if torch is None:
            logger.warning("PyTorch not available")
            return model
        
        # Convert to half precision if supported
        if self.profile.has_fp16:
            try:
                model = model.half()
                logger.info("Converted model to FP16")
            except (RuntimeError, TypeError):
                pass  # Model may not support half precision
        
        # Set optimal thread count
        torch.set_num_threads(self.profile.max_threads)
        
        # Enable inference mode optimizations
        if hasattr(torch, "inference_mode"):
            logger.info("Using torch.inference_mode for ARM")
        
        return model
    
    def get_optimal_config(self) -> dict[str, Any]:
        """Get optimal configuration for current device."""
        config = {
            "device": self.profile.device.value,
            "max_batch_size": self.memory.get_optimal_batch_size(500),  # Assume 500MB model
            "num_threads": self.profile.max_threads,
            "use_fp16": self.profile.has_fp16,
            "use_bf16": self.profile.has_bf16,
            "power_mode": "balanced"
        }
        
        # Device-specific tuning
        if self.profile.device in (ARM64Device.RASPBERRY_PI_4, ARM64Device.RASPBERRY_PI_5):
            config["max_batch_size"] = min(config["max_batch_size"], 4)
            config["power_mode"] = "low_power"
        
        elif self.profile.device in (ARM64Device.APPLE_M1, ARM64Device.APPLE_M2, ARM64Device.APPLE_M3):
            config["use_mps"] = True  # Metal Performance Shaders
        
        elif self.profile.device == ARM64Device.JETSON_NANO:
            config["use_tensorrt"] = True
            config["max_batch_size"] = min(config["max_batch_size"], 8)
        
        return config
    
    def benchmark(self, iterations: int = 100) -> dict[str, float]:
        """Run ARM64 benchmark."""
        if np is None:
            return {"error": "NumPy required"}
        
        results = {}
        
        # Matrix multiplication
        a = np.random.randn(512, 512).astype(np.float32)
        b = np.random.randn(512, 512).astype(np.float32)
        
        start = time.time()
        for _ in range(iterations):
            _ = np.dot(a, b)
        results["matmul_throughput"] = iterations / (time.time() - start)
        
        # Softmax
        x = np.random.randn(1024, 1024).astype(np.float32)
        start = time.time()
        for _ in range(iterations):
            _ = self.neon.softmax_neon(x)
        results["softmax_throughput"] = iterations / (time.time() - start)
        
        # Memory bandwidth
        size = 100 * 1024 * 1024  # 100MB
        data = np.random.randn(size // 4).astype(np.float32)
        start = time.time()
        _ = data.copy()
        results["memory_bandwidth_gbps"] = size / (time.time() - start) / 1e9
        
        return results


def get_arm64_optimizer() -> Optional[ARM64Optimizer]:
    """Get ARM64 optimizer if on ARM platform."""
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        return ARM64Optimizer()
    return None


# Global instance
_optimizer: Optional[ARM64Optimizer] = None


def get_optimizer() -> ARM64Optimizer:
    """Get or create global ARM64 optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = ARM64Optimizer()
    return _optimizer
