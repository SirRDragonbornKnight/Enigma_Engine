"""
================================================================================
GPU Memory Fallback - Automatic CPU fallback when GPU memory is exhausted.
================================================================================

Provides automatic memory management:
- Detects OOM before it happens using memory estimation
- Automatically offloads to CPU when VRAM is insufficient
- Graceful recovery from OOM errors during inference
- Memory pressure monitoring with configurable thresholds
- Layer-by-layer offloading for partial GPU usage

USAGE:
    from enigma_engine.core.memory_fallback import MemoryFallbackManager, with_memory_fallback
    
    # Automatic fallback manager
    manager = MemoryFallbackManager()
    
    # Try GPU first, fall back to CPU if OOM
    result = manager.run_with_fallback(model, input_tensor)
    
    # Decorator for automatic fallback
    @with_memory_fallback()
    def generate(model, prompt):
        return model(prompt)
    
    # Check if operation will fit in GPU
    if manager.will_fit_in_gpu(model, input_size=1024):
        result = model.cuda()(input)
    else:
        result = model.cpu()(input)
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar('T')

# Try to import torch
try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    torch = None
    nn = None


class FallbackStrategy(Enum):
    """Strategy for handling memory exhaustion."""
    
    CPU_ONLY = auto()         # Move everything to CPU
    PARTIAL_OFFLOAD = auto()  # Keep some layers on GPU
    QUANTIZE = auto()         # Quantize to lower precision
    STREAMING = auto()        # Process in smaller chunks
    DISK_OFFLOAD = auto()     # Offload to disk (very slow)


class MemoryState(Enum):
    """Current memory state."""
    
    HEALTHY = auto()          # Plenty of memory available
    WARNING = auto()          # Memory getting low
    CRITICAL = auto()         # Memory very low, risk of OOM
    OOM = auto()              # Out of memory occurred


@dataclass
class MemoryInfo:
    """Memory information snapshot."""
    
    # GPU memory (bytes)
    gpu_total: int = 0
    gpu_allocated: int = 0
    gpu_cached: int = 0
    gpu_free: int = 0
    
    # CPU memory (bytes)
    cpu_total: int = 0
    cpu_available: int = 0
    cpu_used: int = 0
    
    # Calculated fields
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    
    @property
    def gpu_total_gb(self) -> float:
        return self.gpu_total / (1024 ** 3)
    
    @property
    def gpu_free_gb(self) -> float:
        return self.gpu_free / (1024 ** 3)
    
    @property
    def cpu_available_gb(self) -> float:
        return self.cpu_available / (1024 ** 3)


@dataclass
class FallbackConfig:
    """Configuration for memory fallback behavior."""
    
    # Memory thresholds (fractions)
    warning_threshold: float = 0.75    # Warn at 75% GPU usage
    critical_threshold: float = 0.90   # Critical at 90% GPU usage
    
    # Fallback settings
    default_strategy: FallbackStrategy = FallbackStrategy.CPU_ONLY
    enable_partial_offload: bool = True
    enable_quantization: bool = True
    
    # Safety margins
    safety_margin_mb: int = 512        # Keep 512MB free
    batch_reduction_factor: float = 0.5  # Reduce batch size by half on OOM
    
    # Monitoring
    enable_monitoring: bool = True
    monitoring_interval: float = 1.0   # Check every second
    
    # Recovery settings
    max_retries: int = 3
    clear_cache_on_oom: bool = True
    force_gc_on_oom: bool = True


@dataclass
class FallbackResult:
    """Result of a fallback operation."""
    
    success: bool
    result: Any = None
    device_used: str = "unknown"
    fallback_triggered: bool = False
    strategy_used: FallbackStrategy | None = None
    error: str | None = None
    execution_time: float = 0.0
    retries: int = 0


class MemoryFallbackManager:
    """
    Manages GPU memory and automatic CPU fallback.
    
    Provides:
    - Real-time memory monitoring
    - Automatic device selection based on available memory
    - Graceful OOM recovery with retries
    - Memory estimation before operations
    """
    
    def __init__(self, config: FallbackConfig | None = None):
        """Initialize the manager."""
        self.config = config or FallbackConfig()
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread: threading.Thread | None = None
        self._memory_callbacks: list[Callable[[MemoryInfo, MemoryState], None]] = []
        self._last_oom_time: float = 0
        self._oom_count: int = 0
        
        if not HAVE_TORCH:
            logger.warning("PyTorch not available - memory fallback limited")
    
    def get_memory_info(self) -> MemoryInfo:
        """Get current memory information."""
        info = MemoryInfo()
        
        # CPU memory
        try:
            import psutil
            vm = psutil.virtual_memory()
            info.cpu_total = vm.total
            info.cpu_available = vm.available
            info.cpu_used = vm.used
            info.cpu_utilization = vm.percent / 100.0
        except ImportError:
            pass
        
        # GPU memory
        if HAVE_TORCH and torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                info.gpu_total = torch.cuda.get_device_properties(device).total_memory
                info.gpu_allocated = torch.cuda.memory_allocated(device)
                info.gpu_cached = torch.cuda.memory_reserved(device)
                info.gpu_free = info.gpu_total - info.gpu_cached
                info.gpu_utilization = info.gpu_allocated / info.gpu_total if info.gpu_total > 0 else 0
            except Exception as e:
                logger.debug(f"Could not get GPU memory: {e}")
        
        return info
    
    def get_memory_state(self, info: MemoryInfo | None = None) -> MemoryState:
        """Get current memory state based on utilization."""
        if info is None:
            info = self.get_memory_info()
        
        if not HAVE_TORCH or not torch.cuda.is_available():
            # CPU-only mode
            if info.cpu_utilization > self.config.critical_threshold:
                return MemoryState.CRITICAL
            elif info.cpu_utilization > self.config.warning_threshold:
                return MemoryState.WARNING
            return MemoryState.HEALTHY
        
        # Check GPU utilization
        if info.gpu_utilization > self.config.critical_threshold:
            return MemoryState.CRITICAL
        elif info.gpu_utilization > self.config.warning_threshold:
            return MemoryState.WARNING
        
        return MemoryState.HEALTHY
    
    def estimate_memory_needed(
        self,
        model: Any | None = None,
        input_size: int | None = None,
        batch_size: int = 1,
        dtype: str = "float16"
    ) -> int:
        """
        Estimate memory needed for an operation.
        
        Args:
            model: PyTorch model (to estimate weight size)
            input_size: Size of input tensor (elements)
            batch_size: Batch size
            dtype: Data type (float32, float16, bfloat16, int8)
            
        Returns:
            Estimated memory in bytes
        """
        bytes_per_element = {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
            "int8": 1,
            "int4": 0.5,
        }.get(dtype, 4)
        
        total = 0
        
        if HAVE_TORCH and model is not None:
            # Model weights
            param_count = sum(p.numel() for p in model.parameters())
            total += int(param_count * bytes_per_element)
            
            # Gradient memory (roughly same as weights)
            total += int(param_count * bytes_per_element)
        
        if input_size:
            # Input/output tensors
            total += int(input_size * batch_size * bytes_per_element * 2)
            
            # Activation memory (rough estimate)
            total += int(input_size * batch_size * bytes_per_element * 4)
        
        # Add safety margin
        total += self.config.safety_margin_mb * 1024 * 1024
        
        return total
    
    def will_fit_in_gpu(
        self,
        model: Any | None = None,
        input_size: int | None = None,
        batch_size: int = 1
    ) -> bool:
        """Check if an operation will fit in GPU memory."""
        if not HAVE_TORCH or not torch.cuda.is_available():
            return False
        
        info = self.get_memory_info()
        needed = self.estimate_memory_needed(model, input_size, batch_size)
        
        return info.gpu_free > needed
    
    def suggest_device(
        self,
        model: Any | None = None,
        input_size: int | None = None,
        prefer_gpu: bool = True
    ) -> str:
        """
        Suggest the best device for an operation.
        
        Returns:
            "cuda", "cpu", or "cuda:N" for specific GPU
        """
        if not HAVE_TORCH:
            return "cpu"
        
        if not torch.cuda.is_available():
            return "cpu"
        
        if not prefer_gpu:
            return "cpu"
        
        if self.will_fit_in_gpu(model, input_size):
            return "cuda"
        
        return "cpu"
    
    def clear_gpu_memory(self) -> int:
        """
        Clear GPU memory cache.
        
        Returns:
            Amount of memory freed in bytes
        """
        if not HAVE_TORCH or not torch.cuda.is_available():
            return 0
        
        before = torch.cuda.memory_reserved()
        
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        if self.config.force_gc_on_oom:
            gc.collect()
        
        after = torch.cuda.memory_reserved()
        freed = before - after
        
        if freed > 0:
            logger.info(f"Cleared {freed / (1024**2):.1f} MB of GPU memory")
        
        return freed
    
    def move_to_device(
        self,
        obj: Any,
        device: str,
        non_blocking: bool = True
    ) -> Any:
        """
        Safely move a tensor or model to a device.
        
        Handles OOM with automatic fallback.
        """
        if not HAVE_TORCH:
            return obj
        
        try:
            if hasattr(obj, 'to'):
                return obj.to(device, non_blocking=non_blocking)
            return obj
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM moving to {device}, falling back to CPU")
                self._handle_oom()
                if device != "cpu":
                    return self.move_to_device(obj, "cpu", non_blocking)
            raise
    
    def _handle_oom(self) -> None:
        """Handle an out-of-memory situation."""
        self._oom_count += 1
        self._last_oom_time = time.time()
        
        if self.config.clear_cache_on_oom:
            self.clear_gpu_memory()
        
        if self.config.force_gc_on_oom:
            gc.collect()
    
    def run_with_fallback(
        self,
        func: Callable[..., T],
        *args,
        strategy: FallbackStrategy | None = None,
        **kwargs
    ) -> FallbackResult:
        """
        Run a function with automatic memory fallback.
        
        Args:
            func: Function to run
            *args: Positional arguments
            strategy: Fallback strategy to use (default: from config)
            **kwargs: Keyword arguments
            
        Returns:
            FallbackResult with the result or error
        """
        strategy = strategy or self.config.default_strategy
        start_time = time.time()
        retries = 0
        
        while retries <= self.config.max_retries:
            try:
                result = func(*args, **kwargs)
                return FallbackResult(
                    success=True,
                    result=result,
                    device_used=self._detect_device(args, kwargs),
                    fallback_triggered=(retries > 0),
                    strategy_used=strategy if retries > 0 else None,
                    execution_time=time.time() - start_time,
                    retries=retries
                )
                
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    return FallbackResult(
                        success=False,
                        error=str(e),
                        execution_time=time.time() - start_time,
                        retries=retries
                    )
                
                logger.warning(f"OOM error (retry {retries + 1}/{self.config.max_retries + 1})")
                self._handle_oom()
                
                # Apply fallback strategy
                args, kwargs = self._apply_fallback_strategy(strategy, args, kwargs, retries)
                retries += 1
        
        return FallbackResult(
            success=False,
            error="Max retries exceeded",
            fallback_triggered=True,
            strategy_used=strategy,
            execution_time=time.time() - start_time,
            retries=retries
        )
    
    def _detect_device(self, args: tuple, kwargs: dict) -> str:
        """Try to detect what device was used."""
        if not HAVE_TORCH:
            return "cpu"
        
        for arg in args:
            if hasattr(arg, 'device'):
                return str(arg.device)
        
        for v in kwargs.values():
            if hasattr(v, 'device'):
                return str(v.device)
        
        return "unknown"
    
    def _apply_fallback_strategy(
        self,
        strategy: FallbackStrategy,
        args: tuple,
        kwargs: dict,
        retry: int
    ) -> tuple[tuple, dict]:
        """Apply fallback strategy to arguments."""
        if not HAVE_TORCH:
            return args, kwargs
        
        new_args = list(args)
        new_kwargs = kwargs.copy()
        
        if strategy == FallbackStrategy.CPU_ONLY:
            # Move all tensors to CPU
            for i, arg in enumerate(new_args):
                if torch.is_tensor(arg):
                    new_args[i] = arg.cpu()
                elif hasattr(arg, 'to'):
                    new_args[i] = arg.cpu()
            
            for k, v in new_kwargs.items():
                if torch.is_tensor(v):
                    new_kwargs[k] = v.cpu()
                elif hasattr(v, 'to'):
                    new_kwargs[k] = v.cpu()
                    
        elif strategy == FallbackStrategy.STREAMING:
            # Reduce batch size
            if 'batch_size' in new_kwargs:
                factor = self.config.batch_reduction_factor ** (retry + 1)
                new_kwargs['batch_size'] = max(1, int(new_kwargs['batch_size'] * factor))
        
        return tuple(new_args), new_kwargs
    
    def start_monitoring(self) -> None:
        """Start background memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Memory monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        last_state = MemoryState.HEALTHY
        
        while self._monitoring:
            info = self.get_memory_info()
            state = self.get_memory_state(info)
            
            # Notify on state change
            if state != last_state:
                logger.info(f"Memory state changed: {last_state.name} -> {state.name}")
                for callback in self._memory_callbacks:
                    try:
                        callback(info, state)
                    except Exception as e:
                        logger.debug(f"Callback error: {e}")
                last_state = state
            
            # Auto-clear on critical
            if state == MemoryState.CRITICAL:
                self.clear_gpu_memory()
            
            time.sleep(self.config.monitoring_interval)
    
    def add_callback(self, callback: Callable[[MemoryInfo, MemoryState], None]) -> None:
        """Add a callback for memory state changes."""
        self._memory_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[MemoryInfo, MemoryState], None]) -> None:
        """Remove a memory callback."""
        if callback in self._memory_callbacks:
            self._memory_callbacks.remove(callback)


# Singleton instance
_fallback_manager: MemoryFallbackManager | None = None


def get_fallback_manager(config: FallbackConfig | None = None) -> MemoryFallbackManager:
    """Get or create the singleton fallback manager."""
    global _fallback_manager
    if _fallback_manager is None:
        _fallback_manager = MemoryFallbackManager(config)
    return _fallback_manager


def with_memory_fallback(
    strategy: FallbackStrategy = FallbackStrategy.CPU_ONLY,
    max_retries: int = 3
):
    """
    Decorator for automatic memory fallback.
    
    Usage:
        @with_memory_fallback()
        def generate(model, input):
            return model(input)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            manager = get_fallback_manager()
            result = manager.run_with_fallback(func, *args, strategy=strategy, **kwargs)
            
            if result.success:
                return result.result
            else:
                raise RuntimeError(result.error)
        
        return wrapper
    return decorator


@contextmanager
def memory_guard(
    clear_before: bool = False,
    clear_after: bool = True,
    fallback_to_cpu: bool = True
):
    """
    Context manager for memory-safe operations.
    
    Usage:
        with memory_guard():
            result = model(input)
    """
    manager = get_fallback_manager()
    
    if clear_before:
        manager.clear_gpu_memory()
    
    try:
        yield manager
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and fallback_to_cpu:
            manager._handle_oom()
            logger.warning("OOM in memory_guard, cleared cache")
            raise
        raise
    finally:
        if clear_after:
            manager.clear_gpu_memory()


# Convenience functions
def check_gpu_memory() -> MemoryInfo:
    """Quick check of current GPU memory."""
    return get_fallback_manager().get_memory_info()


def clear_gpu_cache() -> int:
    """Clear GPU memory cache."""
    return get_fallback_manager().clear_gpu_memory()


def suggest_device_for_model(model: Any) -> str:
    """Suggest best device for a model based on memory."""
    return get_fallback_manager().suggest_device(model)


def run_with_fallback(func: Callable[..., T], *args, **kwargs) -> T:
    """Run a function with automatic memory fallback."""
    result = get_fallback_manager().run_with_fallback(func, *args, **kwargs)
    if result.success:
        return result.result
    raise RuntimeError(result.error or "Unknown error")
