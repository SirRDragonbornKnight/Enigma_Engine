"""
Resource Manager - Control how much CPU/RAM the AI uses.

This lets you run the AI in the background while gaming or doing other tasks.

Usage:
    from enigma_engine.core.resources import apply_resource_mode, set_cpu_threads

    # Use a preset mode
    apply_resource_mode("minimal")   # Use very few resources
    apply_resource_mode("balanced")  # Default - moderate usage
    apply_resource_mode("performance")  # Use more resources for faster AI
    apply_resource_mode("max")       # Use everything available

    # Or set specific limits
    set_cpu_threads(2)               # Only use 2 CPU threads
    set_low_priority(True)           # Run at lower OS priority
"""

import logging
import os
import platform

from ..config import CONFIG

logger = logging.getLogger(__name__)

# Check for torch
HAVE_TORCH = False
try:
    import torch
    HAVE_TORCH = True
except ImportError:
    pass  # Intentionally silent


# =============================================================================
# RESOURCE MODE PRESETS
# =============================================================================

RESOURCE_MODES = {
    "minimal": {
        "cpu_threads": 1,
        "memory_limit_mb": 512,
        "gpu_memory_fraction": 0.2,
        "low_priority": True,
        "batch_size_limit": 1,
        "description": "Minimum resources - for gaming/heavy multitasking"
    },
    "gaming": {
        "cpu_threads": 2,
        "memory_limit_mb": 1024,
        "gpu_memory_fraction": 0.3,
        "low_priority": True,
        "batch_size_limit": 2,
        "description": "Gaming mode - AI runs in background, prioritizes gaming performance"
    },
    "balanced": {
        "cpu_threads": 0,  # Auto (half of available)
        "memory_limit_mb": 0,  # No hard limit
        "gpu_memory_fraction": 0.5,
        "low_priority": False,
        "batch_size_limit": 4,
        "description": "Balanced - good for normal use"
    },
    "performance": {
        "cpu_threads": 0,  # Auto (most available)
        "memory_limit_mb": 0,
        "gpu_memory_fraction": 0.7,
        "low_priority": False,
        "batch_size_limit": 8,
        "description": "More resources - faster responses"
    },
    "max": {
        "cpu_threads": 0,  # All available
        "memory_limit_mb": 0,
        "gpu_memory_fraction": 0.9,
        "low_priority": False,
        "batch_size_limit": 16,
        "description": "Maximum resources - fastest, but may slow other apps"
    }
}


def get_cpu_count() -> int:
    """Get the number of CPU cores available."""
    return os.cpu_count() or 4


def apply_resource_mode(mode: str = "balanced") -> bool:
    """
    Apply a resource mode preset.

    Args:
        mode: One of "minimal", "balanced", "performance", "max"

    Returns:
        True if applied successfully
    """
    if mode not in RESOURCE_MODES:
        logger.warning(f"Unknown mode '{mode}', using 'balanced'")
        mode = "balanced"

    preset = RESOURCE_MODES[mode]

    # Update CONFIG
    CONFIG["resource_mode"] = mode
    CONFIG["cpu_threads"] = preset["cpu_threads"]
    CONFIG["memory_limit_mb"] = preset["memory_limit_mb"]
    CONFIG["gpu_memory_fraction"] = preset["gpu_memory_fraction"]
    CONFIG["low_priority"] = preset["low_priority"]
    CONFIG["batch_size_limit"] = preset.get("batch_size_limit", 4)

    # Apply the settings
    _apply_current_settings()

    return True


def set_cpu_threads(num_threads: int) -> None:
    """
    Set the number of CPU threads for the AI to use.

    Args:
        num_threads: Number of threads (0 = auto, 1-N = specific)
    """
    if num_threads < 0:
        num_threads = 0

    CONFIG["cpu_threads"] = num_threads

    if HAVE_TORCH:
        if num_threads == 0:
            # Auto - use half of available cores for balanced mode
            mode = CONFIG.get("resource_mode", "balanced")
            if mode == "minimal":
                actual_threads = 1
            elif mode == "max":
                actual_threads = get_cpu_count()
            else:
                actual_threads = max(1, get_cpu_count() // 2)
        else:
            actual_threads = min(num_threads, get_cpu_count())

        torch.set_num_threads(actual_threads)
        torch.set_num_interop_threads(max(1, actual_threads // 2))
        logger.info(f"CPU threads set to {actual_threads}")


def set_gpu_memory_fraction(fraction: float) -> None:
    """
    Set how much GPU memory the AI can use.

    Args:
        fraction: 0.0 to 1.0 (e.g., 0.5 = 50% of GPU memory)
    """
    fraction = max(0.1, min(1.0, fraction))
    CONFIG["gpu_memory_fraction"] = fraction

    if HAVE_TORCH and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(fraction)
            logger.info(f"GPU memory limited to {int(fraction * 100)}%")
        except Exception as e:
            logger.warning(f"Could not set GPU limit: {e}")


def set_low_priority(enabled: bool = True) -> None:
    """
    Set the process to run at lower OS priority.
    This helps other apps (like games) run smoother.

    Args:
        enabled: True to lower priority, False for normal
    """
    CONFIG["low_priority"] = enabled

    try:
        system = platform.system()

        if system == "Windows":
            import ctypes

            # BELOW_NORMAL_PRIORITY_CLASS = 0x4000
            # NORMAL_PRIORITY_CLASS = 0x20
            # IDLE_PRIORITY_CLASS = 0x40
            priority = 0x4000 if enabled else 0x20  # Below normal or normal

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            kernel32.SetPriorityClass(handle, priority)
            logger.info(f"Process priority: {'low' if enabled else 'normal'}")

        elif system in ("Linux", "Darwin"):
            import os

            # Nice values: -20 (high) to 19 (low), 0 is default
            # os.nice(increment) adds to current niceness and returns new value
            try:
                current = os.nice(0)  # Get current niceness without changing
                target = 10 if enabled else 0
                if target > current:  # Can only increase niceness without root
                    os.nice(target - current)
                logger.info(f"Process nice value: {target}")
            except OSError as e:
                logger.warning(f"Could not set nice value: {e}")

    except Exception as e:
        logger.warning(f"Could not set priority: {e}")


def _apply_current_settings() -> None:
    """Apply all current resource settings from CONFIG."""
    # CPU threads
    set_cpu_threads(CONFIG.get("cpu_threads", 0))

    # GPU memory
    if HAVE_TORCH and torch.cuda.is_available():
        set_gpu_memory_fraction(CONFIG.get("gpu_memory_fraction", 0.85))

    # Process priority
    if CONFIG.get("low_priority", False):
        set_low_priority(True)


def get_resource_info() -> dict:
    """Get current resource usage information."""
    info = {
        "mode": CONFIG.get("resource_mode", "balanced"),
        "cpu_count": get_cpu_count(),
        "cpu_threads_setting": CONFIG.get("cpu_threads", 0),
        "memory_limit_mb": CONFIG.get("memory_limit_mb", 0),
        "batch_size_limit": CONFIG.get("batch_size_limit", 4),
        "gpu_available": False,
        "gpu_memory_fraction": CONFIG.get("gpu_memory_fraction", 0.5),
        "low_priority": CONFIG.get("low_priority", False),
    }

    if HAVE_TORCH:
        info["torch_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total_mb"] = torch.cuda.get_device_properties(
                0).total_memory // (1024 * 1024)

    return info


def get_batch_size_limit() -> int:
    """
    Get the current batch size limit based on resource mode.
    
    Returns:
        Maximum batch size to use for training/inference
    """
    return CONFIG.get("batch_size_limit", 4)


def print_resource_info() -> None:
    """Print current resource configuration."""
    info = get_resource_info()

    logger.info("RESOURCE CONFIGURATION")
    logger.info(f"Mode:           {info['mode']}")
    logger.info(f"CPU Cores:      {info['cpu_count']}")
    logger.info(f"Torch Threads:  {info.get('torch_threads', 'N/A')}")
    logger.info(f"Batch Size Limit: {info['batch_size_limit']}")
    logger.info(f"Low Priority:   {'Yes' if info['low_priority'] else 'No'}")

    if info['gpu_available']:
        logger.info(f"GPU:            {info['gpu_name']}")
        logger.info(f"GPU Memory:     {info['gpu_memory_total_mb']} MB")
        logger.info(f"GPU Fraction:   {int(info['gpu_memory_fraction'] * 100)}%")
    else:
        logger.info("GPU:            Not available")

    logger.info("Resource configuration complete")


# Initialize on import
def init_resources() -> None:
    """Initialize resource settings from CONFIG."""
    mode = CONFIG.get("resource_mode", "balanced")
    if mode in RESOURCE_MODES:
        _apply_current_settings()


# Auto-initialize when module loads
init_resources()
