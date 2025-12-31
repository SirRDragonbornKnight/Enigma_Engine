"""
Resource Manager - Control how much CPU/RAM the AI uses.

This lets you run the AI in the background while gaming or doing other tasks.

Usage:
    from enigma.core.resources import apply_resource_mode, set_cpu_threads

    # Use a preset mode
    apply_resource_mode("minimal")   # Use very few resources
    apply_resource_mode("balanced")  # Default - moderate usage
    apply_resource_mode("performance")  # Use more resources for faster AI
    apply_resource_mode("max")       # Use everything available

    # Or set specific limits
    set_cpu_threads(2)               # Only use 2 CPU threads
    set_low_priority(True)           # Run at lower OS priority
"""

import os
import sys
import platform
from typing import Optional
from ..config import CONFIG

# Check for torch
HAVE_TORCH = False
try:
    import torch
    HAVE_TORCH = True
except ImportError:
    pass


# =============================================================================
# RESOURCE MODE PRESETS
# =============================================================================

RESOURCE_MODES = {
    "minimal": {
        "cpu_threads": 1,
        "memory_limit_mb": 512,
        "gpu_memory_fraction": 0.2,
        "low_priority": True,
        "description": "Minimum resources - for gaming/heavy multitasking"
    },
    "balanced": {
        "cpu_threads": 0,  # Auto (half of available)
        "memory_limit_mb": 0,  # No hard limit
        "gpu_memory_fraction": 0.5,
        "low_priority": False,
        "description": "Balanced - good for normal use"
    },
    "performance": {
        "cpu_threads": 0,  # Auto (most available)
        "memory_limit_mb": 0,
        "gpu_memory_fraction": 0.7,
        "low_priority": False,
        "description": "More resources - faster responses"
    },
    "max": {
        "cpu_threads": 0,  # All available
        "memory_limit_mb": 0,
        "gpu_memory_fraction": 0.9,
        "low_priority": False,
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
        print(f"Unknown mode '{mode}', using 'balanced'")
        mode = "balanced"

    preset = RESOURCE_MODES[mode]

    # Update CONFIG
    CONFIG["resource_mode"] = mode
    CONFIG["cpu_threads"] = preset["cpu_threads"]
    CONFIG["memory_limit_mb"] = preset["memory_limit_mb"]
    CONFIG["gpu_memory_fraction"] = preset["gpu_memory_fraction"]
    CONFIG["low_priority"] = preset["low_priority"]

    # Apply the settings
    _apply_current_settings()

    return True


def set_cpu_threads(num_threads: int):
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
        print(f"[Resources] CPU threads set to {actual_threads}")


def set_gpu_memory_fraction(fraction: float):
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
            print(f"[Resources] GPU memory limited to {int(fraction * 100)}%")
        except Exception as e:
            print(f"[Resources] Could not set GPU limit: {e}")


def set_low_priority(enabled: bool = True):
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
            print(f"[Resources] Process priority: {'low' if enabled else 'normal'}")

        elif system in ("Linux", "Darwin"):
            import os
            # Nice values: -20 (high) to 19 (low), 0 is default
            # os.nice(increment) adds to current niceness and returns new value
            try:
                current = os.nice(0)  # Get current niceness without changing
                target = 10 if enabled else 0
                if target > current:  # Can only increase niceness without root
                    os.nice(target - current)
                print(f"[Resources] Process nice value: {target}")
            except OSError as e:
                print(f"[Resources] Could not set nice value: {e}")

    except Exception as e:
        print(f"[Resources] Could not set priority: {e}")


def _apply_current_settings():
    """Apply all current resource settings from CONFIG."""
    # CPU threads
    set_cpu_threads(CONFIG.get("cpu_threads", 0))

    # GPU memory
    if HAVE_TORCH and torch.cuda.is_available():
        set_gpu_memory_fraction(CONFIG.get("gpu_memory_fraction", 0.5))

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


def print_resource_info():
    """Print current resource configuration."""
    info = get_resource_info()

    print("\n" + "=" * 50)
    print("RESOURCE CONFIGURATION")
    print("=" * 50)
    print(f"Mode:           {info['mode']}")
    print(f"CPU Cores:      {info['cpu_count']}")
    print(f"Torch Threads:  {info.get('torch_threads', 'N/A')}")
    print(f"Low Priority:   {'Yes' if info['low_priority'] else 'No'}")

    if info['gpu_available']:
        print(f"GPU:            {info['gpu_name']}")
        print(f"GPU Memory:     {info['gpu_memory_total_mb']} MB")
        print(f"GPU Fraction:   {int(info['gpu_memory_fraction'] * 100)}%")
    else:
        print("GPU:            Not available")

    print("=" * 50 + "\n")


# Initialize on import
def init_resources():
    """Initialize resource settings from CONFIG."""
    mode = CONFIG.get("resource_mode", "balanced")
    if mode in RESOURCE_MODES:
        _apply_current_settings()


# Auto-initialize when module loads
init_resources()
