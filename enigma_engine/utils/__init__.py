"""
Forge Utilities
================

Common utility functions used throughout Enigma AI Engine.
"""
import time
from collections.abc import Iterator
from typing import Any, Optional

# Battery management for mobile/embedded devices
from .battery_manager import (
    POWER_PROFILES,
    BatteryManager,
    PowerProfile,
    PowerState,
    get_battery_manager,
)

# Constants - centralized magic numbers
# Structured error handling
from .errors import (
    ErrorAggregator,
    ErrorCode,
    ErrorContext,
    ErrorSeverity,
    ForgeError,
    Result,
    as_result,
    ensure,
    ensure_not_none,
    from_tool_result,
    handle_errors,
    make_quiet,
    quiet_callback,
    retry,
    safe_call,
)
from .error_handler import ErrorHandler, GracefulFileHandler
from .feedback import FeedbackCollector

# GPU utilities - common operations to avoid duplication
from .json_cache import (
    get_cache_stats,
    invalidate_cache,
    read_json_cached,
    write_json_cached,
)

# Performance monitoring
from .performance_monitor import (
    PerformanceAlert,
    PerformanceMonitor,
    SystemMetrics,
    get_performance_monitor,
)

# JSON storage utilities

# Re-export from submodules for convenience
from .system_messages import (
    MessagePrefix,
    ai_msg,
    debug_msg,
    enigma_msg,
    error_msg,
    forge_msg,
    info_msg,
    memory_msg,
    system_msg,
    thinking_msg,
    user_msg,
    warning_msg,
)
from .text_enhancement import (
    find_closest_match,
    format_did_you_mean,
    levenshtein_distance,
    suggest_command,
)
from .text_formatting import TextFormatter
from .training_validator import TrainingDataValidator


def progress_bar(
    iterable: Iterator[Any],
    total: Optional[int] = None,
    prefix: str = "",
    suffix: str = "",
    length: int = 50,
    fill: str = "█"
) -> Iterator[Any]:
    """
    Create a simple progress bar for iterables.

    Args:
        iterable: Iterator to wrap
        total: Total number of items (required for progress %)
        prefix: Prefix text
        suffix: Suffix text
        length: Character length of bar
        fill: Fill character

    Yields:
        Items from the iterable

    Example:
        >>> for item in progress_bar(range(100), total=100, prefix="Training:"):
        ...     time.sleep(0.01)
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = 0

    start_time = time.time()

    def print_progress(iteration: int) -> None:
        """Print progress bar."""
        if total == 0:
            elapsed = time.time() - start_time
            print(f"\r{prefix} [{iteration} items] | Elapsed: {elapsed:.1f}s {suffix}", end="")
            return

        percent = 100 * (iteration / float(total))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + "-" * (length - filled_length)

        elapsed = time.time() - start_time
        if iteration > 0 and iteration < total:
            eta = elapsed * (total / iteration - 1)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "Done" if iteration >= total else ""

        print(f"\r{prefix} |{bar}| {percent:.1f}% | {eta_str} {suffix}", end="")

        if iteration >= total:
            print()

    print_progress(0)

    for i, item in enumerate(iterable, 1):
        yield item
        print_progress(i)

    if total > 0:
        print_progress(total)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    seconds = seconds % 60

    if minutes < 60:
        return f"{minutes}m {seconds:.0f}s"

    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}m {seconds:.0f}s"


def format_number(num: int) -> str:
    """Format large numbers with commas."""
    return f"{num:,}"


def format_bytes(num_bytes: int) -> str:
    """
    Format bytes into human-readable size.

    Args:
        num_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.23 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def print_section(title: str, width: int = 60, char: str = "=") -> None:
    """Print a section header."""
    print()
    print(char * width)
    print(title)
    print(char * width)
    print()


__all__ = [
    # Progress and formatting
    'progress_bar',
    'format_time',
    'format_number',
    'format_bytes',
    'print_section',
    # Error handling
    'ForgeError',
    'ErrorCode',
    'ErrorSeverity',
    'ErrorContext',
    'ErrorAggregator',
    'Result',
    'safe_call',
    'handle_errors',
    'ensure',
    'ensure_not_none',
    'from_tool_result',
    'as_result',
    'retry',
    # Enigma AI Engine Messages
    'MessagePrefix',
    'system_msg',
    'error_msg',
    'warning_msg',
    'info_msg',
    'debug_msg',
    'ai_msg',
    'forge_msg',
    'enigma_msg',  # Legacy alias
    'user_msg',
    'thinking_msg',
    'memory_msg',
    # Classes
    'TextFormatter',
    'ErrorHandler',
    'GracefulFileHandler',
    'FeedbackCollector',
    'TrainingDataValidator',
    # Text enhancement
    'levenshtein_distance',
    'find_closest_match',
    'suggest_command',
    'format_did_you_mean',
    # JSON caching
    'read_json_cached',
    'write_json_cached',
    'invalidate_cache',
    'get_cache_stats',
    # Battery management
    'BatteryManager',
    'PowerState',
    'PowerProfile',
    'POWER_PROFILES',
    'get_battery_manager',
    # Performance monitoring
    'PerformanceMonitor',
    'SystemMetrics',
    'PerformanceAlert',
    'get_performance_monitor',
]
