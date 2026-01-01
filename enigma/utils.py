"""
Enigma Utilities
================

Common utility functions used throughout Enigma Engine.
"""
import time
from typing import Optional, Iterator, Any


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
        # Try to get length from iterable
        try:
            total = len(iterable)
        except TypeError:
            total = 0

    start_time = time.time()

    def print_progress(iteration: int) -> None:
        """Print progress bar."""
        if total == 0:
            # No progress calculation possible
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

        # Print newline on completion
        if iteration >= total:
            print()

    # Initial call
    print_progress(0)

    # Iterate with progress
    for i, item in enumerate(iterable, 1):
        yield item
        print_progress(i)

    # Ensure we finish at 100%
    if total > 0:
        print_progress(total)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")

    Example:
        >>> format_time(3665)
        '1h 1m 5s'
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
    """
    Format large numbers with commas.

    Args:
        num: Number to format

    Returns:
        Formatted string

    Example:
        >>> format_number(1234567)
        '1,234,567'
    """
    return f"{num:,}"


def format_bytes(bytes: int) -> str:
    """
    Format bytes into human-readable size.

    Args:
        bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.23 MB")

    Example:
        >>> format_bytes(1234567)
        '1.18 MB'
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"


def print_section(title: str, width: int = 60, char: str = "=") -> None:
    """
    Print a section header.

    Args:
        title: Section title
        width: Width of header
        char: Character to use for border

    Example:
        >>> print_section("Training Started")
        ============================================================
        Training Started
        ============================================================
    """
    print()
    print(char * width)
    print(title)
    print(char * width)
    print()
