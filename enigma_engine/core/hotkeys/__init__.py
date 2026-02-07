"""
Platform-specific hotkey backends.
"""

from .linux import LinuxHotkeyBackend
from .macos import MacOSHotkeyBackend
from .windows import WindowsHotkeyBackend

__all__ = ['WindowsHotkeyBackend', 'LinuxHotkeyBackend', 'MacOSHotkeyBackend']
