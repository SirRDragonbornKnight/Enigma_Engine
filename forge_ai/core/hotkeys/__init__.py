"""
Platform-specific hotkey backends.
"""

from .windows import WindowsHotkeyBackend
from .linux import LinuxHotkeyBackend
from .macos import MacOSHotkeyBackend

__all__ = ['WindowsHotkeyBackend', 'LinuxHotkeyBackend', 'MacOSHotkeyBackend']
