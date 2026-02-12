"""
Headless Mode - Run without display.

Provides headless operation utilities:
- Display detection
- GUI fallback handling
- Server/CLI mode support
- Virtual display management
- Environment configuration

Part of the Enigma AI Engine platform utilities.
"""

import logging
import os
import platform
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class DisplayEnvironment(Enum):
    """Display environment types."""
    GUI = "gui"              # Full GUI available
    HEADLESS = "headless"    # No display
    VIRTUAL = "virtual"      # Virtual framebuffer (Xvfb)
    REMOTE = "remote"        # Remote display (X11/RDP)
    MINIMAL = "minimal"      # Minimal display (framebuffer)


@dataclass
class DisplayInfo:
    """Information about the display environment."""
    environment: DisplayEnvironment
    has_display: bool
    display_name: Optional[str] = None
    resolution: Optional[tuple] = None
    is_headless: bool = False
    is_virtual: bool = False
    is_remote: bool = False
    platform: str = ""
    details: str = ""


class HeadlessDetector:
    """
    Detect and manage headless mode.
    
    Usage:
        detector = HeadlessDetector()
        
        # Check display
        if detector.is_headless():
            print("Running in headless mode")
        else:
            print("Display available")
        
        # Get detailed info
        info = detector.get_display_info()
        print(f"Environment: {info.environment.value}")
        
        # Force headless mode
        detector.force_headless()
        
        # Setup virtual display
        detector.setup_virtual_display(1920, 1080)
    """
    
    def __init__(self):
        """Initialize headless detector."""
        self._forced_headless = False
        self._virtual_display = None
        self._display_info: Optional[DisplayInfo] = None
    
    def is_headless(self) -> bool:
        """
        Check if running in headless mode.
        
        Returns:
            True if no display is available
        """
        if self._forced_headless:
            return True
        
        return not self._detect_display()
    
    def has_display(self) -> bool:
        """Check if display is available."""
        return not self.is_headless()
    
    def _detect_display(self) -> bool:
        """Detect if a display is available."""
        system = platform.system()
        
        if system == "Windows":
            return self._detect_windows_display()
        elif system == "Darwin":
            return self._detect_macos_display()
        else:
            return self._detect_linux_display()
    
    def _detect_windows_display(self) -> bool:
        """Detect display on Windows."""
        # Check for Windows service environment
        if os.environ.get("SESSIONNAME") == "Services":
            return False
        
        # Check for remote desktop
        if os.environ.get("SESSIONNAME", "").startswith("RDP"):
            return True
        
        # Check if we can access display
        try:
            import ctypes
            return ctypes.windll.user32.GetDesktopWindow() != 0
        except Exception:
            return False
    
    def _detect_macos_display(self) -> bool:
        """Detect display on macOS."""
        # Check if SSH session without display forwarding
        if os.environ.get("SSH_CONNECTION") and not os.environ.get("DISPLAY"):
            return False
        
        # Check for headless environment variable
        if os.environ.get("__CFBundleIdentifier") == "":
            return False
        
        # Try to access display
        try:
            import subprocess
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                timeout=5
            )
            return b"Resolution" in result.stdout
        except Exception:
            return True  # Assume display available on macOS
    
    def _detect_linux_display(self) -> bool:
        """Detect display on Linux."""
        # Check DISPLAY environment variable
        display = os.environ.get("DISPLAY")
        if not display:
            # Check for Wayland
            wayland = os.environ.get("WAYLAND_DISPLAY")
            if not wayland:
                return False
            return True
        
        # Check for virtual display
        if display.startswith(":99") or "Xvfb" in os.environ.get("DISPLAY_TYPE", ""):
            # Virtual display counts as available
            return True
        
        # Check if SSH without X forwarding
        if os.environ.get("SSH_CONNECTION") and not display:
            return False
        
        # Try to connect to display
        try:
            import subprocess
            result = subprocess.run(
                ["xdpyinfo"],
                capture_output=True,
                timeout=5,
                env=os.environ
            )
            return result.returncode == 0
        except Exception:
            # Can't verify, assume available if DISPLAY is set
            return bool(display)
    
    def get_display_info(self) -> DisplayInfo:
        """Get detailed display information."""
        if self._display_info:
            return self._display_info
        
        system = platform.system()
        has_display = self._detect_display()
        
        info = DisplayInfo(
            environment=self._determine_environment(),
            has_display=has_display,
            is_headless=not has_display or self._forced_headless,
            platform=system
        )
        
        # Get display name
        if system in ("Linux", "Darwin"):
            info.display_name = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        
        # Check for virtual display
        if info.display_name and (":99" in info.display_name or self._virtual_display):
            info.is_virtual = True
        
        # Check for remote display
        if os.environ.get("SSH_CONNECTION") or os.environ.get("SESSIONNAME", "").startswith("RDP"):
            info.is_remote = True
        
        # Get resolution if available
        info.resolution = self._get_resolution()
        
        # Build details string
        details = []
        if info.is_headless:
            details.append("headless")
        if info.is_virtual:
            details.append("virtual")
        if info.is_remote:
            details.append("remote")
        info.details = ", ".join(details) if details else "local display"
        
        self._display_info = info
        return info
    
    def _determine_environment(self) -> DisplayEnvironment:
        """Determine the display environment type."""
        if self._forced_headless:
            return DisplayEnvironment.HEADLESS
        
        if self._virtual_display:
            return DisplayEnvironment.VIRTUAL
        
        if not self._detect_display():
            return DisplayEnvironment.HEADLESS
        
        if os.environ.get("SSH_CONNECTION") or os.environ.get("SESSIONNAME", "").startswith("RDP"):
            return DisplayEnvironment.REMOTE
        
        display = os.environ.get("DISPLAY", "")
        if ":99" in display:
            return DisplayEnvironment.VIRTUAL
        
        return DisplayEnvironment.GUI
    
    def _get_resolution(self) -> Optional[tuple]:
        """Get screen resolution if available."""
        system = platform.system()
        
        try:
            if system == "Windows":
                import ctypes
                user32 = ctypes.windll.user32
                return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
            
            elif system == "Darwin":
                import subprocess
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    timeout=5
                )
                # Parse resolution from output
                for line in result.stdout.decode().split("\n"):
                    if "Resolution" in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.isdigit() and i + 2 < len(parts) and parts[i + 2].isdigit():
                                return (int(parts[i]), int(parts[i + 2]))
            
            else:  # Linux
                import subprocess
                result = subprocess.run(
                    ["xdpyinfo"],
                    capture_output=True,
                    timeout=5,
                    env=os.environ
                )
                for line in result.stdout.decode().split("\n"):
                    if "dimensions" in line:
                        parts = line.split()
                        for part in parts:
                            if "x" in part:
                                dims = part.split("x")
                                if len(dims) == 2 and dims[0].isdigit() and dims[1].isdigit():
                                    return (int(dims[0]), int(dims[1]))
        except Exception:
            pass  # Intentionally silent
        
        return None
    
    def force_headless(self, enabled: bool = True):
        """
        Force headless mode.
        
        Args:
            enabled: Whether to force headless mode
        """
        self._forced_headless = enabled
        self._display_info = None  # Reset cached info
        
        if enabled:
            # Set environment to prevent GUI attempts
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            os.environ["MPLBACKEND"] = "Agg"
    
    def setup_virtual_display(
        self,
        width: int = 1920,
        height: int = 1080,
        depth: int = 24
    ) -> bool:
        """
        Setup a virtual display using Xvfb (Linux only).
        
        Args:
            width: Display width
            height: Display height
            depth: Color depth
            
        Returns:
            True if virtual display is set up
        """
        if platform.system() != "Linux":
            return False
        
        try:
            from pyvirtualdisplay import Display
            
            self._virtual_display = Display(
                visible=False,
                size=(width, height),
                color_depth=depth
            )
            self._virtual_display.start()
            self._display_info = None  # Reset cached info
            return True
            
        except ImportError:
            # Try Xvfb directly
            try:
                import subprocess
                display_num = 99
                os.environ["DISPLAY"] = f":{display_num}"
                
                subprocess.Popen(
                    ["Xvfb", f":{display_num}", "-screen", "0", f"{width}x{height}x{depth}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self._display_info = None
                return True
            except Exception:
                return False
        except Exception:
            return False
    
    def stop_virtual_display(self):
        """Stop the virtual display."""
        if self._virtual_display:
            try:
                self._virtual_display.stop()
            except Exception:
                pass  # Intentionally silent
            self._virtual_display = None
            self._display_info = None


class HeadlessGuard:
    """
    Context manager for headless-safe code execution.
    
    Usage:
        with HeadlessGuard() as guard:
            if guard.has_display:
                # Show GUI
                show_window()
            else:
                # CLI fallback
                print_results()
    """
    
    def __init__(
        self,
        fallback: Optional[Callable] = None,
        require_display: bool = False
    ):
        """
        Initialize headless guard.
        
        Args:
            fallback: Function to call if headless
            require_display: Raise error if no display
        """
        self._detector = HeadlessDetector()
        self._fallback = fallback
        self._require_display = require_display
        self.has_display = False
        self.display_info: Optional[DisplayInfo] = None
    
    def __enter__(self):
        """Enter context."""
        self.display_info = self._detector.get_display_info()
        self.has_display = self.display_info.has_display and not self.display_info.is_headless
        
        if self._require_display and not self.has_display:
            raise RuntimeError("Display required but not available")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type is not None and not self.has_display and self._fallback:
            self._fallback()
            return True  # Suppress exception
        return False


def gui_available(
    gui_func: Callable,
    cli_func: Optional[Callable] = None
) -> Callable:
    """
    Decorator for GUI functions with CLI fallback.
    
    Usage:
        @gui_available(cli_fallback=print_results)
        def show_results(data):
            # GUI code here
            ...
    """
    def wrapper(*args, **kwargs):
        detector = HeadlessDetector()
        if detector.has_display():
            return gui_func(*args, **kwargs)
        elif cli_func:
            return cli_func(*args, **kwargs)
        else:
            logger.warning("GUI not available in headless mode")
            return None
    return wrapper


# Global detector
_global_detector: Optional[HeadlessDetector] = None


def get_detector() -> HeadlessDetector:
    """Get the global headless detector."""
    global _global_detector
    if _global_detector is None:
        _global_detector = HeadlessDetector()
    return _global_detector


def is_headless() -> bool:
    """Check if running in headless mode."""
    return get_detector().is_headless()


def has_display() -> bool:
    """Check if display is available."""
    return get_detector().has_display()


def get_display_info() -> DisplayInfo:
    """Get display information."""
    return get_detector().get_display_info()


def force_headless(enabled: bool = True):
    """Force headless mode."""
    get_detector().force_headless(enabled)


def setup_virtual_display(
    width: int = 1920,
    height: int = 1080
) -> bool:
    """Setup virtual display."""
    return get_detector().setup_virtual_display(width, height)
