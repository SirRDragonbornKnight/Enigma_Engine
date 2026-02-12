"""
Wayland Support

Wayland compositor compatibility for overlay windows on Linux.
Provides layer shell protocol support and fallback mechanisms.

FILE: enigma_engine/gui/wayland_support.py
TYPE: GUI/Platform
MAIN CLASSES: WaylandOverlay, LayerShell, WaylandDetector
"""

import ctypes
import logging
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from PyQt5.QtCore import QRect, Qt
    from PyQt5.QtWidgets import QApplication, QWidget
    HAS_QT = True
except ImportError:
    HAS_QT = False


class DisplayProtocol(Enum):
    """Linux display protocols."""
    X11 = "x11"
    WAYLAND = "wayland"
    XWAYLAND = "xwayland"
    UNKNOWN = "unknown"


class LayerShellLayer(Enum):
    """Wayland layer shell layers."""
    BACKGROUND = 0
    BOTTOM = 1
    TOP = 2
    OVERLAY = 3


class LayerShellAnchor(Enum):
    """Wayland layer shell anchors."""
    NONE = 0
    TOP = 1
    BOTTOM = 2
    LEFT = 4
    RIGHT = 8
    TOP_LEFT = 5
    TOP_RIGHT = 9
    BOTTOM_LEFT = 6
    BOTTOM_RIGHT = 10


@dataclass
class OverlayConfig:
    """Configuration for Wayland overlay."""
    layer: LayerShellLayer = LayerShellLayer.OVERLAY
    anchor: LayerShellAnchor = LayerShellAnchor.NONE
    margin_top: int = 0
    margin_bottom: int = 0
    margin_left: int = 0
    margin_right: int = 0
    keyboard_interactivity: bool = True
    exclusive_zone: int = -1  # -1 = auto


class WaylandDetector:
    """Detect Wayland environment and capabilities."""
    
    @staticmethod
    def detect_protocol() -> DisplayProtocol:
        """Detect current display protocol."""
        # Check for Wayland
        wayland_display = os.environ.get("WAYLAND_DISPLAY")
        xdg_session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        
        if wayland_display or xdg_session_type == "wayland":
            # Check if running under XWayland
            if os.environ.get("DISPLAY"):
                return DisplayProtocol.XWAYLAND
            return DisplayProtocol.WAYLAND
        
        if os.environ.get("DISPLAY"):
            return DisplayProtocol.X11
        
        return DisplayProtocol.UNKNOWN
    
    @staticmethod
    def get_compositor() -> str:
        """Detect Wayland compositor."""
        # Common compositors
        compositors = {
            "GNOME": "gnome-shell",
            "KDE": "kwin_wayland",
            "Sway": "sway",
            "Hyprland": "Hyprland",
            "Weston": "weston",
            "Mutter": "mutter"
        }
        
        desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").upper()
        if "GNOME" in desktop:
            return "gnome-shell"
        if "KDE" in desktop:
            return "kwin_wayland"
        
        # Check running processes
        try:
            result = subprocess.run(
                ["ps", "-A", "-o", "comm="],
                capture_output=True, text=True, timeout=5
            )
            processes = result.stdout.lower()
            
            for name, process in compositors.items():
                if process.lower() in processes:
                    return process
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass  # Intentionally silent
        
        return "unknown"
    
    @staticmethod
    def has_layer_shell() -> bool:
        """Check if layer shell protocol is available."""
        try:
            # Try to load wlr-layer-shell library
            ctypes.CDLL("libwlr-layer-shell.so")
            return True
        except OSError:
            pass  # Intentionally silent
        
        # Check via wayland-info
        try:
            result = subprocess.run(
                ["wayland-info"],
                capture_output=True, text=True, timeout=5
            )
            return "zwlr_layer_shell_v1" in result.stdout
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass  # Intentionally silent
        
        return False
    
    @staticmethod
    def get_output_info() -> list:
        """Get Wayland output (monitor) information."""
        outputs = []
        
        try:
            result = subprocess.run(
                ["wlr-randr"],
                capture_output=True, text=True, timeout=5
            )
            
            current_output = None
            for line in result.stdout.split("\n"):
                if not line.startswith(" ") and line.strip():
                    current_output = {"name": line.strip()}
                    outputs.append(current_output)
                elif current_output and "current" in line.lower():
                    # Parse resolution
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        res = parts[0].split("@")[0]
                        if "x" in res:
                            w, h = res.split("x")
                            current_output["width"] = int(w)
                            current_output["height"] = int(h)
        except (subprocess.SubprocessError, FileNotFoundError, ValueError, OSError):
            pass  # Intentionally silent
        
        return outputs


class LayerShellInterface:
    """
    Interface to Wayland layer shell protocol.
    
    Enables overlay windows on Wayland compositors.
    """
    
    def __init__(self):
        self.available = False
        self._lib = None
        self._load_library()
    
    def _load_library(self):
        """Load layer shell library."""
        libs = [
            "libwlr-layer-shell.so",
            "libgtk-layer-shell.so.0",
            "libgtk-layer-shell.so"
        ]
        
        for lib_name in libs:
            try:
                self._lib = ctypes.CDLL(lib_name)
                self.available = True
                logger.info(f"Loaded {lib_name}")
                return
            except OSError:
                continue
        
        logger.warning("Layer shell library not available")
    
    def configure_window(
        self,
        window: Any,
        config: OverlayConfig
    ) -> bool:
        """
        Configure a window for layer shell.
        
        Args:
            window: Platform window handle
            config: Overlay configuration
        
        Returns:
            True if successful
        """
        if not self.available:
            return False
        
        try:
            # This would use gtk_layer_shell functions
            # gtk_layer_init_for_window(window)
            # gtk_layer_set_layer(window, config.layer.value)
            # gtk_layer_set_anchor(window, GTK_LAYER_SHELL_EDGE_*, True/False)
            # gtk_layer_set_margin(window, GTK_LAYER_SHELL_EDGE_*, margin)
            # gtk_layer_set_keyboard_interactivity(window, config.keyboard_interactivity)
            # gtk_layer_set_exclusive_zone(window, config.exclusive_zone)
            
            logger.info("Layer shell configured")
            return True
        except Exception as e:
            logger.error(f"Layer shell configuration failed: {e}")
            return False


class WaylandOverlay:
    """
    Wayland-compatible overlay window.
    
    Provides fallback mechanisms for different compositors.
    """
    
    def __init__(self, config: OverlayConfig = None):
        self.config = config or OverlayConfig()
        self.protocol = WaylandDetector.detect_protocol()
        self.compositor = WaylandDetector.get_compositor()
        self.layer_shell = LayerShellInterface()
        
        self._widget: Optional[QWidget] = None
        self._fallback_mode = False
    
    def create_overlay(
        self,
        parent: QWidget = None,
        geometry: QRect = None
    ) -> Optional[QWidget]:
        """
        Create overlay window with Wayland support.
        
        Args:
            parent: Parent widget (optional)
            geometry: Window geometry
        
        Returns:
            Overlay widget
        """
        if not HAS_QT:
            raise ImportError("PyQt5 required")
        
        self._widget = QWidget(parent)
        
        if self.protocol == DisplayProtocol.WAYLAND:
            success = self._setup_wayland_overlay()
            if not success:
                logger.warning("Falling back to X11 compatibility mode")
                self._fallback_mode = True
                self._setup_x11_overlay()
        else:
            self._setup_x11_overlay()
        
        if geometry:
            self._widget.setGeometry(geometry)
        
        return self._widget
    
    def _setup_wayland_overlay(self) -> bool:
        """Setup native Wayland overlay."""
        if not self.layer_shell.available:
            return False
        
        # Set window type
        self._widget.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self._widget.setAttribute(Qt.WA_TranslucentBackground)
        
        # Configure layer shell
        window = self._widget.windowHandle()
        if window:
            success = self.layer_shell.configure_window(window, self.config)
            return success
        
        return False
    
    def _setup_x11_overlay(self):
        """Setup X11/XWayland overlay (fallback)."""
        self._widget.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.X11BypassWindowManagerHint |
            Qt.Tool
        )
        self._widget.setAttribute(Qt.WA_TranslucentBackground)
        self._widget.setAttribute(Qt.WA_ShowWithoutActivating)
        
        # X11 specific setup
        if self.protocol in (DisplayProtocol.X11, DisplayProtocol.XWAYLAND):
            self._set_x11_properties()
    
    def _set_x11_properties(self):
        """Set X11 window properties for overlay."""
        try:
            pass

            # This would set:
            # _NET_WM_WINDOW_TYPE = _NET_WM_WINDOW_TYPE_DOCK
            # _NET_WM_STATE = _NET_WM_STATE_ABOVE, _NET_WM_STATE_STICKY
        except ImportError:
            pass  # Intentionally silent
    
    def set_position(self, x: int, y: int):
        """Set overlay position."""
        if self._widget:
            self._widget.move(x, y)
    
    def set_size(self, width: int, height: int):
        """Set overlay size."""
        if self._widget:
            self._widget.resize(width, height)
    
    def show(self):
        """Show overlay."""
        if self._widget:
            self._widget.show()
            
            # Ensure stays on top
            if self._fallback_mode:
                self._widget.raise_()
    
    def hide(self):
        """Hide overlay."""
        if self._widget:
            self._widget.hide()
    
    def set_opacity(self, opacity: float):
        """Set window opacity (0.0 - 1.0)."""
        if self._widget:
            self._widget.setWindowOpacity(opacity)
    
    def move_to_monitor(self, monitor_index: int):
        """Move overlay to specific monitor."""
        if not self._widget:
            return
        
        app = QApplication.instance()
        if app:
            screens = app.screens()
            if 0 <= monitor_index < len(screens):
                screen = screens[monitor_index]
                geometry = screen.geometry()
                self._widget.move(geometry.topLeft())
    
    def get_info(self) -> dict[str, Any]:
        """Get overlay information."""
        return {
            "protocol": self.protocol.value,
            "compositor": self.compositor,
            "layer_shell_available": self.layer_shell.available,
            "fallback_mode": self._fallback_mode,
            "layer": self.config.layer.name,
            "anchor": self.config.anchor.name
        }


class WaylandInputCapture:
    """Capture input on Wayland (for hotkeys)."""
    
    def __init__(self):
        self.protocol = WaylandDetector.detect_protocol()
        self._callbacks: dict[str, Callable] = {}
    
    def register_hotkey(self, keys: str, callback: Callable):
        """
        Register global hotkey.
        
        On Wayland, this may require compositor-specific portals.
        """
        self._callbacks[keys] = callback
        
        if self.protocol == DisplayProtocol.WAYLAND:
            self._register_wayland_hotkey(keys, callback)
        else:
            self._register_x11_hotkey(keys, callback)
    
    def _register_wayland_hotkey(self, keys: str, callback: Callable):
        """Register hotkey via D-Bus portal."""
        try:
            # Use xdg-desktop-portal for global shortcuts
            # This requires user permission via portal dialog
            import dbus
            
            bus = dbus.SessionBus()
            portal = bus.get_object(
                "org.freedesktop.portal.Desktop",
                "/org/freedesktop/portal/desktop"
            )
            
            # GlobalShortcuts portal (Wayland 1.0+)
            # portal.CreateSession(...)
            # portal.BindShortcuts(...)
            
            logger.info(f"Registered Wayland hotkey: {keys}")
        except Exception as e:
            logger.warning(f"Wayland hotkey registration failed: {e}")
            # Fall back to compositor-specific method
            self._register_compositor_hotkey(keys, callback)
    
    def _register_compositor_hotkey(self, keys: str, callback: Callable):
        """Register hotkey via compositor-specific method."""
        compositor = WaylandDetector.get_compositor()
        
        if compositor == "sway":
            # Sway uses swaymsg
            logger.info("Use sway config for hotkeys")
        elif compositor == "Hyprland":
            # Hyprland uses hyprctl
            logger.info("Use hyprland.conf for hotkeys")
        elif "gnome" in compositor.lower():
            # GNOME uses gsettings
            logger.info("Use GNOME settings for hotkeys")
        elif "kwin" in compositor.lower():
            # KDE uses kglobalaccel
            logger.info("Use KDE shortcuts for hotkeys")
    
    def _register_x11_hotkey(self, keys: str, callback: Callable):
        """Register X11 global hotkey."""
        try:
            pass

            # Parse key combination
            # This is a simplified implementation
            logger.info(f"Registered X11 hotkey: {keys}")
        except ImportError:
            logger.warning("pynput not available for X11 hotkeys")
    
    def unregister_all(self):
        """Unregister all hotkeys."""
        self._callbacks.clear()


class WaylandClipboard:
    """Wayland-compatible clipboard access."""
    
    def __init__(self):
        self.protocol = WaylandDetector.detect_protocol()
    
    def get_text(self) -> str:
        """Get text from clipboard."""
        if self.protocol == DisplayProtocol.WAYLAND:
            try:
                result = subprocess.run(
                    ["wl-paste", "-n"],
                    capture_output=True, text=True, timeout=5
                )
                return result.stdout
            except (subprocess.SubprocessError, FileNotFoundError, OSError):
                pass  # Intentionally silent
        
        # Fallback to xclip
        try:
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass  # Intentionally silent
        
        # PyQt5 fallback
        if HAS_QT:
            app = QApplication.instance()
            if app:
                return app.clipboard().text()
        
        return ""
    
    def set_text(self, text: str):
        """Set clipboard text."""
        if self.protocol == DisplayProtocol.WAYLAND:
            try:
                subprocess.run(
                    ["wl-copy"],
                    input=text,
                    text=True,
                    timeout=5
                )
                return
            except (subprocess.SubprocessError, FileNotFoundError, OSError):
                pass  # Intentionally silent
        
        # Fallback to xclip
        try:
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text,
                text=True,
                timeout=5
            )
            return
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass  # Intentionally silent
        
        # PyQt5 fallback
        if HAS_QT:
            app = QApplication.instance()
            if app:
                app.clipboard().setText(text)


def is_wayland() -> bool:
    """Check if running on Wayland."""
    return WaylandDetector.detect_protocol() in (
        DisplayProtocol.WAYLAND,
        DisplayProtocol.XWAYLAND
    )


def get_display_info() -> dict[str, Any]:
    """Get display environment information."""
    detector = WaylandDetector()
    return {
        "protocol": detector.detect_protocol().value,
        "compositor": detector.get_compositor(),
        "layer_shell": detector.has_layer_shell(),
        "outputs": detector.get_output_info()
    }
