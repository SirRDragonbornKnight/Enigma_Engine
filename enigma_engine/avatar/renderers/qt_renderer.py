"""
PyQt5 Avatar Renderer

Renders avatar as a transparent, always-on-top overlay window.
"""

import sys
from typing import Optional

from ..avatar_identity import AvatarAppearance
from .base import BaseRenderer
from .default_sprites import generate_sprite

try:
    from PyQt5.QtCore import QPoint, Qt, QTimer
    from PyQt5.QtSvg import QSvgWidget
    from PyQt5.QtWidgets import QApplication, QWidget
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False


class QtAvatarRenderer(BaseRenderer):
    """
    PyQt5-based renderer for 2D avatar overlay.
    
    Features:
    - Transparent, frameless window
    - Always-on-top
    - Draggable
    - Smooth animations
    """
    
    def __init__(self, controller):
        """
        Initialize Qt renderer.
        
        Args:
            controller: AvatarController instance
        """
        super().__init__(controller)
        
        if not PYQT5_AVAILABLE:
            print("[QtAvatarRenderer] PyQt5 not available, falling back to console output")
            return
        
        self._app = QApplication.instance()
        if self._app is None:
            # Create app if it doesn't exist
            self._app = QApplication(sys.argv)
        
        self._window: Optional[QWidget] = None
        self._svg_widget: Optional[QSvgWidget] = None
        self._drag_position: Optional[QPoint] = None
        self._animation_timer: Optional[QTimer] = None
        self._current_sprite_name = "idle"
    
    def show(self) -> None:
        """Show avatar window."""
        if not PYQT5_AVAILABLE:
            print("[QtAvatarRenderer] PyQt5 not available")
            return
        
        if self._window is None:
            self._create_window()
        
        self._window.show()
        self._window.raise_()
        self._visible = True
        
        # Start animation timer
        if self._animation_timer is None:
            self._animation_timer = QTimer()
            self._animation_timer.timeout.connect(self._update_animation)
            self._animation_timer.start(100)  # 10 FPS
        
        print(f"[QtAvatarRenderer] Window shown")
    
    def hide(self) -> None:
        """Hide avatar window."""
        if self._window:
            self._window.hide()
        
        if self._animation_timer:
            self._animation_timer.stop()
        
        self._visible = False
        print("[QtAvatarRenderer] Window hidden")
    
    def _create_window(self):
        """Create the avatar overlay window."""
        if not PYQT5_AVAILABLE:
            return
        
        # Create frameless, transparent window
        self._window = QWidget()
        self._window.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self._window.setAttribute(Qt.WA_TranslucentBackground)
        
        # Set size based on controller config
        pos = self.controller.position
        self._window.setGeometry(pos.x, pos.y, pos.width, pos.height)
        
        # Create SVG widget for rendering
        self._svg_widget = QSvgWidget(self._window)
        self._svg_widget.setGeometry(0, 0, pos.width, pos.height)
        
        # Load initial sprite
        self._update_sprite_display()
        
        # Enable mouse events for dragging
        self._window.mousePressEvent = self._mouse_press
        self._window.mouseMoveEvent = self._mouse_move
        self._window.mouseReleaseEvent = self._mouse_release
    
    def _mouse_press(self, event):
        """Handle mouse press for dragging."""
        if event.button() == Qt.LeftButton:
            self._drag_position = event.globalPos() - self._window.frameGeometry().topLeft()
            event.accept()
    
    def _mouse_move(self, event):
        """Handle mouse move for dragging."""
        if event.buttons() == Qt.LeftButton and self._drag_position:
            new_pos = event.globalPos() - self._drag_position
            self._window.move(new_pos)
            
            # Update controller position
            self.controller.position.x = new_pos.x()
            self.controller.position.y = new_pos.y()
            
            event.accept()
    
    def _mouse_release(self, event):
        """Handle mouse release."""
        self._drag_position = None
    
    def set_position(self, x: int, y: int) -> None:
        """
        Update avatar position.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        if self._window:
            self._window.move(x, y)
    
    def render_frame(self, animation_data: Optional[dict] = None) -> None:
        """
        Render a single frame.
        
        Args:
            animation_data: Optional animation state
        """
        if not self._visible or not self._window:
            return
        
        # Update sprite based on animation
        if animation_data:
            anim_type = animation_data.get("type", "idle")
            
            if anim_type == "speak":
                # Alternate between speaking frames
                import time
                frame = int(time.time() * 5) % 2  # 5 Hz alternation
                self._current_sprite_name = f"speaking_{frame + 1}"
            elif anim_type == "expression":
                expression = animation_data.get("expression", "neutral")
                self._current_sprite_name = expression
            else:
                sprite_map = {
                    "think": "thinking",
                    "idle": "idle",
                    "move": "idle",
                }
                self._current_sprite_name = sprite_map.get(anim_type, "idle")
        
        self._update_sprite_display()
    
    def _update_animation(self):
        """Periodic animation update."""
        if self._visible and self._current_appearance:
            # Handle idle animation
            if self._current_appearance.idle_animation == "breathe":
                # Could implement breathing effect here
                pass
    
    def _update_sprite_display(self):
        """Update the SVG widget with current sprite."""
        if not self._svg_widget or not self._current_appearance:
            return
        
        # Generate SVG for current sprite
        svg_data = generate_sprite(
            self._current_sprite_name,
            self._current_appearance.primary_color,
            self._current_appearance.secondary_color,
            self._current_appearance.accent_color
        )
        
        # Load SVG into widget
        self._svg_widget.load(svg_data.encode('utf-8'))
    
    def set_appearance(self, appearance: AvatarAppearance) -> None:
        """
        Set complete avatar appearance.
        
        Args:
            appearance: AvatarAppearance instance
        """
        super().set_appearance(appearance)
        
        if self._visible:
            self._update_sprite_display()
    
    def set_expression(self, expression: str) -> None:
        """
        Set avatar facial expression.
        
        Args:
            expression: Expression name
        """
        super().set_expression(expression)
        self._current_sprite_name = expression if expression else "idle"
        
        if self._visible:
            self._update_sprite_display()
    
    def set_scale(self, scale: float) -> None:
        """
        Set avatar scale.
        
        Args:
            scale: Scale factor (1.0 = normal)
        """
        if self._window:
            pos = self.controller.position
            new_width = int(pos.width * scale)
            new_height = int(pos.height * scale)
            self._window.resize(new_width, new_height)
            
            if self._svg_widget:
                self._svg_widget.resize(new_width, new_height)
    
    def set_opacity(self, opacity: float) -> None:
        """
        Set avatar opacity.
        
        Args:
            opacity: 0.0 (transparent) to 1.0 (opaque)
        """
        if self._window:
            self._window.setWindowOpacity(opacity)
    
    def play_animation(self, animation: str, duration: float = 1.0) -> None:
        """
        Play an animation.
        
        Args:
            animation: Animation name
            duration: Duration in seconds
        """
        # Map animation to sprite
        anim_sprite_map = {
            "idle": "idle",
            "speak": "speaking_1",
            "think": "thinking",
            "happy": "happy",
            "sad": "sad",
            "surprised": "surprised",
            "confused": "confused",
            "excited": "excited",
        }
        
        self._current_sprite_name = anim_sprite_map.get(animation, "idle")
        self._update_sprite_display()
