"""
Game Overlay System

In-game overlay UI for displaying AI assistant while gaming.
Supports hotkeys, transparency, and attachment to game windows.

FILE: forge_ai/game/overlay.py
TYPE: Game
MAIN CLASSES: GameOverlay, OverlayWidget, OverlayManager
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from PyQt5.QtCore import QPoint, QRect, Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QColor, QFont, QKeySequence, QPalette
    from PyQt5.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QScrollArea,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    HAS_QT = True
except ImportError:
    HAS_QT = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OverlayPosition(Enum):
    """Overlay position on screen."""
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    CENTER = "center"
    CUSTOM = "custom"


class OverlayMode(Enum):
    """Overlay display modes."""
    COMPACT = "compact"      # Minimal, just shows last message
    EXPANDED = "expanded"    # Full chat interface
    MINIMIZED = "minimized"  # Just icon/indicator
    HIDDEN = "hidden"        # Completely hidden


@dataclass
class OverlayConfig:
    """Overlay configuration."""
    # Position
    position: OverlayPosition = OverlayPosition.TOP_RIGHT
    custom_x: int = 100
    custom_y: int = 100
    width: int = 400
    height: int = 300
    
    # Appearance
    opacity: float = 0.85
    background_color: str = "#1a1a2e"
    text_color: str = "#eaeaea"
    accent_color: str = "#0f3460"
    font_size: int = 12
    font_family: str = "Segoe UI"
    border_radius: int = 10
    
    # Behavior
    always_on_top: bool = True
    click_through: bool = False
    auto_hide_delay: int = 5000  # ms, 0 = never
    show_on_response: bool = True
    
    # Hotkeys
    toggle_hotkey: str = "Ctrl+Shift+O"
    mode_cycle_hotkey: str = "Ctrl+Shift+M"
    send_hotkey: str = "Return"
    
    # Game detection
    attach_to_game: bool = True
    target_games: List[str] = field(default_factory=list)


if HAS_QT:
    
    class OverlayWidget(QWidget):
        """
        Transparent overlay widget for in-game display.
        """
        
        message_sent = pyqtSignal(str)
        mode_changed = pyqtSignal(str)
        
        def __init__(self, config: OverlayConfig = None):
            super().__init__()
            self.config = config or OverlayConfig()
            self._mode = OverlayMode.COMPACT
            self._messages: List[Dict[str, str]] = []
            self._dragging = False
            self._drag_position = QPoint()
            
            self._setup_window()
            self._setup_ui()
            self._apply_style()
            
            # Auto-hide timer
            self._hide_timer = QTimer()
            self._hide_timer.setSingleShot(True)
            self._hide_timer.timeout.connect(self._auto_hide)
        
        def _setup_window(self):
            """Configure window properties."""
            # Frameless, transparent, always on top
            flags = Qt.FramelessWindowHint | Qt.Tool
            
            if self.config.always_on_top:
                flags |= Qt.WindowStaysOnTopHint
            
            self.setWindowFlags(flags)
            self.setAttribute(Qt.WA_TranslucentBackground)
            
            if self.config.click_through:
                self.setAttribute(Qt.WA_TransparentForMouseEvents)
            
            # Set size
            self.resize(self.config.width, self.config.height)
            
            # Set position
            self._set_position()
        
        def _set_position(self):
            """Set overlay position on screen."""
            screen = QApplication.primaryScreen().geometry()
            
            positions = {
                OverlayPosition.TOP_LEFT: (10, 10),
                OverlayPosition.TOP_RIGHT: (screen.width() - self.config.width - 10, 10),
                OverlayPosition.BOTTOM_LEFT: (10, screen.height() - self.config.height - 50),
                OverlayPosition.BOTTOM_RIGHT: (
                    screen.width() - self.config.width - 10,
                    screen.height() - self.config.height - 50
                ),
                OverlayPosition.CENTER: (
                    (screen.width() - self.config.width) // 2,
                    (screen.height() - self.config.height) // 2
                ),
                OverlayPosition.CUSTOM: (self.config.custom_x, self.config.custom_y)
            }
            
            x, y = positions.get(self.config.position, (100, 100))
            self.move(x, y)
        
        def _setup_ui(self):
            """Create UI components."""
            self._main_layout = QVBoxLayout(self)
            self._main_layout.setContentsMargins(0, 0, 0, 0)
            self._main_layout.setSpacing(0)
            
            # Container frame with rounded corners
            self._container = QFrame()
            self._container.setObjectName("overlayContainer")
            self._main_layout.addWidget(self._container)
            
            container_layout = QVBoxLayout(self._container)
            container_layout.setContentsMargins(10, 10, 10, 10)
            container_layout.setSpacing(8)
            
            # Header with title and controls
            header = QHBoxLayout()
            
            self._title = QLabel("AI Assistant")
            self._title.setObjectName("overlayTitle")
            header.addWidget(self._title)
            
            header.addStretch()
            
            # Mode toggle button
            self._mode_btn = QPushButton("_")
            self._mode_btn.setObjectName("overlayBtn")
            self._mode_btn.setFixedSize(24, 24)
            self._mode_btn.clicked.connect(self._cycle_mode)
            header.addWidget(self._mode_btn)
            
            # Close button
            self._close_btn = QPushButton("x")
            self._close_btn.setObjectName("overlayBtn")
            self._close_btn.setFixedSize(24, 24)
            self._close_btn.clicked.connect(self.hide)
            header.addWidget(self._close_btn)
            
            container_layout.addLayout(header)
            
            # Messages area (for expanded mode)
            self._messages_area = QScrollArea()
            self._messages_area.setWidgetResizable(True)
            self._messages_area.setObjectName("messagesArea")
            
            self._messages_widget = QWidget()
            self._messages_layout = QVBoxLayout(self._messages_widget)
            self._messages_layout.setAlignment(Qt.AlignTop)
            self._messages_area.setWidget(self._messages_widget)
            
            container_layout.addWidget(self._messages_area)
            
            # Compact message display
            self._compact_label = QLabel("")
            self._compact_label.setObjectName("compactLabel")
            self._compact_label.setWordWrap(True)
            self._compact_label.setVisible(True)
            container_layout.addWidget(self._compact_label)
            
            # Input area
            input_layout = QHBoxLayout()
            
            self._input = QLineEdit()
            self._input.setObjectName("overlayInput")
            self._input.setPlaceholderText("Type a message...")
            self._input.returnPressed.connect(self._send_message)
            input_layout.addWidget(self._input)
            
            self._send_btn = QPushButton("Send")
            self._send_btn.setObjectName("sendBtn")
            self._send_btn.clicked.connect(self._send_message)
            input_layout.addWidget(self._send_btn)
            
            container_layout.addLayout(input_layout)
            
            # Apply mode
            self._apply_mode()
        
        def _apply_style(self):
            """Apply stylesheet based on config."""
            style = f"""
                #overlayContainer {{
                    background-color: {self.config.background_color};
                    border-radius: {self.config.border_radius}px;
                    border: 1px solid {self.config.accent_color};
                }}
                #overlayTitle {{
                    color: {self.config.text_color};
                    font-size: {self.config.font_size + 2}px;
                    font-weight: bold;
                    font-family: "{self.config.font_family}";
                }}
                #overlayBtn {{
                    background-color: transparent;
                    color: {self.config.text_color};
                    border: none;
                    font-size: 14px;
                    font-weight: bold;
                }}
                #overlayBtn:hover {{
                    background-color: {self.config.accent_color};
                    border-radius: 4px;
                }}
                #messagesArea {{
                    background-color: transparent;
                    border: none;
                }}
                #compactLabel {{
                    color: {self.config.text_color};
                    font-size: {self.config.font_size}px;
                    font-family: "{self.config.font_family}";
                    padding: 5px;
                }}
                #overlayInput {{
                    background-color: {self.config.accent_color};
                    color: {self.config.text_color};
                    border: none;
                    border-radius: 5px;
                    padding: 8px;
                    font-size: {self.config.font_size}px;
                    font-family: "{self.config.font_family}";
                }}
                #sendBtn {{
                    background-color: {self.config.accent_color};
                    color: {self.config.text_color};
                    border: none;
                    border-radius: 5px;
                    padding: 8px 12px;
                    font-size: {self.config.font_size}px;
                }}
                #sendBtn:hover {{
                    background-color: #1a5080;
                }}
                .userMessage {{
                    background-color: {self.config.accent_color};
                    color: {self.config.text_color};
                    border-radius: 8px;
                    padding: 5px 10px;
                    margin: 2px 0;
                }}
                .aiMessage {{
                    background-color: #2a2a4e;
                    color: {self.config.text_color};
                    border-radius: 8px;
                    padding: 5px 10px;
                    margin: 2px 0;
                }}
            """
            self.setStyleSheet(style)
            self.setWindowOpacity(self.config.opacity)
        
        def _cycle_mode(self):
            """Cycle through overlay modes."""
            modes = [OverlayMode.COMPACT, OverlayMode.EXPANDED, OverlayMode.MINIMIZED]
            current_idx = modes.index(self._mode)
            self._mode = modes[(current_idx + 1) % len(modes)]
            self._apply_mode()
            self.mode_changed.emit(self._mode.value)
        
        def _apply_mode(self):
            """Apply current mode to UI."""
            if self._mode == OverlayMode.MINIMIZED:
                self.setFixedSize(50, 50)
                self._messages_area.hide()
                self._compact_label.hide()
                self._input.hide()
                self._send_btn.hide()
                self._title.setText("AI")
            elif self._mode == OverlayMode.COMPACT:
                self.setFixedSize(self.config.width, 120)
                self._messages_area.hide()
                self._compact_label.show()
                self._input.show()
                self._send_btn.show()
                self._title.setText("AI Assistant")
            else:  # EXPANDED
                self.setFixedSize(self.config.width, self.config.height)
                self._messages_area.show()
                self._compact_label.hide()
                self._input.show()
                self._send_btn.show()
                self._title.setText("AI Assistant")
        
        def set_mode(self, mode: OverlayMode):
            """Set specific mode."""
            self._mode = mode
            self._apply_mode()
        
        def _send_message(self):
            """Send user message."""
            text = self._input.text().strip()
            if text:
                self.add_message("user", text)
                self._input.clear()
                self.message_sent.emit(text)
                
                # Reset auto-hide timer
                self._reset_hide_timer()
        
        def add_message(self, role: str, content: str):
            """Add a message to the display."""
            self._messages.append({"role": role, "content": content})
            
            # Update compact label
            self._compact_label.setText(content[:100] + "..." if len(content) > 100 else content)
            
            # Add to expanded view
            msg_label = QLabel(content)
            msg_label.setWordWrap(True)
            msg_label.setProperty("class", f"{role}Message")
            msg_label.setStyleSheet(f"""
                background-color: {'#0f3460' if role == 'user' else '#2a2a4e'};
                color: {self.config.text_color};
                border-radius: 8px;
                padding: 8px;
                margin: 4px 0;
            """)
            self._messages_layout.addWidget(msg_label)
            
            # Scroll to bottom
            QTimer.singleShot(100, lambda: self._messages_area.verticalScrollBar().setValue(
                self._messages_area.verticalScrollBar().maximum()
            ))
            
            # Show overlay if configured
            if self.config.show_on_response and role == "assistant":
                self.show()
            
            self._reset_hide_timer()
        
        def _reset_hide_timer(self):
            """Reset the auto-hide timer."""
            if self.config.auto_hide_delay > 0:
                self._hide_timer.start(self.config.auto_hide_delay)
        
        def _auto_hide(self):
            """Auto-hide the overlay."""
            if self._mode != OverlayMode.MINIMIZED:
                self.set_mode(OverlayMode.MINIMIZED)
        
        # Mouse events for dragging
        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                self._dragging = True
                self._drag_position = event.globalPos() - self.frameGeometry().topLeft()
                event.accept()
        
        def mouseMoveEvent(self, event):
            if self._dragging and event.buttons() == Qt.LeftButton:
                self.move(event.globalPos() - self._drag_position)
                event.accept()
        
        def mouseReleaseEvent(self, event):
            self._dragging = False
        
        def clear_messages(self):
            """Clear all messages."""
            self._messages.clear()
            self._compact_label.clear()
            
            # Clear expanded view
            while self._messages_layout.count():
                item = self._messages_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
    
    
    class OverlayManager:
        """
        Manage game overlay lifecycle and hotkeys.
        """
        
        def __init__(self, config: OverlayConfig = None):
            self.config = config or OverlayConfig()
            self._overlay: Optional[OverlayWidget] = None
            self._message_handler: Optional[Callable[[str], None]] = None
            self._hotkey_registered = False
        
        def create_overlay(self) -> OverlayWidget:
            """Create and return overlay widget."""
            if self._overlay is None:
                self._overlay = OverlayWidget(self.config)
                self._overlay.message_sent.connect(self._handle_message)
            return self._overlay
        
        def show(self):
            """Show the overlay."""
            if self._overlay:
                self._overlay.show()
                self._overlay.raise_()
        
        def hide(self):
            """Hide the overlay."""
            if self._overlay:
                self._overlay.hide()
        
        def toggle(self):
            """Toggle overlay visibility."""
            if self._overlay:
                if self._overlay.isVisible():
                    self.hide()
                else:
                    self.show()
        
        def set_message_handler(self, handler: Callable[[str], None]):
            """Set callback for user messages."""
            self._message_handler = handler
        
        def _handle_message(self, text: str):
            """Handle user message from overlay."""
            if self._message_handler:
                self._message_handler(text)
        
        def add_response(self, content: str):
            """Add AI response to overlay."""
            if self._overlay:
                self._overlay.add_message("assistant", content)
        
        def setup_hotkeys(self):
            """Set up global hotkeys (requires pynput or keyboard library)."""
            try:
                import keyboard

                # Toggle overlay
                keyboard.add_hotkey(
                    self.config.toggle_hotkey.lower().replace('ctrl', 'ctrl').replace('shift', 'shift'),
                    self.toggle
                )
                
                # Cycle mode
                keyboard.add_hotkey(
                    self.config.mode_cycle_hotkey.lower(),
                    lambda: self._overlay._cycle_mode() if self._overlay else None
                )
                
                self._hotkey_registered = True
                logger.info(f"Hotkeys registered: toggle={self.config.toggle_hotkey}")
                
            except ImportError:
                logger.warning("keyboard library not available, hotkeys disabled")
        
        def cleanup(self):
            """Clean up resources."""
            if self._hotkey_registered:
                try:
                    import keyboard
                    keyboard.unhook_all()
                except Exception:
                    pass  # Cleanup should not raise
            
            if self._overlay:
                self._overlay.close()
                self._overlay = None
        
        def save_config(self, path: str):
            """Save configuration to file."""
            config_dict = {
                "position": self.config.position.value,
                "custom_x": self.config.custom_x,
                "custom_y": self.config.custom_y,
                "width": self.config.width,
                "height": self.config.height,
                "opacity": self.config.opacity,
                "background_color": self.config.background_color,
                "text_color": self.config.text_color,
                "accent_color": self.config.accent_color,
                "font_size": self.config.font_size,
                "always_on_top": self.config.always_on_top,
                "auto_hide_delay": self.config.auto_hide_delay,
                "toggle_hotkey": self.config.toggle_hotkey
            }
            
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        def load_config(self, path: str):
            """Load configuration from file."""
            if Path(path).exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                
                if 'position' in data:
                    self.config.position = OverlayPosition(data['position'])
                for key in ['custom_x', 'custom_y', 'width', 'height', 'opacity',
                           'background_color', 'text_color', 'accent_color',
                           'font_size', 'always_on_top', 'auto_hide_delay', 'toggle_hotkey']:
                    if key in data:
                        setattr(self.config, key, data[key])


    def create_game_overlay(
        config: OverlayConfig = None,
        message_handler: Callable[[str], None] = None
    ) -> OverlayManager:
        """
        Create a game overlay manager.
        
        Args:
            config: Overlay configuration
            message_handler: Callback for user messages
        
        Returns:
            OverlayManager instance
        """
        manager = OverlayManager(config)
        if message_handler:
            manager.set_message_handler(message_handler)
        return manager

else:
    class OverlayWidget:
        pass
    
    class OverlayManager:
        pass
    
    def create_game_overlay(*args, **kwargs):
        raise ImportError("PyQt5 required for game overlay")
