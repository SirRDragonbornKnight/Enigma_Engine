"""
AI Overlay Window - Transparent always-on-top interface for gaming and multitasking.

Main overlay window that floats above all applications with transparency,
providing quick AI interaction without leaving games or other apps.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QFrame, QGraphicsOpacityEffect
)
from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSignal
from PyQt5.QtGui import QFont

from .overlay_modes import (
    OverlayMode, OverlayPosition, OverlaySettings,
    MinimalOverlay, CompactOverlay, FullOverlay
)
from .overlay_themes import OverlayTheme, get_theme
from .overlay_chat import OverlayChatBridge

logger = logging.getLogger(__name__)


class AIOverlay(QWidget):
    """
    Transparent overlay window for AI interaction.
    
    Features:
    - Always on top of all windows (including games)
    - Transparent background
    - Click-through option (can interact with game through overlay)
    - Resizable and draggable
    - Multiple display modes (minimal, compact, full)
    - Position memory (remembers where user placed it)
    
    Signals:
        closed: Emitted when overlay is closed
        mode_changed: Emitted when display mode changes
    """
    
    closed = pyqtSignal()
    mode_changed = pyqtSignal(str)  # mode name
    
    def __init__(self, config_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        
        # Initialize settings
        self.settings = OverlaySettings()
        self.config_path = config_path or self._get_default_config_path()
        self._load_settings()
        
        # Initialize chat bridge
        self.chat_bridge = OverlayChatBridge(self)
        self.chat_bridge.response_received.connect(self._on_response_received)
        
        # Dragging state
        self._drag_pos: Optional[QPoint] = None
        self._is_dragging = False
        
        # Setup window
        self._setup_window()
        self._apply_theme()
        self._setup_ui()
        
        # Restore position if enabled
        if self.settings.remember_position:
            self._restore_position()
            
    def _get_default_config_path(self) -> str:
        """Get default config path."""
        try:
            from ...config import CONFIG
            data_dir = Path(CONFIG.get("data_dir", "data"))
        except Exception:
            data_dir = Path("data")
        return str(data_dir / "overlay_settings.json")
        
    def _setup_window(self):
        """Setup window flags and attributes for overlay behavior."""
        # Window flags for overlay behavior
        flags = Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool
        
        if not self.settings.always_on_top:
            flags = Qt.FramelessWindowHint | Qt.Tool
            
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Set opacity
        opacity_effect = QGraphicsOpacityEffect()
        opacity_effect.setOpacity(self.settings.opacity)
        self.setGraphicsEffect(opacity_effect)
        
        # Click-through setup
        if self.settings.click_through:
            self.setAttribute(Qt.WA_TransparentForMouseEvents)
            
    def _setup_ui(self):
        """Setup UI based on current mode."""
        # Clear existing layout
        if self.layout():
            QWidget().setLayout(self.layout())
            
        # Create new layout based on mode
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Main frame
        self.main_frame = QFrame()
        self.main_frame.setObjectName("overlayFrame")
        frame_layout = QVBoxLayout(self.main_frame)
        
        # Build UI based on mode
        if self.settings.mode == OverlayMode.MINIMAL:
            self._build_minimal_ui(frame_layout)
        elif self.settings.mode == OverlayMode.COMPACT:
            self._build_compact_ui(frame_layout)
        elif self.settings.mode == OverlayMode.FULL:
            self._build_full_ui(frame_layout)
        elif self.settings.mode == OverlayMode.HIDDEN:
            self.hide()
            return
            
        layout.addWidget(self.main_frame)
        
        # Set initial size based on mode
        self._resize_for_mode()
        
    def _build_minimal_ui(self, layout: QVBoxLayout):
        """Build minimal mode UI."""
        config = MinimalOverlay()
        
        # Header with drag handle
        header = QFrame()
        header.setFixedHeight(30)
        header.mousePressEvent = self._header_press
        header.mouseMoveEvent = self._header_move
        header.mouseReleaseEvent = self._header_release
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)
        
        if config.show_avatar:
            avatar_label = QLabel("🤖")
            avatar_label.setFixedSize(24, 24)
            header_layout.addWidget(avatar_label)
            
        header_layout.addStretch()
        
        # Mode switch button
        mode_btn = QPushButton("▲")
        mode_btn.setFixedSize(20, 20)
        mode_btn.setToolTip("Expand")
        mode_btn.clicked.connect(lambda: self.set_mode(OverlayMode.COMPACT))
        header_layout.addWidget(mode_btn)
        
        # Close button
        close_btn = QPushButton("×")
        close_btn.setFixedSize(20, 20)
        close_btn.setToolTip("Close overlay")
        close_btn.clicked.connect(self._close_overlay)
        header_layout.addWidget(close_btn)
        
        layout.addWidget(header)
        
        # Response label
        self.response_label = QLabel("Ready")
        self.response_label.setWordWrap(True)
        self.response_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.response_label)
        
    def _build_compact_ui(self, layout: QVBoxLayout):
        """Build compact mode UI."""
        config = CompactOverlay()
        
        # Header with drag handle
        header = QFrame()
        header.setFixedHeight(30)
        header.mousePressEvent = self._header_press
        header.mouseMoveEvent = self._header_move
        header.mouseReleaseEvent = self._header_release
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)
        
        if config.show_avatar:
            avatar_label = QLabel("🤖")
            avatar_label.setFixedSize(24, 24)
            header_layout.addWidget(avatar_label)
            
        if config.show_name:
            name_label = QLabel("AI Assistant")
            name_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
            header_layout.addWidget(name_label)
            
        header_layout.addStretch()
        
        # Minimize button
        min_btn = QPushButton("▼")
        min_btn.setFixedSize(20, 20)
        min_btn.setToolTip("Minimize")
        min_btn.clicked.connect(lambda: self.set_mode(OverlayMode.MINIMAL))
        header_layout.addWidget(min_btn)
        
        # Expand button
        exp_btn = QPushButton("▲")
        exp_btn.setFixedSize(20, 20)
        exp_btn.setToolTip("Expand")
        exp_btn.clicked.connect(lambda: self.set_mode(OverlayMode.FULL))
        header_layout.addWidget(exp_btn)
        
        # Close button
        close_btn = QPushButton("×")
        close_btn.setFixedSize(20, 20)
        close_btn.setToolTip("Close overlay")
        close_btn.clicked.connect(self._close_overlay)
        header_layout.addWidget(close_btn)
        
        layout.addWidget(header)
        
        # Response area
        self.response_area = QTextEdit()
        self.response_area.setReadOnly(True)
        self.response_area.setMaximumHeight(80)
        self.response_area.setPlaceholderText("AI response will appear here...")
        layout.addWidget(self.response_area)
        
        # Input area
        if config.show_input:
            input_layout = QHBoxLayout()
            
            self.input_field = QLineEdit()
            self.input_field.setPlaceholderText("Type here...")
            self.input_field.returnPressed.connect(self._send_message)
            input_layout.addWidget(self.input_field)
            
            send_btn = QPushButton("➤")
            send_btn.setFixedSize(30, 30)
            send_btn.clicked.connect(self._send_message)
            input_layout.addWidget(send_btn)
            
            layout.addLayout(input_layout)
            
    def _build_full_ui(self, layout: QVBoxLayout):
        """Build full mode UI."""
        config = FullOverlay()
        
        # Header with drag handle and controls
        header = QFrame()
        header.setFixedHeight(35)
        header.mousePressEvent = self._header_press
        header.mouseMoveEvent = self._header_move
        header.mouseReleaseEvent = self._header_release
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 0, 10, 0)
        
        if config.show_avatar:
            avatar_label = QLabel("🤖")
            avatar_label.setFixedSize(32, 32)
            header_layout.addWidget(avatar_label)
            
        if config.show_name:
            name_label = QLabel("AI Assistant")
            name_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
            header_layout.addWidget(name_label)
            
        header_layout.addStretch()
        
        if config.show_controls:
            # Minimize button
            min_btn = QPushButton("▼")
            min_btn.setFixedSize(24, 24)
            min_btn.setToolTip("Compact mode")
            min_btn.clicked.connect(lambda: self.set_mode(OverlayMode.COMPACT))
            header_layout.addWidget(min_btn)
            
        # Close button
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.setToolTip("Close overlay")
        close_btn.clicked.connect(self._close_overlay)
        header_layout.addWidget(close_btn)
        
        layout.addWidget(header)
        
        # Chat history area
        if config.show_history:
            self.chat_area = QTextEdit()
            self.chat_area.setReadOnly(True)
            self.chat_area.setPlaceholderText("Chat history will appear here...")
            layout.addWidget(self.chat_area)
        else:
            # Response area only
            self.response_area = QTextEdit()
            self.response_area.setReadOnly(True)
            self.response_area.setPlaceholderText("AI response will appear here...")
            layout.addWidget(self.response_area)
            
        # Input area
        if config.show_input:
            input_layout = QHBoxLayout()
            
            self.input_field = QLineEdit()
            self.input_field.setPlaceholderText("Type your message...")
            self.input_field.returnPressed.connect(self._send_message)
            input_layout.addWidget(self.input_field)
            
            send_btn = QPushButton("➤")
            send_btn.setFixedSize(35, 35)
            send_btn.clicked.connect(self._send_message)
            input_layout.addWidget(send_btn)
            
            layout.addLayout(input_layout)
            
    def _apply_theme(self):
        """Apply current theme to overlay."""
        theme = get_theme(self.settings.theme_name)
        self.setStyleSheet(theme.to_stylesheet())
        
    def _resize_for_mode(self):
        """Resize overlay based on current mode."""
        if self.settings.mode == OverlayMode.MINIMAL:
            config = MinimalOverlay()
            self.setFixedSize(config.width, config.height)
        elif self.settings.mode == OverlayMode.COMPACT:
            config = CompactOverlay()
            self.setFixedSize(config.width, config.height)
        elif self.settings.mode == OverlayMode.FULL:
            config = FullOverlay()
            self.resize(config.width, config.height)
            self.setMinimumSize(300, 300)
            self.setMaximumSize(800, 800)
            
    def set_mode(self, mode: OverlayMode):
        """
        Switch display mode.
        
        Args:
            mode: New display mode
        """
        if self.settings.mode == mode:
            return
            
        self.settings.mode = mode
        self._setup_ui()
        self._apply_theme()
        self.mode_changed.emit(mode.value)
        self._save_settings()
        
        if mode == OverlayMode.HIDDEN:
            self.hide()
        else:
            self.show()
            
        logger.info(f"Overlay mode changed to: {mode.value}")
        
    def set_opacity(self, opacity: float):
        """
        Set overlay transparency.
        
        Args:
            opacity: Transparency value (0.0 to 1.0)
        """
        opacity = max(0.0, min(1.0, opacity))
        self.settings.opacity = opacity
        
        opacity_effect = QGraphicsOpacityEffect()
        opacity_effect.setOpacity(opacity)
        self.setGraphicsEffect(opacity_effect)
        
        self._save_settings()
        
    def set_click_through(self, enabled: bool):
        """
        Enable click-through mode.
        
        When enabled, clicks pass through to the game.
        Only AI-specific elements capture clicks.
        
        Args:
            enabled: Whether to enable click-through
        """
        self.settings.click_through = enabled
        
        if enabled:
            self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        else:
            self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            
        self._save_settings()
        logger.info(f"Click-through mode: {enabled}")
        
    def set_position(self, position: OverlayPosition, x: Optional[int] = None, y: Optional[int] = None):
        """
        Set overlay position.
        
        Args:
            position: Position preset or CUSTOM
            x: Custom x coordinate (if position is CUSTOM)
            y: Custom y coordinate (if position is CUSTOM)
        """
        from PyQt5.QtWidgets import QDesktopWidget
        
        self.settings.position = position
        
        if position == OverlayPosition.CUSTOM:
            if x is not None and y is not None:
                self.move(x, y)
                self.settings.custom_x = x
                self.settings.custom_y = y
        else:
            # Calculate position based on preset
            desktop = QDesktopWidget()
            screen_rect = desktop.availableGeometry()
            
            w, h = self.width(), self.height()
            margin = 20
            
            if position == OverlayPosition.TOP_LEFT:
                x, y = margin, margin
            elif position == OverlayPosition.TOP_RIGHT:
                x, y = screen_rect.width() - w - margin, margin
            elif position == OverlayPosition.BOTTOM_LEFT:
                x, y = margin, screen_rect.height() - h - margin
            elif position == OverlayPosition.BOTTOM_RIGHT:
                x, y = screen_rect.width() - w - margin, screen_rect.height() - h - margin
            elif position == OverlayPosition.CENTER:
                x = (screen_rect.width() - w) // 2
                y = (screen_rect.height() - h) // 2
            else:
                x, y = margin, margin
                
            self.move(x, y)
            
        self._save_settings()
        
    def set_theme(self, theme_name: str):
        """
        Set overlay theme.
        
        Args:
            theme_name: Name of theme to apply
        """
        self.settings.theme_name = theme_name
        self._apply_theme()
        self._save_settings()
        
    def set_engine(self, engine):
        """
        Set AI engine for response generation.
        
        Args:
            engine: ForgeEngine instance
        """
        self.chat_bridge.set_engine(engine)
        
    def _send_message(self):
        """Send message from input field."""
        if not hasattr(self, 'input_field'):
            return
            
        text = self.input_field.text().strip()
        if not text:
            return
            
        # Clear input
        self.input_field.clear()
        
        # Update UI to show user message
        if hasattr(self, 'chat_area'):
            self.chat_area.append(f"<b>You:</b> {text}")
        elif hasattr(self, 'response_area'):
            self.response_area.append(f"<b>You:</b> {text}")
            
        # Send via bridge
        self.chat_bridge.send_message(text)
        
    def _on_response_received(self, response: str):
        """Handle AI response."""
        # Update UI with response
        if hasattr(self, 'chat_area'):
            self.chat_area.append(f"<b>AI:</b> {response}")
        elif hasattr(self, 'response_area'):
            self.response_area.setText(response)
        elif hasattr(self, 'response_label'):
            # Truncate for minimal mode
            truncated = response[:80] + "..." if len(response) > 80 else response
            self.response_label.setText(truncated)
            
    def _header_press(self, event):
        """Handle header press for dragging."""
        if event.button() == Qt.LeftButton:
            self._is_dragging = True
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
            
    def _header_move(self, event):
        """Handle header move for dragging."""
        if self._is_dragging and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()
            
    def _header_release(self, event):
        """Handle header release."""
        if event.button() == Qt.LeftButton:
            self._is_dragging = False
            # Save new position
            if self.settings.remember_position:
                pos = self.pos()
                self.settings.custom_x = pos.x()
                self.settings.custom_y = pos.y()
                self._save_settings()
            event.accept()
            
    def _close_overlay(self):
        """Close the overlay."""
        self.closed.emit()
        self.hide()
        
    def _save_settings(self):
        """Save overlay settings to file."""
        try:
            settings_dict = {
                "mode": self.settings.mode.value,
                "position": self.settings.position.value,
                "custom_x": self.settings.custom_x,
                "custom_y": self.settings.custom_y,
                "opacity": self.settings.opacity,
                "click_through": self.settings.click_through,
                "always_on_top": self.settings.always_on_top,
                "theme_name": self.settings.theme_name,
                "hotkey": self.settings.hotkey,
                "remember_position": self.settings.remember_position,
                "show_on_startup": self.settings.show_on_startup,
            }
            
            config_path = Path(self.config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(settings_dict, f, indent=2)
                
            logger.debug(f"Overlay settings saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save overlay settings: {e}")
            
    def _load_settings(self):
        """Load overlay settings from file."""
        try:
            config_path = Path(self.config_path)
            if not config_path.exists():
                return
                
            with open(config_path, 'r') as f:
                settings_dict = json.load(f)
                
            # Update settings
            self.settings.mode = OverlayMode(settings_dict.get("mode", "compact"))
            self.settings.position = OverlayPosition(settings_dict.get("position", "top_right"))
            self.settings.custom_x = settings_dict.get("custom_x")
            self.settings.custom_y = settings_dict.get("custom_y")
            self.settings.opacity = settings_dict.get("opacity", 0.9)
            self.settings.click_through = settings_dict.get("click_through", False)
            self.settings.always_on_top = settings_dict.get("always_on_top", True)
            self.settings.theme_name = settings_dict.get("theme_name", "gaming")
            self.settings.hotkey = settings_dict.get("hotkey", "Ctrl+Shift+A")
            self.settings.remember_position = settings_dict.get("remember_position", True)
            self.settings.show_on_startup = settings_dict.get("show_on_startup", False)
            
            logger.info("Overlay settings loaded")
        except Exception as e:
            logger.warning(f"Failed to load overlay settings: {e}")
            
    def _restore_position(self):
        """Restore saved position."""
        if self.settings.custom_x is not None and self.settings.custom_y is not None:
            self.move(self.settings.custom_x, self.settings.custom_y)
        else:
            # Use position preset
            self.set_position(self.settings.position)
