"""
Simplified UI Mode for Enigma AI Engine

Minimal interface for accessibility.

Features:
- Large fonts and buttons
- High contrast mode
- Keyboard navigation
- Screen reader support
- Minimal clutter

Usage:
    from enigma_engine.gui.simplified_mode import SimplifiedUI
    
    ui = SimplifiedUI()
    ui.show()
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from PyQt5.QtCore import Qt, QSize, pyqtSignal
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QTextEdit, QLineEdit, QLabel, QScrollArea,
        QFrame, QSizePolicy, QShortcut
    )
    from PyQt5.QtGui import QFont, QKeySequence, QPalette, QColor
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


if HAS_PYQT:
    
    class SimplifiedUI(QMainWindow):
        """Minimal, accessible UI."""
        
        # Signals
        message_sent = pyqtSignal(str)
        
        def __init__(
            self,
            on_send: Optional[Callable[[str], str]] = None,
            high_contrast: bool = False,
            font_size: int = 16
        ):
            """
            Initialize simplified UI.
            
            Args:
                on_send: Callback for sending messages
                high_contrast: Enable high contrast mode
                font_size: Base font size
            """
            super().__init__()
            
            self.on_send = on_send
            self.font_size = font_size
            self.high_contrast = high_contrast
            
            # History
            self._history: List[str] = []
            self._history_index = -1
            
            self._setup_ui()
            self._setup_shortcuts()
            
            if high_contrast:
                self._apply_high_contrast()
            
            logger.info("SimplifiedUI initialized")
        
        def _setup_ui(self):
            """Setup the UI."""
            self.setWindowTitle("Enigma AI - Simple Mode")
            self.setMinimumSize(600, 400)
            
            # Central widget
            central = QWidget()
            self.setCentralWidget(central)
            
            layout = QVBoxLayout(central)
            layout.setSpacing(15)
            layout.setContentsMargins(20, 20, 20, 20)
            
            # Title
            title = QLabel("Enigma AI")
            title.setFont(QFont("Arial", self.font_size + 8, QFont.Bold))
            title.setAlignment(Qt.AlignCenter)
            layout.addWidget(title)
            
            # Status label
            self.status_label = QLabel("Ready")
            self.status_label.setFont(QFont("Arial", self.font_size - 2))
            self.status_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.status_label)
            
            # Separator
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setLineWidth(2)
            layout.addWidget(line)
            
            # Response area
            self.response_area = QTextEdit()
            self.response_area.setReadOnly(True)
            self.response_area.setFont(QFont("Arial", self.font_size))
            self.response_area.setMinimumHeight(200)
            self.response_area.setPlaceholderText("AI responses will appear here...")
            layout.addWidget(self.response_area, stretch=1)
            
            # Input area
            input_layout = QHBoxLayout()
            input_layout.setSpacing(10)
            
            self.input_field = QLineEdit()
            self.input_field.setFont(QFont("Arial", self.font_size))
            self.input_field.setMinimumHeight(50)
            self.input_field.setPlaceholderText("Type your message here...")
            self.input_field.returnPressed.connect(self._send_message)
            input_layout.addWidget(self.input_field, stretch=1)
            
            self.send_button = QPushButton("Send")
            self.send_button.setFont(QFont("Arial", self.font_size, QFont.Bold))
            self.send_button.setMinimumSize(100, 50)
            self.send_button.clicked.connect(self._send_message)
            input_layout.addWidget(self.send_button)
            
            layout.addLayout(input_layout)
            
            # Bottom buttons
            button_layout = QHBoxLayout()
            button_layout.setSpacing(10)
            
            self.clear_button = QPushButton("Clear")
            self.clear_button.setFont(QFont("Arial", self.font_size - 2))
            self.clear_button.setMinimumHeight(40)
            self.clear_button.clicked.connect(self._clear_chat)
            button_layout.addWidget(self.clear_button)
            
            self.voice_button = QPushButton("Voice")
            self.voice_button.setFont(QFont("Arial", self.font_size - 2))
            self.voice_button.setMinimumHeight(40)
            self.voice_button.clicked.connect(self._toggle_voice)
            button_layout.addWidget(self.voice_button)
            
            self.settings_button = QPushButton("Settings")
            self.settings_button.setFont(QFont("Arial", self.font_size - 2))
            self.settings_button.setMinimumHeight(40)
            self.settings_button.clicked.connect(self._open_settings)
            button_layout.addWidget(self.settings_button)
            
            self.quit_button = QPushButton("Quit")
            self.quit_button.setFont(QFont("Arial", self.font_size - 2))
            self.quit_button.setMinimumHeight(40)
            self.quit_button.clicked.connect(self.close)
            button_layout.addWidget(self.quit_button)
            
            layout.addLayout(button_layout)
            
            # Apply base styling
            self._apply_base_style()
        
        def _setup_shortcuts(self):
            """Setup keyboard shortcuts."""
            # Send message
            QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self._send_message)
            
            # Clear
            QShortcut(QKeySequence("Ctrl+L"), self).activated.connect(self._clear_chat)
            
            # Focus input
            QShortcut(QKeySequence("Ctrl+I"), self).activated.connect(
                lambda: self.input_field.setFocus()
            )
            
            # History navigation
            QShortcut(QKeySequence("Up"), self.input_field).activated.connect(
                self._history_up
            )
            QShortcut(QKeySequence("Down"), self.input_field).activated.connect(
                self._history_down
            )
            
            # Increase font
            QShortcut(QKeySequence("Ctrl+="), self).activated.connect(
                lambda: self._change_font_size(2)
            )
            
            # Decrease font
            QShortcut(QKeySequence("Ctrl+-"), self).activated.connect(
                lambda: self._change_font_size(-2)
            )
            
            # Toggle contrast
            QShortcut(QKeySequence("Ctrl+H"), self).activated.connect(
                self._toggle_contrast
            )
        
        def _apply_base_style(self):
            """Apply base styling."""
            self.setStyleSheet(f"""
                QMainWindow {{
                    background-color: #f5f5f5;
                }}
                QLabel {{
                    color: #333333;
                }}
                QLineEdit, QTextEdit {{
                    background-color: white;
                    border: 2px solid #cccccc;
                    border-radius: 5px;
                    padding: 5px;
                }}
                QLineEdit:focus, QTextEdit:focus {{
                    border-color: #0066cc;
                }}
                QPushButton {{
                    background-color: #0066cc;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 10px;
                }}
                QPushButton:hover {{
                    background-color: #0052a3;
                }}
                QPushButton:pressed {{
                    background-color: #003d7a;
                }}
                QPushButton#quit {{
                    background-color: #cc0000;
                }}
            """)
        
        def _apply_high_contrast(self):
            """Apply high contrast styling."""
            self.high_contrast = True
            
            self.setStyleSheet(f"""
                QMainWindow {{
                    background-color: black;
                }}
                QLabel {{
                    color: white;
                }}
                QLineEdit, QTextEdit {{
                    background-color: black;
                    color: white;
                    border: 3px solid white;
                    border-radius: 5px;
                    padding: 5px;
                }}
                QLineEdit:focus, QTextEdit:focus {{
                    border-color: yellow;
                }}
                QPushButton {{
                    background-color: white;
                    color: black;
                    border: 3px solid white;
                    border-radius: 5px;
                    padding: 10px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: yellow;
                    color: black;
                }}
                QPushButton:focus {{
                    border-color: yellow;
                }}
            """)
        
        def _send_message(self):
            """Send message."""
            text = self.input_field.text().strip()
            if not text:
                return
            
            # Add to history
            self._history.append(text)
            self._history_index = len(self._history)
            
            # Display user message
            self.response_area.append(f"\nYOU: {text}")
            
            # Clear input
            self.input_field.clear()
            
            # Update status
            self.status_label.setText("Thinking...")
            QApplication.processEvents()
            
            # Get response
            if self.on_send:
                try:
                    response = self.on_send(text)
                    self.response_area.append(f"\nAI: {response}")
                except Exception as e:
                    self.response_area.append(f"\nERROR: {str(e)}")
            else:
                self.response_area.append("\nAI: (No handler configured)")
            
            # Update status
            self.status_label.setText("Ready")
            
            # Scroll to bottom
            scrollbar = self.response_area.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
            # Emit signal
            self.message_sent.emit(text)
        
        def _clear_chat(self):
            """Clear chat history."""
            self.response_area.clear()
            self.status_label.setText("Chat cleared")
        
        def _toggle_voice(self):
            """Toggle voice mode."""
            self.status_label.setText("Voice mode not implemented")
        
        def _open_settings(self):
            """Open settings."""
            from PyQt5.QtWidgets import QDialog, QFormLayout, QSpinBox, QCheckBox, QDialogButtonBox
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Settings")
            dialog.setMinimumWidth(300)
            
            layout = QFormLayout(dialog)
            
            # Font size
            font_spin = QSpinBox()
            font_spin.setRange(10, 32)
            font_spin.setValue(self.font_size)
            layout.addRow("Font Size:", font_spin)
            
            # High contrast
            contrast_check = QCheckBox()
            contrast_check.setChecked(self.high_contrast)
            layout.addRow("High Contrast:", contrast_check)
            
            # Buttons
            buttons = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel
            )
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addRow(buttons)
            
            if dialog.exec_() == QDialog.Accepted:
                self._change_font_size(font_spin.value() - self.font_size)
                
                if contrast_check.isChecked() != self.high_contrast:
                    self._toggle_contrast()
        
        def _history_up(self):
            """Navigate history up."""
            if self._history and self._history_index > 0:
                self._history_index -= 1
                self.input_field.setText(self._history[self._history_index])
        
        def _history_down(self):
            """Navigate history down."""
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
                self.input_field.setText(self._history[self._history_index])
            elif self._history_index == len(self._history) - 1:
                self._history_index = len(self._history)
                self.input_field.clear()
        
        def _change_font_size(self, delta: int):
            """Change font size."""
            self.font_size = max(10, min(32, self.font_size + delta))
            
            # Update fonts
            font = QFont("Arial", self.font_size)
            self.input_field.setFont(font)
            self.response_area.setFont(font)
            
            self.status_label.setText(f"Font size: {self.font_size}")
        
        def _toggle_contrast(self):
            """Toggle high contrast mode."""
            if self.high_contrast:
                self._apply_base_style()
                self.high_contrast = False
                self.status_label.setText("Normal contrast")
            else:
                self._apply_high_contrast()
                self.status_label.setText("High contrast enabled")
        
        def add_response(self, text: str, sender: str = "AI"):
            """Add response to chat."""
            self.response_area.append(f"\n{sender}: {text}")
            
            scrollbar = self.response_area.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        
        def set_status(self, text: str):
            """Set status text."""
            self.status_label.setText(text)


    class QuickAccessBar(QWidget):
        """Quick access toolbar."""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self._setup_ui()
        
        def _setup_ui(self):
            layout = QHBoxLayout(self)
            layout.setContentsMargins(5, 5, 5, 5)
            layout.setSpacing(5)
            
            # Quick action buttons
            actions = [
                ("Help", "Get help"),
                ("Time", "What time is it?"),
                ("Weather", "How's the weather?"),
                ("News", "What's the latest news?"),
            ]
            
            for name, prompt in actions:
                btn = QPushButton(name)
                btn.setProperty("prompt", prompt)
                btn.clicked.connect(
                    lambda checked, p=prompt: self._send_quick(p)
                )
                layout.addWidget(btn)
        
        def _send_quick(self, prompt: str):
            """Send quick action."""
            parent = self.parent()
            if hasattr(parent, 'input_field'):
                parent.input_field.setText(prompt)
                parent._send_message()


else:
    # Fallback when PyQt5 not available
    class SimplifiedUI:
        """Fallback simplified UI using terminal."""
        
        def __init__(
            self,
            on_send: Optional[Callable[[str], str]] = None,
            **kwargs
        ):
            self.on_send = on_send
            self._running = False
        
        def show(self):
            """Start terminal UI."""
            self._running = True
            print("\n" + "=" * 50)
            print("ENIGMA AI - SIMPLE MODE")
            print("=" * 50)
            print("Type 'quit' to exit\n")
            
            while self._running:
                try:
                    text = input("YOU: ").strip()
                    
                    if text.lower() in ['quit', 'exit', 'q']:
                        print("Goodbye!")
                        break
                    
                    if not text:
                        continue
                    
                    if self.on_send:
                        response = self.on_send(text)
                        print(f"AI: {response}\n")
                    else:
                        print("AI: (No handler configured)\n")
                        
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}\n")
        
        def close(self):
            """Close UI."""
            self._running = False


def create_simplified_interface(
    ai_callback: Callable[[str], str],
    high_contrast: bool = False,
    font_size: int = 16
) -> SimplifiedUI:
    """
    Create simplified UI with AI integration.
    
    Args:
        ai_callback: Function to get AI response
        high_contrast: Enable high contrast
        font_size: Base font size
        
    Returns:
        Configured SimplifiedUI
    """
    return SimplifiedUI(
        on_send=ai_callback,
        high_contrast=high_contrast,
        font_size=font_size
    )


def run_simplified_mode(
    ai_callback: Optional[Callable[[str], str]] = None
):
    """
    Run simplified UI mode.
    
    Args:
        ai_callback: Function to get AI response
    """
    if HAS_PYQT:
        import sys
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        ui = SimplifiedUI(on_send=ai_callback)
        ui.show()
        
        return app.exec_()
    else:
        ui = SimplifiedUI(on_send=ai_callback)
        ui.show()
        return 0
