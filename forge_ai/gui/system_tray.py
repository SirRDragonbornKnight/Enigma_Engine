# type: ignore
# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalMemberAccess=false
"""
System Tray Integration - Keep Forge running in the background.

Features:
  - System tray icon with menu
  - Quick command overlay (hotkey activated)
  - Voice activation support ("Hey ForgeAI")
  - Background AI processing
  - Natural language command execution
"""

import sys
import os
import time
import subprocess
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QWidget
    from PyQt5.QtCore import Qt

try:
    from PyQt5.QtWidgets import (  # type: ignore[import]
        QApplication, QSystemTrayIcon, QMenu, QAction, QWidget,
        QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel,
        QTextEdit, QFrame, QShortcut, QWidgetAction, QMessageBox
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QObject, QThread, QPoint  # type: ignore[import]
    from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QKeySequence, QFont, QCursor  # type: ignore[import]
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QObject = object  # type: ignore[misc,assignment]
    QWidget = object  # type: ignore[misc,assignment]
    pyqtSignal = lambda *args: None  # type: ignore[misc,assignment]
    pyqtSlot = lambda *args: lambda f: f  # type: ignore[misc,assignment]

from ..config import CONFIG


def get_current_model_name() -> str:
    """Get the name of the currently loaded AI model."""
    try:
        from ..config import CONFIG
        # Try to get from gui_settings.json first (most accurate)
        import json
        settings_path = Path(CONFIG.get("info_dir", "information")) / "gui_settings.json"
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                if settings.get("last_model"):
                    return settings["last_model"]
        # Fallback to config
        model_name = CONFIG.get("default_model", "small_forge_ai")
        return model_name
    except:
        return "Unknown Model"


def kill_other_forge_ai_instances() -> dict:
    """Kill other ForgeAI processes to free up resources.
    
    Returns dict with:
      - killed: number of processes killed
      - current_pid: this process's PID (not killed)
      - error: error message if any
    """
    import os
    import signal
    
    current_pid = os.getpid()
    killed = 0
    errors = []
    
    try:
        import subprocess
        import sys
        
        if sys.platform == 'win32':
            # Windows: use tasklist and taskkill
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
                capture_output=True, text=True
            )
            for line in result.stdout.split('\n'):
                if 'forge_ai' in line.lower() or 'run.py' in line.lower():
                    try:
                        # Extract PID from CSV format
                        parts = line.split(',')
                        if len(parts) >= 2:
                            pid = int(parts[1].strip('"'))
                            if pid != current_pid:
                                subprocess.run(['taskkill', '/PID', str(pid), '/F'], capture_output=True)
                                killed += 1
                    except:
                        pass
        else:
            # Linux/Mac: use ps and kill
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True, text=True
            )
            for line in result.stdout.split('\n'):
                if ('forge_ai' in line.lower() or 'run.py' in line.lower()) and 'python' in line.lower():
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            pid = int(parts[1])
                            if pid != current_pid:
                                os.kill(pid, signal.SIGTERM)
                                killed += 1
                    except (ValueError, ProcessLookupError, PermissionError) as e:
                        errors.append(str(e))
        
        return {
            "killed": killed,
            "current_pid": current_pid,
            "error": "; ".join(errors) if errors else None
        }
    except Exception as e:
        return {
            "killed": killed,
            "current_pid": current_pid,
            "error": str(e)
        }


class CommandProcessor(QObject):
    """Process natural language commands in background.
    
    NOTE: This class uses ChatSync's shared engine instead of creating its own.
    The engine property delegates to ChatSync to ensure only one engine exists.
    """
    
    result_ready = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tool_interface = None
        self.model_name = "Not loaded"
    
    @property
    def engine(self):
        """Get engine from ChatSync (shared singleton)."""
        try:
            from .chat_sync import ChatSync
            return ChatSync.instance()._engine
        except Exception:
            return None
    
    @engine.setter
    def engine(self, value):
        """Set engine on ChatSync (shared singleton)."""
        try:
            from .chat_sync import ChatSync
            ChatSync.instance().set_engine(value)
        except Exception:
            pass
    
    def _ensure_engine(self):
        """Ensure tool interface is initialized."""
        if self.tool_interface is None:
            try:
                from ..core.tool_interface import ToolInterface
                self.tool_interface = ToolInterface()
            except Exception:
                pass
        # Update model name from ChatSync
        try:
            from .chat_sync import ChatSync
            chat_sync = ChatSync.instance()
            if chat_sync._model_name:
                self.model_name = chat_sync._model_name
        except Exception:
            pass
    
    def process_command(self, command: str) -> Dict[str, Any]:
        """
        Process a natural language command and determine action.
        
        Returns dict with:
          - action: The action type (chat, image, video, train, file, screen, gui, etc.)
          - params: Parameters for the action
          - response: Text response from AI
        """
        command_lower = command.lower().strip()
        
        # Quick command detection (before AI processing)
        quick_actions = self._detect_quick_action(command_lower)
        if quick_actions:
            return quick_actions
        
        # Use AI for more complex interpretation
        if self.engine:
            try:
                # Add system context for command interpretation
                prompt = f"""User command: {command}

Interpret this command and respond. If it's a request to:
- Generate an image: describe what image to create
- Generate video: describe the video
- Train/learn: acknowledge and confirm
- Edit/modify file: confirm the file operation
- Take screenshot: confirm screen capture
- Record screen: confirm recording
- Open settings/GUI: confirm opening
- Other: respond naturally

Response:"""
                response = self.engine.generate(prompt, max_gen=150, temperature=0.7)
                return {
                    "action": "chat",
                    "params": {"command": command},
                    "response": response
                }
            except Exception as e:
                return {
                    "action": "error",
                    "params": {},
                    "response": f"Error processing: {e}"
                }
        
        return {
            "action": "chat",
            "params": {},
            "response": "I heard you, but my brain isn't loaded yet. Try again in a moment."
        }
    
    def _detect_quick_action(self, command: str) -> Optional[Dict[str, Any]]:
        """Detect quick actions from keywords."""
        
        # Open GUI
        if any(kw in command for kw in ["open gui", "show gui", "open window", "show window", "open forge", "open enigma"]):
            return {"action": "open_gui", "params": {}, "response": "Opening the main window..."}
        
        # Screenshot
        if any(kw in command for kw in ["screenshot", "screen shot", "capture screen", "take a picture of screen"]):
            return {"action": "screenshot", "params": {}, "response": "Taking a screenshot..."}
        
        # Screen recording
        if any(kw in command for kw in ["record screen", "start recording", "screen record"]):
            return {"action": "record_screen", "params": {"start": True}, "response": "Starting screen recording..."}
        if any(kw in command for kw in ["stop recording", "end recording"]):
            return {"action": "record_screen", "params": {"start": False}, "response": "Stopping screen recording..."}
        
        # Image generation
        if any(kw in command for kw in ["generate image", "create image", "make image", "draw", "paint"]):
            # Extract the description after the keyword
            for kw in ["generate image of", "create image of", "make image of", "draw ", "paint "]:
                if kw in command:
                    desc = command.split(kw, 1)[-1].strip()
                    return {"action": "generate_image", "params": {"prompt": desc}, "response": f"Generating image: {desc}"}
            return {"action": "generate_image", "params": {"prompt": command}, "response": "What would you like me to draw?"}
        
        # Video generation
        if any(kw in command for kw in ["generate video", "create video", "make video"]):
            for kw in ["generate video of", "create video of", "make video of"]:
                if kw in command:
                    desc = command.split(kw, 1)[-1].strip()
                    return {"action": "generate_video", "params": {"prompt": desc}, "response": f"Generating video: {desc}"}
            return {"action": "generate_video", "params": {}, "response": "What video would you like me to create?"}
        
        # Training
        if any(kw in command for kw in ["train on", "learn from", "add to training", "train with"]):
            for kw in ["train on ", "learn from ", "add to training ", "train with "]:
                if kw in command:
                    target = command.split(kw, 1)[-1].strip()
                    return {"action": "train", "params": {"data": target}, "response": f"Adding to training data: {target}"}
        
        if any(kw in command for kw in ["start training", "train model", "begin training"]):
            return {"action": "train", "params": {"start": True}, "response": "Starting model training..."}
        
        # File operations
        if any(kw in command for kw in ["open file", "edit file", "read file"]):
            for kw in ["open file ", "edit file ", "read file "]:
                if kw in command:
                    filepath = command.split(kw, 1)[-1].strip()
                    return {"action": "file", "params": {"path": filepath, "operation": "open"}, "response": f"Opening file: {filepath}"}
        
        # Avatar
        if any(kw in command for kw in ["show avatar", "connect avatar", "start avatar"]):
            return {"action": "avatar", "params": {"show": True}, "response": "Connecting avatar..."}
        if any(kw in command for kw in ["hide avatar", "disconnect avatar", "stop avatar"]):
            return {"action": "avatar", "params": {"show": False}, "response": "Disconnecting avatar..."}
        
        # Look at screen / analyze
        if any(kw in command for kw in ["look at screen", "what's on screen", "analyze screen", "see my screen"]):
            return {"action": "vision", "params": {"capture": True}, "response": "Looking at your screen..."}
        
        # Settings
        if any(kw in command for kw in ["open settings", "show settings"]):
            return {"action": "settings", "params": {}, "response": "Opening settings..."}
        
        # Help
        if any(kw in command for kw in ["help", "how to", "show help", "open help"]):
            return {"action": "help", "params": {}, "response": "Opening help..."}
        
        # Open folders
        if "open outputs" in command or "show outputs" in command:
            return {"action": "open_folder", "params": {"folder": "outputs"}, "response": "Opening outputs folder..."}
        if "open models" in command or "show models" in command:
            return {"action": "open_folder", "params": {"folder": "models"}, "response": "Opening models folder..."}
        if "open data" in command or "show data" in command:
            return {"action": "open_folder", "params": {"folder": "data"}, "response": "Opening data folder..."}
        if "open docs" in command or "show docs" in command or "documentation" in command:
            return {"action": "open_folder", "params": {"folder": "docs"}, "response": "Opening documentation..."}
        
        # Exit/quit
        if any(kw in command for kw in ["exit", "quit", "goodbye", "shut down"]):
            return {"action": "exit", "params": {}, "response": "Goodbye! Shutting down..."}
        
        return None


class HelpWindow(QWidget):
    """Separate window to display help content."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ForgeAI - Help")
        self.setWindowFlags(Qt.Window)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title = QLabel("ForgeAI Help")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #3498db; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Content area
        self.content = QTextEdit()
        self.content.setReadOnly(True)
        self.content.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 8px;
                padding: 10px;
                font-size: 12px;
                font-family: Consolas, monospace;
                color: #ddd;
            }
        """)
        layout.addWidget(self.content)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-size: 12px;
                color: white;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)
        
        self.setMinimumSize(500, 400)
        self.resize(600, 500)
        
        # Dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ddd;
            }
        """)
    
    def set_content(self, text: str):
        self.content.setPlainText(text)


# Track all Quick Chat instances for communication
_quick_chat_instances = []


class QuickCommandOverlay(QWidget):
    """
    A floating overlay for quick commands.
    Appears when hotkey is pressed.
    
    Features:
    - Resizable window (drag edges/corners)
    - Close button like a window
    - ESC opens full GUI instead of just closing
    - Alt+F4 closes the entire app
    - Expand button to make it larger
    - Multiple instances can communicate
    """
    
    command_submitted = pyqtSignal(str)
    close_requested = pyqtSignal()
    open_gui_requested = pyqtSignal()  # New signal for ESC to open GUI
    stop_requested = pyqtSignal()  # Signal to stop generation
    instance_message = pyqtSignal(str, str)  # For inter-instance communication (sender_id, message)
    
    @classmethod
    def get_all_instances(cls):
        """Get all active Quick Chat instances."""
        global _quick_chat_instances
        # Clean up dead references
        _quick_chat_instances = [w for w in _quick_chat_instances if w is not None and not w.isHidden()]
        return _quick_chat_instances
    
    @classmethod
    def broadcast_message(cls, sender_id: str, message: str):
        """Send a message to all other Quick Chat instances."""
        for instance in cls.get_all_instances():
            if instance._instance_id != sender_id:
                instance.instance_message.emit(sender_id, message)
    
    @classmethod
    def close_all_instances(cls):
        """Close all Quick Chat instances."""
        from PyQt5.QtWidgets import QApplication
        instances = cls.get_all_instances()
        for instance in instances:
            instance.hide()
        # Also quit the app
        QApplication.quit()
    
    def __init__(self, parent=None):
        # Initialize size attributes BEFORE calling parent __init__
        # (in case any signals trigger setup_ui early)
        self._min_width = 450
        self._min_height = 300  # Taller to show chat history
        self._expanded_height = 350
        self._is_expanded = True  # Always expanded now
        self._drag_pos = None
        self._resize_edge = None
        self._is_responding = False  # Track if AI is responding
        self._stop_requested = False  # Track if stop was requested
        self._auto_speak = False  # Voice output toggle
        self._last_response = None  # Store last response for speak-last
        
        super().__init__(parent)
        
        # Generate unique instance ID and register
        import uuid
        self._instance_id = str(uuid.uuid4())[:8]
        global _quick_chat_instances
        _quick_chat_instances.append(self)
        
        # Connect inter-instance messaging
        self.instance_message.connect(self._on_instance_message)
        
        # Load settings for always-on-top
        always_on_top = self._load_mini_chat_settings().get("mini_chat_always_on_top", True)
        
        # Use Window flag for proper window controls including Alt+F4
        flags = Qt.Window | Qt.FramelessWindowHint
        if always_on_top:
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Enable custom context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
        self.setup_ui()
        self.history = []
        self.history_index = -1
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Main frame with styling
        self.frame = QFrame()
        self.frame.setObjectName("commandFrame")
        self.frame.setStyleSheet("""
            #commandFrame {
                background-color: rgba(30, 30, 30, 0.98);
                border: 2px solid #3498db;
                border-radius: 12px;
            }
        """)
        
        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(15, 8, 15, 10)
        frame_layout.setSpacing(6)
        
        # Header widget for dragging (like a window title bar)
        self._header_widget = QFrame()
        self._header_widget.setFixedHeight(30)
        # No special cursor - keep default arrow like normal windows
        self._header_widget.setStyleSheet("background: rgba(50, 50, 50, 0.5); border-radius: 4px;")
        self._header_widget.setMouseTracking(True)
        
        # Install event filter for proper drag handling
        self._header_widget.mousePressEvent = lambda e: self._header_mouse_press(e)
        self._header_widget.mouseMoveEvent = lambda e: self._header_mouse_move(e)
        self._header_widget.mouseReleaseEvent = lambda e: self._header_mouse_release(e)
        
        header_layout = QHBoxLayout(self._header_widget)
        header_layout.setContentsMargins(8, 0, 0, 0)
        header_layout.setSpacing(8)
        
        self.title_label = QLabel(get_current_model_name())
        self.title_label.setStyleSheet("color: #3498db; font-size: 14px; font-weight: bold;")
        header_layout.addWidget(self.title_label)
        
        header_layout.addStretch()
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        header_layout.addWidget(self.status_label)
        
        # New Chat button
        new_chat_btn = QPushButton("+")
        new_chat_btn.setFixedSize(24, 24)
        new_chat_btn.setToolTip("New Chat - Start a fresh conversation")
        new_chat_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #555;
                border-radius: 4px;
                color: #888;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #444;
                color: #2ecc71;
            }
        """)
        new_chat_btn.clicked.connect(self._new_chat)
        header_layout.addWidget(new_chat_btn)
        
        # Close button - shows close dialog
        self.close_btn = QPushButton("X")
        self.close_btn.setFixedSize(24, 24)
        self.close_btn.setToolTip("Close Forge")
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #555;
                border-radius: 4px;
                color: #888;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e74c3c;
                border-color: #e74c3c;
                color: white;
            }
        """)
        self.close_btn.clicked.connect(self._show_close_dialog)  # X button shows dialog
        header_layout.addWidget(self.close_btn)
        
        frame_layout.addWidget(self._header_widget)
        
        # Chat history area (always visible, above input like main chat)
        self.response_area = QTextEdit()
        self.response_area.setReadOnly(True)
        self.response_area.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        self.response_area.setPlaceholderText("Type a message below to chat with your AI")
        self.response_area.setMinimumHeight(150)
        self.response_area.setLineWrapMode(QTextEdit.WidgetWidth)  # Wrap text at widget edge
        self.response_area.setWordWrapMode(3)  # WrapAtWordBoundaryOrAnywhere
        self.response_area.setStyleSheet("""
            QTextEdit {
                background-color: rgba(40, 40, 40, 0.9);
                border: 1px solid #444;
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
                color: #ccc;
            }
        """)
        frame_layout.addWidget(self.response_area)
        
        # Input row with chat button and voice button
        input_layout = QHBoxLayout()
        input_layout.setSpacing(8)
        
        # Command input
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Chat here...")
        self.command_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(50, 50, 50, 0.9);
                border: 1px solid #555;
                border-radius: 8px;
                padding: 10px 15px;
                font-size: 14px;
                color: white;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """)
        self.command_input.returnPressed.connect(self._on_chat)
        input_layout.addWidget(self.command_input)
        
        # Chat/Send button
        self.chat_btn = QPushButton("Send")
        self.chat_btn.setFixedSize(50, 40)
        self.chat_btn.setToolTip("Send message")
        self.chat_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                border: none;
                border-radius: 8px;
                color: white;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c5980;
            }
        """)
        self.chat_btn.clicked.connect(self._on_chat)
        input_layout.addWidget(self.chat_btn)
        
        # Stop button (hidden by default, shown during generation)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFixedSize(50, 40)
        self.stop_btn.setToolTip("Stop generation")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                border: none;
                border-radius: 8px;
                color: white;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #922b21;
            }
        """)
        self.stop_btn.clicked.connect(self._stop_generation)
        self.stop_btn.hide()  # Hidden by default
        input_layout.addWidget(self.stop_btn)
        
        frame_layout.addLayout(input_layout)
        
        # Bottom row: hint text and open GUI link
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(8)
        
        # Hint label - shows context menu tip
        self.hint_label = QLabel("Right-click for options")
        self.hint_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 10px;
            }
        """)
        bottom_layout.addWidget(self.hint_label)
        
        # Open GUI button - small text link style
        self.gui_btn = QPushButton("Open GUI")
        self.gui_btn.setFixedHeight(18)
        self.gui_btn.setToolTip("Open Full GUI (ESC)")
        self.gui_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #3498db;
                font-size: 10px;
                padding: 0 4px;
            }
            QPushButton:hover {
                color: #5dade2;
                text-decoration: underline;
            }
        """)
        self.gui_btn.clicked.connect(self._open_gui)
        bottom_layout.addWidget(self.gui_btn)
        
        bottom_layout.addStretch()
        
        frame_layout.addLayout(bottom_layout)
        
        layout.addWidget(self.frame)
        
        # Initial size (larger to show chat history)
        self.setMinimumSize(self._min_width, self._min_height)
        self.resize(500, 350)
        
        # Setup responding indicator animation timer
        self._responding_dots = 0
        self._responding_timer = QTimer(self)
        self._responding_timer.timeout.connect(self._update_responding_indicator)
    
    def _show_context_menu(self, position):
        """Show right-click context menu with settings and actions."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2b2b2b;
                border: 2px solid #3498db;
                border-radius: 6px;
                padding: 5px;
            }
            QMenu::item {
                background-color: transparent;
                color: #ecf0f1;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #3498db;
            }
            QMenu::separator {
                height: 1px;
                background: #555;
                margin: 5px 0px;
            }
        """)
        
        # Always on top toggle
        always_on_top_action = QAction("✓ Always on Top" if (self.windowFlags() & Qt.WindowStaysOnTopHint) else "  Always on Top", self)
        always_on_top_action.triggered.connect(self._toggle_always_on_top)
        menu.addAction(always_on_top_action)
        
        menu.addSeparator()
        
        # Voice input
        voice_input_action = QAction("Voice Input...", self)
        voice_input_action.triggered.connect(self._toggle_voice_input_menu)
        menu.addAction(voice_input_action)
        
        # Voice output toggle
        voice_status = "ON" if self._auto_speak else "OFF"
        voice_output_action = QAction(f"Voice Output ({voice_status})", self)
        voice_output_action.triggered.connect(self._toggle_voice_output_menu)
        menu.addAction(voice_output_action)
        
        # Speak last response
        speak_last_action = QAction("Speak Last Response", self)
        speak_last_action.triggered.connect(self._speak_last_response)
        speak_last_action.setEnabled(bool(getattr(self, '_last_response', None)))
        menu.addAction(speak_last_action)
        
        menu.addSeparator()
        
        # Avatar submenu
        avatar_menu = menu.addMenu("Avatar")
        
        # Run avatar (toggle desktop overlay)
        run_avatar_action = QAction("Toggle Desktop Avatar", self)
        run_avatar_action.triggered.connect(self._run_avatar)
        avatar_menu.addAction(run_avatar_action)
        
        # Reset position (if avatar went off-screen)
        reset_pos_action = QAction("Reset Position", self)
        reset_pos_action.triggered.connect(self._reset_avatar_position)
        avatar_menu.addAction(reset_pos_action)
        
        # Open avatar tab
        open_avatar_tab_action = QAction("Open Avatar Tab", self)
        open_avatar_tab_action.triggered.connect(self._open_avatar_tab)
        avatar_menu.addAction(open_avatar_tab_action)
        
        menu.addSeparator()
        
        # New window
        new_window_action = QAction("New Chat Window", self)
        new_window_action.triggered.connect(self._create_new_instance)
        menu.addAction(new_window_action)
        
        menu.addSeparator()
        
        # Close options
        close_action = QAction("Close Window", self)
        close_action.triggered.connect(self._close_overlay)
        menu.addAction(close_action)
        
        quit_action = QAction("Quit Forge", self)
        quit_action.triggered.connect(self._quit_app)
        menu.addAction(quit_action)
        
        # Show menu at cursor position
        menu.exec_(self.mapToGlobal(position))
    
    def _toggle_always_on_top(self):
        """Toggle always on top setting."""
        current_on_top = bool(self.windowFlags() & Qt.WindowStaysOnTopHint)
        new_on_top = not current_on_top
        self.set_always_on_top(new_on_top)
        
        # Save setting
        try:
            import json
            from pathlib import Path
            settings_path = Path(CONFIG.get("info_dir", "information")) / "gui_settings.json"
            settings = {}
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
            settings["mini_chat_always_on_top"] = new_on_top
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            self.set_status("Always on top: " + ("ON" if new_on_top else "OFF"))
        except Exception as e:
            self.set_status(f"Settings error: {e}")
    
    def _toggle_voice_input_menu(self):
        """Toggle voice input from menu."""
        # Simulate button click
        if hasattr(self, '_start_voice_input'):
            self._start_voice_input()
        else:
            self.set_status("Voice input: Click and hold to speak")
    
    def _toggle_voice_output_menu(self):
        """Toggle voice output from menu."""
        self._auto_speak = not self._auto_speak
        status = "ON" if self._auto_speak else "OFF"
        self.set_status(f"Voice output: {status}")
        
        # Sync with main window if available
        main_window = self._get_main_window()
        if main_window:
            main_window.auto_speak = self._auto_speak
    
    def _create_new_instance(self):
        """Create a new Quick Chat window."""
        new_overlay = QuickCommandOverlay()
        new_overlay.show()
        # Position slightly offset from current window
        new_overlay.move(self.x() + 30, self.y() + 30)
        self.set_status("New window created")
    
    def _load_mini_chat_settings(self):
        """Load Quick Chat settings from gui_settings.json."""
        try:
            import json
            from pathlib import Path
            settings_path = Path(CONFIG.get("info_dir", "information")) / "gui_settings.json"
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    # Load user display name
                    self.user_display_name = settings.get("user_display_name", "You")
                    return settings
        except Exception:
            pass
        self.user_display_name = "You"
        return {}
    
    def set_always_on_top(self, on_top: bool):
        """Update the always-on-top setting at runtime."""
        current_flags = self.windowFlags()
        if on_top:
            self.setWindowFlags(current_flags | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(current_flags & ~Qt.WindowStaysOnTopHint)
        # Re-show after changing flags (required by Qt)
        if self.isVisible():
            self.show()
    
    def _update_responding_indicator(self):
        """Update status text only - no chat spam."""
        self._responding_dots = (self._responding_dots + 1) % 4
        dots = "." * self._responding_dots
        self.set_status(f"Thinking{dots}")
    
    def start_responding(self):
        """Show the responding indicator (animated status only, no chat spam)."""
        self._is_responding = True
        self._stop_requested = False
        self._responding_dots = 0
        # Don't add thinking indicator to chat - it shows before user message
        # Instead just animate the status bar and show after a brief delay
        self._responding_timer.start(400)  # Animate status bar only
        self.chat_btn.hide()  # Hide send button
        self.stop_btn.show()  # Show stop button
        self.set_status("Thinking...")
        
        # Add thinking indicator AFTER user message is shown (use timer)
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self._add_thinking_indicator)
    
    def _add_thinking_indicator(self):
        """Add thinking indicator after user message is displayed."""
        if not self._is_responding:
            return  # Already done
        ai_name = getattr(self, 'ai_display_name', None) or getattr(self, 'model_name', 'AI')
        self.response_area.append(
            f'<div id="thinking" style="color: #f9e2af; padding: 4px;"><i>{ai_name} is thinking...</i></div>'
        )
        self.response_area.verticalScrollBar().setValue(
            self.response_area.verticalScrollBar().maximum()
        )
    
    def stop_responding(self):
        """Remove the thinking indicator from chat."""
        self._is_responding = False
        self._responding_timer.stop()
        self._remove_thinking_indicator()
        self.stop_btn.hide()  # Hide stop button
        self.chat_btn.show()  # Show send button
        self.set_status("Ready")
    
    def _stop_generation(self):
        """Stop the current AI generation via ChatSync."""
        self._stop_requested = True
        self.stop_requested.emit()
        
        # Use ChatSync to stop - this stops in both Quick Chat and main GUI
        try:
            from .chat_sync import ChatSync
            chat_sync = ChatSync.instance()
            chat_sync.stop_generation()
        except Exception:
            pass
        
        self.stop_responding()
        # Add stopped message
        self.response_area.append(
            "<div style='color: #e74c3c; padding: 4px;'><i>Generation stopped by user</i></div>"
        )
        self.set_status("Stopped")
    
    def _remove_thinking_indicator(self):
        """Remove the thinking indicator from chat."""
        html = self.response_area.toHtml()
        # Remove the thinking div
        import re
        html = re.sub(r'<div id="thinking"[^>]*>.*?</div>', '', html, flags=re.IGNORECASE | re.DOTALL)
        self.response_area.setHtml(html)
    
    def _quit_app(self):
        """Quit the entire application and clean up tray icons."""
        from PyQt5.QtWidgets import QApplication
        try:
            self.hide()
            
            # Close all Quick Chat instances
            QuickCommandOverlay.close_all_instances()
            
            # Clean up system tray icons to prevent ghost icons
            for widget in QApplication.topLevelWidgets():
                # Hide and clean up any system tray instance
                if hasattr(widget, 'tray_icon'):
                    widget.tray_icon.hide()
                    widget.tray_icon.setVisible(False)
            
            # Also clean up the ForgeSystemTray if accessible
            app = QApplication.instance()
            if app:
                for child in app.children():
                    if hasattr(child, 'tray_icon'):
                        child.tray_icon.hide()
                        child.tray_icon.setVisible(False)
            
            # Process events to ensure tray cleanup completes
            QApplication.processEvents()
            
            # Quit Qt application
            QApplication.quit()
            
            # Force exit if needed
            import sys
            sys.exit(0)
        except Exception as e:
            print(f"Error during quit: {e}")
            import sys
            sys.exit(1)
    
    def _on_instance_message(self, sender_id: str, message: str):
        """Handle message from another Quick Chat instance."""
        # Display system message about inter-instance communication
        self.response_area.append(
            f"<div style='color: #9b59b6; padding: 4px; font-style: italic;'>"
            f"[From window {sender_id}]: {message}</div>"
        )
    
    def _show_close_dialog(self):
        """Show styled dialog with close options."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Close Forge")
        dialog.setFixedSize(350, 180)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                border: 2px solid #3498db;
                border-radius: 10px;
            }
            QLabel {
                color: #ecf0f1;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 6px 10px;
                border-radius: 5px;
                font-size: 10px;
                min-width: 70px;
                max-width: 100px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton#hideBtn {
                background-color: #27ae60;
            }
            QPushButton#hideBtn:hover {
                background-color: #229954;
            }
            QPushButton#closeBtn {
                background-color: #f39c12;
            }
            QPushButton#closeBtn:hover {
                background-color: #d68910;
            }
            QPushButton#quitBtn {
                background-color: #e74c3c;
            }
            QPushButton#quitBtn:hover {
                background-color: #c0392b;
            }
            QPushButton#cancelBtn {
                background-color: #7f8c8d;
            }
            QPushButton#cancelBtn:hover {
                background-color: #95a5a6;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Close Options")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Choose how to close Forge:")
        desc.setStyleSheet("font-weight: normal; font-size: 11px; color: #aaa;")
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)
        
        # Buttons row 1 - main options
        btn_layout1 = QHBoxLayout()
        btn_layout1.setSpacing(8)
        
        # Hide to Tray - keeps AI running in background
        hide_btn = QPushButton("Hide")
        hide_btn.setObjectName("hideBtn")
        hide_btn.setToolTip("Hide window - AI keeps running in tray")
        hide_btn.clicked.connect(lambda: (dialog.accept(), self._hide_to_tray()))
        btn_layout1.addWidget(hide_btn)
        
        # Close Window - closes just this Quick Chat
        close_btn = QPushButton("Close")
        close_btn.setObjectName("closeBtn")
        close_btn.setToolTip("Close this window only")
        close_btn.clicked.connect(lambda: (dialog.accept(), self._close_overlay()))
        btn_layout1.addWidget(close_btn)
        
        # Quit Forge - full exit
        quit_btn = QPushButton("Quit All")
        quit_btn.setObjectName("quitBtn")
        quit_btn.setToolTip("Quit Forge completely")
        quit_btn.clicked.connect(lambda: (dialog.accept(), self._quit_app()))
        btn_layout1.addWidget(quit_btn)
        
        layout.addLayout(btn_layout1)
        
        # Buttons row 2 - cancel
        btn_layout2 = QHBoxLayout()
        btn_layout2.setSpacing(8)
        
        btn_layout2.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancelBtn")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout2.addWidget(cancel_btn)
        btn_layout2.addStretch()
        
        layout.addLayout(btn_layout2)
        
        dialog.exec_()
    
    def _hide_to_tray(self):
        """Hide the window but keep AI running in system tray."""
        self.hide()
        self.set_status("Hidden to tray - AI still running")
        
        dialog.exec_()
    
    def _open_gui(self):
        """Open the full GUI (keeps Quick Chat open)."""
        self.open_gui_requested.emit()
        # Don't hide Quick Chat - keep it open
        # self.hide()
    
    def _close_overlay(self):
        """Close the overlay (goes to tray)."""
        self.close_requested.emit()
        self.hide()
    
    def _get_main_window(self):
        """Get reference to the main window if available."""
        try:
            from PyQt5.QtWidgets import QApplication
            for widget in QApplication.topLevelWidgets():
                if widget.__class__.__name__ == 'EnhancedMainWindow':
                    return widget
        except Exception:
            pass
        return None
    
    def send_to_other_windows(self, message: str):
        """Send a message to all other Quick Chat windows."""
        QuickCommandOverlay.broadcast_message(self._instance_id, message)
    
    def get_instance_count(self) -> int:
        """Get number of active Quick Chat instances."""
        return len(QuickCommandOverlay.get_all_instances())
    
    def _new_chat(self):
        """Clear the chat and start fresh."""
        if self._is_responding:
            return
        self.response_area.clear()
        self.response_area.setPlaceholderText("Chat history will appear here...")
        self.command_input.clear()
        self.history = []
        self.history_index = -1
        self.set_status("New chat")
    
    def _on_chat(self):
        """Handle chat message - use shared ChatSync for single engine."""
        command = self.command_input.text().strip()
        if not command:
            return
        
        # Check if ChatSync is already generating
        from .chat_sync import ChatSync
        chat_sync = ChatSync.instance()
        
        if chat_sync.is_generating:
            self.set_status("Still generating...")
            return
        
        if self._is_responding:
            return
        
        # Add to local history
        self.history.append(command)
        self.history_index = len(self.history)
        self.command_input.clear()
        
        # Start responding indicator
        self.start_responding()
        
        # Use ChatSync for shared generation (updates both UIs)
        chat_sync.generate_response(command, source="quick")
    
    def _on_submit(self):
        """Legacy submit handler - redirects to chat."""
        self._on_chat()
    
    def _toggle_voice_output_small(self):
        """Toggle voice output (removed button - kept for compatibility)."""
        self._auto_speak = not self._auto_speak
        
        status = "ON" if self._auto_speak else "OFF"
        self.set_status(f"Voice output: {status}")
        
        # Sync with main window if available
        main_window = self._get_main_window()
        if main_window:
            main_window.auto_speak = self._auto_speak
            if hasattr(main_window, 'voice_toggle_btn'):
                main_window.voice_toggle_btn.setChecked(self._auto_speak)
                if self._auto_speak:
                    main_window.voice_toggle_btn.setText("Voice ON")
                else:
                    main_window.voice_toggle_btn.setText("Voice OFF")
    
    def _speak_last_response(self):
        """Speak the last AI response."""
        if hasattr(self, '_last_response') and self._last_response:
            try:
                from forge_ai.voice import speak
                # Clean the text for speaking
                import re
                clean_text = re.sub(r'<[^>]+>', '', self._last_response)
                clean_text = clean_text.strip()[:500]
                if clean_text:
                    speak(clean_text)
                    self.set_status("Speaking...")
            except Exception as e:
                self.set_status(f"TTS error: {e}")
        else:
            self.set_status("No response to speak")
    
    def _toggle_voice_input(self):
        """Toggle voice input - now called from context menu."""
        # Start voice recognition
        try:
            self.set_status("Listening...")
            
            # Try to start voice recognition
            if hasattr(self, '_voice_thread') and self._voice_thread:
                return
            
            import threading
            self._voice_thread = threading.Thread(target=self._do_voice_input, daemon=True)
            self._voice_thread.start()
        except Exception as e:
            self.set_status(f"Voice error: {e}")
    
    def _open_avatar_controls(self):
        """Open avatar control menu with quick gestures."""
        try:
            from PyQt5.QtWidgets import QMenu
            from PyQt5.QtCore import QPoint
            
            menu = QMenu(self)
            menu.setStyleSheet("""
                QMenu {
                    background-color: #2c2c2c;
                    border: 1px solid #555;
                    padding: 5px;
                }
                QMenu::item {
                    color: #ddd;
                    padding: 5px 20px;
                }
                QMenu::item:selected {
                    background-color: #9b59b6;
                }
            """)
            
            # Add gesture actions
            gestures = [
                ("Wave", "wave hello"),
                ("Thumbs Up", "give me a thumbs up"),
                ("Thumbs Down", "give me a thumbs down"),
                ("Salute", "salute"),
                ("Shrug", "shrug your shoulders"),
                ("Stop", "make a stop gesture"),
                ("Point", "point at the screen"),
                ("Clap", "clap your hands"),
                ("Facepalm", "do a facepalm"),
                ("Raise Hand", "raise your hand"),
                ("Flex", "flex your muscles"),
            ]
            
            for emoji_text, command in gestures:
                action = menu.addAction(emoji_text)
                action.triggered.connect(lambda checked, cmd=command: self._send_avatar_command(cmd))
            
            menu.addSeparator()
            
            # Add "Open Avatar Tab" option
            open_tab_action = menu.addAction("Open Avatar Tab")
            open_tab_action.triggered.connect(self._open_avatar_tab)
            
            # Show menu at cursor position
            menu.exec_(QCursor.pos())
            
        except Exception as e:
            self.set_status(f"Avatar menu error: {e}")
    
    def _run_avatar(self):
        """Run/launch the avatar window."""
        try:
            # Try to get the main window
            main_window = self._get_main_window()
            
            if main_window:
                # Method 1: Try show_overlay_btn (set by avatar_display.py on main window)
                if hasattr(main_window, 'show_overlay_btn') and main_window.show_overlay_btn:
                    if not main_window.show_overlay_btn.isChecked():
                        main_window.show_overlay_btn.click()
                        self.set_status("Avatar started")
                    else:
                        main_window.show_overlay_btn.click()
                        self.set_status("Avatar stopped")
                    return
                
                # Method 2: Try avatar_controller directly
                if hasattr(main_window, 'avatar_controller') and main_window.avatar_controller:
                    if main_window.avatar_controller.isVisible():
                        main_window.avatar_controller.hide()
                        self.set_status("Avatar hidden")
                    else:
                        main_window.avatar_controller.show()
                        self.set_status("Avatar shown")
                    return
                
                # Method 3: Try desktop_pet overlay
                if hasattr(main_window, 'desktop_pet') and main_window.desktop_pet:
                    if main_window.desktop_pet.isVisible():
                        main_window.desktop_pet.hide()
                        self.set_status("Avatar hidden")
                    else:
                        main_window.desktop_pet.show()
                        self.set_status("Avatar shown")
                    return
                
                # Method 4: Try via avatar module
                try:
                    from ..avatar import get_avatar
                    avatar = get_avatar()
                    if avatar and hasattr(avatar, 'overlay'):
                        if avatar.overlay and avatar.overlay.isVisible():
                            avatar.overlay.hide()
                            self.set_status("Avatar hidden")
                        elif avatar.overlay:
                            avatar.overlay.show()
                            self.set_status("Avatar shown")
                        else:
                            # Need to create overlay
                            from ..avatar.desktop_pet import DesktopPetOverlay
                            avatar.overlay = DesktopPetOverlay()
                            avatar.overlay.show()
                            self.set_status("Avatar started")
                        return
                except Exception as e:
                    print(f"Avatar module error: {e}")
                
                # Method 5: Open avatar tab as last resort
                self._open_avatar_tab()
                self.set_status("Opened avatar tab")
            else:
                # No main window - try to launch standalone avatar
                self.set_status("Opening avatar...")
                try:
                    from ..avatar.desktop_pet import DesktopPetOverlay
                    if not hasattr(self, '_avatar_window') or not self._avatar_window:
                        self._avatar_window = DesktopPetOverlay()
                    if self._avatar_window.isVisible():
                        self._avatar_window.hide()
                        self.set_status("Avatar hidden")
                    else:
                        self._avatar_window.show()
                        self.set_status("Avatar launched")
                except Exception as avatar_err:
                    self.set_status(f"Avatar error: {avatar_err}")
                    # Fallback to opening full GUI with avatar tab
                    self._open_gui()
                    
        except Exception as e:
            self.set_status(f"Run avatar error: {e}")
    
    def _send_avatar_command(self, command: str):
        """Send an avatar control command to the AI."""
        self.command_input.setText(command)
        self._on_chat()
    
    def _open_avatar_tab(self):
        """Open the main GUI avatar tab."""
        try:
            main_window = self._get_main_window()
            if main_window:
                # Show the main window first
                main_window.show()
                main_window.activateWindow()
                
                # Try using the sidebar switch method
                if hasattr(main_window, '_switch_to_tab'):
                    main_window._switch_to_tab('avatar')
                    self.set_status("Opened Avatar tab")
                    return
                
                # Try using content_stack directly
                if hasattr(main_window, 'content_stack') and hasattr(main_window, '_tab_indices'):
                    if 'avatar' in main_window._tab_indices:
                        idx = main_window._tab_indices['avatar']
                        main_window.content_stack.setCurrentIndex(idx)
                        self.set_status("Opened Avatar tab")
                        return
                
                # Legacy: Try tab_widget
                if hasattr(main_window, 'tab_widget'):
                    for i in range(main_window.tab_widget.count()):
                        if "avatar" in main_window.tab_widget.tabText(i).lower():
                            main_window.tab_widget.setCurrentIndex(i)
                            self.set_status("Opened Avatar tab")
                            return
                
                self.set_status("Avatar tab not found")
            else:
                # No main window, just open GUI
                self._open_gui()
                self.set_status("Opening GUI...")
        except Exception as e:
            self.set_status(f"Error: {e}")
    
    def _reset_avatar_position(self):
        """Reset avatar overlay position to center of primary screen."""
        try:
            from PyQt5.QtWidgets import QApplication
            
            # Get primary screen geometry
            screen = QApplication.primaryScreen()
            if screen:
                geo = screen.availableGeometry()
                center_x = geo.x() + (geo.width() // 2) - 150
                center_y = geo.y() + (geo.height() // 2) - 150
            else:
                center_x, center_y = 400, 300
            
            # Try to move via main window's avatar overlays
            main_window = self._get_main_window()
            moved = False
            
            if main_window:
                # Try 2D overlay
                if hasattr(main_window, '_overlay') and main_window._overlay:
                    main_window._overlay.move(center_x, center_y)
                    moved = True
                
                # Try 3D overlay
                if hasattr(main_window, '_overlay_3d') and main_window._overlay_3d:
                    main_window._overlay_3d.move(center_x, center_y)
                    moved = True
            
            # Also try standalone avatar window
            if hasattr(self, '_avatar_window') and self._avatar_window:
                self._avatar_window.move(center_x, center_y)
                moved = True
            
            # Save the new position
            try:
                from ..avatar.persistence import save_position, get_persistence
                save_position(center_x, center_y)
                
                # Clear per-avatar positions
                persistence = get_persistence()
                settings = persistence.load()
                settings.per_avatar_positions.clear()
                persistence.save(settings)
            except Exception as e:
                print(f"Could not save position: {e}")
            
            if moved:
                self.set_status(f"Avatar moved to ({center_x}, {center_y})")
            else:
                self.set_status("Position reset - run avatar to see it")
                
        except Exception as e:
            self.set_status(f"Reset error: {e}")
    
    def _toggle_voice(self):
        """Legacy toggle - redirects to voice input."""
        self._toggle_voice_input()
    
    def _do_voice_input(self):
        """Background voice recognition."""
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            
            text = recognizer.recognize_google(audio)
            
            # Update UI from main thread
            from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
            QMetaObject.invokeMethod(
                self.command_input, "setText",
                Qt.QueuedConnection, Q_ARG(str, text)
            )
            QMetaObject.invokeMethod(
                self, "_voice_done",
                Qt.QueuedConnection
            )
            
        except Exception as e:
            from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
            QMetaObject.invokeMethod(
                self, "_voice_error",
                Qt.QueuedConnection, Q_ARG(str, str(e))
            )
    
    @pyqtSlot()
    def _voice_done(self):
        """Called when voice input completes."""
        self.set_status("Ready")
        self._voice_thread = None
        # Auto-send the voice input
        if self.command_input.text().strip():
            self._on_chat()
    
    @pyqtSlot(str)
    def _voice_error(self, error: str):
        """Called when voice input fails."""
        self.set_status(f"Voice: {error[:30]}")
        self._voice_thread = None
    
    @pyqtSlot(str)
    def show_response(self, text: str):
        """Show a response in the chat area."""
        # Stop the responding indicator
        self.stop_responding()
        
        # Store for speak last
        self._last_response = text
        
        # Use the AI's actual name instead of generic "AI"
        ai_name = getattr(self, 'ai_display_name', None) or getattr(self, 'model_name', 'AI')
        self.response_area.append(f"<div style='color: #3498db; margin-bottom: 8px;'><b>{ai_name}:</b> {text}</div>")
        # Scroll to bottom
        self.response_area.verticalScrollBar().setValue(
            self.response_area.verticalScrollBar().maximum()
        )
        
        # Auto-speak if enabled
        if getattr(self, '_auto_speak', False):
            self._speak_response(text)
    
    def _speak_response(self, text: str):
        """Speak a response using TTS."""
        try:
            import re
            # Clean the text for speaking
            clean_text = re.sub(r'<[^>]+>', '', text)
            clean_text = re.sub(r'```[\s\S]*?```', '', clean_text)  # Remove code blocks
            clean_text = clean_text.strip()[:500]
            
            if clean_text and not clean_text.startswith("⚠️"):
                from forge_ai.voice import speak
                speak(clean_text)
        except Exception as e:
            print(f"[QuickChat] TTS error: {e}")
    
    @pyqtSlot(str)
    def set_status(self, text: str):
        self.status_label.setText(text)
    
    def set_model_name(self, name: str):
        """Update the displayed model name."""
        self.ai_display_name = name  # Store for chat display
        if hasattr(self, 'title_label'):
            self.title_label.setText(name)
    
    def showEvent(self, event):
        super().showEvent(event)
        self.command_input.setFocus()
        self._center_on_screen()
    
    def _center_on_screen(self):
        """Center the overlay near the top of the screen where the main window is."""
        # Try to get the screen where the main window is located
        target_screen = None
        main_window = self._get_main_window()
        
        if main_window and main_window.isVisible():
            from PyQt5.QtGui import QGuiApplication
            target_screen = QGuiApplication.screenAt(main_window.geometry().center())
        
        # Fall back to primary screen if main window not available
        if target_screen is None:
            target_screen = QApplication.primaryScreen()
        
        screen_geo = target_screen.geometry()
        x = screen_geo.x() + (screen_geo.width() - self.width()) // 2
        y = screen_geo.y() + screen_geo.height() // 4  # Upper third of screen
        self.move(x, y)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            # ESC now opens GUI instead of just closing
            self._open_gui()
        elif event.key() == Qt.Key_Up:
            # Navigate history
            if self.history and self.history_index > 0:
                self.history_index -= 1
                self.command_input.setText(self.history[self.history_index])
        elif event.key() == Qt.Key_Down:
            if self.history and self.history_index < len(self.history) - 1:
                self.history_index += 1
                self.command_input.setText(self.history[self.history_index])
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Handle window close (Alt+F4) - hide to tray instead of quitting."""
        # Alt+F4 hides to tray, use X button or menu to quit
        self._close_overlay()
        event.ignore()  # Don't actually close the window
    
    # === Header drag handlers ===
    def _header_mouse_press(self, event):
        """Handle mouse press on header for dragging."""
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
        event.accept()
    
    def _header_mouse_move(self, event):
        """Handle mouse move on header for dragging."""
        if event.buttons() & Qt.LeftButton and self._drag_pos:
            self.move(event.globalPos() - self._drag_pos)
        event.accept()
    
    def _header_mouse_release(self, event):
        """Handle mouse release on header."""
        self._drag_pos = None
        event.accept()
    
    # === Resizing support ===
    def mousePressEvent(self, event):
        """Handle mouse press for dragging/resizing."""
        if event.button() == Qt.LeftButton:
            # Check if near edges for resizing
            edge = self._get_resize_edge(event.pos())
            if edge:
                self._resize_edge = edge
            else:
                # Drag window
                self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
        event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging/resizing."""
        if event.buttons() & Qt.LeftButton:
            if self._resize_edge:
                self._do_resize(event.globalPos())
            elif self._drag_pos:
                self.move(event.globalPos() - self._drag_pos)
        else:
            # Update cursor based on position
            edge = self._get_resize_edge(event.pos())
            if edge in ('left', 'right'):
                self.setCursor(Qt.SizeHorCursor)
            elif edge in ('top', 'bottom'):
                self.setCursor(Qt.SizeVerCursor)
            elif edge in ('topleft', 'bottomright'):
                self.setCursor(Qt.SizeFDiagCursor)
            elif edge in ('topright', 'bottomleft'):
                self.setCursor(Qt.SizeBDiagCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        event.accept()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self._drag_pos = None
        self._resize_edge = None
        self.setCursor(Qt.ArrowCursor)
        event.accept()
    
    def _get_resize_edge(self, pos):
        """Determine which edge the cursor is near."""
        margin = 8
        rect = self.rect()
        
        left = pos.x() < margin
        right = pos.x() > rect.width() - margin
        top = pos.y() < margin
        bottom = pos.y() > rect.height() - margin
        
        if top and left:
            return 'topleft'
        elif top and right:
            return 'topright'
        elif bottom and left:
            return 'bottomleft'
        elif bottom and right:
            return 'bottomright'
        elif left:
            return 'left'
        elif right:
            return 'right'
        elif top:
            return 'top'
        elif bottom:
            return 'bottom'
        return None
    
    def _do_resize(self, global_pos):
        """Perform resize based on edge being dragged."""
        geo = self.geometry()
        
        if 'left' in self._resize_edge:
            new_width = geo.right() - global_pos.x()
            if new_width >= self._min_width:
                geo.setLeft(global_pos.x())
        if 'right' in self._resize_edge:
            new_width = global_pos.x() - geo.left()
            if new_width >= self._min_width:
                geo.setRight(global_pos.x())
        if 'top' in self._resize_edge:
            new_height = geo.bottom() - global_pos.y()
            if new_height >= self._min_height:
                geo.setTop(global_pos.y())
        if 'bottom' in self._resize_edge:
            new_height = global_pos.y() - geo.top()
            if new_height >= self._min_height:
                geo.setBottom(global_pos.y())
        
        self.setGeometry(geo)


class ForgeSystemTray(QObject):
    """
    System tray integration for Forge.
    
    Keeps the AI running in background even when GUI is closed.
    """
    
    show_gui_requested = pyqtSignal()
    
    # Disable Windows notification balloons (set to True to re-enable)
    ENABLE_NOTIFICATIONS = False
    
    def __init__(self, app, main_window=None):
        super().__init__()
        self.app = app
        self.main_window = main_window
        self.is_recording = False
        self.recording_process = None
        self.recording_path = None
        
        # Model info - get from main window if available
        if main_window and hasattr(main_window, 'current_model_name') and main_window.current_model_name:
            self.current_model = main_window.current_model_name
        else:
            self.current_model = get_current_model_name()
        
        # Create system tray icon
        self.tray_icon = QSystemTrayIcon(self.app)
        self.tray_icon.setIcon(self._create_icon())
        self.tray_icon.setToolTip(f"{self.current_model} - Running in background")
        
        # Create menu
        self.menu = QMenu()
        self._build_menu()
        self.tray_icon.setContextMenu(self.menu)
        
        # Double-click to show overlay
        self.tray_icon.activated.connect(self._on_tray_activated)
        
        # Quick command overlay
        self.overlay = QuickCommandOverlay()
        self.overlay.command_submitted.connect(self._on_command)
        self.overlay.close_requested.connect(self._on_overlay_closed)
        self.overlay.open_gui_requested.connect(self._show_main_window)  # ESC opens GUI
        
        # Connect overlay to ChatSync for shared generation
        from .chat_sync import ChatSync
        chat_sync = ChatSync.instance()
        chat_sync.set_quick_chat(self.overlay)
        chat_sync.generation_finished.connect(self._on_chat_sync_finished)
        chat_sync.generation_stopped.connect(self._on_chat_sync_stopped)
        
        # Command processor
        self.processor = CommandProcessor()
        
        # Global hotkey timer (check for hotkey)
        self.hotkey_timer = QTimer()
        self.hotkey_timer.timeout.connect(self._check_hotkey)
        
        # Voice commander
        self.voice_commander = None
        self.voice_enabled = False
        
        # Global ESC key detection (platform specific)
        self._setup_global_hotkeys()
        
        # Show tray icon
        self.tray_icon.show()
        
        # Override showMessage to respect notification setting
        self._original_showMessage = self.tray_icon.showMessage
        self.tray_icon.showMessage = self._filtered_showMessage
    
    def _filtered_showMessage(self, title: str, message: str, icon_type=None, duration: int = 3000):
        """Show a notification balloon only if enabled."""
        if not self.ENABLE_NOTIFICATIONS:
            return
        if icon_type is None:
            icon_type = QSystemTrayIcon.Information
        self._original_showMessage(title, message, icon_type, duration)
    
    def _show_notification(self, title: str, message: str, icon_type=None, duration: int = 3000):
        """Show a notification balloon if enabled."""
        if not self.ENABLE_NOTIFICATIONS:
            return
        if icon_type is None:
            icon_type = QSystemTrayIcon.Information
        self._original_showMessage(title, message, icon_type, duration)
    
    def _setup_global_hotkeys(self):
        """Setup global hotkeys for showing GUI."""
        # Note: True global hotkeys require platform-specific libraries
        # For now, use a polling approach or rely on platform features
        # The overlay ESC key is handled in QuickCommandOverlay.keyPressEvent
        pass
    
    def _create_icon(self):
        """Load icon from file or create a simple one."""
        # Try to load the custom icon
        icon_paths = [
            Path(__file__).parent / "icons" / "forge.ico",
            Path(__file__).parent / "icons" / "forge_32.png",
            Path(CONFIG.get("data_dir", "data")) / "icons" / "forge.ico",
        ]
        
        for icon_path in icon_paths:
            if icon_path.exists():
                return QIcon(str(icon_path))
        
        # Fallback: create a simple icon
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw a simple circle with E
        painter.setBrush(QColor("#1e3a5f"))
        painter.setPen(QColor("#3498db"))
        painter.drawEllipse(2, 2, 28, 28)
        
        # Draw E
        painter.setBrush(QColor("#3498db"))
        painter.setPen(Qt.NoPen)
        painter.drawRect(8, 8, 4, 16)  # Vertical
        painter.drawRect(8, 8, 14, 3)  # Top
        painter.drawRect(8, 14, 10, 3)  # Middle
        painter.drawRect(8, 21, 14, 3)  # Bottom
        
        painter.end()
        
        return QIcon(pixmap)
    
    def _build_menu(self):
        """Build the tray menu."""
        # Load hotkey settings
        hotkeys = self._load_hotkey_settings()
        
        # === Header with model info and X close button ===
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(8, 4, 4, 4)
        
        # Model label on left
        model_label = QLabel(f"Model: {self.current_model}")
        model_label.setStyleSheet("color: #888; font-size: 11px;")
        header_layout.addWidget(model_label)
        self.model_label = model_label
        
        header_layout.addStretch()
        
        # X close button on right
        close_btn = QPushButton("⊗")
        close_btn.setFixedSize(20, 20)
        close_btn.setToolTip("Hide GUI (use Quick Chat to quit)")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #888;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e67e22;
                color: white;
                border-radius: 10px;
            }
        """)
        close_btn.clicked.connect(self._exit_app)
        header_layout.addWidget(close_btn)
        
        # Add header as widget action
        header_action = QWidgetAction(self)
        header_action.setDefaultWidget(header_widget)
        self.menu.addAction(header_action)
        
        # Keep model_action reference for backward compatibility
        model_action = QAction(f"Model: {self.current_model}", self)
        model_action.setVisible(False)  # Hidden, using custom widget instead
        self.model_action = model_action
        
        self.menu.addSeparator()
        
        # Quick command
        action_command = QAction(f"Quick Command ({hotkeys['command']})", self)
        action_command.triggered.connect(self.show_overlay)
        self.menu.addAction(action_command)
        
        # Open GUI
        action_gui = QAction("Open Full Interface", self)
        action_gui.triggered.connect(self._show_main_window)
        self.menu.addAction(action_gui)
        
        self.menu.addSeparator()
        
        # === Chat & Voice Quick Access ===
        action_chat = QAction("Open Chat", self)
        action_chat.triggered.connect(self._open_chat_tab)
        self.menu.addAction(action_chat)
        
        action_new_chat = QAction("New Chat", self)
        action_new_chat.setToolTip("Start a fresh conversation")
        action_new_chat.triggered.connect(self._start_new_chat)
        self.menu.addAction(action_new_chat)
        
        action_voice_input = QAction("Voice Input", self)
        action_voice_input.setToolTip("Open voice input in chat")
        action_voice_input.triggered.connect(self._open_voice_input)
        self.menu.addAction(action_voice_input)
        
        self.menu.addSeparator()
        
        # Help button - direct access
        action_help = QAction(f"Help ({hotkeys['help']})", self)
        action_help.triggered.connect(self._show_help)
        self.menu.addAction(action_help)
        
        self.menu.addSeparator()
        
        # Quick actions submenu
        quick_menu = self.menu.addMenu("Quick Actions")
        
        action_screenshot = QAction("Screenshot", self)
        action_screenshot.triggered.connect(lambda: self._execute_action("screenshot"))
        quick_menu.addAction(action_screenshot)
        
        action_screen_look = QAction("Analyze Screen", self)
        action_screen_look.triggered.connect(lambda: self._execute_action("vision"))
        quick_menu.addAction(action_screen_look)
        
        action_record = QAction("Start Recording", self)
        action_record.triggered.connect(self._toggle_recording)
        self.action_record = action_record
        quick_menu.addAction(action_record)
        
        quick_menu.addSeparator()
        
        action_image = QAction("Generate Image...", self)
        action_image.triggered.connect(lambda: self._prompt_action("generate image of"))
        quick_menu.addAction(action_image)
        
        action_video = QAction("Generate Video...", self)
        action_video.triggered.connect(lambda: self._prompt_action("generate video of"))
        quick_menu.addAction(action_video)
        
        self.menu.addSeparator()
        
        # Files submenu (help moved out)
        files_menu = self.menu.addMenu("Folders")
        
        action_docs = QAction("Documentation", self)
        action_docs.triggered.connect(lambda: self._open_folder("docs"))
        files_menu.addAction(action_docs)
        
        action_outputs = QAction("Outputs", self)
        action_outputs.triggered.connect(lambda: self._open_folder("outputs"))
        files_menu.addAction(action_outputs)
        
        action_models = QAction("Models", self)
        action_models.triggered.connect(lambda: self._open_folder("models"))
        files_menu.addAction(action_models)
        
        action_data = QAction("Data", self)
        action_data.triggered.connect(lambda: self._open_folder("data"))
        files_menu.addAction(action_data)
        
        self.menu.addSeparator()
        
        # Voice toggle
        self.action_voice = QAction(f"Enable Voice ({hotkeys['voice']})", self)
        self.action_voice.setCheckable(True)
        self.action_voice.triggered.connect(self._toggle_voice)
        self.menu.addAction(self.action_voice)
        
        self.menu.addSeparator()
        
        # Status
        self.status_action = QAction("Status: Ready", self)
        self.status_action.setEnabled(False)
        self.menu.addAction(self.status_action)
        
        self.menu.addSeparator()
        
        # Kill other instances
        action_kill = QAction("Kill Other Instances", self)
        action_kill.setToolTip("Kill other Forge processes to free memory")
        action_kill.triggered.connect(self._kill_other_instances)
        self.menu.addAction(action_kill)
        
        # Exit
        action_exit = QAction("Exit Forge", self)
        action_exit.triggered.connect(self._exit_app)
        self.menu.addAction(action_exit)
    
    def _kill_other_instances(self):
        """Kill other Forge instances."""
        result = kill_other_forge_ai_instances()
        killed = result.get("killed", 0)
        error = result.get("error")
        
        if error:
            self.tray_icon.showMessage(
                "Kill Instances",
                f"Killed {killed} instance(s).\nErrors: {error}",
                QSystemTrayIcon.Warning,
                3000
            )
        elif killed > 0:
            self.tray_icon.showMessage(
                "Kill Instances",
                f"Successfully killed {killed} other Forge instance(s).",
                QSystemTrayIcon.Information,
                3000
            )
        else:
            self.tray_icon.showMessage(
                "Kill Instances",
                "No other Forge instances found.",
                QSystemTrayIcon.Information,
                2000
            )
    
    def update_model_name(self, model_name: str):
        """Update the displayed model name."""
        self.current_model = model_name
        self.tray_icon.setToolTip(f"{model_name} - Running in background")
        # Update menu header label
        if hasattr(self, 'model_label'):
            self.model_label.setText(f"Model: {model_name}")
        # Update overlay title
        if hasattr(self, 'overlay') and hasattr(self.overlay, 'set_model_name'):
            self.overlay.set_model_name(model_name)
        # Update hidden action for compatibility
        if hasattr(self, 'model_action'):
            self.model_action.setText(f"Model: {model_name}")
    
    def _on_tray_activated(self, reason):
        """Handle tray icon activation.
        
        Single click: Show the main GUI window
        Double click: Show the main GUI window
        Middle click: Show quick command overlay
        """
        if reason == QSystemTrayIcon.DoubleClick:
            self._show_main_window()
        elif reason == QSystemTrayIcon.Trigger:
            # Single click now shows GUI (more intuitive)
            self._show_main_window()
        elif reason == QSystemTrayIcon.MiddleClick:
            # Middle click shows overlay
            self.show_overlay()
    
    def show_overlay(self):
        """Show the quick command overlay (Quick Chat)."""
        self.overlay.show()
        self.overlay.activateWindow()
        self.overlay.command_input.setFocus()
    
    # Alias for easier access
    def show_quick_command(self):
        """Alias for show_overlay - opens the Quick Chat window."""
        self.show_overlay()
    
    # Another alias 
    def show_mini_chat(self):
        """Alias for show_overlay - opens the Quick Chat window."""
        self.show_overlay()
    
    def hide_overlay(self):
        """Hide the overlay."""
        self.overlay.hide()
    
    def _on_overlay_closed(self):
        """Handle overlay close."""
        pass
    
    def _on_chat_sync_finished(self, response: str):
        """Handle when shared ChatSync finishes generating."""
        if self.overlay:
            self.overlay.stop_responding()
            self.overlay.set_status("Ready")
    
    def _on_chat_sync_stopped(self):
        """Handle when shared ChatSync generation is stopped."""
        if self.overlay:
            self.overlay.stop_responding()
            self.overlay.set_status("Stopped")
    
    def _on_command(self, command: str):
        """Process a command from the overlay - use shared ChatSync."""
        self.overlay.set_status("Thinking...")
        
        # Check if this is a quick action command
        command_lower = command.lower().strip()
        quick_action = self.processor._detect_quick_action(command_lower)
        
        if quick_action:
            # It's a quick action (screenshot, open folder, etc.) - handle locally
            result = quick_action
            self.overlay.show_response(result.get("response", "Done"))
            self.overlay.set_status("Ready")
            self._execute_action(result.get("action", ""), result.get("params", {}))
        # Chat messages are now handled directly in _on_chat via ChatSync
        # So we don't need the else branch anymore
    
    def _process_chat_in_mini(self, message: str):
        """Process a chat message - delegates to ChatSync for shared engine.
        
        NOTE: This is a legacy method. New code should use _on_chat() which
        directly calls ChatSync.generate_response().
        """
        # Use ChatSync for shared generation
        from .chat_sync import ChatSync
        chat_sync = ChatSync.instance()
        
        if chat_sync.is_generating:
            self.overlay.set_status("Still generating...")
            return
        
        # ChatSync handles everything: shows response in both UIs, syncs history
        chat_sync.generate_response(message, source="quick")
    
    def _stop_generation(self):
        """Stop current AI generation via ChatSync."""
        from .chat_sync import ChatSync
        chat_sync = ChatSync.instance()
        chat_sync.stop_generation()
    
    def _execute_action(self, action: str, params: Dict[str, Any] = None):
        """Execute an action."""
        params = params or {}
        
        if action == "open_gui":
            self._show_main_window()
            self.hide_overlay()
        
        elif action == "screenshot":
            self._take_screenshot()
        
        elif action == "record_screen":
            self._toggle_recording()
        
        elif action == "generate_image":
            prompt = params.get("prompt", "")
            self._generate_image(prompt)
        
        elif action == "generate_video":
            prompt = params.get("prompt", "")
            self._generate_video(prompt)
        
        elif action == "train":
            self._handle_training(params)
        
        elif action == "file":
            self._handle_file(params)
        
        elif action == "avatar":
            self._handle_avatar(params)
        
        elif action == "vision":
            self._analyze_screen()
        
        elif action == "settings":
            self._show_main_window()
            # Switch to settings tab if main window exists
            if self.main_window and hasattr(self.main_window, 'tab_widget'):
                # Find settings tab index
                for i in range(self.main_window.tab_widget.count()):
                    if "settings" in self.main_window.tab_widget.tabText(i).lower():
                        self.main_window.tab_widget.setCurrentIndex(i)
                        break
        
        elif action == "help":
            self._show_help()
        
        elif action == "open_folder":
            folder = params.get("folder", "outputs")
            self._open_folder(folder)
        
        elif action == "exit":
            self._exit_app()
    
    def _show_main_window(self):
        """Show the main GUI window on the same screen as Quick Chat."""
        self.show_gui_requested.emit()
        if self.main_window:
            # Position main window on the same monitor as Quick Chat
            if hasattr(self, 'overlay') and self.overlay and self.overlay.isVisible():
                from PyQt5.QtGui import QGuiApplication
                # Get the screen where Quick Chat is
                overlay_center = self.overlay.frameGeometry().center()
                screen = QGuiApplication.screenAt(overlay_center)
                if screen:
                    screen_geo = screen.geometry()
                    # Center main window on that screen
                    win_size = self.main_window.size()
                    x = screen_geo.x() + (screen_geo.width() - win_size.width()) // 2
                    y = screen_geo.y() + (screen_geo.height() - win_size.height()) // 2
                    self.main_window.move(x, y)
            
            self.main_window.show()
            self.main_window.activateWindow()
            # Keep Quick Chat open too - always visible unless explicitly closed
            if hasattr(self, 'overlay') and self.overlay:
                self.overlay.show()
    
    def _open_chat_tab(self):
        """Open the GUI and switch to the chat tab."""
        self._show_main_window()
        if self.main_window:
            # Try to switch to chat tab using the sidebar
            if hasattr(self.main_window, '_switch_to_tab'):
                self.main_window._switch_to_tab('chat')
            elif hasattr(self.main_window, 'content_stack'):
                # Find chat in the content stack (usually index 0)
                self.main_window.content_stack.setCurrentIndex(0)
            # Focus the chat input
            if hasattr(self.main_window, 'chat_input'):
                self.main_window.chat_input.setFocus()
    
    def _start_new_chat(self):
        """Start a new chat conversation."""
        self._show_main_window()
        if self.main_window:
            # Import the new chat function
            try:
                from .tabs.chat_tab import _new_chat
                _new_chat(self.main_window)
            except Exception as e:
                # Fallback: just clear the chat
                if hasattr(self.main_window, 'chat_display'):
                    self.main_window.chat_display.clear()
                if hasattr(self.main_window, 'chat_messages'):
                    self.main_window.chat_messages = []
            
            # Switch to chat tab
            self._open_chat_tab()
            
            self.tray_icon.showMessage(
                "New Chat",
                "Started a new conversation.",
                QSystemTrayIcon.Information,
                2000
            )
    
    def _open_voice_input(self):
        """Open voice input mode in the chat."""
        self._show_main_window()
        if self.main_window:
            # Switch to chat tab first
            self._open_chat_tab()
            
            # Try to activate microphone/voice input
            if hasattr(self.main_window, 'btn_mic') and hasattr(self.main_window.btn_mic, 'click'):
                # Simulate click on microphone button
                self.main_window.btn_mic.click()
                self.tray_icon.showMessage(
                    "Voice Input",
                    "Voice input activated. Speak now...",
                    QSystemTrayIcon.Information,
                    2000
                )
            elif hasattr(self.main_window, '_toggle_microphone'):
                self.main_window._toggle_microphone()
                self.tray_icon.showMessage(
                    "Voice Input",
                    "Voice input activated. Speak now...",
                    QSystemTrayIcon.Information,
                    2000
                )
            else:
                self.tray_icon.showMessage(
                    "Voice Input",
                    "Voice input is ready. Use the microphone button in chat.",
                    QSystemTrayIcon.Information,
                    2000
                )

    def _take_screenshot(self):
        """Take a screenshot."""
        try:
            from ..tools.vision import capture_screen
            result = capture_screen()
            if result.get("success"):
                path = result.get("path", "")
                self.tray_icon.showMessage(
                    "Screenshot Captured",
                    f"Saved to: {path}",
                    QSystemTrayIcon.Information,
                    3000
                )
                # Open in explorer
                from .tabs.output_helpers import open_file_in_explorer
                open_file_in_explorer(path)
        except Exception as e:
            self.tray_icon.showMessage("Error", str(e), QSystemTrayIcon.Warning, 3000)
    
    def _generate_image(self, prompt: str):
        """Generate an image."""
        if not prompt:
            self.show_overlay()
            self.overlay.command_input.setText("generate image of ")
            self.overlay.command_input.setFocus()
            return
        
        self.tray_icon.showMessage("Generating Image", f"Creating: {prompt[:50]}...", QSystemTrayIcon.Information, 2000)
        
        # Try local SD first, fall back to placeholder with warning
        try:
            from .tabs.image_tab import get_provider
            
            # Try local Stable Diffusion first
            provider = get_provider('local')
            if provider:
                if not provider.is_loaded:
                    provider.load()
                
                if provider.is_loaded:
                    result = provider.generate(prompt)
                    if result.get("success"):
                        path = result.get("path", "")
                        self.tray_icon.showMessage("Image Generated", f"Saved to: {path}", QSystemTrayIcon.Information, 3000)
                        from .tabs.output_helpers import open_file_in_explorer
                        open_file_in_explorer(path)
                        return
            
            # Fall back to placeholder with warning
            self.tray_icon.showMessage(
                "Using Placeholder", 
                "Stable Diffusion not available. Install: pip install diffusers transformers accelerate",
                QSystemTrayIcon.Warning, 
                4000
            )
            
            provider = get_provider('placeholder')
            if provider:
                provider.load()
                result = provider.generate(prompt)
                if result.get("success"):
                    path = result.get("path", "")
                    self.tray_icon.showMessage("Placeholder Generated", f"(Not real image) Saved to: {path}", QSystemTrayIcon.Information, 3000)
                    from .tabs.output_helpers import open_file_in_explorer
                    open_file_in_explorer(path)
        except Exception as e:
            self.tray_icon.showMessage("Error", str(e), QSystemTrayIcon.Warning, 3000)
    
    def _generate_video(self, prompt: str):
        """Generate a video."""
        if not prompt:
            self.show_overlay()
            self.overlay.command_input.setText("generate video of ")
            self.overlay.command_input.setFocus()
            return
        
        self.tray_icon.showMessage("Video Generation", "Video generation requires the full GUI.", QSystemTrayIcon.Information, 2000)
        self._show_main_window()
    
    def _handle_training(self, params: Dict):
        """Handle training requests."""
        data = params.get("data", "")
        if params.get("start"):
            self.tray_icon.showMessage("Training", "Opening training interface...", QSystemTrayIcon.Information, 2000)
            self._show_main_window()
        elif data:
            self.tray_icon.showMessage("Training Data", f"Adding to training: {data[:50]}...", QSystemTrayIcon.Information, 2000)
            # Add to training data file
            try:
                from pathlib import Path
                data_dir = Path(CONFIG.get("data_dir", "data"))
                training_file = data_dir / "user_training.txt"
                with open(training_file, "a", encoding="utf-8") as f:
                    f.write(f"\n{data}\n")
                self.tray_icon.showMessage("Training Data Added", f"Saved to {training_file.name}", QSystemTrayIcon.Information, 2000)
            except Exception as e:
                self.tray_icon.showMessage("Error", f"Failed to save: {e}", QSystemTrayIcon.Warning, 3000)
    
    def _handle_file(self, params: Dict):
        """Handle file operations."""
        path = params.get("path", "")
        operation = params.get("operation", "open")
        
        if path:
            from .tabs.output_helpers import open_in_default_viewer
            open_in_default_viewer(path)
    
    def _handle_avatar(self, params: Dict):
        """Handle avatar commands."""
        show = params.get("show", True)
        try:
            # Try to actually control avatar
            if self.main_window and hasattr(self.main_window, 'avatar_controller'):
                if show:
                    self.main_window.avatar_controller.show()
                else:
                    self.main_window.avatar_controller.hide()
                self.tray_icon.showMessage(
                    f"Avatar ({self.current_model})", 
                    "Avatar connected!" if show else "Avatar disconnected.",
                    QSystemTrayIcon.Information, 
                    2000
                )
            else:
                self.tray_icon.showMessage(
                    f"Avatar ({self.current_model})", 
                    "Open full GUI to use avatar.",
                    QSystemTrayIcon.Information, 
                    2000
                )
                if show:
                    self._show_main_window()
        except Exception as e:
            self.tray_icon.showMessage("Avatar Error", str(e), QSystemTrayIcon.Warning, 3000)
    
    def _analyze_screen(self):
        """Capture and analyze the screen with AI."""
        self.tray_icon.showMessage(
            f"Vision ({self.current_model})", 
            "Capturing and analyzing screen...", 
            QSystemTrayIcon.Information, 
            2000
        )
        try:
            from ..tools.vision import capture_screen, get_screen_vision
            
            # Capture screenshot
            result = capture_screen()
            if result.get("success"):
                path = result.get("path", "")
                
                # Try to analyze with vision
                try:
                    vision = get_screen_vision()
                    analysis = vision.see(describe=True, detect_text=True)
                    
                    text_found = analysis.get("text", "")[:200]
                    if text_found:
                        self.tray_icon.showMessage(
                            f"Screen Analysis ({self.current_model})",
                            f"Text detected:\n{text_found}...",
                            QSystemTrayIcon.Information,
                            5000
                        )
                    else:
                        self.tray_icon.showMessage(
                            f"Screen Captured ({self.current_model})",
                            f"Saved to: {path}",
                            QSystemTrayIcon.Information,
                            3000
                        )
                except:
                    self.tray_icon.showMessage(
                        f"Screen Captured ({self.current_model})",
                        f"Saved to: {path}",
                        QSystemTrayIcon.Information,
                        3000
                    )
                
                # Open in explorer
                from .tabs.output_helpers import open_file_in_explorer
                open_file_in_explorer(path)
        except Exception as e:
            self.tray_icon.showMessage("Vision Error", str(e), QSystemTrayIcon.Warning, 3000)
    
    def _prompt_action(self, prefix: str):
        """Show overlay with a pre-filled command."""
        self.show_overlay()
        self.overlay.command_input.setText(f"{prefix} ")
        self.overlay.command_input.setFocus()
        # Move cursor to end
        self.overlay.command_input.setCursorPosition(len(self.overlay.command_input.text()))
    
    def _toggle_voice(self, enabled: bool):
        """Toggle voice activation."""
        self.voice_enabled = enabled
        if enabled:
            success = self._start_voice_listener()
            if not success:
                self.action_voice.setChecked(False)
                self.voice_enabled = False
        else:
            self._stop_voice_listener()
    
    def _start_voice_listener(self) -> bool:
        """Start the voice listener."""
        try:
            from ..voice.listener import VoiceCommander, check_voice_available
            
            # Check if voice is available
            status = check_voice_available()
            if not status["available"]:
                error = status.get("error", "Voice not available")
                self.tray_icon.showMessage(
                    "Voice Error",
                    f"{error}\n\nInstall with: pip install SpeechRecognition pyaudio",
                    QSystemTrayIcon.Warning,
                    5000
                )
                return False
            
            # Create voice commander
            self.voice_commander = VoiceCommander(command_callback=self._on_voice_command)
            self.voice_commander.current_model = self.current_model
            self.voice_commander.set_notification_callback(
                lambda t, m: self.tray_icon.showMessage(t, m, QSystemTrayIcon.Information, 3000)
            )
            self.voice_commander.set_status_callback(self.set_status)
            
            if self.voice_commander.start():
                self.set_status(f"Voice Listening ({self.current_model})")
                return True
            else:
                return False
                
        except ImportError as e:
            self.tray_icon.showMessage(
                "Voice Not Available",
                "Install with: pip install SpeechRecognition pyaudio",
                QSystemTrayIcon.Warning,
                5000
            )
            return False
        except Exception as e:
            self.tray_icon.showMessage("Voice Error", str(e), QSystemTrayIcon.Warning, 3000)
            return False
    
    def _on_voice_command(self, text: str):
        """Handle voice command."""
        self.tray_icon.showMessage(
            f"Voice Command ({self.current_model})",
            f"Heard: {text}",
            QSystemTrayIcon.Information,
            2000
        )
        # Process like any other command
        self._on_command(text)
    
    def _stop_voice_listener(self):
        """Stop the voice listener."""
        if self.voice_commander:
            self.voice_commander.stop()
            self.voice_commander = None
        self.set_status("Ready")
        self.tray_icon.showMessage(
            "Voice Disabled",
            "Voice activation stopped.",
            QSystemTrayIcon.Information,
            2000
        )
    
    def _toggle_recording(self):
        """Toggle screen recording."""
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()
    
    def _start_recording(self):
        """Start screen recording."""
        try:
            # Create output path
            output_dir = Path(CONFIG.get("outputs_dir", "outputs")) / "videos"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            self.recording_path = str(output_dir / f"recording_{timestamp}.mp4")
            
            # Try ffmpeg for recording
            if sys.platform == 'win32':
                # Windows: use ffmpeg with gdigrab
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'gdigrab',
                    '-framerate', '30',
                    '-i', 'desktop',
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    self.recording_path
                ]
            elif sys.platform == 'darwin':
                # macOS: use ffmpeg with avfoundation
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'avfoundation',
                    '-framerate', '30',
                    '-i', '1:',
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    self.recording_path
                ]
            else:
                # Linux: use ffmpeg with x11grab
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'x11grab',
                    '-framerate', '30',
                    '-i', ':0.0',
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    self.recording_path
                ]
            
            self.recording_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            self.is_recording = True
            self.action_record.setText("[Recording] Stop")
            self.tray_icon.showMessage(
                f"Recording Started ({self.current_model})",
                "Screen recording in progress...\nClick Stop Recording when done.",
                QSystemTrayIcon.Information,
                3000
            )
            self.set_status("Recording...")
            
        except FileNotFoundError:
            self.tray_icon.showMessage(
                "Recording Error",
                "ffmpeg not found. Install ffmpeg to record screen.\n"
                "Windows: winget install ffmpeg\n"
                "Mac: brew install ffmpeg\n"
                "Linux: sudo apt install ffmpeg",
                QSystemTrayIcon.Warning,
                5000
            )
        except Exception as e:
            self.tray_icon.showMessage("Recording Error", str(e), QSystemTrayIcon.Warning, 3000)
    
    def _stop_recording(self):
        """Stop screen recording."""
        if self.recording_process:
            try:
                # Send 'q' to ffmpeg to stop gracefully
                self.recording_process.stdin.write(b'q')
                self.recording_process.stdin.flush()
                self.recording_process.wait(timeout=5)
            except:
                self.recording_process.terminate()
            
            self.recording_process = None
        
        self.is_recording = False
        self.action_record.setText("Start Recording")
        self.set_status("Ready")
        
        if self.recording_path and Path(self.recording_path).exists():
            self.tray_icon.showMessage(
                f"Recording Saved ({self.current_model})",
                f"Saved to: {self.recording_path}",
                QSystemTrayIcon.Information,
                3000
            )
            # Open in explorer
            from .tabs.output_helpers import open_file_in_explorer
            open_file_in_explorer(self.recording_path)
        else:
            self.tray_icon.showMessage(
                "Recording Stopped",
                "Recording may not have saved properly.",
                QSystemTrayIcon.Warning,
                3000
            )
    
    def _check_hotkey(self):
        """Check for global hotkey (fallback if pyqtkeybind not available)."""
        # This is a placeholder - actual global hotkey needs platform-specific code
        pass
    
    def _load_hotkey_settings(self) -> dict:
        """Load keyboard shortcut settings."""
        defaults = {
            "command": "Ctrl+Shift+E",
            "help": "Ctrl+Shift+H",
            "voice": "Ctrl+Shift+V"
        }
        try:
            import json
            settings_path = Path(CONFIG.get("info_dir", "information")) / "gui_settings.json"
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    return {
                        "command": settings.get("hotkey_command", defaults["command"]),
                        "help": settings.get("hotkey_help", defaults["help"]),
                        "voice": settings.get("hotkey_voice", defaults["voice"])
                    }
        except:
            pass
        return defaults
    
    def _show_help(self):
        """Show help in a separate window."""
        help_path = Path(CONFIG.get("info_dir", "information")) / "help.txt"
        
        # Create a separate help window
        if not hasattr(self, 'help_window') or self.help_window is None:
            self.help_window = HelpWindow()
        
        # Load help content
        if help_path.exists():
            try:
                with open(help_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.help_window.set_content(content)
            except Exception as e:
                self.help_window.set_content(f"Error loading help: {e}")
        else:
            self.help_window.set_content(
                "Help file not found.\n\n"
                "Check the docs/ folder for detailed guides:\n"
                "- GUI_GUIDE.md\n"
                "- HOW_TO_TRAIN.md\n"
                "- MODULE_GUIDE.md"
            )
        
        self.help_window.show()
        self.help_window.activateWindow()
    
    def _open_folder(self, folder_name: str):
        """Open a project folder in the file explorer."""
        folder_map = {
            "outputs": CONFIG.get("outputs_dir", "outputs"),
            "models": CONFIG.get("models_dir", "models"),
            "data": CONFIG.get("data_dir", "data"),
            "docs": "docs"
        }
        
        folder_path = Path(folder_map.get(folder_name, folder_name))
        if not folder_path.is_absolute():
            folder_path = Path(CONFIG.get("root_dir", ".")) / folder_path
        
        from .tabs.output_helpers import open_folder
        
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
        
        open_folder(folder_path)
        
        self.tray_icon.showMessage(
            "Folder Opened",
            f"Opened: {folder_name}",
            QSystemTrayIcon.Information,
            2000
        )
    
    def _exit_app(self):
        """Hide system tray menu - use Quick Chat to quit ForgeAI."""
        # Just hide the main GUI window, don't exit
        if self.main_window:
            self.main_window.hide()
        # Note: Use Quick Chat's "Quit Forge" to actually exit the application
    
    def set_status(self, text: str):
        """Update the status in the tray menu."""
        self.status_action.setText(f"Status: {text}")
    
    def show_notification(self, title: str, message: str, icon_type=None):
        """Show a system notification."""
        if icon_type is None:
            icon_type = QSystemTrayIcon.Information
        self.tray_icon.showMessage(title, message, icon_type, 3000)


def create_system_tray(app, main_window=None):
    """Create and return the system tray instance."""
    if not HAS_PYQT:
        return None
    
    if not QSystemTrayIcon.isSystemTrayAvailable():
        print("System tray not available on this system")
        return None
    
    return ForgeSystemTray(app, main_window)
