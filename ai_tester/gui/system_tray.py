# type: ignore
# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalMemberAccess=false
"""
System Tray Integration - Keep Enigma running in the background.

Features:
  - System tray icon with menu
  - Quick command overlay (hotkey activated)
  - Voice activation support ("Hey AI Tester")
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
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QObject, QThread  # type: ignore[import]
    from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QKeySequence, QFont  # type: ignore[import]
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
        model_name = CONFIG.get("default_model", "small_ai_tester")
        return model_name
    except:
        return "Unknown Model"


def kill_other_ai_tester_instances() -> dict:
    """Kill other AI Tester processes to free up resources.
    
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
                if 'ai_tester' in line.lower() or 'run.py' in line.lower():
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
                if ('ai_tester' in line.lower() or 'run.py' in line.lower()) and 'python' in line.lower():
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
    """Process natural language commands in background."""
    
    result_ready = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._engine = None
        self._engine_loaded = False
        self.tool_interface = None
        self.model_name = "Not loaded"
        # Don't load engine immediately - lazy load on first use
    
    def _ensure_engine_loaded(self):
        """Lazy-load the engine only when actually needed."""
        if self._engine_loaded:
            return
        self._engine_loaded = True
        self._load_engine()
    
    @property
    def engine(self):
        """Lazy-load engine on first access."""
        if not self._engine_loaded:
            self._ensure_engine_loaded()
        return self._engine
    
    @engine.setter
    def engine(self, value):
        self._engine = value
    
    def _load_engine(self):
        """Load the inference engine using the saved model from settings."""
        try:
            from ..core.tool_interface import ToolInterface
            from ..core.model_registry import ModelRegistry
            
            # Get the model name from saved settings
            self.model_name = get_current_model_name()
            
            # Load the model through the registry (respects HuggingFace models)
            registry = ModelRegistry()
            if self.model_name in registry.registry.get("models", {}):
                model, config = registry.load_model(self.model_name)
                
                # Check if it's a HuggingFace model (already has generate/chat methods)
                is_hf = config.get("source") == "huggingface"
                
                if is_hf:
                    # HuggingFace models are ready to use directly
                    self.engine = model  # HuggingFaceModel wrapper
                else:
                    # For Enigma models, wrap in AITesterEngine
                    from ..core.inference import AITesterEngine
                    import torch
                    self.engine = AITesterEngine.__new__(AITesterEngine)
                    self.engine.model = model
                    self.engine.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.engine.use_half = False
                    self.engine.enable_tools = False
                    self.engine._is_huggingface = False
            else:
                # Fallback: create default engine (untrained)
                from ..core.inference import AITesterEngine
                self.engine = AITesterEngine()
                
            self.tool_interface = ToolInterface()
        except Exception as e:
            print(f"Note: AI engine not loaded yet: {e}")
            self.model_name = "Not loaded"
            self.engine = None
    
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
        if any(kw in command for kw in ["open gui", "show gui", "open window", "show window", "open enigma"]):
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
        self.setWindowTitle("AI Tester - Help")
        self.setWindowFlags(Qt.Window)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title = QLabel("AI Tester Help")
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
    """
    
    command_submitted = pyqtSignal(str)
    close_requested = pyqtSignal()
    open_gui_requested = pyqtSignal()  # New signal for ESC to open GUI
    stop_requested = pyqtSignal()  # Signal to stop generation
    
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
        
        super().__init__(parent)
        
        # Load settings for always-on-top
        always_on_top = self._load_mini_chat_settings().get("mini_chat_always_on_top", True)
        
        # Use Window flag for proper window controls including Alt+F4
        flags = Qt.Window | Qt.FramelessWindowHint
        if always_on_top:
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        
        self.setAttribute(Qt.WA_TranslucentBackground)
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
        
        # Header with title bar controls (like a window)
        header_layout = QHBoxLayout()
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
        new_chat_btn.setToolTip("New Chat")
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
        
        # Minimize button (opens GUI)
        min_btn = QPushButton("_")
        min_btn.setFixedSize(24, 24)
        min_btn.setToolTip("Open Full GUI")
        min_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #555;
                border-radius: 4px;
                color: #888;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #444;
                color: #3498db;
            }
        """)
        min_btn.clicked.connect(self._open_gui)
        header_layout.addWidget(min_btn)
        
        # Close button (like window X)
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setToolTip("Close (to tray)")
        close_btn.setStyleSheet("""
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
        close_btn.clicked.connect(self._close_overlay)
        header_layout.addWidget(close_btn)
        
        frame_layout.addLayout(header_layout)
        
        # Chat history area (always visible, above input like main chat)
        self.response_area = QTextEdit()
        self.response_area.setReadOnly(True)
        self.response_area.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        self.response_area.setPlaceholderText("Chat history will appear here...")
        self.response_area.setMinimumHeight(150)
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
        self.command_input.setPlaceholderText("Chat here... (Esc = Open GUI)")
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
        
        # Voice button
        self.voice_btn = QPushButton("Voice")
        self.voice_btn.setFixedSize(50, 40)
        self.voice_btn.setToolTip("Voice input (hold to speak)")
        self.voice_btn.setCheckable(True)
        self.voice_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                border: none;
                border-radius: 8px;
                color: white;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #666;
            }
            QPushButton:checked {
                background-color: #e74c3c;
            }
            QPushButton:checked:hover {
                background-color: #c0392b;
            }
        """)
        self.voice_btn.clicked.connect(self._toggle_voice)
        input_layout.addWidget(self.voice_btn)
        
        frame_layout.addLayout(input_layout)
        
        # Hint row
        hint_layout = QHBoxLayout()
        hint = QLabel("Enter=Send | Esc=Open GUI | Up/Down=History")
        hint.setStyleSheet("color: #555; font-size: 10px;")
        hint_layout.addWidget(hint)
        hint_layout.addStretch()
        frame_layout.addLayout(hint_layout)
        
        layout.addWidget(self.frame)
        
        # Initial size (larger to show chat history)
        self.setMinimumSize(self._min_width, self._min_height)
        self.resize(500, 350)
        
        # Setup responding indicator animation timer
        self._responding_dots = 0
        self._responding_timer = QTimer(self)
        self._responding_timer.timeout.connect(self._update_responding_indicator)
    
    def _load_mini_chat_settings(self):
        """Load mini chat settings from gui_settings.json."""
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
        """Show the responding indicator (single line in chat, animated status)."""
        self._is_responding = True
        self._stop_requested = False
        self._responding_dots = 0
        # Add single thinking indicator inline
        ai_name = getattr(self, 'ai_display_name', None) or getattr(self, 'model_name', 'AI')
        self.response_area.append(
            f'<div id="thinking" style="color: #f9e2af; padding: 4px;"><i>{ai_name} is thinking...</i></div>'
        )
        self.response_area.verticalScrollBar().setValue(
            self.response_area.verticalScrollBar().maximum()
        )
        self._responding_timer.start(400)  # Animate status bar only
        self.chat_btn.hide()  # Hide send button
        self.stop_btn.show()  # Show stop button
        self.set_status("Thinking...")
    
    def stop_responding(self):
        """Remove the thinking indicator from chat."""
        self._is_responding = False
        self._responding_timer.stop()
        self._remove_thinking_indicator()
        self.stop_btn.hide()  # Hide stop button
        self.chat_btn.show()  # Show send button
        self.set_status("Ready")
    
    def _stop_generation(self):
        """Stop the current AI generation."""
        self._stop_requested = True
        self.stop_requested.emit()
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
    
    def _open_gui(self):
        """Open the full GUI (keeps mini chat open)."""
        self.open_gui_requested.emit()
        # Don't hide mini chat - keep it open
        # self.hide()
    
    def _close_overlay(self):
        """Close the overlay (goes to tray)."""
        self.close_requested.emit()
        self.hide()
    
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
        """Handle chat message - process in mini chat, don't go to main GUI."""
        command = self.command_input.text().strip()
        if command and not self._is_responding:
            self.history.append(command)
            self.history_index = len(self.history)
            
            # Show user message with customized name
            user_name = getattr(self, 'user_display_name', 'You')
            self.response_area.append(
                f"<div style='color: #9b59b6; margin: 4px 0;'><b>{user_name}:</b> {command}</div>"
            )
            self.command_input.clear()
            
            # Start responding indicator
            self.start_responding()
            
            # Scroll to bottom
            self.response_area.verticalScrollBar().setValue(
                self.response_area.verticalScrollBar().maximum()
            )
            
            # Emit signal for processing
            self.command_submitted.emit(command)
    
    def _on_submit(self):
        """Legacy submit handler - redirects to chat."""
        self._on_chat()
    
    def _toggle_voice(self):
        """Toggle voice input."""
        is_listening = self.voice_btn.isChecked()
        
        if is_listening:
            self.voice_btn.setToolTip("Listening... (click to stop)")
            self.set_status("Listening...")
            
            # Try to start voice recognition
            try:
                # Signal to start voice input
                if hasattr(self, '_voice_thread') and self._voice_thread:
                    return
                
                import threading
                self._voice_thread = threading.Thread(target=self._do_voice_input, daemon=True)
                self._voice_thread.start()
            except Exception as e:
                self.voice_btn.setChecked(False)
                self.set_status(f"Voice error: {e}")
        else:
            self.voice_btn.setToolTip("Voice input (click to speak)")
            self.set_status("Ready")
            self._voice_thread = None
    
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
        self.voice_btn.setChecked(False)
        self.set_status("Ready")
        self._voice_thread = None
        # Auto-send the voice input
        if self.command_input.text().strip():
            self._on_chat()
    
    @pyqtSlot(str)
    def _voice_error(self, error: str):
        """Called when voice input fails."""
        self.voice_btn.setChecked(False)
        self.set_status(f"Voice: {error[:30]}")
        self._voice_thread = None
    
    @pyqtSlot(str)
    def show_response(self, text: str):
        """Show a response in the chat area."""
        # Stop the responding indicator
        self.stop_responding()
        
        # Use the AI's actual name instead of generic "AI"
        ai_name = getattr(self, 'ai_display_name', None) or getattr(self, 'model_name', 'AI')
        self.response_area.append(f"<div style='color: #3498db; margin-bottom: 8px;'><b>{ai_name}:</b> {text}</div>")
        # Scroll to bottom
        self.response_area.verticalScrollBar().setValue(
            self.response_area.verticalScrollBar().maximum()
        )
    
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
        """Center the overlay near the top of the screen."""
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = screen.height() // 4  # Upper third of screen
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
        """Handle window close (Alt+F4)."""
        # Alt+F4 will trigger this - exit the entire app
        from PyQt5.QtWidgets import QApplication
        QApplication.quit()
    
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


class EnigmaSystemTray(QObject):
    """
    System tray integration for Enigma.
    
    Keeps the AI running in background even when GUI is closed.
    """
    
    show_gui_requested = pyqtSignal()
    
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
            Path(__file__).parent / "icons" / "enigma.ico",
            Path(__file__).parent / "icons" / "enigma_32.png",
            Path(CONFIG.get("data_dir", "data")) / "icons" / "enigma.ico",
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
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(20, 20)
        close_btn.setToolTip("Exit Enigma completely")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #888;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e74c3c;
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
        action_kill.setToolTip("Kill other Enigma processes to free memory")
        action_kill.triggered.connect(self._kill_other_instances)
        self.menu.addAction(action_kill)
        
        # Exit
        action_exit = QAction("Exit Enigma", self)
        action_exit.triggered.connect(self._exit_app)
        self.menu.addAction(action_exit)
    
    def _kill_other_instances(self):
        """Kill other Enigma instances."""
        result = kill_other_ai_tester_instances()
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
                f"Successfully killed {killed} other Enigma instance(s).",
                QSystemTrayIcon.Information,
                3000
            )
        else:
            self.tray_icon.showMessage(
                "Kill Instances",
                "No other Enigma instances found.",
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
        """Show the quick command overlay (mini chat)."""
        self.overlay.show()
        self.overlay.activateWindow()
        self.overlay.command_input.setFocus()
    
    # Alias for easier access
    def show_quick_command(self):
        """Alias for show_overlay - opens the mini chat window."""
        self.show_overlay()
    
    # Another alias 
    def show_mini_chat(self):
        """Alias for show_overlay - opens the mini chat window."""
        self.show_overlay()
    
    def hide_overlay(self):
        """Hide the overlay."""
        self.overlay.hide()
    
    def _on_overlay_closed(self):
        """Handle overlay close."""
        pass
    
    def _on_command(self, command: str):
        """Process a command from the overlay - chat directly in mini chat."""
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
        else:
            # It's a chat message - process directly in mini chat
            self._process_chat_in_mini(command)
    
    def _process_chat_in_mini(self, message: str):
        """Process a chat message directly in the mini chat and sync to main chat."""
        import threading
        import time
        
        # Store reference to check for stop
        self._current_generation_thread = None
        
        def generate_response():
            try:
                response = None
                
                # Check if stop was requested before starting
                if self.overlay._stop_requested:
                    return
                
                # Try to use the main window's engine if available
                if self.main_window and hasattr(self.main_window, 'engine') and self.main_window.engine:
                    engine = self.main_window.engine
                    response = engine.generate(
                        message,
                        max_gen=200,
                        temperature=0.8
                    )
                else:
                    # Try to create/get engine directly
                    try:
                        from enigma.core.inference import AITesterEngine
                        engine = AITesterEngine()
                        response = engine.generate(
                            message,
                            max_gen=200,
                            temperature=0.8
                        )
                    except Exception as e:
                        response = f"[No AI loaded - open full GUI to load a model]\n\nYour message: {message}"
                
                # Check if stop was requested during generation
                if self.overlay._stop_requested:
                    return
                
                # Sync to main window's chat history
                if self.main_window:
                    self._sync_to_main_chat(message, response)
                
                # Update UI from main thread
                from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(
                    self.overlay, "show_response",
                    Qt.QueuedConnection, Q_ARG(str, response)
                )
                QMetaObject.invokeMethod(
                    self.overlay, "set_status",
                    Qt.QueuedConnection, Q_ARG(str, "Ready")
                )
                
            except Exception as e:
                # Don't show error if stop was requested
                if self.overlay._stop_requested:
                    return
                from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(
                    self.overlay, "show_response",
                    Qt.QueuedConnection, Q_ARG(str, f"<span style='color: #e74c3c;'>Error: {e}</span>")
                )
                QMetaObject.invokeMethod(
                    self.overlay, "set_status",
                    Qt.QueuedConnection, Q_ARG(str, "Error")
                )
        
        # Run in background thread
        thread = threading.Thread(target=generate_response, daemon=True)
        thread.start()
    
    def _sync_to_main_chat(self, user_message: str, ai_response: str):
        """Sync mini chat messages to main window's chat history."""
        import time
        try:
            # Add to chat_messages list
            if hasattr(self.main_window, 'chat_messages'):
                self.main_window.chat_messages.append({
                    "role": "user",
                    "text": user_message,
                    "ts": time.time(),
                    "source": "mini_chat"
                })
                self.main_window.chat_messages.append({
                    "role": "assistant", 
                    "text": ai_response,
                    "ts": time.time(),
                    "source": "mini_chat"
                })
            
            # Update chat display in main window
            if hasattr(self.main_window, 'chat_display'):
                from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                model_name = getattr(self.main_window, 'current_model_name', 'AI')
                
                # Add user message
                user_html = (
                    f'<div style="background-color: #313244; padding: 8px; margin: 4px 0; '
                    f'border-radius: 8px; border-left: 3px solid #9b59b6;">'
                    f'<b style="color: #9b59b6;">You (mini):</b> {user_message}</div>'
                )
                QMetaObject.invokeMethod(
                    self.main_window.chat_display, "append",
                    Qt.QueuedConnection, Q_ARG(str, user_html)
                )
                
                # Add AI response
                ai_html = (
                    f'<div style="background-color: #1e1e2e; padding: 8px; margin: 4px 0; '
                    f'border-radius: 8px; border-left: 3px solid #a6e3a1;">'
                    f'<b style="color: #a6e3a1;">{model_name}:</b> {ai_response}</div>'
                )
                QMetaObject.invokeMethod(
                    self.main_window.chat_display, "append",
                    Qt.QueuedConnection, Q_ARG(str, ai_html)
                )
        except Exception as e:
            print(f"Could not sync to main chat: {e}")
    
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
        """Show the main GUI window (keeps mini chat open)."""
        self.show_gui_requested.emit()
        if self.main_window:
            self.main_window.show()
            self.main_window.activateWindow()
            # Keep mini chat open too - always visible unless explicitly closed
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
            import subprocess
            import sys
            if sys.platform == 'win32':
                subprocess.run(['notepad', path])
            else:
                subprocess.run(['xdg-open', path])
    
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
        
        if folder_path.exists():
            if sys.platform == 'win32':
                os.startfile(str(folder_path))
            elif sys.platform == 'darwin':
                subprocess.run(['open', str(folder_path)])
            else:
                subprocess.run(['xdg-open', str(folder_path)])
            
            self.tray_icon.showMessage(
                "Folder Opened",
                f"Opened: {folder_name}",
                QSystemTrayIcon.Information,
                2000
            )
        else:
            folder_path.mkdir(parents=True, exist_ok=True)
            if sys.platform == 'win32':
                os.startfile(str(folder_path))
            elif sys.platform == 'darwin':
                subprocess.run(['open', str(folder_path)])
            else:
                subprocess.run(['xdg-open', str(folder_path)])
    
    def _exit_app(self):
        """Exit the application completely."""
        self.tray_icon.hide()
        self.app.quit()
    
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
    
    return EnigmaSystemTray(app, main_window)
