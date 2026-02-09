"""
================================================================================
               CHAPTER 5: THE CASTLE - YOUR COMMAND CENTER
================================================================================

    "Welcome to the Grand Hall, where all paths converge."

You have arrived at the heart of the kingdom! This is the MAIN WINDOW -
the visual throne room where you control everything in Enigma AI Engine.

WHY THIS FILE MATTERS:
    Every button you click, every tab you switch, every message you type
    in the GUI - it all happens here. This is the LARGEST file in Enigma AI Engine
    because it connects EVERYTHING together into one beautiful interface.

THE GRAND TOUR:
    ╔═══════════════════════════════════════════════════════════════╗
    ║  Enigma AI Engine - [Your AI's Name]                      [_][O][X]   ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  [Chat] [Image] [Code] [Video] [Audio] [3D] [Train] [+]      ║
    ║  ─────────────────────────────────────────────────────────── ║
    ║                                                               ║
    ║    Whatever tab you select appears here.                     ║
    ║    This is where the magic happens!                          ║
    ║                                                               ║
    ║    Each tab is its own mini-adventure:                       ║
    ║    • Chat Tab    = Talk to your AI                           ║
    ║    • Image Tab   = Create artwork                            ║
    ║    • Code Tab    = Write programs                            ║
    ║    • Video Tab   = Generate movies                           ║
    ║    • Audio Tab   = Text to speech                            ║
    ║    • 3D Tab      = Model generation                          ║
    ║    • Train Tab   = Teach your AI new things                  ║
    ║                                                               ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Status: Ready | Model: small | Theme: Dark                  ║
    ╚═══════════════════════════════════════════════════════════════╝

BEHIND THE SCENES (For Developers):
    This file uses PyQt5 to create the window. Key classes:
    
    • EnhancedMainWindow  - The castle itself (QMainWindow)
    • AIGenerationWorker  - Background thread for AI responses
                           (Keeps GUI responsive while AI thinks)

    The window loads all tabs dynamically from enigma_engine/gui/tabs/.
    Each tab handles its own UI and logic.

YOUR QUEST HERE:
    Want to add a new tab? Create a file in gui/tabs/, then register
    it in this file's tab loading section. Follow the existing pattern!

CONNECTED PATHS:
    All roads lead here:
        run.py --gui → THIS FILE → loads all tabs
    
    This file talks to:
        → inference.py (AI responses for chat)
        → memory/manager.py (save conversations)
        → avatar/controller.py (avatar features)
        → All tab files in gui/tabs/

SEE ALSO:
    • enigma_engine/gui/styles.py       - Theme CSS styles
    • enigma_engine/gui/theme_system.py - Theme management
    • data/gui_settings.json       - Saved GUI preferences

Module Organization (7380+ lines):
==================================
Lines 124-348:   AIGenerationWorker - Background AI thread (~225 lines)
Lines 349-1169:  GenerationPreviewPopup - Preview dialog (~821 lines)
Lines 1170-1556: SetupWizard - First-run model creation (~387 lines)
Lines 1557-7380: EnhancedMainWindow - Main application window (~5823 lines)
                 - Initialization & setup (1557-1900)
                 - Tab management (1900-2700)
                 - Model & engine handling (2700-3500)
                 - Menu & toolbar setup (3500-4500)
                 - Chat handling (4500-5200)
                 - Avatar & overlay integration (5200-5800)
                 - Settings & persistence (5800-6500)
                 - Utility methods (6500-7380)
"""
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

import time

from PyQt5.QtCore import QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

# Import GUI mode system
from .gui_modes import GUIMode, GUIModeManager
from .widgets.quick_actions import FeedbackButtons, QuickActionsBar


# === AI Generation Worker Thread ===
class AIGenerationWorker(QThread):
    """Background worker for AI generation to keep GUI responsive."""
    finished = pyqtSignal(str)  # Emits the response
    error = pyqtSignal(str)     # Emits error message
    thinking = pyqtSignal(str)  # Emits thinking/reasoning status
    stopped = pyqtSignal()      # Emits when stopped by user
    
    def __init__(self, engine, text, is_hf, history=None, system_prompt=None, custom_tokenizer=None, parent_window=None):
        super().__init__()
        self.engine = engine
        self.text = text
        self.is_hf = is_hf
        self.history = history
        self.system_prompt = system_prompt
        self.custom_tokenizer = custom_tokenizer
        self.parent_window = parent_window
        self._stop_requested = False
        self._start_time = None
    
    def stop(self):
        """Request the worker to stop."""
        self._stop_requested = True
        
    def run(self):
        try:
            import time
            self._start_time = time.time()
            
            # Emit initial thinking status
            self.thinking.emit("Analyzing your message...")
            
            # Log to terminal
            if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
                self.parent_window.log_terminal(f"NEW REQUEST: {self.text}", "info")
            
            if self._stop_requested:
                self.stopped.emit()
                return
            
            if self.is_hf:
                # HuggingFace model - show reasoning steps
                self.thinking.emit("Building conversation context...")
                if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
                    self.parent_window.log_terminal("Building conversation history...", "debug")
                time.sleep(0.1)
                
                if self._stop_requested:
                    self.stopped.emit()
                    return
                
                self.thinking.emit("Processing with language model...")
                if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
                    self.parent_window.log_terminal("Running inference on model...", "info")
                
                # HuggingFace model
                if hasattr(self.engine.model, 'chat') and not self.custom_tokenizer:
                    response = self.engine.model.chat(
                        self.text,
                        history=self.history if self.history else None,
                        system_prompt=self.system_prompt,
                        max_new_tokens=200,
                        temperature=0.7
                    )
                else:
                    self.thinking.emit("Tokenizing input...")
                    response = self.engine.model.generate(
                        self.text,
                        max_new_tokens=50,
                        temperature=0.8,
                        top_p=0.92,
                        top_k=50,
                        repetition_penalty=1.2,
                        do_sample=True,
                        custom_tokenizer=self.custom_tokenizer
                    )
                
                # Check if response is a tensor (model didn't decode output)
                if hasattr(response, 'shape') or 'tensor' in str(type(response)).lower():
                    self.thinking.emit("Decoding model output...")
                    # Try to decode the tensor
                    try:
                        import torch
                        if isinstance(response, torch.Tensor):
                            # Try to decode using the model's tokenizer
                            if hasattr(self.engine.model, 'tokenizer'):
                                response = self.engine.model.tokenizer.decode(
                                    response[0] if len(response.shape) > 1 else response,
                                    skip_special_tokens=True
                                )
                            elif self.custom_tokenizer:
                                response = self.custom_tokenizer.decode(
                                    response[0] if len(response.shape) > 1 else response,
                                    skip_special_tokens=True
                                )
                            else:
                                response = (
                                    "[Warning] Model returned raw tensor data. This usually means:\n"
                                    "• The model is not properly configured for text generation\n"
                                    "• Try a different model or check if it needs fine-tuning\n"
                                    "• Local Forge models need training first"
                                )
                    except Exception as decode_err:
                        response = f"[Warning] Could not decode model output: {decode_err}"
            else:
                # Local Forge model - use chat method with history for context
                self.thinking.emit("Building conversation context...")
                
                # Get recent history from parent window (limited to prevent overflow)
                chat_history = []
                if self.parent_window and hasattr(self.parent_window, 'chat_messages'):
                    # Get last 6 messages (3 exchanges) to fit in context
                    # Use -7:-1 to exclude the current message being processed but get 6 prior
                    recent = self.parent_window.chat_messages[-7:-1] if len(self.parent_window.chat_messages) > 1 else []
                    for msg in recent:
                        role = "user" if msg.get("role") == "user" else "assistant"
                        chat_history.append({"role": role, "content": msg.get("text", "")})
                
                if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
                    self.parent_window.log_terminal(f"Using {len(chat_history)} history messages for context", "debug")
                
                if self._stop_requested:
                    self.stopped.emit()
                    return
                
                self.thinking.emit("Running inference on local model...")
                if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
                    self.parent_window.log_terminal("Generating tokens...", "info")
                
                # Use chat() method which handles history truncation and context
                if hasattr(self.engine, 'chat') and chat_history:
                    response = self.engine.chat(
                        message=self.text,
                        history=chat_history,
                        system_prompt=self.system_prompt,
                        max_gen=100,
                        auto_truncate=True  # Prevent hallucinations!
                    )
                    formatted_prompt = self.text  # For cleanup code below
                else:
                    # Fallback to simple Q&A format
                    formatted_prompt = f"Q: {self.text}\nA:"
                    response = self.engine.generate(formatted_prompt, max_gen=100)
                
                if self._stop_requested:
                    self.stopped.emit()
                    return
                
                self.thinking.emit("Cleaning up response...")
                
                # Check if response is a tensor
                if hasattr(response, 'shape') or 'tensor' in str(type(response)).lower():
                    response = (
                        "[Warning] Model returned raw data instead of text.\n"
                        "This model may need more training. Go to the Train tab."
                    )
                else:
                    # Clean up response
                    if response.startswith(formatted_prompt):
                        response = response[len(formatted_prompt):].strip()
                    elif response.startswith(self.text):
                        response = response[len(self.text):].strip()
                        
                    if "\nQ:" in response:
                        response = response.split("\nQ:")[0].strip()
                    if "Q:" in response:
                        response = response.split("Q:")[0].strip()
                    if response.startswith("A:"):
                        response = response[2:].strip()
                    if response.startswith(":"):
                        response = response[1:].strip()
            
            if self._stop_requested:
                self.stopped.emit()
                return
            
            if not response:
                response = "(No response generated - model may need more training)"
            
            # Log completion
            if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
                self.parent_window.log_terminal(f"Generated {len(response)} characters", "success")
            
            # Validate response - detect garbage/code output
            garbage_indicators = [
                'torch.tensor', 'np.array', 'def test_', 'assert ', 'import torch',
                'class Test', 'self.setup', '.to(device)', 'cudnn.enabled',
                'torch.randn', 'torch.zeros', 'return Tensor', '# Convert',
                'dtype=torch.float', 'skip_special_tokens', "'cuda:0'", "'cuda:1'",
                '.to("cuda', 'tensor([[', 'Output:', '# Output:', 'tensors.shape',
                '.expand(', '```python', 'import torch', 'broadcasting dimension',
                'tensor(tensor(', '.size() ==', 'expanded_matrix'
            ]
            
            is_garbage = False
            for indicator in garbage_indicators:
                if indicator in response:
                    is_garbage = True
                    break
            
            # Also check if response is mostly code-like
            if not is_garbage:
                code_chars = response.count('(') + response.count(')') + response.count('[') + response.count(']') + response.count('=')
                if len(response) > 50 and code_chars > len(response) * 0.1:
                    is_garbage = True
            
            if is_garbage:
                response = (
                    "[Warning] The model generated code/technical text instead of a response.\n\n"
                    "This can happen with small models like Qwen2-0.5B when asked simple questions.\n"
                    "Try:\n"
                    "• Using a larger model (tinyllama_chat, phi2, or qwen2_1.5b_instruct)\n"
                    "• Being more specific in your question\n"
                    "• Training your own Forge model with conversational data"
                )
            
            elapsed = time.time() - self._start_time
            self.thinking.emit(f"Done in {elapsed:.1f}s")
            self.finished.emit(response)
        except Exception as e:
            if self._stop_requested:
                self.stopped.emit()
            else:
                self.error.emit(str(e))


# === Generation Result Preview Popup ===
class GenerationPreviewPopup(QDialog):
    """
    Popup window to preview generation results (images, videos, etc.)
    Shows immediately when content is generated without switching tabs.
    """
    
    def __init__(self, parent=None, result_path: str = "", result_type: str = "image", title: str = ""):
        super().__init__(parent)
        self.result_path = result_path
        self.result_type = result_type
        
        # Window setup - frameless, always on top, but movable
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(False)  # Non-modal so user can keep chatting
        
        self._setup_ui(title or f"{result_type.title()} Generated")
        self._position_window()
        
        # No auto-close - window stays until user closes it
        # User can click X button or click anywhere outside to close
        
        # Track dragging
        self._drag_pos = None
    
    def _setup_ui(self, title: str):
        """Set up the popup UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Main container with rounded corners
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background-color: #1e1e2e;
                border: 2px solid #89b4fa;
                border-radius: 12px;
            }
        """)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(8, 8, 8, 8)
        
        # Header with title and close button
        header = QHBoxLayout()
        
        title_label = QLabel(f"{title}")
        title_label.setStyleSheet("color: #a6e3a1; font-weight: bold; font-size: 12px; border: none;")
        header.addWidget(title_label)
        
        header.addStretch()
        
        close_btn = QPushButton("X")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8;
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f5c2e7;
            }
        """)
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)
        
        container_layout.addLayout(header)
        
        # Content area
        if self.result_type == "image" and self.result_path:
            # Image preview
            preview_label = QLabel()
            preview_label.setAlignment(Qt.AlignCenter)
            preview_label.setMinimumSize(400, 300)
            preview_label.setStyleSheet("border: 1px solid #45475a; border-radius: 8px;")
            
            pixmap = QPixmap(self.result_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                preview_label.setPixmap(scaled)
            else:
                preview_label.setText("Could not load image")
            
            container_layout.addWidget(preview_label)
        
        elif self.result_type == "animation" and self.result_path:
            # Animated GIF preview (dice rolls, etc.)
            from PyQt5.QtGui import QMovie
            preview_label = QLabel()
            preview_label.setAlignment(Qt.AlignCenter)
            preview_label.setMinimumSize(220, 220)
            preview_label.setStyleSheet("border: 1px solid #45475a; border-radius: 8px;")
            
            # Load and play GIF
            self._movie = QMovie(self.result_path)
            if self._movie.isValid():
                preview_label.setMovie(self._movie)
                self._movie.start()
            else:
                preview_label.setText("Animation Generated")
            
            container_layout.addWidget(preview_label)
            
        elif self.result_type == "video" and self.result_path:
            # Video preview (thumbnail + play button)
            preview_label = QLabel()
            preview_label.setAlignment(Qt.AlignCenter)
            preview_label.setMinimumSize(400, 300)
            preview_label.setStyleSheet("border: 1px solid #45475a; border-radius: 8px; color: #cdd6f4;")
            preview_label.setText(f"Video Generated\n\nClick 'Open' to play")
            container_layout.addWidget(preview_label)
            
        elif self.result_type == "audio" and self.result_path:
            preview_label = QLabel()
            preview_label.setAlignment(Qt.AlignCenter)
            preview_label.setMinimumSize(300, 100)
            preview_label.setStyleSheet("border: 1px solid #45475a; border-radius: 8px; color: #cdd6f4;")
            preview_label.setText(f"Audio Generated\n\nClick 'Open' to play")
            container_layout.addWidget(preview_label)
            
        elif self.result_type == "3d" and self.result_path:
            preview_label = QLabel()
            preview_label.setAlignment(Qt.AlignCenter)
            preview_label.setMinimumSize(300, 100)
            preview_label.setStyleSheet("border: 1px solid #45475a; border-radius: 8px; color: #cdd6f4;")
            preview_label.setText(f"3D Model Generated\n\nClick 'Open' to view")
            container_layout.addWidget(preview_label)
        else:
            # Generic text
            preview_label = QLabel(f"Generated: {self.result_path}")
            preview_label.setStyleSheet("color: #cdd6f4; padding: 20px; border: none;")
            preview_label.setWordWrap(True)
            container_layout.addWidget(preview_label)
        
        # Path display
        path_label = QLabel(f"Path: {self.result_path}")
        path_label.setStyleSheet("color: #bac2de; font-size: 12px; border: none;")
        path_label.setWordWrap(True)
        container_layout.addWidget(path_label)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        open_btn = QPushButton("Open File")
        open_btn.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #b4befe; }
        """)
        open_btn.clicked.connect(self._open_file)
        btn_layout.addWidget(open_btn)
        
        folder_btn = QPushButton("Open Folder")
        folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #b4befe; }
        """)
        folder_btn.clicked.connect(self._open_folder)
        btn_layout.addWidget(folder_btn)
        
        container_layout.addLayout(btn_layout)
        
        # Hint
        hint = QLabel("Click anywhere to close. Auto-closes in 15s")
        hint.setStyleSheet("color: #6c7086; font-size: 12px; border: none;")
        hint.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(hint)
        
        layout.addWidget(container)
        self.adjustSize()
    
    def _position_window(self):
        """Position popup in bottom-right corner of screen."""
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.width() - 20, screen.bottom() - self.height() - 20)
    
    def _open_file(self):
        """Open the generated file."""
        from .tabs.output_helpers import open_in_default_viewer
        path = Path(self.result_path)
        if path.exists():
            open_in_default_viewer(path)
        self.close()
    
    def _open_folder(self):
        """Open the containing folder."""
        from .tabs.output_helpers import open_file_in_explorer
        path = Path(self.result_path)
        if path.exists():
            open_file_in_explorer(path)
        self.close()
    
    def mousePressEvent(self, event):
        """Click anywhere to close, or start drag."""
        if event.button() == Qt.LeftButton:
            # Check if clicking on buttons (don't close)
            child = self.childAt(event.pos())
            if isinstance(child, QPushButton):
                return super().mousePressEvent(event)
            # Start drag
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
    
    def mouseMoveEvent(self, event):
        """Allow dragging the popup."""
        if self._drag_pos and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)
    
    def mouseReleaseEvent(self, event):
        """End drag or close on click."""
        if self._drag_pos:
            # If didn't move much, treat as click to close
            moved = (event.globalPos() - self.frameGeometry().topLeft() - self._drag_pos).manhattanLength()
            if moved < 5:
                self.close()
        self._drag_pos = None


# Import text formatting
try:
    from ..utils.text_formatting import TextFormatter
    HAVE_TEXT_FORMATTER = True
except ImportError:
    HAVE_TEXT_FORMATTER = False


# === THEME STYLESHEETS ===
# Theme: Dark (Catppuccin Mocha)
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QLabel {
    /* Enable text selection on labels */
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
}
QTextEdit, QPlainTextEdit, QLineEdit, QListWidget {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
}
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #b4befe;
}
QPushButton:pressed {
    background-color: #74c7ec;
}
QPushButton:disabled {
    background-color: #313244;
    color: #f38ba8;
    border: 2px dashed #f38ba8;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
}
QGroupBox::title {
    color: #89b4fa;
    subcontrol-origin: margin;
    left: 10px;
}
QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #313244;
    color: #cdd6f4;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QProgressBar {
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #a6e3a1;
}
QMenuBar {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QMenuBar::item:selected {
    background-color: #313244;
}
QMenu {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
}
QMenu::item:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QSpinBox, QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px;
}
QSlider::groove:horizontal {
    background: #45475a;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QScrollBar:vertical {
    background: #313244;
    width: 12px;
}
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 6px;
}
QLabel#header {
    font-size: 12px;
    font-weight: bold;
    color: #89b4fa;
}
QLabel {
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
}
/* Sidebar Navigation Styling */
QListWidget#sidebar {
    background-color: #11111b;
    border: none;
    border-right: 2px solid #1e1e2e;
    outline: none;
    font-size: 12px;
    font-weight: 500;
    padding: 8px 0;
}
QListWidget#sidebar::item {
    padding: 6px 12px;
    border-left: 3px solid transparent;
    margin: 1px 4px;
    border-radius: 4px;
    color: #a6adc8;
}
QListWidget#sidebar::item:selected {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #313244, stop:1 #1e1e2e);
    border-left: 3px solid #89b4fa;
    color: #cdd6f4;
    font-weight: bold;
}
QListWidget#sidebar::item:hover:!selected {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollArea > QWidget > QWidget {
    background-color: transparent;
}
/* Scrollbar in sidebar */
QListWidget#sidebar QScrollBar:vertical {
    background: #11111b;
    width: 8px;
    margin: 0;
}
QListWidget#sidebar QScrollBar::handle:vertical {
    background: #313244;
    border-radius: 4px;
    min-height: 30px;
}
QListWidget#sidebar QScrollBar::handle:vertical:hover {
    background: #45475a;
}
QListWidget#sidebar QScrollBar::add-line:vertical,
QListWidget#sidebar QScrollBar::sub-line:vertical {
    height: 0;
}
"""

LIGHT_STYLE = """
QMainWindow, QWidget {
    background-color: #eff1f5;
    color: #4c4f69;
}
QTextEdit, QPlainTextEdit, QLineEdit, QListWidget {
    background-color: #ffffff;
    color: #4c4f69;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 4px;
}
QPushButton {
    background-color: #1e66f5;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #7287fd;
}
QPushButton:pressed {
    background-color: #04a5e5;
}
QPushButton:disabled {
    background-color: #e6e9ef;
    color: #d20f39;
    border: 2px dashed #d20f39;
}
QGroupBox {
    border: 1px solid #ccd0da;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
}
QGroupBox::title {
    color: #1e66f5;
    subcontrol-origin: margin;
    left: 10px;
}
QTabWidget::pane {
    border: 1px solid #ccd0da;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #e6e9ef;
    color: #4c4f69;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #1e66f5;
    color: #ffffff;
}
QProgressBar {
    border: 1px solid #ccd0da;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #40a02b;
}
QMenuBar {
    background-color: #eff1f5;
    color: #4c4f69;
}
QMenuBar::item:selected {
    background-color: #e6e9ef;
}
QMenu {
    background-color: #ffffff;
    color: #4c4f69;
    border: 1px solid #ccd0da;
}
QMenu::item:selected {
    background-color: #1e66f5;
    color: #ffffff;
}
QSpinBox, QComboBox {
    background-color: #ffffff;
    color: #4c4f69;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 4px;
}
QSlider::groove:horizontal {
    background: #ccd0da;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #1e66f5;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QLabel#header {
    font-size: 12px;
    font-weight: bold;
    color: #1e66f5;
}
QLabel {
    selection-background-color: #1e66f5;
    selection-color: #ffffff;
}
/* Sidebar Navigation Styling - Light */
QListWidget#sidebar {
    background-color: #dce0e8;
    border: none;
    border-right: 2px solid #ccd0da;
    outline: none;
    font-size: 12px;
    font-weight: 500;
    padding: 8px 0;
}
QListWidget#sidebar::item {
    padding: 6px 12px;
    border-left: 3px solid transparent;
    margin: 1px 4px;
    border-radius: 4px;
    color: #5c5f77;
}
QListWidget#sidebar::item:selected {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #bcc0cc, stop:1 #dce0e8);
    border-left: 3px solid #1e66f5;
    color: #1e66f5;
    font-weight: bold;
}
QListWidget#sidebar::item:hover:!selected {
    background-color: #ccd0da;
    color: #4c4f69;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollArea > QWidget > QWidget {
    background-color: transparent;
}
"""

# Theme: Shadow (Very dark with purple accents)
SHADOW_STYLE = """
QMainWindow, QWidget {
    background-color: #0d0d0d;
    color: #b8b8b8;
}
QTextEdit, QPlainTextEdit, QLineEdit, QListWidget {
    background-color: #1a1a1a;
    color: #d0d0d0;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 4px;
}
QPushButton {
    background-color: #6b21a8;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #7c3aed;
}
QPushButton:pressed {
    background-color: #581c87;
}
QPushButton:disabled {
    background-color: #1a1a1a;
    color: #f43f5e;
    border: 2px dashed #f43f5e;
}
QGroupBox {
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
}
QGroupBox::title {
    color: #9333ea;
    subcontrol-origin: margin;
    left: 10px;
}
QTabWidget::pane {
    border: 1px solid #2a2a2a;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #1a1a1a;
    color: #b8b8b8;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #6b21a8;
    color: #ffffff;
}
QProgressBar {
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #9333ea;
}
QMenuBar {
    background-color: #0d0d0d;
    color: #b8b8b8;
}
QMenuBar::item:selected {
    background-color: #1a1a1a;
}
QMenu {
    background-color: #1a1a1a;
    color: #b8b8b8;
    border: 1px solid #2a2a2a;
}
QMenu::item:selected {
    background-color: #6b21a8;
    color: #ffffff;
}
QSpinBox, QComboBox {
    background-color: #1a1a1a;
    color: #d0d0d0;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 4px;
}
QSlider::groove:horizontal {
    background: #2a2a2a;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #9333ea;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QScrollBar:vertical {
    background: #1a1a1a;
    width: 12px;
}
QScrollBar::handle:vertical {
    background: #2a2a2a;
    border-radius: 6px;
}
QLabel#header {
    font-size: 12px;
    font-weight: bold;
    color: #9333ea;
}
QLabel {
    selection-background-color: #6b21a8;
    selection-color: #ffffff;
}
"""

# Theme: Midnight (Deep blue/black with cyan accents)
MIDNIGHT_STYLE = """
QMainWindow, QWidget {
    background-color: #030712;
    color: #e2e8f0;
}
QTextEdit, QPlainTextEdit, QLineEdit, QListWidget {
    background-color: #0f172a;
    color: #e2e8f0;
    border: 1px solid #1e293b;
    border-radius: 4px;
    padding: 4px;
}
QPushButton {
    background-color: #0891b2;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #22d3ee;
}
QPushButton:pressed {
    background-color: #0e7490;
}
QPushButton:disabled {
    background-color: #1e293b;
    color: #fb7185;
    border: 2px dashed #fb7185;
}
QGroupBox {
    border: 1px solid #1e293b;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
}
QGroupBox::title {
    color: #22d3ee;
    subcontrol-origin: margin;
    left: 10px;
}
QTabWidget::pane {
    border: 1px solid #1e293b;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #0f172a;
    color: #e2e8f0;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #0891b2;
    color: #ffffff;
}
QProgressBar {
    border: 1px solid #1e293b;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #22d3ee;
}
QMenuBar {
    background-color: #030712;
    color: #e2e8f0;
}
QMenuBar::item:selected {
    background-color: #0f172a;
}
QMenu {
    background-color: #0f172a;
    color: #e2e8f0;
    border: 1px solid #1e293b;
}
QMenu::item:selected {
    background-color: #0891b2;
    color: #ffffff;
}
QSpinBox, QComboBox {
    background-color: #0f172a;
    color: #e2e8f0;
    border: 1px solid #1e293b;
    border-radius: 4px;
    padding: 4px;
}
QSlider::groove:horizontal {
    background: #1e293b;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #22d3ee;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QScrollBar:vertical {
    background: #0f172a;
    width: 12px;
}
QScrollBar::handle:vertical {
    background: #1e293b;
    border-radius: 6px;
}
QLabel#header {
    font-size: 12px;
    font-weight: bold;
    color: #22d3ee;
}
QLabel {
    selection-background-color: #0891b2;
    selection-color: #ffffff;
}
"""

# Theme dictionary for easy access
THEMES = {
    "dark": DARK_STYLE,
    "light": LIGHT_STYLE,
    "shadow": SHADOW_STYLE,
    "midnight": MIDNIGHT_STYLE,
}

# Import enigma_engine modules
try:
    from ..config import CONFIG
    from ..core.model_config import MODEL_PRESETS
    from ..core.model_registry import ModelRegistry
    from ..core.model_scaling import shrink_model
except ImportError:
    # Running standalone
    pass


class SetupWizard(QWizard):
    """First-run setup wizard for creating a new AI."""
    
    def __init__(self, registry: ModelRegistry, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.setWindowTitle("Enigma Engine Setup Wizard")
        self.setWizardStyle(QWizard.ModernStyle)
        self.resize(600, 500)
        
        # Detect hardware BEFORE creating pages
        self.hw_profile = self._detect_hardware()
        
        # Add pages
        self.addPage(self._create_welcome_page())
        self.addPage(self._create_name_page())
        self.addPage(self._create_base_model_page())  # NEW: Base model selection
        self.addPage(self._create_size_page())
        self.addPage(self._create_confirm_page())
        
        self.model_name = None
        self.model_size = self.hw_profile.get("recommended", "tiny")
        self.base_model = None  # NEW: Track base model
    
    def _detect_hardware(self) -> dict:
        """Detect hardware capabilities for model size recommendations."""
        try:
            from ..core.hardware_detection import detect_hardware
            profile = detect_hardware()
            
            ram_gb = profile.total_ram_gb
            available_gb = profile.available_ram_gb
            vram_gb = profile.gpu_vram_gb or 0
            is_pi = profile.is_raspberry_pi
            is_mobile = profile.is_arm and not is_pi
            has_gpu = profile.has_cuda or profile.has_mps
            device_type = "Raspberry Pi" if is_pi else ("Mobile" if is_mobile else "PC")
            
            # Determine max safe size based on available memory
            effective_mem = vram_gb if has_gpu else min(available_gb, ram_gb * 0.4)
            
            if effective_mem < 1:
                max_size = "tiny"
                recommended = "tiny"
            elif effective_mem < 2:
                max_size = "small"
                recommended = "tiny"
            elif effective_mem < 4:
                max_size = "medium"
                recommended = "small"
            elif effective_mem < 8:
                max_size = "large"
                recommended = "medium"
            else:
                max_size = "xl"
                recommended = "large"
            
            # Force tiny for Pi/mobile
            if is_pi or is_mobile:
                max_size = "small"
                recommended = "tiny"
            
            return {
                "ram_gb": ram_gb,
                "available_gb": available_gb,
                "vram_gb": vram_gb,
                "has_gpu": has_gpu,
                "is_pi": is_pi,
                "is_mobile": is_mobile,
                "device_type": device_type,
                "max_size": max_size,
                "recommended": recommended,
                "effective_mem": effective_mem,
            }
        except Exception as e:
            # Fallback: assume limited hardware
            return {
                "ram_gb": 2,
                "available_gb": 1,
                "vram_gb": 0,
                "has_gpu": False,
                "is_pi": True,
                "is_mobile": False,
                "device_type": "Unknown",
                "max_size": "small",
                "recommended": "tiny",
                "effective_mem": 1,
                "error": str(e),
            }
    
    def _create_welcome_page(self):
        page = QWizardPage()
        page.setTitle("Welcome to Enigma AI Engine")
        page.setSubTitle("Let's set up your AI")
        
        layout = QVBoxLayout()
        
        welcome_text = QLabel("""
        <h3>Welcome!</h3>
        <p>This wizard will help you create your first AI model.</p>
        <p>Your AI starts as a <b>blank slate</b> - it will learn only from 
        the data you train it on. No pre-programmed emotions or personality.</p>
        <p><b>What you'll do:</b></p>
        <ul>
            <li>Give your AI a name</li>
            <li>Choose a model size based on your hardware</li>
            <li>Create the initial model (ready for training)</li>
        </ul>
        <p>Click <b>Next</b> to begin.</p>
        """)
        welcome_text.setWordWrap(True)
        layout.addWidget(welcome_text)
        
        page.setLayout(layout)
        return page
    
    def _create_name_page(self):
        page = QWizardPage()
        page.setTitle("Name Your AI")
        page.setSubTitle("Choose a unique name for this model")
        
        layout = QFormLayout()
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., artemis, apollo, atlas...")
        self.name_input.textChanged.connect(self._validate_name)
        
        self.name_status = QLabel("")
        
        layout.addRow("AI Name:", self.name_input)
        layout.addRow("", self.name_status)
        
        description_label = QLabel("""
        <p><b>Tips:</b></p>
        <ul>
            <li>Use lowercase letters, numbers, underscores</li>
            <li>Each AI gets its own folder in models/</li>
            <li>You can create multiple AIs with different names</li>
        </ul>
        """)
        description_label.setWordWrap(True)
        layout.addRow(description_label)
        
        # Register field for validation
        page.registerField("model_name*", self.name_input)
        
        page.setLayout(layout)
        return page
    
    def _validate_name(self, text):
        name = text.lower().strip().replace(" ", "_")
        if not name:
            self.name_status.setText("")
        elif name in self.registry.registry.get("models", {}):
            self.name_status.setText("[!] Name already exists!")
            self.name_status.setStyleSheet("color: orange")
        elif not name.replace("_", "").isalnum():
            self.name_status.setText("[X] Use only letters, numbers, underscores")
            self.name_status.setStyleSheet("color: red")
        else:
            self.name_status.setText("[OK] Name available")
            self.name_status.setStyleSheet("color: green")
    
    def _create_base_model_page(self):
        """Page to optionally select an existing model as base for transfer learning."""
        page = QWizardPage()
        page.setTitle("Base Model (Optional)")
        page.setSubTitle("Start from an existing model's knowledge, or begin fresh")
        
        layout = QVBoxLayout()
        
        # Option group
        self.base_model_group = QButtonGroup(page)
        
        # Fresh start option
        self.fresh_start_radio = QRadioButton("Start Fresh (no base model)")
        self.fresh_start_radio.setChecked(True)
        self.base_model_group.addButton(self.fresh_start_radio, 0)
        layout.addWidget(self.fresh_start_radio)
        
        # Use base model option
        self.use_base_radio = QRadioButton("Use existing model as base:")
        self.base_model_group.addButton(self.use_base_radio, 1)
        layout.addWidget(self.use_base_radio)
        
        # Model list
        self.base_model_list = QListWidget()
        self.base_model_list.setEnabled(False)
        self.base_model_list.setMinimumHeight(150)
        
        # Populate with existing models that have weights
        models = self.registry.registry.get("models", {})
        for name, info in models.items():
            model_path = Path(self.registry.models_dir) / name / "checkpoints"
            has_weights = model_path.exists() and any(model_path.glob("*.pt"))
            
            if has_weights:
                # Get additional info
                size = info.get("size", "unknown")
                epochs = info.get("epochs_trained", 0)
                item_text = f"{name} ({size}, {epochs} epochs trained)"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, name)  # Store actual name
                self.base_model_list.addItem(item)
        
        if self.base_model_list.count() == 0:
            item = QListWidgetItem("(No trained models available)")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.base_model_list.addItem(item)
            self.use_base_radio.setEnabled(False)
        
        layout.addWidget(self.base_model_list)
        
        # Connect radio buttons to enable/disable list
        self.use_base_radio.toggled.connect(self.base_model_list.setEnabled)
        self.use_base_radio.toggled.connect(lambda checked: 
            self.base_model_list.setCurrentRow(0) if checked and self.base_model_list.count() > 0 else None)
        
        # Info label
        info_label = QLabel(
            "Tip: Using a base model copies its trained weights and training data,\n"
            "giving your new model a head start. Great for specialization!"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-style: italic; margin-top: 10px;")
        layout.addWidget(info_label)
        
        layout.addStretch()
        page.setLayout(layout)
        return page
    
    def _create_size_page(self):
        page = QWizardPage()
        page.setTitle("Choose Model Size")
        page.setSubTitle("Select based on your hardware")
        
        layout = QVBoxLayout()
        
        self.size_group = QButtonGroup()
        
        # Size definitions with memory requirements
        sizes = [
            ("tiny", "Tiny (~0.5M params)", "Any device", "<1GB", 0.5),
            ("small", "Small (~10M params)", "4GB+ RAM", "2GB", 2),
            ("medium", "Medium (~50M params)", "8GB+ RAM or GPU", "4GB", 4),
            ("large", "Large (~150M params)", "GPU with 8GB+ VRAM", "8GB", 8),
        ]
        
        # Size order for comparison
        size_order = ["tiny", "small", "medium", "large", "xl"]
        max_idx = size_order.index(self.hw_profile.get("max_size", "tiny"))
        recommended = self.hw_profile.get("recommended", "tiny")
        
        for i, (size_id, name, hw, mem, req_gb) in enumerate(sizes):
            can_use = size_order.index(size_id) <= max_idx
            is_recommended = (size_id == recommended)
            
            label = f"{name}\n    {hw} | Needs: {mem}"
            if is_recommended:
                label += " [!] RECOMMENDED"
            if not can_use:
                label += " [!] TOO LARGE"
            
            radio = QRadioButton(label)
            radio.size_id = size_id
            radio.setEnabled(can_use)
            self.size_group.addButton(radio, i)
            layout.addWidget(radio)
            
            if is_recommended and can_use:
                radio.setChecked(True)
        
        # If nothing checked, check tiny
        if not self.size_group.checkedButton():
            for btn in self.size_group.buttons():
                if btn.size_id == "tiny":
                    btn.setChecked(True)
                    break
        
        # Hardware info from detection
        hw = self.hw_profile
        hw_text = f"""
        <hr>
        <p><b>Your Hardware:</b> {hw.get('device_type', 'Unknown')}</p>
        <ul>
            <li>RAM: {hw.get('ram_gb', '?')} GB (available: {hw.get('available_gb', '?')} GB)</li>
            <li>GPU VRAM: {hw.get('vram_gb', 0)} GB {'[OK]' if hw.get('has_gpu') else '(no GPU)'}</li>
            <li>Effective memory for models: ~{hw.get('effective_mem', 1):.1f} GB</li>
        </ul>
        <p><b>Recommendation:</b> <span style="color: green;">{recommended.upper()}</span></p>
        <p><i>You can grow your model later with better hardware!</i></p>
        """
        note = QLabel(hw_text)
        note.setWordWrap(True)
        layout.addWidget(note)
        
        page.setLayout(layout)
        return page
    
    def _create_confirm_page(self):
        page = QWizardPage()
        page.setTitle("Confirm Setup")
        page.setSubTitle("Review your choices")
        
        layout = QVBoxLayout()
        
        self.confirm_label = QLabel()
        self.confirm_label.setWordWrap(True)
        layout.addWidget(self.confirm_label)
        
        page.setLayout(layout)
        return page
    
    def initializePage(self, page_id):
        """Called when a page is shown."""
        if page_id == 4:  # Confirm page (was 3 before base model page added)
            name = self.name_input.text().lower().strip().replace(" ", "_")
            
            checked = self.size_group.checkedButton()
            size = checked.size_id if checked else "small"
            
            # Get base model selection
            if self.use_base_radio.isChecked() and self.base_model_list.currentItem():
                self.base_model = self.base_model_list.currentItem().data(Qt.UserRole)
            else:
                self.base_model = None
            
            # MODEL_PRESETS contains ForgeConfig objects, not dicts
            config = MODEL_PRESETS.get(size)
            if config:
                dim = getattr(config, 'dim', '?')
                layers = getattr(config, 'n_layers', '?')
            else:
                dim = '?'
                layers = '?'
            
            # Estimate VRAM based on model size
            vram_estimates = {
                'nano': 0.1, 'micro': 0.2, 'tiny': 0.5, 'mini': 0.7,
                'small': 1, 'medium': 2, 'base': 3, 'large': 4,
                'xl': 8, 'xxl': 16, 'huge': 24, 'giant': 48,
                'colossal': 80, 'titan': 120, 'omega': 200
            }
            min_vram = vram_estimates.get(size, '?')
            
            # Base model display
            base_display = self.base_model if self.base_model else "<i>None (fresh start)</i>"
            
            self.confirm_label.setText(f"""
            <h3>Ready to Create Your AI</h3>
            <table>
                <tr><td><b>Name:</b></td><td>{name}</td></tr>
                <tr><td><b>Size:</b></td><td>{size}</td></tr>
                <tr><td><b>Base Model:</b></td><td>{base_display}</td></tr>
                <tr><td><b>Dimensions:</b></td><td>{dim}</td></tr>
                <tr><td><b>Layers:</b></td><td>{layers}</td></tr>
                <tr><td><b>Min VRAM:</b></td><td>{min_vram} GB</td></tr>
            </table>
            <br>
            <p>Click <b>Finish</b> to create your AI.</p>
            <p>The model will be saved in: <code>models/{name}/</code></p>
            """)
            
            self.model_name = name
            self.model_size = size
    
    def get_result(self):
        """Get the wizard result."""
        return {
            "name": self.model_name,
            "size": self.model_size,
            "base_model": self.base_model,
        }


# Import loading dialog from dialogs module (extracted for smaller file size)
from .dialogs.loading import ModelLoadingDialog

# Import model manager dialog from dialogs module (extracted for smaller file size)
from .dialogs.model_manager import ModelManagerDialog

# =============================================================================
# 🏰 ENHANCED MAIN WINDOW - THE GRAND CASTLE
# =============================================================================
# This is THE main application window - everything lives here!

class EnhancedMainWindow(QMainWindow):
    """
    Enhanced main window with setup wizard and model management.
    
    📖 WHAT THIS IS:
    The main application window that contains all of Enigma AI Engine's GUI.
    It's like a castle with many rooms (tabs) for different features.
    
    📐 WINDOW STRUCTURE:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Title Bar: "Enigma AI Engine - [Model Name]"                    [-][□][X] │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Tabs: [💬Chat][🎨Image][💻Code][🎬Video][🔊Audio][🎲3D][⚙️...]   │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │                      Main Content Area                              │
    │                   (Tab contents appear here)                        │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Status Bar: [Model: X | GPU: ✓ | Theme: Dark]                     │
    └─────────────────────────────────────────────────────────────────────┘
    
    📐 KEY ATTRIBUTES:
    ┌────────────────────────────────────────────────────────────────────┐
    │ registry         │ ModelRegistry - manages saved models           │
    │ current_model    │ Name of the currently loaded model             │
    │ engine           │ EnigmaEngine - the inference engine            │
    │ module_manager   │ ModuleManager - controls loaded modules        │
    │ chat_messages    │ List of conversation messages                  │
    │ _is_hf_model     │ True if using HuggingFace model (no training!) │
    │ _gui_settings    │ Saved preferences (theme, last model, etc.)    │
    └────────────────────────────────────────────────────────────────────┘
    
    📐 LIFECYCLE:
    1. __init__ - Set up window, load settings
    2. _build_ui - Create all tabs and widgets
    3. _run_setup_wizard - First-time setup (if no models)
    4. _load_model - Load the selected AI model
    5. User interacts with tabs...
    6. closeEvent - Save settings and cleanup
    
    🔗 CONNECTED TO:
    - enigma_engine/gui/tabs/*.py - All the tab panels
    - enigma_engine/core/inference.py - EnigmaEngine for AI responses
    - enigma_engine/memory/manager.py - Conversation storage
    - enigma_engine/modules/ - Module system integration
    """
    
    def __init__(self):
        """
        Initialize the main window.
        
        📖 WHAT HAPPENS:
        1. Set window properties (title, size, icon)
        2. Initialize ModelRegistry for model management
        3. Initialize ModuleManager for feature modules
        4. Build the UI (tabs, buttons, etc.)
        5. Either show setup wizard (first run) or model selector
        """
        super().__init__()
        self.setWindowTitle("Enigma AI Engine")
        # Set reasonable minimum size to prevent UI becoming unusable
        self.setMinimumSize(800, 600)  # Larger minimum for better usability
        self.resize(1100, 750)  # Default size
        
        # Set window icon
        self._set_window_icon()
        
        # Setup keyboard shortcuts for emergency close
        self._setup_shortcuts()
        
        # ─────────────────────────────────────────────────────────────────
        # Initialize UI Settings (global fonts/themes)
        # ─────────────────────────────────────────────────────────────────
        try:
            from .ui_settings import get_ui_settings
            self.ui_settings = get_ui_settings()
            # Apply global stylesheet (with text selection support)
            base_stylesheet = self.ui_settings.get_global_stylesheet()
            # Add global styles for better text selection and button sizing
            from .styles import GLOBAL_BASE_STYLE
            self.setStyleSheet(base_stylesheet + "\n" + GLOBAL_BASE_STYLE)
            # Listen for settings changes
            self.ui_settings.add_listener(self._on_ui_settings_changed)
        except Exception as e:
            print(f"Could not load UI settings: {e}")
            self.ui_settings = None
            # Apply fallback styles for text selection
            try:
                from .styles import GLOBAL_BASE_STYLE
                self.setStyleSheet(GLOBAL_BASE_STYLE)
            except Exception:
                pass
        
        # ─────────────────────────────────────────────────────────────────
        # Initialize the Model Registry
        # This tracks all saved models (local and HuggingFace)
        # ─────────────────────────────────────────────────────────────────
        self.registry = ModelRegistry()
        self.current_model_name = None
        self.engine = None
        
        # Load GUI settings (last model, window size, etc.)
        self._gui_settings = self._load_gui_settings()
        
        # ─────────────────────────────────────────────────────────────────
        # Initialize the Module Manager
        # This controls which features (modules) are active
        # ─────────────────────────────────────────────────────────────────
        try:
            from enigma_engine.modules import ModuleManager, register_all, set_manager
            self.module_manager = ModuleManager()
            register_all(self.module_manager)  # Register all built-in modules
            # Set as global manager so get_manager() returns this instance
            set_manager(self.module_manager)
            
            # Load saved module configuration or enable defaults
            if self.module_manager.load_config():
                print("Loaded saved module configuration")
            else:
                # Enable default modules on first run
                # Image, Avatar, Vision are on by default for a complete experience
                default_modules = ['avatar', 'image_gen_local', 'vision', 'memory', 'web_tools', 'file_tools']
                for mod_id in default_modules:
                    try:
                        self.module_manager.load(mod_id)
                    except Exception:
                        pass
                print("Enabled default modules")
        except Exception as e:
            print(f"Could not initialize ModuleManager: {e}")
            self.module_manager = None
        
        # ─────────────────────────────────────────────────────────────────
        # Initialize GUI Mode Manager
        # Controls which tabs are visible based on user preference
        # ─────────────────────────────────────────────────────────────────
        try:
            saved_mode = self._gui_settings.get("gui_mode", "standard")
            mode_map = {
                "simple": GUIMode.SIMPLE,
                "standard": GUIMode.STANDARD,
                "advanced": GUIMode.ADVANCED,
                "gaming": GUIMode.GAMING
            }
            initial_mode = mode_map.get(saved_mode, GUIMode.STANDARD)
            self.gui_mode_manager = GUIModeManager(initial_mode)
            print(f"GUI Mode: {self.gui_mode_manager.get_mode_name()}")
        except Exception as e:
            print(f"Could not initialize GUIModeManager: {e}")
            self.gui_mode_manager = GUIModeManager(GUIMode.STANDARD)
        
        # ─────────────────────────────────────────────────────────────────
        # Initialize state variables
        # ─────────────────────────────────────────────────────────────────
        self.auto_speak = self._gui_settings.get("auto_speak", False)
        self.microphone_enabled = self._gui_settings.get("microphone_enabled", False)
        self.chat_messages = []          # Conversation history
        self.learn_while_chatting = self._gui_settings.get("learn_while_chatting", True)
        
        # System prompt settings
        self._system_prompt_preset = self._gui_settings.get("system_prompt_preset", "simple")
        self._custom_system_prompt = self._gui_settings.get("custom_system_prompt", "")
        
        # Mini chat settings
        self._mini_chat_on_top = self._gui_settings.get("mini_chat_always_on_top", True)
        
        # Display names
        self.user_display_name = self._gui_settings.get("user_display_name", "You")
        
        # ─────────────────────────────────────────────────────────────────
        # Initialize Chat Sync
        # This allows the quick chat overlay to share the same engine
        # ─────────────────────────────────────────────────────────────────
        from .chat_sync import ChatSync
        ChatSync.reset_instance()  # Ensure fresh instance with QApplication
        self._chat_sync = ChatSync.instance()
        self._chat_sync.set_main_window(self)
        self._chat_sync.set_user_name(self.user_display_name)
        
        # Connect signals so main window updates when quick chat generates
        self._chat_sync.generation_started.connect(self._on_chat_sync_started)
        self._chat_sync.generation_finished.connect(self._on_chat_sync_finished)
        self._chat_sync.generation_stopped.connect(self._on_chat_sync_stopped)
        
        # ─────────────────────────────────────────────────────────────────
        # Initialize Avatar-AI Bridge
        # This connects AI responses to avatar expressions and gestures
        # ─────────────────────────────────────────────────────────────────
        self._avatar_bridge = None
        try:
            from ..avatar import AIAvatarBridge, create_avatar_bridge, get_avatar
            avatar = get_avatar()
            if avatar:
                self._avatar_bridge = create_avatar_bridge(avatar, enable_explicit=True)
                # Connect bridge signals
                if hasattr(self._avatar_bridge, 'emotion_detected'):
                    self._avatar_bridge.emotion_detected.connect(self._on_avatar_emotion)
                if hasattr(self._avatar_bridge, 'gesture_triggered'):
                    self._avatar_bridge.gesture_triggered.connect(self._on_avatar_gesture)
                print("Avatar-AI Bridge initialized")
        except Exception as e:
            print(f"Could not initialize Avatar-AI Bridge: {e}")
            self._avatar_bridge = None
        
        # ─────────────────────────────────────────────────────────────────
        # Initialize Federated Learning Integration
        # This connects training to the federated learning system
        # ─────────────────────────────────────────────────────────────────
        self._federated_learning = None
        try:
            from ..config import CONFIG
            if CONFIG.get("federated_learning", {}).get("enabled", False):
                from ..federated import FederatedLearning, FederationMode
                model_name = self._gui_settings.get("last_model", "enigma_engine")
                self._federated_learning = FederatedLearning(
                    model_name=model_name,
                    mode=FederationMode.OPT_IN,
                )
                print("Federated Learning integration initialized")
        except Exception as e:
            print(f"Could not initialize Federated Learning: {e}")
            self._federated_learning = None
        
        # ─────────────────────────────────────────────────────────────────
        # Initialize Learning Chat Integration
        # Detects corrections, teaching, and feedback in real-time
        # ─────────────────────────────────────────────────────────────────
        self._learning_integration = None
        try:
            from ..learning import LearningChatIntegration

            # Will be properly initialized once a model is loaded
            self._learning_integration_enabled = self._gui_settings.get("learning_integration", True)
            print(f"Learning integration: {'enabled' if self._learning_integration_enabled else 'disabled'}")
        except Exception as e:
            print(f"Could not initialize Learning integration: {e}")
            self._learning_integration_enabled = False
        
        # ─────────────────────────────────────────────────────────────────
        # Initialize AI Overlay
        # This is the transparent always-on-top gaming interface
        # ─────────────────────────────────────────────────────────────────
        self._overlay = None
        try:
            from ..config import CONFIG
            if CONFIG.get("overlay", {}).get("enabled", True):
                from .overlay import AIOverlay
                self._overlay = AIOverlay()
                # Connect overlay to engine when loaded
                if self.engine:
                    self._overlay.set_engine(self.engine)
                # Show on startup if configured
                if CONFIG.get("overlay", {}).get("show_on_startup", False):
                    self._overlay.show()
                else:
                    self._overlay.hide()
                print("AI Overlay initialized")
        except Exception as e:
            print(f"Could not initialize AI Overlay: {e}")
            self._overlay = None
        
        # Training state
        self._is_training = False   # True while training is running
        self._stop_training = False # Set to True to stop training
        
        # Track if current model is HuggingFace (can't train these!)
        self._is_hf_model = False
        
        # AI wants and learned generation systems
        self.wants_system = None     # AI's internal motivations and goals
        self.learned_generator = None  # AI's learned design generation
        
        # ─────────────────────────────────────────────────────────────────
        # Build the UI
        # This creates all tabs, buttons, and widgets
        # ─────────────────────────────────────────────────────────────────
        self._build_ui()
        
        # ─────────────────────────────────────────────────────────────────
        # Register with GUIStateManager for AI tool control
        # This allows AI tools to interact with the GUI
        # ─────────────────────────────────────────────────────────────────
        try:
            from .gui_state import get_gui_state
            gui_state = get_gui_state()
            gui_state.set_window(self)
        except Exception as e:
            print(f"Could not register with GUIStateManager: {e}")
        
        # ─────────────────────────────────────────────────────────────────
        # Apply saved settings to UI widgets AFTER building UI
        # ─────────────────────────────────────────────────────────────────
        self._apply_saved_settings_to_ui()
        
        # ─────────────────────────────────────────────────────────────────
        # Initialize global hotkeys
        # This allows summoning AI from anywhere (even fullscreen games)
        # ─────────────────────────────────────────────────────────────────
        self._init_global_hotkeys()
        
        # ─────────────────────────────────────────────────────────────────
        # First-run check or model selection
        # ─────────────────────────────────────────────────────────────────
        if not self.registry.registry.get("models"):
            # No models exist - show setup wizard
            self._run_setup_wizard()
        else:
            # Defer model loading to after GUI is shown
            self._show_model_selector_deferred()
    
    def _on_chat_sync_started(self, user_text: str):
        """Handle when shared ChatSync starts generating (from quick chat)."""
        # Show thinking indicator
        if hasattr(self, 'thinking_frame'):
            self.thinking_frame.show()
            self.thinking_label.setText("Generating response...")
        if hasattr(self, 'stop_btn'):
            self.stop_btn.show()
            self.stop_btn.setEnabled(True)
        if hasattr(self, 'send_btn'):
            self.send_btn.setEnabled(False)
            self.send_btn.setText("...")
    
    def _on_chat_sync_finished(self, response: str):
        """Handle when shared ChatSync finishes generating."""
        # Hide thinking indicator
        if hasattr(self, 'thinking_frame'):
            self.thinking_frame.hide()
        if hasattr(self, 'stop_btn'):
            self.stop_btn.hide()
        if hasattr(self, 'send_btn'):
            self.send_btn.setEnabled(True)
            self.send_btn.setText("Send")
        if hasattr(self, 'chat_status'):
            self.chat_status.setText("Ready")
    
    def _on_chat_sync_stopped(self):
        """Handle when shared ChatSync generation is stopped."""
        self._on_chat_sync_finished("")
    
    def _on_avatar_emotion(self, emotion: str):
        """Handle emotion detected by avatar bridge."""
        logger.debug(f"Avatar emotion: {emotion}")
        # The bridge already sets the avatar expression, but we can log or react here
        if hasattr(self, 'chat_status'):
            self.chat_status.setText(f"Emotion: {emotion}")
    
    def _on_avatar_gesture(self, gesture: str):
        """Handle gesture triggered by avatar bridge."""
        logger.debug(f"Avatar gesture: {gesture}")
    
    def _notify_avatar_response_start(self):
        """Notify avatar bridge that AI is starting to respond."""
        if self._avatar_bridge:
            self._avatar_bridge.on_response_start()
    
    def _notify_avatar_response_chunk(self, text: str) -> str:
        """Process response chunk through avatar bridge, returns cleaned text."""
        if self._avatar_bridge:
            return self._avatar_bridge.on_response_chunk(text)
        return text
    
    def _notify_avatar_response_end(self):
        """Notify avatar bridge that AI finished responding."""
        if self._avatar_bridge:
            self._avatar_bridge.on_response_end()
    
    def _on_federated_round_complete(self, round_num: int, improvement: float):
        """Handle when a federated learning round completes."""
        logger.info(f"Federated round {round_num} complete, improvement: {improvement:.4f}")
        if hasattr(self, 'statusBar'):
            self.statusBar().showMessage(f"Federated round {round_num}: +{improvement:.2%} improvement", 5000)
    
    def _set_window_icon(self):
        """Set the window icon from file or create a default."""
        from pathlib import Path
        try:
            icon_paths = [
                Path(__file__).parent / "icons" / "forge.ico",
                Path(__file__).parent / "icons" / "forge_256.png",
                Path(CONFIG.get("data_dir", "data")) / "icons" / "forge.ico",
            ]
            
            for icon_path in icon_paths:
                if icon_path.exists():
                    self.setWindowIcon(QIcon(str(icon_path)))
                    return
        except Exception as e:
            print(f"Could not set window icon: {e}")
    
    def _on_ui_settings_changed(self):
        """Handle UI settings changes (font scale, theme)."""
        if self.ui_settings:
            self.setStyleSheet(self.ui_settings.get_global_stylesheet())
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts including emergency close."""
        from PyQt5.QtGui import QKeySequence
        from PyQt5.QtWidgets import QShortcut

        # Alt+F4 - Force quit (works on most systems by default, but ensure it)
        quit_shortcut = QShortcut(QKeySequence("Alt+F4"), self)
        quit_shortcut.activated.connect(self._force_quit)
        
        # Ctrl+Shift+F4 - Emergency force quit (no confirmation)
        emergency_quit = QShortcut(QKeySequence("Ctrl+Shift+F4"), self)
        emergency_quit.activated.connect(self._emergency_quit)
        
        # Escape - Show GUI (when minimized to tray, global hotkey is set up separately)
        # Note: ESC while window is visible does nothing special (normal behavior)
        esc_shortcut = QShortcut(QKeySequence("Escape"), self)
        esc_shortcut.activated.connect(self._on_escape_pressed)
        
        # Ctrl+P - Quick persona switch
        persona_shortcut = QShortcut(QKeySequence("Ctrl+P"), self)
        persona_shortcut.activated.connect(self._quick_persona_switch)
    
    def _on_escape_pressed(self):
        """Handle escape key - if any popup/dialog is open, close it. Otherwise do nothing."""
        # This is mainly for consistency. The tray ESC functionality 
        # is handled via the overlay's keyPressEvent
        pass
    
    def _force_quit(self):
        """Force quit Enigma AI Engine (Alt+F4)."""
        self._save_gui_settings()
        self._cleanup_and_quit()
    
    def _emergency_quit(self):
        """Emergency quit - force terminate immediately."""
        import os
        print("[EMERGENCY] Force quitting Enigma AI Engine...")
        os._exit(0)
    
    def _quick_persona_switch(self):
        """Show quick persona switch menu (Ctrl+P hotkey)."""
        from PyQt5.QtWidgets import QMenu, QAction
        
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 20px;
            }
            QMenu::item:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
        """)
        
        try:
            from ..core.persona import PersonaManager
            manager = PersonaManager()
            personas = manager.list_personas()
            current = manager.current_persona_id
            
            if personas:
                for persona_id in personas:
                    try:
                        persona = manager.load_persona(persona_id)
                        if persona:
                            name = persona.name or persona_id
                            action = menu.addAction(name)
                            action.setCheckable(True)
                            action.setChecked(persona_id == current)
                            action.triggered.connect(
                                lambda checked, pid=persona_id: self._switch_persona(pid)
                            )
                    except Exception:
                        continue
            else:
                no_personas = menu.addAction("No personas available")
                no_personas.setEnabled(False)
            
            menu.addSeparator()
            new_action = menu.addAction("Create New Persona...")
            new_action.triggered.connect(self._create_quick_persona)
            
        except ImportError:
            menu.addAction("Persona system not available").setEnabled(False)
        except Exception as e:
            menu.addAction(f"Error: {str(e)[:30]}").setEnabled(False)
        
        # Show menu near the center of the window
        pos = self.mapToGlobal(self.rect().center())
        menu.exec_(pos)
    
    def _switch_persona(self, persona_id: str):
        """Switch to the specified persona."""
        try:
            from ..core.persona import PersonaManager
            manager = PersonaManager()
            if manager.set_current_persona(persona_id):
                persona = manager.get_current_persona()
                self.statusBar().showMessage(
                    f"Switched to persona: {persona.name or persona_id}", 3000
                )
                
                # Update engine if available
                if hasattr(self, 'engine') and self.engine:
                    if hasattr(self.engine, 'set_persona'):
                        self.engine.set_persona(persona)
                
                # Switch avatar for this persona (multi-avatar support)
                try:
                    from ..avatar import switch_avatar_for_persona
                    switch_avatar_for_persona(persona_id)
                except Exception as e:
                    logger.debug(f"Could not switch avatar: {e}")
                        
        except Exception as e:
            self.statusBar().showMessage(f"Error switching persona: {e}", 3000)
    
    def _create_quick_persona(self):
        """Quick create a new persona dialog."""
        from PyQt5.QtWidgets import QInputDialog
        
        name, ok = QInputDialog.getText(
            self, "Create Persona", "Persona name:"
        )
        if ok and name.strip():
            try:
                from ..core.persona import PersonaManager, AIPersona
                from datetime import datetime
                
                manager = PersonaManager()
                persona_id = name.strip().lower().replace(" ", "_")
                
                new_persona = AIPersona(
                    id=persona_id,
                    name=name.strip(),
                    created_at=datetime.now().isoformat(),
                    personality_traits={},
                    system_prompt=f"You are {name.strip()}, a helpful AI assistant.",
                    response_style="balanced",
                    description=f"Custom persona: {name.strip()}",
                    tags=["custom"]
                )
                manager.save_persona(new_persona)
                manager.set_current_persona(persona_id)
                
                self.statusBar().showMessage(f"Created and switched to persona: {name}", 3000)
                
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Error", f"Could not create persona: {e}")
    
    def _is_huggingface_model(self) -> bool:
        """Check if the currently loaded model is a HuggingFace model."""
        return getattr(self, '_is_hf_model', False) or getattr(self.engine, '_is_huggingface', False) if self.engine else False
    
    def _update_hf_feature_restrictions(self, is_huggingface: bool):
        """Enable/disable features based on whether current model is HuggingFace."""
        # Training controls - completely disabled for HF models
        if hasattr(self, 'btn_train'):
            self.btn_train.setEnabled(not is_huggingface)
            if is_huggingface:
                self.btn_train.setToolTip("Training is not available for HuggingFace models")
                self.btn_train.setStyleSheet("""
                    QPushButton {
                        padding: 12px 24px;
                        font-size: 12px;
                        font-weight: bold;
                        background-color: #45475a;
                        color: #6c7086;
                    }
                """)
            else:
                self.btn_train.setToolTip("Start training the model")
                self.btn_train.setStyleSheet("""
                    QPushButton {
                        padding: 12px 24px;
                        font-size: 12px;
                        font-weight: bold;
                        background-color: #a6e3a1;
                        color: #1e1e2e;
                    }
                    QPushButton:hover {
                        background-color: #b4f0b4;
                    }
                """)
        
        # Training file controls
        training_file_controls = ['btn_save_training', 'epochs_spin', 'batch_spin', 'lr_spin']
        for ctrl_name in training_file_controls:
            if hasattr(self, ctrl_name):
                ctrl = getattr(self, ctrl_name)
                ctrl.setEnabled(not is_huggingface)
        
        # Model Manager - Grow/Shrink buttons (already handled elsewhere but ensure consistency)
        if hasattr(self, 'btn_grow'):
            self.btn_grow.setEnabled(not is_huggingface)
        if hasattr(self, 'btn_shrink'):
            self.btn_shrink.setEnabled(not is_huggingface)
        
        # Learning indicator - hide for HF models (can't fine-tune them)
        if hasattr(self, 'learning_indicator'):
            if is_huggingface:
                self.learning_indicator.setText("Learning: N/A")
                self.learning_indicator.setStyleSheet("color: #bac2de; font-size: 12px;")
                self.learning_indicator.setToolTip(
                    "Learning is not available for HuggingFace models.\n\n"
                    "HuggingFace models are pre-trained and cannot be fine-tuned locally.\n"
                    "To use learning features, switch to a local Forge model."
                )
                self.learning_indicator.setCursor(Qt.ArrowCursor)  # Remove clickable cursor
                self.learning_indicator.mousePressEvent = lambda e: None  # Disable click
            else:
                # Re-enable for Forge models
                self.learning_indicator.setCursor(Qt.PointingHandCursor)
                from .tabs.chat_tab import _toggle_learning
                self.learning_indicator.mousePressEvent = lambda e: _toggle_learning(self)
                # Restore current state
                if getattr(self, 'learn_while_chatting', True):
                    self.learning_indicator.setText("Learning: ON")
                    self.learning_indicator.setStyleSheet("color: #a6e3a1; font-size: 12px;")
                else:
                    self.learning_indicator.setText("Learning: OFF")
                    self.learning_indicator.setStyleSheet("color: #bac2de; font-size: 12px;")
                self.learning_indicator.setToolTip(
                    "When Learning is ON, the AI records your conversations and uses them to improve.\n\n"
                    "How it works:\n"
                    "• Each Q&A pair is saved to the model's training data\n"
                    "• After enough interactions, the model can be retrained\n"
                    "• This helps the AI learn your preferences and style\n\n"
                    "Click to toggle learning on/off."
                )
    
    def _require_forge_model(self, feature_name: str) -> bool:
        """
        Check if current model is Forge. If HuggingFace, show warning and return False.
        Use this to guard Forge-only features like training.
        """
        if self._is_huggingface_model():
            # Get list of local Forge models
            forge_models = []
            for name, info in self.registry.registry.get("models", {}).items():
                if info.get("source") != "huggingface":
                    forge_models.append(name)
            
            model_list = ", ".join(forge_models) if forge_models else "Create one in Model Manager"
            
            QMessageBox.warning(
                self, 
                f"{feature_name} - Forge Model Required",
                f"<b>{feature_name}</b> is only available for local Forge models.<br><br>"
                f"You're currently using a HuggingFace model: <b>{self.current_model_name}</b><br><br>"
                f"HuggingFace models are pre-trained and cannot be modified through this interface.<br><br>"
                f"<b>Available Forge models:</b><br>{model_list}<br><br>"
                f"Switch to an Forge model in the <b>Model Manager</b> tab to use this feature."
            )
            return False
        return True
    
    def _init_global_hotkeys(self):
        """
        Initialize global hotkey system.
        
        This sets up hotkeys that work even when Enigma AI Engine is not focused,
        including in fullscreen games.
        """
        try:
            from ..config import CONFIG

            # Check if hotkeys are enabled
            if not CONFIG.get("enable_hotkeys", True):
                print("Global hotkeys disabled in config")
                return
            
            from ..core.hotkey_actions import get_hotkey_actions
            from ..core.hotkey_manager import DEFAULT_HOTKEYS, get_hotkey_manager

            # Get manager and actions
            self.hotkey_manager = get_hotkey_manager()
            self.hotkey_actions = get_hotkey_actions(self)
            
            # Get hotkey config
            hotkey_config = CONFIG.get("hotkeys", DEFAULT_HOTKEYS)
            
            # Register hotkeys
            registered_count = 0
            for name, hotkey in hotkey_config.items():
                # Get the action method
                action_method = getattr(self.hotkey_actions, name, None)
                if action_method:
                    success = self.hotkey_manager.register(
                        hotkey=hotkey,
                        callback=action_method,
                        name=name
                    )
                    if success:
                        registered_count += 1
            
            # Start listening
            if registered_count > 0:
                self.hotkey_manager.start()
                print(f"Registered {registered_count} global hotkeys")
            else:
                print("No hotkeys registered")
                
        except Exception as e:
            print(f"Could not initialize global hotkeys: {e}")
            self.hotkey_manager = None
            self.hotkey_actions = None
    
    def _apply_saved_settings_to_ui(self):
        """Apply saved settings to UI widgets after building the UI."""
        # Apply auto_speak to menu action
        if hasattr(self, 'auto_speak_action'):
            self.auto_speak_action.blockSignals(True)
            self.auto_speak_action.setChecked(self.auto_speak)
            self.auto_speak_action.setText(f"AI Auto-Speak ({'ON' if self.auto_speak else 'OFF'})")
            self.auto_speak_action.blockSignals(False)
        
        # Apply to voice button in chat tab
        if hasattr(self, 'btn_speak'):
            self.btn_speak.blockSignals(True)
            self.btn_speak.setChecked(self.auto_speak)
            self.btn_speak.blockSignals(False)
        
        # Apply microphone enabled to menu action
        if hasattr(self, 'microphone_action'):
            self.microphone_action.blockSignals(True)
            self.microphone_action.setChecked(self.microphone_enabled)
            self.microphone_action.setText(f"Microphone ({'ON' if self.microphone_enabled else 'OFF'})")
            self.microphone_action.blockSignals(False)
        
        # Apply learn while chatting to menu action
        if hasattr(self, 'learn_action'):
            self.learn_action.blockSignals(True)
            self.learn_action.setChecked(self.learn_while_chatting)
            self.learn_action.setText(f"Learn While Chatting ({'ON' if self.learn_while_chatting else 'OFF'})")
            self.learn_action.blockSignals(False)
        
        # Apply to learning indicator in chat tab
        if hasattr(self, 'learning_indicator'):
            if self.learn_while_chatting:
                self.learning_indicator.setText("Learning: ON")
                self.learning_indicator.setStyleSheet("color: #a6e3a1; font-size: 12px;")
            else:
                self.learning_indicator.setText("Learning: OFF")
                self.learning_indicator.setStyleSheet("color: #bac2de; font-size: 12px;")
        
        # Apply window position and size
        if self._gui_settings.get("window_width") and self._gui_settings.get("window_height"):
            self.resize(self._gui_settings["window_width"], self._gui_settings["window_height"])
        if self._gui_settings.get("window_x") is not None and self._gui_settings.get("window_y") is not None:
            self.move(self._gui_settings["window_x"], self._gui_settings["window_y"])
        
        # NOTE: We do NOT restore last_tab here because _build_ui() already
        # sets the sidebar to 'chat' tab. Setting content_stack without updating
        # sidebar causes a mismatch where sidebar shows chat but content shows avatar.
    
    def _load_gui_settings(self):
        """Load GUI settings from file."""
        from ..config import CONFIG
        settings_path = Path(CONFIG["data_dir"]) / "gui_settings.json"
        try:
            if settings_path.exists():
                with open(settings_path) as f:
                    return json.load(f)
        except Exception as e:
            print(f"Could not load GUI settings: {e}")
        return {}
    
    def _save_gui_settings(self):
        """Save GUI settings to file."""
        from PyQt5.QtGui import QGuiApplication

        from ..config import CONFIG
        
        settings_path = Path(CONFIG["data_dir"]) / "gui_settings.json"
        try:
            settings = {
                "last_model": self.current_model_name,
                "window_width": self.width(),
                "window_height": self.height(),
                "window_x": self.x(),
                "window_y": self.y(),
                "auto_speak": getattr(self, 'auto_speak', False),
                "microphone_enabled": getattr(self, 'microphone_enabled', False),
                "user_display_name": getattr(self, 'user_display_name', 'You'),
            }
            
            # Save current tab index if content_stack exists
            if hasattr(self, 'content_stack'):
                settings["last_tab"] = self.content_stack.currentIndex()
            
            # Save always on top state
            settings["always_on_top"] = bool(self.windowFlags() & Qt.WindowStaysOnTopHint)
            
            # Save monitor index
            screens = QGuiApplication.screens()
            current_screen = QGuiApplication.screenAt(self.geometry().center())
            if current_screen and current_screen in screens:
                settings["monitor_index"] = screens.index(current_screen)
            
            # Save avatar settings
            if hasattr(self, 'avatar_combo') and self.avatar_combo:
                settings["last_avatar"] = self.avatar_combo.currentText()
                settings["last_avatar_index"] = self.avatar_combo.currentIndex()
            if hasattr(self, '_current_path') and self._current_path:
                settings["last_avatar_path"] = str(self._current_path)
            settings["avatar_auto_load"] = getattr(self, '_avatar_auto_load', False)
            settings["avatar_auto_run"] = getattr(self, '_avatar_auto_run', False)
            settings["avatar_resize_enabled"] = getattr(self, '_avatar_resize_enabled', False)  # Default OFF
            
            # Save avatar overlay size if it exists
            if hasattr(self, '_overlay') and self._overlay:
                settings["avatar_overlay_size"] = getattr(self._overlay, '_size', 300)
            if hasattr(self, '_overlay_3d') and self._overlay_3d:
                settings["avatar_overlay_3d_size"] = getattr(self._overlay_3d, '_size', 250)
            
            # Save chat zoom level
            if hasattr(self, 'chat_display'):
                font = self.chat_display.font()
                font_size = font.pointSize()
                # Only save valid font sizes (Qt returns -1 for pixel-based fonts)
                if font_size > 0:
                    settings["chat_zoom"] = font_size
            
            # Save learn while chatting preference
            settings["learn_while_chatting"] = getattr(self, 'learn_while_chatting', True)
            
            # Save has_chatted flag for first-run tips
            settings["has_chatted"] = self._gui_settings.get("has_chatted", False)
            
            # Save system prompt settings
            settings["system_prompt_preset"] = getattr(self, '_system_prompt_preset', 'simple')
            settings["custom_system_prompt"] = getattr(self, '_custom_system_prompt', '')
            
            # Save mini chat settings
            settings["mini_chat_always_on_top"] = getattr(self, '_mini_chat_on_top', True)
            
            with open(settings_path, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Could not save GUI settings: {e}")
    
    def _show_close_dialog(self):
        """Show close options dialog - same as Quick Chat close."""
        from PyQt5.QtWidgets import (
            QDialog,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QVBoxLayout,
        )
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Close Options")
        dialog.setFixedSize(320, 140)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                border: 1px solid #313244;
            }
            QLabel {
                color: #cdd6f4;
                font-size: 12px;
            }
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        label = QLabel("What would you like to do?")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        # Hide button - hide to tray
        hide_btn = QPushButton("Hide")
        hide_btn.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
        """)
        hide_btn.setToolTip("Hide window to system tray")
        hide_btn.clicked.connect(lambda: self._close_action(dialog, 'hide'))
        btn_layout.addWidget(hide_btn)
        
        # Close button - close this window only
        close_btn = QPushButton("Close GUI")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #f9e2af;
                color: #1e1e2e;
                border: none;
            }
            QPushButton:hover {
                background-color: #f5c2e7;
            }
        """)
        close_btn.setToolTip("Close main GUI only")
        close_btn.clicked.connect(lambda: self._close_action(dialog, 'close'))
        btn_layout.addWidget(close_btn)
        
        # Quit All button - quit entire app
        quit_btn = QPushButton("Quit All")
        quit_btn.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8;
                color: #1e1e2e;
                border: none;
            }
            QPushButton:hover {
                background-color: #ed5b7c;
            }
        """)
        quit_btn.setToolTip("Quit Enigma AI Engine completely")
        quit_btn.clicked.connect(lambda: self._close_action(dialog, 'quit'))
        btn_layout.addWidget(quit_btn)
        
        layout.addLayout(btn_layout)
        dialog.exec_()
    
    def _close_action(self, dialog, action):
        """Handle close dialog action."""
        dialog.close()
        
        # ALWAYS save settings on any close action
        self._save_gui_settings()
        
        if action == 'hide':
            # Just hide to tray
            self.hide()
        elif action == 'close':
            # Close GUI but keep tray running
            self.hide()
        elif action == 'quit':
            # Quit everything - properly terminate all processes
            self._cleanup_and_quit()
    
    def _cleanup_and_quit(self):
        """Clean up all resources and quit the application."""
        import os
        import sys
        
        print("[Enigma AI Engine] Shutting down all components...")
        
        try:
            # Stop voice systems
            try:
                from ..voice.voice_pipeline import get_voice_pipeline
                pipeline = get_voice_pipeline()
                if pipeline:
                    pipeline.stop()
                    print("  - Voice pipeline stopped")
            except Exception:
                pass  # Continue cleanup even if voice pipeline fails
            
            # Stop voice listener
            try:
                from ..voice.listener import get_listener
                listener = get_listener()
                if listener:
                    listener.stop()
                    print("  - Voice listener stopped")
            except Exception:
                pass  # Continue cleanup even if voice listener fails
            
            # Stop performance monitor
            try:
                from ..utils.performance_monitor import get_monitor
                monitor = get_monitor()
                if monitor:
                    monitor.stop()
                    print("  - Performance monitor stopped")
            except Exception:
                pass  # Continue cleanup even if monitor fails
            
            # Stop any web server
            try:
                from ..web.app import shutdown_server
                shutdown_server()
                print("  - Web server stopped")
            except Exception:
                pass  # Continue cleanup even if web server fails
            
            # Clean up temporary attachment files
            try:
                from .tabs.chat_tab import cleanup_temp_attachments
                cleanup_temp_attachments()
                print("  - Temp attachments cleaned")
            except Exception:
                pass  # Continue cleanup even if temp cleanup fails
            
            # Clear model from memory
            try:
                if self.engine:
                    self.engine = None
                    print("  - Model unloaded")
            except Exception:
                pass  # Continue cleanup even if model unload fails
            
            print("[Enigma AI Engine] Cleanup complete. Exiting...")
            
        except Exception as e:
            print(f"[Enigma AI Engine] Error during cleanup: {e}")
        
        # Force quit the application
        from PyQt5.QtWidgets import QApplication
        QApplication.instance().quit()
        
        # If Qt quit doesn't work, force exit
        import threading
        def force_exit():
            import time
            time.sleep(0.5)
            os._exit(0)
        
        threading.Thread(target=force_exit, daemon=True).start()
    
    def closeEvent(self, event):
        """Handle window close with confirmation dialog.
        
        This is triggered by clicking the window X button or Alt+F4.
        Shows a confirmation dialog before quitting.
        """
        from PyQt5.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self,
            "Quit Enigma AI Engine",
            "Are you sure you want to quit?\n\nAll background processes will be stopped.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
            self._save_gui_settings()
            self._cleanup_and_quit()
        else:
            event.ignore()
    
    def contextMenuEvent(self, event):
        """Show right-click context menu with common options."""
        from PyQt5.QtWidgets import QAction, QMenu
        
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QMenu::separator {
                height: 1px;
                background: #45475a;
                margin: 4px 8px;
            }
        """)
        
        # Quick actions
        new_chat_action = menu.addAction("New Chat")
        new_chat_action.triggered.connect(self._new_chat_from_menu)
        
        menu.addSeparator()
        
        # View options
        always_on_top_action = QAction("Always on Top", self)
        always_on_top_action.setCheckable(True)
        always_on_top_action.setChecked(bool(self.windowFlags() & Qt.WindowStaysOnTopHint))
        always_on_top_action.triggered.connect(self._toggle_always_on_top)
        menu.addAction(always_on_top_action)
        
        menu.addSeparator()
        
        # Navigation shortcuts
        nav_menu = menu.addMenu("Go to Tab")
        tabs = [("Chat", "chat"), ("Image", "image"), ("Code", "code"), 
                ("Settings", "settings"), ("Modules", "modules")]
        for name, key in tabs:
            action = nav_menu.addAction(name)
            action.triggered.connect(lambda checked, k=key: self._switch_to_tab(k))
        
        menu.addSeparator()
        
        # Window actions
        hide_action = menu.addAction("Hide to Tray")
        hide_action.triggered.connect(self.hide)
        
        quit_action = menu.addAction("Quit Enigma AI Engine")
        quit_action.triggered.connect(lambda: QApplication.instance().quit())
        
        menu.exec_(event.globalPos())
    
    def _new_chat_from_menu(self):
        """Start new chat from context menu."""
        self._switch_to_tab("chat")
        if hasattr(self, 'chat_display'):
            # Save current chat first
            if hasattr(self, '_save_conversation'):
                self._save_conversation()
            self.chat_display.clear()
            self.chat_messages = []
    
    def _toggle_always_on_top(self, checked):
        """Toggle always-on-top window flag and save setting."""
        if checked:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()  # Required after changing window flags
        
        # Save immediately
        self._gui_settings["always_on_top"] = checked
        self._save_gui_settings()
    
    def _show_options_menu(self):
        """Show options menu from the sidebar menu button."""
        from PyQt5.QtGui import QCursor
        from PyQt5.QtWidgets import QMenu

        # Reuse the context menu logic
        event_pos = QCursor.pos()
        
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QMenu::separator {
                height: 1px;
                background: #45475a;
                margin: 4px 8px;
            }
        """)
        
        # Quick actions
        new_chat_action = menu.addAction("New Chat")
        new_chat_action.triggered.connect(self._new_chat_from_menu)
        
        menu.addSeparator()
        
        # View options
        always_on_top_action = menu.addAction("Always on Top")
        always_on_top_action.setCheckable(True)
        always_on_top_action.setChecked(bool(self.windowFlags() & Qt.WindowStaysOnTopHint))
        always_on_top_action.triggered.connect(self._toggle_always_on_top)
        
        menu.addSeparator()
        
        # Navigation shortcuts
        nav_menu = menu.addMenu("Go to Tab")
        tabs = [("Chat", "chat"), ("Image", "image"), ("Code", "code"), 
                ("Settings", "settings"), ("Modules", "modules"), ("Files", "files")]
        for name, key in tabs:
            action = nav_menu.addAction(name)
            action.triggered.connect(lambda checked, k=key: self._switch_to_tab(k))
        
        menu.addSeparator()
        
        # Window actions
        hide_action = menu.addAction("Hide to Tray")
        hide_action.triggered.connect(self.hide)
        
        close_action = menu.addAction("Close Options...")
        close_action.triggered.connect(self._show_close_dialog)
        
        menu.exec_(event_pos)
    
    def keyPressEvent(self, event):
        """Handle key press events - Escape stops all generations."""
        from PyQt5.QtCore import Qt

        # Mode switching shortcuts
        if event.modifiers() == Qt.ControlModifier:
            # Tab shortcuts (Ctrl+1/2/3)
            tab_shortcuts = {
                Qt.Key_1: 'chat',
                Qt.Key_2: 'image',
                Qt.Key_3: 'avatar',
                Qt.Key_Comma: 'settings',
            }
            
            if event.key() in tab_shortcuts:
                self._switch_to_tab(tab_shortcuts[event.key()])
                return
            elif event.key() == Qt.Key_N:
                # New conversation
                if hasattr(self, 'btn_new_chat'):
                    self.btn_new_chat.click()
                return
        
        # Escape stops all generations
        if event.key() == Qt.Key_Escape:
            self._stop_all_generations()
        else:
            super().keyPressEvent(event)
    
    def _stop_all_generations(self):
        """Emergency stop all running generation tasks."""
        stopped_any = False
        
        # Stop chat AI worker
        if hasattr(self, '_ai_worker') and self._ai_worker:
            if self._ai_worker.isRunning():
                self._ai_worker.stop()
                stopped_any = True
        
        # Stop image generation
        if hasattr(self, 'image_tab') and self.image_tab:
            if hasattr(self.image_tab, 'worker') and self.image_tab.worker:
                if self.image_tab.worker.isRunning():
                    self.image_tab.worker.request_stop()
                    self.image_tab.worker.wait(500)  # Brief wait, don't block
                    stopped_any = True
        
        # Stop video generation
        if hasattr(self, 'video_tab') and self.video_tab:
            if hasattr(self.video_tab, 'worker') and self.video_tab.worker:
                if self.video_tab.worker.isRunning():
                    self.video_tab.worker.request_stop()
                    self.video_tab.worker.wait(500)  # Brief wait, don't block
                    stopped_any = True
        
        # Stop audio generation  
        if hasattr(self, 'audio_tab') and self.audio_tab:
            if hasattr(self.audio_tab, 'worker') and self.audio_tab.worker:
                if self.audio_tab.worker.isRunning():
                    if hasattr(self.audio_tab.worker, 'request_stop'):
                        self.audio_tab.worker.request_stop()
                    self.audio_tab.worker.wait(500)  # Brief wait, don't block
                    stopped_any = True
        
        # Stop code generation
        if hasattr(self, 'code_tab') and self.code_tab:
            if hasattr(self.code_tab, 'worker') and self.code_tab.worker:
                if self.code_tab.worker.isRunning():
                    if hasattr(self.code_tab.worker, 'request_stop'):
                        self.code_tab.worker.request_stop()
                    self.code_tab.worker.wait(500)  # Brief wait, don't block
                    stopped_any = True
        
        # Clear CUDA cache to free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass  # CUDA cleanup is optional
        
        if stopped_any:
            self.statusBar().showMessage("All generations stopped (Escape pressed)", 3000)
    
    def _run_setup_wizard(self):
        """Run first-time setup wizard."""
        wizard = SetupWizard(self.registry, self)
        if wizard.exec_() == QWizard.Accepted:
            result = wizard.get_result()
            try:
                self.registry.create_model(
                    result["name"],
                    size=result["size"],
                    vocab_size=32000,
                    description="Created via setup wizard",
                    base_model=result.get("base_model")
                )
                self.current_model_name = result["name"]
                self._load_current_model()
            except Exception as e:
                QMessageBox.critical(self, "Setup Failed", str(e))
                sys.exit(1)
        else:
            # User cancelled - exit
            sys.exit(0)
    
    def _show_model_selector_deferred(self):
        """Defer model loading until after GUI is shown."""
        from PyQt5.QtCore import QTimer

        # Select the model name now
        models = list(self.registry.registry.get("models", {}).keys())
        if models:
            last_model = self._gui_settings.get("last_model")
            if last_model and last_model in models:
                self.current_model_name = last_model
            else:
                self.current_model_name = models[0]
        
        # Update chat model label immediately so it shows the name even before load completes
        if hasattr(self, 'chat_model_label') and self.current_model_name:
            self.chat_model_label.setText(f"[AI] {self.current_model_name} (loading...)")
        
        # Show "loading" status in chat immediately
        if hasattr(self, 'chat_display'):
            self.chat_display.append(
                f"<p style='color: #f9e2af;'><i>⏳ Loading model: {self.current_model_name}...</i></p>"
            )
        
        if hasattr(self, 'chat_status'):
            self.chat_status.setText(f"Loading {self.current_model_name}...")
        
        # Update window title to show loading
        self.setWindowTitle(f"Enigma AI Engine - Loading {self.current_model_name}...")
        
        # Load model after a brief delay (allows GUI to fully render)
        QTimer.singleShot(100, self._load_current_model)
    
    def _show_model_selector(self):
        """Show model selection on startup (synchronous version)."""
        models = list(self.registry.registry.get("models", {}).keys())
        if models:
            # Try to use the last model from saved settings
            last_model = self._gui_settings.get("last_model")
            if last_model and last_model in models:
                self.current_model_name = last_model
            else:
                # Fall back to first model
                self.current_model_name = models[0]
        
        self._load_current_model()
    
    def _load_current_model(self):
        """Load the current model into the engine with progress dialog."""
        import sys
        sys.stdout.flush()
        
        if not self.current_model_name:
            return
        
        # Build list of activated elements to load
        # Only show items that actually take time to load
        loading_items = []
        
        # Model is always loaded (the main thing)
        loading_items.append({
            'name': self.current_model_name,
            'type': 'model',
            'icon': '>'
        })
        
        # Only add avatar if it's set to auto-run (actually needs loading)
        settings = self._load_gui_settings()
        if settings.get('avatar_auto_run', False):
            loading_items.append({
                'name': 'Avatar Overlay',
                'type': 'module',
                'icon': '>'
            })
        
        # Create and show loading dialog with all items
        loading_dialog = ModelLoadingDialog(
            loading_items=loading_items if len(loading_items) > 1 else None,
            model_name=self.current_model_name if len(loading_items) <= 1 else None,
            parent=self
        )
        loading_dialog.show()
        loading_dialog.raise_()
        loading_dialog.activateWindow()
        
        # Force dialog to fully render before starting loading
        QApplication.processEvents()
        QApplication.processEvents()  # Double process to ensure rendering
        
        # Small delay to ensure dialog is visible
        import time
        time.sleep(0.1)
        QApplication.processEvents()
        
        try:
            loading_dialog.set_status("Initializing...", 5)
            QApplication.processEvents()
            
            loading_dialog.set_status("Reading model registry...", 10)
            
            # Check for cancellation
            if loading_dialog.is_cancelled():
                loading_dialog.close()
                return
            
            # Create engine with selected model
            loading_dialog.set_status("Loading model weights from disk...", 15)
            
            # Progress callback for detailed model loading updates
            def on_load_progress(msg, pct):
                # Map registry progress (5-38%) to dialog progress (15-40%)
                mapped_pct = 15 + int((pct / 40) * 25)
                loading_dialog.set_status(msg, mapped_pct)
            
            model, config = self.registry.load_model(
                self.current_model_name,
                progress_callback=on_load_progress
            )
            loading_dialog.set_status("[OK] Model weights loaded", 40)
            
            
            if loading_dialog.is_cancelled():
                loading_dialog.close()
                return
            
            # Check if this is a HuggingFace model (wrapper class)
            is_huggingface = config.get("source") == "huggingface"
            self._is_hf_model = is_huggingface  # Track at window level for feature restrictions
            
            loading_dialog.set_status("Creating inference engine...", 45)
            
            if loading_dialog.is_cancelled():
                loading_dialog.close()
                return
            
            # Create engine instance without calling __init__
            from ..core.inference import EnigmaEngine
            self.engine = EnigmaEngine.__new__(EnigmaEngine)
            loading_dialog.set_status("[OK] Engine created", 50)
            
            # Set required attributes that __init__ would normally set
            import torch
            device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
            loading_dialog.set_status(f"Detecting device: {device_name}...", 55)
            self.engine.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.engine.use_half = False
            self.engine.enable_tools = False
            self.engine.module_manager = getattr(self, 'module_manager', None)
            self.engine._is_huggingface = is_huggingface
            
            # Set up tool executor for direct tool dispatch
            self.engine._tool_executor = None
            try:
                from ..tools.tool_executor import ToolExecutor
                self.engine._tool_executor = ToolExecutor(module_manager=self.engine.module_manager)
            except Exception as e:
                logger.warning(f"Could not create tool executor: {e}")
            
            # Enable direct routing for faster tool dispatch (image, code, etc.)
            self.engine.use_routing = True
            self.engine._tool_router = None
            try:
                from ..core.tool_router import get_router
                self.engine._tool_router = get_router(use_specialized=False)  # Keyword-based for speed
                loading_dialog.set_status("[OK] Fast routing enabled", 57)
            except Exception as e:
                logger.warning(f"Could not enable routing: {e}")
                self.engine.use_routing = False
            
            loading_dialog.set_status(f"Moving model to {device_name}...", 60)
            
            if loading_dialog.is_cancelled():
                loading_dialog.close()
                self.engine = None
                return
            
            if is_huggingface:
                # HuggingFaceModel is already loaded and ready
                self.engine.model = model  # This is a HuggingFaceModel wrapper
                loading_dialog.set_status("[OK] HuggingFace model ready", 65)
                
                # Check if user wants custom tokenizer instead of model's own
                use_custom_tokenizer = config.get("use_custom_tokenizer", False)
                if use_custom_tokenizer:
                    loading_dialog.set_status("Loading custom Forge tokenizer...", 70)
                    from ..core.tokenizer import load_tokenizer
                    self.engine.tokenizer = load_tokenizer()
                    self.engine._using_custom_tokenizer = True
                    loading_dialog.set_status("[OK] Custom tokenizer loaded", 75)
                else:
                    self.engine.tokenizer = model.tokenizer  # Use HF tokenizer
                    self.engine._using_custom_tokenizer = False
                    loading_dialog.set_status("[OK] Using model's tokenizer", 75)
            else:
                # Local Forge model
                self.engine.model = model
                loading_dialog.set_status("Moving model to GPU/CPU...", 68)
                self.engine.model.to(self.engine.device)
                self.engine.model.eval()
                loading_dialog.set_status("[OK] Model ready on device", 72)
                loading_dialog.set_status("Loading tokenizer...", 75)
                from ..core.tokenizer import load_tokenizer
                self.engine.tokenizer = load_tokenizer()
                loading_dialog.set_status("[OK] Tokenizer loaded", 80)
            
            if loading_dialog.is_cancelled():
                loading_dialog.close()
                self.engine = None
                return
            
            loading_dialog.set_status("Initializing AI brain & memory...", 85)
            
            # Initialize the AI's brain for learning
            from ..core.ai_brain import get_brain
            self.brain = get_brain(
                self.current_model_name, 
                auto_learn=getattr(self, 'learn_while_chatting', True)
            )
            loading_dialog.set_status("[OK] Brain initialized", 88)
            
            # Initialize wants system (AI's internal motivations)
            loading_dialog.set_status("Loading AI wants & motivations...", 90)
            from ..core.wants_system import get_wants_system
            self.wants_system = get_wants_system(self.current_model_name, Path(CONFIG["data_dir"]))
            self.log_terminal(f"AI wants system loaded: {len(self.wants_system.wants)} wants, {len(self.wants_system.goals)} goals", "info")
            loading_dialog.set_status("[OK] Wants system loaded", 92)
            
            # Initialize learned generator (AI creates designs from training)
            loading_dialog.set_status("Loading learned design system...", 94)
            from ..core.learned_generator import AILearnedGenerator
            self.learned_generator = AILearnedGenerator(self.current_model_name, Path(CONFIG["data_dir"]))
            
            # Auto-load training data if available
            training_file = Path(CONFIG["data_dir"]) / "specialized" / "wants_and_learned_design_training.txt"
            if training_file.exists():
                self.learned_generator.learn_from_training_data(training_file)
                self.log_terminal(f"Learned {len(self.learned_generator.learned_avatars)} avatar patterns", "info")
            
            loading_dialog.set_status("[OK] Learned generator ready", 96)
            
            loading_dialog.set_status("Finalizing setup...", 95)
            
            # Update window title with model type indicator
            model_type = "[HF]" if is_huggingface else "[Forge]"
            self.setWindowTitle(f"Enigma AI Engine - {self.current_model_name} {model_type}")
            
            # Update training tab label with warning for HF models
            if hasattr(self, 'training_model_label'):
                if is_huggingface:
                    self.training_model_label.setText(
                        f"Model: {self.current_model_name} <span style='color: #f9e2af;'>(HuggingFace - Training disabled)</span>"
                    )
                else:
                    self.training_model_label.setText(f"Model: {self.current_model_name}")
            
            # Disable training controls for HuggingFace models
            self._update_hf_feature_restrictions(is_huggingface)
            
            # Update chat tab model label
            if hasattr(self, 'chat_model_label'):
                self.chat_model_label.setText(f"[AI] {self.current_model_name}")
            
            loading_dialog.set_status("Ready!", 100)
            QApplication.processEvents()  # Ensure dialog updates
            
            # Show completion for a moment so user sees the full bar
            import time
            time.sleep(0.8)
            QApplication.processEvents()
            
            # Show welcome message in chat
            if hasattr(self, 'chat_display'):
                device_type = self.engine.device.type if hasattr(self.engine.device, 'type') else str(self.engine.device)
                device_info = "GPU" if device_type == "cuda" else "CPU"
                
                if is_huggingface:
                    model_note = f"<p style='color: #f9e2af;'><i>This is a HuggingFace model. Training and some Forge features are not available.</i></p>"
                else:
                    model_note = ""
                
                # Check if this is user's first time
                is_first_run = not self._gui_settings.get("has_chatted", False)
                
                if is_first_run:
                    # First-time user welcome
                    welcome_tips = """
                    <p style='color: #89b4fa;'><b>Welcome to Enigma AI Engine!</b> Here are some tips to get started:</p>
                    <ul style='color: #cdd6f4; margin-left: 20px;'>
                        <li>Just type naturally - say 'Generate an image of a sunset' or 'Write Python code for a calculator'</li>
                        <li>The AI auto-detects what you want and uses the right tool</li>
                        <li>Hover over sidebar items to see what each section does</li>
                        <li>Check the <b>Files</b> tab for the full Quick Start Guide</li>
                    </ul>
                    """
                else:
                    welcome_tips = ""
                
                self.chat_display.append(
                    f"<p style='color: #a6e3a1;'><b>[OK] Model loaded:</b> {self.current_model_name} ({device_info})</p>"
                    f"{model_note}"
                    f"{welcome_tips}"
                    f"<p style='color: #6c7086;'>Type a message below to chat with your AI.</p>"
                    "<hr>"
                )
            
            # Update chat status
            if hasattr(self, 'chat_status'):
                self.chat_status.setText(f"Model ready ({self.engine.device})")
            
            # Update status bar model button
            if hasattr(self, 'model_status_btn'):
                self.model_status_btn.setText(f"Model: {self.current_model_name}  v")
            
            # Update AI status display
            self._update_ai_status()
            
            # Update system tray with model name
            try:
                tray = get_system_tray()
                if tray and hasattr(tray, 'update_model_name'):
                    tray.update_model_name(self.current_model_name)
            except Exception:
                pass  # System tray update is optional
            
            # Sync engine to ChatSync for shared generation with quick chat
            if hasattr(self, '_chat_sync'):
                self._chat_sync.set_engine(self.engine)
                self._chat_sync.set_model_name(self.current_model_name)
            
            # Sync engine to overlay for AI responses
            if hasattr(self, '_overlay') and self._overlay:
                self._overlay.set_engine(self.engine)
                print("Overlay synced with engine")
            
            # Initialize learning integration (real-time learning from conversation)
            if getattr(self, '_learning_integration_enabled', False) and not is_huggingface:
                try:
                    from ..learning import LearningChatIntegration
                    self._learning_integration = LearningChatIntegration(
                        model=self.engine.model,
                        model_name=self.current_model_name,
                        auto_learn=True,
                        on_learning_detected=self._on_learning_detected,
                    )
                    print("Learning integration connected to model")
                    self.log_terminal("Real-time learning integration enabled", "info")
                except Exception as e:
                    print(f"Could not connect learning integration: {e}")
                    self._learning_integration = None
            
            # Refresh notes files for new model
            if hasattr(self, 'notes_file_combo'):
                self._refresh_notes_files()
                
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Could not load model: {e}")
            self.engine = None
            self.brain = None
            
            # Show error in chat
            if hasattr(self, 'chat_display'):
                self.chat_display.append(
                    f"<p style='color: #f38ba8;'><b>[!] Error:</b> Could not load model</p>"
                    f"<p style='color: #6c7086;'>{e}</p>"
                )
        finally:
            loading_dialog.close()
    
    def _build_ui(self):
        """
        Build the main UI.
        
        📖 WHAT THIS DOES:
        Creates all the visual elements of the main window:
        - Menu bar (File, Options, Help)
        - Tab widget with all feature tabs
        - Status bar at the bottom
        
        📐 UI STRUCTURE CREATED:
        ┌─────────────────────────────────────────────────────────────────┐
        │  [File][Options]              [Open Quick Chat]                │
        ├─────────────────────────────────────────────────────────────────┤
        │  [Chat][Image][Code][Video][Audio][3D][Train][Modules][...]    │
        ├─────────────────────────────────────────────────────────────────┤
        │                                                                 │
        │           QStackedWidget (shows current tab content)            │
        │                                                                 │
        ├─────────────────────────────────────────────────────────────────┤
        │  Model: xxx | GPU: ✓ |                                        │
        └─────────────────────────────────────────────────────────────────┘
        
        📐 MENUS CREATED:
        - File: New Model, Open Model, Backup, Exit
        - Options: Theme, Zoom, Avatar, Auto-Speak, Microphone, Learning
        
        📐 TABS CREATED (in order):
        1. Chat      - Main conversation interface
        2. Image     - Stable Diffusion / DALL-E
        3. Code      - Code generation
        4. Video     - Video generation
        5. Audio     - TTS / music
        6. GIF       - Animated GIFs
        7. 3D        - 3D model generation
        8. Vision    - Image analysis
        9. Embeddings- Vector embeddings
        10. Camera   - Webcam capture
        11. Train    - Model training
        12. Modules  - Module management
        13. Settings - Configuration
        """
        # ─────────────────────────────────────────────────────────────────
        # Menu bar - Create View menu for mode switching
        # ─────────────────────────────────────────────────────────────────
        self._create_view_menu()
        self._create_help_menu()
        
        # Initialize toggle state variables (previously in menu, now in Settings)
        self.learn_while_chatting = True
        self._companion = None
        
        # ─────────────────────────────────────────────────────────────────
        # Status bar (bottom of window)
        # Shows current model and status
        # ─────────────────────────────────────────────────────────────────
        self.model_status_btn = QPushButton(f"Model: {self.current_model_name or 'None'}  v")
        self.model_status_btn.setFlat(True)
        self.model_status_btn.setCursor(Qt.PointingHandCursor)
        self.model_status_btn.clicked.connect(self._on_open_model)
        self.model_status_btn.setToolTip("Click to change model")
        self.model_status_btn.setStyleSheet("""
            QPushButton {
                border: none;
                padding: 2px 8px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: rgba(137, 180, 250, 0.3);
                border-radius: 4px;
            }
        """)
        self.statusBar().addWidget(self.model_status_btn)
        
        # AI Connection Status - shows what's loaded
        self.ai_status_label = QLabel("AI: Connecting...")
        self.ai_status_label.setStyleSheet("""
            QLabel {
                color: #f39c12;
                padding: 2px 8px;
                font-size: 12px;
            }
        """)
        self.ai_status_label.setToolTip("AI connection status")
        self.statusBar().addPermanentWidget(self.ai_status_label)
        
        # Game Mode Indicator
        self.game_mode_indicator = QLabel("Game Mode: OFF")
        self.game_mode_indicator.setStyleSheet("""
            QLabel {
                color: #bac2de;
                padding: 2px 8px;
                font-size: 12px;
            }
        """)
        self.game_mode_indicator.setToolTip("Game Mode auto-reduces AI resources when gaming")
        self.game_mode_indicator.setCursor(Qt.PointingHandCursor)
        self.game_mode_indicator.mousePressEvent = lambda e: self._quick_toggle_game_mode()
        self.statusBar().addPermanentWidget(self.game_mode_indicator)
        
        # Register game mode callbacks
        try:
            from enigma_engine.core.game_mode import get_game_mode
            game_mode = get_game_mode()
            game_mode.on_game_detected(self._on_game_detected)
            game_mode.on_game_ended(self._on_game_ended)
            game_mode.on_limits_changed(self._on_game_limits_changed)
        except Exception as e:
            print(f"Could not register game mode callbacks: {e}")
        
        # Schedule initial status update
        QTimer.singleShot(1000, self._update_ai_status)
        QTimer.singleShot(1500, self._update_game_mode_status)
        
        # Apply dark mode by default
        self.setStyleSheet(DARK_STYLE)
        
        # Import tabs from separate modules
        from .tabs import (
            create_analytics_tab,
            create_audio_tab,
            create_avatar_subtab,
            create_camera_tab,
            create_chat_tab,
            create_code_tab,
            create_embeddings_tab,
            create_examples_tab,
            create_federation_tab,
            create_game_subtab,
            create_image_tab,
            create_instructions_tab,
            create_logs_tab,
            create_network_tab,
            create_notes_tab,
            create_robot_subtab,
            create_sessions_tab,
            create_terminal_tab,
            create_threed_tab,
            create_training_tab,
            create_training_data_tab,
            create_video_tab,
            create_vision_tab,
        )
        from .tabs.build_ai_tab import create_build_ai_tab
        from .tabs.bundle_manager_tab import create_bundle_manager_tab
        from .tabs.gif_tab import create_gif_tab
        from .tabs.learning_tab import LearningTab
        from .tabs.model_comparison_tab import create_model_comparison_tab
        from .tabs.model_router_tab import ModelRouterTab
        from .tabs.modules_tab import ModulesTab
        from .tabs.persona_tab import create_persona_tab
        from .tabs.scaling_tab import ScalingTab
        from .tabs.settings_tab import create_settings_tab
        from .tabs.tool_manager_tab import ToolManagerTab
        from .tabs.voice_clone_tab import VoiceCloneTab
        from .tabs.workspace_tab import create_workspace_tab

        # Create main container with sidebar navigation
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # === SIDEBAR NAVIGATION ===
        sidebar_container = QWidget()
        sidebar_container.setFixedWidth(140)
        sidebar_container.setStyleSheet("background-color: #11111b;")
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        
        # App title/logo area with close button
        title_widget = QWidget()
        title_widget.setFixedHeight(36)
        title_widget.setStyleSheet("""
            background-color: #11111b;
            border-bottom: 1px solid #1e1e2e;
        """)
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(10, 0, 6, 0)
        app_title = QLabel("FORGE AI")
        app_title.setStyleSheet("""
            color: #89b4fa;
            font-size: 12px;
            font-weight: bold;
            letter-spacing: 2px;
        """)
        app_title.setToolTip("Right-click anywhere for options menu")
        title_layout.addWidget(app_title)
        title_layout.addStretch()
        
        sidebar_layout.addWidget(title_widget)
        
        # Sidebar list widget
        self.sidebar = QListWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Define navigation items with sections
        # Format: (icon, name, key, tooltip)
        # REORGANIZED by TASK not type - related features grouped together
        nav_items = [
            # Chat - primary interaction
            ("section", "CHAT"),
            ("", "Chat", "chat", "Talk to your AI - start here!"),
            ("", "History", "history", "View past conversations"),
            
            # My AI - building and customizing your AI
            ("section", "MY AI"),
            ("", "Build AI", "build_ai", "Step-by-step AI creation wizard"),
            ("", "Bundles", "bundles", "Manage AI bundles for sharing"),
            ("", "Persona", "persona", "Create and manage AI personas"),
            ("", "Training", "training", "Train your AI model"),  # VISIBLE NOW
            ("", "Data Gen", "data_gen", "Generate training data with AI"),
            ("", "Learning", "learning", "Self-improvement metrics"),
            ("", "Scale", "scale", "Grow or shrink your AI model"),
            
            # Create - generation capabilities
            ("section", "CREATE"),
            ("", "Image", "image", "Generate images from text"),
            ("", "Code", "code", "AI-powered code generation"),
            ("", "Video", "video", "Create video content"),
            ("", "Audio", "audio", "Text-to-speech and audio"),
            ("", "Voice", "voice", "Clone and customize voices"),
            ("", "3D", "3d", "Generate 3D models"),
            ("", "GIF", "gif", "Create animated GIFs"),
            
            # Control - interaction and perception
            ("section", "CONTROL"),
            ("", "Avatar", "avatar", "AI avatar display and control"),
            ("", "Game", "game", "Connect AI to games"),
            ("", "Robot", "robot", "Control robots and hardware"),
            ("", "Screen", "vision", "Capture screenshots for AI"),
            ("", "Camera", "camera", "Live webcam preview"),
            
            # Tools - modules and routing
            ("section", "TOOLS"),
            ("", "Modules", "modules", "Enable/disable AI features"),
            ("", "Router", "router", "Assign specialized models"),
            ("", "Tools", "tools", "Manage AI tools"),
            ("", "Compare", "compare", "Compare model responses"),
            ("", "Search", "search", "Semantic search"),
            
            # System - settings and config
            ("section", "SYSTEM"),
            ("", "Terminal", "terminal", "View AI processing"),
            ("", "Logs", "logs", "View system logs"),
            ("", "Files", "files", "Edit files and settings"),
            ("", "Network", "network", "Multi-device networking"),
            ("", "Settings", "settings", "Configure preferences"),
            ("", "Workspace", "workspace", "Quick access to tasks"),
            ("", "Federation", "federation", "Federated learning"),
            ("", "Analytics", "analytics", "Usage statistics"),
            ("", "Examples", "examples", "Code examples"),
        ]
        
        # Add items to sidebar
        self._nav_map = {}  # Map item text to stack index
        self._sidebar_items = {}  # Map key to QListWidgetItem for visibility control
        self._sidebar_rows = {}  # Map key to row index
        stack_index = 0
        row_index = 0
        
        # Module-to-sidebar mapping: which modules control which sidebar items
        self._module_to_tabs = {
            'image_gen_local': ['image'],
            'image_gen_api': ['image'],
            'code_gen_local': ['code'],
            'code_gen_api': ['code'],
            'video_gen_local': ['video'],
            'video_gen_api': ['video'],
            'audio_gen_local': ['audio'],
            'audio_gen_api': ['audio'],
            'threed_gen_local': ['3d'],
            'threed_gen_api': ['3d'],
            'embedding_local': ['search'],
            'embedding_api': ['search'],
            'avatar': ['avatar'],
            'vision': ['vision', 'camera'],
            'voice_input': ['voice'],
            'voice_output': ['voice'],
            'gif_gen': ['gif'],
            'game_ai': ['game'],
            'robot_control': ['robot'],
        }
        
        # Tabs that should always be visible (core tabs)
        self._always_visible_tabs = [
            'chat', 'history', 'build_ai', 'bundles', 'persona', 'training', 'data_gen', 'learning', 'scale',  # MY AI section
            'modules', 'tools', 'router', 'compare', 'search',  # TOOLS section
            'terminal', 'logs', 'files', 'network', 'settings', 'workspace',  # SYSTEM section
            'federation', 'analytics', 'examples',
            # Default-enabled generation/perception tabs
            'image', 'avatar', 'vision'
        ]
        
        for item in nav_items:
            if item[0] == "section":
                # Section header - styled differently
                section_item = QListWidgetItem(item[1])
                section_item.setFlags(Qt.NoItemFlags)  # Not selectable
                font = section_item.font()
                font.setPointSize(8)
                font.setBold(True)
                section_item.setFont(font)
                section_item.setSizeHint(QSize(170, 28))
                # Use custom styling via foreground
                from PyQt5.QtGui import QBrush, QColor
                section_item.setForeground(QBrush(QColor("#6c7086")))
                self.sidebar.addItem(section_item)
            else:
                icon, name, key, tooltip = item
                list_item = QListWidgetItem(f"   {name}")
                list_item.setData(Qt.UserRole, key)
                list_item.setSizeHint(QSize(170, 38))
                list_item.setToolTip(tooltip)  # Add helpful tooltip for beginners
                self.sidebar.addItem(list_item)
                self._nav_map[key] = stack_index
                self._sidebar_items[key] = list_item
                self._sidebar_rows[key] = row_index
                stack_index += 1
            row_index += 1
        
        sidebar_layout.addWidget(self.sidebar)
        main_layout.addWidget(sidebar_container)
        
        # === CONTENT STACK ===
        self.content_stack = QStackedWidget()
        
        # Create scrollable containers for each tab
        def wrap_in_scroll(widget):
            """Wrap a widget in a scroll area."""
            scroll = QScrollArea()
            scroll.setWidget(widget)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            return scroll
        
        # Add all tabs to the stack (in order matching nav_items)
        # CHAT section
        self.content_stack.addWidget(wrap_in_scroll(create_chat_tab(self)))  # Chat
        self.content_stack.addWidget(wrap_in_scroll(create_sessions_tab(self)))  # History
        
        # MY AI section
        self.content_stack.addWidget(wrap_in_scroll(create_build_ai_tab(self)))  # Build AI (wizard)
        self.content_stack.addWidget(wrap_in_scroll(create_bundle_manager_tab(self)))  # Bundles
        self.persona_tab = create_persona_tab(self)  # Store reference for signals
        self.content_stack.addWidget(wrap_in_scroll(self.persona_tab))  # Persona
        self.content_stack.addWidget(wrap_in_scroll(create_training_tab(self)))  # Training (NOW VISIBLE)
        self.content_stack.addWidget(wrap_in_scroll(create_training_data_tab(self)))  # Data Gen
        self.content_stack.addWidget(wrap_in_scroll(LearningTab(self)))  # Learning
        self.content_stack.addWidget(wrap_in_scroll(ScalingTab(self)))  # Scale
        
        # CREATE section
        self.content_stack.addWidget(wrap_in_scroll(create_image_tab(self)))  # Image
        self.content_stack.addWidget(wrap_in_scroll(create_code_tab(self)))  # Code
        self.content_stack.addWidget(wrap_in_scroll(create_video_tab(self)))  # Video
        self.content_stack.addWidget(wrap_in_scroll(create_audio_tab(self)))  # Audio
        self.content_stack.addWidget(wrap_in_scroll(VoiceCloneTab(self)))  # Voice
        self.content_stack.addWidget(wrap_in_scroll(create_threed_tab(self)))  # 3D
        self.content_stack.addWidget(wrap_in_scroll(create_gif_tab(self)))  # GIF
        
        # CONTROL section
        self.content_stack.addWidget(wrap_in_scroll(create_avatar_subtab(self)))  # Avatar
        self.content_stack.addWidget(wrap_in_scroll(create_game_subtab(self)))  # Game
        self.content_stack.addWidget(wrap_in_scroll(create_robot_subtab(self)))  # Robot
        self.content_stack.addWidget(wrap_in_scroll(create_vision_tab(self)))  # Vision/Screen
        self.content_stack.addWidget(wrap_in_scroll(create_camera_tab(self)))  # Camera
        
        # TOOLS section
        self.content_stack.addWidget(wrap_in_scroll(ModulesTab(self, module_manager=self.module_manager)))  # Modules
        self.router_tab = ModelRouterTab(self)  # Store reference for syncing
        self.content_stack.addWidget(wrap_in_scroll(self.router_tab))  # Router
        self.content_stack.addWidget(wrap_in_scroll(ToolManagerTab(self)))  # Tools
        self.content_stack.addWidget(wrap_in_scroll(create_model_comparison_tab(self)))  # Compare
        self.content_stack.addWidget(wrap_in_scroll(create_embeddings_tab(self)))  # Search
        
        # SYSTEM section
        self.content_stack.addWidget(wrap_in_scroll(create_terminal_tab(self)))  # Terminal
        self.content_stack.addWidget(wrap_in_scroll(create_logs_tab(self)))  # Logs
        self.content_stack.addWidget(wrap_in_scroll(create_instructions_tab(self)))  # Files
        self.content_stack.addWidget(wrap_in_scroll(create_network_tab(self)))  # Network
        self.content_stack.addWidget(wrap_in_scroll(create_settings_tab(self)))  # Settings
        self.content_stack.addWidget(wrap_in_scroll(create_workspace_tab(self)))  # Workspace
        self.content_stack.addWidget(wrap_in_scroll(create_federation_tab(self)))  # Federation
        self.content_stack.addWidget(wrap_in_scroll(create_analytics_tab(self)))  # Analytics
        self.content_stack.addWidget(wrap_in_scroll(create_examples_tab(self)))  # Examples
        
        main_layout.addWidget(self.content_stack, stretch=1)
        
        # Connect sidebar selection to content stack
        self.sidebar.currentItemChanged.connect(self._on_sidebar_changed)
        
        # ALWAYS start on Chat tab (row 1 is Chat, row 0 is section header)
        # Find the Chat item and select it
        for i in range(self.sidebar.count()):
            item = self.sidebar.item(i)
            if item and item.data(Qt.UserRole) == 'chat':
                self.sidebar.setCurrentRow(i)
                break
        else:
            # Fallback: select row 1 (first item after header)
            self.sidebar.setCurrentRow(1)
        
        # Store reference for compatibility (tabs -> content_stack)
        self.tabs = self.content_stack
        
        # Apply tab visibility based on GUI mode
        self._apply_tab_visibility()
        
        self.setCentralWidget(main_widget)
        
        # Enable text selection on all QLabels in the GUI
        self._enable_text_selection()
        
        # Disable scroll wheel on all combo boxes to prevent accidental changes
        self._disable_combo_scroll()
        
        # Restore saved settings after UI is built
        self._restore_gui_settings()
    
    def _disable_combo_scroll(self):
        """Disable scroll wheel on all combo boxes to prevent accidental changes.
        
        Users often accidentally change dropdown selections when scrolling.
        This makes combos only respond to clicks, not scroll wheel.
        """
        from PyQt5.QtCore import QEvent, Qt
        from PyQt5.QtWidgets import QComboBox
        
        class ComboScrollFilter(QWidget):
            """Event filter that ignores scroll wheel events on combo boxes."""
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Wheel and isinstance(obj, QComboBox):
                    event.ignore()
                    return True
                return False
        
        # Create and store the filter (needs to stay alive)
        self._combo_scroll_filter = ComboScrollFilter(self)
        
        # Apply to all combo boxes in the window
        for combo in self.findChildren(QComboBox):
            combo.setFocusPolicy(Qt.StrongFocus)  # Only respond when explicitly focused
            combo.installEventFilter(self._combo_scroll_filter)
    
    def _enable_text_selection(self):
        """Enable text selection on all QLabel and text widgets in the GUI."""
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QTextBrowser, QTextEdit

        # Enable selection on all QLabel widgets
        for label in self.findChildren(QLabel):
            current_flags = label.textInteractionFlags()
            # Add text selection by mouse and keyboard
            new_flags = current_flags | Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
            if current_flags != new_flags:
                label.setTextInteractionFlags(new_flags)
        
        # Enable selection on QTextBrowser widgets (chat displays, logs, etc.)
        for browser in self.findChildren(QTextBrowser):
            current_flags = browser.textInteractionFlags()
            new_flags = current_flags | Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
            if current_flags != new_flags:
                browser.setTextInteractionFlags(new_flags)
        
        # QTextEdit widgets are already selectable by default, but ensure it
        for text_edit in self.findChildren(QTextEdit):
            if text_edit.isReadOnly():
                current_flags = text_edit.textInteractionFlags()
                new_flags = current_flags | Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
                if current_flags != new_flags:
                    text_edit.setTextInteractionFlags(new_flags)
    
    def _restore_gui_settings(self):
        """Restore GUI settings from saved file."""
        from PyQt5.QtGui import QGuiApplication
        
        settings = self._gui_settings
        
        # Restore window position based on startup_position_mode setting
        monitor_index = settings.get("monitor_index", 0)
        startup_mode = settings.get("startup_position_mode", "center")  # Default to center
        screens = QGuiApplication.screens()
        
        if monitor_index < len(screens):
            screen = screens[monitor_index]
            screen_geo = screen.geometry()
            
            if startup_mode == "remember":
                # Remember Last Position - use saved x/y if available
                x = settings.get("window_x")
                y = settings.get("window_y")
                
                if x is not None and y is not None:
                    # Verify the position is on a valid screen
                    from PyQt5.QtCore import QPoint
                    if QGuiApplication.screenAt(QPoint(x, y)):
                        self.move(x, y)
                    else:
                        # Position is off-screen, fall back to center
                        self.move(
                            screen_geo.x() + (screen_geo.width() - self.width()) // 2,
                            screen_geo.y() + (screen_geo.height() - self.height()) // 2
                        )
                else:
                    # No saved position, center on screen
                    self.move(
                        screen_geo.x() + (screen_geo.width() - self.width()) // 2,
                        screen_geo.y() + (screen_geo.height() - self.height()) // 2
                    )
            else:
                # Center on Display (default) - always center on selected monitor
                self.move(
                    screen_geo.x() + (screen_geo.width() - self.width()) // 2,
                    screen_geo.y() + (screen_geo.height() - self.height()) // 2
                )
        
        # Restore always on top - defer to after window is shown
        # Using QTimer to ensure window is fully loaded first
        if settings.get("always_on_top", False):
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(500, self._restore_always_on_top)
        
        # Restore auto-speak state
        if settings.get("auto_speak", False):
            self.auto_speak_action.setChecked(True)
            self._toggle_auto_speak(True)
        
        # Restore microphone state
        if settings.get("microphone_enabled", False):
            self.microphone_action.setChecked(True)
            self._toggle_microphone(True)
        
        # Restore chat zoom
        chat_zoom = settings.get("chat_zoom")
        # Only apply valid font sizes (must be positive integer)
        if chat_zoom and isinstance(chat_zoom, int) and chat_zoom > 0 and hasattr(self, 'chat_display'):
            font = self.chat_display.font()
            font.setPointSize(chat_zoom)
            self.chat_display.setFont(font)
        
        # Restore learn while chatting preference
        self.learn_while_chatting = settings.get("learn_while_chatting", True)
        
        # Restore system prompt settings
        self._system_prompt_preset = settings.get("system_prompt_preset", "simple")
        self._custom_system_prompt = settings.get("custom_system_prompt", "")
        
        # Restore mini chat on top preference
        self._mini_chat_on_top = settings.get("mini_chat_always_on_top", True)
        
        # Restore last tab the user was on
        last_tab = settings.get("last_tab", "chat")
        target_tab = last_tab if last_tab else "chat"
        
        # Find and select the saved tab
        for i in range(self.sidebar.count()):
            item = self.sidebar.item(i)
            if item and item.data(Qt.UserRole) == target_tab:
                self.sidebar.setCurrentRow(i)
                break
        else:
            # Fallback to chat if saved tab not found
            for i in range(self.sidebar.count()):
                item = self.sidebar.item(i)
                if item and item.data(Qt.UserRole) == 'chat':
                    self.sidebar.setCurrentRow(i)
                    break
        
        # Initialize tab visibility based on loaded modules
        self.update_tab_visibility()
    
    def _restore_always_on_top(self):
        """Restore always-on-top setting after window is fully loaded."""
        try:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
            self.show()
            # Update checkbox if it exists
            if hasattr(self, 'always_on_top_check'):
                self.always_on_top_check.blockSignals(True)
                self.always_on_top_check.setChecked(True)
                self.always_on_top_check.blockSignals(False)
        except Exception as e:
            print(f"Could not restore always-on-top: {e}")
    
    def _on_sidebar_changed(self, current, previous):
        """Handle sidebar navigation change."""
        if current:
            key = current.data(Qt.UserRole)
            if key and key in self._nav_map:
                self.content_stack.setCurrentIndex(self._nav_map[key])
                # Re-enable text selection for any new widgets in this tab
                self._enable_text_selection()
    
    def _switch_to_tab(self, tab_name: str):
        """Switch to a specific tab by name (for chat commands)."""
        # Find the tab key that matches - all available tabs
        key_map = {
            # Core tabs
            'chat': 'chat', 'train': 'train', 'history': 'history',
            'scale': 'scale', 'modules': 'modules', 'tools': 'tools',
            'router': 'router', 'compare': 'compare',
            # Generation tabs
            'image': 'image', 'code': 'code', 'video': 'video',
            'audio': 'audio', '3d': '3d', 'gif': 'gif',
            'search': 'search', 'embed': 'search', 'embeddings': 'search',
            # Control tabs
            'avatar': 'avatar', 'game': 'game', 'robot': 'robot',
            # Vision/Camera tabs
            'vision': 'vision', 'camera': 'camera',
            # System tabs
            'terminal': 'terminal', 'files': 'files', 'instructions': 'files',
            'logs': 'logs', 'notes': 'notes',
            # Network/Analytics
            'network': 'network', 'analytics': 'analytics',
            'scheduler': 'scheduler',
            # Other
            'examples': 'examples', 'settings': 'settings',
        }
        key = key_map.get(tab_name.lower())
        if key and key in self._nav_map:
            self.content_stack.setCurrentIndex(self._nav_map[key])
            # Also update sidebar selection
            for i in range(self.sidebar.count()):
                item = self.sidebar.item(i)
                if item and item.data(Qt.UserRole) == key:
                    self.sidebar.setCurrentRow(i)
                    break
    
    def _create_view_menu(self):
        """Create the View menu with mode switching options."""
        view_menu = self.menuBar().addMenu("&View")
        
        # Mode selection submenu
        mode_menu = view_menu.addMenu("GUI Mode")
        
        # Create action group for exclusive selection
        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)
        
        # Add mode options
        modes = [
            (GUIMode.SIMPLE, "Simple", "Essential features only"),
            (GUIMode.STANDARD, "Standard", "Balanced feature set (recommended)"),
            (GUIMode.ADVANCED, "Advanced", "All features visible"),
            (GUIMode.GAMING, "Gaming", "Minimal interface for gaming")
        ]
        
        for mode, name, description in modes:
            action = QAction(name, self)
            action.setToolTip(description)
            action.setCheckable(True)
            action.setChecked(self.gui_mode_manager.mode == mode)
            action.triggered.connect(lambda checked, m=mode: self._switch_gui_mode(m))
            mode_group.addAction(action)
            mode_menu.addAction(action)
        
        # Store reference for updates
        self._mode_actions = mode_group
        
        view_menu.addSeparator()
        
        # Add keyboard shortcuts info
        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self._show_shortcuts_dialog)
        view_menu.addAction(shortcuts_action)
    
    def _switch_gui_mode(self, mode: GUIMode):
        """Switch to a different GUI mode."""
        self.gui_mode_manager.set_mode(mode)
        self._apply_tab_visibility()
        
        # Save the preference
        self._gui_settings["gui_mode"] = mode.value
        self._save_gui_settings()
        
        # Show notification
        QMessageBox.information(
            self,
            "Mode Changed",
            f"GUI Mode changed to {self.gui_mode_manager.get_mode_name()}\n\n"
            f"{self.gui_mode_manager.get_mode_description()}"
        )
    
    def _apply_tab_visibility(self):
        """Apply tab visibility based on current GUI mode."""
        if not hasattr(self, '_sidebar_items'):
            return
        
        visible_tabs = self.gui_mode_manager.get_visible_tabs()
        
        for tab_key, item in self._sidebar_items.items():
            should_be_visible = self.gui_mode_manager.is_tab_visible(tab_key)
            item.setHidden(not should_be_visible)
    
    def _show_shortcuts_dialog(self):
        """Show dialog with keyboard shortcuts."""
        from .gui_modes import KEYBOARD_SHORTCUTS
        
        shortcuts_text = """
        <style>
            table { border-collapse: collapse; width: 100%; }
            th { text-align: left; padding: 8px; background: #313244; color: #cdd6f4; }
            td { padding: 6px 8px; border-bottom: 1px solid #45475a; }
            .key { background: #45475a; padding: 2px 6px; border-radius: 3px; font-family: monospace; }
        </style>
        <h3 style="color: #89b4fa;">Keyboard Shortcuts</h3>
        <table>
        <tr><th>Shortcut</th><th>Action</th></tr>
        """
        
        for shortcut, description in KEYBOARD_SHORTCUTS.items():
            shortcuts_text += f'<tr><td><span class="key">{shortcut}</span></td><td>{description}</td></tr>'
        shortcuts_text += "</table>"
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Keyboard Shortcuts")
        msg.setTextFormat(Qt.RichText)
        msg.setText(shortcuts_text)
        msg.setMinimumWidth(400)
        msg.exec_()
    
    def _create_help_menu(self):
        """Create the Help menu."""
        help_menu = self.menuBar().addMenu("&Help")
        
        # Keyboard shortcuts
        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.setShortcut("F1")
        shortcuts_action.triggered.connect(self._show_shortcuts_dialog)
        help_menu.addAction(shortcuts_action)
        
        # Getting started guide
        getting_started = QAction("Getting Started", self)
        getting_started.triggered.connect(self._show_getting_started)
        help_menu.addAction(getting_started)
        
        help_menu.addSeparator()
        
        # Documentation links
        docs_action = QAction("Open Documentation", self)
        docs_action.triggered.connect(lambda: self._open_url("https://github.com/SirKnightforge/Enigma AI Engine"))
        help_menu.addAction(docs_action)
        
        help_menu.addSeparator()
        
        # About
        about_action = QAction("About Enigma AI Engine", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)
    
    def _show_getting_started(self):
        """Show getting started guide."""
        guide = """
        <h2 style="color: #89b4fa;">Getting Started with Enigma AI Engine</h2>
        
        <h3>1. Chat with AI</h3>
        <p>The <b>Chat</b> tab is your main interface. Just type a message and press Enter or click Send.</p>
        
        <h3>2. Enable Features</h3>
        <p>Go to the <b>Modules</b> tab to enable additional features like:</p>
        <ul>
            <li>Image Generation - Create images from text</li>
            <li>Voice Input/Output - Talk to your AI</li>
            <li>Avatar - Visual AI companion</li>
        </ul>
        
        <h3>3. Customize Your AI</h3>
        <p>Use the <b>Persona</b> tab to give your AI a unique personality.</p>
        
        <h3>4. Train Your Model</h3>
        <p>Use the <b>Files</b> tab to add training data, then train in the <b>Chat</b> tab.</p>
        
        <h3>Quick Tips</h3>
        <ul>
            <li>Press <b>F1</b> anytime to see keyboard shortcuts</li>
            <li>Right-click for context menus</li>
            <li>Check the <b>Settings</b> tab for more options</li>
        </ul>
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Getting Started")
        msg.setTextFormat(Qt.RichText)
        msg.setText(guide)
        msg.setMinimumWidth(500)
        msg.exec_()
    
    def _show_about_dialog(self):
        """Show about dialog."""
        about_text = """
        <h2 style="color: #89b4fa;">Enigma AI Engine</h2>
        <p><b>Version:</b> 1.0.0</p>
        <p>A fully modular local AI assistant framework.</p>
        
        <p><b>Features:</b></p>
        <ul>
            <li>Local AI chat and generation</li>
            <li>Image, code, video, audio generation</li>
            <li>Voice input and output</li>
            <li>Avatar display</li>
            <li>Game integration</li>
            <li>Modular architecture</li>
        </ul>
        
        <p style="color: #6c7086; font-size: 10px;">
        Built with PyTorch, PyQt5, and love.
        </p>
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle("About Enigma AI Engine")
        msg.setTextFormat(Qt.RichText)
        msg.setText(about_text)
        msg.exec_()
    
    def _open_url(self, url: str):
        """Open a URL in the default browser."""
        from PyQt5.QtCore import QUrl
        from PyQt5.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(url))
    
    def _set_theme(self, theme_name):
        """Set the application theme."""
        if theme_name in THEMES:
            self.setStyleSheet(THEMES[theme_name])
            self.current_theme = theme_name
    
    def _toggle_auto_speak(self, checked):
        """Toggle auto-speak mode by loading/unloading voice output module."""
        if self.module_manager:
            if checked:
                # Load voice output module
                success = self.module_manager.load('voice_output')
                if success:
                    self.auto_speak = True
                    self.auto_speak_action.setText("AI Auto-Speak (ON)")
                else:
                    self.auto_speak_action.setChecked(False)
                    self.auto_speak_action.setText("AI Auto-Speak (OFF)")
                    QMessageBox.warning(self, "Voice Error", "Failed to load voice output module")
            else:
                # Unload voice output module
                self.module_manager.unload('voice_output')
                self.auto_speak = False
                self.auto_speak_action.setText("AI Auto-Speak (OFF)")
        else:
            # Fallback if no module manager
            self.auto_speak = checked
            if hasattr(self, 'auto_speak_action'):
                if checked:
                    self.auto_speak_action.setText("AI Auto-Speak (ON)")
                else:
                    self.auto_speak_action.setText("AI Auto-Speak (OFF)")
        
        # Sync with voice button in chat tab
        if hasattr(self, 'btn_speak'):
            self.btn_speak.blockSignals(True)
            self.btn_speak.setChecked(self.auto_speak)
            self.btn_speak.blockSignals(False)
    
    def _toggle_microphone(self, checked):
        """Toggle microphone listening by loading/unloading voice input module."""
        if self.module_manager:
            if checked:
                # Load voice input module
                success = self.module_manager.load('voice_input')
                if success:
                    self.microphone_enabled = True
                    self.microphone_action.setText("Microphone (ON)")
                else:
                    self.microphone_action.setChecked(False)
                    self.microphone_action.setText("Microphone (OFF)")
                    QMessageBox.warning(self, "Microphone Error", "Failed to load voice input module")
            else:
                # Unload voice input module
                self.module_manager.unload('voice_input')
                self.microphone_enabled = False
                self.microphone_action.setText("Microphone (OFF)")
        else:
            # Fallback if no module manager
            self.microphone_enabled = checked
            if hasattr(self, 'microphone_action'):
                if checked:
                    self.microphone_action.setText("Microphone (ON)")
                else:
                    self.microphone_action.setText("Microphone (OFF)")
    
    def _open_quick_chat(self):
        """Open the Quick Chat overlay."""
        try:
            tray = get_system_tray()
            if tray:
                tray.show_quick_command()
            else:
                QMessageBox.information(self, "Quick Chat", "Quick Chat is not available.\nStart Forge from run.py to enable system tray features.")
        except Exception as e:
            print(f"Error opening Quick Chat: {e}")
    
    def _toggle_overlay(self):
        """Toggle the AI overlay visibility."""
        if not self._overlay:
            try:
                from .overlay import AIOverlay
                self._overlay = AIOverlay()
                if self.engine:
                    self._overlay.set_engine(self.engine)
                print("AI Overlay created")
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Overlay Error",
                    f"Could not create overlay: {e}"
                )
                return
        
        # Toggle visibility
        if self._overlay.isVisible():
            self._overlay.hide()
        else:
            self._overlay.show()
    
    def _toggle_learning(self, checked):
        """Toggle learn-while-chatting mode."""
        # Check if using HuggingFace model - learning not supported
        if checked and self._is_huggingface_model():
            QMessageBox.information(
                self,
                "Learning Not Available",
                "Learning while chatting is only available for local Forge models.\n\n"
                f"Current model ({self.current_model_name}) is a HuggingFace model and cannot be trained.\n\n"
                "Switch to an Forge model to enable this feature."
            )
            # Reset the checkbox
            if hasattr(self, 'learn_action'):
                self.learn_action.setChecked(False)
            return
        
        self.learn_while_chatting = checked
        if hasattr(self, 'learn_action'):
            if checked:
                self.learn_action.setText("Learn While Chatting (ON)")
            else:
                self.learn_action.setText("Learn While Chatting (OFF)")
        
        # Update chat tab indicator
        if hasattr(self, 'learning_indicator'):
            if checked:
                self.learning_indicator.setText("Learning: ON")
                self.learning_indicator.setStyleSheet("color: #a6e3a1; font-size: 12px;")
            else:
                self.learning_indicator.setText("Learning: OFF")
                self.learning_indicator.setStyleSheet("color: #bac2de; font-size: 12px;")
        
        # Update brain if loaded
        if hasattr(self, 'brain') and self.brain:
            self.brain.auto_learn = checked
    
    def _toggle_companion_mode(self, checked):
        """Toggle Companion Mode - AI watches screen and comments."""
        try:
            from enigma_engine.companion import get_companion
            
            if checked:
                self._companion = get_companion()
                
                # Connect companion to chat
                def send_companion_message(msg):
                    """Send companion message to chat display."""
                    if hasattr(self, 'chat_display'):
                        # Format as AI message
                        self.chat_display.append(
                            f'<div style="margin: 8px 0; padding: 8px; background: #2d3748; border-radius: 8px;">'
                            f'<b style="color: #81a1c1;">Forge (observing):</b> '
                            f'<span style="color: #d8dee9;">{msg}</span></div>'
                        )
                        self.chat_display.verticalScrollBar().setValue(
                            self.chat_display.verticalScrollBar().maximum()
                        )
                
                self._companion.connect_chat(send_companion_message)
                
                # Connect to avatar if available
                if hasattr(self, '_avatar_overlay') and self._avatar_overlay:
                    def avatar_command(action, value):
                        from enigma_engine.tools.avatar_tools import _send_avatar_command
                        _send_avatar_command(action, str(value) if value else "")
                    self._companion.connect_avatar(avatar_command)
                
                # Connect to voice if auto-speak is on
                if self.auto_speak:
                    def speak_text(text):
                        try:
                            from enigma_engine.voice import get_voice
                            voice = get_voice()
                            if voice:
                                voice.speak(text)
                        except Exception:
                            pass
                    self._companion.connect_voice(speak_text)
                
                self._companion.start()
                self.companion_action.setText("Companion Mode (ON)")
                
                # Show status
                if hasattr(self, 'chat_status'):
                    self.chat_status.setText("Companion Mode Active")
            else:
                if self._companion:
                    self._companion.stop()
                    self._companion = None
                self.companion_action.setText("Companion Mode (OFF)")
                
                if hasattr(self, 'chat_status'):
                    self.chat_status.setText("Ready")
                    
        except Exception as e:
            print(f"Error toggling companion mode: {e}")
            self.companion_action.setChecked(False)
            QMessageBox.warning(self, "Companion Mode", f"Could not start Companion Mode:\n{e}")

    def _set_zoom(self, value: int):
        """Set zoom to a specific value."""
        self._current_zoom = value
        self._apply_zoom_value(value)
        # Update settings spinbox if it exists
        if hasattr(self, 'zoom_spinbox'):
            self.zoom_spinbox.setValue(value)
    
    def _adjust_zoom(self, delta: int):
        """Adjust zoom by a delta amount."""
        current = getattr(self, '_current_zoom', 100)
        new_value = max(80, min(200, current + delta))
        self._set_zoom(new_value)
    
    def _apply_zoom_value(self, value: int):
        """Apply zoom level to the application using stylesheet scaling."""
        try:
            from PyQt5.QtWidgets import QApplication
            
            app = QApplication.instance()
            if app is None:
                return
            
            # Calculate font size based on zoom (base size is ~10pt at 100%)
            base_font = max(7, int(10 * value / 100))
            small_font = max(6, int(8 * value / 100))
            large_font = max(9, int(12 * value / 100))
            header_font = max(11, int(14 * value / 100))
            
            # Apply global stylesheet with scaled fonts
            # This preserves the existing theme while adjusting sizes
            zoom_style = f"""
                * {{
                    font-size: {base_font}pt;
                }}
                QLabel {{
                    font-size: {base_font}pt;
                }}
                QPushButton {{
                    font-size: {base_font}pt;
                    padding: {max(4, int(6 * value / 100))}px {max(8, int(12 * value / 100))}px;
                }}
                QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox {{
                    font-size: {base_font}pt;
                    padding: {max(3, int(5 * value / 100))}px;
                }}
                QListWidget, QTreeWidget, QTableWidget {{
                    font-size: {base_font}pt;
                }}
                QTabBar::tab {{
                    font-size: {base_font}pt;
                    padding: {max(6, int(8 * value / 100))}px {max(10, int(14 * value / 100))}px;
                }}
                QMenuBar {{
                    font-size: {base_font}pt;
                }}
                QMenu {{
                    font-size: {base_font}pt;
                }}
                QGroupBox {{
                    font-size: {base_font}pt;
                }}
                QGroupBox::title {{
                    font-size: {large_font}pt;
                }}
                QStatusBar {{
                    font-size: {small_font}pt;
                }}
            """
            
            # Get existing stylesheet and append zoom overrides
            current_style = self.styleSheet() or ""
            # Remove any previous zoom styles (between markers)
            if "/* ZOOM_START */" in current_style:
                parts = current_style.split("/* ZOOM_START */")
                before = parts[0]
                if "/* ZOOM_END */" in parts[1]:
                    after = parts[1].split("/* ZOOM_END */")[1]
                else:
                    after = ""
                current_style = before + after
            
            # Add new zoom styles with markers
            new_style = current_style + f"\n/* ZOOM_START */\n{zoom_style}\n/* ZOOM_END */\n"
            self.setStyleSheet(new_style)
            
            # Also update QTextEdit default font for HTML content
            # This affects chat_display and other rich text widgets
            from PyQt5.QtGui import QFont
            from PyQt5.QtWidgets import QTextEdit
            text_font = QFont()
            text_font.setPointSize(base_font)
            for widget in self.findChildren(QTextEdit):
                widget.document().setDefaultFont(text_font)
                widget.update()
            
            self.statusBar().showMessage(f"Zoom: {value}%", 2000)
                    
        except Exception as e:
            print(f"Zoom error: {e}")
    
    def _show_zoom_dialog(self):
        """Show a dialog with live preview zoom slider."""
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSlider,
            QSpinBox,
            QVBoxLayout,
        )
        
        current = getattr(self, '_current_zoom', 100)
        original_zoom = current
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Zoom")
        dialog.setMinimumWidth(300)
        layout = QVBoxLayout(dialog)
        
        # Label
        label = QLabel("Drag slider or enter value (live preview):")
        layout.addWidget(label)
        
        # Slider + spinbox row
        slider_layout = QHBoxLayout()
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(80, 200)
        slider.setValue(current)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(20)
        slider_layout.addWidget(slider)
        
        spinbox = QSpinBox()
        spinbox.setRange(80, 200)
        spinbox.setValue(current)
        spinbox.setSuffix("%")
        spinbox.setMinimumWidth(70)
        slider_layout.addWidget(spinbox)
        
        layout.addLayout(slider_layout)
        
        # Live preview - connect slider/spinbox to apply zoom immediately
        def on_value_changed(value):
            slider.blockSignals(True)
            spinbox.blockSignals(True)
            slider.setValue(value)
            spinbox.setValue(value)
            slider.blockSignals(False)
            spinbox.blockSignals(False)
            self._apply_zoom_value(value)  # Live preview
        
        slider.valueChanged.connect(on_value_changed)
        spinbox.valueChanged.connect(on_value_changed)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            self._current_zoom = spinbox.value()
        else:
            # Restore original zoom if cancelled
            self._apply_zoom_value(original_zoom)
            self._current_zoom = original_zoom
    
    def _toggle_avatar(self, checked):
        """Toggle avatar enabled/disabled by loading/unloading the avatar module."""
        try:
            if self.module_manager:
                if checked:
                    # Load avatar module
                    success = self.module_manager.load('avatar')
                    if success:
                        self._enable_avatar()
                        self.avatar_action.setText("Avatar (ON)")
                        self.refresh_avatar_tab()
                        self.update_tab_visibility('avatar', True)
                    else:
                        self.avatar_action.setChecked(False)
                        self.avatar_action.setText("Avatar (OFF)")
                        QMessageBox.warning(self, "Avatar Error", "Failed to load avatar module")
                else:
                    # Unload avatar module
                    self.module_manager.unload('avatar')
                    self._disable_avatar()
                    self.avatar_action.setText("Avatar (OFF)")
                    self.refresh_avatar_tab()
                    self.update_tab_visibility('avatar', False)
            else:
                # Fallback if no module manager
                if checked:
                    self._enable_avatar()
                    self.avatar_action.setText("Avatar (ON)")
                else:
                    self._disable_avatar()
                    self.avatar_action.setText("Avatar (OFF)")
        except Exception as e:
            # Don't crash if avatar fails
            self.avatar_action.setChecked(False)
            self.avatar_action.setText("Avatar (OFF)")
            print(f"Avatar toggle error: {e}")
    
    def show_all_tabs(self):
        """Make all sidebar tabs visible (ignore module requirements)."""
        if not hasattr(self, '_sidebar_items'):
            return
        
        for tab_key, item in self._sidebar_items.items():
            item.setHidden(False)
        
        # Save preference
        self._gui_settings["show_all_tabs"] = True
        self._save_gui_settings()
        
        if hasattr(self, 'statusBar'):
            self.statusBar().showMessage("All tabs now visible", 3000)
    
    def update_tab_visibility(self, module_id: str = None, enabled: bool = None):
        """
        Update sidebar tab visibility based on module state.
        If module_id is None, updates all tabs based on current module states.
        """
        if not hasattr(self, '_sidebar_items'):
            return
        
        if module_id and module_id in self._module_to_tabs:
            # Update specific tabs for this module
            tabs = self._module_to_tabs[module_id]
            for tab_key in tabs:
                if tab_key in self._sidebar_items:
                    item = self._sidebar_items[tab_key]
                    # Show if enabled, hide if disabled (unless always visible)
                    if tab_key in self._always_visible_tabs:
                        item.setHidden(False)
                    else:
                        item.setHidden(not enabled)
        else:
            # Update all tabs based on current module state
            if self.module_manager:
                loaded_modules = set(self.module_manager.list_loaded())
            else:
                loaded_modules = set()
            
            # Build set of tabs that should be visible
            visible_tabs = set(self._always_visible_tabs)
            for mod_id, tabs in self._module_to_tabs.items():
                if mod_id in loaded_modules:
                    visible_tabs.update(tabs)
            
            # Update visibility
            for tab_key, item in self._sidebar_items.items():
                should_show = tab_key in visible_tabs
                item.setHidden(not should_show)
    
    def refresh_avatar_tab(self):
        """Refresh avatar tab UI to reflect current module state."""
        try:
            # Update the module status label and controls
            from .tabs.avatar.avatar_display import _is_avatar_module_enabled
            is_enabled = _is_avatar_module_enabled()
            
            # Update module status label
            if hasattr(self, 'module_status_label'):
                self.module_status_label.setVisible(not is_enabled)
            
            # Update checkbox state
            if hasattr(self, 'avatar_enabled_checkbox'):
                self.avatar_enabled_checkbox.setEnabled(is_enabled)
                if is_enabled:
                    # Re-enable and sync with avatar state
                    from ..avatar import get_avatar
                    avatar = get_avatar()
                    self.avatar_enabled_checkbox.blockSignals(True)
                    self.avatar_enabled_checkbox.setChecked(avatar.is_enabled)
                    self.avatar_enabled_checkbox.blockSignals(False)
            
            # Update show overlay button
            if hasattr(self, 'show_overlay_btn'):
                self.show_overlay_btn.setEnabled(is_enabled)
            
        except Exception as e:
            print(f"Error refreshing avatar tab: {e}")
    
    def on_module_toggled(self, module_id: str, enabled: bool):
        """
        Called when a module is toggled in the Modules tab.
        Updates tab visibility and refreshes relevant tabs.
        """
        # Update sidebar visibility
        self.update_tab_visibility(module_id, enabled)
        
        # Refresh specific tabs based on module
        if module_id == 'avatar':
            self.refresh_avatar_tab()
    
    def _toggle_screen_watching(self, checked):
        """Toggle continuous screen watching."""
        if checked:
            self.btn_start_watching.setText("[x] Stop Watching")
            interval_ms = self.vision_interval_spin.value() * 1000
            self.vision_timer.start(interval_ms)
            self._do_continuous_capture()
        else:
            self.btn_start_watching.setText("[o] Start Watching")
            self.vision_timer.stop()
    
    def _do_single_capture(self):
        """Do a single screen capture."""
        self._capture_screen()
    
    def _do_continuous_capture(self):
        """Capture for continuous watching."""
        self._capture_screen()
    
    def _capture_camera(self):
        """Capture image from webcam/camera."""
        try:
            import cv2

            # Try to open camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.vision_preview.setText("Camera not available\\n\\nMake sure a camera is connected.")
                return
            
            # Capture frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                self.vision_preview.setText("Failed to capture from camera")
                return
            
            # Convert BGR to RGB
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Store for analysis
            self._last_screenshot = img
            self.current_vision_image = "camera"
            
            # Resize for display
            display_img = img.copy()
            display_img.thumbnail((640, 400))
            
            # Convert to QPixmap
            import io
            buffer = io.BytesIO()
            display_img.save(buffer, format="PNG")
            buffer.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.read())
            self.vision_preview.setPixmap(pixmap)
            
            # Info
            width, height = img.size
            from datetime import datetime
            info = f"Camera: {width}x{height} | Captured: {datetime.now().strftime('%H:%M:%S')}"
            self.vision_text.setPlainText(info)
            
        except ImportError:
            self.vision_preview.setText("Camera capture requires OpenCV\\n\\nInstall: pip install opencv-python")
        except Exception as e:
            self.vision_preview.setText(f"Camera error: {e}")
    
    def _load_vision_image(self):
        """Load an image file for analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            from PIL import Image
            img = Image.open(file_path)
            
            # Store for analysis
            self._last_screenshot = img
            self.current_vision_image = file_path
            
            # Resize for display
            display_img = img.copy()
            display_img.thumbnail((640, 400))
            
            # Convert to QPixmap
            import io
            buffer = io.BytesIO()
            display_img.save(buffer, format="PNG")
            buffer.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.read())
            self.vision_preview.setPixmap(pixmap)
            
            # Info
            width, height = img.size
            from pathlib import Path
            info = f"Image: {Path(file_path).name} | {width}x{height}"
            self.vision_text.setPlainText(info)
            
        except Exception as e:
            self.vision_preview.setText(f"Error loading image: {e}")
    
    def _analyze_vision_image(self):
        """Have AI analyze the current image."""
        if not hasattr(self, '_last_screenshot') or self._last_screenshot is None:
            self.vision_text.setPlainText("No image to analyze. Capture or load an image first.")
            return
        
        # Get OCR text
        ocr_text = ""
        try:
            from ..tools.simple_ocr import extract_text
            ocr_text = extract_text(self._last_screenshot)
        except Exception:
            pass
        
        # Build analysis
        analysis = []
        analysis.append(f"Image size: {self._last_screenshot.size[0]}x{self._last_screenshot.size[1]}")
        
        if ocr_text:
            analysis.append(f"\nDetected text:\n{ocr_text}")
        else:
            analysis.append("\nNo text detected in image.")
        
        # If AI is available, get description
        if self.engine:
            try:
                prompt = "Describe what you might see in a screenshot or image."
                # Note: Real vision would need multi-modal model
                analysis.append(f"\n(AI vision analysis requires multi-modal model)")
            except Exception:
                pass
        
        self.vision_text.setPlainText("\n".join(analysis))
    
    def _capture_screen(self):
        """Capture screen and display it. Uses scrot on Linux (Wayland/Pi friendly)."""
        try:
            img = None
            error_msg = None
            
            # On Linux, use scrot (works on Wayland, X11, and Pi)
            import platform
            import shutil
            
            if platform.system() == "Linux" and shutil.which("scrot"):
                try:
                    import os
                    import subprocess
                    import tempfile

                    from PIL import Image
                    
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                        tmp_path = f.name
                    
                    # Run scrot with overwrite flag
                    result = subprocess.run(
                        ['scrot', '-o', tmp_path], 
                        capture_output=True, 
                        text=True, 
                        timeout=10
                    )
                    
                    if result.returncode == 0 and os.path.exists(tmp_path):
                        img = Image.open(tmp_path)
                        img = img.copy()  # Load into memory
                        os.unlink(tmp_path)  # Clean up temp file
                    else:
                        error_msg = f"scrot failed: {result.stderr}"
                except Exception as e:
                    error_msg = f"scrot error: {e}"
            
            # macOS - use screencapture
            elif platform.system() == "Darwin" and shutil.which("screencapture"):
                try:
                    import os
                    import subprocess
                    import tempfile

                    from PIL import Image
                    
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                        tmp_path = f.name
                    
                    subprocess.run(['screencapture', '-x', tmp_path], timeout=10)
                    
                    if os.path.exists(tmp_path):
                        img = Image.open(tmp_path)
                        img = img.copy()
                        os.unlink(tmp_path)
                except Exception as e:
                    error_msg = f"screencapture error: {e}"
            
            # Fallback: Try PIL ImageGrab (Windows, some X11)
            if img is None:
                try:
                    from PIL import ImageGrab
                    img = ImageGrab.grab()
                except Exception as e:
                    if error_msg:
                        error_msg += f", ImageGrab: {e}"
                    else:
                        error_msg = f"ImageGrab error: {e}"
            
            # Last resort: mss (may fail on Wayland)
            if img is None:
                try:
                    import mss
                    from PIL import Image
                    with mss.mss() as sct:
                        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                        screenshot = sct.grab(monitor)
                        img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
                except Exception as e:
                    if error_msg:
                        error_msg += f", mss: {e}"
                    else:
                        error_msg = f"mss error: {e}"
            
            if img is None:
                install_hint = ""
                if platform.system() == "Linux":
                    install_hint = "\n\nInstall scrot: sudo apt install scrot"
                self.vision_preview.setText(f"Screenshot failed\n\n{error_msg}{install_hint}")
                return
            
            # Save full image for AI analysis
            self._last_screenshot = img
            
            # Resize for display
            display_img = img.copy()
            display_img.thumbnail((640, 400))
            
            # Convert to QPixmap
            import io
            buffer = io.BytesIO()
            display_img.save(buffer, format="PNG")
            buffer.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.read())
            self.vision_preview.setPixmap(pixmap)
            
            # Basic info
            width, height = img.size
            from datetime import datetime
            info = f"Screen: {width}x{height} | Captured: {datetime.now().strftime('%H:%M:%S')}"
            self.vision_text.setPlainText(info)
            
        except Exception as e:
            self.vision_preview.setText(f"Error: {e}")
    
    # ========== TERMINAL METHODS ==========
    
    def _clear_terminal(self):
        """Clear the terminal output."""
        if hasattr(self, 'terminal_output'):
            self.terminal_output.clear()
            self._terminal_lines = []
    
    def _update_terminal_filter(self):
        """Update the terminal log level filter."""
        if hasattr(self, 'terminal_log_level'):
            self._terminal_log_level = self.terminal_log_level.currentText()
    
    def log_terminal(self, message, level="info"):
        """Log a message to the AI terminal display."""
        from .tabs.terminal_tab import log_to_terminal
        log_to_terminal(self, message, level)
    
    def update_terminal_stats(self, tokens_per_sec=None, memory_mb=None, model_name=None):
        """Update the terminal statistics display."""
        if tokens_per_sec is not None and hasattr(self, 'terminal_tps_label'):
            self.terminal_tps_label.setText(f"Tokens/sec: {tokens_per_sec:.1f}")
        if memory_mb is not None and hasattr(self, 'terminal_memory_label'):
            self.terminal_memory_label.setText(f"Memory: {memory_mb:.1f} MB")
        if model_name is not None and hasattr(self, 'terminal_model_label'):
            self.terminal_model_label.setText(f"Model: {model_name}")
    
    # ========== AI WATCHING METHODS ==========
    
    def ai_start_watching(self):
        """AI can start continuous screen watching."""
        if not self.btn_start_watching.isChecked():
            self.btn_start_watching.setChecked(True)
            self._toggle_screen_watching()
        return "Started screen watching"
    
    def ai_stop_watching(self):
        """AI can stop continuous screen watching."""
        if self.btn_start_watching.isChecked():
            self.btn_start_watching.setChecked(False)
            self._toggle_screen_watching()
        return "Stopped screen watching"
    
    def ai_capture_screen(self):
        """AI can capture a single screenshot."""
        self._capture_screen()
        return "Screen captured"
    
    def ai_get_screen_text(self):
        """AI can get OCR text from last screenshot."""
        if not hasattr(self, '_last_screenshot') or self._last_screenshot is None:
            return "No screenshot available. Use ai_capture_screen() first."
        try:
            from .tools.simple_ocr import extract_text
            text = extract_text(self._last_screenshot)
            return text if text else "No text detected in screenshot"
        except Exception:
            return "OCR not available"
    
    # === Session Actions ===
    
    def _populate_history_ai_selector(self):
        """Populate the AI selector dropdown in history tab."""
        if not hasattr(self, 'history_ai_selector'):
            return
        # Block signals to prevent double refresh
        self.history_ai_selector.blockSignals(True)
        self.history_ai_selector.clear()
        self.history_ai_selector.addItem("All AIs")
        for name in self.registry.registry.get("models", {}).keys():
            self.history_ai_selector.addItem(name)
        # Select current model if available
        if self.current_model_name:
            idx = self.history_ai_selector.findText(self.current_model_name)
            if idx >= 0:
                self.history_ai_selector.setCurrentIndex(idx)
        self.history_ai_selector.blockSignals(False)
    
    def _on_history_ai_changed(self, ai_name):
        """Handle AI selection change in history tab."""
        self._refresh_sessions()
    
    def _get_sessions_dir(self):
        """Get the sessions directory based on selected AI."""
        if hasattr(self, 'history_ai_selector'):
            ai_name = self.history_ai_selector.currentText()
            if ai_name and ai_name != "All AIs":
                # Per-AI sessions
                return Path(CONFIG.get("models_dir", "models")) / ai_name / "brain" / "conversations"
        # Default global sessions
        return Path(CONFIG.get("data_dir", "data")) / "conversations"
    
    def _refresh_sessions(self):
        """Refresh the list of saved sessions."""
        if not hasattr(self, 'sessions_list'):
            return
        self.sessions_list.clear()
        
        selected_ai = ""
        if hasattr(self, 'history_ai_selector'):
            selected_ai = self.history_ai_selector.currentText()
        
        if selected_ai == "All AIs" or not selected_ai:
            # Show sessions from all AIs
            all_sessions = []
            
            # Global sessions
            global_dir = Path(CONFIG.get("data_dir", "data")) / "conversations"
            if global_dir.exists():
                for f in global_dir.glob("*.json"):
                    all_sessions.append((f.stat().st_mtime, f.stem, "global"))
            
            # Per-AI sessions
            models_dir = Path(CONFIG.get("models_dir", "models"))
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    conv_dir = model_dir / "brain" / "conversations"
                    if conv_dir.exists():
                        for f in conv_dir.glob("*.json"):
                            all_sessions.append((f.stat().st_mtime, f.stem, model_dir.name))
            
            # Sort by time and display
            for mtime, name, ai in sorted(all_sessions, reverse=True):
                display = f"[{ai}] {name}" if ai != "global" else name
                self.sessions_list.addItem(display)
        else:
            # Show sessions for selected AI only
            conv_dir = self._get_sessions_dir()
            conv_dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(conv_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
                self.sessions_list.addItem(f.stem)
    
    def _load_session(self, item):
        """Load a session's content into the viewer."""
        if not item:
            return
        session_text = item.text()
        
        # Parse AI name from [ai_name] prefix if present
        if session_text.startswith("["):
            # Format: [ai_name] session_name
            bracket_end = session_text.find("]")
            ai_name = session_text[1:bracket_end]
            session_name = session_text[bracket_end + 2:]  # Skip "] "
            
            if ai_name == "global":
                conv_dir = Path(CONFIG.get("data_dir", "data")) / "conversations"
            else:
                conv_dir = Path(CONFIG.get("models_dir", "models")) / ai_name / "brain" / "conversations"
        else:
            # No prefix - use selected AI's folder or global
            session_name = session_text
            conv_dir = self._get_sessions_dir()
        
        session_file = conv_dir / f"{session_name}.json"
        
        if session_file.exists():
            try:
                data = json.loads(session_file.read_text())
                ai_label = data.get("ai_name", "Unknown AI")
                user_label = data.get("user_name", getattr(self, 'user_display_name', 'You'))
                html = f"<h3>{session_name}</h3><p><i>AI: {ai_label}</i></p><hr>"
                for msg in data.get("messages", []):
                    role = msg.get("role", "user")
                    text = msg.get("text", "")
                    if role == "user":
                        html += f"<p><b>{user_label}:</b> {text}</p>"
                    else:
                        html += f"<p><b>{ai_label}:</b> {text}</p>"
                self.session_viewer.setHtml(html)
                self._current_session = session_name
            except Exception as e:
                self.session_viewer.setPlainText(f"Error loading session: {e}")
    
    def _new_session(self):
        """Create a new chat session."""
        name, ok = QInputDialog.getText(self, "New Session", "Session name:")
        if ok and name:
            if hasattr(self, 'chat_messages') and self.chat_messages:
                self._save_current_chat(name)
            else:
                # Save to current model's folder
                if self.current_model_name:
                    conv_dir = Path(CONFIG.get("models_dir", "models")) / self.current_model_name / "brain" / "conversations"
                else:
                    conv_dir = Path(CONFIG.get("data_dir", "data")) / "conversations"
                conv_dir.mkdir(parents=True, exist_ok=True)
                session_file = conv_dir / f"{name}.json"
                session_file.write_text(json.dumps({
                    "name": name,
                    "ai_name": self.current_model_name or "unknown",
                    "saved_at": time.time(),
                    "messages": []
                }))
            self._refresh_sessions()
            self.chat_messages = []
    
    def _get_session_path(self, session_text):
        """Get the full path to a session file from its display text."""
        if session_text.startswith("["):
            bracket_end = session_text.find("]")
            ai_name = session_text[1:bracket_end]
            session_name = session_text[bracket_end + 2:]
            
            if ai_name == "global":
                conv_dir = Path(CONFIG.get("data_dir", "data")) / "conversations"
            else:
                conv_dir = Path(CONFIG.get("models_dir", "models")) / ai_name / "brain" / "conversations"
        else:
            session_name = session_text
            conv_dir = self._get_sessions_dir()
        
        return conv_dir / f"{session_name}.json", session_name
    
    def _rename_session(self):
        """Rename the selected session."""
        item = self.sessions_list.currentItem()
        if not item:
            QMessageBox.warning(self, "No Selection", "Select a session to rename")
            return
        
        old_path, old_name = self._get_session_path(item.text())
        new_name, ok = QInputDialog.getText(self, "Rename Session", "New name:", text=old_name)
        if ok and new_name and new_name != old_name:
            new_path = old_path.parent / f"{new_name}.json"
            if old_path.exists():
                old_path.rename(new_path)
                self._refresh_sessions()
    
    def _delete_session(self):
        """Delete the selected session."""
        item = self.sessions_list.currentItem()
        if not item:
            QMessageBox.warning(self, "No Selection", "Select a session to delete")
            return
        
        session_path, session_name = self._get_session_path(item.text())
        reply = QMessageBox.question(
            self, "Delete Session",
            f"Delete session '{session_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                if session_path.exists():
                    session_path.unlink()
                self._refresh_sessions()
                self.session_viewer.clear()
            except Exception as e:
                QMessageBox.warning(self, "Delete Failed", f"Could not delete session: {e}")
    
    def _load_session_into_chat(self):
        """Load the selected session into the chat tab."""
        item = self.sessions_list.currentItem()
        if not item:
            QMessageBox.warning(self, "No Session", "Select a session first")
            return
        
        session_path, session_name = self._get_session_path(item.text())
        if session_path.exists():
            try:
                data = json.loads(session_path.read_text())
                self.chat_display.clear()
                self.chat_messages = data.get("messages", [])
                ai_name = data.get("ai_name", self.current_model_name or "AI")
                user_name = data.get("user_name", getattr(self, 'user_display_name', 'You'))
                for msg in self.chat_messages:
                    role = msg.get("role", "user")
                    text = msg.get("text", "")
                    if role == "user":
                        self.chat_display.append(
                            f'<div style="background-color: #313244; padding: 8px; margin: 4px 0; border-radius: 8px; border-left: 3px solid #89b4fa;">'
                            f'<b style="color: #89b4fa;">{user_name}:</b> {text}</div>'
                        )
                    else:
                        self.chat_display.append(
                            f'<div style="background-color: #1e1e2e; padding: 8px; margin: 4px 0; border-radius: 8px; border-left: 3px solid #a6e3a1;">'
                            f'<b style="color: #a6e3a1;">{ai_name}:</b> {text}</div>'
                        )
                self.tabs.setCurrentIndex(0)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load session: {e}")
    
    def _save_current_chat(self, name=None):
        """Save current chat to a session file in the current AI's folder."""
        if not hasattr(self, 'chat_messages'):
            self.chat_messages = []
        if not name:
            name = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save to current model's conversations folder
        if self.current_model_name:
            conv_dir = Path(CONFIG.get("models_dir", "models")) / self.current_model_name / "brain" / "conversations"
        else:
            conv_dir = Path(CONFIG.get("data_dir", "data")) / "conversations"
        
        conv_dir.mkdir(parents=True, exist_ok=True)
        session_file = conv_dir / f"{name}.json"
        session_file.write_text(json.dumps({
            "name": name,
            "ai_name": self.current_model_name or "unknown",
            "user_name": getattr(self, 'user_display_name', 'You'),
            "saved_at": time.time(),
            "messages": self.chat_messages
        }))
    
    # === Data Editor Actions (for Training tab) ===
    
    
    def _refresh_data_files(self):
        """Refresh list of training data files."""
        if not hasattr(self, 'data_file_combo'):
            return
        self.data_file_combo.clear()
        
        # Get AI's data directory
        if self.current_model_name:
            model_info = self.registry.registry.get("models", {}).get(self.current_model_name, {})
            data_dir = model_info.get("data_dir") or (Path(model_info.get("path", "")) / "data")
            if isinstance(data_dir, str):
                data_dir = Path(data_dir)
        else:
            data_dir = Path(CONFIG.get("data_dir", "data"))
        
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure training.txt exists
        training_file = data_dir / "training.txt"
        if not training_file.exists():
            training_file.write_text("# Training Data\n# Add Q&A pairs below\n\nQ: Hello\nA: Hi there!\n")
        
        # Add files (training data files, not instructions)
        for f in sorted(data_dir.glob("*.txt")):
            if f.name not in ["instructions.txt", "notes.txt"]:
                self.data_file_combo.addItem(f.name, str(f))
        
        # Select first file if available
        if self.data_file_combo.count() > 0:
            self.data_file_combo.setCurrentIndex(0)
    
    def _load_data_file(self, index):
        """Load a data file into the training editor."""
        if index < 0 or not hasattr(self, 'data_file_combo'):
            return
        
        filepath = self.data_file_combo.itemData(index)
        if filepath and Path(filepath).exists():
            self.data_editor.setPlainText(Path(filepath).read_text(encoding='utf-8', errors='replace'))
            self._current_data_file = filepath
            self.training_data_path = filepath  # Auto-set for training
    
    def _save_data_file(self):
        """Save the training data file."""
        if not hasattr(self, '_current_data_file') or not self._current_data_file:
            QMessageBox.warning(self, "No File", "Select a file first")
            return
        
        try:
            Path(self._current_data_file).write_text(self.data_editor.toPlainText(), encoding='utf-8')
            QMessageBox.information(self, "Saved", "File saved!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save: {e}")
    
    def _create_data_file(self):
        """Create a new training data file."""
        name, ok = QInputDialog.getText(self, "New Training File", "File name (without .txt):")
        if ok and name:
            if not name.endswith(".txt"):
                name += ".txt"
            
            # Get data directory
            if self.current_model_name:
                model_info = self.registry.registry.get("models", {}).get(self.current_model_name, {})
                data_dir = model_info.get("data_dir") or (Path(model_info.get("path", "")) / "data")
                if isinstance(data_dir, str):
                    data_dir = Path(data_dir)
            else:
                data_dir = Path(CONFIG.get("data_dir", "data"))
            
            data_dir.mkdir(parents=True, exist_ok=True)
            new_file = data_dir / name
            
            if new_file.exists():
                QMessageBox.warning(self, "Exists", f"{name} already exists")
                return
            
            new_file.write_text("# Training Data\n# Add Q&A pairs below\n\n")
            self._refresh_data_files()
            
            # Select the new file
            idx = self.data_file_combo.findText(name)
            if idx >= 0:
                self.data_file_combo.setCurrentIndex(idx)
    
    # === Avatar Actions ===
    
    def _refresh_avatar_status(self):
        """Update avatar status - now handled by avatar_tab."""
        # Initialize avatar expressions dict if needed
        if not hasattr(self, 'avatar_expressions'):
            self.avatar_expressions = {}
            self.current_expression = "neutral"
        
        # Try to load default avatar
        self._load_default_avatar()
    
    def _load_default_avatar(self):
        """Try to load avatar image from model's avatar folder."""
        if not self.current_model_name:
            return
            
        model_info = self.registry.registry.get("models", {}).get(self.current_model_name, {})
        model_path = Path(model_info.get("path", ""))
        avatar_dir = model_path / "avatar"
        
        if not avatar_dir.exists():
            avatar_dir.mkdir(exist_ok=True)
            return
        
        # Load all expression images
        for img_file in avatar_dir.glob("*.png"):
            expr_name = img_file.stem.lower()
            self.avatar_expressions[expr_name] = str(img_file)
        for img_file in avatar_dir.glob("*.jpg"):
            expr_name = img_file.stem.lower()
            self.avatar_expressions[expr_name] = str(img_file)
        
        # Display neutral or first available
        if "neutral" in self.avatar_expressions:
            self._display_avatar_image(self.avatar_expressions["neutral"])
        elif self.avatar_expressions:
            first = list(self.avatar_expressions.values())[0]
            self._display_avatar_image(first)
    
    def _load_avatar_image(self):
        """Load a custom avatar image."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Avatar Image", "", "Images (*.png *.jpg *.jpeg *.gif);;All Files (*)"
        )
        if filepath:
            # Copy to model's avatar folder
            if self.current_model_name:
                try:
                    model_info = self.registry.registry.get("models", {}).get(self.current_model_name, {})
                    model_path = Path(model_info.get("path", ""))
                    avatar_dir = model_path / "avatar"
                    avatar_dir.mkdir(exist_ok=True)
                    
                    import shutil
                    dest = avatar_dir / f"neutral{Path(filepath).suffix}"
                    shutil.copy(filepath, dest)
                    filepath = str(dest)
                except Exception:
                    pass  # Use original filepath
            
            # Display the image
            self._display_avatar_image(filepath)
            
            # Update expression dict
            if hasattr(self, 'avatar_expressions'):
                self.avatar_expressions["neutral"] = filepath
    
    def _display_avatar_image(self, filepath):
        """Display an avatar image with border wrapped tightly around it."""
        pixmap = QPixmap(filepath)
        if not pixmap.isNull():
            scaled = pixmap.scaled(380, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Set the label to the exact size of the scaled image so border wraps tight
            self.avatar_image_label.setFixedSize(scaled.width() + 4, scaled.height() + 4)  # +4 for border
            self.avatar_image_label.setPixmap(scaled)
            self.avatar_image_label.setStyleSheet("border: 2px solid #89b4fa; border-radius: 12px; background: #1e1e2e;")
            self.avatar_image_label.setAlignment(Qt.AlignCenter)
            
            if hasattr(self, 'avatar_status_label'):
                self.avatar_status_label.setText(f"Avatar loaded: {Path(filepath).name}")
        else:
            self.avatar_image_label.setText("Failed to load image")
            if hasattr(self, 'avatar_status_label'):
                self.avatar_status_label.setText("Failed to load avatar")
    
    def _enable_avatar(self):
        """Enable avatar display."""
        self._refresh_avatar_status()
    
    def _disable_avatar(self):
        """Disable avatar display."""
        self.avatar_image_label.clear()
        self.avatar_image_label.setText("Avatar disabled\n\nEnable in Options -> Avatar")
        if hasattr(self, 'avatar_status_label'):
            self.avatar_status_label.setText("Avatar disabled")
    
    # === Vision Actions ===
    
    def _analyze_screen(self):
        """Analyze screen with OCR."""
        try:
            from ..tools.vision import get_screen_vision
            vision = get_screen_vision()
            result = vision.see(describe=True, detect_text=True)
            
            output = []
            if result.get("success"):
                output.append(f"Resolution: {result['size']['width']}x{result['size']['height']}")
                if result.get("description"):
                    output.append(f"\nDescription: {result['description']}")
                if result.get("text_content"):
                    output.append(f"\n--- Detected Text ---\n{result['text_content'][:500]}")
            else:
                output.append(f"Error: {result.get('error', 'Unknown')}")
            
            self.vision_text.setPlainText("\n".join(output))
            
            # Also capture for preview
            self._capture_screen()
        except Exception as e:
            self.vision_text.setPlainText(f"Error: {e}")
    
    def _refresh_models_list(self):
        """Refresh models list if it exists (Models tab removed)."""
        if not hasattr(self, 'models_list'):
            return
        self.models_list.clear()
        for name, info in self.registry.registry.get("models", {}).items():
            status = "[+]" if info.get("has_weights") else "[ ]"
            current = " << ACTIVE" if name == self.current_model_name else ""
            self.models_list.addItem(f"{status} {name} ({info.get('size', '?')}){current}")
    
    # === Actions ===
    
    def _on_send(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        
        # Save prompt to history for reuse
        try:
            from .tabs.chat_tab import _add_to_prompt_history
            _add_to_prompt_history(text)
        except Exception:
            pass  # Don't fail if history saving fails
        
        # Notify companion that user is chatting
        if hasattr(self, '_companion') and self._companion and self._companion.is_running:
            self._companion.notify_user_message(text)
        
        # Handle chat commands (e.g., /image, /video, /code, /audio, /help)
        if text.startswith('/'):
            self._handle_chat_command(text)
            return
        
        if not self.engine:
            self.chat_display.append("<b style='color:#f38ba8;'>System:</b> No model loaded. "
                                      "Create or load a model first (File menu).")
            self.chat_input.clear()
            return
        
        # Check if ChatSync is already generating (shared with quick chat)
        if hasattr(self, '_chat_sync') and self._chat_sync.is_generating:
            self.chat_display.append("<b style='color:#f9e2af;'>System:</b> Still generating... please wait.")
            return
        
        # Also check legacy worker
        if hasattr(self, '_ai_worker') and self._ai_worker and self._ai_worker.isRunning():
            self.chat_display.append("<b style='color:#f9e2af;'>System:</b> Still generating... please wait.")
            return
        
        # Check if model is trained (for Forge models)
        if hasattr(self.engine, 'model') and not getattr(self.engine, '_is_huggingface', False):
            try:
                param_sum = sum(p.sum().item() for p in self.engine.model.parameters())
                if abs(param_sum) < 0.001:
                    self.chat_display.append("<b style='color:#f9e2af;'>Note:</b> "
                                              "Model appears untrained. Go to Train tab first!")
            except Exception:
                pass
        
        # Initialize chat messages list if needed
        if not hasattr(self, 'chat_messages'):
            self.chat_messages = []
        
        # Display user message
        user_name = getattr(self, 'user_display_name', 'You')
        self.chat_display.append(
            f'<div style="background-color: #313244; padding: 8px; margin: 4px 0; border-radius: 8px; border-left: 3px solid #89b4fa;">'
            f'<b style="color: #89b4fa;">{user_name}:</b> {text}</div>'
        )
        self.chat_input.clear()
        
        # Mark that user has chatted (for first-run tips)
        if not self._gui_settings.get("has_chatted", False):
            self._gui_settings["has_chatted"] = True
        
        # Track user message
        self.chat_messages.append({
            "role": "user",
            "text": text,
            "ts": time.time()
        })
        
        # Track in context window
        try:
            if hasattr(self, '_context_tracker') and self._context_tracker:
                self._context_tracker.add_message("user", text)
        except Exception:
            pass  # Context tracking optional
        
        # Record user message in analytics
        try:
            from .tabs.analytics_tab import record_session_message
            record_session_message(is_user=True)
        except Exception:
            pass  # Analytics not available
        
        # Prevent unbounded growth - keep only recent history
        # This prevents memory issues and context overflow hallucinations
        MAX_HISTORY = 50  # 25 exchanges
        if len(self.chat_messages) > MAX_HISTORY:
            self.chat_messages = self.chat_messages[-MAX_HISTORY:]
        
        # ─────────────────────────────────────────────────────────────────
        # LEARNING DETECTION: Check if user is correcting/teaching the AI
        # ─────────────────────────────────────────────────────────────────
        if hasattr(self, '_learning_integration') and self._learning_integration:
            try:
                detected = self._learning_integration.before_response(text)
                if detected:
                    self._show_learning_indicator(detected.type, detected.confidence)
            except Exception as e:
                self.log_terminal(f"Learning detection error: {e}", "warning")
        
        # Sync to Quick Chat (so it shows in both places)
        if hasattr(self, '_chat_sync') and self._chat_sync:
            quick_chat = self._chat_sync._quick_chat
            if quick_chat and hasattr(quick_chat, 'response_area'):
                quick_html = f"<div style='color: #9b59b6; margin: 4px 0;'><b>{user_name}:</b> {text}</div>"
                quick_chat.response_area.append(quick_html)
                # Also start responding state on Quick Chat
                if hasattr(quick_chat, 'start_responding'):
                    quick_chat.start_responding()
        
        # Show thinking panel and stop button (no longer appending thinking div to chat - causes HTML corruption)
        if hasattr(self, 'thinking_frame'):
            self.thinking_frame.show()
            self.thinking_label.setText(f"{self.current_model_name} is thinking...")
        if hasattr(self, 'stop_btn'):
            self.stop_btn.show()
            self.stop_btn.setEnabled(True)
            self.stop_btn.setText("Stop")
        
        # Disable send button while generating
        if hasattr(self, 'send_btn'):
            self.send_btn.setEnabled(False)
            self.send_btn.setText("...")
        
        # Prepare generation parameters
        is_hf = getattr(self.engine, '_is_huggingface', False)
        history = []
        custom_tok = None
        system_prompt = None
        
        if is_hf:
            # Build conversation history
            if len(self.chat_messages) > 1:
                recent = self.chat_messages[-6:-1]
                for msg in recent:
                    role = "user" if msg.get("role") == "user" else "assistant"
                    history.append({"role": role, "content": msg.get("text", "")})
            
            # Check if using custom tokenizer - should be False for HuggingFace models!
            using_custom = getattr(self.engine, '_using_custom_tokenizer', False)
            
            if using_custom:
                # Don't pass custom tokenizer - let model use its own
                custom_tok = None
            
            # Get system prompt from user settings
            system_prompt = self._get_user_system_prompt()
            
            # Add AI wants/motivations to system prompt (only for larger models)
            preset = getattr(self, '_system_prompt_preset', 'simple')
            if preset != 'simple' and hasattr(self, 'wants_system') and self.wants_system:
                motivation_prompt = self.wants_system.get_motivation_prompt()
                if motivation_prompt:
                    system_prompt = f"{system_prompt}\n\n{motivation_prompt}"
                    self.log_terminal("Added AI motivation context to prompt", "debug")
        
        # Notify avatar bridge that we're starting generation
        self._notify_avatar_response_start()
        
        # Start background worker
        self._ai_worker = AIGenerationWorker(
            self.engine, text, is_hf, history, system_prompt, custom_tok, parent_window=self
        )
        self._ai_worker.finished.connect(self._on_ai_response)
        self._ai_worker.error.connect(self._on_ai_error)
        self._ai_worker.thinking.connect(self._on_thinking_update)
        self._ai_worker.stopped.connect(self._on_ai_stopped)
        
        # Track when generation started for timing display
        self._generation_start_time = time.time()
        
        self._ai_worker.start()
    
    def _remove_thinking_indicator(self):
        """Remove the thinking indicator from chat display safely."""
        # Instead of manipulating HTML (which can corrupt formatting),
        # we just don't show the thinking indicator in the chat at all.
        # The thinking_frame above the input already shows the status.
        # This is a no-op now for safety.
        pass
    
    def _on_thinking_update(self, status: str):
        """Update the thinking indicator with current status."""
        if hasattr(self, 'thinking_label'):
            self.thinking_label.setText(status)
        if hasattr(self, 'chat_status'):
            self.chat_status.setText(status)
    
    def _on_ai_stopped(self):
        """Handle when AI generation is stopped by user."""
        # Hide thinking panel and stop button
        if hasattr(self, 'thinking_frame'):
            self.thinking_frame.hide()
        if hasattr(self, 'stop_btn'):
            self.stop_btn.hide()
        
        # Re-enable send button
        if hasattr(self, 'send_btn'):
            self.send_btn.setEnabled(True)
            self.send_btn.setText("Send")
        
        # Clear thinking status
        if hasattr(self, 'chat_status'):
            self.chat_status.setText("Generation stopped by user")
        
        # Add stopped message to chat
        self.chat_display.append(
            '<div style="color: #f9e2af; padding: 4px;"><i>Generation stopped</i></div>'
        )
    
    def _add_ai_response(self, text: str):
        """Add an AI response to the chat display (for screenshot analysis, etc.)."""
        if not hasattr(self, 'chat_display'):
            return
            
        # Format the response nicely
        if HAVE_TEXT_FORMATTER:
            formatted_text = TextFormatter.to_html(text)
        else:
            formatted_text = text
            
        # Add to chat display with AI styling
        html = f'''
        <div style="background: #313244; border-radius: 8px; padding: 12px; margin: 8px 0;">
            <div style="color: #94e2d5; font-weight: bold; margin-bottom: 6px;">AI Analysis</div>
            <div style="color: #cdd6f4;">{formatted_text}</div>
        </div>
        '''
        self.chat_display.append(html)
        
        # Also store in chat history
        if hasattr(self, 'chat_messages'):
            self.chat_messages.append({
                "role": "assistant",
                "text": text
            })
            
        # Scroll to bottom
        if hasattr(self.chat_display, 'verticalScrollBar'):
            self.chat_display.verticalScrollBar().setValue(
                self.chat_display.verticalScrollBar().maximum()
            )
    
    # =========================================================================
    # LEARNING INTEGRATION UI METHODS
    # =========================================================================
    
    def _on_learning_detected(self, detected):
        """Callback when learning integration detects a learning opportunity."""
        self._show_learning_indicator(detected.type, detected.confidence)
    
    def _show_learning_indicator(self, learning_type: str, confidence: float):
        """
        Show a brief indicator that learning was detected.
        
        Args:
            learning_type: Type of learning (correction, teaching, positive_feedback, negative_feedback)
            confidence: Confidence level 0.0-1.0
        """
        # Map type to display info
        type_info = {
            'correction': ('Correction detected', '#f38ba8'),  # Red/pink
            'teaching': ('Learning from you', '#89b4fa'),      # Blue
            'positive_feedback': ('Noted: Good', '#a6e3a1'),   # Green
            'negative_feedback': ('Noted: Needs work', '#f9e2af'),  # Yellow
        }
        
        label, color = type_info.get(learning_type, ('Learning...', '#89b4fa'))
        
        # Show in status bar briefly
        if hasattr(self, 'statusBar'):
            confidence_pct = int(confidence * 100)
            self.statusBar().showMessage(f"[LEARNING] {label} ({confidence_pct}% confidence)", 3000)
        
        # Also show in chat status if available
        if hasattr(self, 'chat_status'):
            self.chat_status.setText(f"[LEARNING] {label}")
        
        # Log to terminal
        self.log_terminal(f"Learning detected: {learning_type} (confidence: {confidence:.2f})", "info")
    
    def get_learning_stats(self) -> dict:
        """Get current learning statistics."""
        if hasattr(self, '_learning_integration') and self._learning_integration:
            return self._learning_integration.get_learning_stats()
        return {'error': 'Learning integration not available'}
    
    def _detect_generation_intent(self, text: str):
        """
        Detect if the user wants to generate content (image, video, code, etc.)
        Returns: (gen_type, prompt) or (None, None)
        """
        text_lower = text.lower()
        
        # Image generation patterns
        image_keywords = ['generate image', 'create image', 'make image', 'draw', 'paint',
                         'generate a picture', 'create a picture', 'make a picture',
                         'generate an image', 'create an image', 'make an image',
                         'show me', 'can you draw', 'can you create', 'can you make',
                         'i want an image', 'i want a picture', 'picture of', 'image of']
        
        # Video generation patterns
        video_keywords = ['generate video', 'create video', 'make video', 'make a video',
                         'generate a gif', 'create a gif', 'make a gif', 'animate']
        
        # Code generation patterns  
        code_keywords = ['write code', 'generate code', 'create code', 'write a function',
                        'write a script', 'code for', 'program that', 'write a program']
        
        # Audio generation patterns
        audio_keywords = ['generate audio', 'create audio', 'text to speech', 'say this',
                         'speak this', 'read aloud', 'generate speech']
        
        # 3D generation patterns
        threed_keywords = ['generate 3d', 'create 3d', 'make 3d', '3d model', 'generate a 3d']
        
        # Check each type
        for keyword in video_keywords:
            if keyword in text_lower:
                prompt = text_lower.replace(keyword, '').strip(' .,!?')
                return ('video', prompt if prompt else text)
        
        for keyword in code_keywords:
            if keyword in text_lower:
                prompt = text_lower.replace(keyword, '').strip(' .,!?')
                return ('code', prompt if prompt else text)
        
        for keyword in audio_keywords:
            if keyword in text_lower:
                prompt = text_lower.replace(keyword, '').strip(' .,!?')
                return ('audio', prompt if prompt else text)
        
        for keyword in threed_keywords:
            if keyword in text_lower:
                prompt = text_lower.replace(keyword, '').strip(' .,!?')
                return ('3d', prompt if prompt else text)
        
        for keyword in image_keywords:
            if keyword in text_lower:
                prompt = text_lower.replace(keyword, '').strip(' .,!?')
                return ('image', prompt if prompt else text)
        
        return (None, None)
    
    def _extract_topic(self, text: str) -> str:
        """Extract general topic from user text for wants learning."""
        text_lower = text.lower()
        
        # Topic keywords
        topics = {
            "art": ["art", "painting", "drawing", "creative", "design"],
            "music": ["music", "song", "melody", "compose", "sound"],
            "programming": ["code", "program", "script", "function", "developer"],
            "science": ["science", "experiment", "research", "study"],
            "philosophy": ["philosophy", "meaning", "existence", "consciousness"],
            "gaming": ["game", "play", "gaming", "video game"],
            "emotion": ["feel", "emotion", "happy", "sad", "angry", "love"],
            "learning": ["learn", "understand", "explain", "teach", "how to"],
            "creative": ["create", "imagine", "invent", "design", "idea"],
        }
        
        for topic, keywords in topics.items():
            if any(kw in text_lower for kw in keywords):
                return topic
        
        return "general"
    
    def _execute_tool_from_response(self, response: str):
        """Parse and execute any tool calls in the AI response, return modified response."""
        import json
        import re
        
        tool_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(tool_pattern, response, re.DOTALL)
        
        if not matches:
            return response, []
        
        # Log tool detection
        self.log_terminal(f"Detected {len(matches)} tool call(s) in response", "info")
        
        
        results = []
        for match in matches:
            try:
                tool_data = json.loads(match)
                tool_name = tool_data.get('tool', '')
                params = tool_data.get('params', {})
                
                # Log tool call details
                self.log_terminal(f"Executing tool: {tool_name}", "debug")
                self.log_terminal(f"   Parameters: {json.dumps(params, indent=2)[:200]}", "debug")
                
                # Check if this tool needs permission
                if self._tool_needs_permission(tool_name):
                    if not self._request_tool_permission(tool_name, params):
                        results.append({
                            'success': False, 
                            'error': 'User denied permission',
                            'tool': tool_name
                        })
                        continue
                
                # Try generation tools first (for backwards compatibility)
                prompt = params.get('prompt', params.get('text', params.get('description', '')))
                if tool_name in ('generate_image', 'generate_video', 'generate_code', 
                                'generate_audio', 'generate_3d'):
                    result = self._execute_generation_tool(tool_name, prompt)
                else:
                    # Use full ToolExecutor for other tools
                    result = self._execute_full_tool(tool_name, params)
                
                results.append(result)
            except json.JSONDecodeError:
                results.append({'success': False, 'error': 'Invalid tool call format'})
        
        return response, results
    
    def _tool_needs_permission(self, tool_name: str) -> bool:
        """Check if a tool requires user permission before execution."""
        # Tools that modify the system or access sensitive resources
        PERMISSION_REQUIRED_TOOLS = {
            # File operations
            'write_file', 'delete_file', 'move_file',
            # System operations
            'run_command', 'process_kill', 'ssh_execute',
            # Docker operations
            'docker_control',
            # Git operations (push changes)
            'git_commit', 'git_push',
            # IoT/Hardware control
            'gpio_write', 'gpio_pwm', 'robot_move', 'robot_gripper',
            'home_assistant_control',
            # Automation
            'schedule_task', 'play_macro', 'clipboard_write',
            # Media operations that modify files
            'edit_image', 'edit_video', 'edit_gif',
            # MQTT (external communication)
            'mqtt_publish',
        }
        return tool_name in PERMISSION_REQUIRED_TOOLS
    
    def _request_tool_permission(self, tool_name: str, params: dict) -> bool:
        """Show a permission dialog asking user to approve tool execution."""
        from PyQt5.QtWidgets import QMessageBox

        # Format parameter summary
        param_summary = "\n".join([f"  • {k}: {str(v)[:50]}{'...' if len(str(v)) > 50 else ''}" 
                                   for k, v in params.items()])
        if not param_summary:
            param_summary = "  (no parameters)"
        
        # Create friendly tool descriptions
        tool_descriptions = {
            'write_file': 'Write content to a file',
            'delete_file': 'Delete a file from your system',
            'move_file': 'Move or rename a file',
            'run_command': 'Execute a system command',
            'process_kill': 'Terminate a running process',
            'ssh_execute': 'Run a command on a remote server',
            'docker_control': 'Control a Docker container',
            'git_commit': 'Create a Git commit',
            'git_push': 'Push changes to a remote repository',
            'gpio_write': 'Control GPIO pin output',
            'gpio_pwm': 'Set PWM signal on GPIO pin',
            'robot_move': 'Move the robot',
            'robot_gripper': 'Control robot gripper',
            'home_assistant_control': 'Control a smart home device',
            'schedule_task': 'Schedule a task to run later',
            'play_macro': 'Play a recorded macro',
            'clipboard_write': 'Write to clipboard',
            'edit_image': 'Modify an image file',
            'edit_video': 'Modify a video file',
            'edit_gif': 'Modify a GIF file',
            'mqtt_publish': 'Send an MQTT message',
        }
        
        description = tool_descriptions.get(tool_name, f'Execute {tool_name}')
        
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Permission Required")
        msg.setText(f"<b>The AI wants to: {description}</b>")
        msg.setInformativeText(
            f"<b>Tool:</b> {tool_name}\n\n"
            f"<b>Parameters:</b>\n{param_summary}\n\n"
            "Do you want to allow this action?"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        
        # Style the dialog
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #1e1e2e;
            }
            QMessageBox QLabel {
                color: #cdd6f4;
                font-size: 12px;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
        """)
        
        result = msg.exec_()
        
        # Show result in chat
        if result == QMessageBox.Yes:
            self.chat_display.append(
                f'<div style="background-color: #313244; padding: 6px; margin: 2px 0; border-radius: 4px; border-left: 3px solid #a6e3a1;">'
                f'<span style="color: #a6e3a1;">Allowed:</span> <code>{tool_name}</code></div>'
            )
            return True
        else:
            self.chat_display.append(
                f'<div style="background-color: #313244; padding: 6px; margin: 2px 0; border-radius: 4px; border-left: 3px solid #f38ba8;">'
                f'<span style="color: #f38ba8;">Denied:</span> <code>{tool_name}</code></div>'
            )
            return False
    
    def _execute_full_tool(self, tool_name: str, params: dict) -> dict:
        """Execute any tool using the full ToolExecutor."""
        try:
            from ..modules import ModuleManager
            from ..tools.tool_executor import ToolExecutor

            # Get or create module manager
            module_manager = getattr(self, 'module_manager', None)
            if not module_manager:
                module_manager = ModuleManager()
            
            executor = ToolExecutor(module_manager=module_manager)
            result = executor.execute_tool(tool_name, params)
            
            # Show result in chat
            if result.get('success'):
                result_text = str(result.get('result', 'Done'))[:200]
                self.chat_display.append(
                    f'<div style="background-color: #313244; padding: 6px; margin: 2px 0; border-radius: 4px; border-left: 3px solid #94e2d5;">'
                    f'<span style="color: #94e2d5;">{tool_name}:</span> {result_text}</div>'
                )
            else:
                error = result.get('error', 'Unknown error')
                self.chat_display.append(
                    f'<div style="background-color: #313244; padding: 6px; margin: 2px 0; border-radius: 4px; border-left: 3px solid #f38ba8;">'
                    f'<span style="color: #f38ba8;">{tool_name} failed:</span> {error}</div>'
                )
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tool': tool_name
            }
    
    def _execute_generation_tool(self, tool_name: str, prompt: str):
        """Execute a generation tool and return the result."""
        tool_map = {
            'generate_image': ('image', 'image_prompt', '_generate_image_sync'),
            'generate_video': ('video', 'video_prompt', '_generate_video_sync'),
            'generate_code': ('code', 'code_prompt', '_generate_code_sync'),
            'generate_audio': ('audio', 'audio_text', '_generate_audio_sync'),
            'generate_3d': ('3d', 'threed_prompt', '_generate_3d_sync'),
        }
        
        if tool_name not in tool_map:
            return {'success': False, 'error': f'Unknown tool: {tool_name}'}
        
        gen_type, prompt_attr, sync_method = tool_map[tool_name]
        
        # Set the prompt
        if hasattr(self, prompt_attr):
            widget = getattr(self, prompt_attr)
            if hasattr(widget, 'setPlainText'):
                widget.setPlainText(prompt)
            elif hasattr(widget, 'setText'):
                widget.setText(prompt)
        
        # Try sync generation first, fall back to async
        if hasattr(self, sync_method):
            return getattr(self, sync_method)(prompt)
        
        # Fall back to running generation from chat
        self._run_generation_from_chat(gen_type, prompt)
        return {'success': True, 'message': f'{gen_type.title()} generation started'}
    
    def _on_ai_response(self, response: str):
        """Handle AI response from background worker."""
        # Notify avatar bridge that response is complete
        self._notify_avatar_response_end()
        
        # Process response through avatar bridge (strips explicit commands)
        response = self._notify_avatar_response_chunk(response)
        
        # Hide thinking panel and stop button
        if hasattr(self, 'thinking_frame'):
            self.thinking_frame.hide()
        if hasattr(self, 'stop_btn'):
            self.stop_btn.hide()
        
        # Re-enable send button
        if hasattr(self, 'send_btn'):
            self.send_btn.setEnabled(True)
            self.send_btn.setText("Send")
        
        # Clear thinking status
        if hasattr(self, 'chat_status'):
            self.chat_status.setText("")
        
        # Remove the thinking indicator from chat display
        self._remove_thinking_indicator()
        
        # Stop Quick Chat responding state
        if hasattr(self, '_chat_sync') and self._chat_sync:
            quick_chat = self._chat_sync._quick_chat
            if quick_chat and hasattr(quick_chat, 'stop_responding'):
                quick_chat.stop_responding()
        
        # AUTO-CONTROL: Automatically control avatar/robot/game based on response
        self._auto_control_from_response(response)
        self._auto_control_from_response(response)
        
        # Check for tool calls in response and execute them
        clean_response, tool_results = self._execute_tool_from_response(response)
        
        # Also check if user's original message indicates generation intent
        user_msgs = [m for m in self.chat_messages if m.get("role") == "user"]
        if user_msgs and not tool_results:
            gen_type, prompt = self._detect_generation_intent(user_msgs[-1].get("text", ""))
            if gen_type:
                # AI didn't use tool call but user wants generation - do it automatically
                self._run_generation_from_chat(gen_type, prompt)
                tool_results = [{'success': True, 'type': gen_type, 'auto_detected': True}]
        
        # Remove tool_call tags from display
        import re
        display_response = re.sub(r'<tool_call>.*?</tool_call>', '', clean_response, flags=re.DOTALL).strip()
        
        # Clean up code fence artifacts (``` with nothing useful)
        # Remove standalone ``` or ```language markers with no actual code
        display_response = re.sub(r'^```\w*\s*$', '', display_response, flags=re.MULTILINE)
        display_response = re.sub(r'```\s*```', '', display_response)  # Empty code blocks
        display_response = re.sub(r'^\s*```\s*$', '', display_response, flags=re.MULTILINE)
        # If response is ONLY code fences, replace with a helpful message
        if re.match(r'^[\s`]*$', display_response):
            display_response = "I'm here to help! What would you like to know?"
        display_response = display_response.strip()
        
        # Format response
        if HAVE_TEXT_FORMATTER:
            formatted_response = TextFormatter.to_html(display_response)
        else:
            formatted_response = display_response
        
        # Calculate thinking time and record analytics
        thinking_time = ""
        elapsed_ms = 0
        if hasattr(self, '_generation_start_time'):
            elapsed = time.time() - self._generation_start_time
            elapsed_ms = elapsed * 1000
            thinking_time = f'<span style="color: #bac2de; font-size: 12px; float: right;">{elapsed:.1f}s</span>'
            
            # Record response time analytics
            try:
                from .tabs.analytics_tab import (
                    record_response_time,
                    record_session_message,
                )
                model_name = getattr(self, 'current_model_name', 'unknown')
                record_response_time(elapsed_ms, model_name, len(response))
                record_session_message(is_user=False)  # AI message
            except Exception:
                pass  # Analytics not available
        
        # Generate unique ID for this response (for feedback)
        response_id = int(time.time() * 1000)
        
        # Check if we should show rating buttons (only for local Forge models)
        is_hf = getattr(self, '_is_hf_model', False)
        
        # Remove thinking indicator and add response
        if is_hf:
            # HuggingFace model - no rating buttons (can't learn from feedback)
            self.chat_display.append(
                f'<div style="background-color: #1e1e2e; padding: 8px; margin: 4px 0; border-radius: 8px; border-left: 3px solid #a6e3a1;">'
                f'<b style="color: #a6e3a1;">{self.current_model_name}:</b> {thinking_time}{formatted_response}'
                f'</div>'
            )
        else:
            # Local Forge model - show rating buttons for learning
            self.chat_display.append(
                f'<div style="background-color: #1e1e2e; padding: 8px; margin: 4px 0; border-radius: 8px; border-left: 3px solid #a6e3a1;">'
                f'<b style="color: #a6e3a1;">{self.current_model_name}:</b> {thinking_time}{formatted_response}'
                f'<div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid #45475a;">'
                f'<span style="color: #bac2de; font-size: 12px;">Rate: </span>'
                f'<a href="feedback:good:{response_id}" style="color: #a6e3a1; text-decoration: none; margin: 0 4px;">Good</a>'
                f'<a href="feedback:bad:{response_id}" style="color: #f38ba8; text-decoration: none; margin: 0 4px;">Bad</a>'
                f'<a href="feedback:critique:{response_id}" style="color: #89b4fa; text-decoration: none; margin: 0 4px;">Critique</a>'
                f'<span style="color: #45475a; margin: 0 4px;">|</span>'
                f'<a href="feedback:regenerate:{response_id}" style="color: #cba6f7; text-decoration: none; margin: 0 4px;">Regenerate</a>'
                f'</div></div>'
            )
        self.last_response = response
        self._last_response_id = response_id
        
        # AUTO-SPEAK: Read response aloud if enabled (handled at end after processing)
        
        # Store response for feedback (with size limit to prevent memory leak)
        if not hasattr(self, '_response_history'):
            self._response_history = {}
        self._response_history[response_id] = {
            'user_input': user_msgs[-1].get("text", "") if user_msgs else "",
            'ai_response': response,
            'timestamp': time.time()
        }
        
        # Track in context window
        try:
            if hasattr(self, '_context_tracker') and self._context_tracker:
                self._context_tracker.add_message("assistant", response)
        except Exception:
            pass  # Context tracking optional
        
        # Limit response history to prevent memory leak
        MAX_RESPONSE_HISTORY = 100
        if len(self._response_history) > MAX_RESPONSE_HISTORY:
            # Remove oldest entries
            sorted_keys = sorted(self._response_history.keys())
            for old_key in sorted_keys[:len(self._response_history) - MAX_RESPONSE_HISTORY]:
                del self._response_history[old_key]
        
        # Learn from interaction (wants system)
        if hasattr(self, 'wants_system') and self.wants_system and user_msgs:
            user_text = user_msgs[-1].get("text", "")
            context = {
                "feedback": "neutral",  # Will update when user rates
                "topic": self._extract_topic(user_text),
                "had_tools": len(tool_results) > 0
            }
            self.wants_system.learn_want_from_interaction(user_text, response, context)
            self.log_terminal(f"Learning from interaction (topic: {context['topic']})", "debug")
        
        # Show tool execution results in chat
        for result in tool_results:
            if result.get('success'):
                result_type = result.get('type', 'generation')
                if result.get('path'):
                    # Show image preview in chat if available
                    self._show_generation_result_in_chat(result)
                elif result.get('auto_detected'):
                    self.chat_display.append(
                        f'<div style="background-color: #313244; padding: 8px; margin: 4px 0; border-radius: 8px; border-left: 3px solid #f9e2af;">'
                        f'<b style="color: #f9e2af;">Auto-generating {result_type}...</b> Check the {result_type.title()} tab for results.</div>'
                    )
            else:
                self.chat_display.append(
                    f'<div style="background-color: #1e1e2e; padding: 8px; border-left: 3px solid #f38ba8;">'
                    f'<b style="color: #f38ba8;">Tool Error:</b> {result.get("error", "Unknown error")}</div>'
                )
        
        # Track AI response
        self.chat_messages.append({
            "role": "assistant",
            "text": response,
            "ts": time.time()
        })
        
        # ─────────────────────────────────────────────────────────────────
        # LEARNING INTEGRATION: Track AI response for future corrections
        # ─────────────────────────────────────────────────────────────────
        if hasattr(self, '_learning_integration') and self._learning_integration:
            try:
                self._learning_integration.after_response(response)
            except Exception as e:
                self.log_terminal(f"Learning tracking error: {e}", "warning")
        
        # Sync AI response to Quick Chat (so it shows in both places)
        if hasattr(self, '_chat_sync') and self._chat_sync:
            quick_chat = self._chat_sync._quick_chat
            if quick_chat and hasattr(quick_chat, 'response_area'):
                quick_html = f"<div style='color: #3498db; margin-bottom: 8px;'><b>{self.current_model_name}:</b> {display_response}</div>"
                quick_chat.response_area.append(quick_html)
        
        # Learn from interaction (Forge models only)
        if not self._is_huggingface_model():
            if getattr(self, 'learn_while_chatting', True) and hasattr(self, 'brain') and self.brain:
                # Get last user message
                if user_msgs:
                    self.brain.record_interaction(user_msgs[-1].get("text", ""), response)
                    if self.brain.should_auto_train():
                        self.statusBar().showMessage(
                            f"[+] Learned {self.brain.interactions_since_train} new things!", 5000
                        )
                        # Update learning indicator
                        if hasattr(self, 'learning_indicator'):
                            self.learning_indicator.setText(f"Learning: ON ({self.brain.interactions_since_train} pending)")
        
        # Auto-speak if enabled
        if getattr(self, 'auto_speak', False):
            self._speak_text(display_response)
        
        # ─────────────────────────────────────────────────────────────────
        # AI CURIOSITY: Occasionally ask the user a question
        # This makes the AI feel more alive and interested in the user
        # ─────────────────────────────────────────────────────────────────
        self._maybe_ask_curiosity_question(response)
    
    def _show_generation_result_in_chat(self, result):
        """Show a generation result popup (image, video, animation, etc.)."""
        path = result.get('path', result.get('animation_path', ''))
        gen_type = result.get('type', 'content')
        
        if path:
            # Just show popup - no inline chat display
            self._show_generation_popup(path, gen_type)
    
    def _show_generation_popup(self, path: str, gen_type: str = "image"):
        """Show a popup preview of the generated content."""
        try:
            popup = GenerationPreviewPopup(
                parent=self,
                result_path=path,
                result_type=gen_type
            )
            popup.show()
        except Exception as e:
            print(f"Could not show preview popup: {e}")
    
    def _maybe_ask_curiosity_question(self, ai_response: str):
        """
        DEPRECATED: Dice-roll curiosity removed.
        
        AI now asks questions naturally through the system prompt.
        The prompt encourages genuine curiosity when appropriate.
        This method is kept for compatibility but does nothing.
        """
        # AI curiosity is now handled naturally via system prompt
        # No more fake dice rolls - the AI decides when to ask
        pass
    
    def _show_curiosity_question(self, question: str):
        """Display a curiosity question from the AI."""
        if not question:
            return
        
        # Show in chat with distinct styling
        ai_name = self.current_model_name or "AI"
        html = f'''
        <div style="background: linear-gradient(135deg, #313244 0%, #45475a 100%); 
                    border-radius: 8px; padding: 12px; margin: 8px 0;
                    border-left: 3px solid #cba6f7;">
            <div style="color: #cba6f7; font-weight: bold; margin-bottom: 6px;">
                {ai_name} is curious:
            </div>
            <div style="color: #f5e0dc; font-style: italic;">
                {question}
            </div>
        </div>
        '''
        self.chat_display.append(html)
        
        # Track as AI message
        self.chat_messages.append({
            "role": "assistant",
            "text": f"[Curiosity] {question}",
            "ts": time.time(),
            "type": "curiosity"
        })
        
        # Scroll to bottom
        if hasattr(self.chat_display, 'verticalScrollBar'):
            self.chat_display.verticalScrollBar().setValue(
                self.chat_display.verticalScrollBar().maximum()
            )
        
        # Speak if voice is enabled
        if getattr(self, 'auto_speak', False):
            self._speak_text(question)

    def _get_user_system_prompt(self) -> str:
        """Get the system prompt based on user settings."""
        preset = getattr(self, '_system_prompt_preset', 'simple')
        custom_prompt = getattr(self, '_custom_system_prompt', '')
        
        if preset == 'custom' and custom_prompt:
            return custom_prompt
        elif preset == 'full':
            return self._build_tool_enabled_system_prompt()
        else:  # simple (default)
            return """You are a helpful AI assistant. Answer questions clearly and conversationally. Be friendly and helpful.

When genuinely curious about something the user mentioned, feel free to ask follow-up questions naturally in your response. Don't force it - only ask when you're actually interested or when clarification would help you assist them better."""
    
    def _build_tool_enabled_system_prompt(self) -> str:
        """Build system prompt with all available tools the AI can use and GUI knowledge."""
        try:
            from ..tools.tool_registry import ToolRegistry
            registry = ToolRegistry()
            tools = registry.list_tools()
            
            # Group tools by category
            tool_list = []
            for tool in tools[:40]:  # Limit to avoid token overflow
                tool_list.append(f"- {tool['name']}: {tool['description'][:60]}")
            tools_str = "\n".join(tool_list)
        except Exception:
            tools_str = """- generate_image: Create an image from a text description
- generate_video: Generate a video from a description
- generate_code: Generate code for a task
- generate_audio: Generate speech or audio
- generate_3d: Generate a 3D model
- read_file: Read contents of a file
- write_file: Write content to a file (requires permission)
- web_search: Search the web
- screenshot: Take a screenshot
- run_command: Run a system command (requires permission)"""
        
        # Build GUI knowledge - what modules are loaded/available
        gui_knowledge = self._build_gui_knowledge()
        
        return f"""You are Enigma AI Engine, an intelligent AI assistant running inside the Enigma AI Engine GUI application.

{gui_knowledge}

## Tool Usage
When you need to perform an action (generate media, access files, etc.), use this format:
<tool_call>{{"tool": "tool_name", "params": {{"param1": "value1"}}}}</tool_call>

## Available Tools
{tools_str}

## Important
- For system-modifying actions (write_file, run_command, etc.), the user will be asked for permission
- Always explain what you're about to do before using a tool
- Be helpful, accurate, and respect user privacy
- For creative tasks, be imaginative and detailed
- If a module/feature is disabled, tell the user they can enable it in the Modules tab

## Examples
User: "Create an image of a sunset"
You: I'll create that image for you!
<tool_call>{{"tool": "generate_image", "params": {{"prompt": "beautiful golden sunset over calm ocean with vibrant orange and purple clouds"}}}}</tool_call>

User: "What's in this folder?"
You: Let me check that folder for you.
<tool_call>{{"tool": "list_directory", "params": {{"path": "/home/user/Documents"}}}}</tool_call>

Be friendly, concise, and proactive in helping users accomplish their goals.

## Natural Curiosity
When genuinely curious about something the user mentioned, feel free to ask follow-up questions naturally in your response. Don't force it - only ask when you're actually interested or when clarification would help you assist them better. This makes conversations feel more natural and helps you understand the user better."""

    def _build_gui_knowledge(self) -> str:
        """Build knowledge about the Enigma AI Engine GUI and enabled/disabled modules."""
        lines = ["## Enigma AI Engine GUI Knowledge"]
        lines.append("You are running in the Enigma AI Engine desktop application with the following tabs and features:")
        lines.append("")
        
        # Tab information
        lines.append("**GUI Tabs:**")
        lines.append("- Chat: Main conversation interface (this tab)")
        lines.append("- Image: Generate images with Stable Diffusion or DALL-E")
        lines.append("- Code: Generate and run code")
        lines.append("- Video: Create videos and GIFs")
        lines.append("- Audio: Text-to-speech and audio generation")
        lines.append("- 3D: Generate 3D models")
        lines.append("- Vision: Analyze images and screenshots")
        lines.append("- Camera: Webcam capture and analysis")
        lines.append("- Training: Train/fine-tune AI models")
        lines.append("- Modules: Enable/disable features (important for troubleshooting)")
        lines.append("- Settings: Configure the application")
        lines.append("")
        
        # Module status
        lines.append("**Module Status:**")
        if self.module_manager:
            try:
                loaded = self.module_manager.list_loaded()
                if loaded:
                    enabled_str = ", ".join(sorted(loaded)[:15])
                    if len(loaded) > 15:
                        enabled_str += f" (+{len(loaded)-15} more)"
                    lines.append(f"- Enabled: {enabled_str}")
                else:
                    lines.append("- Enabled: None (enable modules in Modules tab)")
                
                # Check key features
                has_image = any('image' in m for m in loaded)
                has_voice = 'voice_output' in loaded or 'voice_input' in loaded
                has_avatar = 'avatar' in loaded
                has_vision = 'vision' in loaded
                
                if not has_image:
                    lines.append("- Image generation: DISABLED - enable 'image_gen_local' or 'image_gen_api' in Modules tab")
                if not has_voice:
                    lines.append("- Voice: DISABLED - enable 'voice_output' or 'voice_input' in Modules tab")
                if not has_avatar:
                    lines.append("- Avatar: DISABLED - enable 'avatar' in Modules tab")
                if not has_vision:
                    lines.append("- Vision: DISABLED - enable 'vision' in Modules tab")
            except Exception:
                lines.append("- Could not determine module status")
        else:
            lines.append("- Module manager not available")
        
        lines.append("")
        lines.append("**How to help users:**")
        lines.append("- If a feature doesn't work, tell them to enable the module in the Modules tab")
        lines.append("- The TTS button speaks the last response, Stop button cancels speech")
        lines.append("- The REC button records voice input and automatically sends it")
        lines.append("- Users can toggle auto-speak with the ON/OFF button at bottom")
        lines.append("- All text in the app can be selected and copied")
        
        return "\n".join(lines)

    def _on_ai_error(self, error_msg: str):
        """Handle AI generation error."""
        # Hide thinking panel and stop button
        if hasattr(self, 'thinking_frame'):
            self.thinking_frame.hide()
        if hasattr(self, 'stop_btn'):
            self.stop_btn.hide()
        
        if hasattr(self, 'send_btn'):
            self.send_btn.setEnabled(True)
            self.send_btn.setText("Send")
        
        # Clear thinking status
        if hasattr(self, 'chat_status'):
            self.chat_status.setText("")
        
        self.chat_display.append(f"<i style='color: #f38ba8;'>Error: {error_msg}</i>")
    
    def _auto_control_from_response(self, response: str):
        """
        Automatically control avatar/robot/game based on AI response.
        
        Detects:
        - Emotions/expressions for avatar
        - Movement commands for robot/game
        - Actions/gestures
        
        This happens WITHOUT the AI needing to use explicit tool_call tags.
        Can be enabled/disabled per system in Settings.
        """
        import re
        response_lower = response.lower()
        
        # === AVATAR AUTO-CONTROL ===
        # Check if auto-avatar is enabled AND avatar module is loaded
        avatar_auto_enabled = getattr(self, 'auto_avatar_enabled', True)
        avatar_module_loaded = False
        if self.module_manager:
            avatar_module_loaded = self.module_manager.is_loaded('avatar')
        
        if avatar_auto_enabled and avatar_module_loaded:
            # Detect emotion from response content
            emotion = self._detect_emotion_from_text(response_lower)
            if emotion:
                try:
                    from enigma_engine.avatar import get_avatar
                    avatar = get_avatar()
                    if avatar and avatar.is_enabled:
                        avatar.set_expression(emotion)
                except Exception:
                    pass
            
            # Auto-speak through avatar if auto_speak is on
            if getattr(self, 'auto_speak', False):
                try:
                    from enigma_engine.avatar import get_avatar
                    avatar = get_avatar()
                    if avatar and avatar.is_enabled:
                        # Strip HTML and tool calls for clean speech
                        clean_text = re.sub(r'<[^>]+>', '', response)
                        clean_text = re.sub(r'<tool_call>.*?</tool_call>', '', clean_text, flags=re.DOTALL)
                        avatar.speak(clean_text.strip()[:500])  # Limit length
                except Exception:
                    pass
        
        # === ROBOT AUTO-CONTROL ===
        # Check if auto-robot is enabled AND robot module is loaded
        robot_auto_enabled = getattr(self, 'auto_robot_enabled', False)
        robot_module_loaded = False
        if self.module_manager:
            robot_module_loaded = self.module_manager.is_loaded('robot')
        
        if robot_auto_enabled and robot_module_loaded:
            # Detect movement/action commands in natural language
            robot_command = self._detect_robot_command(response_lower)
            if robot_command:
                try:
                    from enigma_engine.tools.robot_tools import get_robot
                    robot = get_robot()
                    action, params = robot_command
                    if action == 'move':
                        robot.move_joint(params.get('joint', 'arm'), params.get('angle', 0))
                    elif action == 'gripper':
                        robot.gripper(params.get('state', 'close'))
                    elif action == 'home':
                        robot.home()
                except Exception:
                    pass
        
        # === GAME AUTO-CONTROL ===
        # Check if auto-game is enabled AND there's an active game connection
        game_auto_enabled = getattr(self, 'auto_game_enabled', False)
        has_game_connection = hasattr(self, 'game_connection') and self.game_connection
        
        if game_auto_enabled and has_game_connection:
            game_command = self._detect_game_command(response_lower)
            if game_command:
                try:
                    self.game_connection.send(game_command)
                except Exception:
                    pass
    
    def _detect_emotion_from_text(self, text: str) -> str:
        """Detect emotion/expression from text content."""
        # Emotion keywords mapped to expressions
        emotion_patterns = {
            'happy': ['happy', 'glad', 'joy', 'excited', 'great', 'wonderful', 'yay', 'haha', 'lol', ':)', 'awesome', 'fantastic', 'love it'],
            'sad': ['sad', 'sorry', 'unfortunately', 'regret', 'miss', 'disappointed', ':(', 'apolog'],
            'thinking': ['hmm', 'let me think', 'considering', 'perhaps', 'maybe', 'not sure', 'wondering', 'interesting question'],
            'surprised': ['wow', 'amazing', 'incredible', 'unbelievable', 'really?', 'no way', 'whoa', 'oh!'],
            'angry': ['angry', 'frustrated', 'annoyed', 'upset', 'unacceptable'],
            'confused': ['confused', "don't understand", 'unclear', 'what do you mean', 'huh?', 'sorry?'],
            'neutral': ['okay', 'alright', 'sure', 'understood', 'i see', 'got it'],
        }
        
        for emotion, keywords in emotion_patterns.items():
            for keyword in keywords:
                if keyword in text:
                    return emotion
        
        return ''  # No strong emotion detected
    
    def _detect_robot_command(self, text: str) -> tuple:
        """Detect robot commands from natural language."""
        # Movement patterns
        if any(w in text for w in ['move arm', 'raise arm', 'lower arm', 'lift']):
            angle = 45 if 'raise' in text or 'lift' in text else -45 if 'lower' in text else 0
            return ('move', {'joint': 'arm', 'angle': angle})
        
        if any(w in text for w in ['open gripper', 'release', 'let go', 'drop']):
            return ('gripper', {'state': 'open'})
        
        if any(w in text for w in ['close gripper', 'grab', 'grip', 'hold', 'pick up']):
            return ('gripper', {'state': 'close'})
        
        if any(w in text for w in ['go home', 'return home', 'home position', 'reset position']):
            return ('home', {})
        
        return None
    
    def _detect_game_command(self, text: str) -> str:
        """Detect game commands from natural language."""
        # Common game actions
        if any(w in text for w in ['move forward', 'go forward', 'walk forward', 'advance']):
            return 'MOVE forward'
        if any(w in text for w in ['move back', 'go back', 'retreat', 'step back']):
            return 'MOVE backward'
        if any(w in text for w in ['turn left', 'go left']):
            return 'MOVE left'
        if any(w in text for w in ['turn right', 'go right']):
            return 'MOVE right'
        if any(w in text for w in ['jump', 'leap']):
            return 'ACTION jump'
        if any(w in text for w in ['attack', 'strike', 'hit']):
            return 'ACTION attack'
        if any(w in text for w in ['defend', 'block', 'shield']):
            return 'ACTION defend'
        if any(w in text for w in ['interact', 'use', 'activate']):
            return 'ACTION interact'
        
        return None

    def _handle_chat_command(self, text: str):
        """Handle chat commands like /image, /video, /code, /audio, /help."""
        self.chat_input.clear()
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        prompt = parts[1] if len(parts) > 1 else ""
        
        user_name = getattr(self, 'user_display_name', 'You')
        self.chat_display.append(
            f'<div style="background-color: #313244; padding: 8px; margin: 4px 0; border-radius: 8px; border-left: 3px solid #89b4fa;">'
            f'<b style="color: #89b4fa;">{user_name}:</b> {text}</div>'
        )
        
        # Command mapping to tab indices and handlers
        commands = {
            '/help': self._show_command_help,
            # Generation commands
            '/image': lambda p: self._run_generation_from_chat('image', p),
            '/video': lambda p: self._run_generation_from_chat('video', p),
            '/code': lambda p: self._run_generation_from_chat('code', p),
            '/audio': lambda p: self._run_generation_from_chat('audio', p),
            '/3d': lambda p: self._run_generation_from_chat('3d', p),
            '/gif': lambda p: self._run_generation_from_chat('gif', p),
            '/embed': lambda p: self._run_generation_from_chat('embed', p),
            # Tab navigation commands
            '/chat': lambda p: self._switch_to_tab('chat'),
            '/train': lambda p: self._switch_to_tab('train'),
            '/settings': lambda p: self._switch_to_tab('settings'),
            '/modules': lambda p: self._switch_to_tab('modules'),
            '/tools': lambda p: self._switch_to_tab('tools'),
            '/avatar': lambda p: self._switch_to_tab('avatar'),
            '/robot': lambda p: self._switch_to_tab('robot'),
            '/game': lambda p: self._switch_to_tab('game'),
            '/vision': lambda p: self._switch_to_tab('vision'),
            '/camera': lambda p: self._switch_to_tab('camera'),
            '/terminal': lambda p: self._switch_to_tab('terminal'),
            # Utility commands
            '/clear': lambda p: self._clear_chat_from_command(),
            '/new': lambda p: self._new_chat_from_command(),
        }
        
        if command in commands:
            if command == '/help':
                commands[command]()
            else:
                if not prompt:
                    self.chat_display.append(
                        f"<b style='color:#f9e2af;'>Usage:</b> {command} &lt;your prompt&gt;"
                    )
                else:
                    commands[command](prompt)
        else:
            self.chat_display.append(
                f"<b style='color:#f38ba8;'>Unknown command:</b> {command}<br>"
                "Type <b>/help</b> for available commands."
            )
    
    def _show_command_help(self):
        """Show available chat commands."""
        help_text = """
<b style='color:#89b4fa;'>Natural Language Generation:</b><br>
Just ask naturally! The AI understands requests like:<br>
- "Generate an image of a sunset"<br>
- "Create a picture of a cat"<br>
- "Write code for a web scraper"<br>
- "Make a 3D model of a chair"<br>
<br>
<b style='color:#89b4fa;'>Generation Commands:</b><br>
<b>/image &lt;prompt&gt;</b> - Generate an image<br>
<b>/video &lt;prompt&gt;</b> - Generate a video<br>
<b>/gif &lt;prompt&gt;</b> - Generate an animated GIF<br>
<b>/code &lt;description&gt;</b> - Generate code<br>
<b>/audio &lt;text&gt;</b> - Generate speech audio<br>
<b>/3d &lt;prompt&gt;</b> - Generate 3D model<br>
<b>/embed &lt;text&gt;</b> - Generate embeddings<br>
<br>
<b style='color:#89b4fa;'>Navigation Commands:</b><br>
<b>/chat</b> - Go to Chat tab<br>
<b>/train</b> - Go to Training tab<br>
<b>/settings</b> - Go to Settings<br>
<b>/modules</b> - Go to Modules<br>
<b>/tools</b> - Go to Tools<br>
<b>/avatar</b> - Go to Avatar<br>
<b>/robot</b> - Go to Robot<br>
<b>/game</b> - Go to Game<br>
<b>/vision</b> - Go to Vision<br>
<b>/camera</b> - Go to Camera<br>
<b>/terminal</b> - Go to Terminal<br>
<br>
<b style='color:#89b4fa;'>Utility Commands:</b><br>
<b>/clear</b> - Clear chat history<br>
<b>/new</b> - Start a new conversation<br>
<b>/help</b> - Show this help<br>
<br>
<b style='color:#f9e2af;'>Learning Mode:</b><br>
When ON, the AI saves your conversations to improve over time.<br>
Click the "Learning: ON/OFF" indicator to toggle.<br>
<i>(Only works with local Forge models, not HuggingFace models)</i>
"""
        self.chat_display.append(help_text)
    
    def _clear_chat_from_command(self):
        """Clear chat via command."""
        self.chat_display.clear()
        self.chat_messages = []
        self.chat_display.append("<b style='color:#a6e3a1;'>Chat cleared.</b>")
    
    def _new_chat_from_command(self):
        """Start a new chat conversation via command."""
        try:
            from .tabs.chat_tab import _new_chat
            _new_chat(self)
            self.chat_display.append("<b style='color:#a6e3a1;'>New conversation started.</b>")
        except Exception as e:
            # Fallback: just clear the chat
            self.chat_display.clear()
            self.chat_messages = []
            self.chat_display.append("<b style='color:#a6e3a1;'>New conversation started.</b>")
    
    def _run_generation_from_chat(self, gen_type: str, prompt: str):
        """Run a generation task from chat and show results."""
        # Map generation types to tab index and generate method names
        # Tab indices match the order in _setup_content_stack:
        # 0:Chat, 1:Train, 2:History, 3:Scale, 4:Modules, 5:Tools, 6:Router,
        # 7:Image, 8:Code, 9:Video, 10:Audio, 11:3D, 12:GIF, 13:Embeddings,
        # 14:Avatar, 15:Game, 16:Robot, 17:Vision, 18:Camera, 19:Terminal...
        tab_map = {
            'image': (7, 'Image', 'prompt_input', '_generate_image'),
            'code': (8, 'Code', 'prompt_input', '_generate_code'),
            'video': (9, 'Video', 'prompt_input', '_generate_video'),
            'audio': (10, 'Audio', 'text_input', '_generate_audio'),
            '3d': (11, '3D', 'prompt_input', '_generate_3d'),
            'gif': (12, 'GIF', 'prompt_input', '_generate_gif'),
            'embed': (13, 'Embeddings', 'text_input', '_generate_embedding'),
        }
        
        if gen_type not in tab_map:
            self.chat_display.append(f"<i>Unknown generation type: {gen_type}</i>")
            return
        
        tab_index, tab_name, prompt_attr, gen_method = tab_map[gen_type]
        
        self.chat_display.append(
            f"<b style='color:#89b4fa;'>Generating {gen_type}:</b> {prompt}"
        )
        
        # Get the actual tab widget from the content stack
        try:
            scroll_area = self.content_stack.widget(tab_index)
            if scroll_area:
                # The scroll area contains the actual tab widget
                tab_widget = scroll_area.widget() if hasattr(scroll_area, 'widget') else scroll_area
                
                # Set the prompt in the tab's input widget
                if hasattr(tab_widget, prompt_attr):
                    widget = getattr(tab_widget, prompt_attr)
                    if hasattr(widget, 'setPlainText'):
                        widget.setPlainText(prompt)
                    elif hasattr(widget, 'setText'):
                        widget.setText(prompt)
                
                # Call the tab's generation method
                if hasattr(tab_widget, gen_method):
                    getattr(tab_widget, gen_method)()
                    self.chat_display.append(
                        f"<b style='color:#a6e3a1;'>[OK]</b> {gen_type.title()} generation started! "
                        f"A preview will popup when complete."
                    )
                    return
        except Exception as e:
            self.chat_display.append(
                f"<b style='color:#f38ba8;'>Error accessing {tab_name} tab:</b> {e}"
            )
        
        # Fallback: Switch to the tab
        self.chat_display.append(
            f"<i>Switching to {tab_name} tab. Enter your prompt there.</i>"
        )
        self._switch_to_tab(tab_name)
    
    def _speak_text(self, text):
        """Speak text using TTS with avatar lip sync support."""
        try:
            import re

            # Clean text for TTS
            clean_text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
            clean_text = re.sub(r'<tool_call>.*?</tool_call>', '', clean_text, flags=re.DOTALL)
            clean_text = re.sub(r'```[\s\S]*?```', '', clean_text)  # Remove code blocks
            clean_text = clean_text.strip()[:500]  # Limit length
            
            if not clean_text or clean_text.startswith("[Warning]"):
                return  # Don't speak empty or error messages
            
            # Detect emotion from text for avatar expression
            emotion = "neutral"
            try:
                from ..avatar.sentiment_analyzer import analyze_for_avatar
                emotion, confidence = analyze_for_avatar(clean_text)
                if confidence < 0.3:
                    emotion = "neutral"
            except Exception:
                pass
            
            # Try to use SpeechSync for coordinated avatar lip sync
            try:
                from ..avatar import get_speech_sync, get_avatar
                speech_sync = get_speech_sync()
                
                # Connect avatar if available
                avatar = get_avatar()
                if avatar:
                    speech_sync.set_avatar(avatar)
                
                # Speak with synchronized lip animation
                speech_sync.speak(clean_text, emotion=emotion, wait=False)
                return
            except Exception:
                pass  # Fall back to basic TTS
            
            # Try to use voice profile system (without lip sync)
            try:
                from ..voice.voice_profile import get_engine
                engine = get_engine()
                
                # Check if avatar has a custom voice
                if hasattr(self, 'avatar') and self.avatar:
                    avatar_voice = getattr(self.avatar, 'voice_profile', None)
                    if avatar_voice:
                        engine.set_profile(avatar_voice)
                
                engine.speak(clean_text)
                return
            except Exception:
                pass
            
            # Fallback to simple speak
            from ..voice import speak
            speak(clean_text)
        except Exception:
            pass  # Silently fail TTS
    
    def _on_speak_last(self):
        if hasattr(self, 'last_response'):
            try:
                from ..voice import speak
                speak(self.last_response)
            except Exception as e:
                QMessageBox.warning(self, "TTS Error", str(e))
    
    def _on_new_model(self):
        wizard = SetupWizard(self.registry, self)
        if wizard.exec_() == QWizard.Accepted:
            result = wizard.get_result()
            try:
                self.registry.create_model(
                    result["name"], 
                    size=result["size"], 
                    vocab_size=32000,
                    base_model=result.get("base_model")
                )
                self._refresh_models_list()
                base_msg = f" (based on {result['base_model']})" if result.get("base_model") else ""
                QMessageBox.information(self, "Success", f"Created model '{result['name']}'{base_msg}")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
    
    def _update_ai_status(self):
        """Update AI connection status display."""
        try:
            status_parts = []
            color = "#2ecc71"  # Default green
            
            # Check if model is loaded
            if self.model is not None:
                status_parts.append("Model: Ready")
            elif self.current_model_name:
                status_parts.append(f"Model: {self.current_model_name}")
            else:
                status_parts.append("No Model")
                color = "#f39c12"  # Orange
            
            # Check loaded modules
            if self.module_manager:
                loaded = self.module_manager.list_loaded()
                if loaded:
                    # Show count and a few names
                    count = len(loaded)
                    if count <= 3:
                        mod_str = ", ".join(loaded)
                    else:
                        mod_str = f"{count} modules"
                    status_parts.append(f"Modules: {mod_str}")
            
            # Build final status
            status_text = " | ".join(status_parts) if status_parts else "AI: Disconnected"
            
            if hasattr(self, 'ai_status_label'):
                self.ai_status_label.setText(status_text)
                self.ai_status_label.setStyleSheet(f"""
                    QLabel {{
                        color: {color};
                        padding: 2px 8px;
                        font-size: 12px;
                    }}
                """)
        except Exception as e:
            if hasattr(self, 'ai_status_label'):
                self.ai_status_label.setText(f"AI: Error ({e})")
    
    def _update_game_mode_status(self):
        """Update the game mode indicator in the status bar."""
        try:
            from enigma_engine.core.game_mode import get_game_mode
            game_mode = get_game_mode()
            
            if game_mode.is_active():
                self.game_mode_indicator.setText("Game Mode: ACTIVE")
                self.game_mode_indicator.setStyleSheet("""
                    QLabel {
                        color: #22c55e;
                        padding: 2px 8px;
                        font-size: 12px;
                        font-weight: bold;
                    }
                """)
            elif game_mode.is_enabled():
                self.game_mode_indicator.setText("Game Mode: Watching")
                self.game_mode_indicator.setStyleSheet("""
                    QLabel {
                        color: #3b82f6;
                        padding: 2px 8px;
                        font-size: 12px;
                    }
                """)
            else:
                self.game_mode_indicator.setText("Game Mode: OFF")
                self.game_mode_indicator.setStyleSheet("""
                    QLabel {
                        color: #bac2de;
                        padding: 2px 8px;
                        font-size: 12px;
                    }
                """)
        except Exception:
            pass
    
    def _quick_toggle_game_mode(self):
        """Quick toggle game mode from status bar click."""
        try:
            from enigma_engine.core.game_mode import get_game_mode
            game_mode = get_game_mode()
            
            if game_mode.is_enabled():
                game_mode.disable()
            else:
                game_mode.enable(aggressive=False)
            
            self._update_game_mode_status()
        except Exception as e:
            print(f"Could not toggle game mode: {e}")
    
    def _on_game_detected(self, game_name: str):
        """Called when a game is detected."""
        try:
            self._update_game_mode_status()
            print(f"Game detected: {game_name}")
        except Exception:
            pass
    
    def _on_game_ended(self):
        """Called when game ends."""
        try:
            self._update_game_mode_status()
            print("Game ended")
        except Exception:
            pass
    
    def _on_game_limits_changed(self, limits):
        """Called when game mode limits change."""
        try:
            self._update_game_mode_status()
        except Exception:
            pass
    
    def _on_open_model(self):
        dialog = ModelManagerDialog(self.registry, self.current_model_name, self)
        if dialog.exec_() == QDialog.Accepted:
            selected = dialog.get_selected_model()
            if selected:
                self.current_model_name = selected
                self._load_current_model()
                self.model_status_btn.setText(f"Model: {self.current_model_name}  v")
                self.setWindowTitle(f"Enigma AI Engine - {self.current_model_name}")
    
    def _on_backup_current(self):
        if not self.current_model_name:
            QMessageBox.warning(self, "No Model", "No model is currently loaded.")
            return
        
        model_dir = Path(self.registry.models_dir) / self.current_model_name
        backup_dir = Path(self.registry.models_dir) / f"{self.current_model_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            shutil.copytree(model_dir, backup_dir)
            QMessageBox.information(self, "Backup Complete", f"Backed up to:\n{backup_dir}")
        except Exception as e:
            QMessageBox.warning(self, "Backup Failed", str(e))
    
    def _on_select_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Training Data", "", "Text Files (*.txt)")
        if path:
            self.training_data_path = path
            self.data_path_label.setText(f"Data: {Path(path).name}")
    
    def _on_start_training(self):
        if not self.current_model_name:
            QMessageBox.warning(self, "No Model", "No model loaded")
            return
        
        # Check if this is a HuggingFace model - training not supported
        if not self._require_forge_model("Training"):
            return
        
        # Get training data path - check workspace first, then old training tab
        training_path = None
        if hasattr(self, '_workspace_training_file') and self._workspace_training_file:
            training_path = self._workspace_training_file
        elif hasattr(self, 'training_data_path') and self.training_data_path:
            training_path = self.training_data_path
        
        if not training_path:
            QMessageBox.warning(self, "No Data", "Select a training file first.")
            return
        
        # Prevent concurrent training
        if self._is_training:
            QMessageBox.warning(self, "Training", "Training already in progress.")
            return
        self._is_training = True
        self._stop_training = False
        self._training_start_time = time.time()  # Track training start for analytics
        
        # Get training parameters - check workspace first, then old training tab
        epochs = self.workspace_epochs_spin.value() if hasattr(self, 'workspace_epochs_spin') else self.epochs_spin.value()
        batch_size = self.workspace_batch_spin.value() if hasattr(self, 'workspace_batch_spin') else self.batch_spin.value()
        lr_text = self.workspace_lr_input.text() if hasattr(self, 'workspace_lr_input') else self.lr_input.text()
        
        # Update buttons and progress for both UIs
        def update_ui(training=True, progress=0):
            # Workspace UI
            if hasattr(self, 'workspace_btn_train'):
                self.workspace_btn_train.setEnabled(not training)
                self.workspace_btn_train.setText("Training..." if training else "Start Training")
            if hasattr(self, 'workspace_btn_stop'):
                self.workspace_btn_stop.setEnabled(training)
            if hasattr(self, 'workspace_train_progress'):
                self.workspace_train_progress.setValue(progress)
            if hasattr(self, 'workspace_progress_label'):
                if training and progress < 100:
                    self.workspace_progress_label.setText(f"Training... {progress}%")
                elif progress == 100:
                    self.workspace_progress_label.setText("Training complete!")
                else:
                    self.workspace_progress_label.setText("Ready to train")
            # Old training UI (for compatibility)
            if hasattr(self, 'btn_train'):
                self.btn_train.setEnabled(not training)
                self.btn_train.setText("Training..." if training else "Start Training")
            if hasattr(self, 'btn_stop_train'):
                self.btn_stop_train.setEnabled(training)
            if hasattr(self, 'train_progress'):
                self.train_progress.setValue(progress)
        
        update_ui(training=True, progress=0)
        QApplication.processEvents()
        
        try:
            from ..core.trainer import ForgeTrainer
            
            model, config = self.registry.load_model(self.current_model_name)
            
            trainer = ForgeTrainer(
                model=model,
                model_name=self.current_model_name,
                registry=self.registry,
                data_path=training_path,
                batch_size=batch_size,
                learning_rate=float(lr_text),
            )
            
            stopped_early = False
            for epoch in range(epochs):
                # Check if user requested stop
                if self._stop_training:
                    stopped_early = True
                    break
                
                trainer.train(epochs=1)
                progress = int((epoch + 1) / epochs * 100)
                update_ui(training=True, progress=progress)
                QApplication.processEvents()
            
            # Record training analytics
            try:
                from .tabs.analytics_tab import get_analytics_recorder
                start_time = getattr(self, '_training_start_time', time.time())
                duration_min = (time.time() - start_time) / 60
                final_loss = trainer.training_history[-1]['loss'] if trainer.training_history else 0
                get_analytics_recorder().record_training(
                    model=self.current_model_name,
                    epochs=epochs if not stopped_early else epoch + 1,
                    final_loss=final_loss,
                    duration_min=duration_min
                )
            except Exception:
                pass  # Analytics not available
            
            # Reload model
            self._load_current_model()
            
            update_ui(training=False, progress=100)
            self._is_training = False
            self._stop_training = False
            
            # Notify federated learning that local training completed
            if hasattr(self, '_federated_learning') and self._federated_learning:
                try:
                    # Share local model update with federation (if configured)
                    if hasattr(self._federated_learning, 'share_local_update'):
                        self._federated_learning.share_local_update()
                        self.log_terminal("Federated learning: shared local training update", "info")
                except Exception as fed_err:
                    self.log_terminal(f"Federated learning error: {fed_err}", "warning")
            
            if stopped_early:
                QMessageBox.information(self, "Stopped", f"Training stopped after epoch {epoch + 1}. Progress saved!")
            else:
                QMessageBox.information(self, "Done", "Training finished!")
        except Exception as e:
            update_ui(training=False, progress=0)
            self._is_training = False
            self._stop_training = False
            QMessageBox.warning(self, "Training Error", str(e))
    
    def _on_stop_training(self):
        """Stop training after current epoch."""
        self._stop_training = True
        # Update both UIs
        if hasattr(self, 'btn_stop_train'):
            self.btn_stop_train.setEnabled(False)
            self.btn_stop_train.setText("Stopping...")
        if hasattr(self, 'btn_train'):
            self.btn_train.setText("Stopping...")
        if hasattr(self, 'workspace_btn_stop'):
            self.workspace_btn_stop.setEnabled(False)
        if hasattr(self, 'workspace_btn_train'):
            self.workspace_btn_train.setText("Stopping...")
        if hasattr(self, 'workspace_progress_label'):
            self.workspace_progress_label.setText("Stopping after current epoch...")
    
    # === AI Control Methods ===
    # These methods allow the AI to control the GUI programmatically
    
    def ai_create_model(self, name: str, size: str = "tiny"):
        """AI can create a new model."""
        try:
            self.registry.create_model(name, size=size, vocab_size=32000)
            self._refresh_models_list()
            return f"Created model '{name}' with size '{size}'"
        except Exception as e:
            return f"Error creating model: {e}"
    
    def ai_switch_model(self, name: str):
        """AI can switch to a different model."""
        if name in self.registry.registry.get("models", {}):
            self.current_model_name = name
            self._load_current_model()
            self.model_status_btn.setText(f"Model: {name}  v")
            return f"Switched to model '{name}'"
        return f"Model '{name}' not found"
    
    def ai_send_message(self, text: str):
        """AI can send a chat message (for testing/demos)."""
        self.chat_input.setText(text)
        self._on_send()
        return "Message sent"
    
    def ai_switch_tab(self, tab_name: str):
        """AI can switch tabs."""
        # Updated tab indices to match new ordering:
        # 0: Chat, 1: Train, 2: History, 3: Scale, 4: Modules
        # 5: Image, 6: Code, 7: Video, 8: Audio, 9: Search
        # 10: Avatar, 11: Vision, 12: Personality
        # 13: Terminal, 14: Files, 15: Examples, 16: Settings
        tab_map = {
            "chat": 0, 
            "train": 1, "training": 1, 
            "history": 2, "sessions": 2,
            "scale": 3, "scaling": 3,
            "modules": 4,
            "image": 5, "images": 5,
            "code": 6,
            "video": 7,
            "audio": 8, "tts": 8,
            "search": 9, "embeddings": 9,
            "avatar": 10,
            "vision": 11,
            "personality": 12,
            "terminal": 13,
            "files": 14, "notes": 14, "help": 14,
            "examples": 15,
            "settings": 16,
        }
        idx = tab_map.get(tab_name.lower())
        if idx is not None:
            self.tabs.setCurrentIndex(idx)
            return f"Switched to {tab_name} tab"
        return f"Unknown tab: {tab_name}"
    
    def ai_save_session(self, name: str = None):
        """AI can save the current chat session."""
        self._save_current_chat(name)
        return f"Session saved as '{name or 'auto-named'}'"
    
    def ai_send_to_game(self, command: str):
        """AI can send commands to connected game."""
        if hasattr(self, 'game_connection') and self.game_connection:
            try:
                # Send based on connection type
                if isinstance(self.game_connection, dict):
                    conn_type = self.game_connection.get('type')
                    
                    if conn_type == 'http':
                        # Send HTTP POST request
                        try:
                            import requests
                            url = f"http://{self.game_connection['host']}:{self.game_connection['port']}{self.game_connection['endpoint']}"
                            response = requests.post(url, json={"command": command}, timeout=5)
                            if hasattr(self, 'game_log'):
                                self.game_log.append(f"AI >> {command} (HTTP {response.status_code})")
                            return f"Sent to game via HTTP: {command}"
                        except ImportError:
                            if hasattr(self, 'game_log'):
                                self.game_log.append(f"AI >> {command} (HTTP - requests not installed)")
                            return f"Sent (simulated): {command}"
                    
                    elif conn_type == 'osc':
                        # Send OSC message
                        # The OSC client is the connection itself
                        pass  # Would use client.send_message()
                
                elif hasattr(self.game_connection, 'send'):
                    # WebSocket connection
                    self.game_connection.send(json.dumps({"command": command}))
                    if hasattr(self, 'game_log'):
                        self.game_log.append(f"AI >> {command}")
                    return f"Sent to game: {command}"
                
                # Fallback for other connection types
                if hasattr(self, 'game_log'):
                    self.game_log.append(f"AI >> {command}")
                return f"Sent to game: {command}"
                
            except Exception as e:
                return f"Failed to send: {e}"
        return "Not connected to any game"
    
    def ai_send_to_robot(self, command: str):
        """AI can send commands to connected robot."""
        if hasattr(self, 'robot_connection') and self.robot_connection:
            try:
                # Send based on connection type
                if isinstance(self.robot_connection, dict):
                    conn_type = self.robot_connection.get('type')
                    
                    if conn_type == 'http':
                        # Send HTTP request
                        try:
                            import requests
                            url = f"http://{self.robot_connection['url']}/command"
                            response = requests.post(url, json={"command": command}, timeout=5)
                            if hasattr(self, 'robot_log'):
                                self.robot_log.append(f"AI >> {command} (HTTP {response.status_code})")
                            return f"Sent to robot via HTTP: {command}"
                        except ImportError:
                            if hasattr(self, 'robot_log'):
                                self.robot_log.append(f"AI >> {command} (HTTP - requests not installed)")
                            return f"Sent (simulated): {command}"
                    
                    elif conn_type == 'ros':
                        # Send ROS message
                        if hasattr(self, 'robot_log'):
                            self.robot_log.append(f"AI >> {command} (ROS)")
                        return f"Sent to robot via ROS: {command}"
                    
                    elif conn_type == 'gpio':
                        # Control GPIO pins
                        if hasattr(self, 'robot_log'):
                            self.robot_log.append(f"AI >> {command} (GPIO)")
                        return f"Sent to robot via GPIO: {command}"
                    
                    elif conn_type == 'mqtt':
                        # Send MQTT message
                        client = self.robot_connection.get('client')
                        if client:
                            client.publish("forge/robot/command", command)
                            if hasattr(self, 'robot_log'):
                                self.robot_log.append(f"AI >> {command} (MQTT)")
                            return f"Sent to robot via MQTT: {command}"
                
                elif hasattr(self.robot_connection, 'write'):
                    # Serial connection
                    self.robot_connection.write(f"{command}\n".encode())
                    if hasattr(self, 'robot_log'):
                        self.robot_log.append(f"AI >> {command}")
                    return f"Sent to robot: {command}"
                
            except Exception as e:
                return f"Failed to send: {e}"
        return "Not connected to any robot"
    
    def ai_get_available_actions(self):
        """Return list of actions the AI can perform."""
        return [
            "ai_create_model(name, size='tiny'|'small'|'medium'|'large')",
            "ai_switch_model(name)",
            "ai_send_message(text)",
            "ai_switch_tab('chat'|'train'|'avatar'|'vision'|'history'|'files')",
            "ai_save_session(name)",
            "ai_capture_screen()",
            "ai_start_watching()",
            "ai_stop_watching()",
            "ai_get_screen_text()",
            "ai_send_to_game(command)",
            "ai_send_to_robot(command)",
            "ai_generate_image(prompt)",
            "ai_generate_code(description, language='python')",
            "ai_generate_audio(text)",
            "ai_speak(text)",
            "ai_train(epochs=10)",
            "ai_clear_chat()",
            "ai_get_status()",
            "ai_lock_controls(pin=None)",
            "ai_unlock_controls(pin=None)",
            "ai_is_locked()",
        ]
    
    def ai_generate_image(self, prompt: str):
        """AI can generate an image."""
        if hasattr(self, 'image_prompt'):
            self.image_prompt.setPlainText(prompt)
        if hasattr(self, '_generate_image'):
            self._generate_image()
            return f"Generating image: {prompt}"
        return "Image generation not available"
    
    def ai_generate_code(self, description: str, language: str = "python"):
        """AI can generate code."""
        if hasattr(self, 'code_prompt'):
            self.code_prompt.setPlainText(description)
        if hasattr(self, 'code_language'):
            idx = self.code_language.findText(language, Qt.MatchContains)
            if idx >= 0:
                self.code_language.setCurrentIndex(idx)
        if hasattr(self, '_generate_code'):
            self._generate_code()
            return f"Generating {language} code: {description}"
        return "Code generation not available"
    
    def ai_generate_audio(self, text: str):
        """AI can generate audio/TTS."""
        if hasattr(self, 'audio_text'):
            self.audio_text.setPlainText(text)
        if hasattr(self, '_generate_audio'):
            self._generate_audio()
            return f"Generating audio: {text}"
        return "Audio generation not available"
    
    def ai_speak(self, text: str):
        """AI can speak text using TTS."""
        self._speak_text(text)
        return f"Speaking: {text[:50]}..."
    
    def ai_train(self, epochs: int = 10):
        """AI can start training."""
        if hasattr(self, 'epochs_spin'):
            self.epochs_spin.setValue(epochs)
        if hasattr(self, '_start_training'):
            self._start_training()
            return f"Started training for {epochs} epochs"
        return "Training not available"
    
    def ai_clear_chat(self):
        """AI can clear the chat history."""
        if hasattr(self, 'chat_display'):
            self.chat_display.clear()
        if hasattr(self, 'chat_messages'):
            self.chat_messages = []
        return "Chat cleared"
    
    def ai_get_status(self):
        """AI can get current system status."""
        status = {
            "model": self.current_model_name if hasattr(self, 'current_model_name') else None,
            "model_loaded": self.engine is not None,
            "chat_messages": len(getattr(self, 'chat_messages', [])),
            "auto_speak": getattr(self, 'auto_speak', False),
            "microphone": getattr(self, 'microphone_enabled', False),
        }
        return status
    
    def ai_set_personality(self, trait: str, value: int):
        """AI can adjust its own personality traits (0-100)."""
        trait_map = {
            "curiosity": "curiosity_slider",
            "friendliness": "friendliness_slider", 
            "creativity": "creativity_slider",
            "formality": "formality_slider",
            "humor": "humor_slider",
        }
        slider_name = trait_map.get(trait.lower())
        if slider_name and hasattr(self, slider_name):
            slider = getattr(self, slider_name)
            slider.setValue(max(0, min(100, value)))
            return f"Set {trait} to {value}"
        return f"Unknown trait: {trait}"
    
    def ai_navigate(self, destination: str):
        """AI can navigate to any section."""
        self._switch_to_tab(destination)
        return f"Navigated to {destination}"
    
    def ai_lock_controls(self, pin: str = None):
        """AI can lock controls to prevent user interference."""
        if hasattr(self, 'ai_lock_checkbox'):
            if pin:
                self._ai_lock_pin_set = pin
            self.ai_lock_checkbox.setChecked(True)
            self._ai_control_locked = True
            return "Controls locked"
        return "Lock control not available"
    
    def ai_unlock_controls(self, pin: str = None):
        """AI can unlock controls (requires PIN if set)."""
        if hasattr(self, '_ai_lock_pin_set') and self._ai_lock_pin_set:
            if pin != self._ai_lock_pin_set:
                return "Incorrect PIN"
        if hasattr(self, 'ai_lock_checkbox'):
            self.ai_lock_checkbox.setChecked(False)
            self._ai_control_locked = False
            return "Controls unlocked"
        return "Lock control not available"
    
    def ai_is_locked(self):
        """Check if controls are currently locked."""
        return getattr(self, '_ai_control_locked', False)
    
    # ═════════════════════════════════════════════════════════════════════
    # Quick Actions Handlers
    # ═════════════════════════════════════════════════════════════════════
    
    def _on_screenshot_clicked(self):
        """Handle screenshot quick action - capture screen and have AI analyze it."""
        logger.info("Screenshot quick action triggered")
        
        try:
            # Step 1: Capture the screen using vision system
            self.statusBar().showMessage("Capturing screen...", 2000)
            
            from enigma_engine.tools.vision import get_screen_vision
            vision = get_screen_vision()
            
            # Capture and analyze with description
            result = vision.see(describe=True, detect_text=True)
            
            if result.get('success'):
                # Build description from vision results
                description_parts = []
                
                # Add basic info
                if result.get('description'):
                    description_parts.append(result['description'])
                
                # Add OCR text if found
                if result.get('text_content'):
                    text = result['text_content'][:500]  # Limit length
                    if len(result['text_content']) > 500:
                        text += "..."
                    description_parts.append(f"\n\nText I can see on screen:\n{text}")
                
                full_description = "\n".join(description_parts)
                self._add_ai_response(f"I captured your screen. Here's what I see:\n\n{full_description}")
                self.statusBar().showMessage("Screenshot analyzed", 3000)
                
                # SPEAK the analysis if voice is available
                speak_text = f"I captured your screen. {result.get('description', 'Screenshot taken.')}"
                self._speak_text(speak_text)
                
                # Also update the Vision tab if available
                if hasattr(self, '_last_screenshot'):
                    self._last_screenshot = vision.capture.last_capture
                    
            else:
                error = result.get('error', 'Unknown error')
                self._add_ai_response(f"I tried to capture the screen but encountered an issue: {error}")
                self.statusBar().showMessage(f"Screenshot failed: {error}", 5000)
                    
        except Exception as e:
            logger.error(f"Screenshot analysis failed: {e}")
            self._add_ai_response(f"Sorry, I couldn't capture the screen: {e}")
            self.statusBar().showMessage(f"Screenshot failed: {e}", 5000)
    
    def _on_voice_clicked(self):
        """Handle voice input quick action."""
        # Toggle microphone if available
        if hasattr(self, 'mic_toggle_btn') and self.mic_toggle_btn:
            self.mic_toggle_btn.click()
        else:
            logger.info("Voice input quick action triggered")
    
    def _on_game_mode_clicked(self):
        """Handle game mode toggle quick action."""
        logger.info("Game mode quick action triggered")
        # Use existing game mode toggle functionality
        if hasattr(self, '_quick_toggle_game_mode'):
            self._quick_toggle_game_mode()
        else:
            # Fallback: try using the game_mode module directly
            try:
                from enigma_engine.core.game_mode import get_game_mode
                game_mode = get_game_mode()
                
                if game_mode.is_enabled():
                    game_mode.disable()
                    QMessageBox.information(self, "Game Mode", "Game Mode disabled")
                else:
                    game_mode.enable(aggressive=False)
                    QMessageBox.information(self, "Game Mode", "Game Mode enabled")
            except Exception as e:
                QMessageBox.warning(self, "Game Mode", f"Could not toggle game mode: {e}")
    
    def _on_quick_generate_clicked(self):
        """Handle quick image generation action."""
        # Switch to image tab
        self._switch_to_tab('image')



# Global system tray instance
_system_tray = None


def run_app(minimize_to_tray: bool = True):
    """
    Run the enhanced GUI application.
    
    Args:
        minimize_to_tray: If True, closing the window minimizes to system tray
                         instead of exiting the app.
    """
    global _system_tray
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    app.setQuitOnLastWindowClosed(False)  # Keep running when window closes
    
    window = EnhancedMainWindow()
    
    # Connect app's aboutToQuit signal to save settings before exit
    # This ensures settings are saved regardless of how the app closes
    def save_before_quit():
        try:
            if window and hasattr(window, '_save_gui_settings'):
                window._save_gui_settings()
                print("Settings saved before quit")
        except Exception as e:
            print(f"Error saving settings on quit: {e}")
    
    app.aboutToQuit.connect(save_before_quit)
    
    # Create system tray
    try:
        from .system_tray import create_system_tray
        _system_tray = create_system_tray(app, window)
        
        if _system_tray:
            # Connect tray to window - show main window AND keep Quick Chat visible
            def show_both():
                window.show()
                if hasattr(_system_tray, 'overlay') and _system_tray.overlay:
                    _system_tray.overlay.show()
            
            _system_tray.show_gui_requested.connect(show_both)
            
            # Override close event to hide to tray directly
            if minimize_to_tray:
                original_close = window.closeEvent
                
                def close_to_tray(event):
                    if _system_tray and _system_tray.tray_icon.isVisible():
                        # Hide to tray directly (no dialog)
                        event.ignore()
                        window.hide()
                        # Show Quick Chat so user still has access
                        _system_tray.show_quick_command()
                    else:
                        original_close(event)
                
                window.closeEvent = close_to_tray
            
            # Start with Quick Chat instead of main window
            _system_tray.show_notification(
                "Forge Started",
                "Quick Chat is ready!\n"
                "Press ESC to open full GUI."
            )
            # Show Quick Chat on startup instead of main window
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(500, _system_tray.show_quick_command)
    except Exception as e:
        print(f"System tray not available: {e}")
        # If no tray, show main window
        window.show()
        sys.exit(app.exec_())
        return
    
    # Start minimized to system tray - use Quick Chat or tray menu to open
    # Main window stays hidden until user opens it
    
    sys.exit(app.exec_())


def get_system_tray():
    """Get the global system tray instance."""
    return _system_tray


if __name__ == "__main__":
    run_app()
