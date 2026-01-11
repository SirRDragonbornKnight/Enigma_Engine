"""
Enhanced PyQt5 GUI for Enigma with Setup Wizard

Features:
  - First-run setup wizard to create/name your AI
  - Model selection and management
  - Backup before risky operations
  - Grow/shrink models with confirmation
  - Chat, Training, Voice integration
  - Dark/Light/Shadow/Midnight mode toggle
  - Avatar control panel
  - Screen vision preview with camera support
  - Training data editor
  - Per-AI conversation history
  - Multi-AI support (run multiple models)
  - Image upload in chat/vision tabs
  - Selectable (read-only) text throughout
"""
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QLabel, QListWidget, QTabWidget, QFileDialog, QMessageBox, QDialog, QComboBox,
    QRadioButton, QButtonGroup, QDialogButtonBox, QWizard, QWizardPage, QFormLayout,
    QInputDialog, QActionGroup, QGroupBox, QGridLayout, QSplitter, QWidget,
    QStackedWidget, QScrollArea, QListWidgetItem, QFrame, QSizePolicy, QProgressBar,
    QTextEdit
)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QIcon
import time


# === AI Generation Worker Thread ===
class AIGenerationWorker(QThread):
    """Background worker for AI generation to keep GUI responsive."""
    finished = pyqtSignal(str)  # Emits the response
    error = pyqtSignal(str)     # Emits error message
    thinking = pyqtSignal(str)  # Emits thinking/reasoning status
    stopped = pyqtSignal()      # Emits when stopped by user
    
    def __init__(self, engine, text, is_hf, history=None, system_prompt=None, custom_tokenizer=None):
        super().__init__()
        self.engine = engine
        self.text = text
        self.is_hf = is_hf
        self.history = history
        self.system_prompt = system_prompt
        self.custom_tokenizer = custom_tokenizer
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
            
            if self._stop_requested:
                self.stopped.emit()
                return
            
            if self.is_hf:
                # HuggingFace model - show reasoning steps
                self.thinking.emit("Building conversation context...")
                time.sleep(0.1)
                
                if self._stop_requested:
                    self.stopped.emit()
                    return
                
                self.thinking.emit("Processing with language model...")
                
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
                        max_new_tokens=150,
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
                                    "⚠️ Model returned raw tensor data. This usually means:\n"
                                    "• The model is not properly configured for text generation\n"
                                    "• Try a different model or check if it needs fine-tuning\n"
                                    "• Local Enigma models need training first"
                                )
                    except Exception as decode_err:
                        response = f"⚠️ Could not decode model output: {decode_err}"
            else:
                # Local Enigma model - show detailed reasoning
                self.thinking.emit("Formatting prompt for Q&A...")
                formatted_prompt = f"Q: {self.text}\nA:"
                
                if self._stop_requested:
                    self.stopped.emit()
                    return
                
                self.thinking.emit("Running inference on local model...")
                response = self.engine.generate(formatted_prompt, max_gen=100)
                
                if self._stop_requested:
                    self.stopped.emit()
                    return
                
                self.thinking.emit("Cleaning up response...")
                
                # Check if response is a tensor
                if hasattr(response, 'shape') or 'tensor' in str(type(response)).lower():
                    response = (
                        "⚠️ Model returned raw data instead of text.\n"
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
        
        # Auto-close after 15 seconds (or click to close)
        from PyQt5.QtCore import QTimer
        self._auto_close_timer = QTimer(self)
        self._auto_close_timer.timeout.connect(self.close)
        self._auto_close_timer.start(15000)
        
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
        
        title_label = QLabel(f"✨ {title}")
        title_label.setStyleSheet("color: #a6e3a1; font-weight: bold; font-size: 14px; border: none;")
        header.addWidget(title_label)
        
        header.addStretch()
        
        close_btn = QPushButton("✕")
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
                preview_label.setText("🎲 Animation Generated")
            
            container_layout.addWidget(preview_label)
            
        elif self.result_type == "video" and self.result_path:
            # Video preview (thumbnail + play button)
            preview_label = QLabel()
            preview_label.setAlignment(Qt.AlignCenter)
            preview_label.setMinimumSize(400, 300)
            preview_label.setStyleSheet("border: 1px solid #45475a; border-radius: 8px; color: #cdd6f4;")
            preview_label.setText(f"🎬 Video Generated\n\nClick 'Open' to play")
            container_layout.addWidget(preview_label)
            
        elif self.result_type == "audio" and self.result_path:
            preview_label = QLabel()
            preview_label.setAlignment(Qt.AlignCenter)
            preview_label.setMinimumSize(300, 100)
            preview_label.setStyleSheet("border: 1px solid #45475a; border-radius: 8px; color: #cdd6f4;")
            preview_label.setText(f"🔊 Audio Generated\n\nClick 'Open' to play")
            container_layout.addWidget(preview_label)
            
        elif self.result_type == "3d" and self.result_path:
            preview_label = QLabel()
            preview_label.setAlignment(Qt.AlignCenter)
            preview_label.setMinimumSize(300, 100)
            preview_label.setStyleSheet("border: 1px solid #45475a; border-radius: 8px; color: #cdd6f4;")
            preview_label.setText(f"🎲 3D Model Generated\n\nClick 'Open' to view")
            container_layout.addWidget(preview_label)
        else:
            # Generic text
            preview_label = QLabel(f"Generated: {self.result_path}")
            preview_label.setStyleSheet("color: #cdd6f4; padding: 20px; border: none;")
            preview_label.setWordWrap(True)
            container_layout.addWidget(preview_label)
        
        # Path display
        path_label = QLabel(f"📁 {self.result_path}")
        path_label.setStyleSheet("color: #6c7086; font-size: 10px; border: none;")
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
                background-color: #45475a;
                color: #cdd6f4;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #585b70; }
        """)
        folder_btn.clicked.connect(self._open_folder)
        btn_layout.addWidget(folder_btn)
        
        container_layout.addLayout(btn_layout)
        
        # Hint
        hint = QLabel("Click anywhere to close • Auto-closes in 15s")
        hint.setStyleSheet("color: #6c7086; font-size: 9px; border: none;")
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
        import os
        import subprocess
        path = Path(self.result_path)
        if path.exists():
            if sys.platform == 'darwin':
                subprocess.run(['open', str(path)])
            elif sys.platform == 'win32':
                os.startfile(str(path))
            else:
                subprocess.run(['xdg-open', str(path)])
        self.close()
    
    def _open_folder(self):
        """Open the containing folder."""
        import subprocess
        path = Path(self.result_path)
        if path.exists():
            if sys.platform == 'darwin':
                subprocess.run(['open', '-R', str(path)])
            elif sys.platform == 'win32':
                subprocess.run(['explorer', '/select,', str(path)])
            else:
                subprocess.run(['xdg-open', str(path.parent)])
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
    background-color: #45475a;
    color: #6c7086;
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
    font-size: 16px;
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
    font-size: 13px;
    font-weight: 500;
    padding: 8px 0;
}
QListWidget#sidebar::item {
    padding: 10px 20px;
    border-left: 3px solid transparent;
    margin: 1px 8px;
    border-radius: 6px;
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
    background-color: #ccd0da;
    color: #9ca0b0;
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
    font-size: 16px;
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
    font-size: 13px;
    font-weight: 500;
    padding: 8px 0;
}
QListWidget#sidebar::item {
    padding: 10px 20px;
    border-left: 3px solid transparent;
    margin: 1px 8px;
    border-radius: 6px;
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
    background-color: #2a2a2a;
    color: #4a4a4a;
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
    font-size: 16px;
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
    color: #475569;
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
    font-size: 16px;
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

# Import enigma modules
try:
    from ..core.model_registry import ModelRegistry
    from ..core.model_config import MODEL_PRESETS
    from ..core.model_scaling import shrink_model
    from ..config import CONFIG
except ImportError:
    # Running standalone
    pass


class SetupWizard(QWizard):
    """First-run setup wizard for creating a new AI."""
    
    def __init__(self, registry: ModelRegistry, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.setWindowTitle("Enigma Setup Wizard")
        self.setWizardStyle(QWizard.ModernStyle)
        self.resize(600, 450)
        
        # Detect hardware BEFORE creating pages
        self.hw_profile = self._detect_hardware()
        
        # Add pages
        self.addPage(self._create_welcome_page())
        self.addPage(self._create_name_page())
        self.addPage(self._create_size_page())
        self.addPage(self._create_confirm_page())
        
        self.model_name = None
        self.model_size = self.hw_profile.get("recommended", "tiny")
    
    def _detect_hardware(self) -> dict:
        """Detect hardware capabilities for model size recommendations."""
        try:
            from ..core.hardware import HardwareProfile
            hw = HardwareProfile()
            profile = hw.profile
            
            ram_gb = profile.get("memory", {}).get("total_gb", 2)
            available_gb = profile.get("memory", {}).get("available_gb", 1)
            vram_gb = profile.get("gpu", {}).get("vram_gb", 0)
            is_pi = profile.get("platform", {}).get("is_raspberry_pi", False)
            is_mobile = profile.get("platform", {}).get("is_mobile", False)
            has_gpu = profile.get("gpu", {}).get("cuda_available", False)
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
        page.setTitle("Welcome to Enigma")
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
        if page_id == 3:  # Confirm page
            name = self.name_input.text().lower().strip().replace(" ", "_")
            
            checked = self.size_group.checkedButton()
            size = checked.size_id if checked else "small"
            
            # MODEL_PRESETS contains EnigmaConfig objects, not dicts
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
            
            self.confirm_label.setText(f"""
            <h3>Ready to Create Your AI</h3>
            <table>
                <tr><td><b>Name:</b></td><td>{name}</td></tr>
                <tr><td><b>Size:</b></td><td>{size}</td></tr>
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
        }


class ModelLoadingDialog(QDialog):
    """Loading dialog with animated progress bar and terminal output for model loading."""
    
    cancelled = False  # Flag to track cancellation
    
    def __init__(self, model_name: str, parent=None, show_terminal: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Loading Model")
        self.setFixedSize(450, 240)
        self.setModal(False)  # Non-modal so user can move the main window
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)  # Stay on top but movable
        self.cancelled = False
        self.show_terminal = show_terminal
        self._log_lines = []
        self._current_progress = 0
        self._target_progress = 0
        
        # For dragging the dialog
        self._drag_pos = None
        
        # Dark style
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                border: 2px solid #89b4fa;
                border-radius: 12px;
            }
            QLabel {
                color: #cdd6f4;
            }
            QProgressBar {
                background-color: #313244;
                border: none;
                border-radius: 8px;
                height: 18px;
                text-align: center;
                color: white;
                font-size: 11px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #89b4fa, stop:0.5 #74c7ec, stop:1 #89b4fa);
                border-radius: 8px;
            }
            QPushButton {
                background-color: #45475a;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #f38ba8;
            }
            QPushButton#terminal_btn {
                background-color: #313244;
            }
            QPushButton#terminal_btn:hover {
                background-color: #45475a;
            }
            QTextEdit {
                background-color: #11111b;
                color: #a6e3a1;
                border: 1px solid #45475a;
                border-radius: 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                padding: 4px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(10)
        
        # Title with animated emoji
        self.title_label = QLabel(f"⏳ Loading: {model_name}")
        self.title_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #89b4fa;")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Status label with activity dots
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("font-size: 12px; color: #a6adc8;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Progress bar with percentage text
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        layout.addWidget(self.progress)
        
        # Activity indicator (animated dots)
        self.activity_label = QLabel("●○○")
        self.activity_label.setStyleSheet("font-size: 14px; color: #74c7ec;")
        self.activity_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.activity_label)
        
        # Terminal output area (optional)
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setMaximumHeight(100)
        self.terminal.setVisible(show_terminal)
        layout.addWidget(self.terminal)
        
        # Buttons row
        btn_layout = QHBoxLayout()
        
        # Toggle terminal button
        self.terminal_btn = QPushButton("📺 Show Log")
        self.terminal_btn.setObjectName("terminal_btn")
        self.terminal_btn.clicked.connect(self._toggle_terminal)
        btn_layout.addWidget(self.terminal_btn)
        
        btn_layout.addStretch()
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(btn_layout)
        
        # Animation timer for activity dots
        self._dot_state = 0
        from PyQt5.QtCore import QTimer
        self._activity_timer = QTimer(self)
        self._activity_timer.timeout.connect(self._animate_dots)
        self._activity_timer.start(300)  # Update every 300ms
        
        # Smooth progress animation timer
        self._progress_timer = QTimer(self)
        self._progress_timer.timeout.connect(self._animate_progress)
        self._progress_timer.start(30)  # Smooth 30ms updates
    
    def _animate_dots(self):
        """Animate the activity indicator dots."""
        dots = ["●○○", "○●○", "○○●", "○●○"]
        self._dot_state = (self._dot_state + 1) % len(dots)
        self.activity_label.setText(dots[self._dot_state])
    
    def _animate_progress(self):
        """Smoothly animate progress bar to target value."""
        if self._current_progress < self._target_progress:
            # Ease towards target
            diff = self._target_progress - self._current_progress
            step = max(1, diff // 5)
            self._current_progress = min(self._current_progress + step, self._target_progress)
            self.progress.setValue(self._current_progress)
    
    def _toggle_terminal(self):
        """Toggle terminal visibility."""
        self.show_terminal = not self.show_terminal
        self.terminal.setVisible(self.show_terminal)
        if self.show_terminal:
            self.terminal_btn.setText("📺 Hide Log")
            self.setFixedSize(450, 340)
        else:
            self.terminal_btn.setText("📺 Show Log")
            self.setFixedSize(450, 240)
    
    def _on_cancel(self):
        """Handle cancel button click."""
        self.cancelled = True
        self.status_label.setText("Cancelling...")
        self.log("Cancelled by user")
        QApplication.processEvents()
    
    def is_cancelled(self) -> bool:
        """Check if loading was cancelled."""
        QApplication.processEvents()  # Allow UI to update
        return self.cancelled
    
    def log(self, text: str):
        """Add a log line to terminal output."""
        import time
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {text}"
        self._log_lines.append(line)
        self.terminal.append(f"<span style='color: #a6e3a1;'>{line}</span>")
        # Scroll to bottom
        self.terminal.verticalScrollBar().setValue(
            self.terminal.verticalScrollBar().maximum()
        )
        QApplication.processEvents()
    
    def set_status(self, text: str, progress: int):
        """Update status text and progress with smooth animation."""
        self.status_label.setText(text)
        self._target_progress = progress  # Animate towards this
        self.log(text)
        QApplication.processEvents()  # Force UI update
    
    def close(self):
        """Clean up timers before closing."""
        if hasattr(self, '_activity_timer'):
            self._activity_timer.stop()
        if hasattr(self, '_progress_timer'):
            self._progress_timer.stop()
        super().close()
    
    def mousePressEvent(self, event):
        """Handle mouse press for dragging."""
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging."""
        if event.buttons() == Qt.LeftButton and self._drag_pos:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release after dragging."""
        self._drag_pos = None


class ModelManagerDialog(QDialog):
    """Modern model manager dialog - manage, scale, backup, and organize models."""
    
    def __init__(self, registry: ModelRegistry, current_model: str = None, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.current_model = current_model
        self.selected_model = None
        
        self.setWindowTitle("Model Manager")
        self.setMinimumSize(700, 500)
        self.resize(800, 550)
        
        # Make dialog non-modal so it doesn't block
        self.setModal(False)
        
        self._build_ui()
        self._refresh_list()
        
        # Apply dark style to dialog
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QListWidget {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QListWidget::item:hover {
                background-color: #45475a;
            }
            QPushButton {
                background-color: #45475a;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #585b70;
            }
            QPushButton:pressed {
                background-color: #313244;
            }
            QPushButton:disabled {
                background-color: #313244;
                color: #6c7086;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #89b4fa;
                subcontrol-origin: margin;
                left: 12px;
            }
            QLabel {
                color: #cdd6f4;
            }
        """)
    
    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Left panel - Model list
        left_panel = QVBoxLayout()
        
        # Header with refresh
        header = QHBoxLayout()
        title = QLabel("Your Models")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        header.addWidget(title)
        header.addStretch()
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedSize(60, 32)
        refresh_btn.setToolTip("Refresh list")
        refresh_btn.clicked.connect(self._refresh_list)
        header.addWidget(refresh_btn)
        left_panel.addLayout(header)
        
        # Model list
        self.model_list = QListWidget()
        self.model_list.itemClicked.connect(self._on_select_model)
        self.model_list.itemDoubleClicked.connect(self._on_load_model)
        left_panel.addWidget(self.model_list)
        
        # Quick actions under list
        quick_btns = QHBoxLayout()
        
        new_btn = QPushButton("+ New")
        new_btn.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e;")
        new_btn.clicked.connect(self._on_new_model)
        quick_btns.addWidget(new_btn)
        
        load_btn = QPushButton("Load")
        load_btn.setStyleSheet("background-color: #89b4fa; color: #1e1e2e;")
        load_btn.clicked.connect(self._on_load_model)
        quick_btns.addWidget(load_btn)
        
        left_panel.addLayout(quick_btns)
        
        # HuggingFace Models Section
        hf_group = QGroupBox("HuggingFace Models")
        hf_layout = QVBoxLayout(hf_group)
        hf_layout.setSpacing(8)
        
        # Preset dropdown with model sizes and categories
        self.hf_preset_combo = QComboBox()
        self.hf_preset_combo.addItem("Select a preset model...")
        self.hf_preset_combo.addItem("microsoft/DialoGPT-small (162M) [Small] - Fast chat")
        self.hf_preset_combo.addItem("microsoft/DialoGPT-medium (405M) [Medium] - Conversational")
        self.hf_preset_combo.addItem("Salesforce/codegen-350M-mono (350M) [Small] - Code")
        self.hf_preset_combo.addItem("TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B) [Medium] - Fast chat")
        self.hf_preset_combo.addItem("Qwen/Qwen2-1.5B-Instruct (1.5B) [Medium] - Multilingual")
        self.hf_preset_combo.addItem("stabilityai/stablelm-2-zephyr-1_6b (1.6B) [Medium] - Stable chat")
        self.hf_preset_combo.addItem("google/gemma-2b-it (2B) [Medium] - Google Gemma")
        self.hf_preset_combo.addItem("mistralai/Mistral-7B-Instruct-v0.2 (7B) [Large] ⚠️GPU")
        self.hf_preset_combo.addItem("HuggingFaceH4/zephyr-7b-beta (7B) [Large] ⚠️GPU")
        self.hf_preset_combo.addItem("meta-llama/Llama-2-7b-chat-hf (7B) [Large] ⚠️GPU")
        self.hf_preset_combo.addItem("xai-org/grok-1 (314B) [Huge] ⚠️Datacenter")
        hf_layout.addWidget(self.hf_preset_combo)
        
        # Custom input
        hf_input_layout = QHBoxLayout()
        self.hf_model_input = QLineEdit()
        self.hf_model_input.setPlaceholderText("Or enter HuggingFace model ID...")
        hf_input_layout.addWidget(self.hf_model_input)
        
        self.hf_add_btn = QPushButton("Add")
        self.hf_add_btn.setStyleSheet("background-color: #fab387; color: #1e1e2e;")
        self.hf_add_btn.clicked.connect(self._on_add_hf_model)
        hf_input_layout.addWidget(self.hf_add_btn)
        hf_layout.addLayout(hf_input_layout)
        
        # Delete HF model button
        self.hf_delete_btn = QPushButton("🗑️ Delete Selected HF Model")
        self.hf_delete_btn.setStyleSheet("background-color: #f38ba8; color: #1e1e2e; font-weight: bold;")
        self.hf_delete_btn.setToolTip("Delete a HuggingFace model from this system")
        self.hf_delete_btn.clicked.connect(self._on_delete_hf_model)
        hf_layout.addWidget(self.hf_delete_btn)
        
        # Tokenizer option
        tokenizer_layout = QHBoxLayout()
        tokenizer_label = QLabel("Tokenizer:")
        tokenizer_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        tokenizer_layout.addWidget(tokenizer_label)
        
        self.hf_tokenizer_combo = QComboBox()
        self.hf_tokenizer_combo.addItem("Model's Own (Recommended)")
        self.hf_tokenizer_combo.addItem("Custom Enigma Tokenizer")
        self.hf_tokenizer_combo.setToolTip("Choose which tokenizer to use with HuggingFace models")
        self.hf_tokenizer_combo.setStyleSheet("font-size: 11px;")
        tokenizer_layout.addWidget(self.hf_tokenizer_combo)
        tokenizer_layout.addStretch()
        hf_layout.addLayout(tokenizer_layout)
        
        # Info label
        hf_info = QLabel("Note: Large models need good GPU & HF token for gated models")
        hf_info.setStyleSheet("color: #6c7086; font-size: 10px;")
        hf_info.setWordWrap(True)
        hf_layout.addWidget(hf_info)
        
        left_panel.addWidget(hf_group)
        
        layout.addLayout(left_panel, stretch=1)
        
        # Right panel - Details and actions
        right_panel = QVBoxLayout()
        
        # Model info card
        info_group = QGroupBox("Model Details")
        info_layout = QVBoxLayout(info_group)
        
        self.info_name = QLabel("Select a model")
        self.info_name.setStyleSheet("font-size: 18px; font-weight: bold; color: #f9e2af;")
        info_layout.addWidget(self.info_name)
        
        self.info_details = QLabel("Click a model from the list to see its details")
        self.info_details.setWordWrap(True)
        self.info_details.setStyleSheet("color: #a6adc8; font-size: 12px;")
        info_layout.addWidget(self.info_details)
        
        right_panel.addWidget(info_group)
        
        # Actions grouped
        actions_group = QGroupBox("Actions")
        actions_layout = QGridLayout(actions_group)
        actions_layout.setSpacing(8)
        
        # Row 1 - Safe actions
        self.btn_backup = QPushButton("Backup")
        self.btn_backup.setStyleSheet("background-color: #74c7ec; color: #1e1e2e; font-weight: bold;")
        self.btn_backup.clicked.connect(self._on_backup)
        self.btn_backup.setEnabled(False)
        actions_layout.addWidget(self.btn_backup, 0, 0)
        
        self.btn_clone = QPushButton("Clone")
        self.btn_clone.setStyleSheet("background-color: #cba6f7; color: #1e1e2e; font-weight: bold;")
        self.btn_clone.clicked.connect(self._on_clone)
        self.btn_clone.setEnabled(False)
        actions_layout.addWidget(self.btn_clone, 0, 1)
        
        self.btn_test = QPushButton("Test")
        self.btn_test.setStyleSheet("background-color: #94e2d5; color: #1e1e2e; font-weight: bold;")
        self.btn_test.clicked.connect(self._on_test_model)
        self.btn_test.setEnabled(False)
        self.btn_test.setToolTip("Test model with sample prompts")
        actions_layout.addWidget(self.btn_test, 0, 2)
        
        self.btn_folder = QPushButton("Folder")
        self.btn_folder.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; font-weight: bold;")
        self.btn_folder.clicked.connect(self._on_open_folder)
        self.btn_folder.setEnabled(False)
        actions_layout.addWidget(self.btn_folder, 0, 3)
        
        # Row 2 - Scaling
        self.btn_grow = QPushButton("Grow")
        self.btn_grow.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: bold;")
        self.btn_grow.clicked.connect(self._on_grow)
        self.btn_grow.setEnabled(False)
        actions_layout.addWidget(self.btn_grow, 1, 0)
        
        self.btn_shrink = QPushButton("Shrink")
        self.btn_shrink.setStyleSheet("background-color: #f9e2af; color: #1e1e2e; font-weight: bold;")
        self.btn_shrink.clicked.connect(self._on_shrink)
        self.btn_shrink.setEnabled(False)
        actions_layout.addWidget(self.btn_shrink, 1, 1)
        
        self.btn_rename = QPushButton("Rename")
        self.btn_rename.setStyleSheet("background-color: #94e2d5; color: #1e1e2e; font-weight: bold;")
        self.btn_rename.clicked.connect(self._on_rename)
        self.btn_rename.setEnabled(False)
        actions_layout.addWidget(self.btn_rename, 1, 2)
        
        # Row 3 - Delete buttons
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setStyleSheet("background-color: #f38ba8; color: #1e1e2e; font-weight: bold;")
        self.btn_delete.clicked.connect(lambda: self._on_delete_action(False))
        self.btn_delete.setEnabled(False)
        actions_layout.addWidget(self.btn_delete, 2, 0)
        
        self.btn_delete_backup = QPushButton("Delete (Keep Backup)")
        self.btn_delete_backup.setStyleSheet("background-color: #fab387; color: #1e1e2e; font-weight: bold;")
        self.btn_delete_backup.clicked.connect(lambda: self._on_delete_action(True))
        self.btn_delete_backup.setEnabled(False)
        actions_layout.addWidget(self.btn_delete_backup, 2, 1, 1, 2)  # Span 2 columns
        
        right_panel.addWidget(actions_group)
        
        # Close button at bottom
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        right_panel.addWidget(close_btn)
        
        layout.addLayout(right_panel, stretch=1)
    
    def _refresh_list(self):
        """Refresh the model list from disk."""
        try:
            self.registry._load_registry()
        except (IOError, OSError, json.JSONDecodeError):
            pass
        
        self.model_list.clear()
        self.selected_model = None
        self._update_buttons_state()
        self.info_name.setText("Select a model")
        self.info_details.setText("Click a model from the list to see its details")
        
        # Sync models to other tabs (like router)
        self._sync_models_everywhere()
        
        models = self.registry.registry.get("models", {})
        for name, info in sorted(models.items()):
            model_path = Path(self.registry.models_dir) / name
            if model_path.exists():
                has_weights = info.get("has_weights", False)
                size = info.get("size", "?")
                source = info.get("source", "enigma")
                
                # Different icons for different model sources
                if source == "huggingface":
                    icon = "[HF]"
                elif has_weights:
                    icon = "[OK]"
                else:
                    icon = "[--]"
                    
                self.model_list.addItem(f"{icon} {name} ({size})")
    
    def _update_buttons_state(self):
        """Enable/disable buttons based on selection."""
        has_selection = self.selected_model is not None
        self.btn_backup.setEnabled(has_selection)
        self.btn_clone.setEnabled(has_selection)
        self.btn_test.setEnabled(has_selection)
        self.btn_folder.setEnabled(has_selection)
        self.btn_rename.setEnabled(has_selection)
        self.btn_delete.setEnabled(has_selection)
        self.btn_delete_backup.setEnabled(has_selection)
        
        # Check if this is a HuggingFace model - they can't be resized
        is_huggingface = False
        if has_selection:
            model_info = self.registry.registry.get("models", {}).get(self.selected_model, {})
            is_huggingface = model_info.get("source") == "huggingface"
        
        # Disable grow/shrink for HuggingFace models (they have fixed architecture)
        can_scale = has_selection and not is_huggingface
        self.btn_grow.setEnabled(can_scale)
        self.btn_shrink.setEnabled(can_scale)
        
        # Update button tooltips to explain why they're disabled
        if is_huggingface:
            self.btn_grow.setToolTip("HuggingFace models cannot be resized")
            self.btn_shrink.setToolTip("HuggingFace models cannot be resized")
        else:
            self.btn_grow.setToolTip("Grow model to a larger size")
            self.btn_shrink.setToolTip("Shrink model to a smaller size")
    
    def _sync_models_everywhere(self):
        """Notify all components that model list has changed."""
        try:
            # Refresh router tab dropdowns
            if hasattr(self, 'router_tab') and self.router_tab:
                self.router_tab.refresh_models()
        except Exception:
            pass  # Don't crash if sync fails
    
    def _on_select_model(self, item):
        """Handle model selection."""
        text = item.text()
        # Parse "[OK] name (size)" or "[--] name (size)"
        parts = text.split(" ", 1)
        if len(parts) > 1:
            rest = parts[1]  # "name (size)"
            name = rest.rsplit(" (", 1)[0]
        else:
            name = text
        
        self.selected_model = name
        self._update_buttons_state()
        
        try:
            info = self.registry.get_model_info(name)
            meta = info.get("metadata", {})
            reg_info = info.get("registry", {})
            
            self.info_name.setText(f"{name}")
            
            created = str(meta.get('created', 'Unknown'))[:10]
            last_trained = meta.get('last_trained', 'Never')
            if last_trained and last_trained != 'Never':
                last_trained = str(last_trained)[:10]
            
            epochs = meta.get('total_epochs', 0)
            params = meta.get('estimated_parameters', 0)
            params_str = f"{params:,}" if params else "Unknown"
            checkpoints = len(info.get('checkpoints', []))
            size = reg_info.get('size', '?')
            source = reg_info.get('source', 'enigma')
            
            details = f"""
Source: {source.upper()}
Size: {size.upper()}
Parameters: {params_str}
Created: {created}
Last trained: {last_trained}
Total epochs: {epochs}
Checkpoints: {checkpoints}
            """.strip()
            
            # Add note for HuggingFace models
            if source == "huggingface":
                details += "\n\n[HuggingFace models cannot be resized]"
            
            self.info_details.setText(details)
        except Exception as e:
            self.info_details.setText(f"Error loading details:\n{e}")
    
    def _on_load_model(self, item=None):
        """Load the selected model."""
        if not self.selected_model:
            QMessageBox.warning(self, "No Selection", "Select a model first")
            return
        
        # Store selected model and close dialog
        self.accept()
    
    def _on_add_hf_model(self):
        """Add a HuggingFace model to the registry and tool router."""
        # Get model ID from preset or input
        preset_text = self.hf_preset_combo.currentText()
        custom_text = self.hf_model_input.text().strip()
        
        model_id = None
        if custom_text:
            model_id = custom_text
        elif preset_text and not preset_text.startswith("Select"):
            # Parse preset: "gpt2 (124M) - Fast, classic" -> "gpt2"
            model_id = preset_text.split(" (")[0].split(" - ")[0].strip()
        
        if not model_id:
            QMessageBox.warning(self, "No Model", "Select a preset or enter a HuggingFace model ID")
            return
        
        # Clean up the model_id
        model_id = model_id.strip()
        
        # Create a local name for the model
        local_name = model_id.replace("/", "_").replace("-", "_").lower()
        
        # Check if already exists
        if local_name in self.registry.registry.get("models", {}):
            # Ask if they want to just assign it to chat
            reply = QMessageBox.question(
                self, "Model Exists",
                f"'{local_name}' already exists in registry.\n\n"
                "Do you want to set it as the active chat AI?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._assign_hf_to_chat(model_id, local_name)
            return
        
        # Add to registry
        try:
            model_path = Path(self.registry.models_dir) / local_name
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Get tokenizer preference
            use_custom_tokenizer = self.hf_tokenizer_combo.currentIndex() == 1
            
            # Fetch model info (size, params) from HuggingFace
            size_str = "huggingface"
            num_params = 0
            try:
                from ..core.huggingface_loader import get_huggingface_model_info
                info = get_huggingface_model_info(model_id)
                if not info.get("error"):
                    size_str = f"HF-{info['size_str']}"  # e.g., "HF-124M"
                    num_params = info.get("num_parameters", 0)
            except Exception as e:
                print(f"Could not fetch HF model info: {e}")
            
            # Create registry entry
            self.registry.registry.setdefault("models", {})[local_name] = {
                "path": str(model_path),
                "size": size_str,
                "created": datetime.now().isoformat(),
                "has_weights": False,  # Weights are in HF cache, not local
                "source": "huggingface",
                "huggingface_id": model_id,
                "use_custom_tokenizer": use_custom_tokenizer,  # User preference
                "num_parameters": num_params,  # Store actual param count
            }
            self.registry._save_registry()
            
            # Assign to chat tool router
            self._assign_hf_to_chat(model_id, local_name)
            
            self._refresh_list()
            self.hf_model_input.clear()
            self.hf_preset_combo.setCurrentIndex(0)
            
            QMessageBox.information(
                self, "Model Added",
                f"Added HuggingFace model: {model_id}\n\n"
                f"Local name: {local_name}\n"
                f"Estimated size: {size_str}\n\n"
                "It has been set as the active chat AI.\n"
                "The model will download when first used."
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to add model: {e}")
    
    def _assign_hf_to_chat(self, model_id: str, local_name: str):
        """Assign a HuggingFace model to the chat tool router."""
        try:
            from ..core.tool_router import get_router
            router = get_router()
            
            # Format as huggingface:model_id
            full_id = f"huggingface:{model_id}"
            
            # Assign with high priority (assign_model already saves)
            router.assign_model("chat", full_id, priority=100)
        except Exception as e:
            print(f"Could not assign to router: {e}")
    
    def _on_delete_hf_model(self):
        """Delete a HuggingFace model from registry and optionally clear cache."""
        # Get all HF models from registry
        hf_models = []
        for name, info in self.registry.registry.get("models", {}).items():
            if info.get("source") == "huggingface":
                hf_id = info.get("huggingface_id", name)
                size = info.get("size", "unknown")
                hf_models.append((name, hf_id, size))
        
        if not hf_models:
            QMessageBox.information(self, "No HF Models", "No HuggingFace models found in registry.")
            return
        
        # Create selection dialog
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QListWidget, QDialogButtonBox, QCheckBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Delete HuggingFace Model")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)
        
        label = QLabel("Select HuggingFace model to delete:")
        label.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        layout.addWidget(label)
        
        list_widget = QListWidget()
        list_widget.setStyleSheet("""
            QListWidget { 
                background-color: #313244; 
                color: #cdd6f4; 
                border: 1px solid #45475a;
            }
            QListWidget::item:selected { background-color: #f38ba8; color: #1e1e2e; }
        """)
        for name, hf_id, size in hf_models:
            list_widget.addItem(f"{name} ({hf_id}) [{size}]")
        layout.addWidget(list_widget)
        
        # Option to clear HF cache
        cache_checkbox = QCheckBox("Also clear HuggingFace cache for this model")
        cache_checkbox.setToolTip("This will delete the downloaded model files from HuggingFace cache.\n"
                                   "Re-downloading will be needed if you add this model again.")
        cache_checkbox.setStyleSheet("color: #fab387;")
        layout.addWidget(cache_checkbox)
        
        # Warning
        warning = QLabel("⚠️ This cannot be undone!")
        warning.setStyleSheet("color: #f38ba8; font-weight: bold;")
        layout.addWidget(warning)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        selected = list_widget.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "No Selection", "Please select a model to delete.")
            return
        
        model_name, hf_id, _ = hf_models[selected]
        clear_cache = cache_checkbox.isChecked()
        
        # Confirm
        reply = QMessageBox.warning(
            self, "Confirm Delete",
            f"Delete HuggingFace model?\n\n"
            f"Registry name: {model_name}\n"
            f"HuggingFace ID: {hf_id}\n"
            f"Clear cache: {'Yes' if clear_cache else 'No'}\n\n"
            "This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            # Remove from registry
            if model_name in self.registry.registry.get("models", {}):
                del self.registry.registry["models"][model_name]
                self.registry._save_registry()
            
            # Delete local directory if exists
            from pathlib import Path
            model_path = Path(self.registry.models_dir) / model_name
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)
            
            # Clear HuggingFace cache if requested
            cache_msg = ""
            if clear_cache:
                try:
                    from huggingface_hub import scan_cache_dir, HfFolder
                    import os
                    
                    # Get HF cache directory
                    cache_dir = os.environ.get("HF_HOME", 
                                 os.environ.get("HUGGINGFACE_HUB_CACHE",
                                 os.path.expanduser("~/.cache/huggingface/hub")))
                    
                    if os.path.exists(cache_dir):
                        # Look for model in cache
                        cache_info = scan_cache_dir(cache_dir)
                        for repo in cache_info.repos:
                            if repo.repo_id == hf_id:
                                # Delete all revisions of this model
                                delete_strategy = cache_info.delete_revisions(*[rev.commit_hash for rev in repo.revisions])
                                delete_strategy.execute()
                                cache_msg = "\n\nHuggingFace cache cleared."
                                break
                        else:
                            cache_msg = "\n\nModel not found in HF cache (may not have been downloaded)."
                except Exception as e:
                    cache_msg = f"\n\nCould not clear HF cache: {e}"
            
            # Clean up tool routing
            self._cleanup_tool_routing(model_name)
            self._cleanup_tool_routing(f"huggingface:{hf_id}")
            
            self._refresh_list()
            
            QMessageBox.information(
                self, "Deleted",
                f"HuggingFace model '{model_name}' has been deleted from registry.{cache_msg}"
            )
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to delete model: {e}")
    
    def _on_new_model(self):
        """Create a new model via wizard."""
        wizard = SetupWizard(self.registry, self)
        if wizard.exec_() == QWizard.Accepted:
            result = wizard.get_result()
            try:
                self.registry.create_model(
                    result["name"],
                    size=result["size"],
                    vocab_size=32000
                )
                self._refresh_list()
                QMessageBox.information(self, "Success", f"Created model '{result['name']}'")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
    
    def _on_backup(self):
        """Backup the selected model to a zip file."""
        if not self.selected_model:
            return
        
        name = self.selected_model
        model_dir = Path(self.registry.models_dir) / name
        
        # Create backup as a zip file in a backups folder (not as another model)
        backups_dir = Path(self.registry.models_dir) / "_backups"
        backups_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{name}_backup_{timestamp}"
        backup_zip = backups_dir / f"{backup_name}.zip"
        
        try:
            import zipfile
            with zipfile.ZipFile(backup_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in model_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(model_dir)
                        zf.write(file_path, arcname)
            
            QMessageBox.information(
                self, "Backup Complete", 
                f"Backup saved to:\n{backup_zip}\n\n"
                f"To restore, unzip to the models folder."
            )
        except Exception as e:
            QMessageBox.warning(self, "Backup Failed", str(e))
    
    def _on_test_model(self):
        """Test the selected model with sample prompts to verify it works."""
        if not self.selected_model:
            return
        
        name = self.selected_model
        
        # Create test dialog
        test_dialog = QDialog(self)
        test_dialog.setWindowTitle(f"Testing: {name}")
        test_dialog.setMinimumSize(500, 400)
        test_dialog.setStyleSheet("""
            QDialog { background-color: #1e1e2e; }
            QLabel { color: #cdd6f4; }
            QTextEdit { 
                background-color: #313244; 
                color: #cdd6f4; 
                border: 1px solid #45475a;
                border-radius: 6px;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #b4befe; }
        """)
        
        layout = QVBoxLayout(test_dialog)
        
        title = QLabel(f"Model Test: {name}")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        layout.addWidget(title)
        
        # Test results area
        results = QTextEdit()
        results.setReadOnly(True)
        layout.addWidget(results)
        
        # Buttons
        btn_layout = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(test_dialog.close)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        test_dialog.show()
        QApplication.processEvents()
        
        # Run the test
        results.append("<b>Loading model...</b>")
        QApplication.processEvents()
        
        try:
            # Load the model
            model, config = self.registry.load_model(name)
            results.append(f"<span style='color: #a6e3a1;'>✓ Model loaded successfully</span>")
            results.append(f"<span style='color: #6c7086;'>  Source: {config.get('source', 'local')}</span>")
            QApplication.processEvents()
            
            # Get model info
            is_huggingface = config.get("source") == "huggingface"
            
            if is_huggingface:
                results.append(f"<span style='color: #6c7086;'>  HuggingFace ID: {config.get('huggingface_id', 'unknown')}</span>")
            else:
                results.append(f"<span style='color: #6c7086;'>  Size: {config.get('size', 'unknown')}</span>")
            
            # Test prompts
            test_prompts = [
                "Hello",
                "What is 2 + 2?",
                "How are you?",
            ]
            
            results.append("\n<b>Running test prompts...</b>")
            QApplication.processEvents()
            
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            passed = 0
            failed = 0
            
            for prompt in test_prompts:
                results.append(f"\n<b>Prompt:</b> {prompt}")
                QApplication.processEvents()
                
                try:
                    if is_huggingface:
                        # HuggingFace model
                        response = model.generate(prompt, max_length=50)
                    else:
                        # Local Enigma model
                        model.to(device)
                        model.eval()
                        from ..core.tokenizer import load_tokenizer
                        tokenizer = load_tokenizer()
                        
                        tokens = tokenizer.encode(prompt)
                        input_ids = torch.tensor([tokens], device=device)
                        
                        with torch.no_grad():
                            for _ in range(30):
                                logits = model(input_ids)
                                next_token = logits[0, -1, :].argmax().item()
                                input_ids = torch.cat([
                                    input_ids,
                                    torch.tensor([[next_token]], device=device)
                                ], dim=1)
                                if next_token == tokenizer.eos_token_id:
                                    break
                        
                        response = tokenizer.decode(input_ids[0].tolist())
                    
                    # Check response quality
                    response_clean = response.replace(prompt, "").strip()[:100]
                    
                    if len(response_clean) < 2:
                        results.append(f"<span style='color: #f9e2af;'>⚠ Response too short: '{response_clean}'</span>")
                        failed += 1
                    elif response_clean.count(response_clean[0]) == len(response_clean):
                        results.append(f"<span style='color: #f38ba8;'>✗ Repetitive output: '{response_clean}'</span>")
                        failed += 1
                    else:
                        results.append(f"<span style='color: #a6e3a1;'>✓ Response: {response_clean}</span>")
                        passed += 1
                        
                except Exception as e:
                    results.append(f"<span style='color: #f38ba8;'>✗ Error: {e}</span>")
                    failed += 1
                
                QApplication.processEvents()
            
            # Summary
            results.append("\n<b>Test Summary:</b>")
            if failed == 0:
                results.append(f"<span style='color: #a6e3a1;'>All {passed} tests passed! Model looks good.</span>")
            elif passed > failed:
                results.append(f"<span style='color: #f9e2af;'>{passed} passed, {failed} failed. Model may need training.</span>")
            else:
                results.append(f"<span style='color: #f38ba8;'>{passed} passed, {failed} failed. Model needs training or has issues.</span>")
                results.append("<span style='color: #6c7086;'>Tip: Try training with more data or check if weights are corrupted.</span>")
                
        except Exception as e:
            results.append(f"<span style='color: #f38ba8;'>✗ Failed to load model: {e}</span>")
            results.append("<span style='color: #6c7086;'>Check that model files exist and are not corrupted.</span>")
    
    def _on_clone(self):
        """Clone the selected model."""
        if not self.selected_model:
            return
        
        original_name = self.selected_model  # Store before any changes
        
        from PyQt5.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(
            self, "Clone Model",
            f"Enter name for clone of '{original_name}':",
            text=f"{original_name}_clone"
        )
        
        if not ok or not new_name.strip():
            return
        
        new_name = new_name.strip().replace(' ', '_').lower()
        
        if new_name in self.registry.registry.get("models", {}):
            QMessageBox.warning(self, "Name Exists", f"'{new_name}' already exists")
            return
        
        try:
            src = Path(self.registry.models_dir) / original_name
            dst = Path(self.registry.models_dir) / new_name
            shutil.copytree(src, dst)
            
            # Create NEW registry entry with CORRECT path (not copy of old)
            old_info = self.registry.registry["models"][original_name]
            new_info = {
                "path": str(dst),  # NEW path for the clone!
                "size": old_info.get("size", "tiny"),
                "created": datetime.now().isoformat(),
                "has_weights": old_info.get("has_weights", False),
                "data_dir": str(dst / "data"),  # NEW data dir!
                "cloned_from": original_name
            }
            self.registry.registry["models"][new_name] = new_info
            self.registry._save_registry()
            
            self._refresh_list()
            
            # Auto-select the new clone so user can see it's selected
            self.selected_model = new_name
            self._update_buttons_state()
            self.info_name.setText(f"{new_name}")
            self.info_details.setText(f"Clone of: {original_name}\n\nClick to select a different model.")
            
            # Highlight the clone in the list
            for i in range(self.model_list.count()):
                item = self.model_list.item(i)
                if new_name in item.text():
                    self.model_list.setCurrentItem(item)
                    break
            
            QMessageBox.information(self, "Cloned", f"Created clone: {new_name}\n\nThe clone is now selected.")
        except Exception as e:
            QMessageBox.warning(self, "Clone Failed", str(e))
    
    def _on_open_folder(self):
        """Open model folder in file explorer."""
        if not self.selected_model:
            return
        
        from ..config import CONFIG
        folder = Path(CONFIG['models_dir']) / self.selected_model
        
        if not folder.exists():
            QMessageBox.warning(self, "Not Found", "Model folder not found")
            return
        
        import subprocess
        import platform
        
        try:
            if platform.system() == "Windows":
                import os
                os.startfile(str(folder))
            elif platform.system() == "Darwin":
                subprocess.run(["open", str(folder)])
            else:
                subprocess.run(["xdg-open", str(folder)])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open folder: {e}")
    
    def _on_grow(self):
        """Grow the model to a larger size."""
        if not self.selected_model:
            return
        
        current_size = self.registry.registry["models"].get(self.selected_model, {}).get("size", "tiny")
        sizes = ["tiny", "small", "medium", "large", "xl"]
        
        try:
            idx = sizes.index(current_size)
            available = sizes[idx + 1:]
        except ValueError:
            available = sizes
        
        if not available:
            QMessageBox.information(self, "Max Size", "Already at maximum size")
            return
        
        size, ok = self._size_dialog("Grow Model", available, f"Current: {current_size}")
        if not ok or not size:
            return
        
        reply = QMessageBox.question(
            self, "Confirm Grow",
            f"Grow '{self.selected_model}' from {current_size} to {size}?\n\nA backup will be created first."
        )
        
        if reply == QMessageBox.Yes:
            self._on_backup()  # Auto backup
            try:
                from ..core.model_scaling import grow_registered_model
                grow_registered_model(self.registry, self.selected_model, self.selected_model, size)
                self._refresh_list()
                QMessageBox.information(self, "Success", f"Model grown to {size}!")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
    
    def _on_shrink(self):
        """Shrink the model to a smaller size."""
        if not self.selected_model:
            return
        
        current_size = self.registry.registry["models"].get(self.selected_model, {}).get("size", "large")
        sizes = ["nano", "micro", "tiny", "small", "medium", "large"]
        
        try:
            idx = sizes.index(current_size)
            available = sizes[:idx]
        except ValueError:
            available = sizes[:-1]
        
        if not available:
            QMessageBox.information(self, "Min Size", "Already at minimum size")
            return
        
        size, ok = self._size_dialog("Shrink Model", available, f"Current: {current_size}\nWarning: May lose capacity!")
        if not ok or not size:
            return
        
        reply = QMessageBox.warning(
            self, "Confirm Shrink",
            f"Shrink '{self.selected_model}' to {size}?\n\nWarning: This may reduce model quality.\nA backup will be created first.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._on_backup()
            try:
                model, config = self.registry.load_model(self.selected_model)
                shrunk = shrink_model(model, size, config["vocab_size"])
                self.registry.save_model(self.selected_model, shrunk)
                self.registry.registry["models"][self.selected_model]["size"] = size
                self.registry._save_registry()
                self._refresh_list()
                QMessageBox.information(self, "Success", f"Model shrunk to {size}!")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
    
    def _on_rename(self):
        """Rename the selected model."""
        if not self.selected_model:
            return
        
        from PyQt5.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(
            self, "Rename Model",
            f"New name for '{self.selected_model}':",
            text=self.selected_model
        )
        
        if not ok or not new_name.strip() or new_name == self.selected_model:
            return
        
        new_name = new_name.strip().replace(' ', '_')
        
        if new_name in self.registry.registry.get("models", {}):
            QMessageBox.warning(self, "Name Exists", f"'{new_name}' already exists")
            return
        
        try:
            from ..config import CONFIG
            old = Path(CONFIG['models_dir']) / self.selected_model
            new = Path(CONFIG['models_dir']) / new_name
            old.rename(new)
            
            info = self.registry.registry["models"].pop(self.selected_model)
            # Update the path to reflect new name
            info["path"] = str(new)
            self.registry.registry["models"][new_name] = info
            self.registry._save_registry()
            
            self.selected_model = new_name
            self._refresh_list()
            QMessageBox.information(self, "Renamed", f"Model renamed to '{new_name}'")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
    
    def _on_delete_action(self, create_backup: bool):
        """Handle delete button click."""
        if not self.selected_model:
            QMessageBox.warning(self, "No Selection", "Please select a model first")
            return
        
        model_to_delete = self.selected_model
        
        # Check for existing backups
        from pathlib import Path
        models_dir = Path("models")
        backup_found = []
        for backup_dir in models_dir.glob(f"{model_to_delete}_backup*"):
            if backup_dir.is_dir():
                backup_found.append(backup_dir.name)
        
        # Build confirmation message
        if create_backup:
            action_msg = "DELETE WITH BACKUP"
            extra_info = "\n\nA backup will be created before deletion."
        else:
            action_msg = "PERMANENTLY DELETE"
            extra_info = ""
        
        backup_msg = ""
        if backup_found:
            backup_msg = f"\n\nExisting backups:\n" + "\n".join(f"  - {b}" for b in backup_found)
        
        # Single confirmation - no typing required
        reply = QMessageBox.warning(
            self, f"{action_msg}",
            f"Are you sure you want to delete:\n\n"
            f"   {model_to_delete}\n{extra_info}{backup_msg}",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            # Create backup if requested
            if create_backup:
                self._on_backup()
            
            # Actually delete
            self.registry.delete_model(model_to_delete, confirm=True)
            self.selected_model = None
            self._refresh_list()
            
            # Clean up tool routing
            self._cleanup_tool_routing(model_to_delete)
            
            msg = f"Model '{model_to_delete}' has been deleted."
            if create_backup:
                msg += "\n\nA backup was created in models/_backups/"
            QMessageBox.information(self, "Deleted", msg)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
    
    def _cleanup_tool_routing(self, deleted_model: str):
        """Remove deleted model from tool_routing.json assignments."""
        try:
            import json
            from pathlib import Path
            routing_path = Path("information/tool_routing.json")
            if not routing_path.exists():
                return
            
            with open(routing_path, "r") as f:
                routing = json.load(f)
            
            changed = False
            for tool_name, assignments in routing.items():
                if isinstance(assignments, list):
                    # Remove any assignment using this model
                    new_assignments = [
                        a for a in assignments 
                        if not (a.get("model", "").lower() == deleted_model.lower() or
                                deleted_model.lower() in a.get("model", "").lower())
                    ]
                    if len(new_assignments) != len(assignments):
                        routing[tool_name] = new_assignments
                        changed = True
            
            if changed:
                with open(routing_path, "w") as f:
                    json.dump(routing, f, indent=2)
        except Exception:
            pass  # Non-critical, don't block deletion
    
    def _size_dialog(self, title, sizes, message):
        """Show size selection dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setStyleSheet(self.styleSheet())
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel(message))
        
        combo = QComboBox()
        combo.addItems(sizes)
        combo.setStyleSheet("padding: 8px; background: #313244; color: #cdd6f4; border-radius: 4px;")
        layout.addWidget(combo)
        
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        layout.addWidget(btns)
        
        if dialog.exec_() == QDialog.Accepted:
            return combo.currentText(), True
        return None, False
    
    def get_selected_model(self):
        return self.selected_model
    
    def closeEvent(self, event):
        """Handle close - just close, don't block."""
        event.accept()


class EnhancedMainWindow(QMainWindow):
    """Enhanced main window with setup wizard and model management."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enigma Engine")
        # Allow window to resize freely (no fixed constraints)
        self.setMinimumSize(600, 400)  # Reasonable minimum
        self.resize(1000, 700)
        
        # Set window icon
        self._set_window_icon()
        
        # Setup keyboard shortcuts for emergency close
        self._setup_shortcuts()
        
        # Initialize registry
        self.registry = ModelRegistry()
        self.current_model_name = None
        self.engine = None
        
        # Load GUI settings (last model, window size, etc.)
        self._gui_settings = self._load_gui_settings()
        
        # Initialize module manager and register all built-in modules
        try:
            from enigma.modules import ModuleManager, register_all
            self.module_manager = ModuleManager()
            register_all(self.module_manager)
        except Exception as e:
            print(f"Could not initialize ModuleManager: {e}")
            self.module_manager = None
        
        # Initialize toggle states
        self.auto_speak = False
        self.microphone_enabled = False
        
        # Initialize chat state
        self.chat_messages = []
        
        # Initialize display names
        self.user_display_name = self._gui_settings.get("user_display_name", "You")
        
        # Training lock to prevent concurrent training
        self._is_training = False
        self._stop_training = False
        
        # Track if current model is HuggingFace (for feature restrictions)
        self._is_hf_model = False
        
        # Build UI first (before model load so user sees the window immediately)
        self._build_ui()
        
        # Check if first run (no models) - this needs to be synchronous
        if not self.registry.registry.get("models"):
            self._run_setup_wizard()
        else:
            # Defer model loading to after GUI is shown
            self._show_model_selector_deferred()
    
    def _set_window_icon(self):
        """Set the window icon from file or create a default."""
        from pathlib import Path
        try:
            icon_paths = [
                Path(__file__).parent / "icons" / "enigma.ico",
                Path(__file__).parent / "icons" / "enigma_256.png",
                Path(CONFIG.get("data_dir", "data")) / "icons" / "enigma.ico",
            ]
            
            for icon_path in icon_paths:
                if icon_path.exists():
                    self.setWindowIcon(QIcon(str(icon_path)))
                    return
        except Exception as e:
            print(f"Could not set window icon: {e}")
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts including emergency close."""
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        
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
    
    def _on_escape_pressed(self):
        """Handle escape key - if any popup/dialog is open, close it. Otherwise do nothing."""
        # This is mainly for consistency. The tray ESC functionality 
        # is handled via the overlay's keyPressEvent
        pass
    
    def _force_quit(self):
        """Force quit the application with cleanup (Alt+F4)."""
        try:
            # Try to save any unsaved state
            self._save_gui_settings()
            # Hide tray icon if exists
            tray = get_system_tray()
            if tray and hasattr(tray, 'tray_icon'):
                tray.tray_icon.hide()
        except Exception:
            pass
        QApplication.quit()
    
    def _emergency_quit(self):
        """Emergency quit - no questions asked, just exit."""
        import sys
        print("[EMERGENCY] Force quitting application...")
        # Also try to hide tray
        try:
            tray = get_system_tray()
            if tray and hasattr(tray, 'tray_icon'):
                tray.tray_icon.hide()
        except:
            pass
        sys.exit(0)
    
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
                        font-size: 14px;
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
                        font-size: 14px;
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
                self.learning_indicator.setText("📚 Learning: N/A")
                self.learning_indicator.setStyleSheet("color: #6c7086; font-size: 11px;")
                self.learning_indicator.setToolTip(
                    "Learning is not available for HuggingFace models.\n\n"
                    "HuggingFace models are pre-trained and cannot be fine-tuned locally.\n"
                    "To use learning features, switch to a local Enigma model."
                )
                self.learning_indicator.setCursor(Qt.ArrowCursor)  # Remove clickable cursor
                self.learning_indicator.mousePressEvent = lambda e: None  # Disable click
            else:
                # Re-enable for Enigma models
                self.learning_indicator.setCursor(Qt.PointingHandCursor)
                from .tabs.chat_tab import _toggle_learning
                self.learning_indicator.mousePressEvent = lambda e: _toggle_learning(self)
                # Restore current state
                if getattr(self, 'learn_while_chatting', True):
                    self.learning_indicator.setText("📚 Learning: ON")
                    self.learning_indicator.setStyleSheet("color: #a6e3a1; font-size: 11px;")
                else:
                    self.learning_indicator.setText("📚 Learning: OFF")
                    self.learning_indicator.setStyleSheet("color: #6c7086; font-size: 11px;")
                self.learning_indicator.setToolTip(
                    "When Learning is ON, the AI records your conversations and uses them to improve.\n\n"
                    "How it works:\n"
                    "• Each Q&A pair is saved to the model's training data\n"
                    "• After enough interactions, the model can be retrained\n"
                    "• This helps the AI learn your preferences and style\n\n"
                    "Click to toggle learning on/off."
                )
    
    def _require_enigma_model(self, feature_name: str) -> bool:
        """
        Check if current model is Enigma. If HuggingFace, show warning and return False.
        Use this to guard Enigma-only features like training.
        """
        if self._is_huggingface_model():
            # Get list of local Enigma models
            enigma_models = []
            for name, info in self.registry.registry.get("models", {}).items():
                if info.get("source") != "huggingface":
                    enigma_models.append(name)
            
            model_list = ", ".join(enigma_models) if enigma_models else "Create one in Model Manager"
            
            QMessageBox.warning(
                self, 
                f"{feature_name} - Enigma Model Required",
                f"<b>{feature_name}</b> is only available for local Enigma models.<br><br>"
                f"You're currently using a HuggingFace model: <b>{self.current_model_name}</b><br><br>"
                f"HuggingFace models are pre-trained and cannot be modified through this interface.<br><br>"
                f"<b>Available Enigma models:</b><br>{model_list}<br><br>"
                f"Switch to an Enigma model in the <b>Model Manager</b> tab to use this feature."
            )
            return False
        return True
    
    def _load_gui_settings(self):
        """Load GUI settings from file."""
        from ..config import CONFIG
        settings_path = Path(CONFIG["data_dir"]) / "gui_settings.json"
        try:
            if settings_path.exists():
                with open(settings_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Could not load GUI settings: {e}")
        return {}
    
    def _save_gui_settings(self):
        """Save GUI settings to file."""
        from ..config import CONFIG
        from PyQt5.QtGui import QGuiApplication
        
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
            
            with open(settings_path, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Could not save GUI settings: {e}")
    
    def closeEvent(self, event):
        """Handle window close - save settings."""
        self._save_gui_settings()
        event.accept()
    
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
                    description="Created via setup wizard"
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
        
        # Show "loading" status in chat immediately
        if hasattr(self, 'chat_display'):
            self.chat_display.append(
                f"<p style='color: #f9e2af;'><i>⏳ Loading model: {self.current_model_name}...</i></p>"
            )
        
        if hasattr(self, 'chat_status'):
            self.chat_status.setText(f"Loading {self.current_model_name}...")
        
        # Update window title to show loading
        self.setWindowTitle(f"Enigma Engine - Loading {self.current_model_name}...")
        
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
        if not self.current_model_name:
            return
            
        # Create and show loading dialog
        loading_dialog = ModelLoadingDialog(self.current_model_name, self)
        loading_dialog.show()
        
        # Force dialog to fully render before starting loading
        QApplication.processEvents()
        QApplication.processEvents()  # Double process to ensure rendering
        
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
            loading_dialog.set_status("✓ Model weights loaded", 40)
            
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
            loading_dialog.set_status("✓ Engine created", 50)
            
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
                loading_dialog.set_status("✓ Fast routing enabled", 57)
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
                loading_dialog.set_status("✓ HuggingFace model ready", 65)
                
                # Check if user wants custom tokenizer instead of model's own
                use_custom_tokenizer = config.get("use_custom_tokenizer", False)
                if use_custom_tokenizer:
                    loading_dialog.set_status("Loading custom Enigma tokenizer...", 70)
                    from ..core.tokenizer import load_tokenizer
                    self.engine.tokenizer = load_tokenizer()
                    self.engine._using_custom_tokenizer = True
                    loading_dialog.set_status("✓ Custom tokenizer loaded", 75)
                else:
                    self.engine.tokenizer = model.tokenizer  # Use HF tokenizer
                    self.engine._using_custom_tokenizer = False
                    loading_dialog.set_status("✓ Using model's tokenizer", 75)
            else:
                # Local Enigma model
                self.engine.model = model
                loading_dialog.set_status("Moving model to GPU/CPU...", 68)
                self.engine.model.to(self.engine.device)
                self.engine.model.eval()
                loading_dialog.set_status("✓ Model ready on device", 72)
                loading_dialog.set_status("Loading tokenizer...", 75)
                from ..core.tokenizer import load_tokenizer
                self.engine.tokenizer = load_tokenizer()
                loading_dialog.set_status("✓ Tokenizer loaded", 80)
            
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
            loading_dialog.set_status("✓ Brain initialized", 90)
            
            loading_dialog.set_status("Finalizing setup...", 95)
            
            # Update window title with model type indicator
            model_type = "[HF]" if is_huggingface else "[Enigma]"
            self.setWindowTitle(f"Enigma Engine - {self.current_model_name} {model_type}")
            
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
            
            # Show welcome message in chat
            if hasattr(self, 'chat_display'):
                device_type = self.engine.device.type if hasattr(self.engine.device, 'type') else str(self.engine.device)
                device_info = "GPU" if device_type == "cuda" else "CPU"
                
                if is_huggingface:
                    model_note = f"<p style='color: #f9e2af;'><i>This is a HuggingFace model. Training and some Enigma features are not available.</i></p>"
                else:
                    model_note = ""
                
                self.chat_display.append(
                    f"<p style='color: #a6e3a1;'><b>[OK] Model loaded:</b> {self.current_model_name} ({device_info})</p>"
                    f"{model_note}"
                    f"<p style='color: #6c7086;'>Type a message below to chat with your AI.</p>"
                    "<hr>"
                )
            
            # Update chat status
            if hasattr(self, 'chat_status'):
                self.chat_status.setText(f"Model ready ({self.engine.device})")
            
            # Update system tray with model name
            try:
                tray = get_system_tray()
                if tray and hasattr(tray, 'update_model_name'):
                    tray.update_model_name(self.current_model_name)
            except:
                pass
            
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
        """Build the main UI."""
        # Menu bar
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        file_menu.addAction("New Model...", self._on_new_model)
        file_menu.addAction("Open Model...", self._on_open_model)
        file_menu.addSeparator()
        file_menu.addAction("Backup Current Model", self._on_backup_current)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        
        # Options menu with toggles
        options_menu = menubar.addMenu("Options")
        
        # Theme submenu with all 4 themes
        theme_menu = options_menu.addMenu("Theme")
        self.theme_group = QActionGroup(self)
        self.theme_group.setExclusive(True)
        
        theme_dark = theme_menu.addAction("Dark (Catppuccin)")
        theme_dark.setCheckable(True)
        theme_dark.setChecked(True)
        theme_dark.triggered.connect(lambda: self._set_theme("dark"))
        self.theme_group.addAction(theme_dark)
        
        theme_light = theme_menu.addAction("Light")
        theme_light.setCheckable(True)
        theme_light.triggered.connect(lambda: self._set_theme("light"))
        self.theme_group.addAction(theme_light)
        
        theme_shadow = theme_menu.addAction("Shadow (Deep Purple)")
        theme_shadow.setCheckable(True)
        theme_shadow.triggered.connect(lambda: self._set_theme("shadow"))
        self.theme_group.addAction(theme_shadow)
        
        theme_midnight = theme_menu.addAction("Midnight (Deep Blue)")
        theme_midnight.setCheckable(True)
        theme_midnight.triggered.connect(lambda: self._set_theme("midnight"))
        self.theme_group.addAction(theme_midnight)
        
        options_menu.addSeparator()
        
        # Zoom - opens input dialog
        zoom_action = options_menu.addAction("Zoom...")
        zoom_action.setShortcut("Ctrl+Z")
        zoom_action.triggered.connect(self._show_zoom_dialog)
        
        options_menu.addSeparator()
        
        self.avatar_action = options_menu.addAction("Avatar (OFF)")
        self.avatar_action.setCheckable(True)
        self.avatar_action.setChecked(False)
        self.avatar_action.triggered.connect(self._toggle_avatar)
        
        options_menu.addSeparator()
        
        self.auto_speak_action = options_menu.addAction("AI Auto-Speak (OFF)")
        self.auto_speak_action.setCheckable(True)
        self.auto_speak_action.setChecked(False)
        self.auto_speak_action.triggered.connect(self._toggle_auto_speak)
        
        self.microphone_action = options_menu.addAction("Microphone (OFF)")
        self.microphone_action.setCheckable(True)
        self.microphone_action.setChecked(False)
        self.microphone_action.triggered.connect(self._toggle_microphone)
        
        options_menu.addSeparator()
        
        # Learn while chatting toggle
        self.learn_action = options_menu.addAction("Learn While Chatting (ON)")
        self.learn_action.setCheckable(True)
        self.learn_action.setChecked(True)  # On by default
        self.learn_action.triggered.connect(self._toggle_learning)
        self.learn_while_chatting = True
        
        # Status bar with clickable model selector
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
        
        # Apply dark mode by default
        self.setStyleSheet(DARK_STYLE)
        
        # Import tabs from separate modules
        from .tabs import (
            create_chat_tab, create_training_tab, 
            create_avatar_subtab, create_game_subtab, create_robot_subtab,
            create_vision_tab, create_camera_tab, create_sessions_tab, create_instructions_tab,
            create_terminal_tab, create_examples_tab,
            create_image_tab, create_code_tab, create_video_tab,
            create_audio_tab, create_embeddings_tab, create_threed_tab,
            create_logs_tab, create_notes_tab, create_network_tab,
            create_analytics_tab, create_scheduler_tab
        )
        from .tabs.gif_tab import create_gif_tab
        from .tabs.settings_tab import create_settings_tab
        from .tabs.modules_tab import ModulesTab
        from .tabs.scaling_tab import ScalingTab
        from .tabs.model_router_tab import ModelRouterTab
        from .tabs.tool_manager_tab import ToolManagerTab
        
        # Create main container with sidebar navigation
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # === SIDEBAR NAVIGATION ===
        sidebar_container = QWidget()
        sidebar_container.setFixedWidth(180)
        sidebar_container.setStyleSheet("background-color: #11111b;")
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        
        # App title/logo area
        title_widget = QWidget()
        title_widget.setFixedHeight(50)
        title_widget.setStyleSheet("""
            background-color: #11111b;
            border-bottom: 1px solid #1e1e2e;
        """)
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(16, 0, 16, 0)
        app_title = QLabel("ENIGMA")
        app_title.setStyleSheet("""
            color: #89b4fa;
            font-size: 16px;
            font-weight: bold;
            letter-spacing: 2px;
        """)
        title_layout.addWidget(app_title)
        sidebar_layout.addWidget(title_widget)
        
        # Sidebar list widget
        self.sidebar = QListWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Define navigation items with sections
        nav_items = [
            # Core
            ("section", "CORE"),
            ("", "Chat", "chat"),
            ("", "Train", "train"),
            ("", "History", "history"),
            # Model
            ("section", "MODEL"),
            ("", "Scale", "scale"),
            ("", "Modules", "modules"),
            ("", "Tools", "tools"),  # Tool Manager
            ("", "Router", "router"),
            # Generate
            ("section", "GENERATE"),
            ("", "Image", "image"),
            ("", "Code", "code"),
            ("", "Video", "video"),
            ("", "Audio", "audio"),
            ("", "3D", "3d"),
            ("", "GIF", "gif"),
            # Connect
            ("section", "CONNECT"),
            ("", "Search", "search"),
            ("", "Avatar", "avatar"),
            ("", "Game", "game"),
            ("", "Robot", "robot"),
            ("", "Vision", "vision"),
            ("", "Camera", "camera"),
            # Tools
            ("section", "SYSTEM"),
            ("", "Terminal", "terminal"),
            ("", "Files", "files"),
            ("", "Logs", "logs"),
            ("", "Notes", "notes"),
            ("", "Network", "network"),
            ("", "Analytics", "analytics"),
            ("", "Scheduler", "scheduler"),
            ("", "Examples", "examples"),
            ("", "Settings", "settings"),
        ]
        
        # Add items to sidebar
        self._nav_map = {}  # Map item text to stack index
        stack_index = 0
        
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
                from PyQt5.QtGui import QColor, QBrush
                section_item.setForeground(QBrush(QColor("#6c7086")))
                self.sidebar.addItem(section_item)
            else:
                icon, name, key = item
                list_item = QListWidgetItem(f"   {name}")
                list_item.setData(Qt.UserRole, key)
                list_item.setSizeHint(QSize(170, 38))
                self.sidebar.addItem(list_item)
                self._nav_map[key] = stack_index
                stack_index += 1
        
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
        self.content_stack.addWidget(wrap_in_scroll(create_chat_tab(self)))  # Chat
        self.content_stack.addWidget(wrap_in_scroll(create_training_tab(self)))  # Train
        self.content_stack.addWidget(wrap_in_scroll(create_sessions_tab(self)))  # History
        self.content_stack.addWidget(wrap_in_scroll(ScalingTab(self)))  # Scale
        self.content_stack.addWidget(wrap_in_scroll(ModulesTab(self, module_manager=self.module_manager)))  # Modules
        self.content_stack.addWidget(wrap_in_scroll(ToolManagerTab(self)))  # Tools
        self.router_tab = ModelRouterTab(self)  # Store reference for syncing
        self.content_stack.addWidget(wrap_in_scroll(self.router_tab))  # Router
        self.content_stack.addWidget(wrap_in_scroll(create_image_tab(self)))  # Image
        self.content_stack.addWidget(wrap_in_scroll(create_code_tab(self)))  # Code
        self.content_stack.addWidget(wrap_in_scroll(create_video_tab(self)))  # Video
        self.content_stack.addWidget(wrap_in_scroll(create_audio_tab(self)))  # Audio
        self.content_stack.addWidget(wrap_in_scroll(create_threed_tab(self)))  # 3D
        self.content_stack.addWidget(wrap_in_scroll(create_gif_tab(self)))  # GIF
        self.content_stack.addWidget(wrap_in_scroll(create_embeddings_tab(self)))  # Search
        self.content_stack.addWidget(wrap_in_scroll(create_avatar_subtab(self)))  # Avatar
        self.content_stack.addWidget(wrap_in_scroll(create_game_subtab(self)))  # Game
        self.content_stack.addWidget(wrap_in_scroll(create_robot_subtab(self)))  # Robot
        self.content_stack.addWidget(wrap_in_scroll(create_vision_tab(self)))  # Vision
        self.content_stack.addWidget(wrap_in_scroll(create_camera_tab(self)))  # Camera
        self.content_stack.addWidget(wrap_in_scroll(create_terminal_tab(self)))  # Terminal
        self.content_stack.addWidget(wrap_in_scroll(create_instructions_tab(self)))  # Files
        self.content_stack.addWidget(wrap_in_scroll(create_logs_tab(self)))  # Logs
        self.content_stack.addWidget(wrap_in_scroll(create_notes_tab(self)))  # Notes
        self.content_stack.addWidget(wrap_in_scroll(create_network_tab(self)))  # Network
        self.content_stack.addWidget(wrap_in_scroll(create_analytics_tab(self)))  # Analytics
        self.content_stack.addWidget(wrap_in_scroll(create_scheduler_tab(self)))  # Scheduler
        self.content_stack.addWidget(wrap_in_scroll(create_examples_tab(self)))  # Examples
        self.content_stack.addWidget(wrap_in_scroll(create_settings_tab(self)))  # Settings
        
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
        
        self.setCentralWidget(main_widget)
        
        # Enable text selection on all QLabels in the GUI
        self._enable_text_selection()
        
        # Restore saved settings after UI is built
        self._restore_gui_settings()
    
    def _enable_text_selection(self):
        """Enable text selection on all QLabel widgets in the GUI."""
        from PyQt5.QtCore import Qt
        for label in self.findChildren(QLabel):
            # Enable text selection (but not links, which need different flags)
            current_flags = label.textInteractionFlags()
            # Add text selection flag if not already set
            if not (current_flags & Qt.TextSelectableByMouse):
                label.setTextInteractionFlags(current_flags | Qt.TextSelectableByMouse)
    
    def _restore_gui_settings(self):
        """Restore GUI settings from saved file."""
        from PyQt5.QtGui import QGuiApplication
        
        settings = self._gui_settings
        
        # Restore window position and monitor
        monitor_index = settings.get("monitor_index", 0)
        screens = QGuiApplication.screens()
        
        if monitor_index < len(screens):
            screen = screens[monitor_index]
            screen_geo = screen.geometry()
            
            # Restore position if saved, otherwise center on screen
            x = settings.get("window_x")
            y = settings.get("window_y")
            
            if x is not None and y is not None:
                # Verify the position is on a valid screen
                from PyQt5.QtCore import QPoint
                if QGuiApplication.screenAt(QPoint(x, y)):
                    self.move(x, y)
                else:
                    # Position is off-screen, center on target monitor
                    self.move(
                        screen_geo.x() + (screen_geo.width() - self.width()) // 2,
                        screen_geo.y() + (screen_geo.height() - self.height()) // 2
                    )
        
        # Restore always on top
        if settings.get("always_on_top", False):
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
            self.show()
        
        # Restore auto-speak state
        if settings.get("auto_speak", False):
            self.auto_speak_action.setChecked(True)
            self._toggle_auto_speak(True)
        
        # Restore microphone state
        if settings.get("microphone_enabled", False):
            self.microphone_action.setChecked(True)
            self._toggle_microphone(True)
        
        # ALWAYS start on Chat tab - ignore saved last_tab setting
        # This ensures predictable behavior when opening the GUI
        for i in range(self.sidebar.count()):
            item = self.sidebar.item(i)
            if item and item.data(Qt.UserRole) == 'chat':
                self.sidebar.setCurrentRow(i)
                break
    
    def _on_sidebar_changed(self, current, previous):
        """Handle sidebar navigation change."""
        if current:
            key = current.data(Qt.UserRole)
            if key and key in self._nav_map:
                self.content_stack.setCurrentIndex(self._nav_map[key])
    
    def _switch_to_tab(self, tab_name: str):
        """Switch to a specific tab by name (for chat commands)."""
        # Find the tab key that matches - all available tabs
        key_map = {
            # Core tabs
            'chat': 'chat', 'train': 'train', 'history': 'history',
            'scale': 'scale', 'modules': 'modules', 'tools': 'tools',
            'router': 'router',
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
    
    def _toggle_learning(self, checked):
        """Toggle learn-while-chatting mode."""
        # Check if using HuggingFace model - learning not supported
        if checked and self._is_huggingface_model():
            QMessageBox.information(
                self,
                "Learning Not Available",
                "Learning while chatting is only available for local Enigma models.\n\n"
                f"Current model ({self.current_model_name}) is a HuggingFace model and cannot be trained.\n\n"
                "Switch to an Enigma model to enable this feature."
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
                self.learning_indicator.setStyleSheet("color: #a6e3a1; font-size: 11px;")
            else:
                self.learning_indicator.setText("Learning: OFF")
                self.learning_indicator.setStyleSheet("color: #6c7086; font-size: 11px;")
        
        # Update brain if loaded
        if hasattr(self, 'brain') and self.brain:
            self.brain.auto_learn = checked
    
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
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QSpinBox, QPushButton, QDialogButtonBox
        from PyQt5.QtCore import Qt
        
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
                    else:
                        self.avatar_action.setChecked(False)
                        self.avatar_action.setText("Avatar (OFF)")
                        QMessageBox.warning(self, "Avatar Error", "Failed to load avatar module")
                else:
                    # Unload avatar module
                    self.module_manager.unload('avatar')
                    self._disable_avatar()
                    self.avatar_action.setText("Avatar (OFF)")
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
                    import subprocess
                    import tempfile
                    import os
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
                    import subprocess
                    import tempfile
                    import os
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
            if session_path.exists():
                session_path.unlink()
            self._refresh_sessions()
            self.session_viewer.clear()
    
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
                except Exception as e:
                    pass  # Use original filepath
            
            # Display the image
            self._display_avatar_image(filepath)
            
            # Update expression dict
            if hasattr(self, 'avatar_expressions'):
                self.avatar_expressions["neutral"] = filepath
    
    def _display_avatar_image(self, filepath):
        """Display an avatar image."""
        pixmap = QPixmap(filepath)
        if not pixmap.isNull():
            scaled = pixmap.scaled(380, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.avatar_image_label.setPixmap(scaled)
            self.avatar_image_label.setStyleSheet("border: 2px solid #89b4fa; border-radius: 12px; background: #1e1e2e;")
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
        
        # Handle chat commands (e.g., /image, /video, /code, /audio, /help)
        if text.startswith('/'):
            self._handle_chat_command(text)
            return
        
        if not self.engine:
            self.chat_display.append("<b style='color:#f38ba8;'>System:</b> No model loaded. "
                                      "Create or load a model first (File menu).")
            self.chat_input.clear()
            return
        
        # Check if already generating (prevent double-sends)
        if hasattr(self, '_ai_worker') and self._ai_worker and self._ai_worker.isRunning():
            self.chat_display.append("<b style='color:#f9e2af;'>System:</b> Still generating... please wait.")
            return
        
        # Check if model is trained (for Enigma models)
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
        
        # Track user message
        self.chat_messages.append({
            "role": "user",
            "text": text,
            "ts": time.time()
        })
        
        # Show "thinking" indicator in chat
        self.chat_display.append(
            f'<div id="thinking" style="color: #f9e2af; padding: 4px;"><i>{self.current_model_name} is thinking...</i></div>'
        )
        
        # Show thinking panel and stop button
        if hasattr(self, 'thinking_frame'):
            self.thinking_frame.show()
            self.thinking_label.setText("Analyzing your message...")
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
            
            if getattr(self.engine, '_using_custom_tokenizer', False):
                custom_tok = self.engine.tokenizer
            
            system_prompt = (
                "You are an AI assistant running in the Enigma Engine GUI. "
                "You can generate images, videos, code, audio, and 3D models directly. "
                "When the user asks you to create something, output a tool call using this format:\n"
                "<tool_call>{\"tool\": \"tool_name\", \"params\": {\"prompt\": \"description\"}}</tool_call>\n"
                "Available tools: generate_image, generate_video, generate_code, generate_audio, generate_3d\n"
                "For example, if asked to create an image of a sunset, respond with:\n"
                "I'll create that image for you!\n"
                "<tool_call>{\"tool\": \"generate_image\", \"params\": {\"prompt\": \"beautiful sunset over mountains\"}}</tool_call>\n"
                "Be helpful, concise, and friendly."
            )
        
        # Start background worker
        self._ai_worker = AIGenerationWorker(
            self.engine, text, is_hf, history, system_prompt, custom_tok
        )
        self._ai_worker.finished.connect(self._on_ai_response)
        self._ai_worker.error.connect(self._on_ai_error)
        self._ai_worker.thinking.connect(self._on_thinking_update)
        self._ai_worker.stopped.connect(self._on_ai_stopped)
        
        # Track when generation started for timing display
        self._generation_start_time = time.time()
        
        self._ai_worker.start()
    
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
            '<div style="color: #f9e2af; padding: 4px;"><i>⏹ Generation stopped</i></div>'
        )
    
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
    
    def _execute_tool_from_response(self, response: str):
        """Parse and execute any tool calls in the AI response, return modified response."""
        import re
        import json
        
        tool_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(tool_pattern, response, re.DOTALL)
        
        if not matches:
            return response, []
        
        results = []
        for match in matches:
            try:
                tool_data = json.loads(match)
                tool_name = tool_data.get('tool', '')
                params = tool_data.get('params', {})
                prompt = params.get('prompt', params.get('text', params.get('description', '')))
                
                result = self._execute_generation_tool(tool_name, prompt)
                results.append(result)
            except json.JSONDecodeError:
                results.append({'success': False, 'error': 'Invalid tool call format'})
        
        return response, results
    
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
        
        # AUTO-CONTROL: Automatically control avatar/robot/game based on response
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
        
        # Format response
        if HAVE_TEXT_FORMATTER:
            formatted_response = TextFormatter.to_html(display_response)
        else:
            formatted_response = display_response
        
        # Calculate thinking time
        thinking_time = ""
        if hasattr(self, '_generation_start_time'):
            elapsed = time.time() - self._generation_start_time
            thinking_time = f'<span style="color: #6c7086; font-size: 10px; float: right;">⏱️ {elapsed:.1f}s</span>'
        
        # Generate unique ID for this response (for feedback)
        response_id = int(time.time() * 1000)
        
        # Check if we should show rating buttons (only for local Enigma models)
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
            # Local Enigma model - show rating buttons for learning
            self.chat_display.append(
                f'<div style="background-color: #1e1e2e; padding: 8px; margin: 4px 0; border-radius: 8px; border-left: 3px solid #a6e3a1;">'
                f'<b style="color: #a6e3a1;">{self.current_model_name}:</b> {thinking_time}{formatted_response}'
                f'<div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid #45475a;">'
                f'<span style="color: #6c7086; font-size: 11px;">Rate this response: </span>'
                f'<a href="feedback:good:{response_id}" style="color: #a6e3a1; text-decoration: none; margin: 0 4px;">👍 Good</a>'
                f'<a href="feedback:bad:{response_id}" style="color: #f38ba8; text-decoration: none; margin: 0 4px;">👎 Bad</a>'
                f'<a href="feedback:critique:{response_id}" style="color: #89b4fa; text-decoration: none; margin: 0 4px;">✏️ Critique</a>'
                f'</div></div>'
            )
        self.last_response = response
        self._last_response_id = response_id
        
        # Store response for feedback
        if not hasattr(self, '_response_history'):
            self._response_history = {}
        self._response_history[response_id] = {
            'user_input': user_msgs[-1].get("text", "") if user_msgs else "",
            'ai_response': response,
            'timestamp': time.time()
        }
        
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
                        f'<b style="color: #f9e2af;">🎨 Auto-generating {result_type}...</b> Check the {result_type.title()} tab for results.</div>'
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
        
        # Learn from interaction (Enigma models only)
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
                    from enigma.avatar import get_avatar
                    avatar = get_avatar()
                    if avatar and avatar.is_enabled:
                        avatar.set_expression(emotion)
                except Exception:
                    pass
            
            # Auto-speak through avatar if auto_speak is on
            if getattr(self, 'auto_speak', False):
                try:
                    from enigma.avatar import get_avatar
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
                    from enigma.tools.robot_tools import get_robot
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
            'happy': ['happy', 'glad', 'joy', 'excited', 'great', 'wonderful', 'yay', '😊', '😄', 'haha', 'lol', ':)', 'awesome', 'fantastic', 'love it'],
            'sad': ['sad', 'sorry', 'unfortunately', 'regret', 'miss', 'disappointed', '😢', '😔', ':(', 'apolog'],
            'thinking': ['hmm', 'let me think', 'considering', 'perhaps', 'maybe', 'not sure', 'wondering', '🤔', 'interesting question'],
            'surprised': ['wow', 'amazing', 'incredible', 'unbelievable', 'really?', 'no way', '😮', '😲', 'whoa', 'oh!'],
            'angry': ['angry', 'frustrated', 'annoyed', 'upset', '😠', '😤', 'unacceptable'],
            'confused': ['confused', "don't understand", 'unclear', 'what do you mean', '🤨', 'huh?', 'sorry?'],
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
<b style='color:#89b4fa;'>💡 Natural Language Generation:</b><br>
Just ask naturally! The AI understands requests like:<br>
• "Generate an image of a sunset"<br>
• "Create a picture of a cat"<br>
• "Write code for a web scraper"<br>
• "Make a 3D model of a chair"<br>
<br>
<b style='color:#89b4fa;'>🎨 Generation Commands:</b><br>
<b>/image &lt;prompt&gt;</b> - Generate an image<br>
<b>/video &lt;prompt&gt;</b> - Generate a video<br>
<b>/gif &lt;prompt&gt;</b> - Generate an animated GIF<br>
<b>/code &lt;description&gt;</b> - Generate code<br>
<b>/audio &lt;text&gt;</b> - Generate speech audio<br>
<b>/3d &lt;prompt&gt;</b> - Generate 3D model<br>
<b>/embed &lt;text&gt;</b> - Generate embeddings<br>
<br>
<b style='color:#89b4fa;'>📂 Navigation Commands:</b><br>
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
<b style='color:#89b4fa;'>🔧 Utility Commands:</b><br>
<b>/clear</b> - Clear chat history<br>
<b>/new</b> - Start a new conversation<br>
<b>/help</b> - Show this help<br>
<br>
<b style='color:#f9e2af;'>📚 Learning Mode:</b><br>
When ON, the AI saves your conversations to improve over time.<br>
Click the "Learning: ON/OFF" indicator to toggle.<br>
<i>(Only works with local Enigma models, not HuggingFace models)</i>
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
        """Speak text using TTS."""
        try:
            from ..voice import speak
            speak(text)
        except Exception as e:
            pass  # Silent fail for auto-speak
    
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
                self.registry.create_model(result["name"], size=result["size"], vocab_size=32000)
                self._refresh_models_list()
                QMessageBox.information(self, "Success", f"Created model '{result['name']}'")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
    
    def _on_open_model(self):
        dialog = ModelManagerDialog(self.registry, self.current_model_name, self)
        if dialog.exec_() == QDialog.Accepted:
            selected = dialog.get_selected_model()
            if selected:
                self.current_model_name = selected
                self._load_current_model()
                self.model_status_btn.setText(f"Model: {self.current_model_name}  v")
                self.setWindowTitle(f"Enigma Engine - {self.current_model_name}")
    
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
        if not self._require_enigma_model("Training"):
            return
        
        # DON'T auto-save editor - it might overwrite good data with truncated content
        # User should explicitly click Save if they want to save editor changes
        
        if not hasattr(self, 'training_data_path') or not self.training_data_path:
            QMessageBox.warning(self, "No Data", "Select a training file first.")
            return
        
        # Prevent concurrent training
        if self._is_training:
            QMessageBox.warning(self, "Training", "Training already in progress.")
            return
        self._is_training = True
        self._stop_training = False
        
        # Update buttons and progress
        self.btn_train.setEnabled(False)
        self.btn_train.setText("Training...")
        self.btn_stop_train.setEnabled(True)
        self.train_progress.setValue(0)
        QApplication.processEvents()
        
        try:
            from ..core.trainer import EnigmaTrainer
            
            model, config = self.registry.load_model(self.current_model_name)
            
            trainer = EnigmaTrainer(
                model=model,
                model_name=self.current_model_name,
                registry=self.registry,
                data_path=self.training_data_path,
                batch_size=self.batch_spin.value(),
                learning_rate=float(self.lr_input.text()),
            )
            
            epochs = self.epochs_spin.value()
            stopped_early = False
            for epoch in range(epochs):
                # Check if user requested stop
                if self._stop_training:
                    stopped_early = True
                    break
                
                trainer.train(epochs=1)
                progress = int((epoch + 1) / epochs * 100)
                self.train_progress.setValue(progress)
                QApplication.processEvents()
            
            # Reload model
            self._load_current_model()
            
            self.train_progress.setValue(100)
            self.btn_train.setText("Train")
            self.btn_train.setEnabled(True)
            self.btn_stop_train.setEnabled(False)
            self._is_training = False
            self._stop_training = False
            
            if stopped_early:
                self.btn_stop_train.setText("Stop")
                QMessageBox.information(self, "Stopped", f"Training stopped after epoch {epoch + 1}. Progress saved!")
            else:
                QMessageBox.information(self, "Done", "Training finished!")
        except Exception as e:
            self.btn_train.setText("Train")
            self.btn_train.setEnabled(True)
            self.btn_stop_train.setEnabled(False)
            self.btn_stop_train.setText("Stop")
            self._is_training = False
            self._stop_training = False
            QMessageBox.warning(self, "Training Error", str(e))
    
    def _on_stop_training(self):
        """Stop training after current epoch."""
        self._stop_training = True
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.setText("Stopping...")
        self.btn_train.setText("Stopping...")
    
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
                            client.publish("enigma/robot/command", command)
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
    
    # Create system tray
    try:
        from .system_tray import create_system_tray
        _system_tray = create_system_tray(app, window)
        
        if _system_tray:
            # Connect tray to window - show main window AND keep mini chat visible
            def show_both():
                window.show()
                if hasattr(_system_tray, 'overlay') and _system_tray.overlay:
                    _system_tray.overlay.show()
            
            _system_tray.show_gui_requested.connect(show_both)
            
            # Override close event to show options dialog
            if minimize_to_tray:
                original_close = window.closeEvent
                
                def close_to_tray(event):
                    if _system_tray and _system_tray.tray_icon.isVisible():
                        # Process pending events first to ensure UI responsiveness
                        QApplication.processEvents()
                        
                        # Show dialog asking what to do
                        from PyQt5.QtWidgets import QMessageBox
                        
                        msg = QMessageBox(window)
                        msg.setWindowTitle("Close Enigma")
                        msg.setText(f"""<b>Close {window.current_model_name or 'Enigma'}?</b><br><br>
What would you like to do?""")
                        msg.setIcon(QMessageBox.Question)
                        
                        # Make dialog stay on top and be responsive
                        msg.setWindowFlags(msg.windowFlags() | Qt.WindowStaysOnTopHint)
                        
                        # Custom buttons - Mini Chat is now the default/primary option
                        minichat_btn = msg.addButton("🗨️ Mini Chat", QMessageBox.AcceptRole)
                        exit_btn = msg.addButton("Exit Completely", QMessageBox.DestructiveRole)
                        kill_btn = msg.addButton("Kill All && Exit", QMessageBox.DestructiveRole)
                        cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
                        
                        msg.setDefaultButton(minichat_btn)
                        
                        # Process events to ensure dialog is responsive
                        QApplication.processEvents()
                        
                        msg.exec_()
                        
                        clicked = msg.clickedButton()
                        
                        if clicked == minichat_btn:
                            # Open mini chat (Quick Command Overlay)
                            event.ignore()
                            window.hide()
                            _system_tray.show_quick_command()
                        elif clicked == exit_btn:
                            # Exit completely
                            event.accept()
                            window._save_gui_settings()
                            _system_tray.tray_icon.hide()
                            QApplication.quit()
                        elif clicked == kill_btn:
                            # Kill all instances and exit
                            from .system_tray import kill_other_enigma_instances
                            kill_other_enigma_instances()
                            event.accept()
                            window._save_gui_settings()
                            _system_tray.tray_icon.hide()
                            QApplication.quit()
                        else:
                            # Cancel - do nothing
                            event.ignore()
                    else:
                        original_close(event)
                
                window.closeEvent = close_to_tray
            
            # Start with mini chat instead of main window
            _system_tray.show_notification(
                "Enigma Started",
                "Mini Chat is ready!\n"
                "Press ESC to open full GUI."
            )
            # Show mini chat on startup instead of main window
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(500, _system_tray.show_quick_command)
    except Exception as e:
        print(f"System tray not available: {e}")
        # If no tray, show main window
        window.show()
        sys.exit(app.exec_())
        return
    
    # Don't show main window - start with mini chat
    # window.show()  # Commented out - mini chat opens instead
    
    sys.exit(app.exec_())


def get_system_tray():
    """Get the global system tray instance."""
    return _system_tray


if __name__ == "__main__":
    run_app()
