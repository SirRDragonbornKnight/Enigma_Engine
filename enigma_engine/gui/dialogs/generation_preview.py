"""
Generation Preview Popup Dialog

Shows a popup when content is generated (images, videos, etc.)
without switching tabs.
"""
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


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
            preview_label.setText("Video Generated\n\nClick 'Open' to play")
            container_layout.addWidget(preview_label)
            
        elif self.result_type == "audio" and self.result_path:
            preview_label = QLabel()
            preview_label.setAlignment(Qt.AlignCenter)
            preview_label.setMinimumSize(300, 100)
            preview_label.setStyleSheet("border: 1px solid #45475a; border-radius: 8px; color: #cdd6f4;")
            preview_label.setText("Audio Generated\n\nClick 'Open' to play")
            container_layout.addWidget(preview_label)
            
        elif self.result_type == "3d" and self.result_path:
            preview_label = QLabel()
            preview_label.setAlignment(Qt.AlignCenter)
            preview_label.setMinimumSize(300, 100)
            preview_label.setStyleSheet("border: 1px solid #45475a; border-radius: 8px; color: #cdd6f4;")
            preview_label.setText("3D Model Generated\n\nClick 'Open' to view")
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
        from ..tabs.output_helpers import open_in_default_viewer
        path = Path(self.result_path)
        if path.exists():
            open_in_default_viewer(path)
        self.close()
    
    def _open_folder(self):
        """Open the containing folder."""
        from ..tabs.output_helpers import open_file_in_explorer
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


__all__ = ['GenerationPreviewPopup']
