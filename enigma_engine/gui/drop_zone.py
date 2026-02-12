"""
Drag and Drop File Widget for Enigma AI Engine

Provides drag-and-drop file handling for training data.

Usage:
    from enigma_engine.gui.drop_zone import DropZone
    
    drop = DropZone()
    drop.files_dropped.connect(on_files_dropped)
    layout.addWidget(drop)
"""

import logging
from pathlib import Path
from typing import List, Optional, Set

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import (
    QFrame, QLabel, QVBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QHBoxLayout, QFileDialog
)

logger = logging.getLogger(__name__)


# Supported file extensions for training data
SUPPORTED_EXTENSIONS = {
    # Text files
    '.txt', '.text',
    # Structured data
    '.json', '.jsonl', '.csv',
    # Code
    '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt',
    # Documentation
    '.md', '.markdown', '.rst',
    # Config
    '.yaml', '.yml', '.toml', '.ini', '.cfg'
}


class DropZone(QFrame):
    """
    Drag-and-drop zone for training data files.
    
    Features:
    - Visual feedback on drag over
    - File type filtering
    - File list display
    - Clear functionality
    """
    
    files_dropped = pyqtSignal(list)  # Emits list of file paths
    files_changed = pyqtSignal()  # Emits when file list changes
    
    def __init__(
        self,
        allowed_extensions: Optional[Set[str]] = None,
        parent=None
    ):
        super().__init__(parent)
        
        self.allowed_extensions = allowed_extensions or SUPPORTED_EXTENSIONS
        self._files: List[Path] = []
        
        self.setAcceptDrops(True)
        self._setup_ui()
        self._update_style(False)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Drop area label
        self.drop_label = QLabel("Drop training files here\nor click Browse")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet("""
            font-size: 14px;
            color: #6c7086;
            padding: 20px;
        """)
        layout.addWidget(self.drop_label)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setAlternatingRowColors(True)
        self.file_list.setMinimumHeight(100)
        self.file_list.setVisible(False)
        layout.addWidget(self.file_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_files)
        btn_layout.addWidget(self.browse_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_files)
        self.clear_btn.setEnabled(False)
        btn_layout.addWidget(self.clear_btn)
        
        btn_layout.addStretch()
        
        self.count_label = QLabel("0 files")
        self.count_label.setStyleSheet("color: #6c7086;")
        btn_layout.addWidget(self.count_label)
        
        layout.addLayout(btn_layout)
    
    def _update_style(self, drag_over: bool):
        """Update frame style based on drag state."""
        if drag_over:
            self.setStyleSheet("""
                DropZone {
                    border: 2px dashed #89b4fa;
                    border-radius: 8px;
                    background-color: #313244;
                }
            """)
        else:
            self.setStyleSheet("""
                DropZone {
                    border: 2px dashed #45475a;
                    border-radius: 8px;
                    background-color: #1e1e2e;
                }
                DropZone:hover {
                    border-color: #585b70;
                }
            """)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter."""
        if event.mimeData().hasUrls():
            # Check if any valid files
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if path.suffix.lower() in self.allowed_extensions:
                        event.acceptProposedAction()
                        self._update_style(True)
                        return
        event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave."""
        self._update_style(False)
    
    def dropEvent(self, event: QDropEvent):
        """Handle file drop."""
        self._update_style(False)
        
        new_files = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                
                # Handle directories
                if path.is_dir():
                    for file_path in path.rglob('*'):
                        if file_path.is_file() and file_path.suffix.lower() in self.allowed_extensions:
                            if file_path not in self._files:
                                new_files.append(file_path)
                
                # Handle individual files
                elif path.is_file() and path.suffix.lower() in self.allowed_extensions:
                    if path not in self._files:
                        new_files.append(path)
        
        if new_files:
            self._files.extend(new_files)
            self._update_file_list()
            self.files_dropped.emit([str(f) for f in new_files])
            self.files_changed.emit()
        
        event.acceptProposedAction()
    
    def _browse_files(self):
        """Open file browser dialog."""
        # Build filter string
        extensions = ' '.join(f'*{ext}' for ext in sorted(self.allowed_extensions))
        filter_str = f"Training Data ({extensions});;All Files (*.*)"
        
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Training Files",
            "",
            filter_str
        )
        
        if files:
            new_files = []
            for file_path in files:
                path = Path(file_path)
                if path not in self._files:
                    new_files.append(path)
            
            if new_files:
                self._files.extend(new_files)
                self._update_file_list()
                self.files_dropped.emit([str(f) for f in new_files])
                self.files_changed.emit()
    
    def _update_file_list(self):
        """Update the file list display."""
        self.file_list.clear()
        
        if self._files:
            self.file_list.setVisible(True)
            self.drop_label.setVisible(False)
            
            for path in self._files:
                item = QListWidgetItem(path.name)
                item.setData(Qt.UserRole, str(path))
                item.setToolTip(str(path))
                
                # Show file size
                try:
                    size = path.stat().st_size
                    if size >= 1024 * 1024:
                        size_str = f"{size / (1024*1024):.1f} MB"
                    elif size >= 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size} B"
                    item.setText(f"{path.name} ({size_str})")
                except Exception:
                    pass  # Intentionally silent
                
                self.file_list.addItem(item)
            
            self.clear_btn.setEnabled(True)
            self.count_label.setText(f"{len(self._files)} files")
        else:
            self.file_list.setVisible(False)
            self.drop_label.setVisible(True)
            self.clear_btn.setEnabled(False)
            self.count_label.setText("0 files")
    
    def clear_files(self):
        """Clear all files."""
        self._files.clear()
        self._update_file_list()
        self.files_changed.emit()
    
    def get_files(self) -> List[str]:
        """Get list of file paths."""
        return [str(f) for f in self._files]
    
    def get_total_size(self) -> int:
        """Get total size of all files in bytes."""
        total = 0
        for path in self._files:
            try:
                total += path.stat().st_size
            except Exception:
                pass  # Intentionally silent
        return total
    
    def add_file(self, file_path: str):
        """Programmatically add a file."""
        path = Path(file_path)
        if path.is_file() and path not in self._files:
            self._files.append(path)
            self._update_file_list()
            self.files_changed.emit()
    
    def remove_selected(self):
        """Remove currently selected file."""
        current = self.file_list.currentItem()
        if current:
            path_str = current.data(Qt.UserRole)
            self._files = [p for p in self._files if str(p) != path_str]
            self._update_file_list()
            self.files_changed.emit()


class TrainingDataDropZone(DropZone):
    """
    Specialized drop zone for training data with preview.
    
    Shows line count and estimated tokens.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        
        # Add stats label
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        self.layout().addWidget(self.stats_label)
        
        self.files_changed.connect(self._update_stats)
    
    def _update_stats(self):
        """Update training data statistics."""
        if not self._files:
            self.stats_label.setText("")
            return
        
        total_lines = 0
        total_chars = 0
        
        for path in self._files:
            try:
                content = path.read_text(encoding='utf-8', errors='ignore')
                total_lines += content.count('\n') + 1
                total_chars += len(content)
            except Exception:
                pass  # Intentionally silent
        
        # Rough token estimate (4 chars per token average)
        est_tokens = total_chars // 4
        
        size_mb = self.get_total_size() / (1024 * 1024)
        
        self.stats_label.setText(
            f"~{total_lines:,} lines | ~{est_tokens:,} tokens | {size_mb:.1f} MB"
        )
