# type: ignore
"""
Loading Dialog - Shows progress when loading activated elements.

Shows a clean list of what's being loaded (model, avatar, modules, etc.)
with individual progress indicators for each activated element.
"""

import time
from typing import Optional, List, Dict

try:
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QProgressBar, QTextEdit, QWidget, QApplication, QScrollArea,
        QFrame
    )
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QDialog = object


class ModelLoadingDialog(QDialog):
    """Loading dialog showing activated elements and their loading status."""
    
    cancelled = False
    
    def __init__(self, model_name: str = None, parent=None, show_terminal: bool = False, 
                 loading_items: list = None):
        """
        Initialize loading dialog.
        
        Args:
            model_name: Name of model being loaded (for backwards compatibility)
            parent: Parent widget
            show_terminal: Show log output (unused, kept for compatibility)
            loading_items: List of dicts with 'name', 'type', 'icon'
        """
        super().__init__(parent)
        self.setWindowTitle("Loading")
        # Always on top + frameless + stay on top of parent
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setModal(True)  # Block interaction with parent until closed
        self.cancelled = False
        self._log_lines = []
        self._drag_pos = None
        
        # Track loading items
        self._loading_items = []
        if loading_items:
            for item in loading_items:
                self._loading_items.append({
                    'name': item.get('name', 'Unknown'),
                    'type': item.get('type', 'other'),
                    'icon': item.get('icon', '>'),
                    'status': 'Waiting...',
                    'progress': 0,
                    'done': False
                })
        elif model_name:
            # Backwards compatible - just model loading
            self._loading_items.append({
                'name': model_name,
                'type': 'model',
                'icon': '>',
                'status': 'Loading...',
                'progress': 0,
                'done': False
            })
        
        self._setup_styles()
        self._setup_ui()
        self._setup_timers()
        
        # Calculate size based on number of items
        num_items = len(self._loading_items)
        base_height = 100
        item_height = 40 * max(1, num_items)
        total_height = min(400, base_height + item_height)
        self.setFixedSize(340, total_height)
    
    def _setup_styles(self):
        """Apply dark styling to the dialog."""
        try:
            from ..styles import COLORS
            base = COLORS['base']
            blue = COLORS['blue']
            surface0 = COLORS['surface0']
            surface1 = COLORS['surface1']
            text = COLORS['text']
            green = COLORS['green']
            red = COLORS['red']
        except ImportError:
            base = "#1e1e2e"
            blue = "#89b4fa"
            surface0 = "#313244"
            surface1 = "#45475a"
            text = "#cdd6f4"
            green = "#a6e3a1"
            red = "#f38ba8"
        
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {base};
                border: 1px solid {surface1};
                border-radius: 10px;
            }}
            QLabel {{
                color: {text};
                background: transparent;
            }}
            QProgressBar {{
                background-color: {surface0};
                border: none;
                border-radius: 4px;
                height: 6px;
            }}
            QProgressBar::chunk {{
                background-color: {blue};
                border-radius: 4px;
            }}
            QPushButton {{
                background-color: transparent;
                color: {red};
                border: none;
                font-weight: bold;
            }}
            QPushButton:hover {{
                color: white;
            }}
        """)
    
    def _setup_ui(self):
        """Build the dialog UI - simple list of loading elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)
        
        # Title row
        title_row = QHBoxLayout()
        
        self.spinner_label = QLabel(".")
        self.spinner_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #89b4fa;")
        self.spinner_label.setFixedWidth(24)
        title_row.addWidget(self.spinner_label)
        
        self.title_label = QLabel("Loading")
        self.title_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #89b4fa;")
        title_row.addWidget(self.title_label)
        
        title_row.addStretch()
        
        # Cancel button
        self.cancel_btn = QPushButton("X")
        self.cancel_btn.setFixedSize(20, 20)
        self.cancel_btn.clicked.connect(self._on_cancel)
        title_row.addWidget(self.cancel_btn)
        
        layout.addLayout(title_row)
        
        # Container for loading items
        self._items_container = QWidget()
        self._items_layout = QVBoxLayout(self._items_container)
        self._items_layout.setContentsMargins(0, 4, 0, 4)
        self._items_layout.setSpacing(4)
        
        # Create UI for each loading item
        self._item_widgets = []
        for item in self._loading_items:
            row = self._create_item_row(item)
            self._item_widgets.append(row)
            self._items_layout.addWidget(row['container'])
        
        layout.addWidget(self._items_container)
        
        # Status text at bottom (for backwards compatibility)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 12px; color: #6c7086;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Hidden terminal for log compatibility
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.hide()
        
        # Dummy progress for backwards compatibility
        self.progress = QProgressBar()
        self.progress.hide()
    
    def _setup_timers(self):
        """Initialize animation timers."""
        self._spinner_state = 0
        self._spinner_chars = [".", "..", "..."]
        
        # Spinner animation
        self._spinner_timer = QTimer(self)
        self._spinner_timer.timeout.connect(self._animate_spinner)
        self._spinner_timer.start(400)
    
    def _animate_spinner(self):
        """Animate the spinner dots."""
        self._spinner_state = (self._spinner_state + 1) % len(self._spinner_chars)
        self.spinner_label.setText(self._spinner_chars[self._spinner_state])
    
    def _create_item_row(self, item: dict) -> dict:
        """Create a compact row widget for a loading item."""
        container = QWidget()
        container.setStyleSheet("""
            QWidget {
                background-color: #313244;
                border-radius: 6px;
            }
        """)
        row_layout = QHBoxLayout(container)
        row_layout.setContentsMargins(10, 6, 10, 6)
        row_layout.setSpacing(8)
        
        # Icon
        icon_label = QLabel(item['icon'])
        icon_label.setStyleSheet("font-size: 12px;")
        icon_label.setFixedWidth(20)
        row_layout.addWidget(icon_label)
        
        # Name
        name_label = QLabel(item['name'])
        name_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #cdd6f4;")
        name_label.setMinimumWidth(100)
        row_layout.addWidget(name_label, 1)
        
        # Status text
        status_label = QLabel(item['status'])
        status_label.setStyleSheet("font-size: 12px; color: #a6adc8;")
        status_label.setAlignment(Qt.AlignRight)
        row_layout.addWidget(status_label)
        
        # Progress bar
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(False)
        progress_bar.setFixedSize(60, 6)
        row_layout.addWidget(progress_bar)
        
        # Done checkmark (hidden until complete)
        done_label = QLabel("OK")
        done_label.setStyleSheet("font-size: 12px; color: #a6e3a1; font-weight: bold;")
        done_label.setFixedWidth(20)
        done_label.setVisible(False)
        row_layout.addWidget(done_label)
        
        return {
            'container': container,
            'icon': icon_label,
            'name': name_label,
            'status': status_label,
            'progress': progress_bar,
            'done': done_label
        }
    
    def add_loading_item(self, name: str, item_type: str = 'other', icon: str = '[*]'):
        """Dynamically add a new loading item."""
        item = {
            'name': name,
            'type': item_type,
            'icon': icon,
            'status': 'Waiting...',
            'progress': 0,
            'done': False
        }
        self._loading_items.append(item)
        row = self._create_item_row(item)
        self._item_widgets.append(row)
        self._items_layout.addWidget(row['container'])
        
        # Resize dialog
        num_items = len(self._loading_items)
        base_height = 100
        item_height = 40 * num_items
        self.setFixedSize(340, min(400, base_height + item_height))
        QApplication.processEvents()
    
    def set_item_status(self, index: int, status: str, progress: int):
        """Update status and progress for a specific loading item."""
        if 0 <= index < len(self._item_widgets):
            self._item_widgets[index]['status'].setText(status)
            self._item_widgets[index]['progress'].setValue(progress)
            self._loading_items[index]['status'] = status
            self._loading_items[index]['progress'] = progress
            QApplication.processEvents()
    
    def set_item_done(self, index: int):
        """Mark a loading item as complete."""
        if 0 <= index < len(self._item_widgets):
            self._item_widgets[index]['status'].setText("Ready")
            self._item_widgets[index]['status'].setStyleSheet("font-size: 12px; color: #a6e3a1;")
            self._item_widgets[index]['progress'].setValue(100)
            self._item_widgets[index]['progress'].hide()
            self._item_widgets[index]['done'].setVisible(True)
            self._loading_items[index]['done'] = True
            self._loading_items[index]['progress'] = 100
            QApplication.processEvents()
    
    def _on_cancel(self):
        """Handle cancel button click."""
        self.cancelled = True
        self.status_label.setText("Cancelling...")
        self.cancel_btn.setEnabled(False)
        QApplication.processEvents()
    
    def is_cancelled(self) -> bool:
        """Check if loading was cancelled."""
        QApplication.processEvents()
        return self.cancelled
    
    def log(self, text: str):
        """Add a log line (kept for compatibility)."""
        timestamp = time.strftime("%H:%M:%S")
        self._log_lines.append(f"[{timestamp}] {text}")
    
    def set_status(self, text: str, progress: int):
        """Update status text and progress (backwards compatible)."""
        self.status_label.setText(text)
        
        # Also update first loading item if it exists
        if self._loading_items and self._item_widgets:
            # Extract short status from full text
            short_status = text.replace("[OK] ", "").split("...")[0]
            if len(short_status) > 20:
                short_status = short_status[:18] + "..."
            
            self._item_widgets[0]['status'].setText(short_status)
            self._item_widgets[0]['progress'].setValue(progress)
            self._loading_items[0]['progress'] = progress
            
            # Mark done if at 100%
            if progress >= 100:
                self.set_item_done(0)
        
        self.log(text)
        QApplication.processEvents()
    
    def close(self):
        """Clean up timers before closing."""
        if hasattr(self, '_spinner_timer'):
            self._spinner_timer.stop()
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
