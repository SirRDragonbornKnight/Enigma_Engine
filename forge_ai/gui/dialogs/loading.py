# type: ignore
"""
Loading Dialog - Shows progress when loading models and other resources.

Extracted from enhanced_window.py for better organization.
"""

import time
from typing import Optional, List, Dict

try:
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QProgressBar, QTextEdit, QWidget, QApplication
    )
    from PyQt5.QtCore import Qt, QTimer
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QDialog = object


class ModelLoadingDialog(QDialog):
    """Loading dialog with support for multiple loading items (model, avatar, etc.)."""
    
    cancelled = False  # Flag to track cancellation
    
    def __init__(self, model_name: str = None, parent=None, show_terminal: bool = False, 
                 loading_items: list = None):
        """
        Initialize loading dialog.
        
        Args:
            model_name: Name of model being loaded (for backwards compatibility)
            parent: Parent widget
            show_terminal: Show log output
            loading_items: List of dicts with 'name', 'type' (model/avatar/other), 'icon'
        """
        super().__init__(parent)
        self.setWindowTitle("Loading ForgeAI")
        self.setFixedSize(450, 300)
        self.setModal(False)  # Non-modal so user can move the main window
        self.cancelled = False
        self.show_terminal = show_terminal
        self._log_lines = []
        self._current_progress = 0
        self._target_progress = 0
        self._drag_pos = None
        
        # Track loading items - each has: name, type, icon, status, progress
        self._loading_items = []
        if loading_items:
            for item in loading_items:
                self._loading_items.append({
                    'name': item.get('name', 'Unknown'),
                    'type': item.get('type', 'other'),
                    'icon': item.get('icon', '📦'),
                    'status': 'Waiting...',
                    'progress': 0,
                    'done': False
                })
        elif model_name:
            # Backwards compatible - just model loading
            self._loading_items.append({
                'name': model_name,
                'type': 'model',
                'icon': '🧠',
                'status': 'Initializing...',
                'progress': 0,
                'done': False
            })
        
        self._setup_styles()
        self._setup_ui()
        self._setup_timers()
        
        # Adjust size based on number of items
        base_height = 200
        item_height = 50 * len(self._loading_items)
        self.setFixedSize(450, min(400, base_height + item_height))
    
    def _setup_styles(self):
        """Apply dark styling to the dialog."""
        # Import shared styles if available, otherwise use inline
        try:
            from ..styles import COLORS
            base = COLORS['base']
            blue = COLORS['blue']
            surface0 = COLORS['surface0']
            surface1 = COLORS['surface1']
            text = COLORS['text']
            green = COLORS['green']
            red = COLORS['red']
            crust = COLORS['crust']
            sky = COLORS['sky']
        except ImportError:
            base = "#1e1e2e"
            blue = "#89b4fa"
            surface0 = "#313244"
            surface1 = "#45475a"
            text = "#cdd6f4"
            green = "#a6e3a1"
            red = "#f38ba8"
            crust = "#11111b"
            sky = "#74c7ec"
        
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {base};
                border: 2px solid {blue};
                border-radius: 12px;
            }}
            QLabel {{
                color: {text};
            }}
            QProgressBar {{
                background-color: {surface0};
                border: none;
                border-radius: 8px;
                height: 16px;
                text-align: center;
                color: white;
                font-size: 10px;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {blue}, stop:0.5 {sky}, stop:1 {blue});
                border-radius: 8px;
            }}
            QPushButton {{
                background-color: {surface1};
                color: {text};
                border: none;
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {red};
            }}
            QPushButton#terminal_btn {{
                background-color: {surface0};
            }}
            QPushButton#terminal_btn:hover {{
                background-color: {surface1};
            }}
            QTextEdit {{
                background-color: {crust};
                color: {green};
                border: 1px solid {surface1};
                border-radius: 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                padding: 4px;
            }}
        """)
    
    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(8)
        
        # Title
        title_text = "⏳ Loading ForgeAI"
        self.title_label = QLabel(title_text)
        self.title_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #89b4fa;")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Container for loading item rows
        self._items_container = QWidget()
        self._items_layout = QVBoxLayout(self._items_container)
        self._items_layout.setContentsMargins(0, 5, 0, 5)
        self._items_layout.setSpacing(6)
        
        # Create UI for each loading item
        self._item_widgets = []
        for item in self._loading_items:
            row = self._create_item_row(item)
            self._item_widgets.append(row)
            self._items_layout.addWidget(row['container'])
        
        layout.addWidget(self._items_container)
        
        # Overall progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Overall: %p%")
        layout.addWidget(self.progress)
        
        # Activity indicator (animated dots)
        self.activity_label = QLabel("●○○")
        self.activity_label.setStyleSheet("font-size: 14px; color: #74c7ec;")
        self.activity_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.activity_label)
        
        # Status label (for backwards compatibility)
        self.status_label = QLabel("Starting...")
        self.status_label.setStyleSheet("font-size: 11px; color: #a6adc8;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Terminal output area (optional)
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setMaximumHeight(80)
        self.terminal.setVisible(self.show_terminal)
        layout.addWidget(self.terminal)
        
        # Buttons row
        btn_layout = QHBoxLayout()
        
        # Toggle terminal button
        self.terminal_btn = QPushButton("📺 Log")
        self.terminal_btn.setObjectName("terminal_btn")
        self.terminal_btn.clicked.connect(self._toggle_terminal)
        btn_layout.addWidget(self.terminal_btn)
        
        btn_layout.addStretch()
        
        # Cancel button
        self.cancel_btn = QPushButton("⛔ Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8;
                color: #1e1e2e;
                font-weight: bold;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #ef4444;
            }
        """)
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(btn_layout)
    
    def _setup_timers(self):
        """Initialize animation timers."""
        self._dot_state = 0
        
        # Animation timer for activity dots
        self._activity_timer = QTimer(self)
        self._activity_timer.timeout.connect(self._animate_dots)
        self._activity_timer.start(300)
        
        # Smooth progress animation timer
        self._progress_timer = QTimer(self)
        self._progress_timer.timeout.connect(self._animate_progress)
        self._progress_timer.start(30)
    
    def _create_item_row(self, item: dict) -> dict:
        """Create a row widget for a loading item."""
        container = QWidget()
        container.setStyleSheet("""
            QWidget {
                background-color: #313244;
                border-radius: 8px;
                padding: 4px;
            }
        """)
        row_layout = QHBoxLayout(container)
        row_layout.setContentsMargins(10, 6, 10, 6)
        row_layout.setSpacing(8)
        
        # Icon + Name
        icon_label = QLabel(item['icon'])
        icon_label.setStyleSheet("font-size: 16px; background: transparent;")
        row_layout.addWidget(icon_label)
        
        name_label = QLabel(item['name'])
        name_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #cdd6f4; background: transparent;")
        name_label.setMinimumWidth(120)
        row_layout.addWidget(name_label)
        
        # Status
        status_label = QLabel(item['status'])
        status_label.setStyleSheet("font-size: 10px; color: #a6adc8; background: transparent;")
        status_label.setMinimumWidth(100)
        row_layout.addWidget(status_label)
        
        # Mini progress
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(False)
        progress_bar.setFixedHeight(8)
        progress_bar.setFixedWidth(80)
        progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #45475a;
                border-radius: 4px;
                height: 8px;
            }
            QProgressBar::chunk {
                background-color: #89b4fa;
                border-radius: 4px;
            }
        """)
        row_layout.addWidget(progress_bar)
        
        # Check mark (hidden until done)
        done_label = QLabel("✓")
        done_label.setStyleSheet("font-size: 14px; color: #a6e3a1; font-weight: bold; background: transparent;")
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
    
    def add_loading_item(self, name: str, item_type: str = 'other', icon: str = '📦'):
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
        base_height = 200
        item_height = 50 * len(self._loading_items)
        self.setFixedSize(450, min(400, base_height + item_height))
        QApplication.processEvents()
    
    def set_item_status(self, index: int, status: str, progress: int):
        """Update status and progress for a specific loading item."""
        if 0 <= index < len(self._item_widgets):
            self._item_widgets[index]['status'].setText(status)
            self._item_widgets[index]['progress'].setValue(progress)
            self._loading_items[index]['status'] = status
            self._loading_items[index]['progress'] = progress
            
            # Update overall progress
            total_progress = sum(item['progress'] for item in self._loading_items)
            avg_progress = total_progress // len(self._loading_items) if self._loading_items else 0
            self._target_progress = avg_progress
            
            self.log(f"[{self._loading_items[index]['name']}] {status}")
            QApplication.processEvents()
    
    def set_item_done(self, index: int):
        """Mark a loading item as complete."""
        if 0 <= index < len(self._item_widgets):
            self._item_widgets[index]['status'].setText("Complete")
            self._item_widgets[index]['status'].setStyleSheet("font-size: 10px; color: #a6e3a1; background: transparent;")
            self._item_widgets[index]['progress'].setValue(100)
            self._item_widgets[index]['done'].setVisible(True)
            self._loading_items[index]['done'] = True
            self._loading_items[index]['progress'] = 100
            
            # Update overall progress
            total_progress = sum(item['progress'] for item in self._loading_items)
            avg_progress = total_progress // len(self._loading_items) if self._loading_items else 0
            self._target_progress = avg_progress
            QApplication.processEvents()
    
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
            self.terminal_btn.setText("📺 Hide")
            # Expand dialog height
            current_height = self.height()
            self.setFixedSize(450, current_height + 80)
        else:
            self.terminal_btn.setText("📺 Log")
            current_height = self.height()
            self.setFixedSize(450, current_height - 80)
    
    def _on_cancel(self):
        """Handle cancel button click."""
        self.cancelled = True
        self.status_label.setText("⏹ Cancelling...")
        self.status_label.setStyleSheet("font-size: 11px; color: #f38ba8; font-weight: bold;")
        self.cancel_btn.setText("⏹ ...")
        self.cancel_btn.setEnabled(False)
        self.log("❌ Cancelled by user")
        QApplication.processEvents()
    
    def is_cancelled(self) -> bool:
        """Check if loading was cancelled."""
        QApplication.processEvents()  # Allow UI to update
        return self.cancelled
    
    def log(self, text: str):
        """Add a log line to terminal output."""
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
        """Update status text and progress (backwards compatible - updates first item)."""
        self.status_label.setText(text)
        self._target_progress = progress  # Animate towards this
        
        # Also update first loading item if it exists
        if self._loading_items and self._item_widgets:
            self._item_widgets[0]['status'].setText(text)
            self._item_widgets[0]['progress'].setValue(progress)
            self._loading_items[0]['progress'] = progress
            
            # Mark done if at 100%
            if progress >= 100:
                self.set_item_done(0)
        
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
