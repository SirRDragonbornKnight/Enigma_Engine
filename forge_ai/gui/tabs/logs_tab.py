"""
Logs Tab - View system, training, and error logs.

Features:
  - Real-time log viewing
  - Filter by log level (DEBUG, INFO, WARNING, ERROR)
  - Search within logs
  - Clear logs
  - Export logs to file
"""

import os
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit, QGroupBox, QCheckBox,
    QFileDialog, QSplitter, QListWidget, QListWidgetItem,
    QTabWidget, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QFileSystemWatcher
from PyQt5.QtGui import QFont, QColor, QTextCharFormat, QTextCursor

from .shared_components import NoScrollComboBox

# Log directories
LOGS_DIR = Path.home() / ".forge_ai" / "logs"
PROJECT_LOGS = Path(__file__).parent.parent.parent.parent / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)


class LogViewerWidget(QWidget):
    """Widget for viewing a single log file."""
    
    def __init__(self, log_path: Path = None, parent=None):
        super().__init__(parent)
        self.log_path = log_path
        self.auto_scroll = True
        self._setup_ui()
        
        if log_path and log_path.exists():
            self._load_log()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Controls
        controls = QHBoxLayout()
        
        self.filter_combo = NoScrollComboBox()
        self.filter_combo.addItems(["All", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.filter_combo.setToolTip("Filter logs by level")
        self.filter_combo.currentTextChanged.connect(self._apply_filter)
        controls.addWidget(QLabel("Level:"))
        controls.addWidget(self.filter_combo)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search logs...")
        self.search_input.textChanged.connect(self._apply_filter)
        controls.addWidget(self.search_input)
        
        self.auto_scroll_cb = QCheckBox("Auto-scroll")
        self.auto_scroll_cb.setChecked(True)
        self.auto_scroll_cb.stateChanged.connect(lambda s: setattr(self, 'auto_scroll', s == Qt.Checked))
        controls.addWidget(self.auto_scroll_cb)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setMaximumWidth(60)
        refresh_btn.setToolTip("Reload log file from disk")
        refresh_btn.clicked.connect(self._load_log)
        controls.addWidget(refresh_btn)
        
        clear_btn = QPushButton("Clear")
        clear_btn.setMaximumWidth(50)
        clear_btn.setToolTip("Clear log display (file is not deleted)")
        clear_btn.clicked.connect(lambda: self.log_display.clear())
        controls.addWidget(clear_btn)
        
        layout.addLayout(controls)
        
        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 9))
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #333;
            }
        """)
        layout.addWidget(self.log_display)
        
        # Stats
        self.stats_label = QLabel("Lines: 0")
        self.stats_label.setStyleSheet("color: #bac2de;")
        layout.addWidget(self.stats_label)
    
    def _load_log(self):
        """Load log file content."""
        if not self.log_path or not self.log_path.exists():
            self.log_display.setPlainText(
                "No log file found yet.\n\n"
                "Logs are created when:\n"
                "- You train a model (training.log)\n"
                "- Errors occur (errors.log)\n"
                "- System events happen (system.log)\n\n"
                "Try using ForgeAI and come back here to see logs."
            )
            return
        
        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            self._all_lines = content.split('\n')
            self._apply_filter()
            
        except Exception as e:
            self.log_display.setPlainText(f"Error loading log: {e}")
    
    def _apply_filter(self):
        """Apply level and search filters."""
        if not hasattr(self, '_all_lines'):
            return
        
        level_filter = self.filter_combo.currentText()
        search_text = self.search_input.text().lower()
        
        filtered = []
        for line in self._all_lines:
            # Level filter
            if level_filter != "All":
                if level_filter not in line.upper():
                    continue
            
            # Search filter
            if search_text and search_text not in line.lower():
                continue
            
            filtered.append(line)
        
        # Apply syntax highlighting
        self.log_display.clear()
        
        for line in filtered:
            self._append_colored_line(line)
        
        self.stats_label.setText(f"Lines: {len(filtered)} / {len(self._all_lines)}")
        
        if self.auto_scroll:
            self.log_display.moveCursor(QTextCursor.End)
    
    def _append_colored_line(self, line: str):
        """Append a line with color based on log level."""
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        fmt = QTextCharFormat()
        
        line_upper = line.upper()
        if "ERROR" in line_upper or "CRITICAL" in line_upper:
            fmt.setForeground(QColor("#f44336"))  # Red
        elif "WARNING" in line_upper or "WARN" in line_upper:
            fmt.setForeground(QColor("#ff9800"))  # Orange
        elif "INFO" in line_upper:
            fmt.setForeground(QColor("#4caf50"))  # Green
        elif "DEBUG" in line_upper:
            fmt.setForeground(QColor("#9e9e9e"))  # Gray
        else:
            fmt.setForeground(QColor("#d4d4d4"))  # Default
        
        cursor.insertText(line + "\n", fmt)
    
    def set_log_path(self, path: Path):
        """Set and load a new log file."""
        self.log_path = path
        self._load_log()


class LogsTab(QWidget):
    """Main Logs tab with multiple log sources."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._start_auto_refresh()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Compact header row with description
        header_row = QHBoxLayout()
        header = QLabel("Logs")
        header.setFont(QFont("Arial", 11, QFont.Bold))
        header_row.addWidget(header)
        
        desc = QLabel("View training, system, and error logs")
        desc.setStyleSheet("color: #bac2de; font-size: 12px;")
        header_row.addWidget(desc)
        header_row.addStretch()
        layout.addLayout(header_row)
        
        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Log file list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        left_layout.addWidget(QLabel("Log Files:"))
        
        self.log_list = QListWidget()
        self.log_list.setMinimumWidth(150)
        self.log_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.log_list.currentItemChanged.connect(self._on_log_selected)
        left_layout.addWidget(self.log_list, 1)  # stretch factor
        
        refresh_list_btn = QPushButton("Refresh List")
        refresh_list_btn.clicked.connect(self._refresh_log_list)
        left_layout.addWidget(refresh_list_btn)
        
        splitter.addWidget(left_widget)
        
        # Right: Log viewer tabs
        self.log_tabs = QTabWidget()
        
        # System log tab
        self.system_viewer = LogViewerWidget()
        self.log_tabs.addTab(self.system_viewer, "System")
        
        # Training log tab
        self.training_viewer = LogViewerWidget()
        self.log_tabs.addTab(self.training_viewer, "Training")
        
        # Error log tab
        self.error_viewer = LogViewerWidget()
        self.log_tabs.addTab(self.error_viewer, "Errors")
        
        # Custom log tab
        self.custom_viewer = LogViewerWidget()
        self.log_tabs.addTab(self.custom_viewer, "Custom")
        
        splitter.addWidget(self.log_tabs)
        splitter.setStretchFactor(0, 1)  # Log list gets less space
        splitter.setStretchFactor(1, 4)  # Log viewer gets more space
        splitter.setSizes([180, 620])
        
        layout.addWidget(splitter, 1)  # Add stretch factor to splitter
        
        # Bottom controls
        bottom = QHBoxLayout()
        
        export_btn = QPushButton("Export Logs")
        export_btn.clicked.connect(self._export_logs)
        bottom.addWidget(export_btn)
        
        clear_all_btn = QPushButton("Clear All Logs")
        clear_all_btn.clicked.connect(self._clear_all_logs)
        bottom.addWidget(clear_all_btn)
        
        bottom.addStretch()
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #bac2de;")
        bottom.addWidget(self.status_label)
        
        layout.addLayout(bottom)
        
        # Initial load
        self._refresh_log_list()
        self._load_default_logs()
    
    def _refresh_log_list(self):
        """Refresh the list of available log files."""
        self.log_list.clear()
        
        # Check multiple log directories
        log_dirs = [LOGS_DIR, PROJECT_LOGS]
        
        for log_dir in log_dirs:
            if not log_dir.exists():
                continue
            
            for log_file in sorted(log_dir.glob("*.log"), reverse=True):
                item = QListWidgetItem(f"{log_file.name}")
                item.setData(Qt.UserRole, str(log_file))
                
                # Color by type
                name_lower = log_file.name.lower()
                if "error" in name_lower:
                    item.setForeground(QColor("#f44336"))
                elif "training" in name_lower:
                    item.setForeground(QColor("#2196f3"))
                
                self.log_list.addItem(item)
            
            # Also check .txt files that might be logs
            for log_file in sorted(log_dir.glob("*.txt"), reverse=True):
                item = QListWidgetItem(f"{log_file.name}")
                item.setData(Qt.UserRole, str(log_file))
                self.log_list.addItem(item)
    
    def _load_default_logs(self):
        """Load default log files."""
        # Try to find system log
        system_log = LOGS_DIR / "system.log"
        if not system_log.exists():
            system_log = PROJECT_LOGS / "forge.log"
        if system_log.exists():
            self.system_viewer.set_log_path(system_log)
        
        # Try to find training log
        training_log = LOGS_DIR / "training.log"
        if not training_log.exists():
            training_log = PROJECT_LOGS / "training.log"
        if training_log.exists():
            self.training_viewer.set_log_path(training_log)
        
        # Try to find error log
        error_log = LOGS_DIR / "errors.log"
        if not error_log.exists():
            error_log = PROJECT_LOGS / "error.log"
        if error_log.exists():
            self.error_viewer.set_log_path(error_log)
    
    def _on_log_selected(self, current, previous):
        """Handle log file selection."""
        if not current:
            return
        
        log_path = Path(current.data(Qt.UserRole))
        self.custom_viewer.set_log_path(log_path)
        self.log_tabs.setCurrentWidget(self.custom_viewer)
        self.status_label.setText(f"Viewing: {log_path.name}")
    
    def _export_logs(self):
        """Export current log to file."""
        current_viewer = self.log_tabs.currentWidget()
        if not isinstance(current_viewer, LogViewerWidget):
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Log", 
            str(Path.home() / "forge_logs_export.txt"),
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(current_viewer.log_display.toPlainText())
                QMessageBox.information(self, "Exported", f"Log exported to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to export: {e}")
    
    def _clear_all_logs(self):
        """Clear all log displays."""
        reply = QMessageBox.question(
            self, "Clear Logs",
            "Clear all log displays? (Files won't be deleted)",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.system_viewer.log_display.clear()
            self.training_viewer.log_display.clear()
            self.error_viewer.log_display.clear()
            self.custom_viewer.log_display.clear()
    
    def _start_auto_refresh(self):
        """Start auto-refresh timer."""
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._auto_refresh)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
    
    def _auto_refresh(self):
        """Auto-refresh current log."""
        current_viewer = self.log_tabs.currentWidget()
        if isinstance(current_viewer, LogViewerWidget) and current_viewer.auto_scroll:
            current_viewer._load_log()


def create_logs_tab(parent=None):
    """Factory function to create logs tab."""
    return LogsTab(parent)
