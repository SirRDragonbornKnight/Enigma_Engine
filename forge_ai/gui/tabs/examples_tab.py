"""
Examples Tab for ForgeAI GUI
==================================

Provides an easy way to browse and run example scripts.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QListWidget, QListWidgetItem, QLabel,
    QSplitter, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtGui import QFont
from pathlib import Path
import os


# Example descriptions
EXAMPLES = {
    "basic_usage.py": {
        "title": "Basic Usage",
        "description": "The simplest way to use Forge for text generation. Shows basic prompts and temperature settings.",
        "category": "Getting Started"
    },
    "chat_example.py": {
        "title": "Chat Interface", 
        "description": "Interactive chat with conversation history. Type messages and get responses.",
        "category": "Getting Started"
    },
    "run_example.py": {
        "title": "Quick Run",
        "description": "Minimal example - just loads the engine and generates one response.",
        "category": "Getting Started"
    },
    "train_example.py": {
        "title": "Train Model",
        "description": "Train your AI on custom data. Shows basic training workflow.",
        "category": "Training"
    },
    "streaming_example.py": {
        "title": "Streaming Output",
        "description": "Stream tokens as they're generated for real-time output.",
        "category": "Inference"
    },
    "api_client_example.py": {
        "title": "API Client",
        "description": "Connect to ForgeAI's REST API from Python. Requires server running.",
        "category": "Networking"
    },
    "multi_device_example.py": {
        "title": "Multi-Device",
        "description": "Run Forge across multiple devices (Pi + PC). Includes server/client modes.",
        "category": "Networking"
    },
    "multi_model_example.py": {
        "title": "Multiple Models",
        "description": "Create and manage multiple named AI models with different purposes.",
        "category": "Advanced"
    },
    "module_system_demo.py": {
        "title": "Module System",
        "description": "Comprehensive demo of the module system - loading, conflicts, hardware detection.",
        "category": "Advanced"
    },
    "local_cloud_example.py": {
        "title": "Local vs Cloud",
        "description": "Understand local-only vs cloud modules. Privacy-first setup.",
        "category": "Advanced"
    },
    "tool_use_example.py": {
        "title": "Tool Use",
        "description": "Let the AI use tools (web search, file operations, etc.) in conversations.",
        "category": "Tools"
    },
    "gif_creation_example.py": {
        "title": "GIF Creation",
        "description": "Generate animated GIFs from text prompts using image generation.",
        "category": "Media"
    },
    "media_editing_example.py": {
        "title": "Media Editing",
        "description": "Edit images, GIFs, and videos - resize, rotate, filters, and more.",
        "category": "Media"
    },
}


class ExamplesTab(QWidget):
    """Tab for browsing and running examples."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.current_file = None
        self.process = None
        self.examples_dir = Path(__file__).parent.parent.parent.parent / "examples"
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Compact header row
        header_row = QHBoxLayout()
        header = QLabel("Example Scripts")
        header.setStyleSheet("font-size: 14px; font-weight: bold;")
        header_row.addWidget(header)
        header_row.addStretch()
        layout.addLayout(header_row)
        
        # Main splitter (takes up most of the space)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Example list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        
        list_label = QLabel("Select an example:")
        list_label.setStyleSheet("color: #888; font-size: 11px;")
        left_layout.addWidget(list_label)
        
        self.example_list = QListWidget()
        self.example_list.itemClicked.connect(self._on_example_selected)
        self.example_list.itemDoubleClicked.connect(self._on_run_example)
        self._populate_examples()
        left_layout.addWidget(self.example_list)
        
        splitter.addWidget(left_widget)
        
        # Right side - Details and code
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        
        # Compact info section (not in a group box)
        self.title_label = QLabel("Select an example")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        right_layout.addWidget(self.title_label)
        
        self.desc_label = QLabel("")
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self.desc_label.setMaximumHeight(40)
        right_layout.addWidget(self.desc_label)
        
        self.category_label = QLabel("")
        self.category_label.setStyleSheet("color: #89b4fa; font-size: 10px;")
        right_layout.addWidget(self.category_label)
        
        # Code preview (main area)
        code_label = QLabel("Code Preview:")
        code_label.setStyleSheet("font-size: 11px; color: #888;")
        right_layout.addWidget(code_label)
        
        self.code_view = QTextEdit()
        self.code_view.setReadOnly(True)
        self.code_view.setFont(QFont("Monospace", 9))
        self.code_view.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
            }
        """)
        right_layout.addWidget(self.code_view, stretch=1)  # Give code view the most space
        
        # Compact buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)
        
        self.run_btn = QPushButton("Run")
        self.run_btn.setToolTip("Run the selected example")
        self.run_btn.clicked.connect(self._on_run_example)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                font-weight: bold;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: #94e2d5;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #6c7086;
            }
        """)
        btn_layout.addWidget(self.run_btn)
        
        self.open_btn = QPushButton("Open")
        self.open_btn.setToolTip("Open example file in editor")
        self.open_btn.clicked.connect(self._on_open_file)
        self.open_btn.setEnabled(False)
        self.open_btn.setStyleSheet("padding: 6px 12px;")
        btn_layout.addWidget(self.open_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip("Stop running example")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8;
                color: #1e1e2e;
                padding: 6px 12px;
            }
        """)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()
        
        right_layout.addLayout(btn_layout)
        
        # Compact output area
        output_label = QLabel("Output:")
        output_label.setStyleSheet("font-size: 11px; color: #888;")
        right_layout.addWidget(output_label)
        
        self.output_view = QTextEdit()
        self.output_view.setReadOnly(True)
        self.output_view.setFont(QFont("Monospace", 9))
        self.output_view.setMaximumHeight(100)
        self.output_view.setStyleSheet("""
            QTextEdit {
                background-color: #11111b;
                color: #a6e3a1;
                border: 1px solid #45475a;
                border-radius: 4px;
            }
        """)
        right_layout.addWidget(self.output_view)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([200, 600])
        
        layout.addWidget(splitter, stretch=1)  # Give splitter most space
        
    def _populate_examples(self):
        """Populate the example list."""
        # Group by category
        categories = {}
        for filename, info in EXAMPLES.items():
            cat = info["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((filename, info))
        
        # Add to list grouped by category
        for category in ["Getting Started", "Training", "Inference", "Networking", "Advanced", "Tools", "Media"]:
            if category in categories:
                # Category header
                header_item = QListWidgetItem(f"-- {category} --")
                header_item.setFlags(Qt.NoItemFlags)
                header_item.setForeground(Qt.gray)
                self.example_list.addItem(header_item)
                
                # Examples in category
                for filename, info in categories[category]:
                    item = QListWidgetItem(f"  {info['title']}")
                    item.setData(Qt.UserRole, filename)
                    self.example_list.addItem(item)
    
    def _on_example_selected(self, item):
        """Handle example selection."""
        filename = item.data(Qt.UserRole)
        if not filename:
            return
            
        self.current_file = filename
        info = EXAMPLES.get(filename, {})
        
        # Update info
        self.title_label.setText(info.get("title", filename))
        self.desc_label.setText(info.get("description", ""))
        self.category_label.setText(f"Category: {info.get('category', 'Other')}")
        
        # Load code
        file_path = self.examples_dir / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                code = f.read()
            self.code_view.setPlainText(code)
            self.run_btn.setEnabled(True)
            self.open_btn.setEnabled(True)
        else:
            self.code_view.setPlainText(f"File not found: {file_path}")
            self.run_btn.setEnabled(False)
            self.open_btn.setEnabled(False)
    
    def _on_run_example(self, item=None):
        """Run the selected example."""
        if not self.current_file:
            return
            
        file_path = self.examples_dir / self.current_file
        if not file_path.exists():
            QMessageBox.warning(self, "Error", f"File not found: {file_path}")
            return
        
        self.output_view.clear()
        self.output_view.append(f"Running {self.current_file}...\n")
        
        # Start process
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)
        
        self.process.start("python", [str(file_path)])
        
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
    
    def _on_stdout(self):
        """Handle stdout from process."""
        if self.process:
            data = self.process.readAllStandardOutput()
            text = bytes(data).decode('utf-8', errors='replace')
            self.output_view.append(text)
    
    def _on_stderr(self):
        """Handle stderr from process."""
        if self.process:
            data = self.process.readAllStandardError()
            text = bytes(data).decode('utf-8', errors='replace')
            self.output_view.append(f"<span style='color: #f38ba8;'>{text}</span>")
    
    def _on_finished(self, exit_code, exit_status):
        """Handle process finished."""
        self.output_view.append(f"\n--- Finished (exit code: {exit_code}) ---")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.process = None
    
    def _on_stop(self):
        """Stop the running process."""
        if self.process:
            self.process.kill()
            self.output_view.append("\n--- Stopped by user ---")
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def _on_open_file(self):
        """Open the file in the system editor."""
        if not self.current_file:
            return
            
        file_path = self.examples_dir / self.current_file
        if file_path.exists():
            from .output_helpers import open_in_default_viewer
            try:
                open_in_default_viewer(file_path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open file: {e}")


def create_examples_tab(parent):
    """Create the examples tab."""
    return ExamplesTab(parent)
