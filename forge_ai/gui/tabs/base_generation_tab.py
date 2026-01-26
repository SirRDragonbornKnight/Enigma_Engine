"""
Base Generation Tab - Shared foundation for all AI generation tabs.

This base class provides:
  - Consistent header styling
  - Standard progress/status layout
  - Provider management patterns
  - Common button styles
  - Auto-open functionality
  - Standardized error handling

All generation tabs (Image, Code, Video, Audio, Embeddings, 3D) should inherit from this.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QProgressBar, QGroupBox, QFrame,
        QMessageBox, QSizePolicy
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from .shared_components import NoScrollComboBox, disable_scroll_on_combos
from .output_helpers import open_file_in_explorer, open_in_default_viewer, open_folder


# =============================================================================
# Consistent Style Constants
# =============================================================================

# Button styles - use these for consistency across all tabs
BUTTON_STYLE_PRIMARY = """
    QPushButton {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
        font-size: 11px;
    }
    QPushButton:hover {
        background-color: #2980b9;
    }
    QPushButton:pressed {
        background-color: #1c5980;
    }
    QPushButton:disabled {
        background-color: #45475a;
        color: #6c7086;
    }
"""

BUTTON_STYLE_SECONDARY = """
    QPushButton {
        background-color: #45475a;
        color: #cdd6f4;
        border: none;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 11px;
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
"""

BUTTON_STYLE_SUCCESS = """
    QPushButton {
        background-color: #a6e3a1;
        color: #1e1e2e;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
        font-size: 11px;
    }
    QPushButton:hover {
        background-color: #94e2d5;
    }
    QPushButton:pressed {
        background-color: #74c7a0;
    }
    QPushButton:disabled {
        background-color: #45475a;
        color: #6c7086;
    }
"""

BUTTON_STYLE_DANGER = """
    QPushButton {
        background-color: #f38ba8;
        color: #1e1e2e;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
        font-size: 11px;
    }
    QPushButton:hover {
        background-color: #f5c2e7;
    }
    QPushButton:pressed {
        background-color: #d06080;
    }
    QPushButton:disabled {
        background-color: #45475a;
        color: #6c7086;
    }
"""

HEADER_STYLE = """
    QLabel {
        font-size: 16px;
        font-weight: bold;
        color: #cdd6f4;
        padding: 4px 0;
    }
"""

STATUS_LABEL_STYLE = """
    QLabel {
        color: #a6adc8;
        font-size: 11px;
        padding: 2px 4px;
    }
"""

PROGRESS_BAR_STYLE = """
    QProgressBar {
        border: 1px solid #45475a;
        border-radius: 4px;
        background-color: #313244;
        height: 8px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #89b4fa;
        border-radius: 3px;
    }
"""

PREVIEW_FRAME_STYLE = """
    QLabel {
        background-color: #2d2d2d;
        border: 1px solid #45475a;
        border-radius: 4px;
        padding: 8px;
    }
"""

GROUP_BOX_STYLE = """
    QGroupBox {
        font-weight: bold;
        border: 1px solid #45475a;
        border-radius: 6px;
        margin-top: 8px;
        padding-top: 8px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        color: #89b4fa;
    }
"""


# =============================================================================
# Base Generation Worker
# =============================================================================

class BaseGenerationWorker(QThread):
    """
    Base worker class for background generation tasks.
    
    Subclasses should implement run() and emit:
      - progress(int): Progress percentage 0-100
      - finished(dict): Result dictionary with 'success', 'error', 'path', etc.
    """
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    status = pyqtSignal(str)  # Status message updates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_requested = False
    
    def request_stop(self):
        """Request the worker to stop gracefully."""
        self._stop_requested = True
    
    def is_stop_requested(self) -> bool:
        """Check if stop was requested."""
        return self._stop_requested
    
    def emit_cancelled(self):
        """Emit a cancelled result."""
        self.finished.emit({"success": False, "error": "Cancelled by user"})
    
    def run(self):
        """Override this in subclasses."""
        raise NotImplementedError("Subclasses must implement run()")


# =============================================================================
# Base Generation Tab
# =============================================================================

class BaseGenerationTab(QWidget):
    """
    Base class for all AI generation tabs.
    
    Provides:
      - Consistent layout structure
      - Header with title
      - Preview/output area
      - Progress bar and status label
      - Provider selection (using NoScrollComboBox)
      - Common button patterns
      - Auto-open functionality
      - Standard error handling
    
    Subclasses should:
      1. Call super().__init__() 
      2. Set self.tab_title and self.output_dir
      3. Override setup_preview_area(), setup_settings(), setup_input_area(), setup_buttons()
      4. Implement _generate() method
    """
    
    def __init__(self, parent=None, tab_title: str = "Generation", 
                 output_dir: Optional[Path] = None):
        if not HAS_PYQT:
            raise ImportError("PyQt5 is required for generation tabs")
        
        super().__init__(parent)
        self.main_window = parent
        self.tab_title = tab_title
        self.output_dir = output_dir or Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.worker: Optional[BaseGenerationWorker] = None
        self.last_output_path: Optional[str] = None
        
        # Setup base UI structure
        self._setup_base_ui()
        
        # Disable scroll on all combo boxes
        disable_scroll_on_combos(self)
    
    def _setup_base_ui(self):
        """Set up the base UI structure. Called automatically."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 1. Header
        self._create_header()
        
        # 2. Preview/Output area (subclass implements)
        self.setup_preview_area()
        
        # 3. Progress and Status
        self._create_progress_section()
        
        # 4. Settings (provider, options - subclass implements)
        self.setup_settings()
        
        # 5. Input area (subclass implements)
        self.setup_input_area()
        
        # 6. Action buttons
        self._create_button_section()
    
    def _create_header(self):
        """Create the header with title."""
        header_layout = QHBoxLayout()
        
        self.header_label = QLabel(self.tab_title)
        self.header_label.setStyleSheet(HEADER_STYLE)
        header_layout.addWidget(self.header_label)
        
        header_layout.addStretch()
        
        # Subclasses can add widgets to header
        self.header_layout = header_layout
        self.main_layout.addLayout(header_layout)
    
    def _create_progress_section(self):
        """Create progress bar and status label."""
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        self.main_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(STATUS_LABEL_STYLE)
        self.main_layout.addWidget(self.status_label)
    
    def _create_button_section(self):
        """Create the action buttons section."""
        btn_layout = QHBoxLayout()
        
        # Generate button (primary action)
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setStyleSheet(BUTTON_STYLE_PRIMARY)
        self.generate_btn.setMinimumWidth(100)
        self.generate_btn.clicked.connect(self._on_generate_clicked)
        self.generate_btn.setToolTip("Start generation (Ctrl+Enter)")
        btn_layout.addWidget(self.generate_btn)
        
        # Stop button (hidden by default)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(BUTTON_STYLE_DANGER)
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.stop_btn.setToolTip("Stop the current generation")
        btn_layout.addWidget(self.stop_btn)
        
        # Let subclasses add more buttons
        self.setup_buttons(btn_layout)
        
        # Output folder button (common)
        self.folder_btn = QPushButton("Output Folder")
        self.folder_btn.setStyleSheet(BUTTON_STYLE_SECONDARY)
        self.folder_btn.clicked.connect(self._open_output_folder)
        self.folder_btn.setToolTip(f"Open the output folder: {self.output_dir}")
        btn_layout.addWidget(self.folder_btn)
        
        btn_layout.addStretch()
        self.main_layout.addLayout(btn_layout)
        self.button_layout = btn_layout
    
    # =========================================================================
    # Methods for subclasses to override
    # =========================================================================
    
    def setup_preview_area(self):
        """
        Override this to set up the preview/output area.
        
        Example:
            self.preview_label = QLabel("Output will appear here")
            self.preview_label.setStyleSheet(PREVIEW_FRAME_STYLE)
            self.preview_label.setAlignment(Qt.AlignCenter)
            self.preview_label.setMinimumHeight(200)
            self.main_layout.addWidget(self.preview_label, stretch=1)
        """
        pass
    
    def setup_settings(self):
        """
        Override this to set up provider selection and options.
        
        Use NoScrollComboBox for all dropdowns.
        
        Example:
            settings_layout = QHBoxLayout()
            settings_layout.addWidget(QLabel("Provider:"))
            self.provider_combo = NoScrollComboBox()
            self.provider_combo.addItems(['Local', 'Cloud API'])
            settings_layout.addWidget(self.provider_combo)
            settings_layout.addStretch()
            self.main_layout.addLayout(settings_layout)
        """
        pass
    
    def setup_input_area(self):
        """
        Override this to set up the input area (prompt, text, etc.).
        
        Example:
            input_group = QGroupBox("Input")
            input_layout = QVBoxLayout()
            self.prompt_input = QTextEdit()
            self.prompt_input.setMaximumHeight(80)
            self.prompt_input.setPlaceholderText("Enter your prompt...")
            input_layout.addWidget(self.prompt_input)
            input_group.setLayout(input_layout)
            self.main_layout.addWidget(input_group)
        """
        pass
    
    def setup_buttons(self, btn_layout: QHBoxLayout):
        """
        Override this to add custom buttons.
        
        Example:
            self.copy_btn = QPushButton("Copy")
            self.copy_btn.setStyleSheet(BUTTON_STYLE_SECONDARY)
            self.copy_btn.clicked.connect(self._copy_output)
            btn_layout.addWidget(self.copy_btn)
        """
        pass
    
    def _generate(self):
        """
        Override this to implement the generation logic.
        
        Should:
        1. Get input from UI
        2. Create and start a worker thread
        3. Connect worker signals to _on_progress, _on_generation_complete
        
        Example:
            prompt = self.prompt_input.toPlainText().strip()
            if not prompt:
                self.show_error("Please enter a prompt")
                return
            
            self.worker = MyGenerationWorker(prompt, ...)
            self.worker.progress.connect(self._on_progress)
            self.worker.finished.connect(self._on_generation_complete)
            self.worker.start()
        """
        raise NotImplementedError("Subclasses must implement _generate()")
    
    def get_provider_name(self) -> str:
        """
        Override this if you have a provider combo box.
        
        Returns the internal provider name (e.g., 'local', 'openai').
        """
        return 'local'
    
    # =========================================================================
    # Common event handlers
    # =========================================================================
    
    def _on_generate_clicked(self):
        """Handle generate button click."""
        self._start_generation()
    
    def _on_stop_clicked(self):
        """Handle stop button click."""
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.set_status("Stopping...")
    
    def _start_generation(self):
        """Start the generation process."""
        # Show loading state
        self.generate_btn.setEnabled(False)
        self.stop_btn.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.set_status("Starting...")
        
        # Call subclass implementation
        try:
            self._generate()
        except Exception as e:
            self._on_generation_complete({"success": False, "error": str(e)})
    
    def _on_progress(self, value: int):
        """Handle progress updates."""
        self.progress_bar.setValue(value)
    
    def _on_status(self, message: str):
        """Handle status message updates."""
        self.set_status(message)
    
    def _on_generation_complete(self, result: dict):
        """Handle generation completion."""
        # Reset UI state
        self.generate_btn.setEnabled(True)
        self.stop_btn.setVisible(False)
        self.progress_bar.setVisible(False)
        
        if result.get("success"):
            duration = result.get("duration", 0)
            path = result.get("path", "")
            self.last_output_path = path
            
            # Update status
            self.set_status(f"Generated in {duration:.1f}s")
            
            # Handle auto-open if checkbox exists
            if path:
                if hasattr(self, 'auto_open_file_cb') and self.auto_open_file_cb.isChecked():
                    open_file_in_explorer(path)
                if hasattr(self, 'auto_open_viewer_cb') and self.auto_open_viewer_cb.isChecked():
                    open_in_default_viewer(path)
            
            # Let subclass update preview
            self.on_generation_success(result)
        else:
            error = result.get("error", "Unknown error")
            self.set_status(f"Error: {error}")
            self.on_generation_error(result)
    
    def on_generation_success(self, result: dict):
        """
        Override this to handle successful generation.
        
        Example:
            path = result.get("path")
            if path:
                pixmap = QPixmap(path)
                self.preview_label.setPixmap(pixmap.scaled(...))
        """
        pass
    
    def on_generation_error(self, result: dict):
        """
        Override this to handle generation errors.
        
        Default shows a warning message box.
        """
        error = result.get("error", "Unknown error")
        QMessageBox.warning(self, "Generation Error", f"Failed to generate:\n{error}")
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    def set_status(self, message: str):
        """Update the status label."""
        self.status_label.setText(message)
    
    def show_error(self, message: str, title: str = "Error"):
        """Show an error message box."""
        QMessageBox.warning(self, title, message)
    
    def show_info(self, message: str, title: str = "Info"):
        """Show an info message box."""
        QMessageBox.information(self, title, message)
    
    def _open_output_folder(self):
        """Open the output folder in file explorer."""
        open_folder(self.output_dir)
    
    def create_provider_combo(self, providers: List[tuple]) -> 'NoScrollComboBox':
        """
        Create a standard provider combo box.
        
        Args:
            providers: List of (display_name, internal_name) tuples
            
        Returns:
            Configured NoScrollComboBox
            
        Example:
            self.provider_combo = self.create_provider_combo([
                ('Local (Forge)', 'local'),
                ('OpenAI (Cloud)', 'openai'),
            ])
        """
        combo = NoScrollComboBox()
        for display_name, internal_name in providers:
            combo.addItem(display_name, internal_name)
        combo.setToolTip("Select the AI provider for generation")
        return combo
    
    def create_labeled_spinner(self, label_text: str, min_val: int, max_val: int,
                               default: int, suffix: str = "",
                               tooltip: str = "") -> tuple:
        """
        Create a label + spinner pair.
        
        Returns:
            (QLabel, QSpinBox) tuple
        """
        from PyQt5.QtWidgets import QSpinBox
        
        label = QLabel(label_text)
        spinner = QSpinBox()
        spinner.setRange(min_val, max_val)
        spinner.setValue(default)
        if suffix:
            spinner.setSuffix(suffix)
        if tooltip:
            spinner.setToolTip(tooltip)
        
        return label, spinner
    
    def create_auto_open_checkboxes(self) -> QHBoxLayout:
        """
        Create standard auto-open checkboxes.
        
        Creates self.auto_open_file_cb and self.auto_open_viewer_cb
        Returns the layout to add to your UI.
        """
        from PyQt5.QtWidgets import QCheckBox
        
        layout = QHBoxLayout()
        
        self.auto_open_file_cb = QCheckBox("Auto-open in explorer")
        self.auto_open_file_cb.setChecked(True)
        self.auto_open_file_cb.setToolTip("Open the generated file location when done")
        layout.addWidget(self.auto_open_file_cb)
        
        self.auto_open_viewer_cb = QCheckBox("Auto-open in viewer")
        self.auto_open_viewer_cb.setChecked(False)
        self.auto_open_viewer_cb.setToolTip("Open the file in your default application")
        layout.addWidget(self.auto_open_viewer_cb)
        
        layout.addStretch()
        return layout


# =============================================================================
# Utility function for creating consistent group boxes
# =============================================================================

def create_group_box(title: str) -> QGroupBox:
    """Create a styled group box."""
    group = QGroupBox(title)
    group.setStyleSheet(GROUP_BOX_STYLE)
    return group


# Export all
__all__ = [
    'BaseGenerationTab',
    'BaseGenerationWorker',
    'BUTTON_STYLE_PRIMARY',
    'BUTTON_STYLE_SECONDARY',
    'BUTTON_STYLE_SUCCESS',
    'BUTTON_STYLE_DANGER',
    'HEADER_STYLE',
    'STATUS_LABEL_STYLE',
    'PROGRESS_BAR_STYLE',
    'PREVIEW_FRAME_STYLE',
    'GROUP_BOX_STYLE',
    'create_group_box',
]
