"""
Import Models Tab - Import external AI models into Enigma Engine.

Supports:
- HuggingFace models (inference only or converted for training)
- GGUF models (llama.cpp format)
- Local model files

FILE: enigma_engine/gui/tabs/import_models_tab.py

USAGE:
    from enigma_engine.gui.tabs.import_models_tab import create_import_models_tab
    
    widget = create_import_models_tab(parent_window)
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtWidgets import (
        QCheckBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


# =============================================================================
# WORKER THREAD FOR BACKGROUND IMPORT
# =============================================================================

class ImportWorker(QThread):
    """Background worker for model import operations."""
    
    progress = pyqtSignal(str, int)  # message, percentage
    finished = pyqtSignal(bool, str)  # success, result/error message
    
    def __init__(
        self,
        source: str,
        name: str,
        convert_to_forge: bool = False,
        capabilities: list = None,
        models_dir: str = None,
    ):
        super().__init__()
        self.source = source
        self.name = name
        self.convert_to_forge = convert_to_forge
        self.capabilities = capabilities or ["chat"]
        self.models_dir = models_dir
    
    def run(self):
        """Run the import in background."""
        try:
            from enigma_engine.core.model_registry import ModelRegistry
            
            self.progress.emit("Initializing registry...", 10)
            registry = ModelRegistry(self.models_dir)
            
            self.progress.emit(f"Registering model: {self.source}...", 20)
            
            # Use the new register_huggingface_model function
            name = registry.register_huggingface_model(
                model_id=self.source,
                name=self.name if self.name else None,
                convert_to_forge=self.convert_to_forge,
                capabilities=self.capabilities,
            )
            
            self.progress.emit("Import complete!", 100)
            self.finished.emit(True, f"Successfully imported '{name}'")
            
        except Exception as e:
            logger.exception(f"Import failed: {e}")
            self.finished.emit(False, str(e))


# =============================================================================
# MAIN TAB WIDGET
# =============================================================================

def create_import_models_tab(parent) -> QWidget:
    """
    Create the Import Models tab.
    
    Args:
        parent: Parent window (EnhancedMainWindow)
        
    Returns:
        QWidget containing the import models interface
    """
    if not HAS_PYQT:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("PyQt5 not available"))
        return widget
    
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(12)
    
    # Store references on parent for access
    parent._import_worker = None
    
    # Header
    header = QLabel("Import External Models")
    header.setStyleSheet("font-size: 12px; font-weight: bold; color: #a6e3a1;")
    layout.addWidget(header)
    
    desc = QLabel(
        "Import pre-trained models from HuggingFace Hub or local files. "
        "Models can be used for inference or converted for fine-tuning."
    )
    desc.setStyleSheet("color: #64748b; font-size: 10px;")
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Scroll area for content
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setStyleSheet("""
        QScrollArea {
            border: none;
            background: transparent;
        }
    """)
    
    scroll_content = QWidget()
    scroll_layout = QVBoxLayout(scroll_content)
    scroll_layout.setSpacing(12)
    
    # === HUGGINGFACE IMPORT SECTION ===
    hf_group = QGroupBox("HuggingFace Hub")
    hf_group.setStyleSheet("""
        QGroupBox {
            border: 1px solid #3b82f6;
            border-radius: 8px;
            margin-top: 8px;
            padding: 12px;
            font-weight: bold;
        }
        QGroupBox::title {
            color: #3b82f6;
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
    """)
    hf_layout = QVBoxLayout(hf_group)
    
    # Model ID input
    id_row = QHBoxLayout()
    id_label = QLabel("Model ID:")
    id_label.setMinimumWidth(100)
    id_row.addWidget(id_label)
    
    parent._import_model_id = QLineEdit()
    parent._import_model_id.setPlaceholderText("e.g., gpt2, microsoft/phi-2, meta-llama/Llama-2-7b-hf")
    parent._import_model_id.setStyleSheet("""
        QLineEdit {
            padding: 8px;
            border: 1px solid #334155;
            border-radius: 6px;
            background: #1e1e2e;
        }
        QLineEdit:focus {
            border-color: #3b82f6;
        }
    """)
    id_row.addWidget(parent._import_model_id)
    hf_layout.addLayout(id_row)
    
    # Local name
    name_row = QHBoxLayout()
    name_label = QLabel("Local Name:")
    name_label.setMinimumWidth(100)
    name_row.addWidget(name_label)
    
    parent._import_local_name = QLineEdit()
    parent._import_local_name.setPlaceholderText("Optional - auto-generated if empty")
    parent._import_local_name.setStyleSheet("""
        QLineEdit {
            padding: 8px;
            border: 1px solid #334155;
            border-radius: 6px;
            background: #1e1e2e;
        }
    """)
    name_row.addWidget(parent._import_local_name)
    hf_layout.addLayout(name_row)
    
    # Options row
    options_row = QHBoxLayout()
    
    parent._import_convert = QCheckBox("Convert for training")
    parent._import_convert.setToolTip(
        "Convert model weights to Forge format to enable fine-tuning.\n"
        "Without this, the model can only be used for inference."
    )
    parent._import_convert.setChecked(False)
    options_row.addWidget(parent._import_convert)
    
    options_row.addStretch()
    hf_layout.addLayout(options_row)
    
    # Capabilities
    caps_row = QHBoxLayout()
    caps_label = QLabel("Capabilities:")
    caps_label.setMinimumWidth(100)
    caps_row.addWidget(caps_label)
    
    parent._import_caps = QLineEdit()
    parent._import_caps.setPlaceholderText("chat, code, vision (comma-separated)")
    parent._import_caps.setText("chat")
    parent._import_caps.setStyleSheet("""
        QLineEdit {
            padding: 8px;
            border: 1px solid #334155;
            border-radius: 6px;
            background: #1e1e2e;
        }
    """)
    caps_row.addWidget(parent._import_caps)
    hf_layout.addLayout(caps_row)
    
    # Popular models
    popular_row = QHBoxLayout()
    popular_label = QLabel("Popular:")
    popular_label.setMinimumWidth(100)
    popular_row.addWidget(popular_label)
    
    popular_models = [
        ("GPT-2", "gpt2"),
        ("GPT-2 Medium", "gpt2-medium"),
        ("Phi-2", "microsoft/phi-2"),
        ("TinyLlama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        ("Mistral 7B", "mistralai/Mistral-7B-v0.1"),
    ]
    
    for display_name, model_id in popular_models:
        btn = QPushButton(display_name)
        btn.setStyleSheet("""
            QPushButton {
                padding: 4px 8px;
                border: 1px solid #334155;
                border-radius: 4px;
                background: #1e1e2e;
                font-size: 10px;
            }
            QPushButton:hover {
                border-color: #3b82f6;
                background: #1a1a24;
            }
        """)
        btn.clicked.connect(lambda checked, mid=model_id: parent._import_model_id.setText(mid))
        popular_row.addWidget(btn)
    
    popular_row.addStretch()
    hf_layout.addLayout(popular_row)
    
    # Import button
    btn_row = QHBoxLayout()
    btn_row.addStretch()
    
    parent._btn_import_hf = QPushButton("Import from HuggingFace")
    parent._btn_import_hf.setStyleSheet("""
        QPushButton {
            padding: 10px 20px;
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: bold;
        }
        QPushButton:hover {
            background: #2563eb;
        }
        QPushButton:disabled {
            background: #64748b;
        }
    """)
    parent._btn_import_hf.clicked.connect(lambda: _start_import(parent))
    btn_row.addWidget(parent._btn_import_hf)
    
    hf_layout.addLayout(btn_row)
    scroll_layout.addWidget(hf_group)
    
    # === PROGRESS SECTION ===
    progress_group = QGroupBox("Import Progress")
    progress_group.setStyleSheet("""
        QGroupBox {
            border: 1px solid #334155;
            border-radius: 8px;
            margin-top: 8px;
            padding: 12px;
        }
    """)
    progress_layout = QVBoxLayout(progress_group)
    
    parent._import_progress = QProgressBar()
    parent._import_progress.setStyleSheet("""
        QProgressBar {
            border: 1px solid #334155;
            border-radius: 4px;
            text-align: center;
            background: #1e1e2e;
        }
        QProgressBar::chunk {
            background: #3b82f6;
            border-radius: 3px;
        }
    """)
    parent._import_progress.setValue(0)
    progress_layout.addWidget(parent._import_progress)
    
    parent._import_status = QLabel("Ready to import")
    parent._import_status.setStyleSheet("color: #64748b; font-size: 10px;")
    progress_layout.addWidget(parent._import_status)
    
    scroll_layout.addWidget(progress_group)
    
    # === REGISTERED MODELS LIST ===
    models_group = QGroupBox("Registered Models")
    models_group.setStyleSheet("""
        QGroupBox {
            border: 1px solid #334155;
            border-radius: 8px;
            margin-top: 8px;
            padding: 12px;
        }
    """)
    models_layout = QVBoxLayout(models_group)
    
    parent._import_models_list = QListWidget()
    parent._import_models_list.setStyleSheet("""
        QListWidget {
            border: 1px solid #334155;
            border-radius: 4px;
            background: #1e1e2e;
        }
        QListWidget::item {
            padding: 8px;
            border-bottom: 1px solid #334155;
        }
        QListWidget::item:selected {
            background: #3b82f6;
        }
    """)
    parent._import_models_list.setMaximumHeight(200)
    models_layout.addWidget(parent._import_models_list)
    
    # Refresh button
    refresh_row = QHBoxLayout()
    
    btn_refresh = QPushButton("Refresh List")
    btn_refresh.setStyleSheet("""
        QPushButton {
            padding: 6px 12px;
            border: 1px solid #334155;
            border-radius: 4px;
            background: #1e1e2e;
        }
        QPushButton:hover {
            border-color: #3b82f6;
        }
    """)
    btn_refresh.clicked.connect(lambda: _refresh_models_list(parent))
    refresh_row.addWidget(btn_refresh)
    
    btn_delete = QPushButton("Delete Selected")
    btn_delete.setStyleSheet("""
        QPushButton {
            padding: 6px 12px;
            border: 1px solid #ef4444;
            border-radius: 4px;
            background: #1e1e2e;
            color: #ef4444;
        }
        QPushButton:hover {
            background: #ef4444;
            color: white;
        }
    """)
    btn_delete.clicked.connect(lambda: _delete_selected_model(parent))
    refresh_row.addWidget(btn_delete)
    
    refresh_row.addStretch()
    models_layout.addLayout(refresh_row)
    
    scroll_layout.addWidget(models_group)
    
    # === FINE-TUNE QUICK START ===
    finetune_group = QGroupBox("Quick Fine-Tune")
    finetune_group.setStyleSheet("""
        QGroupBox {
            border: 1px solid #a6e3a1;
            border-radius: 8px;
            margin-top: 8px;
            padding: 12px;
            font-weight: bold;
        }
        QGroupBox::title {
            color: #a6e3a1;
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
    """)
    finetune_layout = QVBoxLayout(finetune_group)
    
    # Data file
    data_row = QHBoxLayout()
    data_label = QLabel("Training Data:")
    data_label.setMinimumWidth(100)
    data_row.addWidget(data_label)
    
    parent._finetune_data = QLineEdit()
    parent._finetune_data.setPlaceholderText("Path to training data file (.txt)")
    parent._finetune_data.setStyleSheet("""
        QLineEdit {
            padding: 8px;
            border: 1px solid #334155;
            border-radius: 6px;
            background: #1e1e2e;
        }
    """)
    data_row.addWidget(parent._finetune_data)
    
    btn_browse = QPushButton("Browse")
    btn_browse.clicked.connect(lambda: _browse_data_file(parent))
    data_row.addWidget(btn_browse)
    finetune_layout.addLayout(data_row)
    
    # Options
    ft_options = QHBoxLayout()
    
    parent._finetune_lora = QCheckBox("Use LoRA")
    parent._finetune_lora.setToolTip("Efficient fine-tuning with less memory")
    parent._finetune_lora.setChecked(True)
    ft_options.addWidget(parent._finetune_lora)
    
    epochs_label = QLabel("Epochs:")
    ft_options.addWidget(epochs_label)
    
    parent._finetune_epochs = QSpinBox()
    parent._finetune_epochs.setRange(1, 100)
    parent._finetune_epochs.setValue(10)
    ft_options.addWidget(parent._finetune_epochs)
    
    ft_options.addStretch()
    finetune_layout.addLayout(ft_options)
    
    # Fine-tune button
    ft_btn_row = QHBoxLayout()
    ft_btn_row.addStretch()
    
    parent._btn_finetune = QPushButton("Start Fine-Tuning")
    parent._btn_finetune.setStyleSheet("""
        QPushButton {
            padding: 10px 20px;
            background: #a6e3a1;
            color: #1e1e2e;
            border: none;
            border-radius: 6px;
            font-weight: bold;
        }
        QPushButton:hover {
            background: #86c991;
        }
        QPushButton:disabled {
            background: #64748b;
        }
    """)
    parent._btn_finetune.clicked.connect(lambda: _start_finetune(parent))
    ft_btn_row.addWidget(parent._btn_finetune)
    
    finetune_layout.addLayout(ft_btn_row)
    scroll_layout.addWidget(finetune_group)
    
    # Add stretch at end
    scroll_layout.addStretch()
    
    scroll.setWidget(scroll_content)
    layout.addWidget(scroll)
    
    # Initial refresh
    _refresh_models_list(parent)
    
    return widget


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _start_import(parent):
    """Start the HuggingFace import process."""
    model_id = parent._import_model_id.text().strip()
    if not model_id:
        QMessageBox.warning(parent, "Error", "Please enter a model ID")
        return
    
    local_name = parent._import_local_name.text().strip() or None
    convert = parent._import_convert.isChecked()
    caps_text = parent._import_caps.text().strip()
    capabilities = [c.strip() for c in caps_text.split(",") if c.strip()]
    
    # Disable button
    parent._btn_import_hf.setEnabled(False)
    parent._import_progress.setValue(0)
    parent._import_status.setText(f"Importing {model_id}...")
    
    # Create worker
    parent._import_worker = ImportWorker(
        source=model_id,
        name=local_name,
        convert_to_forge=convert,
        capabilities=capabilities,
    )
    
    # Connect signals
    parent._import_worker.progress.connect(
        lambda msg, pct: _on_import_progress(parent, msg, pct)
    )
    parent._import_worker.finished.connect(
        lambda success, msg: _on_import_finished(parent, success, msg)
    )
    
    # Start
    parent._import_worker.start()


def _on_import_progress(parent, message: str, percent: int):
    """Handle import progress update."""
    parent._import_progress.setValue(percent)
    parent._import_status.setText(message)


def _on_import_finished(parent, success: bool, message: str):
    """Handle import completion."""
    parent._btn_import_hf.setEnabled(True)
    
    if success:
        parent._import_progress.setValue(100)
        parent._import_status.setText(message)
        parent._import_status.setStyleSheet("color: #a6e3a1; font-size: 10px;")
        _refresh_models_list(parent)
        QMessageBox.information(parent, "Success", message)
    else:
        parent._import_progress.setValue(0)
        parent._import_status.setText(f"Error: {message}")
        parent._import_status.setStyleSheet("color: #ef4444; font-size: 10px;")
        QMessageBox.critical(parent, "Import Failed", message)


def _refresh_models_list(parent):
    """Refresh the list of registered models."""
    try:
        from enigma_engine.core.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        models = registry.list_models()
        
        parent._import_models_list.clear()
        
        for name, info in models.items():
            source = info.get("source", "unknown")
            hf_id = info.get("huggingface_id", info.get("source_huggingface_id", ""))
            caps = info.get("capabilities", ["chat"])
            
            # Create display text
            if source == "huggingface":
                display = f"{name} (HF: {hf_id}) [{', '.join(caps)}]"
            elif hf_id:
                display = f"{name} (from: {hf_id}) [{', '.join(caps)}] [trainable]"
            else:
                display = f"{name} [{', '.join(caps)}]"
            
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, name)
            parent._import_models_list.addItem(item)
            
    except Exception as e:
        logger.exception(f"Failed to refresh models list: {e}")


def _delete_selected_model(parent):
    """Delete the selected model from registry."""
    item = parent._import_models_list.currentItem()
    if not item:
        QMessageBox.warning(parent, "Error", "Please select a model to delete")
        return
    
    name = item.data(Qt.UserRole)
    
    reply = QMessageBox.question(
        parent,
        "Confirm Delete",
        f"Are you sure you want to delete model '{name}'?\n\nThis cannot be undone.",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    
    if reply == QMessageBox.Yes:
        try:
            from enigma_engine.core.model_registry import ModelRegistry
            
            registry = ModelRegistry()
            registry.delete_model(name, confirm=True)
            
            _refresh_models_list(parent)
            QMessageBox.information(parent, "Deleted", f"Model '{name}' has been deleted.")
            
        except Exception as e:
            QMessageBox.critical(parent, "Error", f"Failed to delete model: {e}")


def _browse_data_file(parent):
    """Browse for training data file."""
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "Select Training Data",
        "",
        "Text Files (*.txt);;All Files (*)"
    )
    if file_path:
        parent._finetune_data.setText(file_path)


def _start_finetune(parent):
    """Start the fine-tuning process."""
    # Get selected model
    item = parent._import_models_list.currentItem()
    if not item:
        QMessageBox.warning(parent, "Error", "Please select a model to fine-tune")
        return
    
    model_name = item.data(Qt.UserRole)
    data_path = parent._finetune_data.text().strip()
    
    if not data_path:
        QMessageBox.warning(parent, "Error", "Please select a training data file")
        return
    
    if not Path(data_path).exists():
        QMessageBox.warning(parent, "Error", f"Training data file not found: {data_path}")
        return
    
    use_lora = parent._finetune_lora.isChecked()
    epochs = parent._finetune_epochs.value()
    
    # Confirm
    reply = QMessageBox.question(
        parent,
        "Confirm Fine-Tune",
        f"Fine-tune '{model_name}' for {epochs} epochs?\n\n"
        f"Data: {data_path}\n"
        f"Method: {'LoRA (efficient)' if use_lora else 'Full fine-tuning'}",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    
    if reply != QMessageBox.Yes:
        return
    
    # Start fine-tuning (in background ideally, but for now synchronous)
    parent._btn_finetune.setEnabled(False)
    parent._import_status.setText(f"Fine-tuning {model_name}...")
    
    try:
        from enigma_engine.core.model_registry import fine_tune_pretrained
        
        model, result_name = fine_tune_pretrained(
            source=model_name,
            data_path=data_path,
            epochs=epochs,
            use_lora=use_lora,
        )
        
        parent._import_status.setText(f"Fine-tuning complete: {result_name}")
        parent._import_status.setStyleSheet("color: #a6e3a1; font-size: 10px;")
        _refresh_models_list(parent)
        QMessageBox.information(
            parent,
            "Success",
            f"Fine-tuning complete!\n\nModel saved as: {result_name}"
        )
        
    except Exception as e:
        logger.exception(f"Fine-tuning failed: {e}")
        parent._import_status.setText(f"Fine-tuning failed: {e}")
        parent._import_status.setStyleSheet("color: #ef4444; font-size: 10px;")
        QMessageBox.critical(parent, "Fine-Tune Failed", str(e))
    
    finally:
        parent._btn_finetune.setEnabled(True)


# Export
__all__ = ['create_import_models_tab', 'ImportWorker']
