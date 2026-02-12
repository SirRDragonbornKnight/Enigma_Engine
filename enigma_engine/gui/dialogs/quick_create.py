"""
Quick Create AI Dialog - Create an AI from a starter kit template

Provides a simple dialog to create a fully-configured AI in one click.
"""

import logging
from typing import Optional

try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtWidgets import (
        QApplication,
        QDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QVBoxLayout,
        QCheckBox,
    )
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QDialog = object

logger = logging.getLogger(__name__)


class CreateAIWorker(QThread):
    """Background worker for creating AI from starter kit."""
    
    progress = pyqtSignal(str, str, int)  # step, status, percent
    finished = pyqtSignal(dict)  # result dict
    
    def __init__(self, kit_id: str, ai_name: str, registry, auto_train: bool = True):
        super().__init__()
        self.kit_id = kit_id
        self.ai_name = ai_name
        self.registry = registry
        self.auto_train = auto_train
    
    def run(self):
        from ...utils.starter_kits import create_ai_from_kit
        
        result = create_ai_from_kit(
            kit_id=self.kit_id,
            registry=self.registry,
            ai_name=self.ai_name,
            auto_train=self.auto_train,
            progress_callback=lambda step, status, pct: self.progress.emit(step, status, pct),
        )
        
        self.finished.emit(result)


class QuickCreateDialog(QDialog):
    """
    Dialog for quickly creating an AI from a starter kit template.
    
    Usage:
        dialog = QuickCreateDialog(registry, parent)
        if dialog.exec_() == QDialog.Accepted:
            ai_name = dialog.created_ai_name
    """
    
    def __init__(self, registry, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.created_ai_name = None
        self._worker = None
        
        self._setup_ui()
        self._populate_kits()
        self._apply_style()
        
        # Apply transparency
        try:
            from ..ui_settings import apply_dialog_transparency
            apply_dialog_transparency(self)
        except ImportError:
            pass  # Intentionally silent
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Quick Create AI")
        self.setMinimumSize(700, 550)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Header
        header = QLabel("Create Your AI")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header)
        
        subtitle = QLabel("Choose a template and give your AI a name - we'll handle the rest!")
        subtitle.setStyleSheet("color: #999; margin-bottom: 10px;")
        layout.addWidget(subtitle)
        
        # Main content
        content_layout = QHBoxLayout()
        
        # Left: Template list
        left_group = QGroupBox("1. Choose Template")
        left_layout = QVBoxLayout(left_group)
        
        self.kit_list = QListWidget()
        self.kit_list.setMinimumWidth(250)
        self.kit_list.currentItemChanged.connect(self._on_kit_selected)
        left_layout.addWidget(self.kit_list)
        
        content_layout.addWidget(left_group)
        
        # Right: Details and name
        right_group = QGroupBox("2. Configure")
        right_layout = QVBoxLayout(right_group)
        
        # Template details
        self.detail_label = QLabel("Select a template to see details")
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet("background: #2d2d3d; padding: 10px; border-radius: 6px;")
        right_layout.addWidget(self.detail_label)
        
        # Name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("AI Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter a name (e.g., buddy, helper, sparky)")
        self.name_input.textChanged.connect(self._validate_name)
        name_layout.addWidget(self.name_input)
        right_layout.addLayout(name_layout)
        
        self.name_status = QLabel("")
        self.name_status.setStyleSheet("font-size: 11px;")
        right_layout.addWidget(self.name_status)
        
        # Options
        self.auto_train_check = QCheckBox("Auto-train with starter data (recommended)")
        self.auto_train_check.setChecked(True)
        right_layout.addWidget(self.auto_train_check)
        
        right_layout.addStretch()
        content_layout.addWidget(right_group)
        
        layout.addLayout(content_layout)
        
        # Progress section (hidden initially)
        self.progress_group = QGroupBox("3. Creating Your AI")
        progress_layout = QVBoxLayout(self.progress_group)
        
        self.progress_status = QLabel("Ready to create...")
        progress_layout.addWidget(self.progress_status)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_group.hide()
        layout.addWidget(self.progress_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.create_btn = QPushButton("Create My AI")
        self.create_btn.setEnabled(False)
        self.create_btn.clicked.connect(self._on_create)
        self.create_btn.setStyleSheet("""
            QPushButton {
                background: #2ecc71;
                color: white;
                font-weight: bold;
                padding: 10px 24px;
                border-radius: 6px;
            }
            QPushButton:hover { background: #27ae60; }
            QPushButton:disabled { background: #555; color: #999; }
        """)
        btn_layout.addWidget(self.create_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(btn_layout)
    
    def _populate_kits(self):
        """Populate the starter kit list."""
        from ...utils.starter_kits import list_kits
        
        for kit in list_kits():
            item = QListWidgetItem(f"{kit['icon']} {kit['name']}")
            item.setData(Qt.UserRole, kit['id'])
            item.setData(Qt.UserRole + 1, kit)  # Store full kit info
            self.kit_list.addItem(item)
        
        # Select first item
        if self.kit_list.count() > 0:
            self.kit_list.setCurrentRow(0)
    
    def _on_kit_selected(self, current, previous):
        """Handle kit selection change."""
        if not current:
            return
        
        kit_info = current.data(Qt.UserRole + 1)
        if kit_info:
            self.detail_label.setText(f"""
<b>{kit_info['name']}</b><br><br>
{kit_info['description']}<br><br>
<i>Model size: {kit_info['model_size']}</i><br>
<i>Tags: {', '.join(kit_info['tags'])}</i>
""")
        
        self._update_create_button()
    
    def _validate_name(self, text):
        """Validate the AI name."""
        name = text.lower().strip().replace(" ", "_")
        
        if not name:
            self.name_status.setText("")
            self.name_status.setStyleSheet("color: #999;")
        elif name in self.registry.registry.get("models", {}):
            self.name_status.setText("Name already exists")
            self.name_status.setStyleSheet("color: #f39c12;")
        elif not name.replace("_", "").isalnum():
            self.name_status.setText("Use only letters, numbers, underscores")
            self.name_status.setStyleSheet("color: #e74c3c;")
        else:
            self.name_status.setText("Name available")
            self.name_status.setStyleSheet("color: #2ecc71;")
        
        self._update_create_button()
    
    def _update_create_button(self):
        """Update create button enabled state."""
        name = self.name_input.text().lower().strip().replace(" ", "_")
        has_kit = self.kit_list.currentItem() is not None
        name_valid = bool(
            name 
            and name.replace("_", "").isalnum() 
            and name not in self.registry.registry.get("models", {})
        )
        
        self.create_btn.setEnabled(has_kit and name_valid)
    
    def _on_create(self):
        """Start creating the AI."""
        kit_item = self.kit_list.currentItem()
        if not kit_item:
            return
        
        kit_id = kit_item.data(Qt.UserRole)
        ai_name = self.name_input.text().lower().strip().replace(" ", "_")
        
        # Disable inputs during creation
        self.kit_list.setEnabled(False)
        self.name_input.setEnabled(False)
        self.auto_train_check.setEnabled(False)
        self.create_btn.setEnabled(False)
        self.cancel_btn.setText("Close")
        
        # Show progress
        self.progress_group.show()
        
        # Start background worker
        self._worker = CreateAIWorker(
            kit_id=kit_id,
            ai_name=ai_name,
            registry=self.registry,
            auto_train=self.auto_train_check.isChecked()
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
    
    def _on_progress(self, step: str, status: str, percent: int):
        """Handle progress updates."""
        self.progress_status.setText(status)
        self.progress_bar.setValue(percent)
        QApplication.processEvents()
    
    def _on_finished(self, result: dict):
        """Handle creation finished."""
        if result.get("success"):
            self.created_ai_name = result["model_name"]
            
            msg = f"Your AI '{result['model_name']}' has been created!"
            if result.get("trained"):
                msg += f"\n\nTrained for {result.get('epochs', 0)} epochs."
            msg += "\n\nYou can start chatting with it now!"
            
            QMessageBox.information(self, "AI Created!", msg)
            self.accept()
        else:
            QMessageBox.warning(
                self, "Creation Failed",
                f"Failed to create AI:\n{result.get('error', 'Unknown error')}"
            )
            
            # Re-enable inputs
            self.kit_list.setEnabled(True)
            self.name_input.setEnabled(True)
            self.auto_train_check.setEnabled(True)
            self._update_create_button()
            self.cancel_btn.setText("Cancel")
    
    def _apply_style(self):
        """Apply dialog styling."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QListWidget {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 4px;
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
            QLineEdit {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 8px;
                color: #cdd6f4;
            }
            QLineEdit:focus {
                border-color: #89b4fa;
            }
            QProgressBar {
                border-radius: 4px;
                text-align: center;
                background-color: #313244;
            }
            QProgressBar::chunk {
                background-color: #89b4fa;
                border-radius: 4px;
            }
            QCheckBox {
                spacing: 8px;
            }
            QPushButton {
                background-color: #45475a;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #585b70;
            }
        """)


def show_quick_create_dialog(registry, parent=None) -> Optional[str]:
    """
    Show the quick create dialog and return the created AI name.
    
    Args:
        registry: ModelRegistry instance
        parent: Parent widget
        
    Returns:
        Name of created AI, or None if cancelled
    """
    dialog = QuickCreateDialog(registry, parent)
    if dialog.exec_() == QDialog.Accepted:
        return dialog.created_ai_name
    return None


__all__ = ['QuickCreateDialog', 'show_quick_create_dialog']
