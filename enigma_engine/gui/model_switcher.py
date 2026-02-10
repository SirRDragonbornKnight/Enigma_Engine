"""
Quick Model Switcher Widget for Enigma AI Engine GUI

Dropdown widget to quickly swap between loaded models.

Usage:
    from enigma_engine.gui.model_switcher import QuickModelSwitcher
    
    switcher = QuickModelSwitcher()
    toolbar.addWidget(switcher)
    
    # Connect to model change
    switcher.model_changed.connect(on_model_changed)
"""

import logging
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton,
    QWidget, QToolButton, QMenu
)

logger = logging.getLogger(__name__)


class QuickModelSwitcher(QWidget):
    """
    Quick model switching dropdown widget.
    
    Features:
    - Dropdown with all available models
    - Current model indicator
    - Refresh button
    - Memory usage display
    """
    
    model_changed = pyqtSignal(str)  # Emits model path when changed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_model: Optional[str] = None
        self._models: List[str] = []
        
        self._setup_ui()
        self.refresh_models()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)
        
        # Label
        label = QLabel("Model:")
        label.setStyleSheet("font-weight: bold;")
        layout.addWidget(label)
        
        # Dropdown
        self.combo = QComboBox()
        self.combo.setMinimumWidth(150)
        self.combo.currentIndexChanged.connect(self._on_selection_changed)
        layout.addWidget(self.combo)
        
        # Refresh button
        refresh_btn = QToolButton()
        refresh_btn.setText("R")
        refresh_btn.setToolTip("Refresh model list")
        refresh_btn.clicked.connect(self.refresh_models)
        layout.addWidget(refresh_btn)
        
        # Info display
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        layout.addWidget(self.info_label)
    
    def refresh_models(self):
        """Refresh the list of available models."""
        try:
            from ..config import CONFIG
            models_dir = Path(CONFIG.get("models_dir", "models"))
            
            self._models = []
            
            if models_dir.exists():
                for item in models_dir.iterdir():
                    if item.is_dir():
                        # Check if it's a valid model directory
                        model_file = item / "forge_model.pt"
                        config_file = item / "config.json"
                        
                        if model_file.exists() or config_file.exists():
                            self._models.append(str(item))
            
            # Update dropdown
            self.combo.blockSignals(True)
            self.combo.clear()
            self.combo.addItem("(None)")
            
            for model_path in sorted(self._models):
                name = Path(model_path).name
                self.combo.addItem(name, model_path)
            
            # Restore selection
            if self._current_model:
                idx = self.combo.findData(self._current_model)
                if idx >= 0:
                    self.combo.setCurrentIndex(idx)
            
            self.combo.blockSignals(False)
            
            logger.debug(f"Found {len(self._models)} models")
            
        except Exception as e:
            logger.warning(f"Failed to refresh models: {e}")
    
    def _on_selection_changed(self, index: int):
        """Handle model selection change."""
        if index == 0:
            # (None) selected
            return
        
        model_path = self.combo.currentData()
        if model_path and model_path != self._current_model:
            self._current_model = model_path
            self._update_info()
            self.model_changed.emit(model_path)
    
    def _update_info(self):
        """Update the info display."""
        if not self._current_model:
            self.info_label.setText("")
            return
        
        try:
            model_path = Path(self._current_model)
            model_file = model_path / "forge_model.pt"
            
            if model_file.exists():
                size_mb = model_file.stat().st_size / (1024 * 1024)
                self.info_label.setText(f"({size_mb:.1f} MB)")
            else:
                self.info_label.setText("")
        except Exception:
            self.info_label.setText("")
    
    def set_current_model(self, model_path: str):
        """Set the current model without emitting signal."""
        self._current_model = model_path
        
        self.combo.blockSignals(True)
        idx = self.combo.findData(model_path)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)
        self.combo.blockSignals(False)
        
        self._update_info()
    
    def get_current_model(self) -> Optional[str]:
        """Get currently selected model path."""
        return self._current_model


class ModelSwitcherMenu(QMenu):
    """
    Context menu version of model switcher.
    
    Can be attached to a button for quick model switching.
    """
    
    model_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__("Models", parent)
        self._current_model: Optional[str] = None
        self._refresh_action = self.addAction("Refresh")
        self._refresh_action.triggered.connect(self.refresh_models)
        self.addSeparator()
        
        self.refresh_models()
    
    def refresh_models(self):
        """Refresh model list."""
        # Remove all model actions
        for action in self.actions():
            if action != self._refresh_action and not action.isSeparator():
                self.removeAction(action)
        
        try:
            from ..config import CONFIG
            models_dir = Path(CONFIG.get("models_dir", "models"))
            
            if models_dir.exists():
                for item in sorted(models_dir.iterdir()):
                    if item.is_dir():
                        model_file = item / "forge_model.pt"
                        if model_file.exists():
                            action = self.addAction(item.name)
                            action.setData(str(item))
                            action.setCheckable(True)
                            action.setChecked(str(item) == self._current_model)
                            action.triggered.connect(
                                lambda checked, p=str(item): self._on_model_selected(p)
                            )
        except Exception as e:
            logger.warning(f"Failed to refresh models: {e}")
    
    def _on_model_selected(self, model_path: str):
        """Handle model selection."""
        self._current_model = model_path
        
        # Update check marks
        for action in self.actions():
            if action.data():
                action.setChecked(action.data() == model_path)
        
        self.model_changed.emit(model_path)
    
    def set_current_model(self, model_path: str):
        """Set current model without emitting signal."""
        self._current_model = model_path
        
        for action in self.actions():
            if action.data():
                action.setChecked(action.data() == model_path)


def create_model_switcher_button(parent=None) -> QPushButton:
    """
    Create a button with attached model switcher menu.
    
    Returns:
        QPushButton with ModelSwitcherMenu attached
    """
    btn = QPushButton("Model")
    btn.setToolTip("Switch active model")
    
    menu = ModelSwitcherMenu(btn)
    btn.setMenu(menu)
    
    # Store reference to menu for external access
    btn.model_menu = menu
    
    return btn
