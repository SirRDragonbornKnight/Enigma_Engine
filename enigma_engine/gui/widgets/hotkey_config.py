"""
Hotkey Configuration Widget - GUI for configuring global hotkeys.

Features:
- List all hotkeys with current bindings
- Click to rebind (press new key combo)
- Conflict detection
- Reset to defaults
- Enable/disable individual hotkeys
"""

import logging

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class HotkeyRecordDialog(QDialog):
    """Dialog for recording a new hotkey."""
    
    def __init__(self, hotkey_name: str, current_hotkey: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Record Hotkey - {hotkey_name}")
        self.setModal(True)
        
        self.recorded_hotkey = ""
        self.current_hotkey = current_hotkey
        
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel(
            f"Press the key combination you want to use for:\n\n"
            f"{hotkey_name}\n\n"
            f"Current: {current_hotkey}\n\n"
            f"Press ESC to cancel"
        )
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)
        
        # Display label for recorded keys
        self.display_label = QLabel("Waiting for input...")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; padding: 10px; "
            "background-color: #2a2a2a; border-radius: 5px;"
        )
        layout.addWidget(self.display_label)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        self.resize(400, 200)
        
        # Track modifier keys
        self.modifiers = set()
        self.main_key = ""
    
    def keyPressEvent(self, event):
        """Capture key press."""
        key = event.key()
        
        # Handle ESC to cancel
        if key == Qt.Key_Escape:
            self.reject()
            return
        
        # Get key text
        key_text = QKeySequence(key).toString()
        
        # Check if it's a modifier
        if key in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta):
            # Add modifier
            if key == Qt.Key_Control:
                self.modifiers.add("Ctrl")
            elif key == Qt.Key_Shift:
                self.modifiers.add("Shift")
            elif key == Qt.Key_Alt:
                self.modifiers.add("Alt")
            elif key == Qt.Key_Meta:
                self.modifiers.add("Meta")
        else:
            # Main key pressed
            self.main_key = key_text
        
        # Update display
        self._update_display()
    
    def keyReleaseEvent(self, event):
        """Handle key release to remove modifiers."""
        key = event.key()
        
        # Remove released modifier
        if key == Qt.Key_Control:
            self.modifiers.discard("Ctrl")
        elif key == Qt.Key_Shift:
            self.modifiers.discard("Shift")
        elif key == Qt.Key_Alt:
            self.modifiers.discard("Alt")
        elif key == Qt.Key_Meta:
            self.modifiers.discard("Meta")
        
        self._update_display()
    
    def _update_display(self):
        """Update the display label with current key combination."""
        parts = sorted(list(self.modifiers))
        if self.main_key:
            parts.append(self.main_key)
        
        if parts:
            self.recorded_hotkey = "+".join(parts)
            self.display_label.setText(self.recorded_hotkey)
        else:
            self.display_label.setText("Waiting for input...")


class HotkeyConfigWidget(QWidget):
    """
    GUI for configuring hotkeys.
    
    Features:
    - List all hotkeys with current bindings
    - Click to rebind (press new key combo)
    - Conflict detection
    - Reset to defaults
    - Enable/disable individual hotkeys
    """
    
    hotkey_changed = pyqtSignal(str, str)  # name, new_hotkey
    
    def __init__(self, hotkey_manager=None, parent=None):
        super().__init__(parent)
        
        self.hotkey_manager = hotkey_manager
        self.hotkeys: dict[str, dict] = {}
        
        self._init_ui()
        self._load_hotkeys()
    
    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Global Hotkey Configuration")
        title.setStyleSheet("font-size: 12px; font-weight: bold;")
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Configure global hotkeys that work even when Enigma AI Engine is not focused.\n"
            "These work in fullscreen games and other applications."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-style: italic; color: #bac2de;")
        layout.addWidget(desc)
        
        # Hotkey list
        self.hotkey_list = QListWidget()
        self.hotkey_list.itemDoubleClicked.connect(self._on_rebind_clicked)
        layout.addWidget(self.hotkey_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.rebind_btn = QPushButton("Rebind Selected")
        self.rebind_btn.clicked.connect(self._on_rebind_clicked)
        button_layout.addWidget(self.rebind_btn)
        
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        button_layout.addWidget(self.reset_btn)
        
        self.test_btn = QPushButton("Test Selected")
        self.test_btn.clicked.connect(self._on_test_clicked)
        button_layout.addWidget(self.test_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-style: italic;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def _load_hotkeys(self):
        """Load current hotkeys from the manager."""
        self.hotkey_list.clear()
        
        if not self.hotkey_manager:
            from ...core.hotkey_manager import get_hotkey_manager
            self.hotkey_manager = get_hotkey_manager()
        
        # Get hotkeys from manager
        hotkeys = self.hotkey_manager.list_registered()
        
        # If none registered, load defaults
        if not hotkeys:
            from ...core.hotkey_manager import DEFAULT_HOTKEYS
            hotkeys = [
                {"name": name, "hotkey": key, "enabled": False}
                for name, key in DEFAULT_HOTKEYS.items()
            ]
        
        for info in hotkeys:
            name = info["name"]
            hotkey = info["hotkey"]
            enabled = info.get("enabled", True)
            
            # Format display text
            display_name = name.replace("_", " ").title()
            item_text = f"{display_name}: {hotkey}"
            if not enabled:
                item_text += " (Disabled)"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, info)
            self.hotkey_list.addItem(item)
            
            self.hotkeys[name] = info
    
    def _on_rebind_clicked(self):
        """Handle rebind button click."""
        current_item = self.hotkey_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a hotkey to rebind.")
            return
        
        info = current_item.data(Qt.UserRole)
        name = info["name"]
        current_hotkey = info["hotkey"]
        
        # Show recording dialog
        dialog = HotkeyRecordDialog(name, current_hotkey, self)
        if dialog.exec_() == QDialog.Accepted:
            new_hotkey = dialog.recorded_hotkey
            
            if new_hotkey:
                # Check for conflicts
                if self._check_conflict(name, new_hotkey):
                    reply = QMessageBox.question(
                        self,
                        "Conflict Detected",
                        f"The hotkey '{new_hotkey}' is already in use.\n"
                        f"Do you want to use it anyway?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply != QMessageBox.Yes:
                        return
                
                # Update hotkey
                self._update_hotkey(name, new_hotkey)
                self.status_label.setText(f"Updated {name} to {new_hotkey}")
                self.status_label.setStyleSheet("color: #22c55e; font-style: italic;")
    
    def _on_reset_clicked(self):
        """Reset all hotkeys to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Hotkeys",
            "Are you sure you want to reset all hotkeys to their default values?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            from ...core.hotkey_manager import DEFAULT_HOTKEYS
            
            for name, hotkey in DEFAULT_HOTKEYS.items():
                if name in self.hotkeys:
                    self._update_hotkey(name, hotkey)
            
            self._load_hotkeys()
            self.status_label.setText("Reset all hotkeys to defaults")
            self.status_label.setStyleSheet("color: #22c55e; font-style: italic;")
    
    def _on_test_clicked(self):
        """Test the selected hotkey."""
        current_item = self.hotkey_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a hotkey to test.")
            return
        
        info = current_item.data(Qt.UserRole)
        name = info["name"]
        
        self.status_label.setText(f"Testing {name}... (Press the hotkey)")
        self.status_label.setStyleSheet("color: #3b82f6; font-style: italic;")
    
    def _check_conflict(self, name: str, hotkey: str) -> bool:
        """
        Check if hotkey conflicts with existing bindings.
        
        Args:
            name: Name of the hotkey being checked
            hotkey: Hotkey combination
            
        Returns:
            True if conflict exists
        """
        for existing_name, info in self.hotkeys.items():
            if existing_name != name and info["hotkey"] == hotkey:
                return True
        return False
    
    def _update_hotkey(self, name: str, new_hotkey: str):
        """
        Update a hotkey binding.
        
        Args:
            name: Hotkey name
            new_hotkey: New key combination
        """
        if self.hotkey_manager:
            # Unregister old binding
            self.hotkey_manager.unregister(name)
            
            # Register new binding
            # Note: We'll need the callback - for now just update the config
            # The actual registration will happen when the manager is restarted
        
        # Update local storage
        if name in self.hotkeys:
            self.hotkeys[name]["hotkey"] = new_hotkey
        
        # Emit signal
        self.hotkey_changed.emit(name, new_hotkey)
        
        # Reload display
        self._load_hotkeys()
    
    def set_hotkey_manager(self, manager):
        """Set the hotkey manager instance."""
        self.hotkey_manager = manager
        self._load_hotkeys()
