"""
Keyboard Shortcuts System for Enigma AI Engine GUI

Provides customizable keyboard shortcuts with a management dialog.

Usage:
    from enigma_engine.gui.shortcuts import ShortcutManager, get_shortcut_manager
    
    manager = get_shortcut_manager()
    
    # Register an action
    manager.register_action("new_chat", "Start New Chat", lambda: print("New chat"))
    
    # Bind to window
    manager.bind_to_window(my_window)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QMessageBox, QHeaderView, QShortcut,
    QWidget, QKeySequenceEdit, QGroupBox, QComboBox
)

logger = logging.getLogger(__name__)


@dataclass
class ShortcutAction:
    """Represents a keyboard shortcut action."""
    id: str
    name: str
    description: str
    callback: Optional[Callable] = None
    default_shortcut: str = ""
    current_shortcut: str = ""
    category: str = "General"
    enabled: bool = True


class ShortcutManager:
    """
    Manages keyboard shortcuts for the application.
    
    Features:
    - Customizable shortcuts
    - Conflict detection
    - Persistent storage
    - Category organization
    """
    
    # Default shortcuts
    DEFAULT_SHORTCUTS = {
        # Chat actions
        "new_chat": "Ctrl+N",
        "send_message": "Ctrl+Return",
        "clear_chat": "Ctrl+L",
        "copy_response": "Ctrl+Shift+C",
        
        # Tab navigation
        "tab_chat": "Ctrl+1",
        "tab_build": "Ctrl+2",
        "tab_settings": "Ctrl+3",
        "tab_modules": "Ctrl+4",
        
        # Model operations
        "load_model": "Ctrl+O",
        "save_model": "Ctrl+S",
        "start_training": "Ctrl+T",
        "stop_training": "Ctrl+Shift+T",
        
        # Voice control
        "toggle_voice_input": "Ctrl+M",
        "toggle_voice_output": "Ctrl+Shift+M",
        
        # General
        "toggle_fullscreen": "F11",
        "open_settings": "Ctrl+,",
        "quit": "Ctrl+Q",
        "show_shortcuts": "Ctrl+/",
        "focus_input": "Ctrl+I",
        "toggle_sidebar": "Ctrl+B",
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        from ..config import CONFIG
        
        self.config_path = config_path or Path(
            CONFIG.get("data_dir", "data")
        ) / "keyboard_shortcuts.json"
        
        self._actions: Dict[str, ShortcutAction] = {}
        self._shortcuts: Dict[str, QShortcut] = {}
        self._bound_windows: Set[QWidget] = set()
        
        self._register_default_actions()
        self._load_config()
    
    def _register_default_actions(self):
        """Register default actions with their shortcuts."""
        actions_config = [
            # Chat
            ("new_chat", "New Chat", "Start a new conversation", "Chat"),
            ("send_message", "Send Message", "Send the current message", "Chat"),
            ("clear_chat", "Clear Chat", "Clear chat history", "Chat"),
            ("copy_response", "Copy Response", "Copy last AI response", "Chat"),
            
            # Navigation
            ("tab_chat", "Chat Tab", "Switch to Chat tab", "Navigation"),
            ("tab_build", "Build AI Tab", "Switch to Build AI tab", "Navigation"),
            ("tab_settings", "Settings Tab", "Switch to Settings tab", "Navigation"),
            ("tab_modules", "Modules Tab", "Switch to Modules tab", "Navigation"),
            
            # Model
            ("load_model", "Load Model", "Load a model from disk", "Model"),
            ("save_model", "Save Model", "Save current model", "Model"),
            ("start_training", "Start Training", "Begin model training", "Model"),
            ("stop_training", "Stop Training", "Stop current training", "Model"),
            
            # Voice
            ("toggle_voice_input", "Toggle Voice Input", "Enable/disable microphone", "Voice"),
            ("toggle_voice_output", "Toggle Voice Output", "Enable/disable TTS", "Voice"),
            
            # General
            ("toggle_fullscreen", "Toggle Fullscreen", "Toggle fullscreen mode", "General"),
            ("open_settings", "Open Settings", "Open settings dialog", "General"),
            ("quit", "Quit", "Exit the application", "General"),
            ("show_shortcuts", "Show Shortcuts", "Display shortcuts dialog", "General"),
            ("focus_input", "Focus Input", "Focus the message input", "General"),
            ("toggle_sidebar", "Toggle Sidebar", "Show/hide sidebar", "General"),
        ]
        
        for action_id, name, desc, category in actions_config:
            default = self.DEFAULT_SHORTCUTS.get(action_id, "")
            self._actions[action_id] = ShortcutAction(
                id=action_id,
                name=name,
                description=desc,
                default_shortcut=default,
                current_shortcut=default,
                category=category
            )
    
    def _load_config(self):
        """Load custom shortcuts from config file."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)
                
                custom = data.get("shortcuts", {})
                for action_id, shortcut in custom.items():
                    if action_id in self._actions:
                        self._actions[action_id].current_shortcut = shortcut
                
                logger.info(f"Loaded {len(custom)} custom shortcuts")
            except Exception as e:
                logger.warning(f"Could not load shortcuts config: {e}")
    
    def _save_config(self):
        """Save custom shortcuts to config file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Only save shortcuts that differ from defaults
            custom = {}
            for action_id, action in self._actions.items():
                if action.current_shortcut != action.default_shortcut:
                    custom[action_id] = action.current_shortcut
            
            with open(self.config_path, 'w') as f:
                json.dump({"shortcuts": custom}, f, indent=2)
            
            logger.debug("Saved shortcuts config")
        except Exception as e:
            logger.error(f"Could not save shortcuts config: {e}")
    
    def register_action(
        self,
        action_id: str,
        name: str,
        callback: Callable,
        description: str = "",
        category: str = "Custom",
        default_shortcut: str = ""
    ):
        """
        Register a new action.
        
        Args:
            action_id: Unique identifier
            name: Human-readable name
            callback: Function to call when triggered
            description: Action description
            category: Category for grouping
            default_shortcut: Default key sequence
        """
        if action_id in self._actions:
            # Update existing
            self._actions[action_id].callback = callback
            if not self._actions[action_id].current_shortcut:
                self._actions[action_id].current_shortcut = default_shortcut
        else:
            # Create new
            self._actions[action_id] = ShortcutAction(
                id=action_id,
                name=name,
                description=description,
                callback=callback,
                default_shortcut=default_shortcut,
                current_shortcut=default_shortcut,
                category=category
            )
    
    def set_callback(self, action_id: str, callback: Callable):
        """Set or update the callback for an action."""
        if action_id in self._actions:
            self._actions[action_id].callback = callback
            # Re-bind if windows are connected
            for window in self._bound_windows:
                self._bind_action(window, self._actions[action_id])
    
    def set_shortcut(self, action_id: str, shortcut: str) -> bool:
        """
        Set the keyboard shortcut for an action.
        
        Args:
            action_id: Action to modify
            shortcut: New key sequence (empty to disable)
            
        Returns:
            True if successful, False if conflict detected
        """
        if action_id not in self._actions:
            return False
        
        # Check for conflicts
        if shortcut:
            for aid, action in self._actions.items():
                if aid != action_id and action.current_shortcut == shortcut:
                    logger.warning(f"Shortcut conflict: {shortcut} already used by {aid}")
                    return False
        
        self._actions[action_id].current_shortcut = shortcut
        self._save_config()
        
        # Re-bind to all windows
        for window in self._bound_windows:
            self._bind_action(window, self._actions[action_id])
        
        return True
    
    def reset_shortcut(self, action_id: str):
        """Reset an action's shortcut to default."""
        if action_id in self._actions:
            action = self._actions[action_id]
            action.current_shortcut = action.default_shortcut
            self._save_config()
            
            for window in self._bound_windows:
                self._bind_action(window, action)
    
    def reset_all_shortcuts(self):
        """Reset all shortcuts to defaults."""
        for action in self._actions.values():
            action.current_shortcut = action.default_shortcut
        self._save_config()
        
        for window in self._bound_windows:
            self._rebind_all(window)
    
    def _bind_action(self, window: QWidget, action: ShortcutAction):
        """Bind a single action to a window."""
        key = f"{id(window)}_{action.id}"
        
        # Remove existing shortcut if any
        if key in self._shortcuts:
            self._shortcuts[key].setEnabled(False)
            self._shortcuts[key].deleteLater()
            del self._shortcuts[key]
        
        # Create new shortcut if action has one
        if action.current_shortcut and action.callback and action.enabled:
            try:
                shortcut = QShortcut(QKeySequence(action.current_shortcut), window)
                shortcut.activated.connect(action.callback)
                self._shortcuts[key] = shortcut
            except Exception as e:
                logger.warning(f"Failed to bind shortcut {action.current_shortcut}: {e}")
    
    def _rebind_all(self, window: QWidget):
        """Rebind all actions to a window."""
        for action in self._actions.values():
            self._bind_action(window, action)
    
    def bind_to_window(self, window: QWidget):
        """
        Bind all shortcuts to a window.
        
        Args:
            window: QWidget to bind shortcuts to
        """
        self._bound_windows.add(window)
        self._rebind_all(window)
        logger.info(f"Bound shortcuts to window {window}")
    
    def unbind_from_window(self, window: QWidget):
        """Remove all shortcuts from a window."""
        if window in self._bound_windows:
            self._bound_windows.remove(window)
            
            # Remove shortcuts for this window
            to_remove = [k for k in self._shortcuts if k.startswith(f"{id(window)}_")]
            for key in to_remove:
                self._shortcuts[key].setEnabled(False)
                self._shortcuts[key].deleteLater()
                del self._shortcuts[key]
    
    def get_action(self, action_id: str) -> Optional[ShortcutAction]:
        """Get an action by ID."""
        return self._actions.get(action_id)
    
    def get_all_actions(self) -> List[ShortcutAction]:
        """Get all registered actions."""
        return list(self._actions.values())
    
    def get_categories(self) -> List[str]:
        """Get all action categories."""
        return sorted(set(a.category for a in self._actions.values()))
    
    def get_actions_by_category(self, category: str) -> List[ShortcutAction]:
        """Get all actions in a category."""
        return [a for a in self._actions.values() if a.category == category]
    
    def check_conflict(self, shortcut: str, exclude_action: str = "") -> Optional[str]:
        """
        Check if a shortcut conflicts with existing ones.
        
        Returns:
            Action ID that conflicts, or None if no conflict
        """
        for action_id, action in self._actions.items():
            if action_id != exclude_action and action.current_shortcut == shortcut:
                return action_id
        return None


class ShortcutsDialog(QDialog):
    """Dialog for viewing and editing keyboard shortcuts."""
    
    def __init__(self, manager: ShortcutManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.pending_changes: Dict[str, str] = {}
        
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(600, 500)
        
        self._setup_ui()
        self._populate_table()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Category filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItem("All")
        for cat in self.manager.get_categories():
            self.category_combo.addItem(cat)
        self.category_combo.currentTextChanged.connect(self._filter_by_category)
        filter_layout.addWidget(self.category_combo)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        # Shortcuts table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Action", "Description", "Shortcut", "Default"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.cellDoubleClicked.connect(self._edit_shortcut)
        layout.addWidget(self.table)
        
        # Edit shortcut group
        edit_group = QGroupBox("Edit Selected Shortcut")
        edit_layout = QHBoxLayout(edit_group)
        
        edit_layout.addWidget(QLabel("Press keys:"))
        self.key_edit = QKeySequenceEdit()
        self.key_edit.keySequenceChanged.connect(self._on_key_sequence_changed)
        edit_layout.addWidget(self.key_edit)
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply_shortcut)
        edit_layout.addWidget(self.apply_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_shortcut)
        edit_layout.addWidget(self.clear_btn)
        
        layout.addWidget(edit_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        reset_btn = QPushButton("Reset Selected")
        reset_btn.clicked.connect(self._reset_selected)
        btn_layout.addWidget(reset_btn)
        
        reset_all_btn = QPushButton("Reset All")
        reset_all_btn.clicked.connect(self._reset_all)
        btn_layout.addWidget(reset_all_btn)
        
        btn_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
    
    def _populate_table(self, category_filter: str = "All"):
        """Populate the shortcuts table."""
        self.table.setRowCount(0)
        
        actions = self.manager.get_all_actions()
        if category_filter != "All":
            actions = [a for a in actions if a.category == category_filter]
        
        for action in sorted(actions, key=lambda a: (a.category, a.name)):
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            # Store action ID in first item
            name_item = QTableWidgetItem(action.name)
            name_item.setData(Qt.UserRole, action.id)
            self.table.setItem(row, 0, name_item)
            
            self.table.setItem(row, 1, QTableWidgetItem(action.description))
            self.table.setItem(row, 2, QTableWidgetItem(action.current_shortcut))
            self.table.setItem(row, 3, QTableWidgetItem(action.default_shortcut))
    
    def _filter_by_category(self, category: str):
        """Filter table by category."""
        self._populate_table(category)
    
    def _get_selected_action_id(self) -> Optional[str]:
        """Get the action ID of the selected row."""
        row = self.table.currentRow()
        if row >= 0:
            item = self.table.item(row, 0)
            return item.data(Qt.UserRole)
        return None
    
    def _edit_shortcut(self, row: int, column: int):
        """Start editing shortcut when double-clicked."""
        self.key_edit.clear()
        self.key_edit.setFocus()
    
    def _on_key_sequence_changed(self, sequence: QKeySequence):
        """Handle key sequence change in editor."""
        shortcut = sequence.toString()
        action_id = self._get_selected_action_id()
        
        if action_id and shortcut:
            conflict = self.manager.check_conflict(shortcut, action_id)
            if conflict:
                conflict_action = self.manager.get_action(conflict)
                name = conflict_action.name if conflict_action else conflict
                self.apply_btn.setEnabled(False)
                self.apply_btn.setToolTip(f"Conflicts with: {name}")
            else:
                self.apply_btn.setEnabled(True)
                self.apply_btn.setToolTip("")
    
    def _apply_shortcut(self):
        """Apply the edited shortcut."""
        action_id = self._get_selected_action_id()
        if not action_id:
            return
        
        shortcut = self.key_edit.keySequence().toString()
        
        if self.manager.set_shortcut(action_id, shortcut):
            self._populate_table(self.category_combo.currentText())
            self.key_edit.clear()
        else:
            QMessageBox.warning(self, "Conflict", "This shortcut conflicts with another action.")
    
    def _clear_shortcut(self):
        """Clear the shortcut for selected action."""
        action_id = self._get_selected_action_id()
        if action_id:
            self.manager.set_shortcut(action_id, "")
            self._populate_table(self.category_combo.currentText())
            self.key_edit.clear()
    
    def _reset_selected(self):
        """Reset selected shortcut to default."""
        action_id = self._get_selected_action_id()
        if action_id:
            self.manager.reset_shortcut(action_id)
            self._populate_table(self.category_combo.currentText())
    
    def _reset_all(self):
        """Reset all shortcuts to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset All Shortcuts",
            "Reset all shortcuts to their defaults?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.manager.reset_all_shortcuts()
            self._populate_table(self.category_combo.currentText())


# Global instance
_shortcut_manager: Optional[ShortcutManager] = None


def get_shortcut_manager() -> ShortcutManager:
    """Get or create global shortcut manager."""
    global _shortcut_manager
    if _shortcut_manager is None:
        _shortcut_manager = ShortcutManager()
    return _shortcut_manager


def show_shortcuts_dialog(parent=None):
    """Show the shortcuts configuration dialog."""
    dialog = ShortcutsDialog(get_shortcut_manager(), parent)
    dialog.exec_()
