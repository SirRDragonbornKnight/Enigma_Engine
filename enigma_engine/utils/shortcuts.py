"""
Keyboard Shortcuts System for Forge GUI

Provides customizable keyboard shortcuts for faster navigation and actions.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut, QWidget

logger = logging.getLogger(__name__)


class ShortcutManager:
    """
    Manages keyboard shortcuts for the GUI.
    
    Features:
    - Predefined shortcuts for common actions
    - Custom shortcut definition
    - Shortcut persistence
    - Conflict detection
    """
    
    # Default shortcuts
    DEFAULT_SHORTCUTS = {
        # Navigation
        "focus_input": "Ctrl+L",
        "focus_chat": "Ctrl+1",
        "focus_training": "Ctrl+2",
        "focus_vision": "Ctrl+3",
        "focus_settings": "Ctrl+4",
        "next_tab": "Ctrl+Tab",
        "prev_tab": "Ctrl+Shift+Tab",
        
        # Actions
        "send_message": "Ctrl+Return",
        "clear_chat": "Ctrl+Shift+C",
        "new_conversation": "Ctrl+N",
        "save_conversation": "Ctrl+S",
        "copy_response": "Ctrl+Shift+C",
        
        # Training
        "start_training": "Ctrl+T",
        "stop_training": "Ctrl+Shift+T",
        "load_data": "Ctrl+O",
        
        # Undo/Redo
        "undo": "Ctrl+Z",
        "redo": "Ctrl+Y",
        
        # Model management
        "switch_model": "Ctrl+M",
        "reload_model": "Ctrl+R",
        
        # View
        "toggle_sidebar": "Ctrl+B",
        "toggle_fullscreen": "F11",
        "zoom_in": "Ctrl++",
        "zoom_out": "Ctrl+-",
        "reset_zoom": "Ctrl+0",
        
        # Tools
        "open_file": "Ctrl+O",
        "screenshot": "Ctrl+Shift+S",
        "voice_input": "Ctrl+Shift+V",
        
        # Application
        "quit": "Ctrl+Q",
        "preferences": "Ctrl+,",
        "help": "F1",
    }
    
    def __init__(self, parent: QWidget, storage_path: Optional[Path] = None):
        """
        Initialize shortcut manager.
        
        Args:
            parent: Parent widget for shortcuts
            storage_path: Path to save custom shortcuts
        """
        self.parent = parent
        
        if storage_path is None:
            from ..config import CONFIG
            storage_path = Path(CONFIG["data_dir"]) / "shortcuts.json"
        
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load custom shortcuts or use defaults
        self.shortcuts = self._load_shortcuts()
        
        # Store QShortcut objects
        self._shortcut_objects: dict[str, QShortcut] = {}
        
        # Store action callbacks
        self._callbacks: dict[str, Callable] = {}
    
    def _load_shortcuts(self) -> dict[str, str]:
        """Load shortcuts from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    custom = json.load(f)
                # Merge with defaults
                shortcuts = self.DEFAULT_SHORTCUTS.copy()
                shortcuts.update(custom)
                return shortcuts
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted shortcuts file, using defaults: {e}")
                return self.DEFAULT_SHORTCUTS.copy()
            except OSError as e:
                logger.warning(f"Could not read shortcuts file: {e}")
                return self.DEFAULT_SHORTCUTS.copy()
        return self.DEFAULT_SHORTCUTS.copy()
    
    def _save_shortcuts(self):
        """Save shortcuts to storage."""
        with open(self.storage_path, 'w') as f:
            json.dump(self.shortcuts, f, indent=2)
    
    def register(self, action: str, callback: Callable, 
                 custom_key: Optional[str] = None) -> bool:
        """
        Register a keyboard shortcut.
        
        Args:
            action: Action name (e.g., 'send_message')
            callback: Function to call when shortcut is triggered
            custom_key: Optional custom key sequence (overrides default)
            
        Returns:
            True if registered successfully
        """
        # Get key sequence
        if custom_key:
            key_seq = custom_key
        else:
            key_seq = self.shortcuts.get(action)
        
        if not key_seq:
            return False
        
        try:
            # Create QShortcut
            shortcut = QShortcut(QKeySequence(key_seq), self.parent)
            shortcut.activated.connect(callback)
            
            # Store
            self._shortcut_objects[action] = shortcut
            self._callbacks[action] = callback
            
            return True
        except Exception as e:
            print(f"Failed to register shortcut {action}: {e}")
            return False
    
    def unregister(self, action: str):
        """Unregister a shortcut."""
        if action in self._shortcut_objects:
            self._shortcut_objects[action].setEnabled(False)
            del self._shortcut_objects[action]
            if action in self._callbacks:
                del self._callbacks[action]
    
    def update_shortcut(self, action: str, new_key: str) -> bool:
        """
        Update a shortcut's key sequence.
        
        Args:
            action: Action name
            new_key: New key sequence
            
        Returns:
            True if updated successfully
        """
        # Check for conflicts
        for act, key in self.shortcuts.items():
            if act != action and key == new_key:
                print(f"Conflict: {new_key} already used by {act}")
                return False
        
        # Update
        self.shortcuts[action] = new_key
        self._save_shortcuts()
        
        # Re-register if already registered
        if action in self._callbacks:
            callback = self._callbacks[action]
            self.unregister(action)
            self.register(action, callback, new_key)
        
        return True
    
    def get_shortcut(self, action: str) -> Optional[str]:
        """Get the key sequence for an action."""
        return self.shortcuts.get(action)
    
    def get_all_shortcuts(self) -> dict[str, str]:
        """Get all registered shortcuts."""
        return self.shortcuts.copy()
    
    def reset_to_defaults(self):
        """Reset all shortcuts to defaults."""
        self.shortcuts = self.DEFAULT_SHORTCUTS.copy()
        self._save_shortcuts()
        
        # Re-register all
        for action, callback in list(self._callbacks.items()):
            self.unregister(action)
            self.register(action, callback)
    
    def get_shortcut_description(self, action: str) -> str:
        """Get human-readable description of a shortcut."""
        descriptions = {
            "focus_input": "Focus input field",
            "focus_chat": "Go to Chat tab",
            "focus_training": "Go to Training tab",
            "focus_vision": "Go to Vision tab",
            "focus_settings": "Go to Settings tab",
            "next_tab": "Next tab",
            "prev_tab": "Previous tab",
            "send_message": "Send message",
            "clear_chat": "Clear chat history",
            "new_conversation": "Start new conversation",
            "save_conversation": "Save conversation",
            "copy_response": "Copy last response",
            "start_training": "Start training",
            "stop_training": "Stop training",
            "load_data": "Load training data",
            "undo": "Undo last action",
            "redo": "Redo last action",
            "switch_model": "Switch AI model",
            "reload_model": "Reload current model",
            "toggle_sidebar": "Toggle sidebar",
            "toggle_fullscreen": "Toggle fullscreen",
            "zoom_in": "Zoom in",
            "zoom_out": "Zoom out",
            "reset_zoom": "Reset zoom",
            "open_file": "Open file",
            "screenshot": "Take screenshot",
            "voice_input": "Start voice input",
            "quit": "Quit application",
            "preferences": "Open preferences",
            "help": "Show help",
        }
        return descriptions.get(action, action)
    
    def format_for_display(self) -> str:
        """Format all shortcuts for display in help/settings."""
        lines = ["Keyboard Shortcuts", "=" * 50, ""]
        
        # Group by category
        categories = {
            "Navigation": ["focus_input", "focus_chat", "focus_training", "focus_vision", 
                          "focus_settings", "next_tab", "prev_tab"],
            "Chat Actions": ["send_message", "clear_chat", "new_conversation", 
                           "save_conversation", "copy_response"],
            "Training": ["start_training", "stop_training", "load_data"],
            "Edit": ["undo", "redo"],
            "Model": ["switch_model", "reload_model"],
            "View": ["toggle_sidebar", "toggle_fullscreen", "zoom_in", 
                    "zoom_out", "reset_zoom"],
            "Tools": ["open_file", "screenshot", "voice_input"],
            "Application": ["quit", "preferences", "help"],
        }
        
        for category, actions in categories.items():
            lines.append(f"\n{category}:")
            lines.append("-" * 50)
            for action in actions:
                if action in self.shortcuts:
                    key = self.shortcuts[action]
                    desc = self.get_shortcut_description(action)
                    lines.append(f"  {key:20} {desc}")
        
        return "\n".join(lines)


class UndoRedoManager:
    """
    Manages undo/redo functionality for GUI actions.
    
    Tracks reversible actions and allows undo/redo.
    """
    
    def __init__(self, max_history: int = 50):
        """
        Initialize undo/redo manager.
        
        Args:
            max_history: Maximum number of actions to remember
        """
        self.max_history = max_history
        self.undo_stack = []
        self.redo_stack = []
    
    def push_action(self, action: dict):
        """
        Push a new action onto the undo stack.
        
        Args:
            action: Dict with 'type', 'data', and 'undo_func' keys
        """
        self.undo_stack.append(action)
        
        # Limit stack size
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        
        # Clear redo stack when new action is performed
        self.redo_stack.clear()
    
    def undo(self) -> bool:
        """
        Undo the last action.
        
        Returns:
            True if undo was performed
        """
        if not self.undo_stack:
            return False
        
        action = self.undo_stack.pop()
        
        # Execute undo function
        if 'undo_func' in action and action['undo_func']:
            try:
                action['undo_func']()
            except Exception as e:
                print(f"Undo failed: {e}")
                return False
        
        # Move to redo stack
        self.redo_stack.append(action)
        return True
    
    def redo(self) -> bool:
        """
        Redo the last undone action.
        
        Returns:
            True if redo was performed
        """
        if not self.redo_stack:
            return False
        
        action = self.redo_stack.pop()
        
        # Execute redo function
        if 'redo_func' in action and action['redo_func']:
            try:
                action['redo_func']()
            except Exception as e:
                print(f"Redo failed: {e}")
                return False
        
        # Move back to undo stack
        self.undo_stack.append(action)
        return True
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self.redo_stack) > 0
    
    def clear(self):
        """Clear all undo/redo history."""
        self.undo_stack.clear()
        self.redo_stack.clear()
    
    def get_undo_description(self) -> Optional[str]:
        """Get description of the last undoable action."""
        if self.undo_stack:
            return self.undo_stack[-1].get('description', 'Unknown action')
        return None
    
    def get_redo_description(self) -> Optional[str]:
        """Get description of the last redoable action."""
        if self.redo_stack:
            return self.redo_stack[-1].get('description', 'Unknown action')
        return None


if __name__ == "__main__":
    # Test shortcut formatting
    import sys

    from PyQt5.QtWidgets import QApplication, QMainWindow
    
    app = QApplication(sys.argv)
    window = QMainWindow()
    
    manager = ShortcutManager(window)
    print(manager.format_for_display())
    
    # Test undo/redo
    print("\n\nUndo/Redo Test:")
    print("=" * 50)
    
    undo_redo = UndoRedoManager()
    
    # Simulate some actions
    messages = []
    
    def add_message(msg):
        messages.append(msg)
        print(f"Added: {msg}")
    
    def remove_last_message():
        if messages:
            msg = messages.pop()
            print(f"Removed: {msg}")
    
    # Add some messages
    for i in range(3):
        msg = f"Message {i+1}"
        add_message(msg)
        undo_redo.push_action({
            'description': f'Add message "{msg}"',
            'undo_func': remove_last_message,
            'redo_func': lambda m=msg: add_message(m)
        })
    
    print(f"\nMessages: {messages}")
    print(f"Can undo: {undo_redo.can_undo()}")
    
    # Undo
    print("\nUndoing...")
    undo_redo.undo()
    print(f"Messages: {messages}")
    
    # Redo
    print("\nRedoing...")
    undo_redo.redo()
    print(f"Messages: {messages}")
