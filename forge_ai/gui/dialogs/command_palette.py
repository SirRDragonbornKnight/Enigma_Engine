"""
Command Palette for ForgeAI GUI.

Provides a VS Code-style Ctrl+K command palette for quick access
to all GUI actions and features.
"""
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


@dataclass
class Command:
    """A command that can be executed from the palette."""
    id: str
    name: str
    description: str = ""
    category: str = "General"
    shortcut: str = ""
    callback: Optional[Callable] = None
    enabled: bool = True
    icon: str = ""
    keywords: list[str] = field(default_factory=list)


class CommandRegistry:
    """Registry for all available commands."""
    
    _instance: Optional['CommandRegistry'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_storage()
        return cls._instance
    
    def _init_storage(self):
        """Initialize storage attributes."""
        self._commands: dict[str, Command] = {}
        self._categories: dict[str, list[str]] = {}
    
    def register(self, command: Command) -> None:
        """Register a command."""
        self._commands[command.id] = command
        
        if command.category not in self._categories:
            self._categories[command.category] = []
        self._categories[command.category].append(command.id)
        
        logger.debug(f"Registered command: {command.id}")
    
    def unregister(self, command_id: str) -> None:
        """Unregister a command."""
        if command_id in self._commands:
            cmd = self._commands[command_id]
            if cmd.category in self._categories:
                self._categories[cmd.category].remove(command_id)
            del self._commands[command_id]
    
    def get(self, command_id: str) -> Optional[Command]:
        """Get a command by ID."""
        return self._commands.get(command_id)
    
    def get_all(self) -> list[Command]:
        """Get all registered commands."""
        return list(self._commands.values())
    
    def get_by_category(self, category: str) -> list[Command]:
        """Get commands by category."""
        ids = self._categories.get(category, [])
        return [self._commands[id] for id in ids if id in self._commands]
    
    def search(self, query: str) -> list[Command]:
        """
        Search commands by name, description, and keywords.
        
        Args:
            query: Search query (case-insensitive)
            
        Returns:
            List of matching commands, sorted by relevance
        """
        if not query:
            return self.get_all()
        
        query_lower = query.lower()
        results = []
        
        for cmd in self._commands.values():
            if not cmd.enabled:
                continue
                
            score = 0
            
            # Name match (highest priority)
            if query_lower in cmd.name.lower():
                score += 100
                if cmd.name.lower().startswith(query_lower):
                    score += 50
            
            # ID match
            if query_lower in cmd.id.lower():
                score += 80
            
            # Category match
            if query_lower in cmd.category.lower():
                score += 30
            
            # Description match
            if query_lower in cmd.description.lower():
                score += 20
            
            # Keyword match
            for keyword in cmd.keywords:
                if query_lower in keyword.lower():
                    score += 40
                    break
            
            if score > 0:
                results.append((score, cmd))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[0], reverse=True)
        return [cmd for _, cmd in results]
    
    def execute(self, command_id: str, *args, **kwargs) -> Any:
        """Execute a command by ID."""
        cmd = self.get(command_id)
        if not cmd:
            logger.warning(f"Command not found: {command_id}")
            return None
        
        if not cmd.enabled:
            logger.warning(f"Command disabled: {command_id}")
            return None
        
        if not cmd.callback:
            logger.warning(f"Command has no callback: {command_id}")
            return None
        
        try:
            return cmd.callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing command {command_id}: {e}")
            return None


class CommandPaletteDialog(QDialog):
    """
    VS Code-style command palette dialog.
    
    Usage:
        palette = CommandPaletteDialog(parent_window)
        palette.exec_()
    """
    
    command_executed = pyqtSignal(str)  # Emits command ID when executed
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.registry = CommandRegistry()
        self._setup_ui()
        self._setup_shortcuts()
        self._populate_commands()
        
        # Apply transparency
        try:
            from ..ui_settings import apply_dialog_transparency
            apply_dialog_transparency(self)
        except ImportError:
            pass
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Command Palette")
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setFixedWidth(600)
        self.setMaximumHeight(400)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type to search commands...")
        self.search_input.textChanged.connect(self._on_search)
        self.search_input.returnPressed.connect(self._execute_selected)
        layout.addWidget(self.search_input)
        
        # Results list
        self.results_list = QListWidget()
        self.results_list.setAlternatingRowColors(True)
        self.results_list.itemActivated.connect(self._on_item_activated)
        self.results_list.itemDoubleClicked.connect(self._on_item_activated)
        layout.addWidget(self.results_list)
        
        # Status label
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)
        
        # Apply styling
        self._apply_style()
    
    def _apply_style(self):
        """Apply palette styling."""
        self.setStyleSheet("""
            CommandPaletteDialog {
                background-color: #1e1e2e;
                border: 1px solid #45475a;
                border-radius: 8px;
            }
            QLineEdit {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #89b4fa;
            }
            QListWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: none;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #313244;
            }
            QListWidget::item:hover {
                background-color: #45475a;
            }
        """)
    
    def _setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        # Escape to close
        QShortcut(QKeySequence("Escape"), self, self.close)
        
        # Arrow navigation
        self.search_input.installEventFilter(self)
    
    def eventFilter(self, obj, event):
        """Handle keyboard navigation."""
        from PyQt5.QtCore import QEvent
        from PyQt5.QtGui import QKeyEvent
        
        if obj == self.search_input and event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Down:
                self._move_selection(1)
                return True
            elif key == Qt.Key_Up:
                self._move_selection(-1)
                return True
        
        return super().eventFilter(obj, event)
    
    def _move_selection(self, delta: int):
        """Move selection up or down."""
        current = self.results_list.currentRow()
        new_row = max(0, min(current + delta, self.results_list.count() - 1))
        self.results_list.setCurrentRow(new_row)
    
    def _populate_commands(self):
        """Populate the results list with all commands."""
        commands = self.registry.get_all()
        self._show_commands(commands)
    
    def _show_commands(self, commands: list[Command]):
        """Display commands in the results list."""
        self.results_list.clear()
        
        for cmd in commands:
            item = QListWidgetItem()
            
            # Format display text
            display = f"{cmd.name}"
            if cmd.shortcut:
                display += f"  ({cmd.shortcut})"
            
            item.setText(display)
            item.setData(Qt.UserRole, cmd.id)
            
            if cmd.description:
                item.setToolTip(cmd.description)
            
            if not cmd.enabled:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            
            self.results_list.addItem(item)
        
        # Select first item
        if self.results_list.count() > 0:
            self.results_list.setCurrentRow(0)
        
        # Update status
        self.status_label.setText(f"{len(commands)} commands")
    
    def _on_search(self, query: str):
        """Handle search input change."""
        results = self.registry.search(query)
        self._show_commands(results)
    
    def _on_item_activated(self, item: QListWidgetItem):
        """Handle item activation (double-click or enter)."""
        command_id = item.data(Qt.UserRole)
        self._execute_command(command_id)
    
    def _execute_selected(self):
        """Execute the currently selected command."""
        current = self.results_list.currentItem()
        if current:
            command_id = current.data(Qt.UserRole)
            self._execute_command(command_id)
    
    def _execute_command(self, command_id: str):
        """Execute a command and close the palette."""
        self.command_executed.emit(command_id)
        self.registry.execute(command_id)
        self.close()
    
    def showEvent(self, event):
        """Focus search input when shown."""
        super().showEvent(event)
        self.search_input.setFocus()
        self.search_input.selectAll()
        
        # Center on parent
        if self.parent():
            parent_rect = self.parent().geometry()
            x = parent_rect.x() + (parent_rect.width() - self.width()) // 2
            y = parent_rect.y() + 100  # Near top of window
            self.move(x, y)


def get_command_registry() -> CommandRegistry:
    """Get the global command registry."""
    return CommandRegistry()


def register_default_commands(main_window) -> None:
    """
    Register default commands for the main window.
    
    Args:
        main_window: The main EnhancedMainWindow instance
    """
    registry = get_command_registry()
    
    # Navigation commands
    registry.register(Command(
        id="nav.chat",
        name="Go to Chat",
        description="Switch to the Chat tab",
        category="Navigation",
        shortcut="Alt+1",
        callback=lambda: main_window.switch_to_tab(0) if hasattr(main_window, 'switch_to_tab') else None,
        keywords=["talk", "conversation", "message"]
    ))
    
    registry.register(Command(
        id="nav.image",
        name="Go to Image Generation",
        description="Switch to the Image Generation tab",
        category="Navigation",
        shortcut="Alt+2",
        callback=lambda: main_window.switch_to_tab(1) if hasattr(main_window, 'switch_to_tab') else None,
        keywords=["picture", "art", "dalle", "stable diffusion"]
    ))
    
    registry.register(Command(
        id="nav.code",
        name="Go to Code Generation",
        description="Switch to the Code Generation tab",
        category="Navigation",
        shortcut="Alt+3",
        callback=lambda: main_window.switch_to_tab(2) if hasattr(main_window, 'switch_to_tab') else None,
        keywords=["programming", "python", "javascript"]
    ))
    
    registry.register(Command(
        id="nav.settings",
        name="Open Settings",
        description="Switch to the Settings tab",
        category="Navigation",
        callback=lambda: main_window.switch_to_tab('settings') if hasattr(main_window, 'switch_to_tab') else None,
        keywords=["preferences", "config", "options"]
    ))
    
    # File commands
    registry.register(Command(
        id="file.new_conversation",
        name="New Conversation",
        description="Start a new conversation",
        category="File",
        shortcut="Ctrl+N",
        callback=lambda: main_window.new_conversation() if hasattr(main_window, 'new_conversation') else None,
        keywords=["clear", "fresh", "reset"]
    ))
    
    registry.register(Command(
        id="file.save_conversation",
        name="Save Conversation",
        description="Save the current conversation",
        category="File",
        shortcut="Ctrl+S",
        callback=lambda: main_window.save_conversation() if hasattr(main_window, 'save_conversation') else None,
        keywords=["export", "backup"]
    ))
    
    registry.register(Command(
        id="file.load_conversation",
        name="Load Conversation",
        description="Load a saved conversation",
        category="File",
        shortcut="Ctrl+O",
        callback=lambda: main_window.load_conversation() if hasattr(main_window, 'load_conversation') else None,
        keywords=["import", "open", "restore"]
    ))
    
    # Model commands
    registry.register(Command(
        id="model.switch",
        name="Switch Model",
        description="Change the active AI model",
        category="Model",
        callback=lambda: main_window.show_model_selector() if hasattr(main_window, 'show_model_selector') else None,
        keywords=["change", "select", "choose"]
    ))
    
    registry.register(Command(
        id="model.reload",
        name="Reload Model",
        description="Reload the current model",
        category="Model",
        callback=lambda: main_window.reload_model() if hasattr(main_window, 'reload_model') else None,
        keywords=["refresh", "restart"]
    ))
    
    # View commands
    registry.register(Command(
        id="view.toggle_sidebar",
        name="Toggle Sidebar",
        description="Show or hide the sidebar",
        category="View",
        shortcut="Ctrl+B",
        callback=lambda: main_window.toggle_sidebar() if hasattr(main_window, 'toggle_sidebar') else None,
        keywords=["panel", "hide", "show"]
    ))
    
    registry.register(Command(
        id="view.toggle_terminal",
        name="Toggle Terminal",
        description="Show or hide the terminal output",
        category="View",
        shortcut="Ctrl+`",
        callback=lambda: main_window.toggle_terminal() if hasattr(main_window, 'toggle_terminal') else None,
        keywords=["console", "log", "output"]
    ))
    
    registry.register(Command(
        id="view.fullscreen",
        name="Toggle Fullscreen",
        description="Enter or exit fullscreen mode",
        category="View",
        shortcut="F11",
        callback=lambda: main_window.toggle_fullscreen() if hasattr(main_window, 'toggle_fullscreen') else None,
        keywords=["maximize", "window"]
    ))
    
    # Theme commands
    registry.register(Command(
        id="theme.cycle",
        name="Cycle Theme",
        description="Switch to the next theme",
        category="Appearance",
        callback=lambda: main_window.cycle_theme() if hasattr(main_window, 'cycle_theme') else None,
        keywords=["dark", "light", "colors"]
    ))
    
    registry.register(Command(
        id="theme.editor",
        name="Open Theme Editor",
        description="Create or edit custom themes",
        category="Appearance",
        callback=lambda: main_window.open_theme_editor() if hasattr(main_window, 'open_theme_editor') else None,
        keywords=["customize", "colors", "style"]
    ))
    
    # Help commands
    registry.register(Command(
        id="help.documentation",
        name="Open Documentation",
        description="Open the ForgeAI documentation",
        category="Help",
        shortcut="F1",
        callback=lambda: main_window.open_docs() if hasattr(main_window, 'open_docs') else None,
        keywords=["guide", "manual", "tutorial"]
    ))
    
    registry.register(Command(
        id="help.shortcuts",
        name="Show Keyboard Shortcuts",
        description="Display all keyboard shortcuts",
        category="Help",
        shortcut="Ctrl+/",
        callback=lambda: main_window.show_shortcuts() if hasattr(main_window, 'show_shortcuts') else None,
        keywords=["hotkeys", "keys"]
    ))
    
    # Tool commands
    registry.register(Command(
        id="tools.web_search",
        name="Web Search",
        description="Search the web for information",
        category="Tools",
        callback=lambda: main_window.trigger_tool('web_search') if hasattr(main_window, 'trigger_tool') else None,
        keywords=["google", "bing", "internet"]
    ))
    
    registry.register(Command(
        id="tools.screenshot",
        name="Take Screenshot",
        description="Capture a screenshot for analysis",
        category="Tools",
        callback=lambda: main_window.trigger_tool('screenshot') if hasattr(main_window, 'trigger_tool') else None,
        keywords=["capture", "screen", "image"]
    ))
    
    logger.info("Registered default commands for command palette")


def setup_command_palette_shortcut(main_window) -> None:
    """
    Set up Ctrl+K shortcut to open command palette.
    
    Args:
        main_window: The main window to attach the shortcut to
    """
    def show_palette():
        palette = CommandPaletteDialog(main_window)
        palette.exec_()
    
    shortcut = QShortcut(QKeySequence("Ctrl+K"), main_window)
    shortcut.activated.connect(show_palette)
    
    # Also register as a command
    registry = get_command_registry()
    registry.register(Command(
        id="palette.show",
        name="Show Command Palette",
        description="Open the command palette",
        category="Navigation",
        shortcut="Ctrl+K",
        callback=show_palette,
        keywords=["search", "find", "quick"]
    ))
    
    logger.info("Command palette shortcut (Ctrl+K) registered")
