"""
Keyboard Navigation System

Full keyboard-only navigation for the GUI without requiring a mouse.
Implements focus management, keyboard shortcuts, and navigation hints.

FILE: enigma_engine/gui/keyboard_navigation.py
TYPE: GUI Navigation System
MAIN CLASSES: KeyboardNavigator, FocusRing, NavigationHint
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from PyQt5.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QShortcut,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QWidget,
)

logger = logging.getLogger(__name__)


class NavigationDirection(Enum):
    """Direction of keyboard navigation."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    NEXT = "next"
    PREV = "prev"


@dataclass
class FocusableWidget:
    """Information about a focusable widget."""
    widget: QWidget
    group: str = "default"
    order: int = 0
    shortcut: Optional[str] = None
    label: str = ""
    
    
class NavigationHint(QLabel):
    """Visual hint showing keyboard shortcut."""
    
    def __init__(self, shortcut: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setText(shortcut)
        self.setStyleSheet("""
            QLabel {
                background-color: #f0a500;
                color: #000;
                font-weight: bold;
                font-size: 10px;
                padding: 2px 4px;
                border-radius: 3px;
                min-width: 16px;
            }
        """)
        self.setAlignment(Qt.AlignCenter)
        self.hide()


class FocusRing(QWidget):
    """Visual focus ring overlay."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._target: Optional[QWidget] = None
        self._visible = False
        self.hide()
        
    def track(self, widget: QWidget):
        """Track a widget with the focus ring."""
        self._target = widget
        self._update_geometry()
        if self._visible:
            self.show()
            
    def _update_geometry(self):
        """Update ring position to match target."""
        if self._target and self._target.isVisible():
            # Get widget geometry in parent coordinates
            parent = self.parent()
            if parent:
                pos = self._target.mapTo(parent, self._target.rect().topLeft())
                rect = self._target.rect()
                self.setGeometry(
                    pos.x() - 3,
                    pos.y() - 3,
                    rect.width() + 6,
                    rect.height() + 6
                )
                
    def set_visible(self, visible: bool):
        """Show or hide the focus ring."""
        self._visible = visible
        if visible and self._target:
            self._update_geometry()
            self.show()
        else:
            self.hide()
            
    def paintEvent(self, event):
        """Draw the focus ring."""
        from PyQt5.QtGui import QPainter, QPen
        
        if not self._visible:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pen = QPen(QColor("#00b4d8"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 4, 4)


class KeyboardNavigator(QObject):
    """Manages keyboard navigation for a window."""
    
    focus_changed = pyqtSignal(object)  # Emits focused widget
    navigation_mode_changed = pyqtSignal(bool)  # True when nav mode active
    
    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self._window = window
        self._enabled = True
        self._nav_mode = False  # Alt-key navigation mode
        self._widgets: dict[str, FocusableWidget] = {}
        self._groups: dict[str, list[str]] = {}  # group -> widget ids
        self._current_group = "default"
        self._hints: dict[str, NavigationHint] = {}
        self._focus_ring: Optional[FocusRing] = None
        self._shortcut_map: dict[str, str] = {}  # shortcut -> widget id
        self._setup()
        
    def _setup(self):
        """Setup keyboard navigation."""
        # Install event filter on window
        self._window.installEventFilter(self)
        
        # Create focus ring
        self._focus_ring = FocusRing(self._window)
        
        # Setup shortcuts
        self._setup_shortcuts()
        
        # Auto-register widgets
        QTimer.singleShot(100, self._auto_register)
        
    def _setup_shortcuts(self):
        """Setup global navigation shortcuts."""
        shortcuts = {
            "Alt+Left": lambda: self._navigate(NavigationDirection.LEFT),
            "Alt+Right": lambda: self._navigate(NavigationDirection.RIGHT),
            "Alt+Up": lambda: self._navigate(NavigationDirection.UP),
            "Alt+Down": lambda: self._navigate(NavigationDirection.DOWN),
            "Tab": lambda: self._navigate(NavigationDirection.NEXT),
            "Shift+Tab": lambda: self._navigate(NavigationDirection.PREV),
            "Escape": self._cancel_navigation,
            "F6": lambda: self._cycle_group(forward=True),
            "Shift+F6": lambda: self._cycle_group(forward=False),
        }
        
        for key, handler in shortcuts.items():
            shortcut = QShortcut(QKeySequence(key), self._window)
            shortcut.activated.connect(handler)
            
    def _auto_register(self):
        """Auto-register focusable widgets in the window."""
        focusable_types = (
            QPushButton, QLineEdit, QTextEdit, QComboBox,
            QCheckBox, QSpinBox, QSlider, QListWidget,
            QTreeWidget, QTableWidget, QToolButton
        )
        
        widgets = self._window.findChildren(QWidget)
        order = 0
        
        for widget in widgets:
            if isinstance(widget, focusable_types):
                if widget.isEnabled() and widget.isVisible():
                    # Determine group from parent
                    group = self._determine_group(widget)
                    
                    widget_id = f"widget_{id(widget)}"
                    self.register(widget, widget_id, group=group, order=order)
                    order += 1
                    
    def _determine_group(self, widget: QWidget) -> str:
        """Determine navigation group for a widget."""
        # Check parent hierarchy for group indicators
        parent = widget.parent()
        while parent:
            if isinstance(parent, QGroupBox):
                return parent.title() or "group"
            if isinstance(parent, QTabWidget):
                return "tabs"
            parent = parent.parent()
        return "default"
    
    def register(self, widget: QWidget, widget_id: str, 
                 group: str = "default", order: int = 0,
                 shortcut: Optional[str] = None, label: str = ""):
        """Register a widget for keyboard navigation.
        
        Args:
            widget: Widget to register
            widget_id: Unique identifier
            group: Navigation group name
            order: Tab order within group
            shortcut: Optional keyboard shortcut (e.g., "Alt+S")
            label: Label for navigation hints
        """
        self._widgets[widget_id] = FocusableWidget(
            widget=widget,
            group=group,
            order=order,
            shortcut=shortcut,
            label=label or widget_id
        )
        
        # Add to group
        if group not in self._groups:
            self._groups[group] = []
        self._groups[group].append(widget_id)
        
        # Setup shortcut if provided
        if shortcut:
            self._shortcut_map[shortcut] = widget_id
            sc = QShortcut(QKeySequence(shortcut), self._window)
            sc.activated.connect(lambda wid=widget_id: self._focus_widget(wid))
            
    def unregister(self, widget_id: str):
        """Unregister a widget."""
        if widget_id in self._widgets:
            info = self._widgets[widget_id]
            if info.group in self._groups:
                self._groups[info.group].remove(widget_id)
            del self._widgets[widget_id]
            
    def _navigate(self, direction: NavigationDirection):
        """Navigate in a direction."""
        if not self._enabled:
            return
            
        current = QApplication.focusWidget()
        
        # Get current widget id
        current_id = None
        for wid, info in self._widgets.items():
            if info.widget is current:
                current_id = wid
                break
                
        if current_id is None:
            # Focus first widget
            self._focus_first()
            return
            
        # Get widgets in current group, sorted by order
        group = self._widgets[current_id].group
        group_widgets = sorted(
            [self._widgets[wid] for wid in self._groups.get(group, [])],
            key=lambda w: w.order
        )
        
        if not group_widgets:
            return
            
        # Find current index
        current_idx = 0
        for i, w in enumerate(group_widgets):
            if w.widget is current:
                current_idx = i
                break
                
        # Calculate next index
        if direction in [NavigationDirection.NEXT, NavigationDirection.DOWN, NavigationDirection.RIGHT]:
            next_idx = (current_idx + 1) % len(group_widgets)
        else:
            next_idx = (current_idx - 1) % len(group_widgets)
            
        # Focus next widget
        next_widget = group_widgets[next_idx].widget
        self._set_focus(next_widget)
        
    def _focus_first(self):
        """Focus first widget in current group."""
        widgets = self._groups.get(self._current_group, [])
        if widgets:
            sorted_widgets = sorted(
                [self._widgets[wid] for wid in widgets],
                key=lambda w: w.order
            )
            if sorted_widgets:
                self._set_focus(sorted_widgets[0].widget)
                
    def _focus_widget(self, widget_id: str):
        """Focus a specific widget by ID."""
        if widget_id in self._widgets:
            self._set_focus(self._widgets[widget_id].widget)
            
    def _set_focus(self, widget: QWidget):
        """Set focus to a widget."""
        if widget and widget.isEnabled() and widget.isVisible():
            widget.setFocus(Qt.ShortcutFocusReason)
            self._update_focus_ring(widget)
            self.focus_changed.emit(widget)
            
    def _update_focus_ring(self, widget: QWidget):
        """Update focus ring position."""
        if self._focus_ring:
            self._focus_ring.track(widget)
            
    def _cycle_group(self, forward: bool = True):
        """Cycle through navigation groups."""
        groups = list(self._groups.keys())
        if not groups:
            return
            
        try:
            idx = groups.index(self._current_group)
        except ValueError:
            idx = 0
            
        if forward:
            idx = (idx + 1) % len(groups)
        else:
            idx = (idx - 1) % len(groups)
            
        self._current_group = groups[idx]
        self._focus_first()
        
    def _cancel_navigation(self):
        """Cancel navigation mode."""
        self._nav_mode = False
        self._hide_hints()
        self.navigation_mode_changed.emit(False)
        
    def enable_nav_mode(self):
        """Enable navigation mode with hints."""
        self._nav_mode = True
        self._show_hints()
        self.navigation_mode_changed.emit(True)
        
    def _show_hints(self):
        """Show keyboard hints on widgets."""
        for widget_id, info in self._widgets.items():
            if info.shortcut and info.widget.isVisible():
                hint = NavigationHint(info.shortcut, self._window)
                pos = info.widget.mapTo(self._window, info.widget.rect().topLeft())
                hint.move(pos.x(), pos.y() - 20)
                hint.show()
                self._hints[widget_id] = hint
                
    def _hide_hints(self):
        """Hide all keyboard hints."""
        for hint in self._hints.values():
            hint.deleteLater()
        self._hints.clear()
        
    def show_focus_ring(self, show: bool = True):
        """Show or hide the focus ring."""
        if self._focus_ring:
            self._focus_ring.set_visible(show)
            
    def set_enabled(self, enabled: bool):
        """Enable or disable keyboard navigation."""
        self._enabled = enabled
        
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Filter events for navigation handling."""
        if not self._enabled:
            return False
            
        if event.type() == QEvent.KeyPress:
            key_event = event
            
            # Alt key toggles nav mode
            if key_event.key() == Qt.Key_Alt and not key_event.isAutoRepeat():
                self.enable_nav_mode()
                return False
                
        elif event.type() == QEvent.KeyRelease:
            key_event = event
            
            # Alt release hides nav mode
            if key_event.key() == Qt.Key_Alt and not key_event.isAutoRepeat():
                self._cancel_navigation()
                return False
                
        elif event.type() == QEvent.FocusIn:
            # Update focus ring on focus change
            if isinstance(obj, QWidget):
                self._update_focus_ring(obj)
                
        return False


def get_keyboard_navigator(window: QMainWindow) -> KeyboardNavigator:
    """Get or create keyboard navigator for a window."""
    # Check if already exists
    for child in window.children():
        if isinstance(child, KeyboardNavigator):
            return child
    return KeyboardNavigator(window)


def make_focusable(widget: QWidget, order: int = 0, 
                   shortcut: Optional[str] = None,
                   label: str = "") -> QWidget:
    """Helper to make a widget focusable with navigation support.
    
    Args:
        widget: Widget to configure
        order: Tab order
        shortcut: Keyboard shortcut
        label: Navigation label
        
    Returns:
        The configured widget
    """
    widget.setFocusPolicy(Qt.StrongFocus)
    widget.setProperty("nav_order", order)
    if shortcut:
        widget.setProperty("nav_shortcut", shortcut)
    if label:
        widget.setProperty("nav_label", label)
    return widget


__all__ = [
    'KeyboardNavigator',
    'FocusRing',
    'NavigationHint',
    'NavigationDirection',
    'FocusableWidget',
    'get_keyboard_navigator',
    'make_focusable'
]
