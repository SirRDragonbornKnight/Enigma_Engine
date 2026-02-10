"""
Accessibility Support for Enigma AI Engine GUI.

Provides screen reader support, keyboard navigation,
high contrast themes, and reduced motion options.
"""
import logging
from typing import Any, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSlider,
    QWidget,
)

# QAccessible may not be available in all PyQt5 installations
try:
    from PyQt5.QtGui import QAccessible
    QACCESSIBLE_AVAILABLE = True
except ImportError:
    QACCESSIBLE_AVAILABLE = False

logger = logging.getLogger(__name__)


class AccessibilityConfig:
    """Accessibility configuration settings."""
    
    def __init__(self):
        self.screen_reader_enabled = True
        self.high_contrast_enabled = False
        self.reduced_motion_enabled = False
        self.large_text_enabled = False
        self.focus_highlight_enabled = True
        self.keyboard_nav_only = False
        self.announce_notifications = True
        self.text_scale_factor = 1.0
        self.focus_ring_width = 3
        self.animation_duration_ms = 200
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "screen_reader_enabled": self.screen_reader_enabled,
            "high_contrast_enabled": self.high_contrast_enabled,
            "reduced_motion_enabled": self.reduced_motion_enabled,
            "large_text_enabled": self.large_text_enabled,
            "focus_highlight_enabled": self.focus_highlight_enabled,
            "keyboard_nav_only": self.keyboard_nav_only,
            "announce_notifications": self.announce_notifications,
            "text_scale_factor": self.text_scale_factor,
            "focus_ring_width": self.focus_ring_width,
            "animation_duration_ms": self.animation_duration_ms,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'AccessibilityConfig':
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class AccessibilityManager:
    """
    Manager for accessibility features.
    
    Provides:
    - Screen reader announcements
    - Accessible descriptions for widgets
    - High contrast theme support
    - Reduced motion handling
    - Keyboard navigation enhancements
    """
    
    _instance: Optional['AccessibilityManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.config = AccessibilityConfig()
        self._widget_descriptions: dict[int, str] = {}
        self._live_region_widgets: dict[int, QWidget] = {}
        self._announcement_queue = []
        self._initialized = True
        
        logger.info("AccessibilityManager initialized")
    
    def configure(self, config: AccessibilityConfig) -> None:
        """Apply accessibility configuration."""
        self.config = config
        self._apply_config()
    
    def _apply_config(self) -> None:
        """Apply current configuration to the application."""
        app = QApplication.instance()
        if not app:
            return
        
        # Apply text scaling
        if self.config.large_text_enabled or self.config.text_scale_factor != 1.0:
            font = app.font()
            base_size = 11  # Default base size
            font.setPointSize(int(base_size * self.config.text_scale_factor))
            app.setFont(font)
        
        # Apply high contrast if enabled
        if self.config.high_contrast_enabled:
            self._apply_high_contrast()
        
        logger.info("Accessibility config applied")
    
    def _apply_high_contrast(self) -> None:
        """Apply high contrast stylesheet."""
        high_contrast_style = """
            QWidget {
                background-color: #000000;
                color: #ffffff;
            }
            QPushButton {
                background-color: #000000;
                color: #ffff00;
                border: 2px solid #ffff00;
            }
            QPushButton:hover, QPushButton:focus {
                background-color: #ffff00;
                color: #000000;
            }
            QLineEdit, QTextEdit, QPlainTextEdit {
                background-color: #000000;
                color: #ffffff;
                border: 2px solid #ffffff;
            }
            QListWidget::item:selected {
                background-color: #ffff00;
                color: #000000;
            }
            QTabBar::tab:selected {
                background-color: #ffff00;
                color: #000000;
            }
            a, QLabel[link="true"] {
                color: #00ffff;
            }
        """
        
        app = QApplication.instance()
        if app:
            # Append to existing stylesheet
            current = app.styleSheet()
            if "high-contrast" not in current:
                app.setStyleSheet(current + "\n/* high-contrast */\n" + high_contrast_style)
    
    def set_accessible_name(self, widget: QWidget, name: str) -> None:
        """Set the accessible name for a widget."""
        widget.setAccessibleName(name)
        logger.debug(f"Set accessible name for {widget.__class__.__name__}: {name}")
    
    def set_accessible_description(self, widget: QWidget, description: str) -> None:
        """Set the accessible description for a widget."""
        widget.setAccessibleDescription(description)
        self._widget_descriptions[id(widget)] = description
    
    def announce(self, message: str, priority: bool = False) -> None:
        """
        Announce a message to screen readers.
        
        Args:
            message: Message to announce
            priority: If True, interrupt current announcements
        """
        if not self.config.screen_reader_enabled:
            return
        
        if priority:
            self._announcement_queue.insert(0, message)
        else:
            self._announcement_queue.append(message)
        
        # Process queue
        self._process_announcements()
    
    def _process_announcements(self) -> None:
        """Process the announcement queue."""
        if not self._announcement_queue:
            return
        
        message = self._announcement_queue.pop(0)
        
        # Use Qt's accessibility system if available
        if QACCESSIBLE_AVAILABLE:
            try:
                # Create a temporary widget for the announcement
                app = QApplication.instance()
                if app:
                    # Find active window
                    active = app.activeWindow()
                    if active:
                        # Send accessibility event
                        QAccessible.updateAccessibility(
                            QAccessible.queryAccessibleInterface(active).childAt(0, 0)
                            if QAccessible.queryAccessibleInterface(active) else None,
                            0,
                            QAccessible.Event.Alert
                        )
            except Exception as e:
                logger.debug(f"Accessibility announcement fallback: {e}")
        
        logger.info(f"Screen reader announcement: {message}")
    
    def setup_widget(self, widget: QWidget, name: str, description: str = "") -> None:
        """
        Set up accessibility for a widget.
        
        Args:
            widget: Widget to configure
            name: Accessible name (read by screen reader)
            description: Accessible description (additional context)
        """
        self.set_accessible_name(widget, name)
        if description:
            self.set_accessible_description(widget, description)
        
        # Add role-specific enhancements
        self._enhance_widget(widget)
    
    def _enhance_widget(self, widget: QWidget) -> None:
        """Add role-specific accessibility enhancements."""
        # Add keyboard focus indicator
        if self.config.focus_highlight_enabled:
            self._add_focus_indicator(widget)
        
        # Specific widget enhancements
        if isinstance(widget, QPushButton):
            self._enhance_button(widget)
        elif isinstance(widget, QLineEdit):
            self._enhance_line_edit(widget)
        elif isinstance(widget, QComboBox):
            self._enhance_combo_box(widget)
        elif isinstance(widget, QSlider):
            self._enhance_slider(widget)
    
    def _add_focus_indicator(self, widget: QWidget) -> None:
        """Add visible focus indicator."""
        # Override focus style
        current_style = widget.styleSheet()
        focus_style = f"""
            :focus {{
                outline: {self.config.focus_ring_width}px solid #89b4fa;
                outline-offset: 2px;
            }}
        """
        widget.setStyleSheet(current_style + focus_style)
    
    def _enhance_button(self, button: QPushButton) -> None:
        """Enhance button accessibility."""
        if not button.accessibleName():
            button.setAccessibleName(button.text() or "Button")
        
        # Ensure button is keyboard focusable
        button.setFocusPolicy(Qt.StrongFocus)
    
    def _enhance_line_edit(self, edit: QLineEdit) -> None:
        """Enhance line edit accessibility."""
        if not edit.accessibleName():
            # Try to find associated label
            placeholder = edit.placeholderText()
            if placeholder:
                edit.setAccessibleName(placeholder)
            else:
                edit.setAccessibleName("Text input")
    
    def _enhance_combo_box(self, combo: QComboBox) -> None:
        """Enhance combo box accessibility."""
        if not combo.accessibleName():
            combo.setAccessibleName("Dropdown")
        
        # Announce changes
        combo.currentTextChanged.connect(
            lambda text: self.announce(f"Selected: {text}")
        )
    
    def _enhance_slider(self, slider: QSlider) -> None:
        """Enhance slider accessibility."""
        if not slider.accessibleName():
            slider.setAccessibleName("Slider")
        
        # Announce value changes
        slider.valueChanged.connect(
            lambda value: self.announce(f"Value: {value}")
        )
    
    def get_animation_duration(self) -> int:
        """Get animation duration based on reduced motion setting."""
        if self.config.reduced_motion_enabled:
            return 0
        return self.config.animation_duration_ms
    
    def should_animate(self) -> bool:
        """Check if animations should be played."""
        return not self.config.reduced_motion_enabled
    
    def mark_live_region(self, widget: QWidget, level: str = "polite") -> None:
        """
        Mark a widget as a live region for dynamic content.
        
        Args:
            widget: Widget to mark
            level: "polite" (wait for pause) or "assertive" (interrupt)
        """
        widget.setProperty("aria-live", level)
        self._live_region_widgets[id(widget)] = widget
    
    def update_live_region(self, widget: QWidget, message: str) -> None:
        """Announce an update to a live region."""
        level = widget.property("aria-live") or "polite"
        priority = level == "assertive"
        self.announce(message, priority=priority)


def setup_accessibility(main_window: QMainWindow) -> None:
    """
    Set up accessibility features for the main window.
    
    Args:
        main_window: The main application window
    """
    manager = AccessibilityManager()
    
    # Set up main window
    manager.setup_widget(
        main_window,
        "Enigma AI Engine Main Window",
        "AI assistant application with chat, image generation, and more"
    )
    
    # Find and set up child widgets
    _setup_child_widgets(main_window, manager)
    
    logger.info("Accessibility setup complete")


def _setup_child_widgets(parent: QWidget, manager: AccessibilityManager) -> None:
    """Recursively set up accessibility for child widgets."""
    for child in parent.findChildren(QWidget):
        # Skip already configured widgets
        if child.accessibleName():
            continue
        
        # Auto-configure based on widget type and properties
        widget_type = child.__class__.__name__
        object_name = child.objectName()
        
        if object_name:
            # Use object name as base for accessible name
            readable_name = object_name.replace("_", " ").replace("-", " ").title()
            manager.setup_widget(child, readable_name)
        elif isinstance(child, QPushButton) and child.text():
            manager.setup_widget(child, child.text())
        elif isinstance(child, QLabel) and child.text():
            manager.setup_widget(child, child.text())
        elif isinstance(child, QGroupBox) and child.title():
            manager.setup_widget(child, f"{child.title()} group")


def get_accessibility_manager() -> AccessibilityManager:
    """Get the global accessibility manager instance."""
    return AccessibilityManager()


# Convenience decorators
def accessible(name: str, description: str = ""):
    """Decorator to add accessibility info to a widget class."""
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            manager = AccessibilityManager()
            manager.setup_widget(self, name, description)
        
        cls.__init__ = new_init
        return cls
    return decorator
