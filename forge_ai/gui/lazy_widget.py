"""
Lazy Loading Widgets for GUI

Provides widgets that defer heavy initialization until first shown.
Use for tabs that take a long time to initialize but aren't used immediately.

Usage:
    from forge_ai.gui.lazy_widget import LazyWidget
    
    # Instead of:
    # tab = create_heavy_tab(parent)
    
    # Use:
    tab = LazyWidget(parent, lambda p: create_heavy_tab(p))
    
    # The create_heavy_tab function won't be called until the widget
    # is first shown (e.g., when user clicks on the tab).
"""

import logging
from typing import Callable, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class LazyWidget(QWidget):
    """
    A widget container that lazily loads its content.
    
    The actual content widget is not created until this widget
    is first shown, reducing startup time.
    
    Attributes:
        is_loaded: Whether the content has been loaded
        loading_text: Text to show while loading
    """
    
    def __init__(
        self,
        parent: QWidget,
        factory: Callable[[QWidget], QWidget],
        loading_text: str = "Loading...",
        preload_delay_ms: int = 0
    ):
        """
        Initialize lazy widget.
        
        Args:
            parent: Parent widget
            factory: Function that creates the actual content widget
            loading_text: Text to show during loading
            preload_delay_ms: If > 0, start loading after this delay (background)
        """
        super().__init__(parent)
        
        self._factory = factory
        self._parent = parent
        self._content: Optional[QWidget] = None
        self._is_loaded = False
        self._loading_text = loading_text
        
        # Setup layout
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        
        # Create loading placeholder
        self._placeholder = self._create_placeholder()
        self._layout.addWidget(self._placeholder)
        
        # Optional preload after delay
        if preload_delay_ms > 0:
            QTimer.singleShot(preload_delay_ms, self._preload)
    
    @property
    def is_loaded(self) -> bool:
        """Whether the content has been loaded."""
        return self._is_loaded
    
    def _create_placeholder(self) -> QWidget:
        """Create the loading placeholder widget."""
        placeholder = QWidget()
        layout = QVBoxLayout(placeholder)
        layout.setAlignment(Qt.AlignCenter)
        
        # Loading label
        label = QLabel(self._loading_text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("""
            QLabel {
                color: #6c7086;
                font-size: 14px;
            }
        """)
        layout.addWidget(label)
        
        # Progress bar (indeterminate)
        progress = QProgressBar()
        progress.setMaximum(0)  # Indeterminate
        progress.setMinimum(0)
        progress.setFixedWidth(200)
        progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #45475a;
                border-radius: 4px;
                background: #1e1e2e;
                height: 8px;
            }
            QProgressBar::chunk {
                background: #89b4fa;
                border-radius: 3px;
            }
        """)
        layout.addWidget(progress, alignment=Qt.AlignCenter)
        
        return placeholder
    
    def showEvent(self, event):
        """Called when widget is shown - trigger loading."""
        super().showEvent(event)
        if not self._is_loaded:
            # Use timer to allow UI to update before loading
            QTimer.singleShot(0, self._load_content)
    
    def _preload(self):
        """Preload content in background (after startup)."""
        if not self._is_loaded and not self.isVisible():
            self._load_content()
    
    def _load_content(self):
        """Load the actual content widget."""
        if self._is_loaded:
            return
        
        logger.debug(f"Loading lazy widget content: {self._loading_text}")
        
        try:
            # Create the content
            self._content = self._factory(self._parent)
            
            # Replace placeholder with content
            self._layout.removeWidget(self._placeholder)
            self._placeholder.deleteLater()
            self._placeholder = None
            
            self._layout.addWidget(self._content)
            self._is_loaded = True
            
            logger.debug(f"Lazy widget loaded: {self._loading_text}")
            
        except Exception as e:
            logger.error(f"Error loading lazy widget: {e}")
            # Show error in placeholder
            if self._placeholder:
                for child in self._placeholder.findChildren(QLabel):
                    child.setText(f"Error loading: {e}")
    
    def get_content(self) -> Optional[QWidget]:
        """Get the content widget (may be None if not loaded)."""
        return self._content
    
    def force_load(self):
        """Force immediate loading of content."""
        if not self._is_loaded:
            self._load_content()


class LazyTabManager:
    """
    Manages lazy loading of tabs in a QStackedWidget.
    
    Tracks which tabs have been loaded and handles loading
    when tabs are first accessed.
    """
    
    def __init__(self, stack_widget: QStackedWidget):
        """
        Initialize lazy tab manager.
        
        Args:
            stack_widget: The QStackedWidget containing the tabs
        """
        self._stack = stack_widget
        self._tab_factories: dict = {}  # index -> factory
        self._loaded_tabs: set = set()  # indices that are loaded
        
        # Connect to current changed to trigger lazy loading
        self._stack.currentChanged.connect(self._on_tab_changed)
    
    def register_lazy_tab(
        self, 
        index: int, 
        factory: Callable[[QWidget], QWidget],
        name: str = ""
    ):
        """
        Register a tab for lazy loading.
        
        Args:
            index: Tab index in the stack
            factory: Function to create the tab content
            name: Optional name for logging
        """
        self._tab_factories[index] = (factory, name)
        
        # Add placeholder at this index
        placeholder = self._create_placeholder(name or f"Tab {index}")
        self._stack.insertWidget(index, placeholder)
    
    def _create_placeholder(self, name: str) -> QWidget:
        """Create a placeholder widget for unloaded tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)
        
        label = QLabel(f"Click to load {name}...")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #6c7086;")
        layout.addWidget(label)
        
        return widget
    
    def _on_tab_changed(self, index: int):
        """Handle tab change - load if needed."""
        if index in self._tab_factories and index not in self._loaded_tabs:
            self._load_tab(index)
    
    def _load_tab(self, index: int):
        """Load a tab's content."""
        if index not in self._tab_factories:
            return
        
        factory, name = self._tab_factories[index]
        logger.info(f"Loading tab: {name or index}")
        
        # Process events to show loading state
        QApplication.processEvents()
        
        try:
            # Create the actual content
            content = factory(self._stack.parent())
            
            # Replace the placeholder
            old_widget = self._stack.widget(index)
            self._stack.removeWidget(old_widget)
            old_widget.deleteLater()
            
            self._stack.insertWidget(index, content)
            self._stack.setCurrentIndex(index)
            
            self._loaded_tabs.add(index)
            logger.info(f"Tab loaded: {name or index}")
            
        except Exception as e:
            logger.error(f"Error loading tab {name or index}: {e}")
    
    def preload_tabs(self, indices: list, delay_ms: int = 1000):
        """
        Preload specific tabs after a delay.
        
        Args:
            indices: Tab indices to preload
            delay_ms: Delay before starting preload
        """
        def preload():
            for idx in indices:
                if idx not in self._loaded_tabs:
                    self._load_tab(idx)
                    QApplication.processEvents()
        
        QTimer.singleShot(delay_ms, preload)
    
    def is_loaded(self, index: int) -> bool:
        """Check if a tab is loaded."""
        return index in self._loaded_tabs
    
    def get_loaded_count(self) -> int:
        """Get number of loaded tabs."""
        return len(self._loaded_tabs)


# Pre-configured lazy wrappers for common heavy tabs
def create_lazy_image_tab(parent):
    """Create a lazy-loaded image generation tab."""
    return LazyWidget(
        parent,
        lambda p: _import_and_create('forge_ai.gui.tabs.image_tab', 'create_image_tab', p),
        loading_text="Loading Image Generator...",
        preload_delay_ms=3000  # Preload after 3 seconds
    )


def create_lazy_video_tab(parent):
    """Create a lazy-loaded video generation tab."""
    return LazyWidget(
        parent,
        lambda p: _import_and_create('forge_ai.gui.tabs.video_tab', 'create_video_tab', p),
        loading_text="Loading Video Generator..."
    )


def create_lazy_threed_tab(parent):
    """Create a lazy-loaded 3D generation tab."""
    return LazyWidget(
        parent,
        lambda p: _import_and_create('forge_ai.gui.tabs.threed_tab', 'create_threed_tab', p),
        loading_text="Loading 3D Generator..."
    )


def _import_and_create(module_name: str, func_name: str, parent):
    """Import a module and call a factory function."""
    import importlib
    module = importlib.import_module(module_name)
    factory = getattr(module, func_name)
    return factory(parent)
