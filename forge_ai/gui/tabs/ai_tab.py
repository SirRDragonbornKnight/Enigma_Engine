"""
================================================================================
AI TAB - Consolidated AI Configuration Interface
================================================================================

This tab consolidates all AI configuration features into a single unified
interface with sub-tabs for different aspects of AI management.

FILE: forge_ai/gui/tabs/ai_tab.py
TYPE: GUI Tab - Consolidated AI Configuration
MAIN CLASS: AITab

Sub-tabs:
    - Avatar (AI avatar display and control)
    - Modules (Enable/disable AI features)
    - Scaling (Grow or shrink model size)
    - Training (Train the model)

This provides a cleaner interface for Standard mode users while keeping
all AI configuration features accessible in one place.
"""

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QTabWidget, QPushButton, QTextEdit
    )
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


def create_ai_tab(parent):
    """
    Create the consolidated AI tab with sub-tabs for different AI configuration aspects.
    
    This tab combines:
    - Avatar control and display
    - Module management
    - Model scaling
    - Training interface
    
    Args:
        parent: Parent window (EnhancedMainWindow)
    
    Returns:
        QWidget: The AI tab widget
    """
    if not HAS_PYQT:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("PyQt5 not available"))
        return widget
    
    # Import the individual tab creators/classes
    try:
        from .avatar_tab import create_avatar_subtab
        from .modules_tab import ModulesTab
        from .scaling_tab import ScalingTab
        from .training_tab import create_training_tab
    except ImportError as e:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(f"Could not import AI tabs: {e}"))
        return widget
    
    # Main widget
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(12)
    
    # Header
    header = QLabel("AI Configuration")
    header.setObjectName("header")
    header.setStyleSheet("""
        QLabel#header {
            font-size: 12px;
            font-weight: bold;
            color: #6366f1;
            padding: 8px 0;
        }
    """)
    layout.addWidget(header)
    
    # Description
    desc = QLabel("Configure your AI's appearance, capabilities, and learning.")
    desc.setStyleSheet("color: #64748b; font-size: 11px; padding-bottom: 8px;")
    layout.addWidget(desc)
    
    # Create tab widget for sub-tabs
    tab_widget = QTabWidget()
    tab_widget.setStyleSheet("""
        QTabWidget::pane {
            border: 1px solid #1e293b;
            border-radius: 8px;
            background-color: #12121a;
        }
        QTabBar::tab {
            background-color: #12121a;
            color: #64748b;
            padding: 10px 16px;
            border: 1px solid #1e293b;
            border-bottom: none;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            margin-right: 2px;
            font-size: 11px;
        }
        QTabBar::tab:hover {
            color: #e2e8f0;
            background-color: #1a1a24;
        }
        QTabBar::tab:selected {
            background-color: #6366f1;
            color: white;
        }
    """)
    
    # Add sub-tabs
    try:
        # Avatar
        avatar_widget = create_avatar_subtab(parent)
        tab_widget.addTab(avatar_widget, "Avatar")
        
        # Modules
        modules_widget = ModulesTab(parent, module_manager=parent.module_manager if hasattr(parent, 'module_manager') else None)
        tab_widget.addTab(modules_widget, "Modules")
        
        # Scaling
        scaling_widget = ScalingTab(parent)
        tab_widget.addTab(scaling_widget, "Scaling")
        
        # Training (if available)
        try:
            training_widget = create_training_tab(parent)
            tab_widget.addTab(training_widget, "Training")
        except:
            pass  # Training tab might not be available in all configurations
        
    except Exception as e:
        # If any tab fails to load, show error message
        error_widget = QWidget()
        error_layout = QVBoxLayout(error_widget)
        error_layout.addWidget(QLabel(f"Error loading AI tabs: {e}"))
        tab_widget.addTab(error_widget, "Error")
    
    layout.addWidget(tab_widget)
    
    # Quick tips at bottom
    tips_label = QLabel(
        "💡 Tip: Modules control which AI features are active. Enable only what you need to save resources."
    )
    tips_label.setStyleSheet("""
        QLabel {
            color: #64748b;
            font-size: 10px;
            padding: 8px;
            background-color: #1a1a24;
            border-radius: 6px;
        }
    """)
    tips_label.setWordWrap(True)
    layout.addWidget(tips_label)
    
    return widget


class AITab(QWidget):
    """
    Consolidated AI Tab - combines all AI configuration features.
    
    This is an alternative to the function-based approach for more complex needs.
    Currently the function approach (create_ai_tab) is used.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # For now, just use the function-based approach
        # This class can be expanded later if needed
        layout.addWidget(QLabel("Use create_ai_tab() function for now"))
