"""
================================================================================
CREATE TAB - Consolidated Creation Interface
================================================================================

This tab consolidates all content creation features into a single unified
interface with sub-tabs for different content types.

FILE: forge_ai/gui/tabs/create_tab.py
TYPE: GUI Tab - Consolidated Creation Interface
MAIN CLASS: CreateTab

Sub-tabs:
    - Image Generation (Stable Diffusion, DALL-E, etc.)
    - Code Generation (Python, JavaScript, etc.)
    - Video Generation (AnimateDiff, Replicate, etc.)
    - Audio/TTS Generation (local, ElevenLabs, etc.)

This provides a cleaner interface for Standard mode users while keeping
all generation features accessible in one place.
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


def create_create_tab(parent):
    """
    Create the consolidated Create tab with sub-tabs for different generation types.
    
    This tab combines:
    - Image generation
    - Code generation  
    - Video generation
    - Audio/TTS generation
    
    Args:
        parent: Parent window (EnhancedMainWindow)
    
    Returns:
        QWidget: The create tab widget
    """
    if not HAS_PYQT:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("PyQt5 not available"))
        return widget
    
    # Import the individual tab creators
    try:
        from .image_tab import create_image_tab
        from .code_tab import create_code_tab
        from .video_tab import create_video_tab
        from .audio_tab import create_audio_tab
    except ImportError as e:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(f"Could not import generation tabs: {e}"))
        return widget
    
    # Main widget
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(12)
    
    # Header
    header = QLabel("Create Content")
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
    desc = QLabel("Generate images, code, videos, and audio using AI.")
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
        # Image generation
        image_widget = create_image_tab(parent)
        tab_widget.addTab(image_widget, "Images")
        
        # Code generation
        code_widget = create_code_tab(parent)
        tab_widget.addTab(code_widget, "Code")
        
        # Video generation
        video_widget = create_video_tab(parent)
        tab_widget.addTab(video_widget, "Video")
        
        # Audio generation
        audio_widget = create_audio_tab(parent)
        tab_widget.addTab(audio_widget, "Audio")
        
    except Exception as e:
        # If any tab fails to load, show error message
        error_widget = QWidget()
        error_layout = QVBoxLayout(error_widget)
        error_layout.addWidget(QLabel(f"Error loading generation tabs: {e}"))
        tab_widget.addTab(error_widget, "Error")
    
    layout.addWidget(tab_widget)
    
    # Quick tips at bottom
    tips_label = QLabel(
        "💡 Tip: Use the quick actions bar above for common tasks like screenshot or voice input."
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


class CreateTab(QWidget):
    """
    Consolidated Create Tab - combines all generation features.
    
    This is an alternative to the function-based approach for more complex needs.
    Currently the function approach (create_create_tab) is used.
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
        layout.addWidget(QLabel("Use create_create_tab() function for now"))
