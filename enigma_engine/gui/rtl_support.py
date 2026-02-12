"""
RTL Language Support for Enigma AI Engine

Right-to-left layout support for Arabic, Hebrew, etc.

Features:
- Text direction detection
- RTL layout mirroring
- Bidirectional text handling
- Font support
- UI adaptation

Usage:
    from enigma_engine.gui.rtl_support import RTLSupport
    
    rtl = RTLSupport()
    
    # Check if text is RTL
    if rtl.is_rtl_text("مرحبا"):
        print("Arabic text detected")
    
    # Apply RTL layout
    rtl.apply_to_widget(widget)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class TextDirection(Enum):
    """Text direction."""
    LTR = "ltr"  # Left-to-right
    RTL = "rtl"  # Right-to-left
    AUTO = "auto"


@dataclass
class RTLConfig:
    """RTL configuration."""
    enabled: bool = True
    auto_detect: bool = True
    default_direction: TextDirection = TextDirection.LTR
    mirror_ui: bool = True


# RTL language codes
RTL_LANGUAGES = {
    'ar',  # Arabic
    'he',  # Hebrew
    'fa',  # Persian/Farsi
    'ur',  # Urdu
    'yi',  # Yiddish
    'ps',  # Pashto
    'sd',  # Sindhi
    'ug',  # Uyghur
}

# Unicode ranges for RTL scripts
RTL_RANGES = [
    (0x0590, 0x05FF),   # Hebrew
    (0x0600, 0x06FF),   # Arabic
    (0x0700, 0x074F),   # Syriac
    (0x0750, 0x077F),   # Arabic Supplement
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB1D, 0xFB4F),   # Hebrew Presentation Forms
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
]


class RTLSupport:
    """RTL language support handler."""
    
    def __init__(self, config: Optional[RTLConfig] = None):
        """
        Initialize RTL support.
        
        Args:
            config: RTL configuration
        """
        self.config = config or RTLConfig()
        
        # Current direction
        self._current_direction = self.config.default_direction
        
        # Registered widgets
        self._widgets: List[Any] = []
    
    def is_rtl_character(self, char: str) -> bool:
        """
        Check if a character is RTL.
        
        Args:
            char: Character to check
            
        Returns:
            True if RTL
        """
        code = ord(char)
        
        for start, end in RTL_RANGES:
            if start <= code <= end:
                return True
        
        return False
    
    def is_rtl_text(self, text: str) -> bool:
        """
        Check if text should be displayed RTL.
        
        Args:
            text: Text to check
            
        Returns:
            True if majority RTL
        """
        if not text:
            return False
        
        rtl_count = 0
        total_count = 0
        
        for char in text:
            if char.isalpha():
                total_count += 1
                if self.is_rtl_character(char):
                    rtl_count += 1
        
        if total_count == 0:
            return False
        
        return rtl_count / total_count > 0.5
    
    def detect_direction(self, text: str) -> TextDirection:
        """
        Detect text direction.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected direction
        """
        if self.is_rtl_text(text):
            return TextDirection.RTL
        return TextDirection.LTR
    
    def detect_language_direction(self, language_code: str) -> TextDirection:
        """
        Get direction for a language code.
        
        Args:
            language_code: ISO language code (e.g., 'ar', 'he')
            
        Returns:
            Text direction
        """
        code = language_code.lower().split('-')[0].split('_')[0]
        
        if code in RTL_LANGUAGES:
            return TextDirection.RTL
        
        return TextDirection.LTR
    
    def set_direction(self, direction: TextDirection):
        """
        Set current text direction.
        
        Args:
            direction: New direction
        """
        self._current_direction = direction
        
        if direction == TextDirection.RTL and self.config.mirror_ui:
            self._apply_rtl_to_widgets()
        else:
            self._apply_ltr_to_widgets()
    
    def get_direction(self) -> TextDirection:
        """Get current direction."""
        return self._current_direction
    
    def apply_to_widget(self, widget: Any):
        """
        Apply RTL support to a Qt widget.
        
        Args:
            widget: PyQt/PySide widget
        """
        self._widgets.append(widget)
        
        if self._current_direction == TextDirection.RTL:
            self._apply_rtl_to_widget(widget)
    
    def _apply_rtl_to_widget(self, widget: Any):
        """Apply RTL layout to widget."""
        try:
            from PyQt5.QtCore import Qt
            from PyQt5.QtWidgets import QWidget
            
            if isinstance(widget, QWidget):
                widget.setLayoutDirection(Qt.RightToLeft)
                
        except ImportError:
            pass  # Intentionally silent
    
    def _apply_ltr_to_widget(self, widget: Any):
        """Apply LTR layout to widget."""
        try:
            from PyQt5.QtCore import Qt
            from PyQt5.QtWidgets import QWidget
            
            if isinstance(widget, QWidget):
                widget.setLayoutDirection(Qt.LeftToRight)
                
        except ImportError:
            pass  # Intentionally silent
    
    def _apply_rtl_to_widgets(self):
        """Apply RTL to all registered widgets."""
        for widget in self._widgets:
            self._apply_rtl_to_widget(widget)
    
    def _apply_ltr_to_widgets(self):
        """Apply LTR to all registered widgets."""
        for widget in self._widgets:
            self._apply_ltr_to_widget(widget)
    
    def wrap_bidi_text(self, text: str) -> str:
        """
        Wrap text with bidirectional control characters.
        
        Args:
            text: Text to wrap
            
        Returns:
            Text with bidi marks
        """
        if self.is_rtl_text(text):
            # Right-to-Left Embedding + Pop Directional Formatting
            return f"\u202B{text}\u202C"
        else:
            # Left-to-Right Embedding + Pop Directional Formatting
            return f"\u202A{text}\u202C"
    
    def get_rtl_css(self) -> str:
        """
        Get CSS for RTL layout.
        
        Returns:
            CSS string
        """
        if self._current_direction == TextDirection.RTL:
            return """
* {
    direction: rtl;
    text-align: right;
}

.ltr {
    direction: ltr;
    text-align: left;
}
"""
        else:
            return """
* {
    direction: ltr;
    text-align: left;
}

.rtl {
    direction: rtl;
    text-align: right;
}
"""
    
    def get_qt_stylesheet(self) -> str:
        """
        Get Qt stylesheet for RTL.
        
        Returns:
            Qt stylesheet
        """
        if self._current_direction == TextDirection.RTL:
            return """
QWidget {
    layoutDirection: RightToLeft;
}

QLineEdit, QTextEdit, QPlainTextEdit {
    alignment: AlignRight;
}

QMenuBar {
    direction: rtl;
}

QToolBar {
    direction: rtl;
}
"""
        return ""


class BidirectionalText:
    """Handle bidirectional text rendering."""
    
    def __init__(self):
        """Initialize bidi handler."""
        self._rtl = RTLSupport()
    
    def render(self, text: str) -> str:
        """
        Render bidirectional text correctly.
        
        Args:
            text: Mixed direction text
            
        Returns:
            Properly formatted text
        """
        # Split into segments by direction
        segments = self._split_by_direction(text)
        
        # Wrap each segment appropriately
        result = []
        for segment, direction in segments:
            if direction == TextDirection.RTL:
                result.append(f"\u202B{segment}\u202C")
            else:
                result.append(segment)
        
        return ''.join(result)
    
    def _split_by_direction(self, text: str) -> List[tuple]:
        """Split text into direction segments."""
        if not text:
            return []
        
        segments = []
        current_segment = ""
        current_direction = None
        
        for char in text:
            char_direction = (
                TextDirection.RTL
                if self._rtl.is_rtl_character(char)
                else TextDirection.LTR
            )
            
            # Non-directional characters (spaces, numbers) keep current direction
            if not char.isalpha():
                current_segment += char
                continue
            
            if current_direction is None:
                current_direction = char_direction
            
            if char_direction == current_direction:
                current_segment += char
            else:
                # Direction change
                if current_segment:
                    segments.append((current_segment, current_direction))
                current_segment = char
                current_direction = char_direction
        
        if current_segment:
            segments.append((current_segment, current_direction or TextDirection.LTR))
        
        return segments


class RTLAwareWidget:
    """Mixin for RTL-aware widgets."""
    
    def __init__(self):
        """Initialize RTL awareness."""
        self._rtl_support = get_rtl_support()
    
    def set_text_rtl(self, text: str):
        """
        Set text with automatic RTL detection.
        
        Args:
            text: Text to set
        """
        direction = self._rtl_support.detect_direction(text)
        
        if direction == TextDirection.RTL:
            self._apply_rtl()
        else:
            self._apply_ltr()
    
    def _apply_rtl(self):
        """Apply RTL layout. Override in subclass."""
    
    def _apply_ltr(self):
        """Apply LTR layout. Override in subclass."""


# Global instance
_rtl_support: Optional[RTLSupport] = None


def get_rtl_support() -> RTLSupport:
    """Get or create global RTL support."""
    global _rtl_support
    if _rtl_support is None:
        _rtl_support = RTLSupport()
    return _rtl_support


def is_rtl(text: str) -> bool:
    """Quick check if text is RTL."""
    return get_rtl_support().is_rtl_text(text)


def detect_direction(text: str) -> str:
    """Detect text direction."""
    return get_rtl_support().detect_direction(text).value
