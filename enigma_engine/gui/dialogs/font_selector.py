"""
Font Selection Dialog

User-configurable font family and size selection for the GUI.
Integrates with UISettings for persistent storage.

FILE: enigma_engine/gui/dialogs/font_selector.py
TYPE: GUI Dialog for font customization
MAIN CLASSES: FontSelector, FontPreview, FontSettings
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFontComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


# Font categories for organization
FONT_CATEGORIES = {
    "Sans-Serif": [
        "Segoe UI", "Arial", "Helvetica", "Verdana", "Tahoma",
        "Open Sans", "Roboto", "Lato", "Source Sans Pro", "Noto Sans"
    ],
    "Serif": [
        "Times New Roman", "Georgia", "Palatino", "Book Antiqua",
        "Cambria", "Constantia", "Garamond"
    ],
    "Monospace": [
        "Consolas", "Courier New", "Lucida Console", "Monaco",
        "Source Code Pro", "Fira Code", "JetBrains Mono", "Cascadia Code"
    ],
    "Display": [
        "Impact", "Comic Sans MS", "Trebuchet MS", "Century Gothic"
    ]
}


@dataclass
class FontSettings:
    """Font configuration settings."""
    family: str = "Segoe UI"
    size: int = 14
    code_family: str = "Consolas"
    code_size: int = 13
    line_height: float = 1.4
    letter_spacing: float = 0.0
    bold_headings: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "family": self.family,
            "size": self.size,
            "code_family": self.code_family,
            "code_size": self.code_size,
            "line_height": self.line_height,
            "letter_spacing": self.letter_spacing,
            "bold_headings": self.bold_headings
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FontSettings':
        """Create from dictionary."""
        return cls(
            family=data.get("family", "Segoe UI"),
            size=data.get("size", 14),
            code_family=data.get("code_family", "Consolas"),
            code_size=data.get("code_size", 13),
            line_height=data.get("line_height", 1.4),
            letter_spacing=data.get("letter_spacing", 0.0),
            bold_headings=data.get("bold_headings", True)
        )


class FontPreview(QFrame):
    """Live preview of font settings."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup preview area."""
        self.setFrameStyle(QFrame.StyledPanel)
        self.setMinimumHeight(200)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Sample heading
        self._heading = QLabel("The Quick Brown Fox")
        self._heading.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._heading)
        
        # Sample body text
        self._body = QLabel(
            "jumps over the lazy dog. Lorem ipsum dolor sit amet, "
            "consectetur adipiscing elit. Sed do eiusmod tempor "
            "incididunt ut labore et dolore magna aliqua."
        )
        self._body.setWordWrap(True)
        layout.addWidget(self._body)
        
        # Sample code
        self._code = QLabel(
            "def hello_world():\n    print('Hello, World!')"
        )
        self._code.setStyleSheet("background: rgba(0,0,0,0.1); padding: 8px; border-radius: 4px;")
        layout.addWidget(self._code)
        
        layout.addStretch()
        
    def update_preview(self, settings: FontSettings):
        """Update preview with new settings."""
        # Main font
        font = QFont(settings.family, settings.size)
        self._body.setFont(font)
        
        # Heading font
        heading_font = QFont(settings.family, int(settings.size * 1.4))
        if settings.bold_headings:
            heading_font.setBold(True)
        self._heading.setFont(heading_font)
        
        # Code font
        code_font = QFont(settings.code_family, settings.code_size)
        self._code.setFont(code_font)
        
        # Update line height via stylesheet
        line_height_pct = int(settings.line_height * 100)
        self._body.setStyleSheet(f"line-height: {line_height_pct}%;")


class FontSelector(QDialog):
    """Font selection dialog with live preview."""
    
    font_changed = pyqtSignal(object)  # Emits FontSettings
    
    def __init__(self, 
                 current_settings: Optional[FontSettings] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._settings = current_settings or FontSettings()
        self._original_settings = FontSettings(
            family=self._settings.family,
            size=self._settings.size,
            code_family=self._settings.code_family,
            code_size=self._settings.code_size,
            line_height=self._settings.line_height,
            letter_spacing=self._settings.letter_spacing,
            bold_headings=self._settings.bold_headings
        )
        self._callbacks: list[Callable[[FontSettings], None]] = []
        self._setup_ui()
        self._update_preview()
        
        # Apply transparency
        try:
            from ..ui_settings import apply_dialog_transparency
            apply_dialog_transparency(self)
        except ImportError:
            pass
        
    def _setup_ui(self):
        """Setup dialog UI."""
        self.setWindowTitle("Font Settings")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        
        layout = QHBoxLayout(self)
        layout.setSpacing(16)
        
        # Left panel: Settings
        settings_panel = QWidget()
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        
        # Main font section
        main_group = QGroupBox("Main Font")
        main_layout = QVBoxLayout(main_group)
        
        # Font family combo
        family_layout = QHBoxLayout()
        family_layout.addWidget(QLabel("Family:"))
        self._family_combo = QFontComboBox()
        self._family_combo.setCurrentFont(QFont(self._settings.family))
        self._family_combo.currentFontChanged.connect(self._on_family_changed)
        family_layout.addWidget(self._family_combo, 1)
        main_layout.addLayout(family_layout)
        
        # Quick category selection
        cat_layout = QHBoxLayout()
        cat_layout.addWidget(QLabel("Category:"))
        self._category_combo = QComboBox()
        self._category_combo.addItems(list(FONT_CATEGORIES.keys()))
        self._category_combo.currentTextChanged.connect(self._on_category_changed)
        cat_layout.addWidget(self._category_combo, 1)
        main_layout.addLayout(cat_layout)
        
        # Size spinner
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self._size_spin = QSpinBox()
        self._size_spin.setRange(8, 32)
        self._size_spin.setValue(self._settings.size)
        self._size_spin.valueChanged.connect(self._on_size_changed)
        size_layout.addWidget(self._size_spin)
        
        self._size_slider = QSlider(Qt.Horizontal)
        self._size_slider.setRange(8, 32)
        self._size_slider.setValue(self._settings.size)
        self._size_slider.valueChanged.connect(self._size_spin.setValue)
        self._size_spin.valueChanged.connect(self._size_slider.setValue)
        size_layout.addWidget(self._size_slider, 1)
        main_layout.addLayout(size_layout)
        
        settings_layout.addWidget(main_group)
        
        # Code font section
        code_group = QGroupBox("Code Font")
        code_layout = QVBoxLayout(code_group)
        
        # Code family
        code_fam_layout = QHBoxLayout()
        code_fam_layout.addWidget(QLabel("Family:"))
        self._code_family_combo = QComboBox()
        # Populate with available monospace fonts
        self._populate_monospace_fonts()
        code_fam_layout.addWidget(self._code_family_combo, 1)
        code_layout.addLayout(code_fam_layout)
        
        # Code size
        code_size_layout = QHBoxLayout()
        code_size_layout.addWidget(QLabel("Size:"))
        self._code_size_spin = QSpinBox()
        self._code_size_spin.setRange(8, 28)
        self._code_size_spin.setValue(self._settings.code_size)
        self._code_size_spin.valueChanged.connect(self._on_code_size_changed)
        code_size_layout.addWidget(self._code_size_spin)
        code_size_layout.addStretch()
        code_layout.addLayout(code_size_layout)
        
        settings_layout.addWidget(code_group)
        
        # Typography section
        typo_group = QGroupBox("Typography")
        typo_layout = QVBoxLayout(typo_group)
        
        # Line height
        lh_layout = QHBoxLayout()
        lh_layout.addWidget(QLabel("Line Height:"))
        self._line_height_slider = QSlider(Qt.Horizontal)
        self._line_height_slider.setRange(100, 200)  # 1.0 to 2.0
        self._line_height_slider.setValue(int(self._settings.line_height * 100))
        self._line_height_slider.valueChanged.connect(self._on_line_height_changed)
        lh_layout.addWidget(self._line_height_slider, 1)
        self._line_height_label = QLabel(f"{self._settings.line_height:.1f}")
        lh_layout.addWidget(self._line_height_label)
        typo_layout.addLayout(lh_layout)
        
        # Bold headings checkbox
        self._bold_headings_check = QCheckBox("Bold Headings")
        self._bold_headings_check.setChecked(self._settings.bold_headings)
        self._bold_headings_check.toggled.connect(self._on_bold_headings_changed)
        typo_layout.addWidget(self._bold_headings_check)
        
        settings_layout.addWidget(typo_group)
        
        # Presets
        preset_group = QGroupBox("Presets")
        preset_layout = QVBoxLayout(preset_group)
        
        presets_row = QHBoxLayout()
        
        default_btn = QPushButton("Default")
        default_btn.clicked.connect(lambda: self._apply_preset("default"))
        presets_row.addWidget(default_btn)
        
        compact_btn = QPushButton("Compact")
        compact_btn.clicked.connect(lambda: self._apply_preset("compact"))
        presets_row.addWidget(compact_btn)
        
        readable_btn = QPushButton("Readable")
        readable_btn.clicked.connect(lambda: self._apply_preset("readable"))
        presets_row.addWidget(readable_btn)
        
        preset_layout.addLayout(presets_row)
        settings_layout.addWidget(preset_group)
        
        settings_layout.addStretch()
        
        layout.addWidget(settings_panel)
        
        # Right panel: Preview
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        
        preview_label = QLabel("Preview")
        preview_label.setStyleSheet("font-weight: bold;")
        preview_layout.addWidget(preview_label)
        
        self._preview = FontPreview()
        preview_layout.addWidget(self._preview, 1)
        
        layout.addWidget(preview_panel, 1)
        
        # Button row
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset)
        button_layout.addWidget(reset_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply)
        apply_btn.setDefault(True)
        button_layout.addWidget(apply_btn)
        
        # Add button layout at bottom
        main_layout = self.layout()
        outer_layout = QVBoxLayout()
        
        content_widget = QWidget()
        content_widget.setLayout(main_layout)
        outer_layout.addWidget(content_widget, 1)
        outer_layout.addLayout(button_layout)
        
        # Reset layout
        QWidget().setLayout(self.layout())  # Clear old
        self.setLayout(outer_layout)
        
    def _populate_monospace_fonts(self):
        """Populate code font combo with monospace fonts."""
        db = QFontDatabase()
        monospace_fonts = []
        
        # Check preferred fonts first
        preferred = FONT_CATEGORIES["Monospace"]
        for font in preferred:
            if font in db.families():
                monospace_fonts.append(font)
        
        # Add other monospace fonts
        for family in db.families():
            if db.isFixedPitch(family) and family not in monospace_fonts:
                monospace_fonts.append(family)
        
        self._code_family_combo.addItems(monospace_fonts[:20])
        
        # Set current
        idx = self._code_family_combo.findText(self._settings.code_family)
        if idx >= 0:
            self._code_family_combo.setCurrentIndex(idx)
        
        self._code_family_combo.currentTextChanged.connect(self._on_code_family_changed)
        
    def _on_family_changed(self, font: QFont):
        """Handle font family change."""
        self._settings.family = font.family()
        self._update_preview()
        
    def _on_category_changed(self, category: str):
        """Handle category change - show fonts from that category."""
        fonts = FONT_CATEGORIES.get(category, [])
        if fonts:
            db = QFontDatabase()
            for font in fonts:
                if font in db.families():
                    self._family_combo.setCurrentFont(QFont(font))
                    break
                    
    def _on_size_changed(self, size: int):
        """Handle size change."""
        self._settings.size = size
        self._update_preview()
        
    def _on_code_family_changed(self, family: str):
        """Handle code font family change."""
        self._settings.code_family = family
        self._update_preview()
        
    def _on_code_size_changed(self, size: int):
        """Handle code size change."""
        self._settings.code_size = size
        self._update_preview()
        
    def _on_line_height_changed(self, value: int):
        """Handle line height change."""
        self._settings.line_height = value / 100.0
        self._line_height_label.setText(f"{self._settings.line_height:.1f}")
        self._update_preview()
        
    def _on_bold_headings_changed(self, checked: bool):
        """Handle bold headings toggle."""
        self._settings.bold_headings = checked
        self._update_preview()
        
    def _apply_preset(self, preset: str):
        """Apply a font preset."""
        presets = {
            "default": FontSettings(),
            "compact": FontSettings(
                family="Segoe UI",
                size=12,
                code_family="Consolas",
                code_size=11,
                line_height=1.2,
                bold_headings=True
            ),
            "readable": FontSettings(
                family="Georgia",
                size=16,
                code_family="Source Code Pro",
                code_size=14,
                line_height=1.6,
                bold_headings=True
            )
        }
        
        if preset in presets:
            self._settings = presets[preset]
            self._update_ui_from_settings()
            self._update_preview()
            
    def _update_ui_from_settings(self):
        """Update UI controls from current settings."""
        self._family_combo.setCurrentFont(QFont(self._settings.family))
        self._size_spin.setValue(self._settings.size)
        self._code_family_combo.setCurrentText(self._settings.code_family)
        self._code_size_spin.setValue(self._settings.code_size)
        self._line_height_slider.setValue(int(self._settings.line_height * 100))
        self._bold_headings_check.setChecked(self._settings.bold_headings)
        
    def _update_preview(self):
        """Update the preview panel."""
        self._preview.update_preview(self._settings)
        
    def _reset(self):
        """Reset to original settings."""
        self._settings = FontSettings(
            family=self._original_settings.family,
            size=self._original_settings.size,
            code_family=self._original_settings.code_family,
            code_size=self._original_settings.code_size,
            line_height=self._original_settings.line_height,
            letter_spacing=self._original_settings.letter_spacing,
            bold_headings=self._original_settings.bold_headings
        )
        self._update_ui_from_settings()
        self._update_preview()
        
    def _apply(self):
        """Apply and close."""
        self.font_changed.emit(self._settings)
        for cb in self._callbacks:
            cb(self._settings)
        self.accept()
        
    def get_settings(self) -> FontSettings:
        """Get current font settings."""
        return self._settings
        
    def on_change(self, callback: Callable[[FontSettings], None]):
        """Register callback for font changes."""
        self._callbacks.append(callback)


def get_font_selector(parent: Optional[QWidget] = None,
                      current_settings: Optional[FontSettings] = None) -> FontSelector:
    """Factory function to create font selector dialog."""
    return FontSelector(current_settings, parent)


# Integration helper for UISettings
def apply_font_to_stylesheet(stylesheet: str, settings: FontSettings) -> str:
    """Apply font settings to a stylesheet.
    
    Args:
        stylesheet: Original stylesheet
        settings: Font settings to apply
        
    Returns:
        Modified stylesheet with font settings
    """
    import re

    # Replace font-family in QWidget
    pattern = r"(QWidget\s*\{[^}]*font-family:)[^;]*(;)"
    replacement = f"\\1 '{settings.family}', Arial, sans-serif\\2"
    stylesheet = re.sub(pattern, replacement, stylesheet, flags=re.MULTILINE)
    
    # Replace font-size in QWidget  
    pattern = r"(QWidget\s*\{[^}]*font-size:)[^;]*(;)"
    replacement = f"\\1 {settings.size}px\\2"
    stylesheet = re.sub(pattern, replacement, stylesheet, flags=re.MULTILINE)
    
    return stylesheet


__all__ = [
    'FontSelector',
    'FontSettings', 
    'FontPreview',
    'get_font_selector',
    'apply_font_to_stylesheet',
    'FONT_CATEGORIES'
]
