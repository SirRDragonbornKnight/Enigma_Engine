"""
Theme Editor Dialog for ForgeAI.

Allows users to create and edit custom themes with a visual editor.
"""
import logging
from typing import Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..theme_system import Theme, ThemeColors, ThemeManager

logger = logging.getLogger(__name__)


class ColorButton(QPushButton):
    """A button that shows a color and opens a color picker when clicked."""
    
    color_changed = pyqtSignal(str)  # Emits hex color
    
    def __init__(self, color: str = "#ffffff", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(60, 30)
        self.clicked.connect(self._pick_color)
        self._update_style()
    
    def _update_style(self):
        """Update button style to show current color."""
        # Calculate contrasting text color
        r, g, b = int(self._color[1:3], 16), int(self._color[3:5], 16), int(self._color[5:7], 16)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        text_color = "#000000" if brightness > 128 else "#ffffff"
        
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._color};
                color: {text_color};
                border: 2px solid #45475a;
                border-radius: 4px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                border-color: #89b4fa;
            }}
        """)
        self.setText(self._color)
    
    def _pick_color(self):
        """Open color picker dialog."""
        current = QColor(self._color)
        color = QColorDialog.getColor(current, self, "Pick Color")
        
        if color.isValid():
            self._color = color.name()
            self._update_style()
            self.color_changed.emit(self._color)
    
    @property
    def color(self) -> str:
        return self._color
    
    @color.setter
    def color(self, value: str):
        self._color = value
        self._update_style()


class ThemeEditorDialog(QDialog):
    """
    Visual theme editor for creating and modifying custom themes.
    
    Usage:
        editor = ThemeEditorDialog(theme_manager, parent)
        if editor.exec_() == QDialog.Accepted:
            # Theme was saved
            pass
    """
    
    theme_saved = pyqtSignal(str)  # Emits theme name when saved
    
    def __init__(
        self,
        theme_manager: ThemeManager,
        parent: Optional[QWidget] = None,
        edit_theme: Optional[Theme] = None
    ):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.edit_theme = edit_theme
        self.color_buttons: dict[str, ColorButton] = {}
        
        self._setup_ui()
        
        # Apply transparency
        try:
            from ..ui_settings import apply_dialog_transparency
            apply_dialog_transparency(self)
        except ImportError:
            pass
        
        if edit_theme:
            self._load_theme(edit_theme)
    
    def _setup_ui(self):
        """Set up the editor UI."""
        self.setWindowTitle("Theme Editor")
        self.setMinimumSize(700, 600)
        
        layout = QVBoxLayout(self)
        
        # Theme info section
        info_group = QGroupBox("Theme Information")
        info_layout = QFormLayout()
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter theme name...")
        info_layout.addRow("Name:", self.name_edit)
        
        self.desc_edit = QLineEdit()
        self.desc_edit.setPlaceholderText("Brief description of the theme...")
        info_layout.addRow("Description:", self.desc_edit)
        
        self.base_combo = QComboBox()
        self.base_combo.addItems(["dark", "light", "midnight", "forest", "sunset", "cerulean"])
        self.base_combo.currentTextChanged.connect(self._apply_base_theme)
        info_layout.addRow("Base Theme:", self.base_combo)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Color editor tabs
        tabs = QTabWidget()
        
        # Background colors tab
        bg_tab = self._create_color_tab([
            ("bg_primary", "Primary Background", "Main window background"),
            ("bg_secondary", "Secondary Background", "Cards, panels, inputs"),
            ("bg_tertiary", "Tertiary Background", "Highlights, active states"),
        ])
        tabs.addTab(bg_tab, "Backgrounds")
        
        # Text colors tab
        text_tab = self._create_color_tab([
            ("text_primary", "Primary Text", "Main text color"),
            ("text_secondary", "Secondary Text", "Less important text"),
            ("text_disabled", "Disabled Text", "Inactive elements"),
        ])
        tabs.addTab(text_tab, "Text")
        
        # Accent colors tab
        accent_tab = self._create_color_tab([
            ("accent_primary", "Primary Accent", "Buttons, links, focus"),
            ("accent_secondary", "Secondary Accent", "Hover states"),
            ("accent_hover", "Hover Accent", "Active/pressed states"),
        ])
        tabs.addTab(accent_tab, "Accents")
        
        # Semantic colors tab
        semantic_tab = self._create_color_tab([
            ("success", "Success", "Success messages, positive actions"),
            ("warning", "Warning", "Warnings, caution states"),
            ("error", "Error", "Errors, destructive actions"),
            ("info", "Info", "Information, tips"),
        ])
        tabs.addTab(semantic_tab, "Semantic")
        
        # Border colors tab
        border_tab = self._create_color_tab([
            ("border_primary", "Primary Border", "Main borders"),
            ("border_secondary", "Secondary Border", "Subtle borders"),
        ])
        tabs.addTab(border_tab, "Borders")
        
        layout.addWidget(tabs)
        
        # Preview section
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_widget = QWidget()
        self.preview_widget.setMinimumHeight(120)
        preview_layout.addWidget(self.preview_widget)
        
        preview_btn = QPushButton("Update Preview")
        preview_btn.clicked.connect(self._update_preview)
        preview_layout.addWidget(preview_btn)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export CSS")
        export_btn.clicked.connect(self._export_css)
        btn_layout.addWidget(export_btn)
        
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("Save Theme")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save_theme)
        btn_layout.addWidget(save_btn)
        
        layout.addLayout(btn_layout)
        
        # Apply current theme styling
        self._apply_style()
    
    def _create_color_tab(self, colors: list) -> QWidget:
        """Create a tab with color editors."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setSpacing(12)
        
        for row, (key, label, description) in enumerate(colors):
            # Label
            label_widget = QLabel(label)
            label_widget.setToolTip(description)
            layout.addWidget(label_widget, row, 0)
            
            # Color button
            btn = ColorButton("#ffffff")
            btn.color_changed.connect(lambda c, k=key: self._on_color_changed(k, c))
            layout.addWidget(btn, row, 1)
            
            # Description
            desc_label = QLabel(description)
            desc_label.setStyleSheet("color: #888; font-size: 10px;")
            layout.addWidget(desc_label, row, 2)
            
            self.color_buttons[key] = btn
        
        layout.setRowStretch(len(colors), 1)
        return widget
    
    def _apply_style(self):
        """Apply current theme to the dialog."""
        if self.theme_manager.current_theme:
            self.setStyleSheet(self.theme_manager.get_current_stylesheet())
    
    def _apply_base_theme(self, theme_name: str):
        """Apply a base theme's colors to the editor."""
        theme = self.theme_manager.get_theme(theme_name)
        if theme:
            self._load_theme(theme)
    
    def _load_theme(self, theme: Theme):
        """Load a theme into the editor."""
        self.name_edit.setText(theme.name)
        self.desc_edit.setText(theme.description)
        
        colors = theme.colors.to_dict()
        for key, btn in self.color_buttons.items():
            if key in colors:
                btn.color = colors[key]
        
        self._update_preview()
    
    def _on_color_changed(self, key: str, color: str):
        """Handle color change."""
        # Auto-update preview after short delay
        self._update_preview()
    
    def _get_current_colors(self) -> ThemeColors:
        """Get current colors from the editor."""
        colors_dict = {key: btn.color for key, btn in self.color_buttons.items()}
        return ThemeColors(**colors_dict)
    
    def _update_preview(self):
        """Update the preview widget."""
        colors = self._get_current_colors()
        theme = Theme("Preview", colors)
        
        # Create a mini preview layout
        preview_style = f"""
            QWidget {{
                background-color: {colors.bg_primary};
                border-radius: 8px;
            }}
        """
        self.preview_widget.setStyleSheet(preview_style)
        
        # Clear existing preview content
        old_layout = self.preview_widget.layout()
        if old_layout:
            while old_layout.count():
                item = old_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        else:
            old_layout = QVBoxLayout(self.preview_widget)
        
        # Add preview elements
        preview_layout = self.preview_widget.layout()
        
        # Sample text
        text_label = QLabel("Sample text in primary color")
        text_label.setStyleSheet(f"color: {colors.text_primary}; font-size: 12px;")
        preview_layout.addWidget(text_label)
        
        # Sample button
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        
        primary_btn = QPushButton("Primary Button")
        primary_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors.accent_primary};
                color: {colors.bg_primary};
                padding: 6px 12px;
                border-radius: 4px;
                border: none;
            }}
        """)
        btn_layout.addWidget(primary_btn)
        
        secondary_btn = QPushButton("Secondary")
        secondary_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors.bg_secondary};
                color: {colors.text_primary};
                padding: 6px 12px;
                border-radius: 4px;
                border: 1px solid {colors.border_primary};
            }}
        """)
        btn_layout.addWidget(secondary_btn)
        btn_layout.addStretch()
        preview_layout.addWidget(btn_widget)
        
        # Status labels
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        
        for color, label in [
            (colors.success, "Success"),
            (colors.warning, "Warning"),
            (colors.error, "Error"),
            (colors.info, "Info")
        ]:
            status_label = QLabel(label)
            status_label.setStyleSheet(f"""
                background-color: {color};
                color: #000;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 10px;
            """)
            status_layout.addWidget(status_label)
        status_layout.addStretch()
        preview_layout.addWidget(status_widget)
    
    def _export_css(self):
        """Export the theme as CSS."""
        colors = self._get_current_colors()
        theme = Theme(
            self.name_edit.text() or "Custom",
            colors,
            self.desc_edit.text()
        )
        
        css = theme.generate_stylesheet()
        
        # Show in a dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Theme CSS")
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        text_edit = QTextEdit()
        text_edit.setPlainText(css)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def _save_theme(self):
        """Save the current theme."""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter a theme name.")
            return
        
        # Check for reserved names
        if name.lower() in ["dark", "light", "midnight", "forest", "sunset", "cerulean"]:
            QMessageBox.warning(
                self,
                "Error",
                "Cannot overwrite built-in themes. Please choose a different name."
            )
            return
        
        try:
            colors = self._get_current_colors()
            self.theme_manager.create_custom_theme(
                name,
                colors,
                self.desc_edit.text()
            )
            
            self.theme_saved.emit(name)
            QMessageBox.information(self, "Success", f"Theme '{name}' saved successfully!")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save theme: {e}")


def show_theme_editor(parent: Optional[QWidget] = None) -> Optional[str]:
    """
    Show the theme editor dialog.
    
    Args:
        parent: Parent widget
        
    Returns:
        Name of saved theme, or None if cancelled
    """
    from ..theme_system import ThemeManager
    
    manager = ThemeManager()
    dialog = ThemeEditorDialog(manager, parent)
    
    saved_theme = None
    
    def on_saved(name):
        nonlocal saved_theme
        saved_theme = name
    
    dialog.theme_saved.connect(on_saved)
    dialog.exec_()
    
    return saved_theme
