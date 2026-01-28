"""
Gaming Theme - Modern gaming aesthetic for ForgeAI.

Dark theme with indigo/cyan accents, smooth transitions, and gaming-style effects.
"""

# Gaming color palette
GAMING_THEME = {
    "background": "#0a0a0f",      # Very dark background
    "surface": "#12121a",          # Slightly lighter surface
    "surface_elevated": "#1a1a24", # Elevated surfaces
    "primary": "#6366f1",          # Indigo accent
    "primary_hover": "#818cf8",    # Lighter indigo
    "secondary": "#22d3ee",        # Cyan accent
    "secondary_hover": "#67e8f9",  # Lighter cyan
    "text": "#e2e8f0",             # Light text
    "text_muted": "#64748b",       # Muted text
    "text_dim": "#475569",         # Dim text
    "success": "#22c55e",          # Green
    "warning": "#f59e0b",          # Orange
    "error": "#ef4444",            # Red
    "border": "#1e293b",           # Dark border
    "border_focus": "#6366f1",     # Focus border (indigo)
    "border_radius": "8px",
    "font_family": "'Inter', 'Segoe UI', 'Roboto', sans-serif",
}

# Gaming effects configuration
GAMING_EFFECTS = {
    "hover_glow": True,
    "smooth_transitions": True,
    "animated_borders": False,  # Can be enabled for more visual effects
    "rgb_accents": False,       # Can be enabled for RGB effects
}

# Generate QSS stylesheet from theme
def get_gaming_stylesheet() -> str:
    """Generate complete QSS stylesheet for gaming theme."""
    theme = GAMING_THEME
    
    return f"""
    /* ===== GAMING THEME ===== */
    
    /* Base Window */
    QMainWindow, QWidget {{
        background-color: {theme['background']};
        color: {theme['text']};
        font-family: {theme['font_family']};
        font-size: 11px;
    }}
    
    /* Elevated surfaces */
    QGroupBox, QFrame {{
        background-color: {theme['surface']};
        border: 1px solid {theme['border']};
        border-radius: {theme['border_radius']};
        padding: 8px;
    }}
    
    QGroupBox::title {{
        color: {theme['primary']};
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }}
    
    /* Text Input Fields */
    QTextEdit, QPlainTextEdit, QLineEdit {{
        background-color: {theme['surface']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        border-radius: {theme['border_radius']};
        padding: 6px;
        selection-background-color: {theme['primary']};
        selection-color: {theme['text']};
    }}
    
    QTextEdit:focus, QPlainTextEdit:focus, QLineEdit:focus {{
        border: 2px solid {theme['border_focus']};
    }}
    
    /* Buttons - Primary */
    QPushButton {{
        background-color: {theme['primary']};
        color: white;
        border: none;
        border-radius: {theme['border_radius']};
        padding: 8px 16px;
        font-weight: 500;
        font-size: 11px;
    }}
    
    QPushButton:hover {{
        background-color: {theme['primary_hover']};
    }}
    
    QPushButton:pressed {{
        background-color: {theme['primary']};
        padding: 9px 15px 7px 17px;
    }}
    
    QPushButton:disabled {{
        background-color: {theme['surface']};
        color: {theme['text_dim']};
    }}
    
    /* Buttons - Secondary (use property) */
    QPushButton[secondary="true"] {{
        background-color: {theme['surface_elevated']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
    }}
    
    QPushButton[secondary="true"]:hover {{
        background-color: {theme['surface']};
        border-color: {theme['primary']};
    }}
    
    /* List Widgets */
    QListWidget {{
        background-color: {theme['surface']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        border-radius: {theme['border_radius']};
        padding: 4px;
        outline: none;
    }}
    
    QListWidget::item {{
        padding: 8px;
        border-radius: 4px;
        color: {theme['text_muted']};
    }}
    
    QListWidget::item:hover {{
        background-color: {theme['surface_elevated']};
        color: {theme['text']};
    }}
    
    QListWidget::item:selected {{
        background-color: {theme['primary']};
        color: white;
    }}
    
    /* Sidebar (special list widget) */
    QListWidget#sidebar {{
        background-color: {theme['background']};
        border: none;
        border-right: 1px solid {theme['border']};
        border-radius: 0;
    }}
    
    QListWidget#sidebar::item {{
        padding: 10px 12px;
        border-radius: 4px;
        margin: 2px 8px;
    }}
    
    QListWidget#sidebar::item:hover {{
        background-color: {theme['surface']};
    }}
    
    QListWidget#sidebar::item:selected {{
        background-color: {theme['primary']};
        color: white;
    }}
    
    /* Combo Boxes */
    QComboBox {{
        background-color: {theme['surface']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        border-radius: {theme['border_radius']};
        padding: 6px 12px;
        min-height: 20px;
    }}
    
    QComboBox:hover {{
        border-color: {theme['primary']};
    }}
    
    QComboBox::drop-down {{
        border: none;
        padding-right: 8px;
    }}
    
    QComboBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 6px solid {theme['text_muted']};
        margin-right: 4px;
    }}
    
    QComboBox QAbstractItemView {{
        background-color: {theme['surface_elevated']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        selection-background-color: {theme['primary']};
        selection-color: white;
        padding: 4px;
    }}
    
    /* Spin Boxes */
    QSpinBox, QDoubleSpinBox {{
        background-color: {theme['surface']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        border-radius: {theme['border_radius']};
        padding: 4px 8px;
        min-height: 20px;
    }}
    
    /* Sliders */
    QSlider::groove:horizontal {{
        background: {theme['surface']};
        height: 6px;
        border-radius: 3px;
        border: 1px solid {theme['border']};
    }}
    
    QSlider::handle:horizontal {{
        background: {theme['primary']};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
        border: 2px solid {theme['primary_hover']};
    }}
    
    QSlider::handle:horizontal:hover {{
        background: {theme['primary_hover']};
    }}
    
    /* Progress Bars */
    QProgressBar {{
        background-color: {theme['surface']};
        border: 1px solid {theme['border']};
        border-radius: {theme['border_radius']};
        text-align: center;
        color: {theme['text']};
        height: 20px;
    }}
    
    QProgressBar::chunk {{
        background-color: {theme['primary']};
        border-radius: 4px;
    }}
    
    /* Tabs */
    QTabWidget::pane {{
        border: 1px solid {theme['border']};
        border-radius: {theme['border_radius']};
        background-color: {theme['surface']};
        top: -1px;
    }}
    
    QTabBar::tab {{
        background-color: {theme['surface']};
        color: {theme['text_muted']};
        padding: 10px 16px;
        border: 1px solid {theme['border']};
        border-bottom: none;
        border-top-left-radius: {theme['border_radius']};
        border-top-right-radius: {theme['border_radius']};
        margin-right: 2px;
    }}
    
    QTabBar::tab:hover {{
        color: {theme['text']};
        background-color: {theme['surface_elevated']};
    }}
    
    QTabBar::tab:selected {{
        background-color: {theme['primary']};
        color: white;
    }}
    
    /* Scroll Bars */
    QScrollBar:vertical {{
        background: {theme['background']};
        width: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:vertical {{
        background: {theme['surface']};
        border-radius: 6px;
        min-height: 30px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background: {theme['surface_elevated']};
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    
    QScrollBar:horizontal {{
        background: {theme['background']};
        height: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:horizontal {{
        background: {theme['surface']};
        border-radius: 6px;
        min-width: 30px;
    }}
    
    QScrollBar::handle:horizontal:hover {{
        background: {theme['surface_elevated']};
    }}
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}
    
    /* Menu Bar */
    QMenuBar {{
        background-color: {theme['background']};
        color: {theme['text']};
        border-bottom: 1px solid {theme['border']};
        padding: 4px;
    }}
    
    QMenuBar::item {{
        padding: 6px 12px;
        border-radius: 4px;
    }}
    
    QMenuBar::item:selected {{
        background-color: {theme['surface']};
    }}
    
    QMenuBar::item:pressed {{
        background-color: {theme['primary']};
        color: white;
    }}
    
    /* Menus */
    QMenu {{
        background-color: {theme['surface_elevated']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        border-radius: {theme['border_radius']};
        padding: 4px;
    }}
    
    QMenu::item {{
        padding: 8px 24px 8px 12px;
        border-radius: 4px;
    }}
    
    QMenu::item:selected {{
        background-color: {theme['primary']};
        color: white;
    }}
    
    QMenu::separator {{
        height: 1px;
        background: {theme['border']};
        margin: 4px 8px;
    }}
    
    /* Status Bar */
    QStatusBar {{
        background-color: {theme['surface']};
        color: {theme['text_muted']};
        border-top: 1px solid {theme['border']};
    }}
    
    QStatusBar::item {{
        border: none;
    }}
    
    /* Tool Tips */
    QToolTip {{
        background-color: {theme['surface_elevated']};
        color: {theme['text']};
        border: 1px solid {theme['primary']};
        border-radius: 4px;
        padding: 4px 8px;
    }}
    
    /* Labels */
    QLabel {{
        color: {theme['text']};
        background: transparent;
    }}
    
    QLabel#header {{
        color: {theme['primary']};
        font-size: 14px;
        font-weight: bold;
    }}
    
    QLabel#muted {{
        color: {theme['text_muted']};
    }}
    
    /* Check Boxes */
    QCheckBox {{
        color: {theme['text']};
        spacing: 8px;
    }}
    
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid {theme['border']};
        border-radius: 4px;
        background-color: {theme['surface']};
    }}
    
    QCheckBox::indicator:hover {{
        border-color: {theme['primary']};
    }}
    
    QCheckBox::indicator:checked {{
        background-color: {theme['primary']};
        border-color: {theme['primary']};
    }}
    
    /* Radio Buttons */
    QRadioButton {{
        color: {theme['text']};
        spacing: 8px;
    }}
    
    QRadioButton::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid {theme['border']};
        border-radius: 9px;
        background-color: {theme['surface']};
    }}
    
    QRadioButton::indicator:hover {{
        border-color: {theme['primary']};
    }}
    
    QRadioButton::indicator:checked {{
        background-color: {theme['primary']};
        border-color: {theme['primary']};
    }}
    
    /* Text Browser (for chat display) */
    QTextBrowser {{
        background-color: {theme['surface']};
        color: {theme['text']};
        border: 1px solid {theme['border']};
        border-radius: {theme['border_radius']};
        padding: 8px;
        selection-background-color: {theme['primary']};
        selection-color: white;
    }}
    
    /* Dialogs */
    QDialog {{
        background-color: {theme['background']};
        color: {theme['text']};
    }}
    
    /* Stacked Widget */
    QStackedWidget {{
        background-color: {theme['background']};
        border: none;
    }}
    """


# Quick access to stylesheet
GAMING_STYLESHEET = get_gaming_stylesheet()
