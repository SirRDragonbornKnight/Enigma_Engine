# type: ignore
"""
Shared GUI Styles - Common CSS/QSS styles used across ForgeAI UI.

This module centralizes styling to:
1. Keep files smaller
2. Ensure consistent look across all windows
3. Make theme changes easier
"""

# =============================================================================
# COLOR PALETTE (Catppuccin Mocha inspired)
# =============================================================================

COLORS = {
    # Base colors
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    
    # Text colors
    "text": "#cdd6f4",
    "subtext0": "#a6adc8",
    "subtext1": "#bac2de",
    "overlay": "#6c7086",
    
    # Accent colors
    "blue": "#89b4fa",
    "green": "#a6e3a1",
    "red": "#f38ba8",
    "yellow": "#f9e2af",
    "peach": "#fab387",
    "teal": "#94e2d5",
    "lavender": "#b4befe",
    "pink": "#f5c2e7",
    "mauve": "#cba6f7",
    "sky": "#89dceb",
    
    # Semantic colors
    "success": "#a6e3a1",
    "error": "#f38ba8",
    "warning": "#f9e2af",
    "info": "#89b4fa",
}


# =============================================================================
# COMMON WIDGET STYLES
# =============================================================================

DARK_WINDOW = f"""
    QWidget {{
        background-color: {COLORS['base']};
        color: {COLORS['text']};
    }}
"""

DARK_DIALOG = f"""
    QDialog {{
        background-color: {COLORS['base']};
        border: 2px solid {COLORS['blue']};
        border-radius: 12px;
    }}
    QLabel {{
        color: {COLORS['text']};
    }}
"""

BUTTON_PRIMARY = f"""
    QPushButton {{
        background-color: {COLORS['blue']};
        color: {COLORS['crust']};
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: {COLORS['sky']};
    }}
    QPushButton:pressed {{
        background-color: {COLORS['lavender']};
    }}
    QPushButton:disabled {{
        background-color: {COLORS['surface0']};
        color: {COLORS['red']};
        border: 2px dashed {COLORS['red']};
    }}
"""

BUTTON_SECONDARY = f"""
    QPushButton {{
        background-color: {COLORS['surface1']};
        color: {COLORS['text']};
        border: none;
        border-radius: 6px;
        padding: 6px 16px;
        font-size: 12px;
    }}
    QPushButton:hover {{
        background-color: {COLORS['red']};
    }}
"""

BUTTON_DANGER = f"""
    QPushButton {{
        background-color: {COLORS['red']};
        color: {COLORS['crust']};
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: #ef4444;
    }}
    QPushButton:disabled {{
        background-color: {COLORS['surface0']};
        color: {COLORS['red']};
        border: 2px dashed {COLORS['red']};
    }}
"""

BUTTON_SUCCESS = f"""
    QPushButton {{
        background-color: {COLORS['green']};
        color: {COLORS['crust']};
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: #7dd87d;
    }}
"""

BUTTON_TOGGLE = f"""
    QPushButton {{
        background-color: {COLORS['surface0']};
        color: {COLORS['subtext0']};
        border: 1px solid {COLORS['surface1']};
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 12px;
    }}
    QPushButton:hover {{
        background-color: {COLORS['surface1']};
        border-color: {COLORS['blue']};
    }}
    QPushButton:checked {{
        background-color: {COLORS['green']};
        color: {COLORS['crust']};
        border-color: {COLORS['green']};
    }}
"""


# =============================================================================
# INPUT STYLES
# =============================================================================

TEXT_INPUT = f"""
    QLineEdit {{
        background-color: {COLORS['surface0']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['surface1']};
        border-radius: 8px;
        padding: 10px 15px;
        font-size: 12px;
    }}
    QLineEdit:focus {{
        border-color: {COLORS['blue']};
    }}
    QLineEdit:disabled {{
        background-color: {COLORS['mantle']};
        color: {COLORS['overlay']};
    }}
"""

TEXT_AREA = f"""
    QTextEdit {{
        background-color: {COLORS['surface0']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['surface1']};
        border-radius: 8px;
        padding: 8px;
        font-size: 12px;
    }}
    QTextEdit:focus {{
        border-color: {COLORS['blue']};
    }}
"""

CHAT_DISPLAY = f"""
    QTextEdit {{
        background-color: {COLORS['mantle']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['surface1']};
        border-radius: 8px;
        padding: 10px;
        font-size: 12px;
    }}
"""

COMBO_BOX = f"""
    QComboBox {{
        background-color: {COLORS['surface0']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['surface1']};
        border-radius: 6px;
        padding: 6px 12px;
        min-width: 120px;
    }}
    QComboBox:hover {{
        border-color: {COLORS['blue']};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {COLORS['surface0']};
        color: {COLORS['text']};
        selection-background-color: {COLORS['blue']};
        selection-color: {COLORS['crust']};
    }}
"""


# =============================================================================
# PROGRESS BARS
# =============================================================================

PROGRESS_BAR = f"""
    QProgressBar {{
        background-color: {COLORS['surface0']};
        border: none;
        border-radius: 8px;
        height: 18px;
        text-align: center;
        color: white;
        font-size: 12px;
        font-weight: bold;
    }}
    QProgressBar::chunk {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {COLORS['blue']}, stop:0.5 {COLORS['sky']}, stop:1 {COLORS['blue']});
        border-radius: 8px;
    }}
"""

PROGRESS_BAR_MINI = f"""
    QProgressBar {{
        background-color: {COLORS['surface1']};
        border-radius: 4px;
        height: 8px;
    }}
    QProgressBar::chunk {{
        background-color: {COLORS['blue']};
        border-radius: 4px;
    }}
"""


# =============================================================================
# LIST WIDGETS
# =============================================================================

LIST_WIDGET = f"""
    QListWidget {{
        background-color: {COLORS['surface0']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['surface1']};
        border-radius: 8px;
        padding: 8px;
        font-size: 12px;
    }}
    QListWidget::item {{
        padding: 8px;
        border-radius: 4px;
    }}
    QListWidget::item:selected {{
        background-color: {COLORS['blue']};
        color: {COLORS['crust']};
    }}
    QListWidget::item:hover {{
        background-color: {COLORS['surface1']};
    }}
"""


# =============================================================================
# TERMINAL/LOG OUTPUT
# =============================================================================

TERMINAL_OUTPUT = f"""
    QTextEdit {{
        background-color: {COLORS['crust']};
        color: {COLORS['green']};
        border: 1px solid {COLORS['surface1']};
        border-radius: 6px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
        padding: 4px;
    }}
"""


# =============================================================================
# MENUS
# =============================================================================

CONTEXT_MENU = f"""
    QMenu {{
        background-color: {COLORS['surface0']};
        border: 1px solid {COLORS['surface1']};
        border-radius: 4px;
        padding: 4px;
    }}
    QMenu::item {{
        color: {COLORS['text']};
        padding: 6px 20px;
        border-radius: 2px;
    }}
    QMenu::item:selected {{
        background-color: {COLORS['blue']};
        color: white;
    }}
    QMenu::separator {{
        height: 1px;
        background-color: {COLORS['surface1']};
        margin: 4px 8px;
    }}
"""


# =============================================================================
# TABS
# =============================================================================

TAB_WIDGET = f"""
    QTabWidget::pane {{
        border: 1px solid {COLORS['surface1']};
        border-radius: 8px;
        background-color: {COLORS['base']};
    }}
    QTabBar::tab {{
        background-color: {COLORS['surface0']};
        color: {COLORS['subtext0']};
        padding: 8px 16px;
        margin-right: 2px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
    }}
    QTabBar::tab:selected {{
        background-color: {COLORS['blue']};
        color: {COLORS['crust']};
    }}
    QTabBar::tab:hover:!selected {{
        background-color: {COLORS['surface1']};
    }}
"""


# =============================================================================
# SCROLLBARS
# =============================================================================

SCROLLBAR = f"""
    QScrollBar:vertical {{
        background-color: {COLORS['mantle']};
        width: 10px;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical {{
        background-color: {COLORS['surface1']};
        border-radius: 5px;
        min-height: 20px;
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: {COLORS['surface2']};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background-color: {COLORS['mantle']};
        height: 10px;
        border-radius: 5px;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {COLORS['surface1']};
        border-radius: 5px;
        min-width: 20px;
    }}
"""


# =============================================================================
# SPECIAL COMPONENTS
# =============================================================================

LOADING_ITEM_ROW = f"""
    QWidget {{
        background-color: {COLORS['surface0']};
        border-radius: 8px;
        padding: 4px;
    }}
"""

HEADER_LABEL = f"""
    QLabel {{
        font-size: 12px;
        font-weight: bold;
        color: {COLORS['blue']};
    }}
"""

STATUS_LABEL = f"""
    QLabel {{
        font-size: 12px;
        color: {COLORS['subtext0']};
    }}
"""

HINT_LABEL = f"""
    QLabel {{
        font-size: 12px;
        color: {COLORS['overlay']};
    }}
"""

# =============================================================================
# TEXT SELECTION STYLE - Makes all text selectable
# =============================================================================

SELECTABLE_LABEL = f"""
    QLabel {{
        color: {COLORS['text']};
    }}
"""

# Global base style that makes text selectable and sets sensible defaults
GLOBAL_BASE_STYLE = f"""
    /* Make QLabel text selectable by default */
    QLabel {{
        color: {COLORS['text']};
    }}
    
    /* Ensure text widgets allow selection */
    QTextEdit, QTextBrowser, QPlainTextEdit {{
        selection-background-color: {COLORS['blue']};
        selection-color: {COLORS['crust']};
    }}
    
    /* Standard button sizes - prevent tiny buttons */
    QPushButton {{
        min-height: 28px;
        min-width: 60px;
        padding: 6px 12px;
        font-size: 12px;
    }}
    
    /* Checkboxes - larger click target */
    QCheckBox {{
        spacing: 8px;
        min-height: 24px;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
    }}
"""


# =============================================================================
# CHAT MESSAGE STYLES (HTML)
# =============================================================================

def user_message_html(user_name: str, text: str) -> str:
    """Generate HTML for a user message."""
    return (
        f'<div style="background-color: {COLORS["surface0"]}; padding: 8px; margin: 4px 0; '
        f'border-radius: 8px; border-left: 3px solid {COLORS["blue"]};">'
        f'<b style="color: {COLORS["blue"]};">{user_name}:</b> {text}</div>'
    )


def ai_message_html(ai_name: str, text: str) -> str:
    """Generate HTML for an AI message."""
    return (
        f'<div style="background-color: {COLORS["mantle"]}; padding: 8px; margin: 4px 0; '
        f'border-radius: 8px; border-left: 3px solid {COLORS["green"]};">'
        f'<b style="color: {COLORS["green"]};">{ai_name}:</b> {text}</div>'
    )


def system_message_html(text: str, color: str = None) -> str:
    """Generate HTML for a system message."""
    color = color or COLORS["subtext0"]
    return f'<div style="color: {color}; padding: 4px; font-style: italic;">{text}</div>'


def error_message_html(text: str) -> str:
    """Generate HTML for an error message."""
    return f'<div style="color: {COLORS["red"]}; padding: 4px;">{text}</div>'


def success_message_html(text: str) -> str:
    """Generate HTML for a success message."""
    return f'<div style="color: {COLORS["green"]}; padding: 4px;">{text}</div>'


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def apply_dark_theme(widget):
    """Apply the standard dark theme to a widget."""
    widget.setStyleSheet(DARK_WINDOW + SCROLLBAR)


def get_color(name: str) -> str:
    """Get a color by name from the palette."""
    return COLORS.get(name, COLORS["text"])
