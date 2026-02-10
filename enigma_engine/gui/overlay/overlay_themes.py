"""
Overlay Themes - Visual customization for the AI overlay.

Defines different visual themes for the overlay to match user preferences
and different use cases (gaming, productivity, etc.).
"""

from dataclasses import dataclass


@dataclass
class OverlayTheme:
    """Visual theme for the overlay."""
    name: str = "dark"
    background_color: str = "rgba(0, 0, 0, 0.7)"
    text_color: str = "#ffffff"
    accent_color: str = "#4a9eff"
    font_family: str = "Segoe UI"
    font_size: int = 12
    border_radius: int = 10
    border_color: str = "#4a9eff"
    border_width: int = 2
    avatar_size: int = 48
    padding: int = 15
    
    def to_stylesheet(self) -> str:
        """Convert theme to Qt stylesheet."""
        return f"""
            QWidget {{
                background-color: {self.background_color};
                color: {self.text_color};
                font-family: {self.font_family};
                font-size: {self.font_size}px;
                border-radius: {self.border_radius}px;
            }}
            QFrame#overlayFrame {{
                background-color: {self.background_color};
                border: {self.border_width}px solid {self.border_color};
                border-radius: {self.border_radius}px;
            }}
            QPushButton {{
                background-color: rgba(255, 255, 255, 0.1);
                border: 1px solid {self.accent_color};
                border-radius: 6px;
                padding: 6px 12px;
                color: {self.text_color};
            }}
            QPushButton:hover {{
                background-color: {self.accent_color};
                color: white;
            }}
            QLineEdit, QTextEdit {{
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 6px;
                padding: 6px;
                color: {self.text_color};
            }}
            QLabel {{
                color: {self.text_color};
            }}
        """


# Preset themes
OVERLAY_THEMES: dict[str, OverlayTheme] = {
    "dark": OverlayTheme(
        name="dark",
        background_color="rgba(0, 0, 0, 0.7)",
        text_color="#ffffff",
        accent_color="#4a9eff",
        font_family="Segoe UI",
        border_color="#4a9eff",
    ),
    
    "light": OverlayTheme(
        name="light",
        background_color="rgba(255, 255, 255, 0.9)",
        text_color="#000000",
        accent_color="#2196F3",
        font_family="Segoe UI",
        border_color="#2196F3",
    ),
    
    "gaming": OverlayTheme(
        name="gaming",
        background_color="rgba(0, 0, 0, 0.5)",
        text_color="#00ff00",
        accent_color="#00ff00",
        font_family="Consolas",
        font_size=11,
        border_color="#00ff00",
        border_radius=5,
    ),
    
    "minimal": OverlayTheme(
        name="minimal",
        background_color="rgba(0, 0, 0, 0.3)",
        text_color="#ffffff",
        accent_color="#888888",
        font_family="Segoe UI",
        font_size=11,
        border_radius=5,
        border_width=1,
        border_color="#666666",
    ),
    
    "cyberpunk": OverlayTheme(
        name="cyberpunk",
        background_color="rgba(0, 0, 0, 0.8)",
        text_color="#00ffff",
        accent_color="#ff00ff",
        font_family="Consolas",
        font_size=12,
        border_color="#ff00ff",
        border_width=2,
    ),
    
    "stealth": OverlayTheme(
        name="stealth",
        background_color="rgba(0, 0, 0, 0.2)",
        text_color="#cccccc",
        accent_color="#555555",
        font_family="Segoe UI",
        font_size=10,
        border_radius=8,
        border_width=1,
        border_color="#444444",
    ),
}


def get_theme(name: str) -> OverlayTheme:
    """Get a theme by name, returns dark theme if not found."""
    return OVERLAY_THEMES.get(name.lower(), OVERLAY_THEMES["dark"])
