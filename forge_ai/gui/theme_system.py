"""
Advanced Theme System for ForgeAI GUI
Provides multiple theme presets and custom theme support.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ThemeColors:
    """Theme color scheme."""
    # Background colors
    bg_primary: str = "#1e1e2e"
    bg_secondary: str = "#313244"
    bg_tertiary: str = "#45475a"
    
    # Text colors
    text_primary: str = "#cdd6f4"
    text_secondary: str = "#bac2de"
    text_disabled: str = "#6c7086"
    
    # Accent colors
    accent_primary: str = "#89b4fa"
    accent_secondary: str = "#b4befe"
    accent_hover: str = "#74c7ec"
    
    # Semantic colors
    success: str = "#a6e3a1"
    warning: str = "#f9e2af"
    error: str = "#f38ba8"
    info: str = "#89dceb"
    
    # Border colors
    border_primary: str = "#45475a"
    border_secondary: str = "#313244"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'ThemeColors':
        """Create from dictionary."""
        return cls(**data)


class Theme:
    """A complete theme with colors and stylesheet."""
    
    def __init__(
        self,
        name: str,
        colors: ThemeColors,
        description: str = ""
    ):
        """
        Initialize theme.
        
        Args:
            name: Theme name
            colors: Theme colors
            description: Theme description
        """
        self.name = name
        self.colors = colors
        self.description = description
    
    def generate_stylesheet(self) -> str:
        """Generate Qt stylesheet from theme colors."""
        c = self.colors
        
        return f"""
        /* Theme: {self.name} */
        QMainWindow, QWidget {{
            background-color: {c.bg_primary};
            color: {c.text_primary};
            font-size: 14px;
        }}
        
        QTextEdit, QPlainTextEdit, QLineEdit, QListWidget {{
            background-color: {c.bg_secondary};
            color: {c.text_primary};
            border: 1px solid {c.border_primary};
            border-radius: 4px;
            padding: 4px;
            font-size: 15px;
            selection-background-color: {c.accent_primary};
            selection-color: {c.bg_primary};
        }}
        
        QPushButton {{
            background-color: {c.accent_primary};
            color: {c.bg_primary};
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-size: 14px;
            font-weight: bold;
        }}
        
        QPushButton:hover {{
            background-color: {c.accent_secondary};
        }}
        
        QPushButton:pressed {{
            background-color: {c.accent_hover};
        }}
        
        QPushButton:disabled {{
            background-color: {c.bg_tertiary};
            color: {c.text_disabled};
        }}
        
        QGroupBox {{
            border: 1px solid {c.border_primary};
            border-radius: 4px;
            margin-top: 12px;
            padding-top: 8px;
            color: {c.text_primary};
        }}
        
        QGroupBox::title {{
            color: {c.accent_primary};
            subcontrol-origin: margin;
            left: 10px;
        }}
        
        QTabWidget::pane {{
            border: 1px solid {c.border_primary};
            border-radius: 4px;
            background-color: {c.bg_primary};
        }}
        
        QTabBar::tab {{
            background-color: {c.bg_secondary};
            color: {c.text_primary};
            padding: 8px 16px;
            font-size: 14px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 2px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {c.accent_primary};
            color: {c.bg_primary};
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: {c.bg_tertiary};
        }}
        
        QProgressBar {{
            border: 1px solid {c.border_primary};
            border-radius: 4px;
            background-color: {c.bg_secondary};
            text-align: center;
            color: {c.text_primary};
        }}
        
        QProgressBar::chunk {{
            background-color: {c.accent_primary};
            border-radius: 3px;
        }}
        
        QComboBox {{
            background-color: {c.bg_secondary};
            color: {c.text_primary};
            border: 1px solid {c.border_primary};
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 14px;
        }}
        
        QComboBox:hover {{
            border-color: {c.accent_primary};
        }}
        
        QComboBox::drop-down {{
            border: none;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {c.text_primary};
        }}
        
        QSlider::groove:horizontal {{
            border: 1px solid {c.border_primary};
            height: 8px;
            background: {c.bg_secondary};
            border-radius: 4px;
        }}
        
        QSlider::handle:horizontal {{
            background: {c.accent_primary};
            border: 1px solid {c.border_primary};
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }}
        
        QSlider::handle:horizontal:hover {{
            background: {c.accent_secondary};
        }}
        
        QCheckBox {{
            color: {c.text_primary};
            spacing: 8px;
            font-size: 14px;
        }}
        
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 1px solid {c.border_primary};
            border-radius: 3px;
            background-color: {c.bg_secondary};
        }}
        
        QCheckBox::indicator:checked {{
            background-color: {c.accent_primary};
        }}
        
        QLabel {{
            color: {c.text_primary};
            font-size: 14px;
        }}
        
        QScrollBar:vertical {{
            border: none;
            background: {c.bg_secondary};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background: {c.bg_tertiary};
            min-height: 20px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background: {c.accent_primary};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
        
        QMenuBar {{
            background-color: {c.bg_secondary};
            color: {c.text_primary};
            font-size: 14px;
        }}
        
        QMenuBar::item:selected {{
            background-color: {c.accent_primary};
            color: {c.bg_primary};
        }}
        
        QMenu {{
            background-color: {c.bg_secondary};
            color: {c.text_primary};
            border: 1px solid {c.border_primary};
            font-size: 14px;
        }}
        
        QMenu::item:selected {{
            background-color: {c.accent_primary};
            color: {c.bg_primary};
        }}
        
        /* Status indicators */
        .status-success {{
            color: {c.success};
        }}
        
        .status-warning {{
            color: {c.warning};
        }}
        
        .status-error {{
            color: {c.error};
        }}
        
        .status-info {{
            color: {c.info};
        }}
        """
    
    def save(self, path: Path):
        """Save theme to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'name': self.name,
            'description': self.description,
            'colors': self.colors.to_dict()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'Theme':
        """Load theme from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        colors = ThemeColors.from_dict(data['colors'])
        return cls(
            name=data['name'],
            colors=colors,
            description=data.get('description', '')
        )


class ThemeManager:
    """Manages themes and theme switching."""
    
    # Preset themes
    PRESETS = {
        'dark': Theme(
            name='Dark (Catppuccin Mocha)',
            colors=ThemeColors(
                bg_primary='#1e1e2e',
                bg_secondary='#313244',
                bg_tertiary='#45475a',
                text_primary='#cdd6f4',
                text_secondary='#bac2de',
                text_disabled='#6c7086',
                accent_primary='#89b4fa',
                accent_secondary='#b4befe',
                accent_hover='#74c7ec',
                success='#a6e3a1',
                warning='#f9e2af',
                error='#f38ba8',
                info='#89dceb',
                border_primary='#45475a',
                border_secondary='#313244'
            ),
            description='Dark theme with soft colors (default)'
        ),
        
        'light': Theme(
            name='Light',
            colors=ThemeColors(
                bg_primary='#eff1f5',
                bg_secondary='#e6e9ef',
                bg_tertiary='#ccd0da',
                text_primary='#4c4f69',
                text_secondary='#5c5f77',
                text_disabled='#9ca0b0',
                accent_primary='#1e66f5',
                accent_secondary='#7287fd',
                accent_hover='#04a5e5',
                success='#40a02b',
                warning='#df8e1d',
                error='#d20f39',
                info='#209fb5',
                border_primary='#ccd0da',
                border_secondary='#e6e9ef'
            ),
            description='Light theme for bright environments'
        ),
        
        'high_contrast': Theme(
            name='High Contrast',
            colors=ThemeColors(
                bg_primary='#000000',
                bg_secondary='#1a1a1a',
                bg_tertiary='#333333',
                text_primary='#ffffff',
                text_secondary='#e0e0e0',
                text_disabled='#808080',
                accent_primary='#00ff00',
                accent_secondary='#00cc00',
                accent_hover='#00ff99',
                success='#00ff00',
                warning='#ffff00',
                error='#ff0000',
                info='#00ffff',
                border_primary='#ffffff',
                border_secondary='#808080'
            ),
            description='High contrast for accessibility'
        ),
        
        'midnight': Theme(
            name='Midnight',
            colors=ThemeColors(
                bg_primary='#0a0e27',
                bg_secondary='#1a1f3a',
                bg_tertiary='#2a2f4a',
                text_primary='#b8c5d6',
                text_secondary='#a0b0c8',
                text_disabled='#5a6a7a',
                accent_primary='#6366f1',
                accent_secondary='#818cf8',
                accent_hover='#4f46e5',
                success='#34d399',
                warning='#fbbf24',
                error='#f87171',
                info='#60a5fa',
                border_primary='#2a2f4a',
                border_secondary='#1a1f3a'
            ),
            description='Deep blue midnight theme'
        ),
        
        'forest': Theme(
            name='Forest',
            colors=ThemeColors(
                bg_primary='#1a2e1a',
                bg_secondary='#2d4a2d',
                bg_tertiary='#3d5a3d',
                text_primary='#d4e4d4',
                text_secondary='#c0d8c0',
                text_disabled='#6a8a6a',
                accent_primary='#52b788',
                accent_secondary='#74c69d',
                accent_hover='#40916c',
                success='#95d5b2',
                warning='#f4a261',
                error='#e76f51',
                info='#52b788',
                border_primary='#3d5a3d',
                border_secondary='#2d4a2d'
            ),
            description='Nature-inspired green theme'
        ),
        
        'sunset': Theme(
            name='Sunset',
            colors=ThemeColors(
                bg_primary='#2d1b2e',
                bg_secondary='#472d4f',
                bg_tertiary='#5a3d5c',
                text_primary='#f4e4d7',
                text_secondary='#e8d4c7',
                text_disabled='#8a7a7c',
                accent_primary='#ff6b9d',
                accent_secondary='#ff8fab',
                accent_hover='#ffa5c0',
                success='#a8dadc',
                warning='#f1c40f',
                error='#e74c3c',
                info='#c9aff0',
                border_primary='#5a3d5c',
                border_secondary='#472d4f'
            ),
            description='Warm sunset colors'
        ),
        
        'cerulean': Theme(
            name='Cerulean',
            colors=ThemeColors(
                bg_primary='#0a1628',
                bg_secondary='#122640',
                bg_tertiary='#1a3550',
                text_primary='#e0f0ff',
                text_secondary='#b8d4e8',
                text_disabled='#5a7a8a',
                accent_primary='#00bcd4',
                accent_secondary='#4dd0e1',
                accent_hover='#00acc1',
                success='#26a69a',
                warning='#ffb74d',
                error='#ef5350',
                info='#29b6f6',
                border_primary='#1a3550',
                border_secondary='#122640'
            ),
            description='Cool cerulean blue ocean theme'
        ),
    }
    
    def __init__(self, themes_dir: Optional[Path] = None):
        """
        Initialize theme manager.
        
        Args:
            themes_dir: Directory for custom themes
        """
        self.themes_dir = themes_dir or Path('data/themes')
        self.themes_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_theme: Optional[Theme] = None
        self.custom_themes: Dict[str, Theme] = {}
        
        # Load custom themes
        self._load_custom_themes()
        
        # Set default theme
        self.set_theme('dark')
    
    def _load_custom_themes(self):
        """Load custom themes from themes directory."""
        for theme_file in self.themes_dir.glob('*.json'):
            try:
                theme = Theme.load(theme_file)
                self.custom_themes[theme.name] = theme
                logger.info(f"Loaded custom theme: {theme.name}")
            except Exception as e:
                logger.error(f"Failed to load theme {theme_file}: {e}")
    
    def get_theme(self, name: str) -> Optional[Theme]:
        """Get a theme by name."""
        # Check presets first
        if name in self.PRESETS:
            return self.PRESETS[name]
        
        # Check custom themes
        return self.custom_themes.get(name)
    
    def set_theme(self, name: str) -> bool:
        """
        Set the current theme.
        
        Args:
            name: Theme name
            
        Returns:
            True if successful
        """
        theme = self.get_theme(name)
        if not theme:
            logger.warning(f"Theme not found: {name}")
            return False
        
        self.current_theme = theme
        logger.info(f"Theme set to: {name}")
        return True
    
    def get_current_stylesheet(self) -> str:
        """Get stylesheet for current theme."""
        if not self.current_theme:
            return ""
        return self.current_theme.generate_stylesheet()
    
    def list_themes(self) -> Dict[str, str]:
        """List all available themes with descriptions."""
        themes = {}
        
        # Add presets
        for name, theme in self.PRESETS.items():
            themes[name] = theme.description
        
        # Add custom themes
        for name, theme in self.custom_themes.items():
            themes[f"custom:{name}"] = theme.description
        
        return themes
    
    def create_custom_theme(
        self,
        name: str,
        colors: ThemeColors,
        description: str = ""
    ) -> Theme:
        """
        Create a custom theme.
        
        Args:
            name: Theme name
            colors: Theme colors
            description: Theme description
            
        Returns:
            Created Theme
        """
        theme = Theme(name, colors, description)
        theme.save(self.themes_dir / f"{name}.json")
        self.custom_themes[name] = theme
        
        logger.info(f"Created custom theme: {name}")
        return theme
    
    def delete_custom_theme(self, name: str) -> bool:
        """Delete a custom theme."""
        if name not in self.custom_themes:
            return False
        
        # Delete file
        theme_file = self.themes_dir / f"{name}.json"
        if theme_file.exists():
            theme_file.unlink()
        
        # Remove from cache
        del self.custom_themes[name]
        
        logger.info(f"Deleted custom theme: {name}")
        return True
