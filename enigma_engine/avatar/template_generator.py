"""
Avatar Sprite Template Generator

Creates template sprite sheets and emotion image sets for artists to customize.
Generates PSD-like layered structures and PNG templates with guidelines.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPixmap
    from PyQt5.QtWidgets import QApplication
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from ..config import CONFIG

# Standard emotions for Enigma AI Engine avatars
STANDARD_EMOTIONS = [
    "neutral", "happy", "sad", "surprised",
    "thinking", "confused", "angry", "excited"
]

# Extended emotions
EXTENDED_EMOTIONS = STANDARD_EMOTIONS + [
    "curious", "sleepy", "loving", "worried",
    "determined", "mischievous", "embarrassed", "proud"
]


@dataclass
class TemplateConfig:
    """Configuration for template generation."""
    size: int = 256  # Square size
    emotions: list[str] = None
    include_guidelines: bool = True
    include_face_guides: bool = True
    background_color: str = "#00000000"  # Transparent
    guideline_color: str = "#FF000050"  # Red, semi-transparent
    
    def __post_init__(self):
        if self.emotions is None:
            self.emotions = STANDARD_EMOTIONS.copy()


class SpriteTemplateGenerator:
    """
    Generates template images for creating custom avatars.
    
    Usage:
        generator = SpriteTemplateGenerator()
        
        # Create individual emotion templates
        generator.generate_emotion_templates("my_avatar", size=256)
        
        # Create sprite sheet template
        generator.generate_sprite_sheet("my_avatar", cols=4)
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        if output_dir is None:
            output_dir = Path(CONFIG.get("data_dir", "data")) / "avatar" / "templates"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_emotion_templates(
        self,
        name: str,
        config: Optional[TemplateConfig] = None
    ) -> list[Path]:
        """
        Generate individual template images for each emotion.
        
        Returns list of created file paths.
        """
        if not HAS_PYQT:
            return []
        
        if config is None:
            config = TemplateConfig()
        
        # Ensure Qt app exists
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        output_folder = self.output_dir / name
        output_folder.mkdir(parents=True, exist_ok=True)
        
        created = []
        
        for emotion in config.emotions:
            path = output_folder / f"{emotion}.png"
            pixmap = self._create_template_image(config, emotion)
            pixmap.save(str(path))
            created.append(path)
        
        # Create manifest
        manifest = {
            "name": name,
            "version": "1.0",
            "author": "",
            "description": f"Custom avatar: {name}",
            "avatar_type": "HUMAN",
            "emotions": {e: f"{e}.png" for e in config.emotions},
            "template": True
        }
        
        import json
        manifest_path = output_folder / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        created.append(manifest_path)
        
        # Create README
        readme = self._create_readme(name, config)
        readme_path = output_folder / "README.txt"
        readme_path.write_text(readme)
        created.append(readme_path)
        
        return created
    
    def generate_sprite_sheet(
        self,
        name: str,
        config: Optional[TemplateConfig] = None,
        cols: int = 4
    ) -> Optional[Path]:
        """
        Generate a sprite sheet template.
        
        All emotions in a single image, arranged in a grid.
        """
        if not HAS_PYQT:
            return None
        
        if config is None:
            config = TemplateConfig()
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        n_emotions = len(config.emotions)
        rows = (n_emotions + cols - 1) // cols
        
        width = config.size * cols
        height = config.size * rows
        
        sheet = QPixmap(width, height)
        sheet.fill(QColor(config.background_color))
        
        painter = QPainter(sheet)
        painter.setRenderHint(QPainter.Antialiasing)
        
        for i, emotion in enumerate(config.emotions):
            col = i % cols
            row = i // cols
            x = col * config.size
            y = row * config.size
            
            # Draw cell
            cell = self._create_template_image(config, emotion)
            painter.drawPixmap(x, y, cell)
            
            # Draw grid
            pen = QPen(QColor("#808080"))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawRect(x, y, config.size, config.size)
        
        painter.end()
        
        output_path = self.output_dir / f"{name}_spritesheet.png"
        sheet.save(str(output_path))
        
        return output_path
    
    def _create_template_image(
        self,
        config: TemplateConfig,
        emotion: str
    ) -> QPixmap:
        """Create a single template image."""
        size = config.size
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(config.background_color))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if config.include_guidelines:
            self._draw_guidelines(painter, size, config.guideline_color)
        
        if config.include_face_guides:
            self._draw_face_guides(painter, size)
        
        # Draw emotion label
        painter.setPen(QColor("#888888"))
        font = QFont("Arial", 10)
        painter.setFont(font)
        painter.drawText(4, 14, emotion.upper())
        
        painter.end()
        return pixmap
    
    def _draw_guidelines(self, painter: "QPainter", size: int, color: str):
        """Draw center and margin guidelines."""
        pen = QPen(QColor(color))
        pen.setWidth(1)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        
        # Center lines
        mid = size // 2
        painter.drawLine(mid, 0, mid, size)
        painter.drawLine(0, mid, size, mid)
        
        # Margin guides (10%)
        margin = size // 10
        painter.drawRect(margin, margin, size - 2 * margin, size - 2 * margin)
    
    def _draw_face_guides(self, painter: "QPainter", size: int):
        """Draw face proportion guidelines."""
        pen = QPen(QColor("#0000FF40"))  # Blue, semi-transparent
        pen.setWidth(1)
        pen.setStyle(Qt.DotLine)
        painter.setPen(pen)
        
        # Face oval area (centered, 60% of image)
        face_size = int(size * 0.6)
        offset = (size - face_size) // 2
        painter.drawEllipse(offset, offset, face_size, face_size)
        
        # Eye line (1/3 from top of face)
        eye_y = offset + face_size // 3
        painter.drawLine(offset, eye_y, offset + face_size, eye_y)
        
        # Mouth line (2/3 from top of face)
        mouth_y = offset + (face_size * 2) // 3
        painter.drawLine(offset, mouth_y, offset + face_size, mouth_y)
    
    def _create_readme(self, name: str, config: TemplateConfig) -> str:
        """Create a README file for the template."""
        return f"""
Enigma AI Engine Avatar Template: {name}
================================

This template contains placeholder images for creating a custom avatar.

IMAGE SIZE: {config.size}x{config.size} pixels

EMOTIONS TO CREATE:
{chr(10).join(f"  - {e}.png" for e in config.emotions)}

GUIDELINES:
  - Red dashed lines: Center and margin guides
  - Blue dotted lines: Face proportion guides (eyes, mouth)
  - Keep main content within the margin guides
  - The center cross helps align the avatar

HOW TO USE:
1. Open each .png file in your image editor
2. Draw your avatar expression on a new layer
3. Delete the template layer
4. Save as PNG with transparency

TIPS:
  - Keep consistent face position across all emotions
  - Only change facial features between emotions
  - Use a consistent art style
  - Test in Enigma AI Engine by importing the folder

After completing your avatar:
1. Edit manifest.json with your name and description
2. Delete this README.txt
3. Import via Avatar tab -> Import Avatar...

Enjoy creating!
"""


def generate_template(
    name: str,
    size: int = 256,
    emotions: Optional[list[str]] = None,
    include_spritesheet: bool = True
) -> dict[str, list[Path]]:
    """
    Quick function to generate avatar templates.
    
    Args:
        name: Template name
        size: Image size in pixels
        emotions: List of emotions (defaults to standard 8)
        include_spritesheet: Also generate sprite sheet
    
    Returns:
        Dict with 'individual' and 'spritesheet' paths
    """
    generator = SpriteTemplateGenerator()
    
    config = TemplateConfig(
        size=size,
        emotions=emotions or STANDARD_EMOTIONS
    )
    
    result = {
        "individual": generator.generate_emotion_templates(name, config),
        "spritesheet": []
    }
    
    if include_spritesheet:
        sheet = generator.generate_sprite_sheet(name, config)
        if sheet:
            result["spritesheet"] = [sheet]
    
    return result
