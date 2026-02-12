"""
Sprite Sheet Loader

Load and manage sprite sheets for 2D avatar animation.
Sprite sheets are single images containing multiple frames/expressions.

Supported formats:
- Grid-based: Equal-sized cells in rows/columns
- JSON atlas: Coordinates specified in JSON file
- Individual frames: Separate image files in a directory
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class SpriteFrame:
    """A single frame/sprite from a sprite sheet."""
    name: str
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    # For individual files
    file_path: str = ""
    # Cached image data
    _image: Any = None


@dataclass
class SpriteAnimation:
    """An animation sequence of frames."""
    name: str
    frames: list[str] = field(default_factory=list)  # Frame names
    fps: float = 12.0
    loop: bool = True


@dataclass
class SpriteSheet:
    """
    A sprite sheet with multiple frames.
    
    Contains frame definitions and optional animations.
    """
    
    name: str = ""
    source_path: str = ""
    
    # Size of the full sheet
    width: int = 0
    height: int = 0
    
    # Individual frames
    frames: dict[str, SpriteFrame] = field(default_factory=dict)
    
    # Animations (sequences of frames)
    animations: dict[str, SpriteAnimation] = field(default_factory=dict)
    
    # The full image (if loaded)
    _image: Any = None
    
    def get_frame_names(self) -> list[str]:
        """Get all frame names."""
        return list(self.frames.keys())
    
    def get_animation_names(self) -> list[str]:
        """Get all animation names."""
        return list(self.animations.keys())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "source": self.source_path,
            "width": self.width,
            "height": self.height,
            "frame_count": len(self.frames),
            "animation_count": len(self.animations),
            "frames": list(self.frames.keys()),
            "animations": list(self.animations.keys()),
        }


class SpriteSheetLoader:
    """
    Load sprite sheets in various formats.
    
    Supports:
    - Grid layout (automatic cell detection)
    - JSON atlas files
    - Directory of individual images
    
    Usage:
        loader = SpriteSheetLoader()
        
        # Load grid-based sheet (4 columns, 3 rows)
        sheet = loader.load_grid("sprites.png", cols=4, rows=3)
        
        # Load with JSON atlas
        sheet = loader.load_atlas("sprites.json")
        
        # Load from directory
        sheet = loader.load_directory("sprites/")
    """
    
    # Common expression names to look for
    EXPRESSION_NAMES = [
        "idle", "neutral", "happy", "sad", "angry", "surprised",
        "thinking", "confused", "excited", "winking", "sleeping",
        "speaking_1", "speaking_2", "speaking_3"
    ]
    
    def __init__(self):
        """Initialize loader."""
        self._cache: dict[str, SpriteSheet] = {}
    
    @staticmethod
    def is_available() -> bool:
        """Check if PIL is available for image processing."""
        return PIL_AVAILABLE
    
    def load_grid(
        self, 
        filepath: str, 
        cols: int, 
        rows: int,
        frame_names: Optional[list[str]] = None
    ) -> Optional[SpriteSheet]:
        """
        Load a grid-based sprite sheet.
        
        Args:
            filepath: Path to image file
            cols: Number of columns
            rows: Number of rows
            frame_names: Optional names for frames (left-to-right, top-to-bottom)
            
        Returns:
            SpriteSheet or None
        """
        if not PIL_AVAILABLE:
            logger.warning("Pillow not installed")
            return None
        
        path = Path(filepath)
        if not path.exists():
            logger.warning("File not found: %s", filepath)
            return None
        
        try:
            from PIL import Image as PILImage
            img = PILImage.open(path)
            
            sheet = SpriteSheet()
            sheet.name = path.stem
            sheet.source_path = str(path)
            sheet.width = img.width
            sheet.height = img.height
            sheet._image = img
            
            # Calculate cell size
            cell_width = img.width // cols
            cell_height = img.height // rows
            
            # Generate frames
            frame_index = 0
            for row in range(rows):
                for col in range(cols):
                    # Get frame name
                    if frame_names and frame_index < len(frame_names):
                        name = frame_names[frame_index]
                    else:
                        name = f"frame_{frame_index}"
                    
                    frame = SpriteFrame(
                        name=name,
                        x=col * cell_width,
                        y=row * cell_height,
                        width=cell_width,
                        height=cell_height
                    )
                    sheet.frames[name] = frame
                    frame_index += 1
            
            return sheet
            
        except Exception as e:
            logger.error("Error loading grid: %s", e)
            return None
    
    def load_atlas(self, filepath: str) -> Optional[SpriteSheet]:
        """
        Load a sprite sheet with JSON atlas.
        
        Expected JSON format:
        {
            "image": "sprites.png",
            "frames": {
                "idle": {"x": 0, "y": 0, "width": 64, "height": 64},
                "happy": {"x": 64, "y": 0, "width": 64, "height": 64},
                ...
            },
            "animations": {
                "speak": {
                    "frames": ["speak_1", "speak_2", "speak_3"],
                    "fps": 12,
                    "loop": true
                }
            }
        }
        
        Args:
            filepath: Path to JSON atlas file
            
        Returns:
            SpriteSheet or None
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning("File not found: %s", filepath)
            return None
        
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            
            sheet = SpriteSheet()
            sheet.name = path.stem
            sheet.source_path = str(path)
            
            # Load image if specified
            if PIL_AVAILABLE and 'image' in data:
                img_path = path.parent / data['image']
                if img_path.exists():
                    from PIL import Image as PILImage
                    img = PILImage.open(img_path)
                    sheet.width = img.width
                    sheet.height = img.height
                    sheet._image = img
            
            # Parse frames
            for name, frame_data in data.get('frames', {}).items():
                frame = SpriteFrame(
                    name=name,
                    x=frame_data.get('x', 0),
                    y=frame_data.get('y', 0),
                    width=frame_data.get('width', 0),
                    height=frame_data.get('height', 0)
                )
                sheet.frames[name] = frame
            
            # Parse animations
            for name, anim_data in data.get('animations', {}).items():
                anim = SpriteAnimation(
                    name=name,
                    frames=anim_data.get('frames', []),
                    fps=anim_data.get('fps', 12.0),
                    loop=anim_data.get('loop', True)
                )
                sheet.animations[name] = anim
            
            return sheet
            
        except Exception as e:
            logger.error("Error loading atlas: %s", e)
            return None
    
    def load_directory(self, dirpath: str) -> Optional[SpriteSheet]:
        """
        Load sprites from a directory of individual images.
        
        Files are named by their expression/frame name.
        Example: idle.png, happy.png, sad.png, etc.
        
        Args:
            dirpath: Path to directory
            
        Returns:
            SpriteSheet or None
        """
        path = Path(dirpath)
        if not path.is_dir():
            logger.warning("Not a directory: %s", dirpath)
            return None
        
        # Find image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
        image_files = [f for f in path.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            logger.warning("No images found in %s", dirpath)
            return None
        
        sheet = SpriteSheet()
        sheet.name = path.name
        sheet.source_path = str(path)
        
        for img_file in sorted(image_files):
            frame_name = img_file.stem
            
            # Get dimensions if PIL available
            width, height = 0, 0
            if PIL_AVAILABLE:
                try:
                    from PIL import Image as PILImage
                    img = PILImage.open(img_file)
                    width, height = img.size
                except Exception:
                    pass  # Intentionally silent
            
            frame = SpriteFrame(
                name=frame_name,
                width=width,
                height=height,
                file_path=str(img_file)
            )
            sheet.frames[frame_name] = frame
        
        return sheet
    
    def get_frame_image(
        self, 
        sheet: SpriteSheet, 
        frame_name: str
    ) -> Optional[Any]:
        """
        Get a specific frame as an image.
        
        Args:
            sheet: SpriteSheet to get frame from
            frame_name: Name of frame
            
        Returns:
            PIL Image or None
        """
        if not PIL_AVAILABLE:
            return None
        
        if frame_name not in sheet.frames:
            return None
        
        frame = sheet.frames[frame_name]
        
        # If frame has its own file
        if frame.file_path:
            try:
                from PIL import Image as PILImage
                return PILImage.open(frame.file_path)
            except Exception:
                return None
        
        # Otherwise crop from sheet
        if sheet._image:
            try:
                return sheet._image.crop((
                    frame.x, 
                    frame.y,
                    frame.x + frame.width,
                    frame.y + frame.height
                ))
            except Exception:
                return None
        
        return None
    
    def create_atlas_template(self, output_path: str):
        """
        Create a template atlas JSON file.
        
        Args:
            output_path: Where to save the template
        """
        template = {
            "image": "sprites.png",
            "frames": {
                "idle": {"x": 0, "y": 0, "width": 64, "height": 64},
                "happy": {"x": 64, "y": 0, "width": 64, "height": 64},
                "sad": {"x": 128, "y": 0, "width": 64, "height": 64},
                "thinking": {"x": 0, "y": 64, "width": 64, "height": 64},
            },
            "animations": {
                "speak": {
                    "frames": ["speak_1", "speak_2", "speak_3"],
                    "fps": 12,
                    "loop": True
                },
                "idle_anim": {
                    "frames": ["idle"],
                    "fps": 1,
                    "loop": True
                }
            }
        }
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2)
        
        logger.info("Created template at: %s", output_path)


def get_sprite_loader() -> SpriteSheetLoader:
    """Get or create sprite sheet loader."""
    return SpriteSheetLoader()
