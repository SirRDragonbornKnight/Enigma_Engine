"""
Real-Time Avatar Part Editor

Allows AI to edit individual avatar parts (hair, eyes, clothes) while the avatar
is visible, with morphing transitions between changes.

Features:
- Part-by-part editing (swap hair, eyes, clothes while visible)
- Morphing transitions between avatars
- AI describes changes, system generates just that part
- Layer-based composition for efficient updates

FILE: enigma_engine/avatar/part_editor.py
TYPE: Avatar Editing
MAIN CLASSES: AvatarPartEditor, AvatarPart, PartLayer, MorphTransition
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Qt imports with fallback
try:
    from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal, QObject
    from PyQt5.QtGui import QPixmap, QPainter, QColor, QImage, QTransform
    HAS_QT = True
except ImportError:
    HAS_QT = False
    logger.warning("PyQt5 not available - part editor GUI features disabled")


class PartType(Enum):
    """Types of avatar parts that can be edited separately."""
    BASE = "base"           # Base body/shape
    HEAD = "head"           # Head shape
    FACE = "face"           # Face features
    EYES = "eyes"           # Eye style
    EYEBROWS = "eyebrows"   # Eyebrow style
    NOSE = "nose"           # Nose style
    MOUTH = "mouth"         # Mouth/lips style
    EARS = "ears"           # Ear style
    HAIR = "hair"           # Hair style
    HAIR_BACK = "hair_back" # Hair behind head (for layering)
    SKIN = "skin"           # Skin color/texture
    OUTFIT = "outfit"       # Full outfit
    TOP = "top"             # Upper body clothing
    BOTTOM = "bottom"       # Lower body clothing
    SHOES = "shoes"         # Footwear
    ACCESSORY = "accessory" # Accessories (glasses, hat, etc.)
    EXPRESSION = "expression"  # Facial expression overlay
    EFFECT = "effect"       # Visual effects (glow, sparkles)
    CUSTOM = "custom"       # User-defined parts


@dataclass
class PartLayer:
    """A single layer in the avatar composition."""
    part_type: PartType
    name: str
    z_order: int = 0        # Draw order (higher = on top)
    image: Optional[Any] = None  # QPixmap or PIL Image
    image_path: Optional[str] = None
    offset_x: float = 0.0
    offset_y: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0
    opacity: float = 1.0
    visible: bool = True
    blend_mode: str = "normal"  # normal, multiply, screen, overlay
    tint_color: Optional[str] = None  # Hex color for tinting
    
    # Animation state
    _morph_progress: float = 1.0  # 0-1 for morph transitions
    _morph_source: Optional[Any] = None  # Previous image during morph


@dataclass 
class AvatarPart:
    """Definition of an avatar part with variants."""
    part_type: PartType
    name: str
    variants: List[str] = field(default_factory=list)  # List of variant names
    current_variant: str = "default"
    layer: Optional[PartLayer] = None
    
    # Part-specific settings
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # Generation prompt for AI
    generation_prompt: str = ""


@dataclass
class MorphTransition:
    """Tracks a morph transition between part states."""
    part_type: PartType
    source_image: Any
    target_image: Any
    duration: float = 0.5   # Seconds
    start_time: float = 0.0
    easing: str = "ease_in_out"
    
    @property
    def progress(self) -> float:
        """Get current progress (0-1)."""
        elapsed = time.time() - self.start_time
        raw_progress = min(1.0, elapsed / self.duration)
        
        # Apply easing
        if self.easing == "linear":
            return raw_progress
        elif self.easing == "ease_in":
            return raw_progress * raw_progress
        elif self.easing == "ease_out":
            return 1 - (1 - raw_progress) * (1 - raw_progress)
        else:  # ease_in_out
            if raw_progress < 0.5:
                return 2 * raw_progress * raw_progress
            else:
                return 1 - 2 * (1 - raw_progress) * (1 - raw_progress)
    
    @property
    def is_complete(self) -> bool:
        return self.progress >= 1.0


# Default layer z-order for proper composition
DEFAULT_Z_ORDER = {
    PartType.HAIR_BACK: 0,
    PartType.BASE: 10,
    PartType.SKIN: 15,
    PartType.OUTFIT: 20,
    PartType.BOTTOM: 21,
    PartType.TOP: 22,
    PartType.SHOES: 23,
    PartType.HEAD: 30,
    PartType.EARS: 35,
    PartType.FACE: 40,
    PartType.EYES: 50,
    PartType.EYEBROWS: 51,
    PartType.NOSE: 52,
    PartType.MOUTH: 53,
    PartType.HAIR: 60,
    PartType.ACCESSORY: 70,
    PartType.EXPRESSION: 80,
    PartType.EFFECT: 90,
    PartType.CUSTOM: 100,
}


class AvatarPartEditor:
    """
    Real-time avatar part editor with morphing transitions.
    
    Allows editing individual avatar parts while maintaining the overall
    composition and providing smooth transitions between changes.
    """
    
    def __init__(self, avatar_controller=None):
        self._lock = Lock()
        self._avatar_controller = avatar_controller
        
        # Part layers and definitions
        self._parts: Dict[PartType, AvatarPart] = {}
        self._layers: Dict[PartType, PartLayer] = {}
        
        # Active morph transitions
        self._active_morphs: List[MorphTransition] = []
        
        # Composition state
        self._composite_image: Optional[Any] = None
        self._composite_dirty = True
        self._canvas_width = 512
        self._canvas_height = 512
        
        # Animation
        self._animation_timer = None
        self._morph_callbacks: List[Callable[[], None]] = []
        
        # File paths
        self._parts_dir = Path("data/avatar/parts")
        self._parts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize default parts
        self._init_default_parts()
    
    def _init_default_parts(self):
        """Initialize default part definitions."""
        for part_type in PartType:
            z_order = DEFAULT_Z_ORDER.get(part_type, 50)
            self._parts[part_type] = AvatarPart(
                part_type=part_type,
                name=part_type.value,
                variants=["default"],
            )
            self._layers[part_type] = PartLayer(
                part_type=part_type,
                name=part_type.value,
                z_order=z_order,
            )
    
    def set_canvas_size(self, width: int, height: int):
        """Set the canvas size for composition."""
        self._canvas_width = width
        self._canvas_height = height
        self._composite_dirty = True
    
    # ========================================================================
    # Part Management
    # ========================================================================
    
    def set_part_image(
        self,
        part_type: PartType,
        image: Any,
        morph: bool = True,
        morph_duration: float = 0.5,
    ) -> bool:
        """
        Set the image for a part, optionally with morph transition.
        
        Args:
            part_type: The type of part to update
            image: QPixmap, PIL Image, or path to image file
            morph: Whether to morph from current to new
            morph_duration: Duration of morph transition
            
        Returns:
            True if successful
        """
        with self._lock:
            try:
                layer = self._layers.get(part_type)
                if not layer:
                    logger.warning(f"Unknown part type: {part_type}")
                    return False
                
                # Load image if path
                if isinstance(image, (str, Path)):
                    image = self._load_image(image)
                    if image is None:
                        return False
                
                # Start morph transition if enabled
                if morph and layer.image is not None:
                    transition = MorphTransition(
                        part_type=part_type,
                        source_image=layer.image,
                        target_image=image,
                        duration=morph_duration,
                        start_time=time.time(),
                    )
                    self._active_morphs.append(transition)
                    layer._morph_source = layer.image
                    layer._morph_progress = 0.0
                    
                    # Start animation timer if not running
                    self._start_animation()
                
                # Set the new image
                layer.image = image
                self._composite_dirty = True
                
                logger.debug(f"Set part image: {part_type.value}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to set part image: {e}")
                return False
    
    def set_part_from_file(
        self,
        part_type: PartType,
        file_path: str,
        morph: bool = True,
    ) -> bool:
        """Load and set part image from file."""
        try:
            path = Path(file_path)
            if not path.exists():
                # Try relative to parts directory
                path = self._parts_dir / file_path
            
            if not path.exists():
                logger.error(f"Part file not found: {file_path}")
                return False
            
            return self.set_part_image(part_type, path, morph=morph)
            
        except Exception as e:
            logger.error(f"Failed to load part file: {e}")
            return False
    
    def clear_part(self, part_type: PartType, morph: bool = True) -> bool:
        """Clear/hide a part."""
        with self._lock:
            layer = self._layers.get(part_type)
            if layer:
                if morph and layer.image:
                    # Fade out
                    layer._morph_source = layer.image
                    layer._morph_progress = 0.0
                    transition = MorphTransition(
                        part_type=part_type,
                        source_image=layer.image,
                        target_image=None,
                        duration=0.3,
                        start_time=time.time(),
                    )
                    self._active_morphs.append(transition)
                    self._start_animation()
                
                layer.image = None
                layer.visible = False
                self._composite_dirty = True
                return True
        return False
    
    def set_part_visible(self, part_type: PartType, visible: bool):
        """Show or hide a part."""
        with self._lock:
            layer = self._layers.get(part_type)
            if layer:
                layer.visible = visible
                self._composite_dirty = True
    
    def set_part_transform(
        self,
        part_type: PartType,
        offset_x: Optional[float] = None,
        offset_y: Optional[float] = None,
        scale: Optional[float] = None,
        rotation: Optional[float] = None,
        opacity: Optional[float] = None,
    ):
        """Set transform properties for a part."""
        with self._lock:
            layer = self._layers.get(part_type)
            if layer:
                if offset_x is not None:
                    layer.offset_x = offset_x
                if offset_y is not None:
                    layer.offset_y = offset_y
                if scale is not None:
                    layer.scale = max(0.1, min(10.0, scale))
                if rotation is not None:
                    layer.rotation = rotation
                if opacity is not None:
                    layer.opacity = max(0.0, min(1.0, opacity))
                self._composite_dirty = True
    
    def set_part_tint(self, part_type: PartType, color: Optional[str]):
        """Set tint color for a part (hex color or None to clear)."""
        with self._lock:
            layer = self._layers.get(part_type)
            if layer:
                layer.tint_color = color
                self._composite_dirty = True
    
    # ========================================================================
    # Variant Management
    # ========================================================================
    
    def register_variant(
        self,
        part_type: PartType,
        variant_name: str,
        image_path: str,
        settings: Optional[Dict] = None,
    ):
        """Register a variant for a part type."""
        part = self._parts.get(part_type)
        if part:
            if variant_name not in part.variants:
                part.variants.append(variant_name)
            if settings:
                part.settings[variant_name] = settings
            
            # Store image path mapping
            variant_dir = self._parts_dir / part_type.value
            variant_dir.mkdir(parents=True, exist_ok=True)
            
            mapping_file = variant_dir / "variants.json"
            mapping = {}
            if mapping_file.exists():
                mapping = json.loads(mapping_file.read_text())
            
            mapping[variant_name] = image_path
            mapping_file.write_text(json.dumps(mapping, indent=2))
    
    def set_variant(
        self,
        part_type: PartType,
        variant_name: str,
        morph: bool = True,
    ) -> bool:
        """Switch to a different variant of a part."""
        part = self._parts.get(part_type)
        if not part:
            return False
        
        # Load variant mapping
        variant_dir = self._parts_dir / part_type.value
        mapping_file = variant_dir / "variants.json"
        
        if mapping_file.exists():
            mapping = json.loads(mapping_file.read_text())
            if variant_name in mapping:
                image_path = mapping[variant_name]
                if self.set_part_from_file(part_type, image_path, morph=morph):
                    part.current_variant = variant_name
                    return True
        
        return False
    
    def get_variants(self, part_type: PartType) -> List[str]:
        """Get available variants for a part type."""
        part = self._parts.get(part_type)
        return part.variants if part else []
    
    # ========================================================================
    # Composition
    # ========================================================================
    
    def compose(self) -> Optional[Any]:
        """
        Compose all visible layers into a single image.
        
        Returns:
            QPixmap or PIL Image of the composed avatar
        """
        if not HAS_QT:
            return self._compose_pil()
        
        with self._lock:
            try:
                # Create canvas
                canvas = QPixmap(self._canvas_width, self._canvas_height)
                canvas.fill(Qt.transparent)
                
                painter = QPainter(canvas)
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
                
                # Sort layers by z-order
                sorted_layers = sorted(
                    self._layers.values(),
                    key=lambda l: l.z_order
                )
                
                # Draw each visible layer
                for layer in sorted_layers:
                    if not layer.visible or layer.image is None:
                        continue
                    
                    self._draw_layer(painter, layer)
                
                painter.end()
                
                self._composite_image = canvas
                self._composite_dirty = False
                
                return canvas
                
            except Exception as e:
                logger.error(f"Composition failed: {e}")
                return None
    
    def _draw_layer(self, painter: 'QPainter', layer: PartLayer):
        """Draw a single layer with transforms and effects."""
        try:
            # Get image to draw (handle morph transition)
            image = layer.image
            if layer._morph_progress < 1.0 and layer._morph_source is not None:
                # Blend between source and target
                image = self._blend_images(
                    layer._morph_source,
                    layer.image,
                    layer._morph_progress
                )
            
            if image is None:
                return
            
            # Save painter state
            painter.save()
            
            # Apply transforms
            center_x = self._canvas_width / 2 + layer.offset_x
            center_y = self._canvas_height / 2 + layer.offset_y
            
            painter.translate(center_x, center_y)
            painter.rotate(layer.rotation)
            painter.scale(layer.scale, layer.scale)
            painter.translate(-image.width() / 2, -image.height() / 2)
            
            # Apply opacity
            painter.setOpacity(layer.opacity)
            
            # Apply tint if set
            if layer.tint_color:
                tinted = self._apply_tint(image, layer.tint_color)
                painter.drawPixmap(0, 0, tinted)
            else:
                painter.drawPixmap(0, 0, image)
            
            painter.restore()
            
        except Exception as e:
            logger.error(f"Failed to draw layer {layer.name}: {e}")
    
    def _blend_images(
        self,
        source: Any,
        target: Any,
        progress: float,
    ) -> Any:
        """Blend two images together for morph transition."""
        if not HAS_QT:
            return target
        
        try:
            if target is None:
                # Fading out
                result = QPixmap(source.size())
                result.fill(Qt.transparent)
                
                painter = QPainter(result)
                painter.setOpacity(1.0 - progress)
                painter.drawPixmap(0, 0, source)
                painter.end()
                
                return result
            
            # Blend source and target
            result = QPixmap(target.size())
            result.fill(Qt.transparent)
            
            painter = QPainter(result)
            
            # Draw source with fading opacity
            painter.setOpacity(1.0 - progress)
            if source:
                painter.drawPixmap(0, 0, source)
            
            # Draw target with increasing opacity
            painter.setOpacity(progress)
            painter.drawPixmap(0, 0, target)
            
            painter.end()
            return result
            
        except Exception as e:
            logger.error(f"Image blend failed: {e}")
            return target
    
    def _apply_tint(self, image: Any, color: str) -> Any:
        """Apply color tint to an image."""
        if not HAS_QT:
            return image
        
        try:
            tinted = image.toImage()
            tint = QColor(color)
            
            for x in range(tinted.width()):
                for y in range(tinted.height()):
                    pixel = tinted.pixelColor(x, y)
                    if pixel.alpha() > 0:
                        # Blend with tint
                        blended = QColor(
                            int(pixel.red() * 0.5 + tint.red() * 0.5),
                            int(pixel.green() * 0.5 + tint.green() * 0.5),
                            int(pixel.blue() * 0.5 + tint.blue() * 0.5),
                            pixel.alpha()
                        )
                        tinted.setPixelColor(x, y, blended)
            
            return QPixmap.fromImage(tinted)
            
        except Exception as e:
            logger.error(f"Tint failed: {e}")
            return image
    
    def _compose_pil(self) -> Optional[Any]:
        """Compose using PIL (fallback when Qt not available)."""
        try:
            from PIL import Image
            
            # Create canvas
            canvas = Image.new('RGBA', (self._canvas_width, self._canvas_height), (0, 0, 0, 0))
            
            # Sort and draw layers
            sorted_layers = sorted(self._layers.values(), key=lambda l: l.z_order)
            
            for layer in sorted_layers:
                if not layer.visible or layer.image is None:
                    continue
                
                img = layer.image
                if isinstance(img, str):
                    img = Image.open(img).convert('RGBA')
                
                # Apply transforms (simplified)
                if layer.scale != 1.0:
                    new_size = (int(img.width * layer.scale), int(img.height * layer.scale))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                if layer.rotation != 0:
                    img = img.rotate(layer.rotation, expand=True, resample=Image.Resampling.BICUBIC)
                
                # Calculate position
                x = int(self._canvas_width / 2 - img.width / 2 + layer.offset_x)
                y = int(self._canvas_height / 2 - img.height / 2 + layer.offset_y)
                
                # Apply opacity
                if layer.opacity < 1.0:
                    alpha = img.split()[3]
                    alpha = alpha.point(lambda p: int(p * layer.opacity))
                    img.putalpha(alpha)
                
                # Paste with alpha
                canvas.paste(img, (x, y), img)
            
            self._composite_image = canvas
            self._composite_dirty = False
            return canvas
            
        except Exception as e:
            logger.error(f"PIL composition failed: {e}")
            return None
    
    # ========================================================================
    # Animation
    # ========================================================================
    
    def _start_animation(self):
        """Start the animation timer for morph transitions."""
        if not HAS_QT:
            return
        
        if self._animation_timer is None:
            from PyQt5.QtCore import QTimer
            self._animation_timer = QTimer()
            self._animation_timer.timeout.connect(self._update_morphs)
            self._animation_timer.start(16)  # ~60fps
    
    def _update_morphs(self):
        """Update active morph transitions."""
        with self._lock:
            completed = []
            
            for morph in self._active_morphs:
                layer = self._layers.get(morph.part_type)
                if layer:
                    layer._morph_progress = morph.progress
                    self._composite_dirty = True
                
                if morph.is_complete:
                    completed.append(morph)
                    if layer:
                        layer._morph_progress = 1.0
                        layer._morph_source = None
            
            # Remove completed morphs
            for morph in completed:
                self._active_morphs.remove(morph)
            
            # Stop timer if no active morphs
            if not self._active_morphs and self._animation_timer:
                self._animation_timer.stop()
                self._animation_timer = None
            
            # Notify callbacks
            for callback in self._morph_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Morph callback error: {e}")
    
    def on_morph_update(self, callback: Callable[[], None]):
        """Register callback for morph updates (for triggering redraws)."""
        self._morph_callbacks.append(callback)
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def _load_image(self, path) -> Optional[Any]:
        """Load an image from file."""
        path = Path(path)
        if not path.exists():
            logger.error(f"Image not found: {path}")
            return None
        
        if HAS_QT:
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                logger.error(f"Failed to load image: {path}")
                return None
            return pixmap
        else:
            try:
                from PIL import Image
                return Image.open(path).convert('RGBA')
            except Exception as e:
                logger.error(f"PIL image load failed: {e}")
                return None
    
    def get_part_info(self, part_type: PartType) -> Dict[str, Any]:
        """Get information about a part."""
        part = self._parts.get(part_type)
        layer = self._layers.get(part_type)
        
        if not part or not layer:
            return {}
        
        return {
            "type": part_type.value,
            "name": part.name,
            "variants": part.variants,
            "current_variant": part.current_variant,
            "visible": layer.visible,
            "z_order": layer.z_order,
            "offset": (layer.offset_x, layer.offset_y),
            "scale": layer.scale,
            "rotation": layer.rotation,
            "opacity": layer.opacity,
            "has_image": layer.image is not None,
            "tint_color": layer.tint_color,
        }
    
    def get_all_parts_info(self) -> Dict[str, Dict]:
        """Get information about all parts."""
        return {
            pt.value: self.get_part_info(pt)
            for pt in PartType
        }
    
    def save_composition(self, path: str) -> bool:
        """Save the composed avatar to a file."""
        try:
            image = self.compose()
            if image is None:
                return False
            
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if HAS_QT:
                return image.save(str(path))
            else:
                image.save(str(path))
                return True
                
        except Exception as e:
            logger.error(f"Failed to save composition: {e}")
            return False
    
    def load_preset(self, preset_path: str) -> bool:
        """Load a preset configuration for all parts."""
        try:
            path = Path(preset_path)
            if not path.exists():
                logger.error(f"Preset not found: {preset_path}")
                return False
            
            data = json.loads(path.read_text())
            
            for part_data in data.get("parts", []):
                part_type = PartType(part_data["type"])
                
                if "image" in part_data:
                    self.set_part_from_file(part_type, part_data["image"], morph=False)
                
                if "transform" in part_data:
                    self.set_part_transform(part_type, **part_data["transform"])
                
                if "tint" in part_data:
                    self.set_part_tint(part_type, part_data["tint"])
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load preset: {e}")
            return False
    
    def save_preset(self, preset_path: str) -> bool:
        """Save current configuration as a preset."""
        try:
            parts_data = []
            
            for part_type, layer in self._layers.items():
                part = self._parts.get(part_type)
                if layer.image is not None:
                    parts_data.append({
                        "type": part_type.value,
                        "variant": part.current_variant if part else "default",
                        "transform": {
                            "offset_x": layer.offset_x,
                            "offset_y": layer.offset_y,
                            "scale": layer.scale,
                            "rotation": layer.rotation,
                            "opacity": layer.opacity,
                        },
                        "tint": layer.tint_color,
                        "visible": layer.visible,
                    })
            
            data = {"parts": parts_data}
            
            path = Path(preset_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save preset: {e}")
            return False


# Global instance
_part_editor: Optional[AvatarPartEditor] = None


def get_part_editor() -> AvatarPartEditor:
    """Get or create the global part editor instance."""
    global _part_editor
    if _part_editor is None:
        _part_editor = AvatarPartEditor()
    return _part_editor


__all__ = [
    'AvatarPartEditor',
    'AvatarPart',
    'PartLayer',
    'PartType',
    'MorphTransition',
    'get_part_editor',
]
