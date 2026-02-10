"""
Spawnable Objects System - Avatar can generate and place objects on screen.

The AI's avatar can:
- Create speech/thought bubbles
- Hold items (sword, book, coffee, etc.)
- Spawn decorations/items around the screen
- Leave notes, drawings, stickers
- Create 2D sprites or 3D objects

Objects can:
- Be attached to avatar (holding)
- Float freely on screen
- Have physics (fall, bounce)
- Be persistent or temporary
- Be interactive (clickable)

Usage:
    from enigma_engine.avatar.spawnable_objects import ObjectSpawner, SpawnedObject
    
    spawner = ObjectSpawner()
    
    # Avatar holds something
    sword = spawner.create_held_object("sword", hand="right")
    
    # Leave a note on screen
    note = spawner.spawn_note("Remember to save!", x=100, y=100)
    
    # Create a floating decoration
    star = spawner.spawn_decoration("star", x=500, y=200, animated=True)
    
    # Spawn a custom image
    pic = spawner.spawn_image("path/to/image.png", x=300, y=400)
    
    # AI generates and spawns
    ai_art = spawner.spawn_ai_generated("a cute cat", style="pixel")
"""

import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

# Try PyQt5
try:
    from PyQt5.QtCore import (
        QPoint,
        QRect,
        Qt,
        QTimer,
        pyqtSignal,
    )
    from PyQt5.QtGui import (
        QColor,
        QFont,
        QLinearGradient,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
    )
    from PyQt5.QtWidgets import (
        QApplication,
        QMenu,
        QWidget,
    )
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QWidget = object
    pyqtSignal = lambda *args: None

# Try SVG support
try:
    from PyQt5.QtSvg import QSvgRenderer
    HAS_SVG = True
except ImportError:
    HAS_SVG = False


@dataclass
class SpawnSettings:
    """Settings for object spawning - AI should know when disabled."""
    allow_spawned_objects: bool = True   # Master toggle for all spawning
    allow_held_items: bool = True        # Can AI hold things?
    allow_screen_effects: bool = True    # Particles, effects, etc.
    allow_notes: bool = True             # Sticky notes, signs
    allow_bubbles: bool = True           # Speech/thought bubbles
    gaming_mode: bool = False            # Disable all overlays except avatar
    
    def is_type_allowed(self, obj_type: 'ObjectType') -> tuple[bool, str]:
        """Check if object type is allowed and return reason if not."""
        if self.gaming_mode:
            return False, "Gaming mode is active - object spawning disabled"
        
        if not self.allow_spawned_objects:
            return False, "Object spawning is currently disabled by user"
        
        # Check specific categories
        if obj_type in (ObjectType.HELD_ITEM,):
            if not self.allow_held_items:
                return False, "Held items are currently disabled by user"
        
        elif obj_type in (ObjectType.EFFECT,):
            if not self.allow_screen_effects:
                return False, "Screen effects are currently disabled by user"
        
        elif obj_type in (ObjectType.NOTE, ObjectType.SIGN):
            if not self.allow_notes:
                return False, "Notes/signs are currently disabled by user"
        
        elif obj_type in (ObjectType.SPEECH_BUBBLE, ObjectType.THOUGHT_BUBBLE):
            if not self.allow_bubbles:
                return False, "Speech bubbles are currently disabled by user"
        
        return True, ""


class ObjectType(Enum):
    """Types of spawnable objects."""
    SPEECH_BUBBLE = auto()      # Text bubble
    THOUGHT_BUBBLE = auto()     # Thought cloud
    HELD_ITEM = auto()          # Item avatar holds
    DECORATION = auto()         # Screen decoration
    NOTE = auto()               # Sticky note
    STICKER = auto()            # Fun sticker
    DRAWING = auto()            # AI-generated drawing
    IMAGE = auto()              # Custom image
    EFFECT = auto()             # Visual effect (sparkles, etc.)
    EMOJI = auto()              # Large emoji
    SIGN = auto()               # Sign/placard


class AttachPoint(Enum):
    """Where objects can attach to avatar."""
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    HEAD = "head"
    BACK = "back"
    FLOATING = "floating"       # Near avatar but not attached
    NONE = "none"               # Free on screen


@dataclass
class SpawnedObject:
    """A spawned object on screen."""
    id: str
    object_type: ObjectType
    x: float
    y: float
    width: int = 100
    height: int = 100
    
    # Content
    text: str = ""
    image_path: str = ""
    svg_data: str = ""
    color: str = "#ffffff"
    
    # Attachment
    attach_point: AttachPoint = AttachPoint.NONE
    attached_to_avatar: bool = False
    
    # Behavior
    animated: bool = False
    physics: bool = False
    temporary: bool = True
    lifetime: float = 0.0       # 0 = permanent
    created_at: float = field(default_factory=time.time)
    
    # State
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    rotation: float = 0.0
    scale: float = 1.0
    opacity: float = 1.0
    
    # Spawn blocking (for AI feedback)
    blocked: bool = False
    blocked_reason: str = ""
    
    def is_expired(self) -> bool:
        """Check if temporary object has expired."""
        if not self.temporary or self.lifetime <= 0:
            return False
        return time.time() - self.created_at > self.lifetime

class ObjectWindow(QWidget):
    """Window for displaying a spawned object with pixel-perfect click detection."""
    
    clicked = pyqtSignal(str)  # Emits object ID
    
    def __init__(self, obj: SpawnedObject, parent=None):
        if not HAS_PYQT:
            return
        super().__init__(parent)
        
        self.obj = obj
        self._rendered_pixmap: Optional[QPixmap] = None  # Cache for hit testing
        self._is_dragging = False
        
        # Frameless, transparent, always on top
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        
        self.setFixedSize(obj.width, obj.height)
        self.move(int(obj.x), int(obj.y))
        
        # Animation timer
        self._anim_phase = 0.0
        if obj.animated:
            self._anim_timer = QTimer()
            self._anim_timer.timeout.connect(self._animate)
            self._anim_timer.start(50)
    
    def _animate(self):
        """Update animation."""
        self._anim_phase += 0.1
        self.update()
    
    def nativeEvent(self, eventType, message):
        """Handle Windows native events for per-pixel hit testing.
        
        WM_NCHITTEST determines what part of the window the mouse is over.
        Return HTTRANSPARENT for transparent pixels so clicks pass through.
        """
        import sys
        if sys.platform != 'win32':
            return super().nativeEvent(eventType, message)
        
        try:
            import ctypes
            from ctypes import wintypes
            
            WM_NCHITTEST = 0x0084
            HTTRANSPARENT = -1
            HTCLIENT = 1
            
            msg = ctypes.cast(int(message), ctypes.POINTER(wintypes.MSG)).contents
            
            if msg.message == WM_NCHITTEST:
                # Don't do hit testing while dragging
                if self._is_dragging:
                    return super().nativeEvent(eventType, message)
                
                # Get mouse position from lParam
                x = msg.lParam & 0xFFFF
                y = (msg.lParam >> 16) & 0xFFFF
                
                # Handle signed coordinates (negative on multi-monitor)
                if x > 32767:
                    x -= 65536
                if y > 32767:
                    y -= 65536
                
                # Convert screen coords to widget coords
                local_pos = self.mapFromGlobal(QPoint(x, y))
                
                # Check if pixel is opaque
                if not self._is_pixel_opaque(local_pos.x(), local_pos.y()):
                    return True, HTTRANSPARENT  # Click passes through
                
                return True, HTCLIENT  # Handle this click
                
        except Exception:
            pass
        
        return super().nativeEvent(eventType, message)
    
    def _is_pixel_opaque(self, x: int, y: int, threshold: int = 10) -> bool:
        """Check if the pixel at (x, y) is opaque (visible part of object).
        
        Args:
            x, y: Position relative to widget
            threshold: Alpha value below which pixel is considered transparent
            
        Returns:
            True if pixel is opaque (should handle click), False if transparent
        """
        if not self._rendered_pixmap or self._rendered_pixmap.isNull():
            return False
        
        # Bounds check
        if x < 0 or x >= self._rendered_pixmap.width():
            return False
        if y < 0 or y >= self._rendered_pixmap.height():
            return False
        
        # Get pixel color from cached pixmap
        img = self._rendered_pixmap.toImage()
        if img.isNull():
            return False
        
        pixel = img.pixelColor(x, y)
        return pixel.alpha() > threshold
    
    def paintEvent(self, event):
        """Draw the object and cache for hit testing."""
        # Create pixmap to render into (for hit testing cache)
        self._rendered_pixmap = QPixmap(self.size())
        self._rendered_pixmap.fill(Qt.transparent)
        
        # Render to cached pixmap
        cache_painter = QPainter(self._rendered_pixmap)
        cache_painter.setRenderHint(QPainter.Antialiasing)
        self._draw_content(cache_painter)
        cache_painter.end()
        
        # Draw cached pixmap to screen
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._rendered_pixmap)
    
    def _draw_content(self, painter: QPainter):
        """Draw the object content."""
        # Apply transformations
        center = self.rect().center()
        painter.translate(center)
        painter.rotate(self.obj.rotation)
        painter.scale(self.obj.scale, self.obj.scale)
        painter.translate(-center)
        
        painter.setOpacity(self.obj.opacity)
        
        if self.obj.object_type == ObjectType.SPEECH_BUBBLE:
            self._draw_speech_bubble(painter)
        elif self.obj.object_type == ObjectType.THOUGHT_BUBBLE:
            self._draw_thought_bubble(painter)
        elif self.obj.object_type == ObjectType.NOTE:
            self._draw_note(painter)
        elif self.obj.object_type == ObjectType.STICKER:
            self._draw_sticker(painter)
        elif self.obj.object_type == ObjectType.EMOJI:
            self._draw_emoji(painter)
        elif self.obj.object_type == ObjectType.SIGN:
            self._draw_sign(painter)
        elif self.obj.object_type == ObjectType.EFFECT:
            self._draw_effect(painter)
        elif self.obj.image_path or self.obj.svg_data:
            self._draw_image(painter)
        else:
            self._draw_generic(painter)
    
    def _draw_speech_bubble(self, painter: QPainter):
        """Draw a speech bubble."""
        rect = self.rect().adjusted(5, 5, -5, -25)
        
        # Bubble background
        path = QPainterPath()
        path.addRoundedRect(rect, 15, 15)
        
        # Pointer
        pointer = QPainterPath()
        px = rect.center().x()
        pointer.moveTo(px - 10, rect.bottom())
        pointer.lineTo(px, rect.bottom() + 20)
        pointer.lineTo(px + 10, rect.bottom())
        path.addPath(pointer)
        
        # Draw
        painter.setBrush(QColor(255, 255, 255, 240))
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.drawPath(path)
        
        # Text
        painter.setPen(QColor(30, 30, 30))
        font = QFont("Segoe UI", 11)
        painter.setFont(font)
        painter.drawText(rect.adjusted(10, 5, -10, -5), Qt.AlignCenter | Qt.TextWordWrap, self.obj.text)
    
    def _draw_thought_bubble(self, painter: QPainter):
        """Draw a thought bubble (cloud style)."""
        rect = self.rect().adjusted(10, 10, -10, -30)
        
        # Cloud shape
        path = QPainterPath()
        path.addEllipse(rect.adjusted(0, rect.height()//4, -rect.width()//3, 0))
        path.addEllipse(rect.adjusted(rect.width()//4, 0, 0, -rect.height()//3))
        path.addEllipse(rect.adjusted(rect.width()//3, rect.height()//4, -10, -10))
        
        # Thought dots
        cx = rect.center().x()
        painter.setBrush(QColor(255, 255, 255, 230))
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        painter.drawEllipse(cx - 5, rect.bottom() + 5, 10, 10)
        painter.drawEllipse(cx - 3, rect.bottom() + 18, 6, 6)
        
        # Main cloud
        painter.setBrush(QColor(255, 255, 255, 240))
        painter.setPen(QPen(QColor(150, 150, 150), 2))
        painter.drawPath(path)
        
        # Text
        painter.setPen(QColor(80, 80, 80))
        font = QFont("Segoe UI", 10)
        font.setItalic(True)
        painter.setFont(font)
        painter.drawText(rect.adjusted(15, 10, -15, -10), Qt.AlignCenter | Qt.TextWordWrap, self.obj.text)
    
    def _draw_note(self, painter: QPainter):
        """Draw a sticky note."""
        rect = self.rect().adjusted(5, 5, -5, -5)
        
        # Yellow note background
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, QColor("#fff9c4"))
        gradient.setColorAt(1, QColor("#ffee58"))
        
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor("#fbc02d"), 1))
        
        # Slightly tilted
        if self.obj.animated:
            angle = math.sin(self._anim_phase) * 2
            painter.translate(rect.center())
            painter.rotate(angle)
            painter.translate(-rect.center())
        
        painter.drawRect(rect)
        
        # Folded corner
        fold = QPainterPath()
        fold.moveTo(rect.right() - 15, rect.top())
        fold.lineTo(rect.right(), rect.top() + 15)
        fold.lineTo(rect.right() - 15, rect.top() + 15)
        fold.closeSubpath()
        painter.setBrush(QColor("#fdd835"))
        painter.drawPath(fold)
        
        # Text
        painter.setPen(QColor(50, 50, 50))
        font = QFont("Comic Sans MS", 10)
        painter.setFont(font)
        painter.drawText(rect.adjusted(10, 20, -10, -10), Qt.AlignLeft | Qt.TextWordWrap, self.obj.text)
    
    def _draw_sticker(self, painter: QPainter):
        """Draw a fun sticker."""
        rect = self.rect().adjusted(10, 10, -10, -10)
        
        # Colorful background
        color = QColor(self.obj.color)
        painter.setBrush(color)
        painter.setPen(QPen(color.darker(120), 3))
        
        # Star shape for fun
        path = self._create_star_path(rect)
        painter.drawPath(path)
        
        # Text or emoji
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Segoe UI Emoji", 20)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, self.obj.text)
    
    def _draw_emoji(self, painter: QPainter):
        """Draw a large emoji."""
        rect = self.rect()
        
        font = QFont("Segoe UI Emoji", min(rect.width(), rect.height()) - 20)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, self.obj.text)
    
    def _draw_sign(self, painter: QPainter):
        """Draw a sign/placard."""
        rect = self.rect().adjusted(5, 20, -5, -5)
        
        # Sign board
        painter.setBrush(QColor("#5d4037"))
        painter.setPen(QPen(QColor("#3e2723"), 2))
        painter.drawRect(rect)
        
        # Inner panel
        inner = rect.adjusted(5, 5, -5, -5)
        painter.setBrush(QColor("#efebe9"))
        painter.drawRect(inner)
        
        # Stick
        stick_rect = QRect(rect.center().x() - 5, rect.bottom(), 10, 20)
        painter.setBrush(QColor("#6d4c41"))
        painter.drawRect(stick_rect)
        
        # Text
        painter.setPen(QColor(30, 30, 30))
        font = QFont("Impact", 12)
        painter.setFont(font)
        painter.drawText(inner, Qt.AlignCenter | Qt.TextWordWrap, self.obj.text)
    
    def _draw_effect(self, painter: QPainter):
        """Draw visual effects (sparkles, hearts, etc.)."""
        rect = self.rect()
        
        # Sparkle pattern
        painter.setPen(Qt.NoPen)
        
        for i in range(5):
            angle = (self._anim_phase + i * 72) * math.pi / 180 * 10
            size = 10 + math.sin(self._anim_phase + i) * 5
            x = rect.center().x() + math.cos(angle) * 30
            y = rect.center().y() + math.sin(angle) * 30
            
            gradient = QLinearGradient(x - size, y - size, x + size, y + size)
            gradient.setColorAt(0, QColor("#ffeb3b"))
            gradient.setColorAt(1, QColor("#ff9800"))
            painter.setBrush(gradient)
            
            star = self._create_star_path(QRect(int(x - size), int(y - size), int(size * 2), int(size * 2)), points=4)
            painter.drawPath(star)
    
    def _draw_image(self, painter: QPainter):
        """Draw custom image."""
        if self.obj.image_path and Path(self.obj.image_path).exists():
            pixmap = QPixmap(self.obj.image_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                x = (self.width() - scaled.width()) // 2
                y = (self.height() - scaled.height()) // 2
                painter.drawPixmap(x, y, scaled)
        elif self.obj.svg_data and HAS_SVG:
            from PyQt5.QtCore import QByteArray
            renderer = QSvgRenderer(QByteArray(self.obj.svg_data.encode()))
            renderer.render(painter, self.rect())
    
    def _draw_generic(self, painter: QPainter):
        """Draw generic object."""
        rect = self.rect().adjusted(5, 5, -5, -5)
        painter.setBrush(QColor(self.obj.color))
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.drawRoundedRect(rect, 10, 10)
        
        if self.obj.text:
            painter.setPen(QColor(30, 30, 30))
            painter.drawText(rect, Qt.AlignCenter, self.obj.text)
    
    def _create_star_path(self, rect: QRect, points: int = 5) -> QPainterPath:
        """Create a star shape path."""
        path = QPainterPath()
        cx, cy = rect.center().x(), rect.center().y()
        outer_r = min(rect.width(), rect.height()) / 2
        inner_r = outer_r * 0.5
        
        for i in range(points * 2):
            angle = (i * math.pi / points) - (math.pi / 2)
            r = outer_r if i % 2 == 0 else inner_r
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()
        return path
    
    def mousePressEvent(self, event):
        """Handle click."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.obj.id)
    
    def contextMenuEvent(self, event):
        """Right-click menu."""
        menu = QMenu(self)
        menu.addAction("Remove", self.close)
        menu.addAction("Make Permanent", lambda: setattr(self.obj, 'temporary', False))
        menu.exec_(event.globalPos())


class ObjectSpawner:
    """
    Creates and manages spawned objects for the avatar.
    
    Settings can be used to disable specific object types. AI gets feedback
    when attempting to spawn disabled types.
    """
    
    def __init__(self, settings: Optional[SpawnSettings] = None):
        self._objects: dict[str, SpawnedObject] = {}
        self._windows: dict[str, ObjectWindow] = {}
        self._next_id = 0
        self._settings = settings or SpawnSettings()
        
        # Physics/cleanup timer
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        
        # Avatar attachment callback
        self._avatar_position_callback: Optional[Callable] = None
    
    @property
    def settings(self) -> SpawnSettings:
        """Get spawn settings."""
        return self._settings
    
    @settings.setter
    def settings(self, value: SpawnSettings):
        """Set spawn settings."""
        self._settings = value
    
    def _check_allowed(self, obj_type: ObjectType) -> tuple[bool, str]:
        """Check if object type is allowed. Returns (allowed, reason_if_not)."""
        return self._settings.is_type_allowed(obj_type)
    
    def _generate_id(self) -> str:
        """Generate unique object ID."""
        self._next_id += 1
        return f"obj_{self._next_id}_{int(time.time())}"
    
    def start(self):
        """Start the spawner (physics/cleanup)."""
        if self._running:
            return
        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
    
    def stop(self):
        """Stop and clean up."""
        self._running = False
        for win in list(self._windows.values()):
            try:
                win.close()
            except Exception:
                pass
        self._windows.clear()
        self._objects.clear()
    
    def _update_loop(self):
        """Update physics and clean up expired objects."""
        while self._running:
            # Check for expired objects
            expired = [
                obj_id for obj_id, obj in self._objects.items()
                if obj.is_expired()
            ]
            for obj_id in expired:
                self.remove(obj_id)
            
            # Update physics
            for obj_id, obj in list(self._objects.items()):
                if obj.physics:
                    # Gravity
                    obj.velocity_y += 0.5
                    obj.x += obj.velocity_x
                    obj.y += obj.velocity_y
                    
                    # Bounce off bottom
                    screen_height = 1080  # Default
                    if HAS_PYQT:
                        app = QApplication.instance()
                        if app:
                            screen = app.primaryScreen()
                            if screen:
                                screen_height = screen.geometry().height()
                    
                    if obj.y > screen_height - obj.height:
                        obj.y = screen_height - obj.height
                        obj.velocity_y = -obj.velocity_y * 0.6
                        obj.velocity_x *= 0.8
                    
                    # Update window position
                    if obj_id in self._windows:
                        try:
                            self._windows[obj_id].move(int(obj.x), int(obj.y))
                        except Exception:
                            pass
            
            time.sleep(0.033)  # ~30 FPS
    
    def set_avatar_position_callback(self, callback: Callable[[], tuple[int, int]]):
        """Set callback to get avatar position for attached objects."""
        self._avatar_position_callback = callback
    
    # === Creation Methods ===
    
    def create_speech_bubble(
        self,
        text: str,
        x: int,
        y: int,
        lifetime: float = 5.0,
        width: int = 200,
        height: int = 100
    ) -> SpawnedObject:
        """Create a speech bubble."""
        obj = SpawnedObject(
            id=self._generate_id(),
            object_type=ObjectType.SPEECH_BUBBLE,
            x=x, y=y,
            width=width, height=height,
            text=text,
            temporary=True,
            lifetime=lifetime,
        )
        return self._spawn(obj)
    
    def create_thought_bubble(
        self,
        text: str,
        x: int,
        y: int,
        lifetime: float = 4.0,
    ) -> SpawnedObject:
        """Create a thought bubble."""
        obj = SpawnedObject(
            id=self._generate_id(),
            object_type=ObjectType.THOUGHT_BUBBLE,
            x=x, y=y,
            width=180, height=120,
            text=text,
            temporary=True,
            lifetime=lifetime,
        )
        return self._spawn(obj)
    
    def spawn_note(
        self,
        text: str,
        x: int,
        y: int,
        animated: bool = True,
        permanent: bool = False,
    ) -> SpawnedObject:
        """Spawn a sticky note on screen."""
        obj = SpawnedObject(
            id=self._generate_id(),
            object_type=ObjectType.NOTE,
            x=x, y=y,
            width=150, height=150,
            text=text,
            animated=animated,
            temporary=not permanent,
            lifetime=30.0 if not permanent else 0,
        )
        return self._spawn(obj)
    
    def spawn_emoji(
        self,
        emoji: str,
        x: int,
        y: int,
        size: int = 80,
        physics: bool = False,
        lifetime: float = 5.0,
    ) -> SpawnedObject:
        """Spawn a large emoji."""
        obj = SpawnedObject(
            id=self._generate_id(),
            object_type=ObjectType.EMOJI,
            x=x, y=y,
            width=size, height=size,
            text=emoji,
            physics=physics,
            temporary=True,
            lifetime=lifetime,
        )
        return self._spawn(obj)
    
    def spawn_sticker(
        self,
        text: str,
        x: int,
        y: int,
        color: str = "#e91e63",
        size: int = 100,
    ) -> SpawnedObject:
        """Spawn a fun sticker."""
        obj = SpawnedObject(
            id=self._generate_id(),
            object_type=ObjectType.STICKER,
            x=x, y=y,
            width=size, height=size,
            text=text,
            color=color,
            temporary=False,
        )
        return self._spawn(obj)
    
    def spawn_sign(
        self,
        text: str,
        x: int,
        y: int,
    ) -> SpawnedObject:
        """Spawn a sign/placard."""
        obj = SpawnedObject(
            id=self._generate_id(),
            object_type=ObjectType.SIGN,
            x=x, y=y,
            width=150, height=120,
            text=text,
            temporary=False,
        )
        return self._spawn(obj)
    
    def spawn_effect(
        self,
        x: int,
        y: int,
        effect_type: str = "sparkle",
        duration: float = 3.0,
    ) -> SpawnedObject:
        """Spawn visual effects."""
        obj = SpawnedObject(
            id=self._generate_id(),
            object_type=ObjectType.EFFECT,
            x=x, y=y,
            width=100, height=100,
            text=effect_type,
            animated=True,
            temporary=True,
            lifetime=duration,
        )
        return self._spawn(obj)
    
    def spawn_image(
        self,
        image_path: str,
        x: int,
        y: int,
        width: int = 100,
        height: int = 100,
        physics: bool = False,
    ) -> SpawnedObject:
        """Spawn a custom image."""
        obj = SpawnedObject(
            id=self._generate_id(),
            object_type=ObjectType.IMAGE,
            x=x, y=y,
            width=width, height=height,
            image_path=image_path,
            physics=physics,
            temporary=False,
        )
        return self._spawn(obj)
    
    def create_held_object(
        self,
        item_type: str,
        hand: str = "right",
    ) -> SpawnedObject:
        """
        Create an object for avatar to hold.
        
        item_type: "sword", "book", "coffee", "flower", "sign", etc.
        """
        svg_data = self._get_item_svg(item_type)
        
        attach = AttachPoint.RIGHT_HAND if hand == "right" else AttachPoint.LEFT_HAND
        
        obj = SpawnedObject(
            id=self._generate_id(),
            object_type=ObjectType.HELD_ITEM,
            x=0, y=0,  # Will be updated based on avatar
            width=60, height=60,
            svg_data=svg_data,
            text=item_type,
            attach_point=attach,
            attached_to_avatar=True,
            temporary=False,
        )
        return self._spawn(obj)
    
    def _get_item_svg(self, item_type: str) -> str:
        """Get SVG data for common items."""
        items = {
            "sword": '''<svg viewBox="0 0 100 100"><path d="M50 10 L55 70 L50 75 L45 70 Z" fill="#b0bec5"/><rect x="42" y="70" width="16" height="8" fill="#795548"/><rect x="35" y="75" width="30" height="5" fill="#5d4037"/></svg>''',
            "book": '''<svg viewBox="0 0 100 100"><rect x="20" y="15" width="60" height="70" rx="3" fill="#1565c0"/><rect x="25" y="20" width="50" height="60" fill="#e3f2fd"/><line x1="50" y1="20" x2="50" y2="80" stroke="#90caf9" stroke-width="2"/></svg>''',
            "coffee": '''<svg viewBox="0 0 100 100"><ellipse cx="50" cy="30" rx="25" ry="8" fill="#5d4037"/><path d="M25 30 L30 80 L70 80 L75 30" fill="#795548"/><path d="M75 40 Q90 40 90 55 Q90 70 75 70" fill="none" stroke="#795548" stroke-width="5"/><path d="M35 25 Q38 15 42 25" fill="none" stroke="#bdbdbd" stroke-width="2"/><path d="M50 20 Q53 10 57 20" fill="none" stroke="#bdbdbd" stroke-width="2"/></svg>''',
            "flower": '''<svg viewBox="0 0 100 100"><line x1="50" y1="50" x2="50" y2="95" stroke="#4caf50" stroke-width="4"/><circle cx="50" cy="40" r="8" fill="#ffeb3b"/><circle cx="40" cy="35" r="10" fill="#e91e63"/><circle cx="60" cy="35" r="10" fill="#e91e63"/><circle cx="35" cy="45" r="10" fill="#e91e63"/><circle cx="65" cy="45" r="10" fill="#e91e63"/><circle cx="45" cy="50" r="10" fill="#e91e63"/><circle cx="55" cy="50" r="10" fill="#e91e63"/></svg>''',
            "wand": '''<svg viewBox="0 0 100 100"><line x1="20" y1="80" x2="70" y2="30" stroke="#5d4037" stroke-width="6" stroke-linecap="round"/><circle cx="75" cy="25" r="8" fill="#ffeb3b"/><circle cx="80" cy="15" r="4" fill="#fff59d"/><circle cx="85" cy="28" r="3" fill="#fff59d"/></svg>''',
            "heart": '''<svg viewBox="0 0 100 100"><path d="M50 85 C20 55 10 30 30 20 C45 12 50 25 50 25 C50 25 55 12 70 20 C90 30 80 55 50 85" fill="#e91e63"/></svg>''',
        }
        return items.get(item_type, items.get("heart", ""))
    
    def _spawn(self, obj: SpawnedObject) -> SpawnedObject:
        """Actually spawn the object on screen.
        
        Checks settings before spawning. If blocked, the object is returned
        with a special 'blocked' flag set so AI knows why it failed.
        """
        # Check if this type is allowed
        allowed, reason = self._check_allowed(obj.object_type)
        if not allowed:
            # Mark as blocked - AI tools can check this
            obj.blocked = True
            obj.blocked_reason = reason
            return obj
        
        # Clear any blocked flags
        obj.blocked = False
        obj.blocked_reason = ""
        
        self._objects[obj.id] = obj
        
        if HAS_PYQT:
            try:
                window = ObjectWindow(obj)
                window.clicked.connect(lambda oid: self._on_object_clicked(oid))
                window.show()
                self._windows[obj.id] = window
                
                # Register with fullscreen controller for visibility management
                try:
                    from ..core.fullscreen_mode import get_fullscreen_controller
                    controller = get_fullscreen_controller()
                    controller.register_element(f'spawned_{obj.id}', window, category='spawned_objects')
                except Exception:
                    pass
            except Exception as e:
                print(f"[ObjectSpawner] Failed to create window: {e}")
        
        return obj
    
    def remove(self, obj_id: str):
        """Remove a spawned object."""
        if obj_id in self._objects:
            del self._objects[obj_id]
        if obj_id in self._windows:
            # Unregister from fullscreen controller
            try:
                from ..core.fullscreen_mode import get_fullscreen_controller
                controller = get_fullscreen_controller()
                controller.unregister_element(f'spawned_{obj_id}')
            except Exception:
                pass
            try:
                self._windows[obj_id].close()
            except Exception:
                pass
            del self._windows[obj_id]
    
    def remove_all(self):
        """Remove all spawned objects."""
        for obj_id in list(self._objects.keys()):
            self.remove(obj_id)
    
    def _on_object_clicked(self, obj_id: str):
        """Handle object click."""
        print(f"[ObjectSpawner] Object clicked: {obj_id}")
    
    def get_objects(self) -> list[SpawnedObject]:
        """Get all current objects."""
        return list(self._objects.values())


# Global spawner instance
_spawner_instance: Optional[ObjectSpawner] = None


def get_spawner() -> ObjectSpawner:
    """Get or create the global object spawner."""
    global _spawner_instance
    if _spawner_instance is None:
        _spawner_instance = ObjectSpawner()
        _spawner_instance.start()
    return _spawner_instance
