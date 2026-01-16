"""
Desktop Pet Avatar - DesktopMate-style companion

A desktop companion avatar that:
- Walks along the screen edges (taskbar, window edges)
- Sits on top of windows
- Reacts to mouse cursor
- Has idle behaviors (sleep, wave, dance)
- Falls with gravity simulation
- Interacts with desktop elements
- AI-controlled behaviors and responses

Based on DesktopMate/Shimeji concepts but fully AI-driven.

Works on both Windows and Linux.

Usage:
    from forge_ai.avatar.desktop_pet import DesktopPet
    
    pet = DesktopPet()
    pet.start()  # Avatar appears and starts behaving autonomously
    
    pet.set_mood("happy")  # Change mood
    pet.say("Hello!")  # Speech bubble
    pet.walk_to(500, 800)  # Move to position
    
    pet.stop()  # Hide avatar
"""

import math
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Dict, Callable, Tuple, Any

# Platform detection
IS_WINDOWS = sys.platform == 'win32'
IS_LINUX = sys.platform.startswith('linux')
IS_MAC = sys.platform == 'darwin'

# Try PyQt5 imports - may not be available on headless systems
try:
    from PyQt5.QtWidgets import (
        QWidget, QApplication, QLabel, QVBoxLayout, QGraphicsOpacityEffect, QMenu
    )
    from PyQt5.QtCore import (
        Qt, QTimer, QPoint, QRect, QPropertyAnimation, QEasingCurve,
        pyqtSignal, QSize, QByteArray, QObject
    )
    from PyQt5.QtGui import (
        QPixmap, QPainter, QColor, QCursor, QFont, QFontMetrics,
        QPainterPath, QBrush, QPen, QScreen
    )
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    # Dummy classes for type hints
    QWidget = object  # type: ignore
    QObject = object  # type: ignore
    pyqtSignal = lambda *args: None  # type: ignore
    Qt = None  # type: ignore

# Optional SVG support
try:
    from PyQt5.QtSvg import QSvgRenderer
    HAS_SVG = True
except ImportError:
    QSvgRenderer = None  # type: ignore
    HAS_SVG = False

from ..config import CONFIG

# Qt flags - get safely to work on both platforms
def _get_qt_flag(name: str, default: int) -> int:
    """Get Qt flag safely."""
    if Qt is None:
        return default
    return getattr(Qt, name, default)

Qt_FramelessWindowHint = _get_qt_flag('FramelessWindowHint', 0x00000800)
Qt_WindowStaysOnTopHint = _get_qt_flag('WindowStaysOnTopHint', 0x00040000)
Qt_Tool = _get_qt_flag('Tool', 0x00000008)
Qt_WA_TranslucentBackground = _get_qt_flag('WA_TranslucentBackground', 120)
Qt_LeftButton = _get_qt_flag('LeftButton', 0x00000001)
Qt_KeepAspectRatio = _get_qt_flag('KeepAspectRatio', 1)
Qt_SmoothTransformation = _get_qt_flag('SmoothTransformation', 1)
Qt_NoPen = _get_qt_flag('NoPen', 0)
Qt_AlignCenter = _get_qt_flag('AlignCenter', 0x0084)
Qt_DashLine = _get_qt_flag('DashLine', 2)
Qt_ClosedHandCursor = _get_qt_flag('ClosedHandCursor', 18)
Qt_ArrowCursor = _get_qt_flag('ArrowCursor', 0)


class PetState(Enum):
    """Pet behavior states."""
    IDLE = auto()
    WALKING = auto()
    FALLING = auto()
    SITTING = auto()
    SLEEPING = auto()
    CLIMBING = auto()
    DRAGGED = auto()
    WAVING = auto()
    DANCING = auto()
    LOOKING = auto()
    TALKING = auto()


class PetDirection(Enum):
    """Pet facing direction."""
    LEFT = -1
    RIGHT = 1


@dataclass
class PetConfig:
    """Desktop pet configuration."""
    # Size
    size: int = 128
    
    # Physics
    gravity: float = 0.8
    max_fall_speed: float = 15.0
    walk_speed: float = 3.0
    climb_speed: float = 2.0
    
    # Behavior timing
    idle_min_duration: float = 2.0
    idle_max_duration: float = 8.0
    walk_min_duration: float = 1.0
    walk_max_duration: float = 5.0
    
    # Behavior chances (0-1)
    chance_walk: float = 0.4
    chance_sit: float = 0.2
    chance_sleep: float = 0.1
    chance_wave: float = 0.1
    chance_dance: float = 0.1
    chance_climb: float = 0.1
    
    # Interaction
    follow_cursor_chance: float = 0.3
    react_to_windows: bool = True
    
    # Appearance
    enable_shadow: bool = True
    enable_speech_bubbles: bool = True
    speech_bubble_duration: float = 4.0
    
    # Colors (for built-in sprites)
    primary_color: str = "#6366f1"
    secondary_color: str = "#8b5cf6"
    accent_color: str = "#10b981"


@dataclass
class SpeechBubble:
    """A speech/thought bubble."""
    text: str
    is_thought: bool = False
    duration: float = 4.0
    created_at: float = 0.0


class DesktopPetWindow(QWidget):
    """The actual pet window that appears on desktop."""
    
    clicked = pyqtSignal()
    double_clicked = pyqtSignal()
    dragging = pyqtSignal(QPoint)
    dropped = pyqtSignal(QPoint)
    
    def __init__(self, size: int = 128):
        super().__init__(None)
        
        # Transparent, always-on-top, no taskbar
        self.setWindowFlags(
            Qt_FramelessWindowHint |
            Qt_WindowStaysOnTopHint |
            Qt_Tool
        )
        self.setAttribute(Qt_WA_TranslucentBackground, True)
        
        self._size = size
        self._padding = 20  # Extra space for speech bubble
        self.setFixedSize(self._size + self._padding * 2, self._size + self._padding * 2 + 60)
        
        self._pixmap: Optional[QPixmap] = None
        self._direction = PetDirection.RIGHT
        self._speech_bubble: Optional[SpeechBubble] = None
        self._drag_pos: Optional[QPoint] = None
        self._is_dragging = False
        
        # Animation state
        self._frame = 0
        self._animation_frames: List[QPixmap] = []
        
        self.setMouseTracking(True)
    
    def set_sprite(self, pixmap: QPixmap):
        """Set the pet sprite."""
        # Scale to size
        self._pixmap = pixmap.scaled(
            self._size, self._size,
            Qt_KeepAspectRatio, Qt_SmoothTransformation
        )
        self.update()
    
    def set_direction(self, direction: PetDirection):
        """Set facing direction (flips sprite)."""
        self._direction = direction
        self.update()
    
    def set_speech(self, text: str, is_thought: bool = False, duration: float = 4.0):
        """Show a speech bubble."""
        self._speech_bubble = SpeechBubble(
            text=text,
            is_thought=is_thought,
            duration=duration,
            created_at=time.time()
        )
        self.update()
        
        # Auto-hide after duration
        QTimer.singleShot(int(duration * 1000), self._clear_speech)
    
    def _clear_speech(self):
        """Clear speech bubble."""
        self._speech_bubble = None
        self.update()
    
    def paintEvent(self, event):
        """Draw the pet and speech bubble."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        # Calculate pet position (centered in widget)
        pet_x = self._padding
        pet_y = self._padding + 60  # Leave room for speech bubble above
        
        if self._pixmap:
            pixmap = self._pixmap
            
            # Flip horizontally if facing left
            if self._direction == PetDirection.LEFT:
                pixmap = pixmap.transformed(
                    pixmap.trueMatrix(
                        pixmap.transform().scale(-1, 1),
                        pixmap.width(), pixmap.height()
                    )
                )
            
            # Draw shadow
            shadow_color = QColor(0, 0, 0, 40)
            painter.setPen(Qt_NoPen)
            painter.setBrush(shadow_color)
            painter.drawEllipse(
                pet_x + 10, pet_y + self._size - 10,
                self._size - 20, 15
            )
            
            # Draw pet
            painter.drawPixmap(pet_x, pet_y, pixmap)
        else:
            # Placeholder
            painter.setPen(QColor("#6c7086"))
            painter.setBrush(QColor(30, 30, 46, 150))
            painter.drawEllipse(pet_x, pet_y, self._size, self._size)
            painter.drawText(
                pet_x, pet_y, self._size, self._size,
                Qt_AlignCenter, "?"
            )
        
        # Draw speech bubble
        if self._speech_bubble:
            self._draw_speech_bubble(painter, pet_x, pet_y)
    
    def _draw_speech_bubble(self, painter: QPainter, pet_x: int, pet_y: int):
        """Draw speech/thought bubble above pet."""
        text = self._speech_bubble.text
        is_thought = self._speech_bubble.is_thought
        
        # Calculate text size
        font = QFont("Segoe UI", 10)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        
        # Word wrap
        max_width = 150
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if metrics.horizontalAdvance(test_line) > max_width:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)
        
        if not lines:
            return
        
        # Bubble dimensions
        padding = 10
        line_height = metrics.height()
        bubble_width = min(max_width, max(metrics.horizontalAdvance(l) for l in lines)) + padding * 2
        bubble_height = line_height * len(lines) + padding * 2
        
        # Position above pet, centered
        bubble_x = pet_x + (self._size - bubble_width) // 2
        bubble_y = 5
        
        # Draw bubble
        path = QPainterPath()
        bubble_rect = QRect(bubble_x, bubble_y, bubble_width, bubble_height)
        path.addRoundedRect(float(bubble_x), float(bubble_y), float(bubble_width), float(bubble_height), 10, 10)
        
        # Pointer triangle
        pointer_x = bubble_x + bubble_width // 2
        pointer_y = bubble_y + bubble_height
        path.moveTo(pointer_x - 8, pointer_y)
        path.lineTo(pointer_x, pointer_y + 10)
        path.lineTo(pointer_x + 8, pointer_y)
        
        # Fill and stroke
        if is_thought:
            painter.setBrush(QColor(200, 200, 220, 230))
            painter.setPen(QPen(QColor("#9ca3af"), 2, Qt_DashLine))
        else:
            painter.setBrush(QColor(255, 255, 255, 240))
            painter.setPen(QPen(QColor("#374151"), 2))
        
        painter.drawPath(path)
        
        # Draw text
        painter.setPen(QColor("#1f2937"))
        text_y = bubble_y + padding + metrics.ascent()
        for line in lines:
            painter.drawText(bubble_x + padding, int(text_y), line)
            text_y += line_height
    
    def mousePressEvent(self, event):
        """Start drag."""
        if event.button() == Qt_LeftButton:
            self._drag_pos = event.globalPos() - self.pos()
            self._is_dragging = True
            self.setCursor(QCursor(Qt_ClosedHandCursor))
            self.dragging.emit(event.globalPos())
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle drag."""
        if self._is_dragging and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """End drag."""
        if self._is_dragging:
            self._is_dragging = False
            self._drag_pos = None
            self.setCursor(QCursor(Qt_ArrowCursor))
            self.dropped.emit(self.pos())
            event.accept()
    
    def mouseDoubleClickEvent(self, event):
        """Handle double click."""
        self.double_clicked.emit()
    
    def contextMenuEvent(self, event):
        """Right-click context menu."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 8px;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: #45475a;
            }
        """)
        
        menu.addAction("👋 Wave", self._action_wave)
        menu.addAction("💃 Dance", self._action_dance)
        menu.addAction("😴 Sleep", self._action_sleep)
        menu.addSeparator()
        menu.addAction("🏠 Go Home", self._action_go_home)
        menu.addAction("❌ Hide", self.hide)
        
        menu.exec_(event.globalPos())
    
    def _action_wave(self):
        """Trigger wave action via signal."""
        # Parent DesktopPet will handle this
        pass
    
    def _action_dance(self):
        pass
    
    def _action_sleep(self):
        pass
    
    def _action_go_home(self):
        # Move to bottom right of screen
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - self.width() - 50, screen.height() - self.height() - 80)


class DesktopPet(QObject):
    """
    AI-controlled desktop pet.
    
    Features:
    - Physics-based movement (gravity, collision)
    - Autonomous behaviors
    - Window awareness
    - AI personality integration
    - Speech bubbles
    - Multiple expressions/animations
    """
    
    # Signals
    state_changed = pyqtSignal(str)  # Emits state name
    speech_started = pyqtSignal(str)  # Emits speech text
    
    def __init__(self, config: Optional[PetConfig] = None):
        super().__init__()
        
        self.config = config or PetConfig()
        self._window: Optional[DesktopPetWindow] = None
        
        # State
        self._state = PetState.IDLE
        self._direction = PetDirection.RIGHT
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        self._on_ground = True
        
        # Position (float for smooth movement)
        self._x = 0.0
        self._y = 0.0
        
        # Behavior
        self._behavior_timer = 0.0
        self._current_behavior_duration = 0.0
        self._target_x: Optional[float] = None
        self._target_y: Optional[float] = None
        
        # AI integration
        self._ai_controller = None
        self._personality = None
        
        # Sprites for different states
        self._sprites: Dict[str, QPixmap] = {}
        self._current_sprite_key = "idle"
        
        # Animation
        self._animation_timer: Optional[QTimer] = None
        self._physics_timer: Optional[QTimer] = None
        self._behavior_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Screen info
        self._screen_rect = QRect(0, 0, 1920, 1080)
        
        # Initialize
        self._load_default_sprites()
    
    def _load_default_sprites(self):
        """Load/generate default sprites."""
        if not HAS_PYQT or not HAS_SVG:
            return
        
        try:
            from ..avatar.renderers.default_sprites import generate_sprite, SPRITE_TEMPLATES
        except ImportError:
            return
        
        for expression in SPRITE_TEMPLATES.keys():
            svg_data = generate_sprite(
                expression,
                self.config.primary_color,
                self.config.secondary_color,
                self.config.accent_color
            )
            
            # Convert SVG to pixmap
            renderer = QSvgRenderer(QByteArray(svg_data.encode('utf-8')))
            pixmap = QPixmap(self.config.size, self.config.size)
            pixmap.fill(QColor(0, 0, 0, 0))
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            
            self._sprites[expression] = pixmap
    
    def set_sprite(self, state: str, pixmap: QPixmap):
        """Set sprite for a state."""
        self._sprites[state] = pixmap
    
    def load_sprite_sheet(self, path: str, state_mapping: Dict[str, Tuple[int, int]]):
        """
        Load sprites from a sprite sheet.
        
        Args:
            path: Path to sprite sheet image
            state_mapping: Dict of state -> (x, y) position in sheet
        """
        sheet = QPixmap(path)
        if sheet.isNull():
            return
        
        for state, (x, y) in state_mapping.items():
            sprite = sheet.copy(
                x * self.config.size,
                y * self.config.size,
                self.config.size,
                self.config.size
            )
            self._sprites[state] = sprite
    
    def start(self):
        """Start the desktop pet."""
        if self._running:
            return
        
        if not HAS_PYQT:
            print("[DesktopPet] PyQt5 not available - cannot start desktop pet")
            return
        
        # Create window
        self._window = DesktopPetWindow(self.config.size)
        self._window.dropped.connect(self._on_dropped)
        self._window.dragging.connect(self._on_dragging)
        self._window.double_clicked.connect(self._on_double_click)
        
        # Get screen info
        screen = QApplication.primaryScreen()
        if screen:
            self._screen_rect = screen.geometry()
        
        # Initial position (bottom right)
        self._x = self._screen_rect.width() - self.config.size - 100
        self._y = self._screen_rect.height() - self.config.size - 80
        self._update_window_position()
        
        # Set initial sprite
        self._set_sprite("idle")
        
        # Start timers
        self._running = True
        
        # Physics timer (60 FPS)
        self._physics_timer = QTimer()
        self._physics_timer.timeout.connect(self._physics_tick)
        self._physics_timer.start(16)
        
        # Animation timer (10 FPS for sprite changes)
        self._animation_timer = QTimer()
        self._animation_timer.timeout.connect(self._animation_tick)
        self._animation_timer.start(100)
        
        # Behavior thread
        self._behavior_thread = threading.Thread(target=self._behavior_loop, daemon=True)
        self._behavior_thread.start()
        
        # Show window
        self._window.show()
        
        print("[DesktopPet] Started!")
    
    def stop(self):
        """Stop and hide the desktop pet."""
        self._running = False
        
        if self._physics_timer:
            self._physics_timer.stop()
        if self._animation_timer:
            self._animation_timer.stop()
        
        if self._window:
            self._window.hide()
            self._window.deleteLater()
            self._window = None
        
        print("[DesktopPet] Stopped")
    
    def is_running(self) -> bool:
        return self._running
    
    # === Movement ===
    
    def walk_to(self, x: float, y: Optional[float] = None):
        """
        Walk to a position.
        
        If y is None, walks along the ground to x.
        """
        self._target_x = x
        self._target_y = y
        
        # Determine direction
        if x > self._x:
            self._direction = PetDirection.RIGHT
        else:
            self._direction = PetDirection.LEFT
        
        if self._window:
            self._window.set_direction(self._direction)
        
        self._set_state(PetState.WALKING)
    
    def jump(self, strength: float = 10.0):
        """Make the pet jump."""
        if self._on_ground:
            self._velocity_y = -strength
            self._on_ground = False
    
    def teleport(self, x: float, y: float):
        """Instantly move to position."""
        self._x = x
        self._y = y
        self._update_window_position()
    
    # === Behavior ===
    
    def set_state(self, state: PetState):
        """Set pet state."""
        self._set_state(state)
    
    def _set_state(self, state: PetState):
        """Internal state setter."""
        if state == self._state:
            return
        
        self._state = state
        self.state_changed.emit(state.name)
        
        # Update sprite based on state
        sprite_map = {
            PetState.IDLE: "idle",
            PetState.WALKING: "neutral",
            PetState.SLEEPING: "sleeping",
            PetState.SITTING: "neutral",
            PetState.WAVING: "winking",
            PetState.DANCING: "excited",
            PetState.LOOKING: "thinking",
            PetState.TALKING: "happy",
            PetState.FALLING: "surprised",
            PetState.DRAGGED: "surprised",
        }
        self._set_sprite(sprite_map.get(state, "idle"))
    
    def say(self, text: str, duration: float = 4.0):
        """Show speech bubble."""
        if self._window:
            self._window.set_speech(text, is_thought=False, duration=duration)
            self._set_state(PetState.TALKING)
            self.speech_started.emit(text)
    
    def think(self, text: str, duration: float = 3.0):
        """Show thought bubble."""
        if self._window:
            self._window.set_speech(text, is_thought=True, duration=duration)
            self._set_state(PetState.LOOKING)
    
    def set_expression(self, expression: str):
        """Set facial expression."""
        self._set_sprite(expression)
    
    def set_mood(self, mood: str):
        """Set mood (maps to expression)."""
        mood_map = {
            "happy": "happy",
            "sad": "sad",
            "angry": "angry",
            "excited": "excited",
            "curious": "thinking",
            "sleepy": "sleeping",
            "neutral": "neutral",
            "love": "love",
        }
        self._set_sprite(mood_map.get(mood, "neutral"))
    
    # === AI Integration ===
    
    def link_ai(self, controller):
        """Link to AI controller for autonomous behavior."""
        self._ai_controller = controller
    
    def link_personality(self, personality):
        """Link to AI personality."""
        self._personality = personality
    
    # === Internal ===
    
    def _set_sprite(self, key: str):
        """Set current sprite."""
        if key in self._sprites and self._window:
            self._current_sprite_key = key
            self._window.set_sprite(self._sprites[key])
    
    def _update_window_position(self):
        """Update window position from internal coordinates."""
        if self._window:
            # Offset for padding in window
            padding = 20
            self._window.move(int(self._x) - padding, int(self._y) - padding - 60)
    
    def _physics_tick(self):
        """Physics update (called every frame)."""
        if not self._running or self._state == PetState.DRAGGED:
            return
        
        dt = 0.016  # ~60 FPS
        
        # Gravity
        if not self._on_ground:
            self._velocity_y += self.config.gravity
            self._velocity_y = min(self._velocity_y, self.config.max_fall_speed)
        
        # Walking movement
        if self._state == PetState.WALKING and self._target_x is not None:
            diff = self._target_x - self._x
            if abs(diff) < self.config.walk_speed:
                # Reached target
                self._x = self._target_x
                self._target_x = None
                self._set_state(PetState.IDLE)
            else:
                # Move toward target
                self._velocity_x = self.config.walk_speed * (1 if diff > 0 else -1)
        elif self._state != PetState.FALLING:
            self._velocity_x = 0
        
        # Apply velocity
        self._x += self._velocity_x
        self._y += self._velocity_y
        
        # Ground collision (bottom of screen / taskbar area)
        ground_y = self._screen_rect.height() - self.config.size - 50  # Leave room for taskbar
        if self._y >= ground_y:
            self._y = ground_y
            self._velocity_y = 0
            self._on_ground = True
            if self._state == PetState.FALLING:
                self._set_state(PetState.IDLE)
        else:
            self._on_ground = False
        
        # Screen edge collision
        if self._x < 0:
            self._x = 0
            self._velocity_x = 0
        elif self._x > self._screen_rect.width() - self.config.size:
            self._x = self._screen_rect.width() - self.config.size
            self._velocity_x = 0
        
        self._update_window_position()
    
    def _animation_tick(self):
        """Animation update (sprite changes)."""
        if not self._running:
            return
        
        # Could add frame-by-frame animation here
        pass
    
    def _behavior_loop(self):
        """Autonomous behavior loop (runs in thread)."""
        while self._running:
            time.sleep(0.1)
            
            # Don't change behavior if being dragged or falling
            if self._state in (PetState.DRAGGED, PetState.FALLING):
                continue
            
            # Check if current behavior is done
            self._behavior_timer += 0.1
            if self._behavior_timer >= self._current_behavior_duration:
                self._choose_next_behavior()
    
    def _choose_next_behavior(self):
        """Choose next autonomous behavior."""
        self._behavior_timer = 0
        
        # Random behavior selection
        roll = random.random()
        
        cumulative = 0
        behaviors = [
            (self.config.chance_walk, self._behavior_walk),
            (self.config.chance_sit, self._behavior_sit),
            (self.config.chance_sleep, self._behavior_sleep),
            (self.config.chance_wave, self._behavior_wave),
            (self.config.chance_dance, self._behavior_dance),
        ]
        
        for chance, behavior in behaviors:
            cumulative += chance
            if roll < cumulative:
                behavior()
                return
        
        # Default: idle
        self._behavior_idle()
    
    def _behavior_idle(self):
        """Idle behavior."""
        self._set_state(PetState.IDLE)
        self._current_behavior_duration = random.uniform(
            self.config.idle_min_duration,
            self.config.idle_max_duration
        )
    
    def _behavior_walk(self):
        """Walk behavior."""
        # Pick random destination on screen
        target_x = random.uniform(50, self._screen_rect.width() - self.config.size - 50)
        self.walk_to(target_x)
        self._current_behavior_duration = random.uniform(
            self.config.walk_min_duration,
            self.config.walk_max_duration
        )
    
    def _behavior_sit(self):
        """Sit behavior."""
        self._set_state(PetState.SITTING)
        self._current_behavior_duration = random.uniform(3.0, 8.0)
    
    def _behavior_sleep(self):
        """Sleep behavior."""
        self._set_state(PetState.SLEEPING)
        self._current_behavior_duration = random.uniform(5.0, 15.0)
    
    def _behavior_wave(self):
        """Wave behavior."""
        self._set_state(PetState.WAVING)
        self._current_behavior_duration = 2.0
    
    def _behavior_dance(self):
        """Dance behavior."""
        self._set_state(PetState.DANCING)
        self._current_behavior_duration = random.uniform(3.0, 6.0)
    
    def _on_dragging(self, pos: QPoint):
        """Handle being dragged."""
        self._set_state(PetState.DRAGGED)
        self._velocity_x = 0
        self._velocity_y = 0
    
    def _on_dropped(self, pos: QPoint):
        """Handle being dropped."""
        # Update internal position from window position
        if self._window:
            padding = 20
            self._x = pos.x() + padding
            self._y = pos.y() + padding + 60
        
        # Start falling if not on ground
        ground_y = self._screen_rect.height() - self.config.size - 50
        if self._y < ground_y - 5:
            self._set_state(PetState.FALLING)
            self._on_ground = False
        else:
            self._set_state(PetState.IDLE)
    
    def _on_double_click(self):
        """Handle double click."""
        # Could trigger interaction
        self.say("Hi there! 👋")


# Singleton
_pet_instance: Optional[DesktopPet] = None


def get_desktop_pet() -> DesktopPet:
    """Get the singleton DesktopPet instance."""
    global _pet_instance
    if _pet_instance is None:
        _pet_instance = DesktopPet()
    return _pet_instance
