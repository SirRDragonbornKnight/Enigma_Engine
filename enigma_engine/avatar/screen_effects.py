"""
Fullscreen Effect Overlay System

A transparent fullscreen overlay that AI can use to spawn visual effects
anywhere on screen - particles, explosions, trails, spell effects, etc.

Features:
- Click-through by default (doesn't block user input)
- Multi-monitor support (effects can span all screens)
- Gaming mode aware (auto-hides when gaming)
- AI-controlled via ScreenEffectManager

FILE: enigma_engine/avatar/screen_effects.py
TYPE: Visual Effects
MAIN CLASSES: ScreenEffectManager, EffectOverlay, ParticleEmitter, Particle
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Qt imports with fallbacks
try:
    from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF, pyqtSignal, QObject
    from PyQt5.QtGui import (
        QPainter, QColor, QPen, QBrush, QRadialGradient, 
        QPainterPath, QPixmap, QImage, QLinearGradient
    )
    from PyQt5.QtWidgets import QWidget, QApplication
    HAS_QT = True
except ImportError:
    HAS_QT = False
    logger.warning("PyQt5 not available - screen effects disabled")

# Qt flag compatibility
if HAS_QT:
    Qt_Tool = getattr(Qt, 'Tool', 0x00000004)
    Qt_FramelessWindowHint = getattr(Qt, 'FramelessWindowHint', 0x00000800)
    Qt_WindowStaysOnTopHint = getattr(Qt, 'WindowStaysOnTopHint', 0x00040000)
    Qt_WA_TranslucentBackground = getattr(Qt, 'WA_TranslucentBackground', 120)
    Qt_WA_TransparentForMouseEvents = getattr(Qt, 'WA_TransparentForMouseEvents', 51)
    Qt_WA_ShowWithoutActivating = getattr(Qt, 'WA_ShowWithoutActivating', 70)


# =============================================================================
# EFFECT TYPES AND PRESETS
# =============================================================================

class EffectType(Enum):
    """Types of visual effects."""
    PARTICLES = auto()      # General particle system
    EXPLOSION = auto()      # Burst outward
    TRAIL = auto()          # Following path/cursor
    RAIN = auto()           # Falling particles
    SNOW = auto()           # Gentle falling  
    FIRE = auto()           # Flames rising
    SPARKLE = auto()        # Twinkling stars
    MAGIC = auto()          # Magical swirls
    SMOKE = auto()          # Rising smoke
    CONFETTI = auto()       # Celebration
    HEARTS = auto()         # Love/affection
    LIGHTNING = auto()      # Electric bolts
    BUBBLE = auto()         # Rising bubbles
    SPIRAL = auto()         # Spinning spiral
    BEAM = auto()           # Laser/light beam
    RIPPLE = auto()         # Expanding rings
    CUSTOM = auto()         # User-defined


@dataclass
class EffectConfig:
    """Configuration for an effect."""
    effect_type: EffectType = EffectType.PARTICLES
    
    # Position and area
    x: float = 0.0
    y: float = 0.0
    width: float = 100.0
    height: float = 100.0
    
    # Timing
    duration: float = 3.0       # 0 = infinite
    spawn_rate: float = 20.0    # Particles per second
    
    # Particle properties
    particle_count: int = 50
    particle_size: tuple[float, float] = (4.0, 12.0)   # min, max
    particle_speed: tuple[float, float] = (50.0, 150.0)
    particle_lifetime: tuple[float, float] = (0.5, 2.0)
    
    # Colors (list of hex strings or QColor-compatible)
    colors: list[str] = field(default_factory=lambda: ["#ff6b6b", "#ffd93d", "#6bcb77"])
    
    # Physics
    gravity: float = 0.0        # Positive = down
    wind: float = 0.0           # Horizontal force
    friction: float = 0.98      # Velocity dampening
    
    # Appearance
    glow: bool = True
    glow_intensity: float = 0.5
    fade_out: bool = True
    shape: str = "circle"       # circle, square, star, heart, image, custom
    
    # Custom texture/image for particles
    texture: str = ""           # Path to texture image (relative to assets/effects/textures/)
    texture_tint: bool = True   # Apply color tint to texture
    
    # Direction
    direction: float = 0.0      # Degrees (0 = right, 90 = down)
    spread: float = 360.0       # Spread angle in degrees
    
    # Special
    follow_cursor: bool = False
    respect_bounds: bool = False    # Bounce off screen edges


# Preset effect configurations
EFFECT_PRESETS: dict[str, EffectConfig] = {
    "sparkle": EffectConfig(
        effect_type=EffectType.SPARKLE,
        spawn_rate=15,
        particle_size=(2, 8),
        particle_speed=(10, 50),
        particle_lifetime=(0.3, 1.0),
        colors=["#ffffff", "#fffacd", "#ffd700", "#87ceeb"],
        gravity=0,
        glow=True,
        glow_intensity=0.8,
    ),
    "fire": EffectConfig(
        effect_type=EffectType.FIRE,
        spawn_rate=40,
        particle_size=(6, 20),
        particle_speed=(80, 180),
        particle_lifetime=(0.4, 1.2),
        colors=["#ff4500", "#ff6347", "#ffd700", "#ffff00"],
        gravity=-120,  # Rise up
        direction=270,  # Up
        spread=45,
        glow=True,
    ),
    "snow": EffectConfig(
        effect_type=EffectType.SNOW,
        spawn_rate=25,
        particle_size=(3, 8),
        particle_speed=(20, 60),
        particle_lifetime=(3.0, 6.0),
        colors=["#ffffff", "#f0f8ff", "#e6e6fa"],
        gravity=30,
        wind=15,
        direction=90,  # Down
        spread=30,
        fade_out=True,
    ),
    "rain": EffectConfig(
        effect_type=EffectType.RAIN,
        spawn_rate=60,
        particle_size=(2, 4),
        particle_speed=(300, 500),
        particle_lifetime=(0.5, 1.5),
        colors=["#4a90d9", "#6bb3f0", "#a8d4f7"],
        gravity=500,
        direction=100,  # Slightly angled
        spread=10,
        glow=False,
        shape="line",
    ),
    "explosion": EffectConfig(
        effect_type=EffectType.EXPLOSION,
        duration=1.5,
        spawn_rate=0,  # Burst mode
        particle_count=80,
        particle_size=(4, 16),
        particle_speed=(200, 500),
        particle_lifetime=(0.3, 1.0),
        colors=["#ff4500", "#ff6347", "#ffd700", "#ffffff"],
        gravity=100,
        spread=360,
        glow=True,
    ),
    "confetti": EffectConfig(
        effect_type=EffectType.CONFETTI,
        spawn_rate=30,
        particle_size=(6, 14),
        particle_speed=(100, 250),
        particle_lifetime=(2.0, 4.0),
        colors=["#ff6b6b", "#ffd93d", "#6bcb77", "#4ecdc4", "#9b59b6", "#3498db"],
        gravity=80,
        direction=270,  # Up initially
        spread=120,
        shape="square",
        glow=False,
    ),
    "hearts": EffectConfig(
        effect_type=EffectType.HEARTS,
        spawn_rate=8,
        particle_size=(12, 24),
        particle_speed=(30, 80),
        particle_lifetime=(1.5, 3.0),
        colors=["#ff69b4", "#ff1493", "#dc143c", "#ff6b6b"],
        gravity=-40,  # Float up
        spread=60,
        direction=270,
        shape="heart",
        glow=True,
        glow_intensity=0.4,
    ),
    "magic": EffectConfig(
        effect_type=EffectType.MAGIC,
        spawn_rate=25,
        particle_size=(4, 12),
        particle_speed=(60, 150),
        particle_lifetime=(0.8, 2.0),
        colors=["#9b59b6", "#8e44ad", "#3498db", "#00ffff", "#ff00ff"],
        gravity=0,
        spread=360,
        glow=True,
        glow_intensity=0.7,
    ),
    "smoke": EffectConfig(
        effect_type=EffectType.SMOKE,
        spawn_rate=15,
        particle_size=(20, 50),
        particle_speed=(20, 60),
        particle_lifetime=(2.0, 4.0),
        colors=["#696969", "#808080", "#a9a9a9", "#d3d3d3"],
        gravity=-30,
        direction=270,
        spread=40,
        glow=False,
        fade_out=True,
    ),
    "bubble": EffectConfig(
        effect_type=EffectType.BUBBLE,
        spawn_rate=10,
        particle_size=(8, 25),
        particle_speed=(30, 80),
        particle_lifetime=(2.0, 5.0),
        colors=["#87ceeb", "#add8e6", "#b0e0e6", "#afeeee"],
        gravity=-50,
        wind=10,
        direction=270,
        spread=30,
        shape="circle",
        glow=True,
        glow_intensity=0.3,
    ),
    "lightning": EffectConfig(
        effect_type=EffectType.LIGHTNING,
        duration=0.5,
        spawn_rate=0,
        particle_count=20,
        particle_size=(2, 6),
        particle_speed=(500, 1000),
        particle_lifetime=(0.05, 0.15),
        colors=["#ffffff", "#00ffff", "#87ceeb"],
        direction=90,
        spread=30,
        glow=True,
        glow_intensity=1.0,
    ),
    "ripple": EffectConfig(
        effect_type=EffectType.RIPPLE,
        duration=2.0,
        spawn_rate=3,
        particle_size=(5, 5),
        particle_speed=(100, 200),
        particle_lifetime=(1.0, 2.0),
        colors=["#4a90d9", "#6bb3f0"],
        spread=360,
        glow=True,
    ),
}


# =============================================================================
# PARTICLE CLASS
# =============================================================================

@dataclass
class Particle:
    """A single particle in an effect."""
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0             # Velocity X
    vy: float = 0.0             # Velocity Y
    size: float = 5.0
    color: str = "#ffffff"
    alpha: float = 1.0
    lifetime: float = 1.0       # Total lifetime
    age: float = 0.0            # Current age
    rotation: float = 0.0
    rotation_speed: float = 0.0
    shape: str = "circle"
    texture: str = ""           # Texture path if using image
    
    # Computed
    _start_size: float = field(default=0.0, repr=False)
    _start_alpha: float = field(default=1.0, repr=False)
    
    def __post_init__(self):
        self._start_size = self.size
        self._start_alpha = self.alpha
    
    @property
    def is_dead(self) -> bool:
        """Check if particle has expired."""
        return self.age >= self.lifetime
    
    @property
    def life_ratio(self) -> float:
        """Progress through lifetime (0 to 1)."""
        return min(1.0, self.age / self.lifetime) if self.lifetime > 0 else 1.0
    
    def update(self, dt: float, config: EffectConfig):
        """Update particle state."""
        self.age += dt
        
        # Apply forces
        self.vy += config.gravity * dt
        self.vx += config.wind * dt
        
        # Apply friction
        self.vx *= config.friction
        self.vy *= config.friction
        
        # Move
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Rotate
        self.rotation += self.rotation_speed * dt
        
        # Fade out
        if config.fade_out:
            # Fade in first 10%, fade out last 50%
            if self.life_ratio < 0.1:
                self.alpha = self._start_alpha * (self.life_ratio / 0.1)
            elif self.life_ratio > 0.5:
                fade_progress = (self.life_ratio - 0.5) / 0.5
                self.alpha = self._start_alpha * (1.0 - fade_progress)
            else:
                self.alpha = self._start_alpha
        
        # Shrink over time for some effects
        if config.effect_type in (EffectType.FIRE, EffectType.SMOKE, EffectType.EXPLOSION):
            self.size = self._start_size * (1.0 - self.life_ratio * 0.7)


# =============================================================================
# PARTICLE EMITTER
# =============================================================================

class ParticleEmitter:
    """Spawns and manages particles for an effect."""
    
    def __init__(self, effect_id: str, config: EffectConfig):
        self.id = effect_id
        self.config = config
        self.particles: list[Particle] = []
        self.created_at = time.time()
        self.last_spawn_time = 0.0
        self._spawn_accumulator = 0.0
        self._is_burst = config.spawn_rate <= 0
        self._burst_done = False
        
        # Burst mode: spawn all particles immediately
        if self._is_burst:
            self._spawn_burst()
    
    @property
    def is_finished(self) -> bool:
        """Check if effect is complete."""
        # Check duration
        if self.config.duration > 0:
            if time.time() - self.created_at > self.config.duration:
                # Duration expired - finish when particles die
                return len(self.particles) == 0
        
        # Burst mode: done when all particles dead
        if self._is_burst and self._burst_done:
            return len(self.particles) == 0
        
        return False
    
    def _spawn_burst(self):
        """Spawn all particles at once (explosion mode)."""
        for _ in range(self.config.particle_count):
            self._spawn_particle()
        self._burst_done = True
    
    def _spawn_particle(self) -> Particle:
        """Create a new particle."""
        cfg = self.config
        
        # Random position within spawn area
        x = cfg.x + random.uniform(0, cfg.width)
        y = cfg.y + random.uniform(0, cfg.height)
        
        # Random direction within spread
        base_angle = math.radians(cfg.direction)
        spread_rad = math.radians(cfg.spread / 2)
        angle = base_angle + random.uniform(-spread_rad, spread_rad)
        
        # Random speed
        speed = random.uniform(cfg.particle_speed[0], cfg.particle_speed[1])
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        
        # Random properties
        size = random.uniform(cfg.particle_size[0], cfg.particle_size[1])
        lifetime = random.uniform(cfg.particle_lifetime[0], cfg.particle_lifetime[1])
        color = random.choice(cfg.colors)
        rotation_speed = random.uniform(-180, 180) if cfg.shape != "circle" else 0
        
        particle = Particle(
            x=x, y=y, vx=vx, vy=vy,
            size=size, color=color,
            lifetime=lifetime, shape=cfg.shape,
            rotation_speed=rotation_speed,
        )
        
        self.particles.append(particle)
        return particle
    
    def update(self, dt: float):
        """Update all particles and spawn new ones."""
        cfg = self.config
        
        # Spawn new particles (continuous mode)
        if not self._is_burst and cfg.spawn_rate > 0:
            # Check if still within duration
            if cfg.duration <= 0 or (time.time() - self.created_at < cfg.duration):
                self._spawn_accumulator += cfg.spawn_rate * dt
                while self._spawn_accumulator >= 1.0:
                    self._spawn_particle()
                    self._spawn_accumulator -= 1.0
        
        # Update existing particles
        for particle in self.particles:
            particle.update(dt, cfg)
        
        # Remove dead particles
        self.particles = [p for p in self.particles if not p.is_dead]
    
    def stop(self):
        """Stop spawning (let existing particles finish)."""
        self.config.duration = 0.001  # Tiny duration to stop spawning
        self._is_burst = True
        self._burst_done = True


# =============================================================================
# EFFECT OVERLAY WINDOW
# =============================================================================

if HAS_QT:
    
    class EffectOverlay(QWidget):
        """
        Fullscreen transparent overlay for rendering visual effects.
        Click-through by default, spans entire screen or monitor.
        """
        
        # Signals
        effect_finished = pyqtSignal(str)  # effect_id
        
        def __init__(
            self, 
            screen_index: int = 0,
            parent: QWidget = None,
            manager: 'ScreenEffectManager' = None,
        ):
            super().__init__(parent)
            
            self.screen_index = screen_index
            self._manager = manager
            self._emitters: dict[str, ParticleEmitter] = {}
            self._hidden_for_gaming = False
            
            # Setup window
            self._setup_window()
            
            # Animation timer (60 FPS)
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._last_tick = time.time()
            
            # Start animation when first effect added
            self._timer_active = False
        
        def _setup_window(self):
            """Configure window flags for transparent, click-through overlay."""
            # Window flags: tool window, frameless, always on top
            self.setWindowFlags(
                Qt_Tool |
                Qt_FramelessWindowHint |
                Qt_WindowStaysOnTopHint
            )
            
            # Transparent background
            self.setAttribute(Qt_WA_TranslucentBackground, True)
            
            # Click-through (mouse events pass to windows below)
            self.setAttribute(Qt_WA_TransparentForMouseEvents, True)
            
            # Don't steal focus
            self.setAttribute(Qt_WA_ShowWithoutActivating, True)
            
            # Position and size to cover screen
            self._position_on_screen()
        
        def _position_on_screen(self):
            """Position overlay to cover specified screen."""
            app = QApplication.instance()
            if not app:
                return
            
            screens = app.screens()
            if self.screen_index >= len(screens):
                self.screen_index = 0
            
            if screens:
                screen = screens[self.screen_index]
                geometry = screen.geometry()
                self.setGeometry(geometry)
        
        def _start_timer(self):
            """Start animation timer if not running."""
            if not self._timer_active:
                self._last_tick = time.time()
                self._timer.start(16)  # ~60 FPS
                self._timer_active = True
        
        def _stop_timer(self):
            """Stop animation timer."""
            if self._timer_active:
                self._timer.stop()
                self._timer_active = False
        
        def _tick(self):
            """Animation tick - update all effects."""
            now = time.time()
            dt = min(now - self._last_tick, 0.1)  # Cap delta to prevent huge jumps
            self._last_tick = now
            
            # Update emitters
            finished = []
            for effect_id, emitter in self._emitters.items():
                emitter.update(dt)
                if emitter.is_finished:
                    finished.append(effect_id)
            
            # Remove finished effects
            for effect_id in finished:
                del self._emitters[effect_id]
                self.effect_finished.emit(effect_id)
            
            # Stop timer if no effects
            if not self._emitters:
                self._stop_timer()
                self.hide()
            
            # Trigger repaint
            self.update()
        
        def add_effect(self, effect_id: str, config: EffectConfig):
            """Add a new effect to this overlay."""
            emitter = ParticleEmitter(effect_id, config)
            self._emitters[effect_id] = emitter
            
            # Show and start animating
            if not self._hidden_for_gaming:
                self.show()
            self._start_timer()
        
        def remove_effect(self, effect_id: str):
            """Remove an effect (stops spawning, lets particles finish)."""
            if effect_id in self._emitters:
                self._emitters[effect_id].stop()
        
        def clear_all(self):
            """Remove all effects immediately."""
            self._emitters.clear()
            self._stop_timer()
            self.hide()
        
        def set_gaming_mode(self, enabled: bool):
            """Hide/show for gaming mode."""
            self._hidden_for_gaming = enabled
            if enabled:
                self.hide()
            elif self._emitters:
                self.show()
        
        def paintEvent(self, event):
            """Render all particle effects."""
            if not self._emitters:
                return
            
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            
            # Draw each emitter's particles
            for emitter in self._emitters.values():
                self._draw_emitter(painter, emitter)
            
            painter.end()
        
        def _draw_emitter(self, painter: QPainter, emitter: ParticleEmitter):
            """Draw all particles for an emitter."""
            cfg = emitter.config
            
            for particle in emitter.particles:
                self._draw_particle(painter, particle, cfg)
        
        def _draw_particle(self, painter: QPainter, p: Particle, cfg: EffectConfig):
            """Draw a single particle."""
            # Parse color
            color = QColor(p.color)
            color.setAlphaF(p.alpha)
            
            # Save painter state
            painter.save()
            
            # Translate to particle position and rotate
            painter.translate(p.x, p.y)
            if p.rotation != 0:
                painter.rotate(p.rotation)
            
            half_size = p.size / 2
            
            # Check for texture (image-based particle)
            texture_path = p.texture or cfg.texture
            if texture_path and self._manager:
                pixmap = self._manager.get_texture(texture_path)
                if pixmap:
                    # Draw textured particle
                    self._draw_textured_particle(painter, pixmap, p, cfg, color)
                    painter.restore()
                    return
            
            # Draw glow if enabled
            if cfg.glow and cfg.glow_intensity > 0:
                glow_size = p.size * 2.5
                gradient = QRadialGradient(0, 0, glow_size)
                glow_color = QColor(p.color)
                glow_color.setAlphaF(p.alpha * cfg.glow_intensity * 0.5)
                gradient.setColorAt(0, glow_color)
                gradient.setColorAt(1, QColor(0, 0, 0, 0))
                painter.setBrush(QBrush(gradient))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(0, 0), glow_size, glow_size)
            
            # Draw particle shape
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            
            shape = p.shape or cfg.shape
            if shape == "circle":
                painter.drawEllipse(QPointF(0, 0), half_size, half_size)
            
            elif shape == "square":
                painter.drawRect(QRectF(-half_size, -half_size, p.size, p.size))
            
            elif shape == "line":
                # Line particle (for rain)
                pen = QPen(color)
                pen.setWidthF(p.size / 2)
                painter.setPen(pen)
                # Draw line in direction of velocity
                length = p.size * 3
                painter.drawLine(QPointF(0, 0), QPointF(0, length))
            
            elif shape == "star":
                self._draw_star(painter, half_size, 5)
            
            elif shape == "heart":
                self._draw_heart(painter, p.size)
            
            elif shape == "triangle":
                path = QPainterPath()
                path.moveTo(0, -half_size)
                path.lineTo(half_size, half_size)
                path.lineTo(-half_size, half_size)
                path.closeSubpath()
                painter.drawPath(path)
            
            # Restore painter state
            painter.restore()
        
        def _draw_textured_particle(
            self, 
            painter: QPainter, 
            pixmap: QPixmap, 
            p: Particle, 
            cfg: EffectConfig,
            color: QColor
        ):
            """Draw a particle using a texture image."""
            # Scale pixmap to particle size
            target_size = int(p.size * 2)
            scaled = pixmap.scaled(
                target_size, target_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # Apply alpha
            painter.setOpacity(p.alpha)
            
            # Draw centered
            x = -scaled.width() / 2
            y = -scaled.height() / 2
            
            # Apply color tint if enabled
            if cfg.texture_tint and color.alpha() > 0:
                # Create tinted version
                tinted = self._tint_pixmap(scaled, color)
                painter.drawPixmap(int(x), int(y), tinted)
            else:
                painter.drawPixmap(int(x), int(y), scaled)
            
            painter.setOpacity(1.0)
        
        def _tint_pixmap(self, pixmap: QPixmap, color: QColor) -> QPixmap:
            """Apply a color tint to a pixmap."""
            # Convert to image for manipulation
            image = pixmap.toImage()
            
            # Tint by blending with color
            tint_r, tint_g, tint_b = color.red(), color.green(), color.blue()
            
            for y in range(image.height()):
                for x in range(image.width()):
                    pixel = image.pixelColor(x, y)
                    if pixel.alpha() > 0:
                        # Blend original with tint (multiply blend mode)
                        r = int(pixel.red() * tint_r / 255)
                        g = int(pixel.green() * tint_g / 255)
                        b = int(pixel.blue() * tint_b / 255)
                        image.setPixelColor(x, y, QColor(r, g, b, pixel.alpha()))
            
            return QPixmap.fromImage(image)
        
        def _draw_star(self, painter: QPainter, radius: float, points: int = 5):
            """Draw a star shape."""
            path = QPainterPath()
            inner_radius = radius * 0.4
            
            for i in range(points * 2):
                angle = math.pi * i / points - math.pi / 2
                r = radius if i % 2 == 0 else inner_radius
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            
            path.closeSubpath()
            painter.drawPath(path)
        
        def _draw_heart(self, painter: QPainter, size: float):
            """Draw a heart shape."""
            path = QPainterPath()
            scale = size / 20  # Normalize
            
            # Heart curve
            path.moveTo(0, -5 * scale)
            path.cubicTo(
                -5 * scale, -12 * scale,
                -12 * scale, -5 * scale,
                0, 8 * scale
            )
            path.cubicTo(
                12 * scale, -5 * scale,
                5 * scale, -12 * scale,
                0, -5 * scale
            )
            
            painter.drawPath(path)


# =============================================================================
# SCREEN EFFECT MANAGER (Singleton)
# =============================================================================

class ScreenEffectManager:
    """
    Central manager for all screen effects.
    
    Provides simple API for AI to spawn effects:
        manager = get_effect_manager()
        manager.spawn("sparkle", x=500, y=300)
        manager.spawn("explosion", x=800, y=400, duration=2.0)
        manager.spawn("custom", texture="star.png", x=500, y=300)
    
    Asset directories:
        assets/effects/textures/  - Particle texture images (PNG with transparency)
        assets/effects/presets/   - Custom effect preset JSON files
    """
    
    _instance: Optional['ScreenEffectManager'] = None
    
    # Asset directories
    TEXTURES_DIR = "assets/effects/textures"
    PRESETS_DIR = "assets/effects/presets"
    
    def __init__(self):
        self._overlays: dict[int, 'EffectOverlay'] = {}  # screen_index -> overlay
        self._active_effects: dict[str, int] = {}  # effect_id -> screen_index
        self._effect_counter = 0
        self._enabled = True
        self._gaming_mode = False
        
        # Texture cache (path -> QPixmap)
        self._texture_cache: dict[str, Any] = {}
        
        # Ensure asset directories exist
        self._ensure_asset_dirs()
        
        # Load custom presets
        self._custom_presets: dict[str, EffectConfig] = {}
        self._load_custom_presets()
        
        # Callbacks
        self._on_effect_start: list[Callable[[str, EffectConfig], None]] = []
        self._on_effect_end: list[Callable[[str], None]] = []
    
    def _ensure_asset_dirs(self):
        """Create asset directories if they don't exist."""
        from pathlib import Path
        Path(self.TEXTURES_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.PRESETS_DIR).mkdir(parents=True, exist_ok=True)
    
    def _load_custom_presets(self):
        """Load custom effect presets from JSON files."""
        import json
        from pathlib import Path
        
        presets_path = Path(self.PRESETS_DIR)
        if not presets_path.exists():
            return
        
        for preset_file in presets_path.glob("*.json"):
            try:
                with open(preset_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to EffectConfig
                config = EffectConfig(
                    effect_type=EffectType[data.get('effect_type', 'PARTICLES')],
                    duration=data.get('duration', 3.0),
                    spawn_rate=data.get('spawn_rate', 20.0),
                    particle_count=data.get('particle_count', 50),
                    particle_size=tuple(data.get('particle_size', [4.0, 12.0])),
                    particle_speed=tuple(data.get('particle_speed', [50.0, 150.0])),
                    particle_lifetime=tuple(data.get('particle_lifetime', [0.5, 2.0])),
                    colors=data.get('colors', ["#ffffff"]),
                    gravity=data.get('gravity', 0.0),
                    wind=data.get('wind', 0.0),
                    friction=data.get('friction', 0.98),
                    glow=data.get('glow', True),
                    glow_intensity=data.get('glow_intensity', 0.5),
                    fade_out=data.get('fade_out', True),
                    shape=data.get('shape', 'circle'),
                    texture=data.get('texture', ''),
                    texture_tint=data.get('texture_tint', True),
                    direction=data.get('direction', 0.0),
                    spread=data.get('spread', 360.0),
                )
                
                preset_name = preset_file.stem
                self._custom_presets[preset_name] = config
                logger.info(f"Loaded custom effect preset: {preset_name}")
                
            except Exception as e:
                logger.warning(f"Failed to load preset {preset_file}: {e}")
    
    def get_texture(self, texture_path: str) -> Optional[Any]:
        """
        Load and cache a texture image.
        
        Args:
            texture_path: Filename in assets/effects/textures/ or absolute path
        
        Returns:
            QPixmap or None if not found
        """
        if not HAS_QT:
            return None
        
        if not texture_path:
            return None
        
        # Check cache
        if texture_path in self._texture_cache:
            return self._texture_cache[texture_path]
        
        # Resolve path
        from pathlib import Path
        
        if Path(texture_path).is_absolute():
            full_path = Path(texture_path)
        else:
            full_path = Path(self.TEXTURES_DIR) / texture_path
        
        if not full_path.exists():
            logger.warning(f"Texture not found: {full_path}")
            return None
        
        try:
            pixmap = QPixmap(str(full_path))
            if not pixmap.isNull():
                self._texture_cache[texture_path] = pixmap
                return pixmap
        except Exception as e:
            logger.error(f"Failed to load texture {texture_path}: {e}")
        
        return None
    
    def list_textures(self) -> list[str]:
        """List available texture files."""
        from pathlib import Path
        textures_path = Path(self.TEXTURES_DIR)
        if not textures_path.exists():
            return []
        
        extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        return [f.name for f in textures_path.iterdir() 
                if f.is_file() and f.suffix.lower() in extensions]
    
    def list_presets(self) -> list[str]:
        """List all available presets (built-in + custom)."""
        builtin = list(EFFECT_PRESETS.keys())
        custom = list(self._custom_presets.keys())
        return builtin + custom
    
    def save_preset(self, name: str, config: EffectConfig) -> bool:
        """
        Save a custom effect preset to JSON.
        
        Args:
            name: Preset name (will be saved as {name}.json)
            config: Effect configuration to save
        
        Returns:
            True if saved successfully
        """
        import json
        from pathlib import Path
        
        try:
            data = {
                'effect_type': config.effect_type.name,
                'duration': config.duration,
                'spawn_rate': config.spawn_rate,
                'particle_count': config.particle_count,
                'particle_size': list(config.particle_size),
                'particle_speed': list(config.particle_speed),
                'particle_lifetime': list(config.particle_lifetime),
                'colors': config.colors,
                'gravity': config.gravity,
                'wind': config.wind,
                'friction': config.friction,
                'glow': config.glow,
                'glow_intensity': config.glow_intensity,
                'fade_out': config.fade_out,
                'shape': config.shape,
                'texture': config.texture,
                'texture_tint': config.texture_tint,
                'direction': config.direction,
                'spread': config.spread,
            }
            
            preset_file = Path(self.PRESETS_DIR) / f"{name}.json"
            with open(preset_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self._custom_presets[name] = config
            logger.info(f"Saved custom preset: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save preset {name}: {e}")
            return False
    
    @classmethod
    def instance(cls) -> 'ScreenEffectManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _get_overlay(self, screen_index: int = 0) -> Optional['EffectOverlay']:
        """Get or create overlay for screen."""
        if not HAS_QT:
            return None
        
        if screen_index not in self._overlays:
            overlay = EffectOverlay(screen_index=screen_index, manager=self)
            overlay.effect_finished.connect(self._on_effect_finished)
            overlay.set_gaming_mode(self._gaming_mode)
            self._overlays[screen_index] = overlay
            
            # Register with fullscreen controller for visibility management
            try:
                from ..core.fullscreen_mode import get_fullscreen_controller
                controller = get_fullscreen_controller()
                controller.register_element(f'effect_overlay_{screen_index}', overlay, category='effects')
            except Exception:
                pass
        
        return self._overlays[screen_index]
    
    def _generate_id(self) -> str:
        """Generate unique effect ID."""
        self._effect_counter += 1
        return f"effect_{self._effect_counter}_{int(time.time() * 1000) % 10000}"
    
    def _on_effect_finished(self, effect_id: str):
        """Handle effect completion."""
        if effect_id in self._active_effects:
            del self._active_effects[effect_id]
        
        for callback in self._on_effect_end:
            try:
                callback(effect_id)
            except Exception as e:
                logger.error(f"Effect end callback error: {e}")
    
    def _clone_config(self, base: EffectConfig) -> EffectConfig:
        """Create a copy of an EffectConfig."""
        return EffectConfig(
            effect_type=base.effect_type,
            x=base.x, y=base.y,
            width=base.width, height=base.height,
            duration=base.duration,
            spawn_rate=base.spawn_rate,
            particle_count=base.particle_count,
            particle_size=base.particle_size,
            particle_speed=base.particle_speed,
            particle_lifetime=base.particle_lifetime,
            colors=list(base.colors),
            gravity=base.gravity,
            wind=base.wind,
            friction=base.friction,
            glow=base.glow,
            glow_intensity=base.glow_intensity,
            fade_out=base.fade_out,
            shape=base.shape,
            texture=base.texture,
            texture_tint=base.texture_tint,
            direction=base.direction,
            spread=base.spread,
            follow_cursor=base.follow_cursor,
            respect_bounds=base.respect_bounds,
        )
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    def spawn(
        self,
        preset: str = "sparkle",
        x: float = None,
        y: float = None,
        duration: float = None,
        screen: int = 0,
        custom_config: EffectConfig = None,
        **kwargs
    ) -> str:
        """
        Spawn a visual effect.
        
        Args:
            preset: Effect preset name (sparkle, fire, explosion, etc.) or custom preset
            x: X position (default: center of screen)
            y: Y position (default: center of screen)
            duration: Override effect duration
            screen: Screen index for multi-monitor
            custom_config: Full custom EffectConfig
            **kwargs: Override individual config values (texture, colors, gravity, etc.)
        
        Returns:
            Effect ID for tracking/removal
        """
        if not self._enabled:
            return ""
        
        # Get or create config
        if custom_config:
            config = custom_config
        elif preset in EFFECT_PRESETS:
            # Copy built-in preset config
            base = EFFECT_PRESETS[preset]
            config = self._clone_config(base)
        elif preset in self._custom_presets:
            # Copy custom preset config
            base = self._custom_presets[preset]
            config = self._clone_config(base)
        else:
            # Unknown preset - use sparkle as fallback
            logger.warning(f"Unknown effect preset: {preset}, using sparkle")
            return self.spawn("sparkle", x=x, y=y, duration=duration, screen=screen)
        
        # Apply position
        if x is not None:
            config.x = x - config.width / 2
        elif config.x == 0:
            # Default to screen center
            overlay = self._get_overlay(screen)
            if overlay:
                config.x = overlay.width() / 2 - config.width / 2
        
        if y is not None:
            config.y = y - config.height / 2
        elif config.y == 0:
            overlay = self._get_overlay(screen)
            if overlay:
                config.y = overlay.height() / 2 - config.height / 2
        
        # Apply duration override
        if duration is not None:
            config.duration = duration
        
        # Apply any extra kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Get overlay and spawn effect
        overlay = self._get_overlay(screen)
        if not overlay:
            return ""
        
        effect_id = self._generate_id()
        overlay.add_effect(effect_id, config)
        self._active_effects[effect_id] = screen
        
        # Notify callbacks
        for callback in self._on_effect_start:
            try:
                callback(effect_id, config)
            except Exception as e:
                logger.error(f"Effect start callback error: {e}")
        
        logger.debug(f"Spawned effect: {effect_id} ({preset}) at ({config.x}, {config.y})")
        return effect_id
    
    def spawn_at_avatar(
        self,
        preset: str = "sparkle",
        offset_x: float = 0,
        offset_y: float = 0,
        **kwargs
    ) -> str:
        """Spawn effect at avatar's current position."""
        try:
            from .persistence import load_avatar_state
            state = load_avatar_state()
            if state:
                x = state.get('x', 500) + offset_x
                y = state.get('y', 300) + offset_y
                return self.spawn(preset, x=x, y=y, **kwargs)
        except Exception:
            pass
        
        # Fallback to center
        return self.spawn(preset, **kwargs)
    
    def stop(self, effect_id: str):
        """Stop an effect (lets particles finish, stops spawning)."""
        if effect_id in self._active_effects:
            screen = self._active_effects[effect_id]
            if screen in self._overlays:
                self._overlays[screen].remove_effect(effect_id)
    
    def clear_all(self):
        """Remove all effects immediately."""
        for overlay in self._overlays.values():
            overlay.clear_all()
        self._active_effects.clear()
    
    def set_enabled(self, enabled: bool):
        """Enable or disable all effects."""
        self._enabled = enabled
        if not enabled:
            self.clear_all()
    
    def set_gaming_mode(self, enabled: bool):
        """Set gaming mode (hides all overlays)."""
        self._gaming_mode = enabled
        for overlay in self._overlays.values():
            overlay.set_gaming_mode(enabled)
    
    def get_active_effects(self) -> list[str]:
        """Get list of active effect IDs."""
        return list(self._active_effects.keys())
    
    def get_available_presets(self) -> list[str]:
        """Get list of available effect presets."""
        return list(EFFECT_PRESETS.keys())
    
    def on_effect_start(self, callback: Callable[[str, EffectConfig], None]):
        """Register callback for effect start."""
        self._on_effect_start.append(callback)
    
    def on_effect_end(self, callback: Callable[[str], None]):
        """Register callback for effect end."""
        self._on_effect_end.append(callback)


# =============================================================================
# MODULE-LEVEL API
# =============================================================================

def get_effect_manager() -> ScreenEffectManager:
    """Get the global screen effect manager instance."""
    return ScreenEffectManager.instance()


def spawn_effect(
    preset: str = "sparkle",
    x: float = None,
    y: float = None,
    duration: float = None,
    **kwargs
) -> str:
    """
    Spawn a visual effect on screen.
    
    Simple API for quick effect spawning:
        spawn_effect("fire", x=500, y=300)
        spawn_effect("explosion", x=800, y=400, duration=2.0)
        spawn_effect("hearts", x=600, y=200, colors=["#ff69b4", "#ff1493"])
    
    Available presets: sparkle, fire, snow, rain, explosion, confetti,
                       hearts, magic, smoke, bubble, lightning, ripple
    
    Args:
        preset: Effect preset name
        x: X position (default: center)
        y: Y position (default: center)
        duration: Effect duration in seconds
        **kwargs: Override config values (colors, gravity, etc.)
    
    Returns:
        Effect ID
    """
    return get_effect_manager().spawn(preset, x=x, y=y, duration=duration, **kwargs)


def stop_effect(effect_id: str):
    """Stop a specific effect."""
    get_effect_manager().stop(effect_id)


def clear_effects():
    """Clear all effects from screen."""
    get_effect_manager().clear_all()


# =============================================================================
# GAMING MODE INTEGRATION
# =============================================================================

def setup_gaming_mode_integration():
    """
    Hook screen effects into gaming mode.
    Auto-hides effects when a game is detected.
    Call this once during application startup.
    """
    try:
        from ..core.gaming_mode import get_gaming_mode
        
        gaming = get_gaming_mode()
        manager = get_effect_manager()
        
        def on_game_start(game_name: str, profile):
            """Called when a game is detected."""
            logger.info(f"[ScreenEffects] Hiding effects for gaming: {game_name}")
            manager.set_gaming_mode(True)
        
        def on_game_end(game_name: str):
            """Called when game ends."""
            logger.info(f"[ScreenEffects] Restoring effects: {game_name} ended")
            manager.set_gaming_mode(False)
        
        gaming.on_game_start(on_game_start)
        gaming.on_game_end(on_game_end)
        
        # If gaming mode is already active, sync state
        status = gaming.get_status()
        if status.get('active_game'):
            manager.set_gaming_mode(True)
        
        logger.info("[ScreenEffects] Gaming mode integration enabled")
        return True
        
    except Exception as e:
        logger.warning(f"[ScreenEffects] Gaming mode integration failed: {e}")
        return False


# =============================================================================
# MULTI-MONITOR HELPERS
# =============================================================================

def spawn_effect_on_screen(
    preset: str,
    screen_index: int = 0,
    x: float = None,
    y: float = None,
    **kwargs
) -> str:
    """
    Spawn effect on a specific monitor.
    
    Args:
        preset: Effect preset name
        screen_index: Monitor index (0 = primary)
        x, y: Position relative to that screen
        **kwargs: Additional effect config
    
    Returns:
        Effect ID
    """
    return get_effect_manager().spawn(preset, x=x, y=y, screen=screen_index, **kwargs)


def spawn_effect_all_screens(
    preset: str,
    duration: float = 3.0,
    **kwargs
) -> list[str]:
    """
    Spawn the same effect on all screens.
    
    Returns:
        List of effect IDs (one per screen)
    """
    if not HAS_QT:
        return []
    
    app = QApplication.instance()
    if not app:
        return []
    
    effect_ids = []
    for i, screen in enumerate(app.screens()):
        effect_id = get_effect_manager().spawn(
            preset, 
            screen=i, 
            duration=duration,
            **kwargs
        )
        if effect_id:
            effect_ids.append(effect_id)
    
    return effect_ids


def get_screen_count() -> int:
    """Get number of available screens."""
    if not HAS_QT:
        return 1
    
    app = QApplication.instance()
    return len(app.screens()) if app else 1


def get_screen_geometry(screen_index: int = 0) -> tuple[int, int, int, int]:
    """
    Get geometry of a screen.
    
    Returns:
        (x, y, width, height) tuple
    """
    if not HAS_QT:
        return (0, 0, 1920, 1080)
    
    app = QApplication.instance()
    if not app:
        return (0, 0, 1920, 1080)
    
    screens = app.screens()
    if screen_index >= len(screens):
        screen_index = 0
    
    if screens:
        geo = screens[screen_index].geometry()
        return (geo.x(), geo.y(), geo.width(), geo.height())
    
    return (0, 0, 1920, 1080)
