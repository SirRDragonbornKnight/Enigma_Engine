"""
Unified Avatar System

Combines multiple avatar modes into one AI-controlled companion:
- 3D Skeletal (Cortana style) - glTF/GLB with animations
- 2D Animated (Desktop Mate style) - GIF/sprite sheets  
- Simple PNG (DougDoug style) - static image with bounce/reactions
- Non-human support - animals, robots, fantasy creatures

The AI controls the avatar through a unified interface regardless of mode.

Usage:
    from enigma_engine.avatar.unified_avatar import UnifiedAvatar, AvatarMode
    
    # Create avatar
    avatar = UnifiedAvatar()
    
    # Choose your mode
    avatar.set_mode(AvatarMode.PNG_BOUNCE)  # DougDoug style
    avatar.load("my_character.png")
    
    # Or 3D mode
    avatar.set_mode(AvatarMode.SKELETAL_3D)
    avatar.load("character.glb")
    
    # Or 2D animated
    avatar.set_mode(AvatarMode.ANIMATED_2D)
    avatar.load_animations({
        "idle": "idle.gif",
        "talk": "talk.gif",
    })
    
    # AI controls (same for all modes)
    avatar.start_talking()
    avatar.set_emotion("happy")
    avatar.gesture("wave")
    
    # Get widget for display
    widget = avatar.get_widget()
"""

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

try:
    from PyQt5.QtCore import (
        QObject,
        Qt,
        QTimer,
        pyqtSignal,
    )
    from PyQt5.QtGui import QPainter, QPixmap, QTransform
    from PyQt5.QtWidgets import QLabel, QStackedWidget, QWidget
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QObject = object
    pyqtSignal = lambda *args: None


class AvatarMode(Enum):
    """Available avatar display modes."""
    PNG_BOUNCE = auto()      # Simple PNG with bounce (DougDoug style)
    ANIMATED_2D = auto()     # GIF/sprite sheet animation (Desktop Mate style)
    SKELETAL_3D = auto()     # 3D model with skeletal animation (Cortana style)


class AvatarType(Enum):
    """Type of avatar for appropriate behavior mapping."""
    HUMAN = auto()           # Humanoid - standard emotions
    ANIMAL = auto()          # Animal - ears, tail expressions
    ROBOT = auto()           # Robot - lights, mechanical movements
    FANTASY = auto()         # Fantasy creature - wings, magic effects
    ABSTRACT = auto()        # Abstract/geometric - color/shape changes
    CUSTOM = auto()          # User-defined mappings


@dataclass
class AvatarConfig:
    """Configuration for avatar behavior."""
    mode: AvatarMode = AvatarMode.PNG_BOUNCE
    avatar_type: AvatarType = AvatarType.HUMAN
    
    # PNG Bounce settings (DougDoug style)
    bounce_enabled: bool = True
    bounce_amplitude: int = 10          # Pixels to bounce
    bounce_speed: float = 2.0           # Bounces per second
    talk_bounce_multiplier: float = 1.5  # Faster bounce when talking
    reaction_squash: bool = True        # Squash/stretch on reactions
    
    # Size
    width: int = 256
    height: int = 256
    
    # Behavior
    idle_fidget: bool = True            # Small random movements when idle
    look_at_cursor: bool = False        # Eyes/head follow mouse
    breathing: bool = True              # Subtle breathing animation


class EmotionMapping:
    """
    Maps emotions to avatar-appropriate responses.
    Different avatar types express emotions differently.
    """
    
    # Human emotions → standard expressions
    HUMAN = {
        "neutral": {"expression": "neutral", "posture": "relaxed"},
        "happy": {"expression": "smile", "posture": "upright", "gesture": "none"},
        "sad": {"expression": "frown", "posture": "slouched"},
        "surprised": {"expression": "wide_eyes", "posture": "lean_back"},
        "thinking": {"expression": "thoughtful", "posture": "hand_on_chin"},
        "angry": {"expression": "frown", "posture": "tense"},
        "excited": {"expression": "big_smile", "posture": "bouncy"},
    }
    
    # Animal emotions → ear/tail/body language
    ANIMAL = {
        "neutral": {"ears": "relaxed", "tail": "still"},
        "happy": {"ears": "perked", "tail": "wagging", "bounce": True},
        "sad": {"ears": "drooped", "tail": "tucked"},
        "surprised": {"ears": "alert", "tail": "puffed"},
        "thinking": {"ears": "tilted", "tail": "slow_swish"},
        "angry": {"ears": "flat", "tail": "bristled"},
        "excited": {"ears": "perked", "tail": "fast_wag", "bounce": True},
    }
    
    # Robot emotions → lights/sounds/mechanical
    ROBOT = {
        "neutral": {"lights": "blue", "movement": "idle_hum"},
        "happy": {"lights": "green", "movement": "head_tilt"},
        "sad": {"lights": "dim_blue", "movement": "droop"},
        "surprised": {"lights": "yellow_flash", "movement": "jerk_back"},
        "thinking": {"lights": "pulsing_blue", "movement": "spin_thinking"},
        "angry": {"lights": "red", "movement": "shake"},
        "excited": {"lights": "rainbow", "movement": "spin_happy"},
    }
    
    # Fantasy emotions → magical effects
    FANTASY = {
        "neutral": {"aura": "soft_glow", "particles": "none"},
        "happy": {"aura": "bright", "particles": "sparkles"},
        "sad": {"aura": "dim", "particles": "rain"},
        "surprised": {"aura": "flash", "particles": "stars"},
        "thinking": {"aura": "pulsing", "particles": "symbols"},
        "angry": {"aura": "fire", "particles": "embers"},
        "excited": {"aura": "rainbow", "particles": "fireworks"},
    }
    
    @classmethod
    def get_mapping(cls, avatar_type: AvatarType) -> dict:
        """Get emotion mapping for avatar type."""
        mappings = {
            AvatarType.HUMAN: cls.HUMAN,
            AvatarType.ANIMAL: cls.ANIMAL,
            AvatarType.ROBOT: cls.ROBOT,
            AvatarType.FANTASY: cls.FANTASY,
            AvatarType.ABSTRACT: cls.HUMAN,  # Fallback
            AvatarType.CUSTOM: cls.HUMAN,    # Fallback
        }
        return mappings.get(avatar_type, cls.HUMAN)


class PNGBounceWidget(QWidget):
    """
    Simple PNG avatar with bounce animation (DougDoug style).
    
    Features:
    - Idle bounce (sine wave up/down)
    - Faster bounce when talking
    - Squash/stretch on reactions
    - Optional fidget movements
    """
    
    def __init__(self, config: AvatarConfig, parent=None):
        super().__init__(parent)
        self.config = config
        
        self._pixmap: Optional[QPixmap] = None
        self._base_pixmap: Optional[QPixmap] = None
        
        # Animation state
        self._bounce_phase: float = 0.0
        self._is_talking: bool = False
        self._current_emotion: str = "neutral"
        self._squash_factor: float = 1.0
        self._stretch_factor: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._target_squash: float = 1.0
        self._target_stretch: float = 1.0
        
        # Fidget
        self._fidget_timer: float = 0.0
        self._fidget_offset_x: float = 0.0
        self._fidget_offset_y: float = 0.0
        
        # Setup
        self.setFixedSize(config.width, config.height)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_animation)
        self._timer.start(16)  # ~60 FPS
    
    def load_image(self, path: str) -> bool:
        """Load PNG image."""
        try:
            self._base_pixmap = QPixmap(path)
            if self._base_pixmap.isNull():
                return False
            
            # Scale to fit
            self._base_pixmap = self._base_pixmap.scaled(
                self.config.width, self.config.height,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._pixmap = self._base_pixmap
            self.update()
            return True
        except Exception as e:
            print(f"[PNG Avatar] Failed to load: {e}")
            return False
    
    def set_talking(self, talking: bool):
        """Set talking state."""
        self._is_talking = talking
    
    def set_emotion(self, emotion: str):
        """Set emotion with squash/stretch reaction."""
        if emotion != self._current_emotion:
            self._current_emotion = emotion
            
            # Trigger squash/stretch reaction
            if self.config.reaction_squash:
                if emotion in ("surprised", "excited"):
                    self._target_squash = 0.8
                    self._target_stretch = 1.3
                elif emotion in ("sad", "thinking"):
                    self._target_squash = 1.1
                    self._target_stretch = 0.9
                elif emotion == "happy":
                    self._target_squash = 0.9
                    self._target_stretch = 1.1
                else:
                    self._target_squash = 1.0
                    self._target_stretch = 1.0
    
    def trigger_reaction(self, intensity: float = 1.0):
        """Trigger a bounce reaction (for emphasis)."""
        self._target_squash = 0.7 * intensity
        self._target_stretch = 1.4 * intensity
    
    def _update_animation(self):
        """Update animation frame."""
        dt = 0.016  # ~60 FPS
        
        # Update bounce phase
        speed = self.config.bounce_speed
        if self._is_talking:
            speed *= self.config.talk_bounce_multiplier
        self._bounce_phase += dt * speed * 2 * math.pi
        
        # Calculate bounce offset
        if self.config.bounce_enabled:
            amplitude = self.config.bounce_amplitude
            if self._is_talking:
                amplitude *= 1.3
            self._offset_y = math.sin(self._bounce_phase) * amplitude
        
        # Smooth squash/stretch
        lerp_speed = 8.0 * dt
        self._squash_factor += (self._target_squash - self._squash_factor) * lerp_speed
        self._stretch_factor += (self._target_stretch - self._stretch_factor) * lerp_speed
        
        # Gradually return to normal
        self._target_squash += (1.0 - self._target_squash) * 2.0 * dt
        self._target_stretch += (1.0 - self._target_stretch) * 2.0 * dt
        
        # Idle fidget - smooth oscillation instead of random jumps
        if self.config.idle_fidget and not self._is_talking:
            self._fidget_timer += dt
            # Smooth sine-based fidget (gentle swaying)
            fidget_speed = 0.3  # Slow, natural movement
            self._fidget_offset_x = math.sin(self._fidget_timer * fidget_speed) * 3
            self._fidget_offset_y = math.sin(self._fidget_timer * fidget_speed * 1.3) * 2
        
        # Smooth fidget return
        self._fidget_offset_x *= 0.95
        self._fidget_offset_y *= 0.95
        
        self.update()
    
    def paintEvent(self, event):
        """Render the avatar."""
        if not self._base_pixmap:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Calculate transform
        w = self._base_pixmap.width()
        h = self._base_pixmap.height()
        
        # Center position with offsets
        x = (self.width() - w) / 2 + self._offset_x + self._fidget_offset_x
        y = (self.height() - h) / 2 + self._offset_y + self._fidget_offset_y
        
        # Apply squash/stretch
        if abs(self._squash_factor - 1.0) > 0.01 or abs(self._stretch_factor - 1.0) > 0.01:
            transform = QTransform()
            # Move to center, scale, move back
            cx = x + w / 2
            cy = y + h
            transform.translate(cx, cy)
            transform.scale(self._squash_factor, self._stretch_factor)
            transform.translate(-cx, -cy)
            painter.setTransform(transform)
        
        painter.drawPixmap(int(x), int(y), self._base_pixmap)


class UnifiedAvatar(QObject):
    """
    Unified avatar system that supports multiple display modes.
    
    Modes:
    - PNG_BOUNCE: Simple PNG with bounce animation (DougDoug style)
    - ANIMATED_2D: GIF/sprite sheet animations (Desktop Mate style)
    - SKELETAL_3D: 3D model with skeletal animation (Cortana style)
    
    All modes are controlled through the same AI interface.
    """
    
    # Signals
    state_changed = pyqtSignal(str)
    emotion_changed = pyqtSignal(str)
    talking_changed = pyqtSignal(bool)
    
    def __init__(self, config: Optional[AvatarConfig] = None):
        super().__init__()
        
        self.config = config or AvatarConfig()
        
        # Widgets for each mode
        self._png_widget: Optional[PNGBounceWidget] = None
        self._2d_widget = None  # AvatarAnimator widget
        self._3d_widget = None  # NativeAvatar3D widget
        
        # Container widget
        self._container: Optional[QStackedWidget] = None
        
        # State
        self._current_mode = self.config.mode
        self._is_talking = False
        self._current_emotion = "neutral"
        self._animation_mappings: dict[str, str] = {}
        
        # Emotion mapping for avatar type
        self._emotion_map = EmotionMapping.get_mapping(self.config.avatar_type)
    
    def set_mode(self, mode: AvatarMode):
        """Switch avatar display mode."""
        self._current_mode = mode
        self.config.mode = mode
        
        if self._container:
            index = {
                AvatarMode.PNG_BOUNCE: 0,
                AvatarMode.ANIMATED_2D: 1,
                AvatarMode.SKELETAL_3D: 2,
            }.get(mode, 0)
            self._container.setCurrentIndex(index)
    
    def set_avatar_type(self, avatar_type: AvatarType):
        """Set avatar type for appropriate emotion mapping."""
        self.config.avatar_type = avatar_type
        self._emotion_map = EmotionMapping.get_mapping(avatar_type)
    
    def get_widget(self) -> QWidget:
        """Get the display widget."""
        if not self._container:
            self._create_widgets()
        return self._container
    
    def _create_widgets(self):
        """Create all mode widgets."""
        self._container = QStackedWidget()
        self._container.setFixedSize(self.config.width, self.config.height)
        self._container.setAttribute(Qt.WA_TranslucentBackground)
        
        # PNG Bounce widget (index 0)
        self._png_widget = PNGBounceWidget(self.config)
        self._container.addWidget(self._png_widget)
        
        # 2D Animated widget (index 1)
        from .animation_system import AvatarAnimator
        self._2d_animator = AvatarAnimator()
        label_2d = QLabel()
        label_2d.setFixedSize(self.config.width, self.config.height)
        label_2d.setAttribute(Qt.WA_TranslucentBackground)
        self._2d_animator.frame_changed.connect(label_2d.setPixmap)
        self._container.addWidget(label_2d)
        
        # 3D widget (index 2)
        try:
            from .animation_3d_native import NativeAvatar3D
            self._3d_avatar = NativeAvatar3D()
            self._3d_widget = self._3d_avatar.get_widget(self.config.width, self.config.height)
            self._container.addWidget(self._3d_widget)
        except Exception as e:
            # Fallback placeholder
            placeholder = QLabel("3D not available")
            placeholder.setFixedSize(self.config.width, self.config.height)
            self._container.addWidget(placeholder)
        
        # Set initial mode
        self.set_mode(self._current_mode)
    
    # =========================================================================
    # LOADING
    # =========================================================================
    
    def load(self, path: str) -> bool:
        """
        Load avatar from file.
        
        For PNG_BOUNCE: path to PNG image
        For ANIMATED_2D: path to GIF or sprite sheet
        For SKELETAL_3D: path to GLB/glTF file
        """
        if not self._container:
            self._create_widgets()
        
        path = str(path)
        
        if self._current_mode == AvatarMode.PNG_BOUNCE:
            return self._png_widget.load_image(path)
        
        elif self._current_mode == AvatarMode.ANIMATED_2D:
            # Load as default idle animation
            return self._2d_animator.load_animation("idle", path)
        
        elif self._current_mode == AvatarMode.SKELETAL_3D:
            success = self._3d_avatar.load_model(path)
            if success:
                # Auto-map animations
                anims = self._3d_avatar.get_available_animations()
                self._auto_map_animations(anims)
            return success
        
        return False
    
    def load_animations(self, animations: dict[str, str]):
        """
        Load multiple animations (for ANIMATED_2D mode).
        
        Args:
            animations: Dict of name -> file path
                e.g. {"idle": "idle.gif", "talk": "talk.gif", "wave": "wave.gif"}
        """
        if not self._container:
            self._create_widgets()
        
        for name, path in animations.items():
            self._2d_animator.load_animation(name, path)
        
        self._auto_map_animations(list(animations.keys()))
    
    def _auto_map_animations(self, animation_names: list[str]):
        """Auto-map animation names to states."""
        name_lower = {n.lower(): n for n in animation_names}
        
        # Common mappings
        mappings = {
            "idle": ["idle", "stand", "default", "rest", "breathe"],
            "talk": ["talk", "talking", "speak", "speaking", "chat"],
            "happy": ["happy", "joy", "smile", "laugh", "cheerful"],
            "sad": ["sad", "unhappy", "cry", "frown", "depressed"],
            "surprised": ["surprised", "shock", "gasp", "amazed"],
            "thinking": ["think", "thinking", "ponder", "hmm", "curious"],
            "wave": ["wave", "hi", "hello", "greet", "bye"],
            "nod": ["nod", "yes", "agree", "affirmative"],
            "shake": ["shake", "no", "disagree", "negative"],
        }
        
        for state, keywords in mappings.items():
            for kw in keywords:
                if kw in name_lower:
                    self._animation_mappings[state] = name_lower[kw]
                    break
        
        print(f"[Avatar] Auto-mapped animations: {self._animation_mappings}")
    
    # =========================================================================
    # AI CONTROL INTERFACE
    # =========================================================================
    
    def start_talking(self):
        """Called when AI starts speaking."""
        self._is_talking = True
        self.talking_changed.emit(True)
        
        if self._current_mode == AvatarMode.PNG_BOUNCE:
            self._png_widget.set_talking(True)
        elif self._current_mode == AvatarMode.ANIMATED_2D:
            if "talk" in self._animation_mappings:
                from .animation_system import AnimationState
                self._2d_animator.set_state(AnimationState.TALKING)
        elif self._current_mode == AvatarMode.SKELETAL_3D:
            from .animation_3d_native import Animation3DState
            self._3d_avatar.set_state(Animation3DState.TALKING)
    
    def stop_talking(self):
        """Called when AI finishes speaking."""
        self._is_talking = False
        self.talking_changed.emit(False)
        
        if self._current_mode == AvatarMode.PNG_BOUNCE:
            self._png_widget.set_talking(False)
        else:
            # Return to emotion state
            self._apply_emotion()
    
    def set_emotion(self, emotion: str):
        """
        Set avatar emotion.
        
        Args:
            emotion: "neutral", "happy", "sad", "surprised", "thinking", "angry", "excited"
        """
        emotion = emotion.lower()
        if emotion == self._current_emotion:
            return
        
        self._current_emotion = emotion
        self.emotion_changed.emit(emotion)
        
        if not self._is_talking:
            self._apply_emotion()
    
    def _apply_emotion(self):
        """Apply current emotion to avatar."""
        emotion = self._current_emotion
        
        if self._current_mode == AvatarMode.PNG_BOUNCE:
            self._png_widget.set_emotion(emotion)
        
        elif self._current_mode == AvatarMode.ANIMATED_2D:
            from .animation_system import AnimationState
            state_map = {
                "neutral": AnimationState.IDLE,
                "happy": AnimationState.HAPPY,
                "sad": AnimationState.SAD,
                "surprised": AnimationState.SURPRISED,
                "thinking": AnimationState.THINKING,
            }
            state = state_map.get(emotion, AnimationState.IDLE)
            self._2d_animator.set_state(state)
        
        elif self._current_mode == AvatarMode.SKELETAL_3D:
            from .animation_3d_native import Animation3DState
            state_map = {
                "neutral": Animation3DState.IDLE,
                "happy": Animation3DState.HAPPY,
                "sad": Animation3DState.SAD,
                "surprised": Animation3DState.SURPRISED,
                "thinking": Animation3DState.THINKING,
            }
            state = state_map.get(emotion, Animation3DState.IDLE)
            self._3d_avatar.set_state(state)
    
    def gesture(self, gesture_name: str):
        """
        Play a one-shot gesture.
        
        Args:
            gesture_name: "wave", "nod", "shrug", etc.
        """
        if self._current_mode == AvatarMode.PNG_BOUNCE:
            # Trigger a bounce reaction
            self._png_widget.trigger_reaction(1.2)
        
        elif self._current_mode == AvatarMode.ANIMATED_2D:
            anim_name = self._animation_mappings.get(gesture_name, gesture_name)
            self._2d_animator.play_gesture(anim_name)
        
        elif self._current_mode == AvatarMode.SKELETAL_3D:
            anim_name = self._animation_mappings.get(gesture_name, gesture_name)
            self._3d_avatar.play_gesture(anim_name)
    
    def react(self, intensity: float = 1.0):
        """
        Trigger a reaction animation (for emphasis).
        
        Args:
            intensity: 0.0 to 2.0, how strong the reaction
        """
        if self._current_mode == AvatarMode.PNG_BOUNCE:
            self._png_widget.trigger_reaction(intensity)
        else:
            # For animated modes, could trigger a quick gesture
            pass
    
    def listen(self):
        """Put avatar in listening mode."""
        if self._current_mode == AvatarMode.SKELETAL_3D:
            from .animation_3d_native import Animation3DState
            self._3d_avatar.set_state(Animation3DState.LISTENING)
        elif self._current_mode == AvatarMode.ANIMATED_2D:
            from .animation_system import AnimationState
            self._2d_animator.set_state(AnimationState.LISTENING)
    
    def think(self):
        """Put avatar in thinking mode."""
        self.set_emotion("thinking")
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    def set_bounce_enabled(self, enabled: bool):
        """Enable/disable bounce animation (PNG mode)."""
        self.config.bounce_enabled = enabled
        if self._png_widget:
            self._png_widget.config.bounce_enabled = enabled
    
    def set_bounce_speed(self, speed: float):
        """Set bounce speed (bounces per second)."""
        self.config.bounce_speed = speed
        if self._png_widget:
            self._png_widget.config.bounce_speed = speed
    
    def set_bounce_amplitude(self, pixels: int):
        """Set bounce height in pixels."""
        self.config.bounce_amplitude = pixels
        if self._png_widget:
            self._png_widget.config.bounce_amplitude = pixels
    
    def set_fidget(self, enabled: bool):
        """Enable/disable idle fidget movements."""
        self.config.idle_fidget = enabled
        if self._png_widget:
            self._png_widget.config.idle_fidget = enabled
    
    def set_squash_stretch(self, enabled: bool):
        """Enable/disable squash/stretch reactions."""
        self.config.reaction_squash = enabled
        if self._png_widget:
            self._png_widget.config.reaction_squash = enabled


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_png_avatar(image_path: str, width: int = 256, height: int = 256) -> UnifiedAvatar:
    """Create a simple PNG bounce avatar (DougDoug style)."""
    config = AvatarConfig(
        mode=AvatarMode.PNG_BOUNCE,
        width=width,
        height=height,
    )
    avatar = UnifiedAvatar(config)
    avatar.load(image_path)
    return avatar


def create_2d_avatar(animations: dict[str, str], width: int = 256, height: int = 256) -> UnifiedAvatar:
    """Create a 2D animated avatar (Desktop Mate style)."""
    config = AvatarConfig(
        mode=AvatarMode.ANIMATED_2D,
        width=width,
        height=height,
    )
    avatar = UnifiedAvatar(config)
    avatar.load_animations(animations)
    return avatar


def create_3d_avatar(model_path: str, width: int = 512, height: int = 512) -> UnifiedAvatar:
    """Create a 3D skeletal avatar (Cortana style)."""
    config = AvatarConfig(
        mode=AvatarMode.SKELETAL_3D,
        width=width,
        height=height,
    )
    avatar = UnifiedAvatar(config)
    avatar.load(model_path)
    return avatar


def create_animal_avatar(path: str, mode: AvatarMode = AvatarMode.PNG_BOUNCE) -> UnifiedAvatar:
    """Create an animal avatar with appropriate emotion mappings."""
    config = AvatarConfig(
        mode=mode,
        avatar_type=AvatarType.ANIMAL,
    )
    avatar = UnifiedAvatar(config)
    avatar.load(path)
    return avatar


def create_robot_avatar(path: str, mode: AvatarMode = AvatarMode.SKELETAL_3D) -> UnifiedAvatar:
    """Create a robot avatar with appropriate emotion mappings."""
    config = AvatarConfig(
        mode=mode,
        avatar_type=AvatarType.ROBOT,
    )
    avatar = UnifiedAvatar(config)
    avatar.load(path)
    return avatar
