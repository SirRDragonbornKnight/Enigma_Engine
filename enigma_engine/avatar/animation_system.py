"""
Avatar Animation System

A proper animation system for AI-controlled avatars like Cortana or JARVIS.

Features:
- State machine for animation states (idle, talking, gesture, emotion)
- Sprite sheet and GIF animation support
- Direct AI control via signals (real-time, no file polling)
- Lip sync support
- Smooth transitions between states

Animation File Formats Supported:
- Sprite sheets (PNG with frames in a row or grid)
- Animated GIF/APNG
- Image sequences (frame_001.png, frame_002.png, etc.)

Usage:
    from enigma_engine.avatar.animation_system import AvatarAnimator, AnimationState
    
    animator = AvatarAnimator()
    animator.load_animation("idle", "data/avatar/animations/idle.gif")
    animator.load_animation("talking", "data/avatar/animations/talking.gif")
    animator.load_animation("wave", "data/avatar/animations/wave.gif")
    
    # Connect to display
    animator.frame_changed.connect(my_display.set_pixmap)
    
    # AI controls the state
    animator.set_state(AnimationState.TALKING)
    animator.set_state(AnimationState.IDLE)
    animator.play_gesture("wave")  # Plays once, returns to idle
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

try:
    from PyQt5.QtCore import QObject, QTimer, pyqtSignal
    from PyQt5.QtGui import QImage, QPixmap
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QObject = object
    pyqtSignal = lambda *args: None

# Try to import imageio for GIF support
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# Try PIL for additional format support
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class AnimationState(Enum):
    """Avatar animation states."""
    IDLE = auto()          # Default state - breathing, blinking
    TALKING = auto()       # Lip sync / talking animation
    LISTENING = auto()     # Attentive pose
    THINKING = auto()      # Processing / thinking
    HAPPY = auto()         # Positive emotion
    SAD = auto()           # Negative emotion
    SURPRISED = auto()     # Surprise reaction
    GESTURE = auto()       # Playing a one-shot gesture


@dataclass
class Animation:
    """A single animation (sequence of frames)."""
    name: str
    frames: list[QPixmap] = field(default_factory=list)
    frame_durations: list[int] = field(default_factory=list)  # ms per frame
    loop: bool = True
    default_fps: int = 12
    
    @property
    def frame_count(self) -> int:
        return len(self.frames)
    
    @property
    def total_duration(self) -> int:
        """Total animation duration in ms."""
        if self.frame_durations:
            return sum(self.frame_durations)
        return int(self.frame_count * (1000 / self.default_fps))


class AvatarAnimator(QObject):
    """
    AI-controlled avatar animation system.
    
    The AI can control:
    - Current state (idle, talking, emotion)
    - Trigger gestures (wave, nod, etc.)
    - Set emotion overlays
    
    Signals are emitted for the display to update.
    """
    
    # Signals
    frame_changed = pyqtSignal(QPixmap)  # Emitted when frame changes
    state_changed = pyqtSignal(str)       # Emitted when state changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Animation storage
        self._animations: dict[str, Animation] = {}
        self._state_animations: dict[AnimationState, str] = {}  # State -> animation name
        
        # Current state
        self._current_state = AnimationState.IDLE
        self._current_animation: Optional[Animation] = None
        self._current_frame = 0
        self._gesture_queue: list[str] = []
        self._return_state = AnimationState.IDLE  # State to return to after gesture
        
        # Playback
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)
        self._playing = False
        self._last_frame_time = 0
        
        # Callbacks for AI control
        self._on_animation_complete: Optional[Callable] = None
        
    # =========================================================================
    # LOADING ANIMATIONS
    # =========================================================================
    
    def load_animation(self, name: str, path: str, loop: bool = True, fps: int = 12) -> bool:
        """
        Load an animation from file.
        
        Supports:
        - GIF/APNG (animated)
        - Sprite sheet (single image with frames in a row)
        - Image sequence (folder with frame_001.png, etc.)
        
        Args:
            name: Animation name (e.g., "idle", "wave", "happy")
            path: Path to animation file or folder
            loop: Whether to loop the animation
            fps: Frames per second (for sprite sheets)
        
        Returns:
            True if loaded successfully
        """
        path = Path(path)
        
        if not path.exists():
            print(f"[Animator] Animation not found: {path}")
            return False
        
        animation = Animation(name=name, loop=loop, default_fps=fps)
        
        if path.is_dir():
            # Image sequence
            success = self._load_image_sequence(animation, path)
        elif path.suffix.lower() in ('.gif', '.apng', '.png'):
            if path.suffix.lower() == '.gif' or self._is_animated_png(path):
                success = self._load_animated_image(animation, path)
            else:
                # Assume sprite sheet
                success = self._load_sprite_sheet(animation, path)
        else:
            # Try as sprite sheet
            success = self._load_sprite_sheet(animation, path)
        
        if success and animation.frame_count > 0:
            self._animations[name] = animation
            print(f"[Animator] Loaded '{name}': {animation.frame_count} frames")
            return True
        
        print(f"[Animator] Failed to load '{name}' from {path}")
        return False
    
    def _is_animated_png(self, path: Path) -> bool:
        """Check if PNG is animated (APNG)."""
        try:
            with open(path, 'rb') as f:
                header = f.read(100)
                return b'acTL' in header  # APNG animation control chunk
        except Exception:
            return False
    
    def _load_animated_image(self, animation: Animation, path: Path) -> bool:
        """Load GIF or APNG."""
        if HAS_PIL:
            try:
                img = Image.open(path)
                
                # Get all frames
                frames = []
                durations = []
                
                try:
                    while True:
                        # Convert frame to QPixmap
                        frame = img.convert('RGBA')
                        data = frame.tobytes('raw', 'RGBA')
                        qimg = QImage(data, frame.width, frame.height, QImage.Format_RGBA8888)
                        pixmap = QPixmap.fromImage(qimg)
                        frames.append(pixmap)
                        
                        # Get frame duration (GIF stores in centiseconds)
                        duration = img.info.get('duration', 100)
                        durations.append(duration)
                        
                        img.seek(img.tell() + 1)
                except EOFError:
                    pass
                
                animation.frames = frames
                animation.frame_durations = durations
                return len(frames) > 0
                
            except Exception as e:
                print(f"[Animator] PIL error: {e}")
        
        if HAS_IMAGEIO:
            try:
                reader = imageio.get_reader(str(path))
                for frame_data in reader:
                    # Convert numpy array to QPixmap
                    h, w = frame_data.shape[:2]
                    if frame_data.shape[2] == 4:
                        fmt = QImage.Format_RGBA8888
                    else:
                        fmt = QImage.Format_RGB888
                    qimg = QImage(frame_data.data, w, h, fmt)
                    pixmap = QPixmap.fromImage(qimg)
                    animation.frames.append(pixmap)
                
                # Try to get duration from metadata
                meta = reader.get_meta_data()
                if 'duration' in meta:
                    animation.frame_durations = [meta['duration']] * len(animation.frames)
                
                return len(animation.frames) > 0
                
            except Exception as e:
                print(f"[Animator] imageio error: {e}")
        
        return False
    
    def _load_sprite_sheet(self, animation: Animation, path: Path, 
                           frame_width: Optional[int] = None,
                           frame_height: Optional[int] = None,
                           columns: Optional[int] = None) -> bool:
        """
        Load a sprite sheet (single image with frames arranged in grid).
        
        If frame dimensions not specified, assumes horizontal strip
        (all frames in one row, square frames).
        """
        try:
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                return False
            
            w, h = pixmap.width(), pixmap.height()
            
            # Auto-detect frame size if not specified
            if frame_width is None:
                # Assume horizontal strip with square frames
                frame_height = frame_height or h
                frame_width = frame_height  # Square
                columns = w // frame_width
            
            if columns is None:
                columns = w // frame_width
            
            rows = h // frame_height
            
            # Extract frames
            for row in range(rows):
                for col in range(columns):
                    x = col * frame_width
                    y = row * frame_height
                    frame = pixmap.copy(x, y, frame_width, frame_height)
                    if not frame.isNull():
                        animation.frames.append(frame)
            
            return len(animation.frames) > 0
            
        except Exception as e:
            print(f"[Animator] Sprite sheet error: {e}")
            return False
    
    def _load_image_sequence(self, animation: Animation, folder: Path) -> bool:
        """Load animation from numbered image files."""
        try:
            # Find all image files
            extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
            files = sorted([
                f for f in folder.iterdir()
                if f.suffix.lower() in extensions
            ])
            
            for file in files:
                pixmap = QPixmap(str(file))
                if not pixmap.isNull():
                    animation.frames.append(pixmap)
            
            return len(animation.frames) > 0
            
        except Exception as e:
            print(f"[Animator] Image sequence error: {e}")
            return False
    
    # =========================================================================
    # STATE MANAGEMENT (AI Control)
    # =========================================================================
    
    def map_state_to_animation(self, state: AnimationState, animation_name: str):
        """Map an animation state to a loaded animation."""
        if animation_name in self._animations:
            self._state_animations[state] = animation_name
    
    def set_state(self, state: AnimationState):
        """
        Set the current animation state.
        
        This is how the AI controls the avatar - by setting states like:
        - IDLE when not doing anything
        - TALKING when speaking
        - HAPPY, SAD, etc. for emotions
        """
        if state == self._current_state and state != AnimationState.GESTURE:
            return
        
        self._current_state = state
        self._current_frame = 0
        self.state_changed.emit(state.name)
        
        # Get animation for this state
        anim_name = self._state_animations.get(state)
        if anim_name and anim_name in self._animations:
            self._current_animation = self._animations[anim_name]
            self._start_playback()
        else:
            # Fallback to idle
            idle_name = self._state_animations.get(AnimationState.IDLE)
            if idle_name and idle_name in self._animations:
                self._current_animation = self._animations[idle_name]
                self._start_playback()
    
    def play_gesture(self, gesture_name: str, then_state: AnimationState = AnimationState.IDLE):
        """
        Play a one-shot gesture animation, then return to specified state.
        
        Args:
            gesture_name: Name of the gesture animation to play
            then_state: State to return to after gesture completes
        """
        if gesture_name not in self._animations:
            print(f"[Animator] Unknown gesture: {gesture_name}")
            return
        
        self._return_state = then_state
        self._current_animation = self._animations[gesture_name]
        self._current_frame = 0
        self._current_state = AnimationState.GESTURE
        self.state_changed.emit(f"GESTURE:{gesture_name}")
        self._start_playback()
    
    def queue_gesture(self, gesture_name: str):
        """Queue a gesture to play after current animation."""
        if gesture_name in self._animations:
            self._gesture_queue.append(gesture_name)
    
    # =========================================================================
    # PLAYBACK
    # =========================================================================
    
    def _start_playback(self):
        """Start or restart animation playback."""
        if not self._current_animation or self._current_animation.frame_count == 0:
            return
        
        self._playing = True
        self._emit_current_frame()
        
        # Calculate frame delay
        if self._current_animation.frame_durations:
            delay = self._current_animation.frame_durations[0]
        else:
            delay = int(1000 / self._current_animation.default_fps)
        
        self._timer.start(delay)
    
    def _advance_frame(self):
        """Advance to next frame."""
        if not self._current_animation:
            return
        
        self._current_frame += 1
        
        # Check if animation complete
        if self._current_frame >= self._current_animation.frame_count:
            if self._current_animation.loop and self._current_state != AnimationState.GESTURE:
                self._current_frame = 0
            else:
                # Animation complete
                self._current_frame = self._current_animation.frame_count - 1
                self._on_animation_finished()
                return
        
        self._emit_current_frame()
        
        # Update timer for variable frame rates
        if self._current_animation.frame_durations:
            delay = self._current_animation.frame_durations[self._current_frame]
            self._timer.setInterval(delay)
    
    def _emit_current_frame(self):
        """Emit the current frame."""
        if self._current_animation and self._current_frame < self._current_animation.frame_count:
            frame = self._current_animation.frames[self._current_frame]
            self.frame_changed.emit(frame)
    
    def _on_animation_finished(self):
        """Handle animation completion."""
        self._timer.stop()
        
        # Check gesture queue
        if self._gesture_queue:
            next_gesture = self._gesture_queue.pop(0)
            self.play_gesture(next_gesture, self._return_state)
        elif self._current_state == AnimationState.GESTURE:
            # Return to previous state
            self.set_state(self._return_state)
        
        # Callback
        if self._on_animation_complete:
            self._on_animation_complete(self._current_animation.name if self._current_animation else None)
    
    def stop(self):
        """Stop animation playback."""
        self._timer.stop()
        self._playing = False
    
    def pause(self):
        """Pause animation playback."""
        self._timer.stop()
    
    def resume(self):
        """Resume animation playback."""
        if self._current_animation:
            self._start_playback()
    
    # =========================================================================
    # AI INTERFACE
    # =========================================================================
    
    def on_animation_complete(self, callback: Callable):
        """Set callback for when animations complete."""
        self._on_animation_complete = callback
    
    def get_current_state(self) -> AnimationState:
        """Get current animation state."""
        return self._current_state
    
    def get_available_animations(self) -> list[str]:
        """Get list of loaded animation names."""
        return list(self._animations.keys())
    
    def get_available_gestures(self) -> list[str]:
        """Get list of animations that can be used as gestures."""
        # All non-looping animations or specifically marked ones
        return [name for name, anim in self._animations.items() if not anim.loop]


# =============================================================================
# AI AVATAR CONTROLLER
# =============================================================================

class AIAvatarController:
    """
    High-level controller for AI to manage avatar.
    
    This is the interface the AI uses to control the avatar.
    It translates AI intents into animation commands.
    
    Usage:
        controller = AIAvatarController(animator)
        
        # When AI starts speaking
        controller.start_talking()
        
        # When AI finishes speaking
        controller.stop_talking()
        
        # When AI detects emotion in its response
        controller.set_emotion("happy")
        
        # When AI wants to gesture
        controller.gesture("wave")
    """
    
    def __init__(self, animator: AvatarAnimator):
        self.animator = animator
        self._is_talking = False
        self._current_emotion = "neutral"
    
    def start_talking(self):
        """Called when AI starts speaking."""
        self._is_talking = True
        self.animator.set_state(AnimationState.TALKING)
    
    def stop_talking(self):
        """Called when AI finishes speaking."""
        self._is_talking = False
        # Return to emotion state or idle
        self._apply_emotion()
    
    def set_emotion(self, emotion: str):
        """
        Set avatar emotion based on AI's response.
        
        Args:
            emotion: One of "neutral", "happy", "sad", "surprised", "thinking"
        """
        self._current_emotion = emotion.lower()
        if not self._is_talking:
            self._apply_emotion()
    
    def _apply_emotion(self):
        """Apply current emotion as animation state."""
        emotion_map = {
            "neutral": AnimationState.IDLE,
            "happy": AnimationState.HAPPY,
            "sad": AnimationState.SAD,
            "surprised": AnimationState.SURPRISED,
            "thinking": AnimationState.THINKING,
        }
        state = emotion_map.get(self._current_emotion, AnimationState.IDLE)
        self.animator.set_state(state)
    
    def gesture(self, gesture_name: str):
        """
        Play a gesture animation.
        
        Args:
            gesture_name: Name of gesture (e.g., "wave", "nod", "shrug")
        """
        # After gesture, return to current emotion state
        emotion_map = {
            "neutral": AnimationState.IDLE,
            "happy": AnimationState.HAPPY,
            "sad": AnimationState.SAD,
        }
        return_state = emotion_map.get(self._current_emotion, AnimationState.IDLE)
        self.animator.play_gesture(gesture_name, return_state)
    
    def listen(self):
        """Put avatar in listening mode."""
        self.animator.set_state(AnimationState.LISTENING)
    
    def think(self):
        """Put avatar in thinking mode."""
        self.animator.set_state(AnimationState.THINKING)
