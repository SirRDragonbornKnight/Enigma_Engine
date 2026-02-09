"""
Detailed AI Avatar Controls for Enigma AI Engine

Provides fine-grained AI control over avatar behavior:
- Micro-expressions and facial details
- Gesture library with variations
- Emotional state machine
- Attention/gaze system
- Reactive behaviors
- Body language
- Personality expression

Usage:
    from enigma_engine.avatar.ai_controls import AIAvatarController
    
    ai_ctrl = AIAvatarController(avatar_controller)
    
    # Express emotion with intensity
    ai_ctrl.express_emotion("happy", intensity=0.8)
    
    # Make gesture
    ai_ctrl.gesture("nod", intensity=0.5)
    
    # Set attention target
    ai_ctrl.look_at_user()
    ai_ctrl.look_at_point(0.3, -0.1)  # Normalized screen coords
    
    # Enable reactive mode
    ai_ctrl.enable_reactions(True)
"""

import logging
import random
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)


class EmotionType(Enum):
    """Core emotion types for avatar expression."""
    NEUTRAL = auto()
    HAPPY = auto()
    SAD = auto()
    ANGRY = auto()
    SURPRISED = auto()
    FEARFUL = auto()
    DISGUSTED = auto()
    CONFUSED = auto()
    CURIOUS = auto()
    THINKING = auto()
    EXCITED = auto()
    TIRED = auto()
    SMUG = auto()
    EMBARRASSED = auto()


class GestureType(Enum):
    """Available gesture types."""
    # Head gestures
    NOD = auto()
    SHAKE = auto()
    TILT_LEFT = auto()
    TILT_RIGHT = auto()
    CHIN_UP = auto()
    CHIN_DOWN = auto()
    
    # Eye gestures  
    BLINK = auto()
    WINK_LEFT = auto()
    WINK_RIGHT = auto()
    SQUINT = auto()
    WIDE_EYES = auto()
    EYE_ROLL = auto()
    
    # Body gestures (if supported)
    LEAN_FORWARD = auto()
    LEAN_BACK = auto()
    SHRUG = auto()
    WAVE = auto()
    THUMBS_UP = auto()
    
    # Compound gestures
    THINKING_POSE = auto()      # Head tilt + look up
    DISMISSIVE = auto()         # Eye roll + head shake
    ACKNOWLEDGMENT = auto()     # Small nod + smile


@dataclass
class EmotionState:
    """Current emotional state with multiple emotion blend."""
    primary: EmotionType = EmotionType.NEUTRAL
    primary_intensity: float = 1.0
    secondary: Optional[EmotionType] = None
    secondary_intensity: float = 0.0
    decay_rate: float = 0.1  # How fast emotion fades
    
    def blend_value(self, emotion: EmotionType) -> float:
        """Get blended intensity for an emotion."""
        value = 0.0
        if self.primary == emotion:
            value += self.primary_intensity
        if self.secondary == emotion:
            value += self.secondary_intensity
        return min(1.0, value)


@dataclass 
class AttentionTarget:
    """Where the avatar is looking."""
    x: float = 0.0  # -1 to 1 (left to right)
    y: float = 0.0  # -1 to 1 (down to up)
    target_type: str = "center"  # center, user, point, random
    saccade_enabled: bool = True  # Small eye movements


@dataclass
class GestureConfig:
    """Configuration for a gesture."""
    duration: float
    keyframes: List[Dict[str, float]]  # Time-keyed parameter changes
    can_interrupt: bool = True
    priority: int = 50


@dataclass
class ReactionTrigger:
    """Trigger for reactive behavior."""
    event_type: str  # "user_speaks", "silence", "question", etc.
    response_gesture: GestureType
    response_emotion: Optional[EmotionType] = None
    probability: float = 1.0  # Chance to trigger
    cooldown: float = 2.0  # Seconds before can trigger again


class AIAvatarController:
    """
    High-level AI control interface for avatars.
    
    Provides semantic control (emotions, gestures, attention)
    rather than raw parameter manipulation.
    """
    
    def __init__(self, avatar_controller: Any = None, live2d_controller: Any = None):
        """
        Initialize AI avatar controller.
        
        Args:
            avatar_controller: 3D avatar controller (BoneController or AvatarController)
            live2d_controller: Live2D controller (optional)
        """
        self._avatar = avatar_controller
        self._live2d = live2d_controller
        self._lock = threading.Lock()
        
        # State
        self._emotion_state = EmotionState()
        self._attention = AttentionTarget()
        self._current_gesture: Optional[str] = None
        self._gesture_thread: Optional[threading.Thread] = None
        
        # Control settings
        self._detail_level = 1.0  # 0=minimal, 1=full detail
        self._reaction_enabled = True
        self._idle_behavior = True
        self._personality_variation = 0.2  # Random variation amount
        
        # Reaction system
        self._reactions: List[ReactionTrigger] = []
        self._last_reaction_time: Dict[str, float] = {}
        self._setup_default_reactions()
        
        # Animation loop
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._update_fps = 30
        
        # Idle behavior
        self._last_blink_time = time.time()
        self._blink_interval = (3.0, 7.0)  # Random interval range
        self._next_blink = random.uniform(*self._blink_interval)
        
        # Saccade (micro eye movements)
        self._saccade_offset = (0.0, 0.0)
        self._saccade_timer = 0.0
        self._saccade_interval = 0.3
        
        logger.info("AI Avatar Controller initialized")
    
    def _setup_default_reactions(self):
        """Set up default reactive behaviors."""
        self._reactions = [
            ReactionTrigger(
                event_type="user_speaks",
                response_gesture=GestureType.NOD,
                probability=0.3,
                cooldown=4.0
            ),
            ReactionTrigger(
                event_type="question",
                response_gesture=GestureType.TILT_RIGHT,
                response_emotion=EmotionType.CURIOUS,
                probability=0.5,
                cooldown=3.0
            ),
            ReactionTrigger(
                event_type="user_laughs",
                response_gesture=GestureType.NOD,
                response_emotion=EmotionType.HAPPY,
                probability=0.7,
                cooldown=2.0
            ),
            ReactionTrigger(
                event_type="silence_long",
                response_gesture=GestureType.THINKING_POSE,
                response_emotion=EmotionType.THINKING,
                probability=0.4,
                cooldown=10.0
            ),
        ]
    
    def start(self):
        """Start the update loop."""
        if self._running:
            return
        
        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
    
    def stop(self):
        """Stop the update loop."""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=1.0)
    
    def _update_loop(self):
        """Main update loop for animations and idle behavior."""
        frame_time = 1.0 / self._update_fps
        
        while self._running:
            start = time.perf_counter()
            
            # Update idle behaviors
            if self._idle_behavior and self._detail_level > 0.3:
                self._update_idle_behaviors()
            
            # Update attention/gaze
            self._update_attention()
            
            # Decay emotions
            self._decay_emotions(frame_time)
            
            elapsed = time.perf_counter() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
    
    def _update_idle_behaviors(self):
        """Handle idle animations like blinking."""
        now = time.time()
        
        # Blinking
        if now - self._last_blink_time >= self._next_blink:
            self.gesture(GestureType.BLINK, intensity=1.0, blocking=False)
            self._last_blink_time = now
            self._next_blink = random.uniform(*self._blink_interval)
        
        # Saccades (micro eye movements)
        if self._attention.saccade_enabled:
            self._saccade_timer += 1.0 / self._update_fps
            if self._saccade_timer >= self._saccade_interval:
                self._saccade_offset = (
                    random.gauss(0, 0.02),
                    random.gauss(0, 0.02)
                )
                self._saccade_timer = 0.0
                self._saccade_interval = random.uniform(0.2, 0.5)
    
    def _update_attention(self):
        """Update gaze direction."""
        # Base attention target
        x = self._attention.x
        y = self._attention.y
        
        # Add saccade
        x += self._saccade_offset[0]
        y += self._saccade_offset[1]
        
        # Clamp
        x = max(-1, min(1, x))
        y = max(-1, min(1, y))
        
        # Apply to avatar
        if self._live2d:
            self._live2d.look_at(x, y)
        
        if self._avatar and hasattr(self._avatar, 'set_eye_target'):
            self._avatar.set_eye_target(x, y)
    
    def _decay_emotions(self, dt: float):
        """Gradually return to neutral."""
        with self._lock:
            if self._emotion_state.primary != EmotionType.NEUTRAL:
                self._emotion_state.primary_intensity -= self._emotion_state.decay_rate * dt
                
                if self._emotion_state.primary_intensity <= 0:
                    self._emotion_state.primary = EmotionType.NEUTRAL
                    self._emotion_state.primary_intensity = 1.0
            
            if self._emotion_state.secondary:
                self._emotion_state.secondary_intensity -= self._emotion_state.decay_rate * dt
                
                if self._emotion_state.secondary_intensity <= 0:
                    self._emotion_state.secondary = None
                    self._emotion_state.secondary_intensity = 0.0
    
    # ===================
    # PUBLIC API
    # ===================
    
    def set_detail_level(self, level: float):
        """
        Set detail level for avatar control.
        
        Args:
            level: 0.0 (minimal) to 1.0 (full detail)
                  - 0.0-0.3: Basic expressions only
                  - 0.3-0.6: + gestures, simple idle
                  - 0.6-1.0: + micro-expressions, saccades, reactions
        """
        self._detail_level = max(0.0, min(1.0, level))
        self._attention.saccade_enabled = level > 0.5
        self._idle_behavior = level > 0.3
        logger.debug(f"Detail level set to {level}")
    
    def express_emotion(
        self,
        emotion: EmotionType | str,
        intensity: float = 0.8,
        duration: float = 3.0,
        blend: bool = True
    ):
        """
        Express an emotion.
        
        Args:
            emotion: Emotion type or name
            intensity: How strong (0.0-1.0)
            duration: How long before decay
            blend: If True, blend with current emotion
        """
        if isinstance(emotion, str):
            emotion = EmotionType[emotion.upper()]
        
        intensity *= self._detail_level
        
        with self._lock:
            if blend and self._emotion_state.primary != EmotionType.NEUTRAL:
                # Move current to secondary
                self._emotion_state.secondary = self._emotion_state.primary
                self._emotion_state.secondary_intensity = self._emotion_state.primary_intensity * 0.5
            
            self._emotion_state.primary = emotion
            self._emotion_state.primary_intensity = intensity
            self._emotion_state.decay_rate = intensity / duration
        
        # Apply to avatar
        self._apply_emotion(emotion, intensity)
        logger.debug(f"Expressing {emotion.name} at {intensity:.1%}")
    
    def _apply_emotion(self, emotion: EmotionType, intensity: float):
        """Apply emotion to avatar parameters."""
        # Emotion to parameter mapping
        emotion_params = {
            EmotionType.HAPPY: {
                "mouth_smile": 0.7,
                "eye_smile": 0.5,
                "brow_up": 0.2,
            },
            EmotionType.SAD: {
                "mouth_smile": -0.5,
                "eye_droop": 0.4,
                "brow_down": 0.3,
            },
            EmotionType.ANGRY: {
                "brow_down": 0.6,
                "eye_narrow": 0.4,
                "mouth_tight": 0.3,
            },
            EmotionType.SURPRISED: {
                "eye_wide": 0.8,
                "brow_up": 0.7,
                "mouth_open": 0.3,
            },
            EmotionType.CONFUSED: {
                "brow_furrow": 0.5,
                "head_tilt": 0.3,
                "eye_squint": 0.2,
            },
            EmotionType.CURIOUS: {
                "brow_up": 0.4,
                "eye_wide": 0.3,
                "head_tilt": 0.2,
                "lean_forward": 0.1,
            },
            EmotionType.THINKING: {
                "eye_look_up": 0.4,
                "head_tilt": 0.2,
                "mouth_purse": 0.2,
            },
            EmotionType.EXCITED: {
                "eye_wide": 0.5,
                "mouth_smile": 0.8,
                "brow_up": 0.4,
                "body_bounce": 0.3,
            },
        }
        
        params = emotion_params.get(emotion, {})
        
        # Apply with intensity scaling
        scaled_params = {k: v * intensity for k, v in params.items()}
        
        # Apply to Live2D
        if self._live2d:
            live2d_map = {
                "mouth_smile": "ParamMouthForm",
                "mouth_open": "ParamMouthOpenY",
                "eye_wide": "ParamEyeLOpen",  # Would also set right eye
                "brow_up": "ParamBrowLY",
            }
            for param, value in scaled_params.items():
                if param in live2d_map:
                    self._live2d.set_parameter(live2d_map[param], value)
        
        # Apply to 3D avatar
        if self._avatar:
            # Would map to blend shapes or bone rotations
            pass
    
    def gesture(
        self,
        gesture: GestureType | str,
        intensity: float = 0.7,
        speed: float = 1.0,
        blocking: bool = False
    ):
        """
        Perform a gesture.
        
        Args:
            gesture: Gesture type or name
            intensity: How pronounced (0.0-1.0)
            speed: Playback speed multiplier
            blocking: If True, wait for completion
        """
        if isinstance(gesture, str):
            gesture = GestureType[gesture.upper()]
        
        if self._detail_level < 0.3:
            # Only basic gestures at low detail
            if gesture not in [GestureType.NOD, GestureType.SHAKE, GestureType.BLINK]:
                return
        
        intensity *= self._detail_level
        
        # Get gesture config
        config = self._get_gesture_config(gesture)
        if not config:
            return
        
        def run_gesture():
            self._current_gesture = gesture.name
            
            duration = config.duration / speed
            frame_count = int(duration * self._update_fps)
            
            for i, keyframe in enumerate(config.keyframes):
                if not self._running and not blocking:
                    break
                    
                # Apply keyframe parameters
                for param, value in keyframe.items():
                    if param == "time":
                        continue
                    scaled = value * intensity
                    self._apply_gesture_param(param, scaled)
                
                # Wait for next keyframe
                if i < len(config.keyframes) - 1:
                    next_time = config.keyframes[i + 1].get("time", 0)
                    current_time = keyframe.get("time", 0)
                    wait = (next_time - current_time) / speed
                    time.sleep(wait)
            
            self._current_gesture = None
        
        if blocking:
            run_gesture()
        else:
            self._gesture_thread = threading.Thread(target=run_gesture, daemon=True)
            self._gesture_thread.start()
        
        logger.debug(f"Gesture: {gesture.name}")
    
    def _get_gesture_config(self, gesture: GestureType) -> Optional[GestureConfig]:
        """Get configuration for a gesture."""
        configs = {
            GestureType.NOD: GestureConfig(
                duration=0.6,
                keyframes=[
                    {"time": 0.0, "head_y": 0},
                    {"time": 0.15, "head_y": -10},
                    {"time": 0.3, "head_y": 5},
                    {"time": 0.45, "head_y": -5},
                    {"time": 0.6, "head_y": 0},
                ]
            ),
            GestureType.SHAKE: GestureConfig(
                duration=0.8,
                keyframes=[
                    {"time": 0.0, "head_z": 0},
                    {"time": 0.13, "head_z": -15},
                    {"time": 0.27, "head_z": 15},
                    {"time": 0.4, "head_z": -10},
                    {"time": 0.53, "head_z": 10},
                    {"time": 0.67, "head_z": -5},
                    {"time": 0.8, "head_z": 0},
                ]
            ),
            GestureType.BLINK: GestureConfig(
                duration=0.15,
                keyframes=[
                    {"time": 0.0, "eye_close": 0},
                    {"time": 0.05, "eye_close": 1},
                    {"time": 0.1, "eye_close": 0.2},
                    {"time": 0.15, "eye_close": 0},
                ]
            ),
            GestureType.TILT_RIGHT: GestureConfig(
                duration=0.5,
                keyframes=[
                    {"time": 0.0, "head_z": 0},
                    {"time": 0.25, "head_z": 15},
                    {"time": 0.5, "head_z": 10},
                ]
            ),
            GestureType.THINKING_POSE: GestureConfig(
                duration=1.0,
                keyframes=[
                    {"time": 0.0, "head_x": 0, "head_z": 0, "eye_y": 0},
                    {"time": 0.3, "head_x": 10, "head_z": 10, "eye_y": 0.3},
                    {"time": 0.6, "head_x": 15, "head_z": 12, "eye_y": 0.5},
                    {"time": 1.0, "head_x": 10, "head_z": 8, "eye_y": 0.4},
                ]
            ),
        }
        return configs.get(gesture)
    
    def _apply_gesture_param(self, param: str, value: float):
        """Apply a gesture parameter."""
        # Live2D mapping
        if self._live2d:
            live2d_map = {
                "head_x": "ParamAngleX",
                "head_y": "ParamAngleY", 
                "head_z": "ParamAngleZ",
                "eye_close": "ParamEyeLOpen",  # Inverse
                "eye_y": "ParamEyeBallY",
            }
            if param in live2d_map:
                if param == "eye_close":
                    value = 1.0 - value
                self._live2d.set_parameter(live2d_map[param], value)
        
        # 3D avatar mapping
        if self._avatar:
            pass  # Would apply to bone rotations
    
    def look_at_user(self):
        """Make avatar look at user (center of screen)."""
        self._attention.x = 0.0
        self._attention.y = 0.0
        self._attention.target_type = "user"
    
    def look_at_point(self, x: float, y: float):
        """
        Look at screen position.
        
        Args:
            x: -1 (left) to 1 (right)
            y: -1 (down) to 1 (up)
        """
        self._attention.x = max(-1, min(1, x))
        self._attention.y = max(-1, min(1, y))
        self._attention.target_type = "point"
    
    def look_random(self, range_x: float = 0.5, range_y: float = 0.3):
        """Look at random point within range."""
        self._attention.x = random.uniform(-range_x, range_x)
        self._attention.y = random.uniform(-range_y, range_y)
        self._attention.target_type = "random"
    
    def enable_reactions(self, enabled: bool):
        """Enable/disable reactive behaviors."""
        self._reaction_enabled = enabled
    
    def trigger_reaction(self, event_type: str):
        """
        Trigger a reaction to an event.
        
        Args:
            event_type: Event that occurred
        """
        if not self._reaction_enabled:
            return
        
        now = time.time()
        
        for reaction in self._reactions:
            if reaction.event_type != event_type:
                continue
            
            # Check cooldown
            last_time = self._last_reaction_time.get(event_type, 0)
            if now - last_time < reaction.cooldown:
                continue
            
            # Check probability
            if random.random() > reaction.probability:
                continue
            
            # Trigger reaction
            self._last_reaction_time[event_type] = now
            
            if reaction.response_emotion:
                self.express_emotion(reaction.response_emotion, intensity=0.6)
            
            if reaction.response_gesture:
                self.gesture(reaction.response_gesture, intensity=0.5)
            
            logger.debug(f"Triggered reaction for {event_type}")
            break
    
    def set_speech_intensity(self, intensity: float):
        """
        Set mouth movement for speech.
        
        Args:
            intensity: Volume/intensity (0-1)
        """
        mouth_open = intensity * 0.5 * self._detail_level
        
        if self._live2d:
            self._live2d.set_mouth_open(mouth_open)
        
        if self._avatar and hasattr(self._avatar, 'set_mouth_open'):
            self._avatar.set_mouth_open(mouth_open)
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current avatar state."""
        with self._lock:
            return {
                "emotion": self._emotion_state.primary.name,
                "emotion_intensity": self._emotion_state.primary_intensity,
                "secondary_emotion": self._emotion_state.secondary.name if self._emotion_state.secondary else None,
                "current_gesture": self._current_gesture,
                "attention": {
                    "x": self._attention.x,
                    "y": self._attention.y,
                    "type": self._attention.target_type
                },
                "detail_level": self._detail_level,
                "reactions_enabled": self._reaction_enabled,
            }


# Convenience functions

def create_lightweight_controller(avatar: Any = None, live2d: Any = None) -> AIAvatarController:
    """Create controller optimized for low-power devices."""
    ctrl = AIAvatarController(avatar, live2d)
    ctrl.set_detail_level(0.3)
    ctrl._update_fps = 15
    ctrl._idle_behavior = False
    ctrl._reaction_enabled = False
    return ctrl


def create_full_controller(avatar: Any = None, live2d: Any = None) -> AIAvatarController:
    """Create controller with all features enabled."""
    ctrl = AIAvatarController(avatar, live2d)
    ctrl.set_detail_level(1.0)
    ctrl._update_fps = 30
    ctrl._idle_behavior = True
    ctrl._reaction_enabled = True
    return ctrl
