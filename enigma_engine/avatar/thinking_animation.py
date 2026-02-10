"""
Thinking Animation System

Shows visual "thinking" indicators during AI inference.
Provides animated feedback while the model generates responses.

FILE: enigma_engine/avatar/thinking_animation.py
TYPE: Avatar Animation
MAIN CLASSES: ThinkingAnimator, ThinkingState, ThinkingIndicator
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ThinkingStyle(Enum):
    """Visual styles for thinking indication."""
    SUBTLE = "subtle"           # Minimal - slight head tilt
    MODERATE = "moderate"       # Default - chin rub, looking up
    EXPRESSIVE = "expressive"   # Full - pacing, dramatic poses
    MINIMAL = "minimal"         # Just eyes/expression change


class ThinkingPhase(Enum):
    """Phases of the thinking animation."""
    START = "start"         # Initial pose transition
    ACTIVE = "active"       # Main thinking loop
    EUREKA = "eureka"       # Found answer moment
    TRANSITION = "transition"  # Returning to normal


@dataclass  
class ThinkingState:
    """Current state of thinking animation."""
    is_thinking: bool = False
    phase: ThinkingPhase = ThinkingPhase.START
    started_at: float = 0.0
    duration: float = 0.0
    style: ThinkingStyle = ThinkingStyle.MODERATE
    progress: float = 0.0  # 0.0 to 1.0
    tokens_generated: int = 0
    expected_tokens: int = 0


@dataclass
class ThinkingPose:
    """A pose in the thinking animation sequence."""
    name: str
    duration: float  # seconds
    head_tilt: float = 0.0  # degrees
    eye_target: tuple = (0.0, 0.0, 0.0)  # look direction
    hand_position: str = "neutral"
    expression: str = "thinking"
    blend_in: float = 0.3
    blend_out: float = 0.3


# Default thinking pose sequences
THINKING_SEQUENCES: dict[ThinkingStyle, list[ThinkingPose]] = {
    ThinkingStyle.MINIMAL: [
        ThinkingPose("think_eyes_up", 2.0, head_tilt=5, eye_target=(0, 0.5, 0)),
        ThinkingPose("think_eyes_side", 2.0, head_tilt=-5, eye_target=(0.5, 0, 0)),
    ],
    ThinkingStyle.SUBTLE: [
        ThinkingPose("think_head_tilt", 2.5, head_tilt=10, expression="curious"),
        ThinkingPose("think_look_up", 2.0, head_tilt=5, eye_target=(0, 0.3, 0)),
        ThinkingPose("think_return", 1.5, head_tilt=0),
    ],
    ThinkingStyle.MODERATE: [
        ThinkingPose("think_chin_touch", 2.5, head_tilt=8, hand_position="chin"),
        ThinkingPose("think_look_away", 2.0, head_tilt=-5, eye_target=(0.4, 0.2, 0)),
        ThinkingPose("think_nod_slight", 1.5, head_tilt=3),
        ThinkingPose("think_return", 1.5, head_tilt=0, hand_position="neutral"),
    ],
    ThinkingStyle.EXPRESSIVE: [
        ThinkingPose("think_dramatic_pause", 1.5, head_tilt=15, expression="concentrated"),
        ThinkingPose("think_pace_left", 2.0, head_tilt=5),
        ThinkingPose("think_chin_stroke", 2.5, hand_position="chin_stroke"),
        ThinkingPose("think_pace_right", 2.0, head_tilt=-5),
        ThinkingPose("think_eureka_prep", 1.0, head_tilt=0, expression="curious"),
    ]
}

# Eureka animations for when thinking completes
EUREKA_ANIMATIONS: dict[ThinkingStyle, ThinkingPose] = {
    ThinkingStyle.MINIMAL: ThinkingPose("eureka_subtle", 0.5, expression="happy"),
    ThinkingStyle.SUBTLE: ThinkingPose("eureka_nod", 0.8, head_tilt=5, expression="pleased"),
    ThinkingStyle.MODERATE: ThinkingPose("eureka_gesture", 1.0, hand_position="open_palm", expression="happy"),
    ThinkingStyle.EXPRESSIVE: ThinkingPose("eureka_dramatic", 1.5, head_tilt=-10, hand_position="point_up", expression="excited"),
}


class ThinkingIndicator:
    """Visual indicator widget (for non-avatar UI)."""
    
    def __init__(self):
        self._active = False
        self._text = "Thinking"
        self._dots = 0
        self._max_dots = 3
        self._callbacks: list[Callable[[str], None]] = []
        self._timer: Optional[threading.Timer] = None
        
    def start(self, text: str = "Thinking"):
        """Start the indicator."""
        self._active = True
        self._text = text
        self._dots = 0
        self._update_loop()
        
    def stop(self):
        """Stop the indicator."""
        self._active = False
        if self._timer:
            self._timer.cancel()
            self._timer = None
            
    def _update_loop(self):
        """Animation loop for dots."""
        if not self._active:
            return
            
        self._dots = (self._dots + 1) % (self._max_dots + 1)
        display_text = self._text + "." * self._dots
        
        for callback in self._callbacks:
            try:
                callback(display_text)
            except Exception as e:
                logger.debug(f"Thinking callback error: {e}")
                
        self._timer = threading.Timer(0.5, self._update_loop)
        self._timer.start()
        
    def on_update(self, callback: Callable[[str], None]):
        """Register callback for text updates."""
        self._callbacks.append(callback)
        
    def get_text(self) -> str:
        """Get current indicator text."""
        return self._text + "." * self._dots


class ThinkingAnimator:
    """Manages thinking animations for avatar."""
    
    def __init__(self, avatar_controller=None):
        """
        Initialize thinking animator.
        
        Args:
            avatar_controller: Avatar controller instance
        """
        self._avatar = avatar_controller
        self._state = ThinkingState()
        self._style = ThinkingStyle.MODERATE
        self._current_pose_index = 0
        self._pose_start_time = 0.0
        self._animation_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[ThinkingState], None]] = []
        self._indicator = ThinkingIndicator()
        
    def set_avatar(self, avatar_controller):
        """Set the avatar controller."""
        self._avatar = avatar_controller
        
    def set_style(self, style: ThinkingStyle):
        """Set the thinking animation style."""
        self._style = style
        
    def start_thinking(self, 
                       expected_duration: float = 5.0,
                       expected_tokens: int = 100):
        """
        Start thinking animation.
        
        Args:
            expected_duration: Expected time in seconds
            expected_tokens: Expected number of tokens to generate
        """
        with self._lock:
            if self._state.is_thinking:
                return
                
            self._state = ThinkingState(
                is_thinking=True,
                phase=ThinkingPhase.START,
                started_at=time.time(),
                duration=expected_duration,
                style=self._style,
                expected_tokens=expected_tokens
            )
            
            self._current_pose_index = 0
            self._pose_start_time = time.time()
            self._stop_flag.clear()
            
        # Start animation thread
        self._animation_thread = threading.Thread(target=self._animation_loop, daemon=True)
        self._animation_thread.start()
        
        # Start text indicator
        self._indicator.start()
        
        self._notify_state()
        logger.debug("Started thinking animation")
        
    def stop_thinking(self, show_eureka: bool = True):
        """
        Stop thinking animation.
        
        Args:
            show_eureka: Whether to show eureka animation
        """
        with self._lock:
            if not self._state.is_thinking:
                return
                
            self._state.phase = ThinkingPhase.EUREKA if show_eureka else ThinkingPhase.TRANSITION
            self._stop_flag.set()
            
        # Play eureka animation
        if show_eureka:
            self._play_eureka()
            
        # Stop indicator
        self._indicator.stop()
        
        with self._lock:
            self._state.is_thinking = False
            self._state.phase = ThinkingPhase.START
            
        self._notify_state()
        logger.debug("Stopped thinking animation")
        
    def update_progress(self, tokens_generated: int):
        """Update thinking progress."""
        with self._lock:
            self._state.tokens_generated = tokens_generated
            if self._state.expected_tokens > 0:
                self._state.progress = min(tokens_generated / self._state.expected_tokens, 1.0)
                
    def _animation_loop(self):
        """Main animation loop."""
        sequence = THINKING_SEQUENCES.get(self._style, THINKING_SEQUENCES[ThinkingStyle.MODERATE])
        
        while not self._stop_flag.is_set():
            with self._lock:
                if not self._state.is_thinking:
                    break
                self._state.phase = ThinkingPhase.ACTIVE
                
            # Get current pose
            pose = sequence[self._current_pose_index]
            
            # Apply pose to avatar
            self._apply_pose(pose)
            
            # Wait for pose duration
            elapsed = 0.0
            while elapsed < pose.duration and not self._stop_flag.is_set():
                time.sleep(0.1)
                elapsed += 0.1
                
                # Update elapsed time in state
                with self._lock:
                    self._state.duration = time.time() - self._state.started_at
                    
            # Move to next pose
            self._current_pose_index = (self._current_pose_index + 1) % len(sequence)
            self._pose_start_time = time.time()
            
    def _apply_pose(self, pose: ThinkingPose):
        """Apply a pose to the avatar."""
        if not self._avatar:
            return
            
        try:
            # Set head rotation
            if hasattr(self._avatar, 'set_head_rotation'):
                self._avatar.set_head_rotation(pose.head_tilt, 0, 0)
                
            # Set eye target
            if hasattr(self._avatar, 'set_look_target'):
                self._avatar.set_look_target(*pose.eye_target)
                
            # Set expression
            if hasattr(self._avatar, 'set_expression'):
                self._avatar.set_expression(pose.expression)
                
            # Set hand position
            if hasattr(self._avatar, 'set_hand_pose') and pose.hand_position != "neutral":
                self._avatar.set_hand_pose(pose.hand_position)
                
        except Exception as e:
            logger.warning(f"Failed to apply thinking pose: {e}")
            
    def _play_eureka(self):
        """Play eureka animation."""
        eureka = EUREKA_ANIMATIONS.get(self._style, EUREKA_ANIMATIONS[ThinkingStyle.MODERATE])
        self._apply_pose(eureka)
        time.sleep(eureka.duration)
        
        # Return to neutral
        if self._avatar:
            try:
                if hasattr(self._avatar, 'reset_pose'):
                    self._avatar.reset_pose()
                if hasattr(self._avatar, 'set_expression'):
                    self._avatar.set_expression('neutral')
            except Exception as e:
                logger.debug(f"Avatar reset error: {e}")
                
    def get_state(self) -> ThinkingState:
        """Get current thinking state."""
        with self._lock:
            return ThinkingState(
                is_thinking=self._state.is_thinking,
                phase=self._state.phase,
                started_at=self._state.started_at,
                duration=self._state.duration,
                style=self._state.style,
                progress=self._state.progress,
                tokens_generated=self._state.tokens_generated,
                expected_tokens=self._state.expected_tokens
            )
            
    def on_state_change(self, callback: Callable[[ThinkingState], None]):
        """Register callback for state changes."""
        self._callbacks.append(callback)
        
    def _notify_state(self):
        """Notify callbacks of state change."""
        state = self.get_state()
        for callback in self._callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Thinking callback error: {e}")
                
    @property
    def indicator(self) -> ThinkingIndicator:
        """Get the text indicator."""
        return self._indicator


# Singleton
_thinking_animator: Optional[ThinkingAnimator] = None


def get_thinking_animator(avatar_controller=None) -> ThinkingAnimator:
    """Get the thinking animator singleton."""
    global _thinking_animator
    if _thinking_animator is None:
        _thinking_animator = ThinkingAnimator(avatar_controller)
    elif avatar_controller:
        _thinking_animator.set_avatar(avatar_controller)
    return _thinking_animator


__all__ = [
    'ThinkingAnimator',
    'ThinkingState',
    'ThinkingStyle',
    'ThinkingPhase',
    'ThinkingPose',
    'ThinkingIndicator',
    'get_thinking_animator',
    'THINKING_SEQUENCES',
    'EUREKA_ANIMATIONS'
]
