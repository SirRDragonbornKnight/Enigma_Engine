"""
Procedural Animation

Generates procedural animations for breathing, idle movements,
micro-expressions, and other natural movements.

FILE: enigma_engine/avatar/procedural_animation.py
TYPE: Avatar Animation
MAIN CLASSES: ProceduralAnimator, BreathingController, IdleAnimator
"""

import logging
import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AnimationType(Enum):
    """Types of procedural animation."""
    BREATHING = "breathing"
    BLINKING = "blinking"
    IDLE_SWAY = "idle_sway"
    MICRO_EXPRESSION = "micro_expression"
    LOOK_AROUND = "look_around"
    WEIGHT_SHIFT = "weight_shift"
    HEAD_TILT = "head_tilt"
    FINGER_TWITCH = "finger_twitch"


@dataclass
class AnimationLayer:
    """Animation layer with blending."""
    name: str
    weight: float = 1.0
    active: bool = True
    additive: bool = True  # Add to other animations vs replace


@dataclass
class Vector3:
    """3D vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def lerp(self, target: 'Vector3', t: float) -> 'Vector3':
        return Vector3(
            self.x + (target.x - self.x) * t,
            self.y + (target.y - self.y) * t,
            self.z + (target.z - self.z) * t
        )


@dataclass
class ProceduralConfig:
    """Configuration for procedural animation."""
    # Breathing
    breath_rate: float = 0.2        # Breaths per second
    breath_intensity: float = 1.0   # Amplitude multiplier
    chest_expansion: float = 0.02   # Max chest scale increase
    shoulder_rise: float = 0.01     # Max shoulder rise
    
    # Blinking
    blink_rate: float = 0.05        # Blinks per second (avg)
    blink_duration: float = 0.15    # Seconds for a blink
    blink_variance: float = 0.5     # Randomness in blink timing
    
    # Idle sway
    sway_enabled: bool = True
    sway_amount: float = 0.005      # Max sway distance
    sway_speed: float = 0.3         # Sway cycle speed
    
    # Look around
    look_enabled: bool = True
    look_interval: float = 5.0      # Seconds between looks
    look_variance: float = 3.0      # Randomness in timing
    look_range: float = 15.0        # Max look angle (degrees)
    
    # Micro expressions
    micro_expression_rate: float = 0.02  # Per second
    micro_expression_intensity: float = 0.3


class BreathingController:
    """Controls breathing animation."""
    
    def __init__(self, config: ProceduralConfig):
        """
        Initialize breathing controller.
        
        Args:
            config: Animation configuration
        """
        self._config = config
        self._phase = 0.0
        self._paused = False
    
    def update(self, delta_time: float) -> dict[str, Any]:
        """
        Update breathing animation.
        
        Args:
            delta_time: Time since last update
            
        Returns:
            Animation values for this frame
        """
        if self._paused:
            return {}
        
        # Advance phase
        self._phase += delta_time * self._config.breath_rate * 2 * math.pi
        if self._phase > 2 * math.pi:
            self._phase -= 2 * math.pi
        
        # Calculate breath cycle (smooth sine wave)
        breath = (math.sin(self._phase) + 1) / 2  # 0 to 1
        breath = self._ease_breath(breath)
        
        # Apply intensity
        breath *= self._config.breath_intensity
        
        return {
            "chest_scale": Vector3(
                1.0 + breath * self._config.chest_expansion,
                1.0 + breath * self._config.chest_expansion * 0.5,
                1.0 + breath * self._config.chest_expansion * 0.8
            ),
            "shoulder_offset": Vector3(
                0, breath * self._config.shoulder_rise, 0
            ),
            "spine_rotation": Vector3(
                breath * 0.5, 0, 0  # Slight spine extension
            ),
            "breath_value": breath
        }
    
    def _ease_breath(self, t: float) -> float:
        """Apply easing for natural breath curve."""
        # Inhale faster than exhale
        if t < 0.4:
            return t / 0.4
        else:
            return 1.0 - (t - 0.4) / 0.6
    
    def pause(self):
        """Pause breathing (e.g., during speech)."""
        self._paused = True
    
    def resume(self):
        """Resume breathing."""
        self._paused = False
    
    def hold_breath(self, duration: float):
        """Hold breath at current phase."""
        self._paused = True


class BlinkController:
    """Controls eye blinking."""
    
    def __init__(self, config: ProceduralConfig):
        """
        Initialize blink controller.
        
        Args:
            config: Animation configuration
        """
        self._config = config
        self._time_to_blink = self._next_blink_time()
        self._blink_progress = 0.0
        self._is_blinking = False
    
    def update(self, delta_time: float) -> dict[str, float]:
        """
        Update blink animation.
        
        Args:
            delta_time: Time since last update
            
        Returns:
            Blend shape values
        """
        result = {
            "blink_left": 0.0,
            "blink_right": 0.0
        }
        
        if self._is_blinking:
            self._blink_progress += delta_time / self._config.blink_duration
            
            if self._blink_progress >= 1.0:
                self._is_blinking = False
                self._blink_progress = 0.0
                self._time_to_blink = self._next_blink_time()
            else:
                # Blink curve: quick close, slower open
                blink_value = self._blink_curve(self._blink_progress)
                result["blink_left"] = blink_value
                result["blink_right"] = blink_value
        else:
            self._time_to_blink -= delta_time
            if self._time_to_blink <= 0:
                self._is_blinking = True
        
        return result
    
    def _blink_curve(self, t: float) -> float:
        """Calculate blink value at time t (0-1)."""
        if t < 0.3:
            # Quick close
            return t / 0.3
        elif t < 0.5:
            # Hold closed
            return 1.0
        else:
            # Slower open
            return 1.0 - (t - 0.5) / 0.5
    
    def _next_blink_time(self) -> float:
        """Calculate time until next blink."""
        base = 1.0 / self._config.blink_rate
        variance = base * self._config.blink_variance
        return base + random.uniform(-variance, variance)
    
    def trigger_blink(self):
        """Manually trigger a blink."""
        if not self._is_blinking:
            self._is_blinking = True
            self._blink_progress = 0.0


class IdleAnimator:
    """Controls idle movements (sway, weight shift)."""
    
    def __init__(self, config: ProceduralConfig):
        """
        Initialize idle animator.
        
        Args:
            config: Animation configuration
        """
        self._config = config
        self._sway_phase_x = random.uniform(0, 2 * math.pi)
        self._sway_phase_z = random.uniform(0, 2 * math.pi)
        self._weight_shift_time = 0.0
        self._weight_target = 0.0
    
    def update(self, delta_time: float) -> dict[str, Any]:
        """
        Update idle animation.
        
        Args:
            delta_time: Time since last update
            
        Returns:
            Animation values
        """
        result = {}
        
        if self._config.sway_enabled:
            # Update sway phases
            self._sway_phase_x += delta_time * self._config.sway_speed
            self._sway_phase_z += delta_time * self._config.sway_speed * 0.7
            
            # Calculate sway offset
            sway_x = math.sin(self._sway_phase_x) * self._config.sway_amount
            sway_z = math.sin(self._sway_phase_z) * self._config.sway_amount * 0.5
            
            result["root_offset"] = Vector3(sway_x, 0, sway_z)
            result["spine_rotation"] = Vector3(
                sway_z * 10,  # Slight forward/back lean
                sway_x * 5,   # Slight side twist
                sway_x * -3   # Counter-lean
            )
        
        # Weight shift
        self._weight_shift_time += delta_time
        if self._weight_shift_time > 8.0 + random.uniform(-2, 4):
            self._weight_target = random.uniform(-0.5, 0.5)
            self._weight_shift_time = 0.0
        
        result["weight_shift"] = self._weight_target
        
        return result


class LookAroundController:
    """Controls random look-around movements."""
    
    def __init__(self, config: ProceduralConfig):
        """
        Initialize look controller.
        
        Args:
            config: Animation configuration
        """
        self._config = config
        self._time_to_look = self._next_look_time()
        self._current_look = Vector3()
        self._target_look = Vector3()
        self._look_progress = 1.0
    
    def update(self, delta_time: float) -> dict[str, Any]:
        """
        Update look animation.
        
        Args:
            delta_time: Time since last update
            
        Returns:
            Head/eye rotation values
        """
        if not self._config.look_enabled:
            return {}
        
        # Update timing
        self._time_to_look -= delta_time
        
        if self._time_to_look <= 0:
            # Pick new look target
            self._target_look = Vector3(
                random.uniform(-self._config.look_range, self._config.look_range),
                random.uniform(-self._config.look_range * 0.5, self._config.look_range * 0.5),
                0
            )
            self._look_progress = 0.0
            self._time_to_look = self._next_look_time()
        
        # Interpolate to target
        if self._look_progress < 1.0:
            self._look_progress = min(1.0, self._look_progress + delta_time * 2)
            self._current_look = self._current_look.lerp(
                self._target_look, 
                self._ease_look(self._look_progress)
            )
        
        return {
            "head_rotation": self._current_look,
            "eye_target": self._current_look * 2  # Eyes move more
        }
    
    def _ease_look(self, t: float) -> float:
        """Ease function for natural head movement."""
        # Smooth step
        return t * t * (3 - 2 * t)
    
    def _next_look_time(self) -> float:
        """Time until next look."""
        return self._config.look_interval + random.uniform(
            -self._config.look_variance, 
            self._config.look_variance
        )
    
    def set_focus_point(self, point: Vector3):
        """Set a specific focus point to look at."""
        self._target_look = point
        self._look_progress = 0.0


class MicroExpressionController:
    """Generates subtle facial micro-expressions."""
    
    EXPRESSIONS = [
        {"name": "slight_smile", "shapes": {"mouth_smile": 0.2}},
        {"name": "brow_raise", "shapes": {"brow_up": 0.3}},
        {"name": "brow_furrow", "shapes": {"brow_down": 0.2}},
        {"name": "nose_wrinkle", "shapes": {"nose_wrinkle": 0.15}},
        {"name": "lip_press", "shapes": {"lip_press": 0.2}},
        {"name": "cheek_puff", "shapes": {"cheek_puff": 0.1}},
    ]
    
    def __init__(self, config: ProceduralConfig):
        """
        Initialize micro-expression controller.
        
        Args:
            config: Animation configuration
        """
        self._config = config
        self._current_expression = None
        self._expression_value = 0.0
        self._time_to_next = self._next_expression_time()
        self._phase = "none"  # none, rising, holding, falling
    
    def update(self, delta_time: float) -> dict[str, float]:
        """
        Update micro-expressions.
        
        Args:
            delta_time: Time since last update
            
        Returns:
            Blend shape values
        """
        result = {}
        
        if self._phase == "none":
            self._time_to_next -= delta_time
            if self._time_to_next <= 0:
                self._start_expression()
        
        elif self._phase == "rising":
            self._expression_value = min(1.0, self._expression_value + delta_time * 4)
            if self._expression_value >= 1.0:
                self._phase = "holding"
                self._time_to_next = random.uniform(0.3, 1.0)
        
        elif self._phase == "holding":
            self._time_to_next -= delta_time
            if self._time_to_next <= 0:
                self._phase = "falling"
        
        elif self._phase == "falling":
            self._expression_value = max(0.0, self._expression_value - delta_time * 3)
            if self._expression_value <= 0:
                self._phase = "none"
                self._time_to_next = self._next_expression_time()
                self._current_expression = None
        
        # Apply current expression
        if self._current_expression:
            intensity = self._expression_value * self._config.micro_expression_intensity
            for shape, value in self._current_expression["shapes"].items():
                result[shape] = value * intensity
        
        return result
    
    def _start_expression(self):
        """Start a new micro-expression."""
        self._current_expression = random.choice(self.EXPRESSIONS)
        self._expression_value = 0.0
        self._phase = "rising"
    
    def _next_expression_time(self) -> float:
        """Time until next expression."""
        if self._config.micro_expression_rate <= 0:
            return 999999
        return 1.0 / self._config.micro_expression_rate + random.uniform(-2, 5)


class ProceduralAnimator:
    """Main procedural animation controller."""
    
    def __init__(self, config: Optional[ProceduralConfig] = None):
        """
        Initialize procedural animator.
        
        Args:
            config: Animation configuration
        """
        self._config = config or ProceduralConfig()
        self._layers: dict[AnimationType, AnimationLayer] = {}
        
        # Create controllers
        self._breathing = BreathingController(self._config)
        self._blinking = BlinkController(self._config)
        self._idle = IdleAnimator(self._config)
        self._look = LookAroundController(self._config)
        self._micro = MicroExpressionController(self._config)
        
        # Initialize layers
        for anim_type in AnimationType:
            self._layers[anim_type] = AnimationLayer(
                name=anim_type.value,
                weight=1.0,
                active=True
            )
        
        self._last_update = time.time()
    
    def update(self) -> dict[str, Any]:
        """
        Update all procedural animations.
        
        Returns:
            Combined animation values
        """
        current_time = time.time()
        delta = current_time - self._last_update
        self._last_update = current_time
        
        result = {
            "bone_transforms": {},
            "blend_shapes": {},
            "parameters": {}
        }
        
        # Update each system
        if self._is_active(AnimationType.BREATHING):
            breath = self._breathing.update(delta)
            weight = self._layers[AnimationType.BREATHING].weight
            self._apply_with_weight(result, breath, weight)
        
        if self._is_active(AnimationType.BLINKING):
            blink = self._blinking.update(delta)
            weight = self._layers[AnimationType.BLINKING].weight
            for shape, value in blink.items():
                result["blend_shapes"][shape] = result["blend_shapes"].get(shape, 0) + value * weight
        
        if self._is_active(AnimationType.IDLE_SWAY):
            idle = self._idle.update(delta)
            weight = self._layers[AnimationType.IDLE_SWAY].weight
            self._apply_with_weight(result, idle, weight)
        
        if self._is_active(AnimationType.LOOK_AROUND):
            look = self._look.update(delta)
            weight = self._layers[AnimationType.LOOK_AROUND].weight
            self._apply_with_weight(result, look, weight)
        
        if self._is_active(AnimationType.MICRO_EXPRESSION):
            micro = self._micro.update(delta)
            weight = self._layers[AnimationType.MICRO_EXPRESSION].weight
            for shape, value in micro.items():
                result["blend_shapes"][shape] = result["blend_shapes"].get(shape, 0) + value * weight
        
        return result
    
    def _is_active(self, anim_type: AnimationType) -> bool:
        """Check if animation type is active."""
        layer = self._layers.get(anim_type)
        return layer and layer.active and layer.weight > 0
    
    def _apply_with_weight(self, result: dict, values: dict, weight: float):
        """Apply values with weight blending."""
        for key, value in values.items():
            if isinstance(value, Vector3):
                if key not in result["bone_transforms"]:
                    result["bone_transforms"][key] = Vector3()
                result["bone_transforms"][key] = result["bone_transforms"][key] + value * weight
            elif isinstance(value, (int, float)):
                result["parameters"][key] = result["parameters"].get(key, 0) + value * weight
    
    def set_layer_weight(self, anim_type: AnimationType, weight: float):
        """Set weight for an animation layer."""
        if anim_type in self._layers:
            self._layers[anim_type].weight = max(0.0, min(1.0, weight))
    
    def set_layer_active(self, anim_type: AnimationType, active: bool):
        """Enable/disable an animation layer."""
        if anim_type in self._layers:
            self._layers[anim_type].active = active
    
    def trigger_blink(self):
        """Trigger a manual blink."""
        self._blinking.trigger_blink()
    
    def pause_breathing(self):
        """Pause breathing animation."""
        self._breathing.pause()
    
    def resume_breathing(self):
        """Resume breathing animation."""
        self._breathing.resume()
    
    def set_look_target(self, point: Vector3):
        """Set a specific look target."""
        self._look.set_focus_point(point)
    
    @property
    def config(self) -> ProceduralConfig:
        """Get configuration."""
        return self._config
    
    @config.setter
    def config(self, value: ProceduralConfig):
        """Update configuration."""
        self._config = value
        self._breathing = BreathingController(value)
        self._blinking = BlinkController(value)
        self._idle = IdleAnimator(value)
        self._look = LookAroundController(value)
        self._micro = MicroExpressionController(value)


# Singleton
_procedural_animator: Optional[ProceduralAnimator] = None


def get_procedural_animator() -> ProceduralAnimator:
    """Get the procedural animator singleton."""
    global _procedural_animator
    if _procedural_animator is None:
        _procedural_animator = ProceduralAnimator()
    return _procedural_animator


__all__ = [
    'ProceduralAnimator',
    'ProceduralConfig',
    'AnimationType',
    'AnimationLayer',
    'BreathingController',
    'BlinkController',
    'IdleAnimator',
    'LookAroundController',
    'MicroExpressionController',
    'Vector3',
    'get_procedural_animator'
]
