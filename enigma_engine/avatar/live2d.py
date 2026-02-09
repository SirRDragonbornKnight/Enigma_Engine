"""
Live2D Support for Enigma AI Engine

Import and control Live2D models for 2D VTuber-style avatars.

Live2D models (.moc3, .model3.json) provide 2D character animation
with layered deformation, offering a lighter alternative to 3D models.

Usage:
    from enigma_engine.avatar.live2d import Live2DController, load_live2d_model
    
    # Load a Live2D model
    controller = load_live2d_model("path/to/model.model3.json")
    
    # Set parameters
    controller.set_parameter("ParamAngleX", 15.0)  # Head tilt
    controller.set_parameter("ParamEyeLOpen", 0.8)  # Left eye
    
    # Apply expression
    controller.set_expression("happy")
    
    # Animate with motion
    controller.play_motion("idle", loop=True)

Requirements:
    pip install live2d-py  # Optional Live2D SDK wrapper
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import time
import threading

logger = logging.getLogger(__name__)


class Live2DParameterType(Enum):
    """Common Live2D parameter types."""
    # Head/Face
    ANGLE_X = "ParamAngleX"         # Head left/right tilt (-30 to 30)
    ANGLE_Y = "ParamAngleY"         # Head up/down (-30 to 30)  
    ANGLE_Z = "ParamAngleZ"         # Head rotation (-30 to 30)
    
    # Eyes
    EYE_L_OPEN = "ParamEyeLOpen"    # Left eye open (0-1)
    EYE_R_OPEN = "ParamEyeROpen"    # Right eye open (0-1)
    EYE_BALL_X = "ParamEyeBallX"    # Eye look left/right (-1 to 1)
    EYE_BALL_Y = "ParamEyeBallY"    # Eye look up/down (-1 to 1)
    EYE_L_SMILE = "ParamEyeLSmile"  # Left eye smile (0-1)
    EYE_R_SMILE = "ParamEyeRSmile"  # Right eye smile (0-1)
    
    # Eyebrows
    BROW_L_Y = "ParamBrowLY"        # Left brow position (-1 to 1)
    BROW_R_Y = "ParamBrowRY"        # Right brow position (-1 to 1)
    BROW_L_ANGLE = "ParamBrowLAngle"
    BROW_R_ANGLE = "ParamBrowRAngle"
    
    # Mouth
    MOUTH_OPEN_Y = "ParamMouthOpenY"  # Mouth open (0-1)
    MOUTH_FORM = "ParamMouthForm"     # Smile/frown (-1 to 1)
    
    # Body
    BODY_ANGLE_X = "ParamBodyAngleX"  # Body lean (-10 to 10)
    BODY_ANGLE_Y = "ParamBodyAngleY"
    BODY_ANGLE_Z = "ParamBodyAngleZ"
    
    # Breathing
    BREATH = "ParamBreath"            # Breathing (0-1)
    
    # Arms (if available)
    ARM_L = "ParamArmL"               # Left arm position
    ARM_R = "ParamArmR"               # Right arm position


@dataclass
class Live2DParameter:
    """A single Live2D parameter."""
    id: str
    min_value: float = -30.0
    max_value: float = 30.0
    default_value: float = 0.0
    current_value: float = 0.0


@dataclass
class Live2DExpression:
    """A Live2D expression preset."""
    name: str
    fade_in_time: float = 0.3
    fade_out_time: float = 0.3
    parameters: Dict[str, float] = field(default_factory=dict)


@dataclass
class Live2DMotion:
    """A Live2D motion/animation."""
    name: str
    path: str
    duration: float = 0.0
    loop: bool = False
    fade_in_time: float = 0.0
    fade_out_time: float = 0.0


@dataclass
class Live2DModelInfo:
    """Information about a loaded Live2D model."""
    name: str
    path: Path
    version: int = 3  # moc3 version
    parameters: Dict[str, Live2DParameter] = field(default_factory=dict)
    expressions: Dict[str, Live2DExpression] = field(default_factory=dict)
    motions: Dict[str, List[Live2DMotion]] = field(default_factory=dict)
    hit_areas: List[str] = field(default_factory=list)
    
    # Physics groups
    physics_groups: List[str] = field(default_factory=list)
    
    # Canvas size
    width: int = 1024
    height: int = 1024


class Live2DController:
    """
    Controller for Live2D model animation.
    
    Provides:
    - Parameter control (head angle, eyes, mouth, etc.)
    - Expression presets
    - Motion playback
    - Lip sync integration
    - Physics simulation toggle
    
    Works with or without actual Live2D SDK - can emit parameter
    changes for external renderer.
    """
    
    def __init__(self, model_info: Live2DModelInfo):
        self.model = model_info
        self._lock = threading.Lock()
        
        # Current state
        self._current_parameters: Dict[str, float] = {}
        self._target_parameters: Dict[str, float] = {}
        self._current_expression: Optional[str] = None
        self._current_motion: Optional[str] = None
        
        # Animation
        self._animation_thread: Optional[threading.Thread] = None
        self._running = False
        self._animation_fps = 30
        
        # Smoothing
        self._smooth_factor = 0.15  # Lower = smoother but slower
        
        # Callbacks
        self._callbacks: List[Callable[[Dict[str, float]], None]] = []
        
        # Physics
        self._physics_enabled = True
        self._breathing_enabled = True
        self._breathing_speed = 3.0  # Seconds per cycle
        
        # Initialize parameters to defaults
        for param_id, param in self.model.parameters.items():
            self._current_parameters[param_id] = param.default_value
            self._target_parameters[param_id] = param.default_value
        
        logger.info(f"Live2D controller initialized for {model_info.name}")
    
    def start(self):
        """Start the animation loop."""
        if self._running:
            return
        
        self._running = True
        self._animation_thread = threading.Thread(
            target=self._animation_loop,
            daemon=True
        )
        self._animation_thread.start()
        logger.debug("Live2D animation loop started")
    
    def stop(self):
        """Stop the animation loop."""
        self._running = False
        if self._animation_thread:
            self._animation_thread.join(timeout=1.0)
            self._animation_thread = None
        logger.debug("Live2D animation loop stopped")
    
    def _animation_loop(self):
        """Main animation loop - updates parameters smoothly."""
        frame_time = 1.0 / self._animation_fps
        breath_phase = 0.0
        
        while self._running:
            start = time.perf_counter()
            
            with self._lock:
                # Smooth interpolation to target
                for param_id, target in self._target_parameters.items():
                    current = self._current_parameters.get(param_id, 0.0)
                    diff = target - current
                    
                    if abs(diff) > 0.001:
                        self._current_parameters[param_id] = current + diff * self._smooth_factor
                    else:
                        self._current_parameters[param_id] = target
                
                # Add breathing animation
                if self._breathing_enabled and "ParamBreath" in self.model.parameters:
                    import math
                    breath_phase += frame_time / self._breathing_speed * 2 * math.pi
                    breath_value = (math.sin(breath_phase) + 1) * 0.5
                    self._current_parameters["ParamBreath"] = breath_value
                
                # Notify callbacks
                params_copy = self._current_parameters.copy()
            
            for callback in self._callbacks:
                try:
                    callback(params_copy)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            # Sleep for remaining frame time
            elapsed = time.perf_counter() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
    
    def set_parameter(self, param_id: str, value: float, immediate: bool = False):
        """
        Set a parameter value.
        
        Args:
            param_id: Parameter ID (e.g., "ParamAngleX")
            value: Target value
            immediate: If True, skip smoothing
        """
        # Clamp to valid range if parameter is known
        if param_id in self.model.parameters:
            param = self.model.parameters[param_id]
            value = max(param.min_value, min(param.max_value, value))
        
        with self._lock:
            self._target_parameters[param_id] = value
            if immediate:
                self._current_parameters[param_id] = value
    
    def get_parameter(self, param_id: str) -> float:
        """Get current parameter value."""
        with self._lock:
            return self._current_parameters.get(param_id, 0.0)
    
    def set_parameters(self, params: Dict[str, float], immediate: bool = False):
        """Set multiple parameters at once."""
        for param_id, value in params.items():
            self.set_parameter(param_id, value, immediate)
    
    def set_expression(self, expression_name: str):
        """
        Apply an expression preset.
        
        Args:
            expression_name: Name of expression (e.g., "happy", "sad")
        """
        if expression_name not in self.model.expressions:
            logger.warning(f"Unknown expression: {expression_name}")
            return
        
        expr = self.model.expressions[expression_name]
        self._current_expression = expression_name
        
        # Apply expression parameters
        for param_id, value in expr.parameters.items():
            self.set_parameter(param_id, value)
        
        logger.debug(f"Applied expression: {expression_name}")
    
    def play_motion(self, motion_group: str, index: int = 0, loop: bool = False):
        """
        Play a motion animation.
        
        Args:
            motion_group: Motion group (e.g., "idle", "tap")
            index: Motion index within group
            loop: Whether to loop
        """
        if motion_group not in self.model.motions:
            logger.warning(f"Unknown motion group: {motion_group}")
            return
        
        motions = self.model.motions[motion_group]
        if index >= len(motions):
            logger.warning(f"Motion index {index} out of range")
            return
        
        motion = motions[index]
        self._current_motion = f"{motion_group}_{index}"
        
        logger.debug(f"Playing motion: {motion.name}")
        # Note: Actual motion playback would require Live2D SDK
        # This is a stub for the motion system
    
    def stop_motion(self):
        """Stop current motion."""
        self._current_motion = None
    
    def set_head_angle(self, x: float = 0, y: float = 0, z: float = 0):
        """
        Convenience method to set head angle.
        
        Args:
            x: Left/right tilt (-30 to 30)
            y: Up/down (-30 to 30)
            z: Rotation (-30 to 30)
        """
        self.set_parameters({
            "ParamAngleX": x,
            "ParamAngleY": y,
            "ParamAngleZ": z
        })
    
    def set_eye_openness(self, left: float = 1.0, right: float = 1.0):
        """
        Set eye openness.
        
        Args:
            left: Left eye (0=closed, 1=open)
            right: Right eye (0=closed, 1=open)
        """
        self.set_parameters({
            "ParamEyeLOpen": left,
            "ParamEyeROpen": right
        })
    
    def set_mouth_open(self, value: float):
        """
        Set mouth openness for lip sync.
        
        Args:
            value: Mouth open amount (0=closed, 1=open)
        """
        self.set_parameter("ParamMouthOpenY", value)
    
    def look_at(self, x: float, y: float):
        """
        Make eyes look at screen position.
        
        Args:
            x: Horizontal position (-1=left, 1=right)
            y: Vertical position (-1=down, 1=up)
        """
        self.set_parameters({
            "ParamEyeBallX": x,
            "ParamEyeBallY": y
        })
    
    def blink(self):
        """Trigger a blink animation."""
        # Quick blink
        self.set_eye_openness(0, 0)
        # Schedule eye open after 100ms
        threading.Timer(0.1, lambda: self.set_eye_openness(1, 1)).start()
    
    def add_parameter_callback(self, callback: Callable[[Dict[str, float]], None]):
        """Add callback for parameter updates."""
        self._callbacks.append(callback)
    
    def remove_parameter_callback(self, callback: Callable[[Dict[str, float]], None]):
        """Remove parameter callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def enable_physics(self, enabled: bool):
        """Enable/disable physics simulation."""
        self._physics_enabled = enabled
    
    def enable_breathing(self, enabled: bool):
        """Enable/disable breathing animation."""
        self._breathing_enabled = enabled
    
    def reset_to_defaults(self):
        """Reset all parameters to default values."""
        with self._lock:
            for param_id, param in self.model.parameters.items():
                self._target_parameters[param_id] = param.default_value
                self._current_parameters[param_id] = param.default_value
        
        self._current_expression = None
        self._current_motion = None


def load_live2d_model(model_path: Union[str, Path]) -> Optional[Live2DController]:
    """
    Load a Live2D model from .model3.json file.
    
    Args:
        model_path: Path to .model3.json file
        
    Returns:
        Live2DController or None if loading failed
    """
    model_path_obj = Path(model_path)
    
    if not model_path_obj.exists():
        logger.error(f"Model file not found: {model_path_obj}")
        return None
    
    if not model_path_obj.suffix == '.json':
        logger.error("Expected .model3.json file")
        return None
    
    try:
        with open(model_path_obj, encoding='utf-8') as f:
            model_json = json.load(f)
        
        model_dir = model_path_obj.parent
        
        # Parse model info
        name = model_path_obj.stem.replace('.model3', '')
        
        # Parse parameters
        parameters = {}
        
        # Standard Live2D parameters
        standard_params = [
            ("ParamAngleX", -30, 30, 0),
            ("ParamAngleY", -30, 30, 0),
            ("ParamAngleZ", -30, 30, 0),
            ("ParamEyeLOpen", 0, 1, 1),
            ("ParamEyeROpen", 0, 1, 1),
            ("ParamEyeBallX", -1, 1, 0),
            ("ParamEyeBallY", -1, 1, 0),
            ("ParamEyeLSmile", 0, 1, 0),
            ("ParamEyeRSmile", 0, 1, 0),
            ("ParamBrowLY", -1, 1, 0),
            ("ParamBrowRY", -1, 1, 0),
            ("ParamMouthOpenY", 0, 1, 0),
            ("ParamMouthForm", -1, 1, 0),
            ("ParamBodyAngleX", -10, 10, 0),
            ("ParamBreath", 0, 1, 0),
        ]
        
        for param_id, min_v, max_v, default in standard_params:
            parameters[param_id] = Live2DParameter(
                id=param_id,
                min_value=min_v,
                max_value=max_v,
                default_value=default,
                current_value=default
            )
        
        # Parse expressions
        expressions = {}
        if "FileReferences" in model_json:
            expr_list = model_json["FileReferences"].get("Expressions", [])
            for expr_data in expr_list:
                expr_name = expr_data.get("Name", "")
                expr_file = expr_data.get("File", "")
                
                if expr_name:
                    # Try to load expression file
                    expr_path = model_dir / expr_file
                    expr_params = {}
                    
                    if expr_path.exists():
                        try:
                            with open(expr_path, encoding='utf-8') as ef:
                                expr_json = json.load(ef)
                            for p in expr_json.get("Parameters", []):
                                expr_params[p["Id"]] = p.get("Value", 0)
                        except:
                            pass
                    
                    expressions[expr_name] = Live2DExpression(
                        name=expr_name,
                        parameters=expr_params
                    )
        
        # Parse motions
        motions = {}
        if "FileReferences" in model_json:
            motion_groups = model_json["FileReferences"].get("Motions", {})
            for group_name, motion_list in motion_groups.items():
                motions[group_name] = []
                for i, motion_data in enumerate(motion_list):
                    motion_file = motion_data.get("File", "")
                    motions[group_name].append(Live2DMotion(
                        name=f"{group_name}_{i}",
                        path=str(model_dir / motion_file),
                        loop=motion_data.get("FadeInTime", 0.0) > 0
                    ))
        
        # Parse hit areas
        hit_areas = []
        for hit in model_json.get("HitAreas", []):
            hit_areas.append(hit.get("Name", ""))
        
        model_info = Live2DModelInfo(
            name=name,
            path=model_path_obj,
            parameters=parameters,
            expressions=expressions,
            motions=motions,
            hit_areas=hit_areas
        )
        
        controller = Live2DController(model_info)
        logger.info(f"Loaded Live2D model: {name}")
        logger.info(f"  Expressions: {list(expressions.keys())}")
        logger.info(f"  Motion groups: {list(motions.keys())}")
        
        return controller
        
    except Exception as e:
        logger.error(f"Failed to load Live2D model: {e}")
        return None


def is_live2d_supported() -> bool:
    """
    Check if Live2D SDK is available.
    
    Returns:
        True if live2d-py is installed
    """
    try:
        import live2d
        return True
    except ImportError:
        return False


def get_live2d_requirements() -> str:
    """Get installation instructions for Live2D support."""
    return """
Live2D Support Requirements:

1. Install the Live2D Python wrapper:
   pip install live2d-py
   
2. Download Cubism SDK for your platform:
   https://www.live2d.com/en/download/cubism-sdk/

3. Place your Live2D model files (.moc3, .model3.json, textures)
   in the models/avatars/live2d/ directory

Supported formats:
- *.model3.json (Cubism 3+)
- *.moc3 (model data)
- *.exp3.json (expressions)
- *.motion3.json (animations)

Note: Without the Live2D SDK, you can still use this controller
to emit parameter values to external Live2D renderers.
"""
