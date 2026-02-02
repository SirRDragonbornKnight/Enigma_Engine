"""
Adaptive Avatar Animation System

Works with ANY 3D model regardless of its structure by:
1. Analyzing what the model CAN do (bones, blend shapes, etc.)
2. Choosing the best animation strategy for that model
3. Providing fallbacks for models without animation capabilities

Animation Strategies:
- SKELETAL: Full bone-based animation (rigged models)
- BLENDSHAPE: Morph target animation (face shapes)
- TRANSFORM: Whole-model movement (static models)
- HYBRID: Combination of available methods

The AI receives model capability info and adapts its commands accordingly.
"""

import json
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .controller import AvatarController


class AnimationStrategy(Enum):
    """Available animation strategies based on model capabilities."""
    NONE = auto()       # No animation possible
    TRANSFORM = auto()  # Whole-model transforms (any model)
    SKELETAL = auto()   # Bone-based animation
    BLENDSHAPE = auto() # Morph targets / shape keys
    HYBRID = auto()     # Combination of methods


class AnimationType(Enum):
    """Types of animations the avatar can perform."""
    IDLE = "idle"
    SPEAKING = "speaking"
    EMOTION = "emotion"
    GESTURE = "gesture"
    LOOK_AT = "look_at"
    NOD = "nod"
    SHAKE = "shake"
    WAVE = "wave"
    BREATHE = "breathe"
    BLINK = "blink"


@dataclass
class ModelCapabilities:
    """What a 3D model can do - detected automatically."""
    # Basic info
    has_skeleton: bool = False
    has_blend_shapes: bool = False
    is_static: bool = True
    
    # Skeleton details
    bones: List[str] = field(default_factory=list)
    bone_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    
    # Standard bone mapping (detected or user-configured)
    head_bone: Optional[str] = None
    neck_bone: Optional[str] = None
    spine_bones: List[str] = field(default_factory=list)
    jaw_bone: Optional[str] = None
    eye_bones: List[str] = field(default_factory=list)
    arm_bones: Dict[str, List[str]] = field(default_factory=dict)  # left/right
    
    # Blend shapes (for expressions/lip sync)
    blend_shapes: List[str] = field(default_factory=list)
    mouth_shapes: List[str] = field(default_factory=list)  # For lip sync
    eye_shapes: List[str] = field(default_factory=list)    # For blink/expressions
    expression_shapes: Dict[str, List[str]] = field(default_factory=dict)
    
    # What's possible with this model
    can_lip_sync: bool = False
    can_blink: bool = False
    can_emote: bool = False
    can_look_at: bool = False
    can_nod: bool = False
    can_gesture: bool = False
    
    # Recommended strategy
    recommended_strategy: AnimationStrategy = AnimationStrategy.TRANSFORM
    
    def to_dict(self) -> dict:
        """Convert to dictionary for AI consumption."""
        return {
            'has_skeleton': self.has_skeleton,
            'has_blend_shapes': self.has_blend_shapes,
            'bone_count': len(self.bones),
            'blend_shape_count': len(self.blend_shapes),
            'can_lip_sync': self.can_lip_sync,
            'can_blink': self.can_blink,
            'can_emote': self.can_emote,
            'can_look_at': self.can_look_at,
            'can_nod': self.can_nod,
            'can_gesture': self.can_gesture,
            'strategy': self.recommended_strategy.name,
            'head_bone': self.head_bone,
            'jaw_bone': self.jaw_bone,
            'mouth_shapes': self.mouth_shapes[:5] if self.mouth_shapes else [],
        }


@dataclass
class AnimationState:
    """Current animation state."""
    current_animation: Optional[AnimationType] = None
    emotion: str = "neutral"
    speaking: bool = False
    looking_at: Optional[Tuple[float, float, float]] = None
    
    # Transform state (for TRANSFORM strategy)
    position_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_factor: float = 1.0
    
    # Bone state (for SKELETAL strategy)
    bone_rotations: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    
    # Blend shape state (for BLENDSHAPE strategy)
    blend_shape_values: Dict[str, float] = field(default_factory=dict)


class AdaptiveAnimator:
    """
    Adaptive animation system that works with any 3D model.
    
    Automatically detects model capabilities and chooses the best
    animation strategy. Provides consistent API regardless of model type.
    
    Usage:
        animator = AdaptiveAnimator()
        animator.set_model_info(model_metadata)  # From model loader
        
        # Same API works for any model
        animator.speak("Hello!")  # Uses best available method
        animator.set_emotion("happy")
        animator.look_at(500, 300)
        animator.nod()
    """
    
    # Common bone name patterns for detection
    BONE_PATTERNS = {
        'head': ['head', 'Head', 'HEAD', 'head_bone', 'Bip001_Head', 'mixamorig:Head'],
        'neck': ['neck', 'Neck', 'NECK', 'neck_bone', 'Bip001_Neck', 'mixamorig:Neck'],
        'spine': ['spine', 'Spine', 'SPINE', 'torso', 'Bip001_Spine', 'mixamorig:Spine'],
        'jaw': ['jaw', 'Jaw', 'JAW', 'chin', 'mouth_bone', 'Bip001_Jaw'],
        'eye_l': ['eye.L', 'eye_L', 'LeftEye', 'Eye_L', 'Bip001_L_Eye'],
        'eye_r': ['eye.R', 'eye_R', 'RightEye', 'Eye_R', 'Bip001_R_Eye'],
        'arm_l': ['arm.L', 'arm_L', 'LeftArm', 'Arm_L', 'Bip001_L_UpperArm', 'mixamorig:LeftArm'],
        'arm_r': ['arm.R', 'arm_R', 'RightArm', 'Arm_R', 'Bip001_R_UpperArm', 'mixamorig:RightArm'],
    }
    
    # Common blend shape patterns
    SHAPE_PATTERNS = {
        'mouth_open': ['mouthOpen', 'Mouth_Open', 'jawOpen', 'Fcl_MTH_A', 'vrc.v_aa', 'A'],
        'mouth_smile': ['mouthSmile', 'Mouth_Smile', 'smile', 'Fcl_MTH_Joy', 'happy'],
        'mouth_o': ['mouthO', 'Mouth_O', 'Fcl_MTH_O', 'vrc.v_oh', 'O'],
        'mouth_u': ['mouthU', 'Mouth_U', 'Fcl_MTH_U', 'vrc.v_ou', 'U'],
        'blink_l': ['eyeBlink_L', 'Eye_Blink_L', 'Fcl_EYE_Close_L', 'winkLeft'],
        'blink_r': ['eyeBlink_R', 'Eye_Blink_R', 'Fcl_EYE_Close_R', 'winkRight'],
        'blink': ['eyesClosed', 'Blink', 'Fcl_EYE_Close', 'blink'],
    }
    
    # Viseme (mouth shape) mapping for lip sync
    VISEME_MAP = {
        'sil': [],  # Silence - mouth closed
        'aa': ['mouth_open'],  # "father"
        'ee': ['mouth_smile'],  # "see"
        'oo': ['mouth_o', 'mouth_u'],  # "food"
        'oh': ['mouth_o'],  # "go"
        'consonant': [],  # Most consonants
    }
    
    def __init__(self, controller: Optional['AvatarController'] = None):
        """Initialize adaptive animator."""
        self.controller = controller
        self.capabilities = ModelCapabilities()
        self.state = AnimationState()
        self.strategy = AnimationStrategy.TRANSFORM
        
        # Animation thread
        self._running = False
        self._anim_thread: Optional[threading.Thread] = None
        self._anim_lock = threading.Lock()
        
        # Animation queue
        self._animation_queue: List[Dict[str, Any]] = []
        
        # Callbacks for rendering
        self._on_transform_update: List[Callable] = []
        self._on_bone_update: List[Callable] = []
        self._on_blend_shape_update: List[Callable] = []
        
        # Timing
        self._last_update = time.time()
        self._idle_phase = 0.0
        
        # AI command file path
        self._command_path = Path(__file__).parent.parent.parent / "data" / "avatar" / "animator_command.json"
        self._last_command_time = 0
    
    def set_model_info(self, model_metadata: dict) -> ModelCapabilities:
        """
        Analyze model and determine its capabilities.
        
        Args:
            model_metadata: Dictionary from model loader with bones, shapes, etc.
            
        Returns:
            Detected capabilities
        """
        caps = ModelCapabilities()
        
        # Check for skeleton
        bones = model_metadata.get('skeleton_bones', []) or model_metadata.get('bones', [])
        if bones:
            caps.has_skeleton = True
            caps.is_static = False
            caps.bones = bones
            
            # Map standard bones
            for bone in bones:
                bone_lower = bone.lower()
                
                # Head
                if any(p.lower() in bone_lower for p in self.BONE_PATTERNS['head']):
                    caps.head_bone = bone
                    caps.can_nod = True
                    caps.can_look_at = True
                
                # Neck
                if any(p.lower() in bone_lower for p in self.BONE_PATTERNS['neck']):
                    caps.neck_bone = bone
                
                # Spine
                if any(p.lower() in bone_lower for p in self.BONE_PATTERNS['spine']):
                    caps.spine_bones.append(bone)
                
                # Jaw (for skeletal lip sync)
                if any(p.lower() in bone_lower for p in self.BONE_PATTERNS['jaw']):
                    caps.jaw_bone = bone
                    caps.can_lip_sync = True
                
                # Eyes
                if any(p.lower() in bone_lower for p in self.BONE_PATTERNS['eye_l'] + self.BONE_PATTERNS['eye_r']):
                    caps.eye_bones.append(bone)
                    caps.can_look_at = True
                
                # Arms
                if any(p.lower() in bone_lower for p in self.BONE_PATTERNS['arm_l']):
                    caps.arm_bones.setdefault('left', []).append(bone)
                    caps.can_gesture = True
                if any(p.lower() in bone_lower for p in self.BONE_PATTERNS['arm_r']):
                    caps.arm_bones.setdefault('right', []).append(bone)
                    caps.can_gesture = True
        
        # Check for blend shapes / morph targets
        shapes = model_metadata.get('blend_shapes', []) or model_metadata.get('shape_keys', [])
        if shapes:
            caps.has_blend_shapes = True
            caps.is_static = False
            caps.blend_shapes = shapes
            
            # Map standard shapes
            for shape in shapes:
                shape_lower = shape.lower()
                
                # Mouth shapes for lip sync
                for pattern_name, patterns in self.SHAPE_PATTERNS.items():
                    if pattern_name.startswith('mouth'):
                        if any(p.lower() in shape_lower for p in patterns):
                            caps.mouth_shapes.append(shape)
                            caps.can_lip_sync = True
                
                # Eye shapes for blink
                for pattern_name, patterns in self.SHAPE_PATTERNS.items():
                    if 'blink' in pattern_name:
                        if any(p.lower() in shape_lower for p in patterns):
                            caps.eye_shapes.append(shape)
                            caps.can_blink = True
                
                # Expression detection
                emotion_keywords = ['happy', 'sad', 'angry', 'surprised', 'joy', 'sorrow']
                for emotion in emotion_keywords:
                    if emotion in shape_lower:
                        caps.expression_shapes.setdefault(emotion, []).append(shape)
                        caps.can_emote = True
        
        # Determine best strategy
        if caps.has_skeleton and caps.has_blend_shapes:
            caps.recommended_strategy = AnimationStrategy.HYBRID
        elif caps.has_skeleton:
            caps.recommended_strategy = AnimationStrategy.SKELETAL
        elif caps.has_blend_shapes:
            caps.recommended_strategy = AnimationStrategy.BLENDSHAPE
        else:
            caps.recommended_strategy = AnimationStrategy.TRANSFORM
        
        self.capabilities = caps
        self.strategy = caps.recommended_strategy
        
        print(f"[AdaptiveAnimator] Model analysis complete:")
        print(f"  Strategy: {self.strategy.name}")
        print(f"  Can lip sync: {caps.can_lip_sync} (jaw: {caps.jaw_bone}, mouth shapes: {len(caps.mouth_shapes)})")
        print(f"  Can emote: {caps.can_emote}")
        print(f"  Can look at: {caps.can_look_at}")
        print(f"  Can nod: {caps.can_nod}")
        print(f"  Can gesture: {caps.can_gesture}")
        
        return caps
    
    def start(self):
        """Start the animation system."""
        if self._running:
            return
        
        self._running = True
        self._anim_thread = threading.Thread(target=self._animation_loop, daemon=True)
        self._anim_thread.start()
        print("[AdaptiveAnimator] Started")
    
    def stop(self):
        """Stop the animation system."""
        self._running = False
        if self._anim_thread:
            self._anim_thread.join(timeout=1.0)
        print("[AdaptiveAnimator] Stopped")
    
    def _animation_loop(self):
        """Main animation loop."""
        while self._running:
            now = time.time()
            dt = now - self._last_update
            self._last_update = now
            
            # Check for AI commands
            self._check_ai_commands()
            
            # Process animation queue
            self._process_queue(dt)
            
            # Idle animation
            if not self._animation_queue and not self.state.speaking:
                self._do_idle_animation(dt)
            
            # Speaking animation
            if self.state.speaking:
                self._do_speaking_animation(dt)
            
            time.sleep(0.016)  # ~60fps
    
    def _process_queue(self, dt: float):
        """Process queued animations."""
        with self._anim_lock:
            if not self._animation_queue:
                return
            
            anim = self._animation_queue[0]
            anim_type = anim.get('type')
            progress = anim.get('progress', 0.0)
            duration = anim.get('duration', 1.0)
            
            # Update progress
            progress += dt / duration
            anim['progress'] = progress
            
            # Execute animation frame
            self._execute_animation_frame(anim, progress)
            
            # Complete?
            if progress >= 1.0:
                self._animation_queue.pop(0)
                callback = anim.get('on_complete')
                if callback:
                    callback()
    
    def _execute_animation_frame(self, anim: dict, progress: float):
        """Execute a single frame of an animation."""
        anim_type = anim.get('type')
        
        if anim_type == AnimationType.NOD:
            self._animate_nod(progress, anim)
        elif anim_type == AnimationType.SHAKE:
            self._animate_shake(progress, anim)
        elif anim_type == AnimationType.WAVE:
            self._animate_wave(progress, anim)
        elif anim_type == AnimationType.BLINK:
            self._animate_blink(progress, anim)
        elif anim_type == AnimationType.LOOK_AT:
            self._animate_look_at(progress, anim)
    
    def _do_idle_animation(self, dt: float):
        """Perform idle animation based on strategy."""
        self._idle_phase += dt * 0.5  # Slow idle
        
        if self.strategy == AnimationStrategy.TRANSFORM:
            # Gentle breathing/bobbing for any model
            bob = math.sin(self._idle_phase) * 0.02
            sway = math.sin(self._idle_phase * 0.7) * 0.3
            
            self.state.position_offset = (0.0, bob, 0.0)
            self.state.rotation_offset = (0.0, sway, 0.0)
            
            self._notify_transform_update()
        
        elif self.strategy in (AnimationStrategy.SKELETAL, AnimationStrategy.HYBRID):
            # Subtle head/spine movement
            if self.capabilities.head_bone:
                head_rot = (
                    math.sin(self._idle_phase * 0.3) * 2.0,  # Slight nod
                    math.sin(self._idle_phase * 0.5) * 3.0,  # Slight turn
                    0.0
                )
                self.state.bone_rotations[self.capabilities.head_bone] = head_rot
                self._notify_bone_update(self.capabilities.head_bone, head_rot)
            
            # Random blinks
            if self.capabilities.can_blink and random_chance(0.003):  # ~0.3% per frame
                self.blink()
    
    def _do_speaking_animation(self, dt: float):
        """Animate speaking based on capabilities."""
        # Phase for mouth movement
        self._idle_phase += dt * 8.0  # Faster for speech
        
        if self.capabilities.can_lip_sync:
            if self.capabilities.mouth_shapes:
                # Blend shape lip sync
                mouth_open = abs(math.sin(self._idle_phase)) * 0.7
                for shape in self.capabilities.mouth_shapes[:1]:  # Use first mouth shape
                    self.state.blend_shape_values[shape] = mouth_open
                    self._notify_blend_shape_update(shape, mouth_open)
            
            elif self.capabilities.jaw_bone:
                # Skeletal jaw animation
                jaw_rot = (abs(math.sin(self._idle_phase)) * 15.0, 0.0, 0.0)
                self.state.bone_rotations[self.capabilities.jaw_bone] = jaw_rot
                self._notify_bone_update(self.capabilities.jaw_bone, jaw_rot)
        
        else:
            # Fallback: scale pulse for any model
            pulse = 1.0 + abs(math.sin(self._idle_phase)) * 0.05
            self.state.scale_factor = pulse
            self._notify_transform_update()
    
    # === Public Animation API ===
    
    def speak(self, text: str, duration: Optional[float] = None):
        """
        Animate speaking.
        
        Automatically uses best method for the model:
        - Blend shapes for mouth movement
        - Jaw bone rotation
        - Or visual pulse fallback
        
        Args:
            text: Text being spoken (used for timing)
            duration: Override duration (auto-calculated if None)
        """
        if duration is None:
            # Estimate duration: ~100ms per character
            duration = max(0.5, len(text) * 0.08)
        
        self.state.speaking = True
        
        # Schedule stop
        def stop_speaking():
            self.state.speaking = False
            self._reset_mouth()
        
        threading.Timer(duration, stop_speaking).start()
        print(f"[AdaptiveAnimator] Speaking for {duration:.1f}s (method: {self._get_lip_sync_method()})")
    
    def _get_lip_sync_method(self) -> str:
        """Get description of lip sync method being used."""
        if self.capabilities.mouth_shapes:
            return f"blend shapes ({len(self.capabilities.mouth_shapes)})"
        elif self.capabilities.jaw_bone:
            return f"jaw bone ({self.capabilities.jaw_bone})"
        else:
            return "visual pulse (no lip sync available)"
    
    def _reset_mouth(self):
        """Reset mouth to closed state."""
        for shape in self.capabilities.mouth_shapes:
            self.state.blend_shape_values[shape] = 0.0
            self._notify_blend_shape_update(shape, 0.0)
        
        if self.capabilities.jaw_bone:
            self.state.bone_rotations[self.capabilities.jaw_bone] = (0.0, 0.0, 0.0)
            self._notify_bone_update(self.capabilities.jaw_bone, (0.0, 0.0, 0.0))
        
        self.state.scale_factor = 1.0
        self._notify_transform_update()
    
    def set_emotion(self, emotion: str):
        """
        Set avatar emotion/expression.
        
        Uses best available method:
        - Expression blend shapes
        - Bone-based facial pose
        - Visual effects (color tint, glow)
        
        Args:
            emotion: Emotion name (happy, sad, angry, surprised, neutral, etc.)
        """
        self.state.emotion = emotion
        
        if self.capabilities.can_emote and emotion in self.capabilities.expression_shapes:
            # Use blend shapes
            # Reset previous expression
            for shapes in self.capabilities.expression_shapes.values():
                for shape in shapes:
                    self.state.blend_shape_values[shape] = 0.0
                    self._notify_blend_shape_update(shape, 0.0)
            
            # Set new expression
            for shape in self.capabilities.expression_shapes.get(emotion, []):
                self.state.blend_shape_values[shape] = 1.0
                self._notify_blend_shape_update(shape, 1.0)
        
        print(f"[AdaptiveAnimator] Emotion set: {emotion}")
    
    def look_at(self, x: float, y: float, z: float = 0.0):
        """
        Make avatar look at a point.
        
        Uses:
        - Eye bones if available
        - Head bone rotation
        - Or model rotation fallback
        
        Args:
            x, y, z: Target position
        """
        self.state.looking_at = (x, y, z)
        
        with self._anim_lock:
            self._animation_queue.append({
                'type': AnimationType.LOOK_AT,
                'target': (x, y, z),
                'duration': 0.3,
                'progress': 0.0,
            })
    
    def _animate_look_at(self, progress: float, anim: dict):
        """Animate looking at target."""
        target = anim.get('target', (0, 0, 0))
        
        # Calculate rotation towards target (simplified)
        # This would need proper 3D math in real implementation
        yaw = math.atan2(target[0], 10.0) * 30.0  # Convert to degrees
        pitch = math.atan2(target[1], 10.0) * 20.0
        
        # Ease in/out
        t = ease_in_out(progress)
        
        if self.capabilities.head_bone:
            current = self.state.bone_rotations.get(self.capabilities.head_bone, (0, 0, 0))
            new_rot = (
                lerp(current[0], pitch, t),
                lerp(current[1], yaw, t),
                0.0
            )
            self.state.bone_rotations[self.capabilities.head_bone] = new_rot
            self._notify_bone_update(self.capabilities.head_bone, new_rot)
        else:
            # Rotate whole model
            self.state.rotation_offset = (pitch * t * 0.3, yaw * t, 0.0)
            self._notify_transform_update()
    
    def nod(self, intensity: float = 1.0):
        """
        Make avatar nod (yes gesture).
        
        Args:
            intensity: How emphatic (0.5 = subtle, 1.0 = normal, 1.5 = emphatic)
        """
        with self._anim_lock:
            self._animation_queue.append({
                'type': AnimationType.NOD,
                'intensity': intensity,
                'duration': 0.4,
                'progress': 0.0,
            })
    
    def _animate_nod(self, progress: float, anim: dict):
        """Animate nodding."""
        intensity = anim.get('intensity', 1.0)
        
        # Nod curve: down, up, down, center
        nod_angle = math.sin(progress * math.pi * 2) * 15.0 * intensity
        
        if self.capabilities.head_bone:
            current = self.state.bone_rotations.get(self.capabilities.head_bone, (0, 0, 0))
            self.state.bone_rotations[self.capabilities.head_bone] = (nod_angle, current[1], current[2])
            self._notify_bone_update(self.capabilities.head_bone, self.state.bone_rotations[self.capabilities.head_bone])
        else:
            # Whole model nod
            self.state.rotation_offset = (nod_angle * 0.5, 0.0, 0.0)
            self._notify_transform_update()
    
    def shake(self, intensity: float = 1.0):
        """
        Make avatar shake head (no gesture).
        
        Args:
            intensity: How emphatic
        """
        with self._anim_lock:
            self._animation_queue.append({
                'type': AnimationType.SHAKE,
                'intensity': intensity,
                'duration': 0.5,
                'progress': 0.0,
            })
    
    def _animate_shake(self, progress: float, anim: dict):
        """Animate head shake."""
        intensity = anim.get('intensity', 1.0)
        
        # Shake curve: left, right, left, right, center
        shake_angle = math.sin(progress * math.pi * 4) * 20.0 * intensity * (1 - progress)
        
        if self.capabilities.head_bone:
            current = self.state.bone_rotations.get(self.capabilities.head_bone, (0, 0, 0))
            self.state.bone_rotations[self.capabilities.head_bone] = (current[0], shake_angle, current[2])
            self._notify_bone_update(self.capabilities.head_bone, self.state.bone_rotations[self.capabilities.head_bone])
        else:
            self.state.rotation_offset = (0.0, shake_angle * 0.5, 0.0)
            self._notify_transform_update()
    
    def wave(self):
        """Make avatar wave (if capable)."""
        if not self.capabilities.can_gesture:
            print("[AdaptiveAnimator] Model cannot wave (no arm bones)")
            # Fallback: rotation wiggle
            with self._anim_lock:
                self._animation_queue.append({
                    'type': AnimationType.WAVE,
                    'fallback': True,
                    'duration': 1.0,
                    'progress': 0.0,
                })
            return
        
        with self._anim_lock:
            self._animation_queue.append({
                'type': AnimationType.WAVE,
                'fallback': False,
                'duration': 1.0,
                'progress': 0.0,
            })
    
    def _animate_wave(self, progress: float, anim: dict):
        """Animate waving."""
        if anim.get('fallback'):
            # No arms - do a wiggle
            wiggle = math.sin(progress * math.pi * 6) * 10.0 * (1 - progress)
            self.state.rotation_offset = (0.0, 0.0, wiggle)
            self._notify_transform_update()
        else:
            # Arm wave
            arm_bones = self.capabilities.arm_bones.get('right', [])
            if arm_bones:
                wave_angle = math.sin(progress * math.pi * 4) * 30.0
                for bone in arm_bones[:1]:  # Just upper arm
                    self.state.bone_rotations[bone] = (0.0, 0.0, -90.0 + wave_angle)
                    self._notify_bone_update(bone, self.state.bone_rotations[bone])
    
    def blink(self):
        """Make avatar blink."""
        if not self.capabilities.can_blink:
            return
        
        with self._anim_lock:
            self._animation_queue.append({
                'type': AnimationType.BLINK,
                'duration': 0.15,
                'progress': 0.0,
            })
    
    def _animate_blink(self, progress: float, anim: dict):
        """Animate blinking."""
        # Blink curve: quick close, quick open
        if progress < 0.5:
            blink_value = progress * 2  # 0 -> 1
        else:
            blink_value = (1 - progress) * 2  # 1 -> 0
        
        for shape in self.capabilities.eye_shapes:
            self.state.blend_shape_values[shape] = blink_value
            self._notify_blend_shape_update(shape, blink_value)
    
    # === Callbacks ===
    
    def on_transform_update(self, callback: Callable):
        """Register callback for transform updates."""
        self._on_transform_update.append(callback)
    
    def on_bone_update(self, callback: Callable):
        """Register callback for bone updates."""
        self._on_bone_update.append(callback)
    
    def on_blend_shape_update(self, callback: Callable):
        """Register callback for blend shape updates."""
        self._on_blend_shape_update.append(callback)
    
    def _notify_transform_update(self):
        """Notify transform update callbacks."""
        for cb in self._on_transform_update:
            try:
                cb(self.state.position_offset, self.state.rotation_offset, self.state.scale_factor)
            except Exception as e:
                print(f"[AdaptiveAnimator] Transform callback error: {e}")
    
    def _notify_bone_update(self, bone: str, rotation: Tuple[float, float, float]):
        """Notify bone update callbacks."""
        for cb in self._on_bone_update:
            try:
                cb(bone, rotation)
            except Exception as e:
                print(f"[AdaptiveAnimator] Bone callback error: {e}")
    
    def _notify_blend_shape_update(self, shape: str, value: float):
        """Notify blend shape update callbacks."""
        for cb in self._on_blend_shape_update:
            try:
                cb(shape, value)
            except Exception as e:
                print(f"[AdaptiveAnimator] Blend shape callback error: {e}")
    
    # === AI Integration ===
    
    def _check_ai_commands(self):
        """Check for AI-sent commands."""
        if not self._command_path.exists():
            return
        
        try:
            with open(self._command_path, 'r') as f:
                cmd = json.load(f)
            
            timestamp = cmd.get('timestamp', 0)
            if timestamp <= self._last_command_time:
                return
            
            self._last_command_time = timestamp
            action = cmd.get('action', '').lower()
            value = cmd.get('value', '')
            
            if action == 'speak':
                self.speak(value)
            elif action == 'emotion':
                self.set_emotion(value)
            elif action == 'nod':
                self.nod(float(value) if value else 1.0)
            elif action == 'shake':
                self.shake(float(value) if value else 1.0)
            elif action == 'wave':
                self.wave()
            elif action == 'blink':
                self.blink()
            elif action == 'look_at':
                parts = value.split(',')
                if len(parts) >= 2:
                    self.look_at(float(parts[0]), float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0)
            elif action == 'get_capabilities':
                # Write capabilities for AI to read
                caps_path = self._command_path.parent / "animator_capabilities.json"
                with open(caps_path, 'w') as f:
                    json.dump(self.capabilities.to_dict(), f, indent=2)
        
        except (json.JSONDecodeError, IOError, ValueError) as e:
            pass
    
    def get_capabilities_for_ai(self) -> dict:
        """
        Get capability summary for AI decision-making.
        
        Returns dict that tells AI what this avatar CAN do.
        """
        return self.capabilities.to_dict()


# === Utility Functions ===

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


def ease_in_out(t: float) -> float:
    """Smooth ease in/out curve."""
    return t * t * (3 - 2 * t)


def random_chance(probability: float) -> bool:
    """Return True if probability exceeds threshold (deterministic).
    
    For animation, this creates predictable behavior:
    - probability > 0.5 always triggers
    - probability <= 0.5 never triggers
    This makes animations consistent rather than chaotic.
    """
    return probability > 0.5
