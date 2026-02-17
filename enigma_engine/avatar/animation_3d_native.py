"""
3D Avatar Animation System (No External Dependencies)

Real-time 3D skeletal animation using ONLY PyQt5's built-in OpenGL.
No Panda3D, no extra packages - just PyQt5 and numpy (both already required).

Features:
- Load glTF/GLB models (parsed manually, no external libs)
- Skeletal animation with bone transforms
- AI-controlled animation states
- Render to PyQt widget for desktop overlay
- Transparent background support

Supported Formats:
- glTF/GLB (JSON-based, easy to parse)
- OBJ (for static models)
- Custom JSON animation format

Usage:
    from enigma_engine.avatar.animation_3d_native import NativeAvatar3D, AI3DController
    
    avatar = NativeAvatar3D()
    avatar.load_model("character.glb")
    
    # Get widget to embed in your GUI
    widget = avatar.get_widget()
    layout.addWidget(widget)
    
    # AI controls
    controller = AI3DController(avatar)
    controller.start_talking()
"""

import base64
import json
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

# Numpy for matrix math (already in requirements)
try:
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[3D Native] numpy not available, using pure Python math (slower)")

# PyQt5 OpenGL (built into PyQt5, no extra install)
try:
    from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QMatrix4x4, QQuaternion, QVector3D
    from PyQt5.QtWidgets import QOpenGLWidget
    HAS_PYQT_GL = True
except ImportError:
    HAS_PYQT_GL = False
    # Don't print - this is optional and the fallback works fine

# OpenGL functions from PyQt5
# NOTE: Star imports from OpenGL.GL and OpenGL.GLU are intentional here.
# OpenGL bindings require many constants (GL_DEPTH_TEST, GL_BLEND, etc.)
# and functions (glEnable, glClear, etc.) - explicit imports would be verbose.
try:
    from OpenGL.GL import *  # noqa: F401,F403 - OpenGL star import is standard practice
    from OpenGL.GLU import *  # noqa: F401,F403
    HAS_OPENGL = True
except ImportError:
    # Try PyOpenGL-accelerate or fall back
    try:
        # PyQt5 has basic GL bindings
        HAS_OPENGL = False
    except Exception as e:
        logger.debug(f"OpenGL fallback failed: {e}")
        HAS_OPENGL = False


class Animation3DState(Enum):
    """Avatar animation states."""
    IDLE = auto()
    TALKING = auto()
    LISTENING = auto()
    THINKING = auto()
    HAPPY = auto()
    SAD = auto()
    SURPRISED = auto()
    GESTURE = auto()


@dataclass
class Bone:
    """A single bone in the skeleton."""
    name: str
    index: int
    parent_index: int = -1
    local_position: tuple[float, float, float] = (0, 0, 0)
    local_rotation: tuple[float, float, float, float] = (0, 0, 0, 1)  # Quaternion xyzw
    local_scale: tuple[float, float, float] = (1, 1, 1)
    inverse_bind_matrix: Optional[list[float]] = None  # 4x4 matrix as flat list
    children: list[int] = field(default_factory=list)


@dataclass
class AnimationKeyframe:
    """A single keyframe."""
    time: float
    value: tuple  # Position (xyz), Rotation (xyzw quaternion), or Scale (xyz)


@dataclass  
class AnimationChannel:
    """Animation data for one bone property."""
    bone_index: int
    property_type: str  # "translation", "rotation", "scale"
    keyframes: list[AnimationKeyframe] = field(default_factory=list)


@dataclass
class AnimationClip:
    """A complete animation."""
    name: str
    duration: float
    channels: list[AnimationChannel] = field(default_factory=list)
    loop: bool = True


@dataclass
class Mesh:
    """3D mesh data."""
    name: str
    vertices: list[float] = field(default_factory=list)  # xyz xyz xyz...
    normals: list[float] = field(default_factory=list)   # xyz xyz xyz...
    uvs: list[float] = field(default_factory=list)       # uv uv uv...
    indices: list[int] = field(default_factory=list)     # Triangle indices
    bone_weights: list[float] = field(default_factory=list)  # 4 weights per vertex
    bone_indices: list[int] = field(default_factory=list)    # 4 bone indices per vertex


class GLTFLoader:
    """
    Load glTF/GLB files without external dependencies.
    glTF is JSON-based, so we can parse it with stdlib.
    """
    
    @staticmethod
    def load(path: str) -> tuple[list[Mesh], list[Bone], list[AnimationClip]]:
        """Load a glTF/GLB file."""
        path = Path(path)
        
        if path.suffix.lower() == '.glb':
            return GLTFLoader._load_glb(path)
        else:
            return GLTFLoader._load_gltf(path)
    
    @staticmethod
    def _load_glb(path: Path) -> tuple[list[Mesh], list[Bone], list[AnimationClip]]:
        """Load binary GLB format."""
        with open(path, 'rb') as f:
            # GLB header
            magic = f.read(4)
            if magic != b'glTF':
                raise ValueError("Not a valid GLB file")
            
            version = struct.unpack('<I', f.read(4))[0]
            length = struct.unpack('<I', f.read(4))[0]
            
            # Read chunks
            json_data = None
            bin_data = None
            
            while f.tell() < length:
                chunk_length = struct.unpack('<I', f.read(4))[0]
                chunk_type = struct.unpack('<I', f.read(4))[0]
                chunk_data = f.read(chunk_length)
                
                if chunk_type == 0x4E4F534A:  # JSON
                    json_data = json.loads(chunk_data.decode('utf-8'))
                elif chunk_type == 0x004E4942:  # BIN
                    bin_data = chunk_data
            
            return GLTFLoader._parse_gltf(json_data, bin_data, path.parent)
    
    @staticmethod
    def _load_gltf(path: Path) -> tuple[list[Mesh], list[Bone], list[AnimationClip]]:
        """Load JSON glTF format."""
        with open(path) as f:
            json_data = json.load(f)
        
        # Load external binary buffer if referenced
        bin_data = None
        if 'buffers' in json_data and json_data['buffers']:
            buffer_uri = json_data['buffers'][0].get('uri', '')
            if buffer_uri and not buffer_uri.startswith('data:'):
                bin_path = path.parent / buffer_uri
                if bin_path.exists():
                    with open(bin_path, 'rb') as f:
                        bin_data = f.read()
            elif buffer_uri.startswith('data:'):
                # Base64 encoded
                _, encoded = buffer_uri.split(',', 1)
                bin_data = base64.b64decode(encoded)
        
        return GLTFLoader._parse_gltf(json_data, bin_data, path.parent)
    
    @staticmethod
    def _parse_gltf(gltf: dict, bin_data: bytes, base_path: Path) -> tuple[list[Mesh], list[Bone], list[AnimationClip]]:
        """Parse glTF JSON structure."""
        meshes = []
        bones = []
        animations = []
        
        # Helper to get accessor data
        def get_accessor_data(accessor_idx: int) -> list:
            if accessor_idx is None or 'accessors' not in gltf:
                return []
            
            accessor = gltf['accessors'][accessor_idx]
            buffer_view = gltf['bufferViews'][accessor['bufferView']]
            
            offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
            count = accessor['count']
            
            # Determine component type
            component_type = accessor['componentType']
            type_map = {
                5120: ('b', 1),  # BYTE
                5121: ('B', 1),  # UNSIGNED_BYTE
                5122: ('h', 2),  # SHORT
                5123: ('H', 2),  # UNSIGNED_SHORT
                5125: ('I', 4),  # UNSIGNED_INT
                5126: ('f', 4),  # FLOAT
            }
            fmt, size = type_map.get(component_type, ('f', 4))
            
            # Determine number of components
            type_sizes = {
                'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4,
                'MAT2': 4, 'MAT3': 9, 'MAT4': 16
            }
            num_components = type_sizes.get(accessor['type'], 1)
            
            # Read data
            result = []
            stride = buffer_view.get('byteStride', size * num_components)
            
            for i in range(count):
                pos = offset + i * stride
                for j in range(num_components):
                    value = struct.unpack_from(fmt, bin_data, pos + j * size)[0]
                    result.append(value)
            
            return result
        
        # Parse meshes
        for mesh_data in gltf.get('meshes', []):
            for primitive in mesh_data.get('primitives', []):
                mesh = Mesh(name=mesh_data.get('name', 'mesh'))
                
                attrs = primitive.get('attributes', {})
                mesh.vertices = get_accessor_data(attrs.get('POSITION'))
                mesh.normals = get_accessor_data(attrs.get('NORMAL'))
                mesh.uvs = get_accessor_data(attrs.get('TEXCOORD_0'))
                mesh.indices = get_accessor_data(primitive.get('indices'))
                
                # Skinning data
                mesh.bone_weights = get_accessor_data(attrs.get('WEIGHTS_0'))
                mesh.bone_indices = [int(x) for x in get_accessor_data(attrs.get('JOINTS_0'))]
                
                meshes.append(mesh)
        
        # Parse skeleton
        for skin_data in gltf.get('skins', []):
            joint_indices = skin_data.get('joints', [])
            inv_bind_matrices = get_accessor_data(skin_data.get('inverseBindMatrices'))
            
            for i, joint_idx in enumerate(joint_indices):
                node = gltf['nodes'][joint_idx]
                bone = Bone(
                    name=node.get('name', f'bone_{i}'),
                    index=i,
                    local_position=tuple(node.get('translation', [0, 0, 0])),
                    local_rotation=tuple(node.get('rotation', [0, 0, 0, 1])),
                    local_scale=tuple(node.get('scale', [1, 1, 1])),
                )
                
                # Inverse bind matrix (16 floats)
                if inv_bind_matrices:
                    start = i * 16
                    bone.inverse_bind_matrix = inv_bind_matrices[start:start + 16]
                
                # Find parent
                for j, other_idx in enumerate(joint_indices):
                    other_node = gltf['nodes'][other_idx]
                    if joint_idx in other_node.get('children', []):
                        bone.parent_index = j
                        break
                
                bones.append(bone)
        
        # Parse animations
        for anim_data in gltf.get('animations', []):
            clip = AnimationClip(
                name=anim_data.get('name', 'animation'),
                duration=0.0
            )
            
            # Map node indices to bone indices
            skin = gltf.get('skins', [{}])[0] if gltf.get('skins') else {}
            joint_indices = skin.get('joints', [])
            node_to_bone = {j: i for i, j in enumerate(joint_indices)}
            
            for channel_data in anim_data.get('channels', []):
                target = channel_data['target']
                node_idx = target.get('node')
                bone_idx = node_to_bone.get(node_idx, -1)
                
                if bone_idx < 0:
                    continue
                
                sampler = anim_data['samplers'][channel_data['sampler']]
                times = get_accessor_data(sampler['input'])
                values = get_accessor_data(sampler['output'])
                
                prop = target['path']  # translation, rotation, scale
                
                channel = AnimationChannel(
                    bone_index=bone_idx,
                    property_type=prop
                )
                
                # Determine values per keyframe
                if prop == 'rotation':
                    stride = 4
                elif prop in ('translation', 'scale'):
                    stride = 3
                else:
                    stride = 1
                
                for i, t in enumerate(times):
                    value = tuple(values[i * stride:(i + 1) * stride])
                    channel.keyframes.append(AnimationKeyframe(time=t, value=value))
                    clip.duration = max(clip.duration, t)
                
                clip.channels.append(channel)
            
            animations.append(clip)
        
        return meshes, bones, animations


class Avatar3DWidget(QOpenGLWidget):
    """
    OpenGL widget for rendering 3D avatar.
    Uses PyQt5's built-in OpenGL support.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.meshes: list[Mesh] = []
        self.bones: list[Bone] = []
        self.animations: dict[str, AnimationClip] = {}
        
        # Current state
        self._current_animation: Optional[str] = None
        self._animation_time: float = 0.0
        self._bone_matrices: list[QMatrix4x4] = []
        
        # Camera
        self._camera_pos = QVector3D(0, 0, 3)
        self._camera_target = QVector3D(0, 0.5, 0)
        
        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_animation)
        self._timer.start(33)  # ~30 FPS
        
        # Set format for transparency
        fmt = self.format()
        fmt.setAlphaBufferSize(8)
        self.setFormat(fmt)
        
        self.setAttribute(Qt.WA_TranslucentBackground)
    
    def load_model(self, path: str) -> bool:
        """Load a 3D model."""
        try:
            self.meshes, self.bones, animations = GLTFLoader.load(path)
            
            for clip in animations:
                self.animations[clip.name] = clip
            
            # Initialize bone matrices
            self._bone_matrices = [QMatrix4x4() for _ in self.bones]
            
            print(f"[3D Native] Loaded: {len(self.meshes)} meshes, {len(self.bones)} bones, {len(animations)} animations")
            print(f"[3D Native] Animations: {list(self.animations.keys())}")
            
            self.update()
            return True
            
        except Exception as e:
            print(f"[3D Native] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def play_animation(self, name: str, loop: bool = True):
        """Play an animation by name."""
        if name in self.animations:
            self._current_animation = name
            self.animations[name].loop = loop
            self._animation_time = 0.0
    
    def _update_animation(self):
        """Update animation frame."""
        if not self._current_animation or self._current_animation not in self.animations:
            return
        
        clip = self.animations[self._current_animation]
        
        # Advance time
        self._animation_time += 0.033  # ~30 FPS
        
        if self._animation_time > clip.duration:
            if clip.loop:
                self._animation_time = 0.0
            else:
                self._animation_time = clip.duration
                self._current_animation = None
                return
        
        # Update bone transforms
        self._update_bone_transforms(clip)
        self.update()
    
    def _update_bone_transforms(self, clip: AnimationClip):
        """Update bone matrices from animation."""
        for channel in clip.channels:
            if channel.bone_index >= len(self.bones):
                continue
            
            bone = self.bones[channel.bone_index]
            
            # Interpolate keyframes
            value = self._interpolate_keyframes(channel.keyframes, self._animation_time)
            
            if channel.property_type == 'translation':
                bone.local_position = value
            elif channel.property_type == 'rotation':
                bone.local_rotation = value
            elif channel.property_type == 'scale':
                bone.local_scale = value
        
        # Calculate final matrices
        self._calculate_bone_matrices()
    
    def set_bone_rotation(self, bone_name: str, pitch: float, yaw: float, roll: float) -> bool:
        """
        Set bone rotation directly (from BoneController).
        
        Converts euler angles to quaternion and applies to bone.
        
        Args:
            bone_name: Name of bone to rotate
            pitch: Rotation around X axis (degrees)
            yaw: Rotation around Y axis (degrees)
            roll: Rotation around Z axis (degrees)
            
        Returns:
            True if bone was found and rotated
        """
        # Find bone by name
        bone_idx = None
        for i, bone in enumerate(self.bones):
            if bone.name.lower() == bone_name.lower():
                bone_idx = i
                break
        
        if bone_idx is None:
            return False
        
        bone = self.bones[bone_idx]
        
        # Convert euler to quaternion
        # Order: pitch (X) -> yaw (Y) -> roll (Z)
        import math
        
        hp = math.radians(pitch) * 0.5
        hy = math.radians(yaw) * 0.5
        hr = math.radians(roll) * 0.5
        
        cp, sp = math.cos(hp), math.sin(hp)
        cy, sy = math.cos(hy), math.sin(hy)
        cr, sr = math.cos(hr), math.sin(hr)
        
        # Quaternion xyzw
        qx = sp * cy * cr - cp * sy * sr
        qy = cp * sy * cr + sp * cy * sr
        qz = cp * cy * sr - sp * sy * cr
        qw = cp * cy * cr + sp * sy * sr
        
        bone.local_rotation = (qx, qy, qz, qw)
        
        # Recalculate matrices and redraw
        self._calculate_bone_matrices()
        self.update()
        
        return True
    
    def get_bone_names(self) -> list[str]:
        """Get list of available bone names."""
        return [bone.name for bone in self.bones]
    
    def connect_bone_controller(self):
        """
        Connect to BoneController to receive rotation updates.
        
        Call this after loading a model to enable AI bone control.
        """
        try:
            from .bone_control import get_bone_controller
            controller = get_bone_controller()
            
            # Set available bones
            controller.set_avatar_bones(self.get_bone_names())
            
            # Register callback
            controller.add_callback(self._on_bone_update)
            print(f"[3D Native] Connected to BoneController with {len(self.bones)} bones")
            
        except Exception as e:
            print(f"[3D Native] Could not connect to BoneController: {e}")
    
    def _on_bone_update(self, bone_name: str, pitch: float, yaw: float, roll: float):
        """Callback from BoneController when a bone is moved."""
        self.set_bone_rotation(bone_name, pitch, yaw, roll)
    
    def _interpolate_keyframes(self, keyframes: list[AnimationKeyframe], time: float) -> tuple:
        """Linear interpolation between keyframes."""
        if not keyframes:
            return (0, 0, 0)
        
        if len(keyframes) == 1:
            return keyframes[0].value
        
        # Find surrounding keyframes
        prev_kf = keyframes[0]
        next_kf = keyframes[-1]
        
        for i, kf in enumerate(keyframes):
            if kf.time > time:
                next_kf = kf
                if i > 0:
                    prev_kf = keyframes[i - 1]
                break
            prev_kf = kf
        
        if prev_kf.time == next_kf.time:
            return prev_kf.value
        
        # Lerp factor
        t = (time - prev_kf.time) / (next_kf.time - prev_kf.time)
        t = max(0, min(1, t))
        
        # Interpolate
        return tuple(
            prev_kf.value[i] + t * (next_kf.value[i] - prev_kf.value[i])
            for i in range(len(prev_kf.value))
        )
    
    def _calculate_bone_matrices(self):
        """Calculate world matrices for all bones."""
        for i, bone in enumerate(self.bones):
            matrix = QMatrix4x4()
            
            # Local transform
            matrix.translate(*bone.local_position)
            
            # Rotation (quaternion xyzw)
            q = QQuaternion(bone.local_rotation[3], bone.local_rotation[0],
                           bone.local_rotation[1], bone.local_rotation[2])
            matrix.rotate(q)
            
            matrix.scale(*bone.local_scale)
            
            # Apply parent transform
            if bone.parent_index >= 0 and bone.parent_index < len(self._bone_matrices):
                matrix = self._bone_matrices[bone.parent_index] * matrix
            
            self._bone_matrices[i] = matrix
    
    def initializeGL(self):
        """Initialize OpenGL."""
        if HAS_OPENGL:
            glClearColor(0.0, 0.0, 0.0, 0.0)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    def resizeGL(self, w, h):
        """Handle resize."""
        if HAS_OPENGL:
            glViewport(0, 0, w, h)
            
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45.0, w / max(h, 1), 0.1, 100.0)
            
            glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        """Render the scene."""
        if not HAS_OPENGL:
            return
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Camera
        gluLookAt(
            self._camera_pos.x(), self._camera_pos.y(), self._camera_pos.z(),
            self._camera_target.x(), self._camera_target.y(), self._camera_target.z(),
            0, 1, 0
        )
        
        # Simple lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        
        # Material
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.7, 0.7, 0.8, 1])
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        
        # Draw meshes
        for mesh in self.meshes:
            self._draw_mesh(mesh)
    
    def _draw_mesh(self, mesh: Mesh):
        """Draw a single mesh."""
        if not mesh.vertices:
            return
        
        glBegin(GL_TRIANGLES)
        
        num_vertices = len(mesh.vertices) // 3
        
        for idx in mesh.indices if mesh.indices else range(num_vertices):
            if idx >= num_vertices:
                continue
            
            vi = idx * 3
            
            # Apply skeletal animation if we have bone data
            vx, vy, vz = mesh.vertices[vi], mesh.vertices[vi + 1], mesh.vertices[vi + 2]
            
            if mesh.bone_weights and mesh.bone_indices and self._bone_matrices:
                bi = idx * 4
                if bi + 3 < len(mesh.bone_indices) and bi + 3 < len(mesh.bone_weights):
                    # Transform vertex by bone weights
                    new_v = QVector3D(0, 0, 0)
                    for j in range(4):
                        bone_idx = int(mesh.bone_indices[bi + j])
                        weight = mesh.bone_weights[bi + j]
                        if weight > 0 and bone_idx < len(self._bone_matrices):
                            v = QVector3D(vx, vy, vz)
                            transformed = self._bone_matrices[bone_idx].map(v)
                            new_v += transformed * weight
                    vx, vy, vz = new_v.x(), new_v.y(), new_v.z()
            
            # Normal
            if mesh.normals and vi + 2 < len(mesh.normals):
                glNormal3f(mesh.normals[vi], mesh.normals[vi + 1], mesh.normals[vi + 2])
            
            glVertex3f(vx, vy, vz)
        
        glEnd()
    
    def capture_frame(self) -> Optional['QImage']:
        """
        Capture the current avatar frame as an image.
        
        This allows the AI to see what it looks like (self-vision).
        Used for embodied AI control feedback loop.
        
        Returns:
            QImage of current avatar state, or None if capture fails
        """
        try:
            # Force a render update
            self.update()
            self.repaint()
            
            # Grab the widget as a pixmap
            pixmap = self.grab()
            
            # Convert to QImage
            return pixmap.toImage()
        except Exception as e:
            print(f"[Avatar3D] Failed to capture frame: {e}")
            return None
    
    def capture_frame_bytes(self, format: str = "PNG") -> Optional[bytes]:
        """
        Capture current frame and return as bytes.
        
        Useful for sending to vision models or saving.
        
        Args:
            format: Image format (PNG, JPEG, etc.)
            
        Returns:
            Image bytes or None if capture fails
        """
        try:
            from PyQt5.QtCore import QBuffer, QIODevice
            
            image = self.capture_frame()
            if not image:
                return None
            
            # Convert to bytes
            buffer = QBuffer()
            buffer.open(QIODevice.WriteOnly)
            image.save(buffer, format)
            return bytes(buffer.data())
        except Exception as e:
            print(f"[Avatar3D] Failed to capture frame bytes: {e}")
            return None


class NativeAvatar3D(QObject):
    """
    Main 3D avatar class using native PyQt5 OpenGL.
    No external dependencies required.
    """
    
    state_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        self._widget: Optional[Avatar3DWidget] = None
        self._state_animations: dict[Animation3DState, str] = {}
        self._current_state = Animation3DState.IDLE
        self._gesture_queue: list[str] = []
        self._return_state = Animation3DState.IDLE
    
    def get_widget(self, width: int = 512, height: int = 512) -> QOpenGLWidget:
        """Get the OpenGL widget for embedding."""
        if not self._widget:
            self._widget = Avatar3DWidget()
            self._widget.setFixedSize(width, height)
        return self._widget
    
    def load_model(self, path: str, connect_bones: bool = True) -> bool:
        """
        Load a 3D model.
        
        Args:
            path: Path to GLB/GLTF file
            connect_bones: If True, connect to BoneController for AI control
            
        Returns:
            True if model loaded successfully
        """
        if not self._widget:
            self.get_widget()
        
        success = self._widget.load_model(path)
        if success and connect_bones:
            self._widget.connect_bone_controller()
        return success
    
    def get_bone_names(self) -> list[str]:
        """Get list of available bone names."""
        if self._widget:
            return self._widget.get_bone_names()
        return []
    
    def set_bone_rotation(self, bone_name: str, pitch: float = 0, yaw: float = 0, roll: float = 0) -> bool:
        """Set bone rotation directly (bypasses BoneController)."""
        if self._widget:
            return self._widget.set_bone_rotation(bone_name, pitch, yaw, roll)
        return False
    
    def map_state_to_animation(self, state: Animation3DState, animation_name: str):
        """Map state to animation name."""
        self._state_animations[state] = animation_name
    
    def set_state(self, state: Animation3DState):
        """Set current animation state (AI control)."""
        if state == self._current_state and state != Animation3DState.GESTURE:
            return
        
        self._current_state = state
        self.state_changed.emit(state.name)
        
        anim_name = self._state_animations.get(state)
        if anim_name and self._widget:
            self._widget.play_animation(anim_name, loop=(state != Animation3DState.GESTURE))
    
    def play_gesture(self, name: str, then_state: Animation3DState = Animation3DState.IDLE):
        """Play one-shot gesture."""
        self._return_state = then_state
        self._current_state = Animation3DState.GESTURE
        if self._widget:
            self._widget.play_animation(name, loop=False)
    
    def get_available_animations(self) -> list[str]:
        """Get list of loaded animations."""
        if self._widget:
            return list(self._widget.animations.keys())
        return []


class AI3DController:
    """AI controller for 3D avatar (same interface as 2D)."""
    
    def __init__(self, avatar: NativeAvatar3D):
        self.avatar = avatar
        self._is_talking = False
        self._current_emotion = "neutral"
    
    def start_talking(self):
        self._is_talking = True
        self.avatar.set_state(Animation3DState.TALKING)
    
    def stop_talking(self):
        self._is_talking = False
        self._apply_emotion()
    
    def set_emotion(self, emotion: str):
        self._current_emotion = emotion.lower()
        if not self._is_talking:
            self._apply_emotion()
    
    def _apply_emotion(self):
        emotion_map = {
            "neutral": Animation3DState.IDLE,
            "happy": Animation3DState.HAPPY,
            "sad": Animation3DState.SAD,
            "surprised": Animation3DState.SURPRISED,
            "thinking": Animation3DState.THINKING,
        }
        self.avatar.set_state(emotion_map.get(self._current_emotion, Animation3DState.IDLE))
    
    def gesture(self, name: str):
        emotion_map = {"neutral": Animation3DState.IDLE, "happy": Animation3DState.HAPPY}
        self.avatar.play_gesture(name, emotion_map.get(self._current_emotion, Animation3DState.IDLE))
    
    def listen(self):
        self.avatar.set_state(Animation3DState.LISTENING)
    
    def think(self):
        self.avatar.set_state(Animation3DState.THINKING)
