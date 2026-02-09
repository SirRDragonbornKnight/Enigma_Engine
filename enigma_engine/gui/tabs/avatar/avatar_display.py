"""
Avatar Display Module

Features:
  - 2D images (PNG, JPG) - lightweight display
  - 3D models (GLB, GLTF, OBJ, FBX) - optional OpenGL rendering
  - Desktop overlay (transparent, always on top, draggable)
  - Toggle between 2D preview and 3D rendering to save resources
  - Expression controls for live avatar expression changes
  - Color customization with presets
  - Avatar preset system for quick switching

Module Organization (7462+ lines, 10 classes):
=============================================
Lines 1-160:     Imports, constants, Qt flag compatibility
Lines 159-1244:  OpenGL3DWidget - 3D model rendering (~1086 lines)
                 - Sketchfab-style controls (orbit, zoom, pan)
                 - Lighting, wireframe, auto-rotate
Lines 1245-2003: AvatarOverlayWindow - Desktop overlay base (~759 lines)
Lines 2004-2171: DragBarWidget - Draggable title bar (~168 lines)
Lines 2172-2436: FloatingDragBar - Floating drag bar (~265 lines)
Lines 2437-2864: AvatarHitLayer - Click-through hit detection (~428 lines)
Lines 2865-3164: BoneHitRegion - Bone-specific click regions (~300 lines)
Lines 3165-3256: ResizeHandle - Corner resize handles (~92 lines)
Lines 3257-3507: BoneHitManager - Manages all hit regions (~251 lines)
Lines 3508-4778: Avatar3DOverlayWindow - 3D overlay window (~1271 lines)
Lines 4779-5800: AvatarPreviewWidget - Main preview widget (~1022 lines)
Lines 5800-7462: Helper functions (create_avatar_subtab, load/save, presets)
"""
# type: ignore[attr-defined]
# PyQt5 type stubs are incomplete; runtime works correctly

import logging
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Any, Dict, Optional

from PyQt5.QtCore import QByteArray, QPoint, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QBitmap,
    QColor,
    QCursor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QRegion,
    QWheelEvent,
)
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QOpenGLWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..shared_components import NoScrollComboBox

# Optional SVG support - not all PyQt5 installs have it
try:
    from PyQt5.QtSvg import QSvgRenderer, QSvgWidget
    HAS_SVG = True
except ImportError:
    QSvgWidget = None  # type: ignore
    QSvgRenderer = None  # type: ignore
    HAS_SVG = False

# Define Qt flags for compatibility with different PyQt5 versions
# These work at runtime even if type checker complains
Qt_FramelessWindowHint: Any = getattr(Qt, 'FramelessWindowHint', 0x00000800)
Qt_WindowStaysOnTopHint: Any = getattr(Qt, 'WindowStaysOnTopHint', 0x00040000)
Qt_Tool: Any = getattr(Qt, 'Tool', 0x00000008)
Qt_WA_TranslucentBackground: Any = getattr(Qt, 'WA_TranslucentBackground', 120)
Qt_WA_TransparentForMouseEvents: Any = getattr(Qt, 'WA_TransparentForMouseEvents', 51)
Qt_LeftButton: Any = getattr(Qt, 'LeftButton', 0x00000001)
Qt_RightButton: Any = getattr(Qt, 'RightButton', 0x00000002)
Qt_KeepAspectRatio: Any = getattr(Qt, 'KeepAspectRatio', 1)
Qt_SmoothTransformation: Any = getattr(Qt, 'SmoothTransformation', 1)
Qt_AlignCenter: Any = getattr(Qt, 'AlignCenter', 0x0084)
Qt_transparent: Any = getattr(Qt, 'transparent', QColor(0, 0, 0, 0))
Qt_NoPen: Any = getattr(Qt, 'NoPen', 0)
Qt_NoBrush: Any = getattr(Qt, 'NoBrush', 0)
Qt_OpenHandCursor: Any = getattr(Qt, 'OpenHandCursor', 17)
Qt_ClosedHandCursor: Any = getattr(Qt, 'ClosedHandCursor', 18)
Qt_ArrowCursor: Any = getattr(Qt, 'ArrowCursor', 0)
Qt_SizeHorCursor: Any = getattr(Qt, 'SizeHorCursor', 6)
Qt_SizeVerCursor: Any = getattr(Qt, 'SizeVerCursor', 7)
Qt_SizeFDiagCursor: Any = getattr(Qt, 'SizeFDiagCursor', 8)  # \ diagonal
Qt_SizeBDiagCursor: Any = getattr(Qt, 'SizeBDiagCursor', 9)  # / diagonal
Qt_ShiftModifier: Any = getattr(Qt, 'ShiftModifier', 0x02000000)
import json
import os
import time

from ....avatar import AvatarState, get_avatar
from ....avatar.customizer import AvatarCustomizer
from ....avatar.renderers.default_sprites import SPRITE_TEMPLATES, generate_sprite
from ....config import CONFIG

# Try importing 3D libraries
HAS_TRIMESH = False
HAS_OPENGL = False
trimesh = None
np = None

try:
    import numpy as _np
    import trimesh as _trimesh
    trimesh = _trimesh
    np = _np
    HAS_TRIMESH = True
except ImportError:
    pass

# OpenGL imports with explicit names to avoid wildcard import issues
try:
    import OpenGL.GL as GL
    import OpenGL.GLU as GLU
    HAS_OPENGL = True
except ImportError:
    GL = None  # type: ignore
    GLU = None  # type: ignore

# Supported file extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
MODEL_3D_EXTENSIONS = {'.glb', '.gltf', '.obj', '.fbx', '.dae'}
ALL_AVATAR_EXTENSIONS = IMAGE_EXTENSIONS | MODEL_3D_EXTENSIONS

# Avatar directories
AVATAR_CONFIG_DIR = Path(CONFIG["data_dir"]) / "avatar"
AVATAR_MODELS_DIR = AVATAR_CONFIG_DIR / "models"
AVATAR_IMAGES_DIR = AVATAR_CONFIG_DIR / "images"

# Activity log file for AI avatar commands
AVATAR_ACTIVITY_LOG = AVATAR_CONFIG_DIR / "activity_log.txt"


def _log_avatar_activity(action: str, value: str = ""):
    """Log an avatar command to the activity log file."""
    from datetime import datetime
    try:
        AVATAR_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {action}"
        if value:
            log_line += f": {value}"
        log_line += "\n"
        
        # Append to log file (keep last 50 lines)
        lines = []
        if AVATAR_ACTIVITY_LOG.exists():
            lines = AVATAR_ACTIVITY_LOG.read_text().splitlines()[-49:]
        lines.append(log_line.strip())
        AVATAR_ACTIVITY_LOG.write_text("\n".join(lines))
    except Exception:
        pass


class OpenGL3DWidget(QOpenGLWidget):
    """Sketchfab-style OpenGL widget for rendering 3D models.
    
    Features:
    - Dark gradient background (or transparent)
    - Grid floor (toggleable)
    - Smooth orbit controls (drag to rotate)
    - Scroll to zoom
    - Adjustable lighting
    - Auto-rotate option
    - Wireframe mode
    - Double-click to reset view
    """
    
    def __init__(self, parent=None, transparent_bg=False):
        super().__init__(parent)
        
        # Enable transparency for desktop overlay
        if transparent_bg:
            try:
                from PyQt5.QtGui import QSurfaceFormat
                fmt = QSurfaceFormat()
                fmt.setAlphaBufferSize(8)
                fmt.setSamples(4)  # Anti-aliasing
                self.setFormat(fmt)
            except Exception:
                pass  # Format not supported
        
        self.mesh = None
        self.vertices = None
        self.faces = None
        self.normals = None
        self.colors = None
        self.texture_colors = None  # Per-vertex colors from texture
        
        # Camera
        self.rotation_x = 15.0  # Slight tilt for better view
        self.rotation_y = 0.0  # Face forward (front-facing)
        self.zoom = 3.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        # Default camera settings (for reset)
        self._default_rotation_x = 15.0
        self._default_rotation_y = 0.0
        self._default_zoom = 3.0
        
        # Interaction
        self.last_pos = None
        self.is_panning = False
        
        # Auto-rotate
        self.auto_rotate = False
        self.auto_rotate_speed = 0.5
        self._rotate_timer = None
        
        # Display options
        self.transparent_bg = transparent_bg
        self.show_grid = True
        self.wireframe_mode = False
        self.ambient_strength = 0.15
        self.light_intensity = 1.0
        self.model_color = [0.75, 0.75, 0.82]  # Default color when no texture
        
        # Loading state
        self.is_loading = False
        self.model_name = ""
        self._model_path = None
        self._model_metadata = {}  # AI-readable model analysis
        
        # Model orientation (user-adjustable, saved per model)
        self.model_pitch = 0.0  # Rotation around X axis (radians)
        self.model_yaw = 0.0    # Rotation around Y axis (radians)
        self.model_roll = 0.0   # Rotation around Z axis (radians)
        
        # Flag to disable mouse interaction (used in overlay mode where parent handles mouse)
        self.disable_mouse_interaction = False
        
        self.setMinimumSize(250, 250)
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt_ArrowCursor))  # Use normal cursor, not grabbing hand
        
    def load_model(self, path: str) -> bool:
        """Load a 3D model file with texture support."""
        if not HAS_TRIMESH or trimesh is None or np is None:
            return False
        
        self.is_loading = True
        self.model_name = Path(path).stem
        self._model_path = path
        self.update()
        
        try:
            # Force texture resolver for GLTF files
            resolver = None
            model_dir = Path(path).parent
            
            # Create a resolver that can find textures in the model directory
            try:
                from trimesh.visual.resolvers import FilePathResolver
                resolver = FilePathResolver(str(model_dir))
            except ImportError:
                pass
            
            # Load with force='scene' to get materials properly
            if resolver:
                scene = trimesh.load(str(path), resolver=resolver, force='scene')
            else:
                scene = trimesh.load(str(path), force='scene')
            
            # Collect all meshes with their colors
            all_vertices = []
            all_faces = []
            all_normals = []
            all_colors = []
            vertex_offset = 0
            mesh_names_list = []  # Store mesh names for front/back detection
            
            # Get list of meshes (handle both Scene and single Mesh)
            if hasattr(scene, 'geometry') and scene.geometry:
                meshes = list(scene.geometry.values())
                # Also get mesh names from the geometry dict
                mesh_names_list = list(scene.geometry.keys())
            else:
                meshes = [scene]
            
            for mesh in meshes:
                if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                    continue
                    
                verts = np.array(mesh.vertices, dtype=np.float32)
                faces = np.array(mesh.faces, dtype=np.uint32) + vertex_offset
                
                all_vertices.append(verts)
                all_faces.append(faces)
                
                # Normals
                if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                    all_normals.append(np.array(mesh.vertex_normals, dtype=np.float32))
                else:
                    # Generate normals if missing
                    all_normals.append(np.zeros_like(verts))
                
                # Extract colors from this mesh's visual
                mesh_colors = self._extract_mesh_colors(mesh, len(verts))
                all_colors.append(mesh_colors)
                
                vertex_offset += len(verts)
            
            if not all_vertices:
                self.is_loading = False
                return False
            
            # Combine all meshes
            self.vertices = np.vstack(all_vertices)
            self.faces = np.vstack(all_faces)
            self.normals = np.vstack(all_normals)
            self.colors = np.vstack(all_colors) if all_colors else None
            
            # Store mesh names for front/back detection (before auto-orient is called)
            self._mesh_names = mesh_names_list
            
            # Check if we actually got colors
            if self.colors is not None:
                # Check if colors are all the same (default gray)
                unique_colors = np.unique(self.colors, axis=0)
                if len(unique_colors) <= 1:
                    print("[Avatar] Colors appear to be uniform - trying texture loading...")
                    loaded_colors = self._load_textures_from_files(path, meshes)
                    if loaded_colors is not None:
                        self.colors = loaded_colors
            
            # Center and scale the mesh
            centroid = self.vertices.mean(axis=0)
            self.vertices -= centroid
            max_extent = max(self.vertices.max(axis=0) - self.vertices.min(axis=0))
            if max_extent > 0:
                scale = 1.5 / max_extent
                self.vertices *= scale
            
            self.is_loading = False
            self.reset_view()
            
            # Try to load saved orientation for this model
            if not self.load_saved_orientation(path):
                # No saved orientation - apply auto-orientation as fallback
                self.apply_auto_orientation()
            
            # Log info
            has_real_colors = self.colors is not None and len(np.unique(self.colors, axis=0)) > 1
            print(f"[Avatar] Loaded {Path(path).name}: {len(self.vertices)} vertices, "
                  f"{len(self.faces)} faces, colors: {'yes (varied)' if has_real_colors else 'no/uniform'}")
            
            # Store model metadata for AI awareness
            self._model_metadata = self._analyze_model_structure(scene, meshes)
            
            return True
            
        except Exception as e:
            print(f"Error loading 3D model: {e}")
            import traceback
            traceback.print_exc()
            self.is_loading = False
            return False
    
    def _extract_mesh_colors(self, mesh, num_verts):
        """Extract colors from a single mesh's visual."""
        default_color = np.tile(self.model_color, (num_verts, 1)).astype(np.float32)
        
        if not hasattr(mesh, 'visual'):
            return default_color
        
        visual = mesh.visual
        
        # Method 1: Direct vertex colors (most reliable)
        if hasattr(visual, 'vertex_colors') and visual.vertex_colors is not None:
            vc = visual.vertex_colors
            if len(vc) == num_verts:
                colors = np.array(vc[:, :3] / 255.0, dtype=np.float32)
                # Check if colors are varied (not all the same)
                if len(np.unique(colors, axis=0)) > 1:
                    print(f"[Avatar] Got vertex colors from mesh")
                    return colors
        
        # Method 2: TextureVisuals - sample from texture using UVs
        if hasattr(visual, 'kind') and visual.kind == 'texture':
            try:
                # Get UV coordinates
                uv = None
                if hasattr(visual, 'uv') and visual.uv is not None:
                    uv = visual.uv
                
                # Get the texture image
                material = getattr(visual, 'material', None)
                img = None
                
                if material:
                    # Try different attribute names for the texture
                    for attr in ['baseColorTexture', 'image', 'diffuse']:
                        tex = getattr(material, attr, None)
                        if tex is not None:
                            img = tex
                            break
                
                if img is not None and uv is not None and len(uv) == num_verts:
                    from PIL import Image
                    if not isinstance(img, np.ndarray):
                        img = np.array(img)
                    
                    h, w = img.shape[:2]
                    
                    # Sample using UV (flip V for OpenGL convention)
                    u = np.clip(uv[:, 0] % 1.0, 0, 0.9999)
                    v = np.clip((1.0 - uv[:, 1]) % 1.0, 0, 0.9999)
                    
                    px = (u * w).astype(int)
                    py = (v * h).astype(int)
                    
                    if img.ndim == 3 and img.shape[2] >= 3:
                        colors = img[py, px, :3].astype(np.float32) / 255.0
                        print(f"[Avatar] Sampled {len(colors)} colors from texture ({w}x{h})")
                        return colors
            except Exception as e:
                print(f"[Avatar] Texture sampling error: {e}")
        
        # Method 3: Try trimesh's built-in to_color() conversion
        if hasattr(visual, 'to_color'):
            try:
                color_visual = visual.to_color()
                if hasattr(color_visual, 'vertex_colors') and color_visual.vertex_colors is not None:
                    vc = color_visual.vertex_colors
                    if len(vc) == num_verts:
                        colors = np.array(vc[:, :3] / 255.0, dtype=np.float32)
                        if len(np.unique(colors, axis=0)) > 1:
                            print(f"[Avatar] Got colors via to_color()")
                            return colors
            except Exception as e:
                print(f"[Avatar] to_color() failed: {e}")
        
        # Method 4: Material base color as fallback
        if hasattr(visual, 'material'):
            material = visual.material
            color = None
            
            if hasattr(material, 'main_color') and material.main_color is not None:
                color = np.array(material.main_color[:3]) / 255.0
            elif hasattr(material, 'baseColorFactor') and material.baseColorFactor is not None:
                color = np.array(material.baseColorFactor[:3])
                if np.max(color) > 1:
                    color = color / 255.0
            elif hasattr(material, 'diffuse') and material.diffuse is not None:
                color = np.array(material.diffuse[:3])
                if np.max(color) > 1:
                    color = color / 255.0
            
            if color is not None:
                print(f"[Avatar] Using material base color: {color}")
                return np.tile(color, (num_verts, 1)).astype(np.float32)
        
        return default_color
    
    def _load_textures_from_files(self, model_path, meshes):
        """Try to load textures directly from files in the model directory."""
        try:
            from PIL import Image
            model_dir = Path(model_path).parent
            
            # Look for texture files in multiple locations
            texture_dirs = [
                model_dir / "textures",
                model_dir / "texture", 
                model_dir / "tex",
                model_dir,
            ]
            
            # Also check numbered subdirectories (common in GLB/GLTF exports)
            for subdir in model_dir.iterdir():
                if subdir.is_dir() and subdir.name.isdigit():
                    texture_dirs.append(subdir)
            
            # Also check any immediate subdirectory that might have textures
            for subdir in model_dir.iterdir():
                if subdir.is_dir() and subdir.name not in ['textures', 'texture', 'tex']:
                    texture_dirs.append(subdir)
            
            texture_files = {}
            for tex_dir in texture_dirs:
                if tex_dir.exists():
                    for f in tex_dir.iterdir():
                        if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tga', '.bmp'}:
                            name = f.stem.lower()
                            texture_files[name] = f
            
            if not texture_files:
                print(f"[Avatar] No texture files found near {model_path}")
                return None
            
            print(f"[Avatar] Found {len(texture_files)} texture files: {list(texture_files.keys())[:5]}")
            
            # Find the best texture to use
            tex_img = None
            tex_name = None
            
            # Priority order for texture names
            priority_patterns = ['basecolor', 'diffuse', 'albedo', 'color', 'body', 'skin', 'face', 'main']
            
            for pattern in priority_patterns:
                for name, tex_path in texture_files.items():
                    if pattern in name:
                        try:
                            tex_img = Image.open(tex_path).convert('RGB')
                            tex_name = tex_path.name
                            print(f"[Avatar] Using texture (matched '{pattern}'): {tex_name}")
                            break
                        except Exception:
                            continue
                if tex_img:
                    break
            
            # If no priority match, use the largest texture file (likely the main one)
            if tex_img is None:
                largest_size = 0
                largest_path = None
                for name, tex_path in texture_files.items():
                    try:
                        size = tex_path.stat().st_size
                        if size > largest_size:
                            largest_size = size
                            largest_path = tex_path
                    except Exception:
                        continue
                
                if largest_path:
                    try:
                        tex_img = Image.open(largest_path).convert('RGB')
                        tex_name = largest_path.name
                        print(f"[Avatar] Using largest texture: {tex_name}")
                    except Exception:
                        pass
            
            if tex_img is None:
                return None
            
            img_array = np.array(tex_img)
            h, w = img_array.shape[:2]
            print(f"[Avatar] Texture size: {w}x{h}")
            
            # Apply texture to all meshes using their UV coordinates
            all_colors = []
            for mesh in meshes:
                if not hasattr(mesh, 'vertices'):
                    continue
                    
                num_verts = len(mesh.vertices)
                
                # Check if mesh has UV coordinates
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                    uv = mesh.visual.uv
                    if len(uv) == num_verts:
                        # Sample texture using UV coordinates
                        u = np.clip(uv[:, 0] % 1.0, 0, 0.9999)
                        v = np.clip((1.0 - uv[:, 1]) % 1.0, 0, 0.9999)  # Flip V
                        
                        px = (u * w).astype(int)
                        py = (v * h).astype(int)
                        
                        mesh_colors = img_array[py, px, :3].astype(np.float32) / 255.0
                        all_colors.append(mesh_colors)
                        continue
                
                # Fallback: use average texture color
                avg_color = img_array.mean(axis=(0, 1))[:3] / 255.0
                mesh_colors = np.tile(avg_color, (num_verts, 1)).astype(np.float32)
                all_colors.append(mesh_colors)
            
            if all_colors:
                result = np.vstack(all_colors).astype(np.float32)
                unique_count = len(np.unique(result, axis=0))
                print(f"[Avatar] Applied texture colors: {unique_count} unique colors")
                return result
            
            return None
            
        except Exception as e:
            print(f"[Avatar] Texture file loading failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _analyze_model_structure(self, scene, meshes) -> dict:
        """Analyze the 3D model structure for AI awareness.
        
        Returns a dictionary with:
        - Basic geometry info (vertices, faces, size)
        - Detected parts/meshes
        - Bone/skeleton info if available (GLTF/GLB armature)
        - Material info
        - Estimated model type (humanoid, object, etc.)
        """
        info = {
            'total_vertices': 0,
            'total_faces': 0,
            'mesh_count': len(meshes),
            'mesh_names': [],
            'bounding_box': None,
            'center': None,
            'size': None,
            'has_skeleton': False,
            'skeleton_bones': [],
            'materials': [],
            'estimated_type': 'unknown',
            'has_textures': False,
        }
        
        try:
            # Basic geometry
            if self.vertices is not None:
                info['total_vertices'] = len(self.vertices)
                info['bounding_box'] = {
                    'min': self.vertices.min(axis=0).tolist(),
                    'max': self.vertices.max(axis=0).tolist(),
                }
                info['center'] = self.vertices.mean(axis=0).tolist()
                dims = self.vertices.max(axis=0) - self.vertices.min(axis=0)
                info['size'] = {'width': float(dims[0]), 'height': float(dims[1]), 'depth': float(dims[2])}
            
            if self.faces is not None:
                info['total_faces'] = len(self.faces)
            
            # Mesh names
            for mesh in meshes:
                name = getattr(mesh, 'name', None) or getattr(mesh.metadata, 'name', None) if hasattr(mesh, 'metadata') else None
                if name:
                    info['mesh_names'].append(name)
            
            # Check for skeleton/armature in GLTF scenes
            if hasattr(scene, 'graph') and scene.graph is not None:
                try:
                    # GLTF scenes store skeleton info in the graph
                    graph = scene.graph
                    if hasattr(graph, 'nodes_geometry'):
                        # Check for nodes that might be bones
                        for node_name in graph.nodes:
                            node_name_lower = str(node_name).lower()
                            # Common bone naming patterns
                            if any(bone_keyword in node_name_lower for bone_keyword in [
                                'bone', 'joint', 'skeleton', 'armature', 'spine', 'hip', 'head',
                                'arm', 'leg', 'hand', 'foot', 'finger', 'neck', 'shoulder', 'elbow',
                                'knee', 'ankle', 'wrist', 'pelvis', 'chest', 'root'
                            ]):
                                info['skeleton_bones'].append(str(node_name))
                                info['has_skeleton'] = True
                except Exception as e:
                    print(f"[Avatar] Skeleton detection error: {e}")
            
            # Materials
            for mesh in meshes:
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                    mat = mesh.visual.material
                    mat_info = {'name': getattr(mat, 'name', 'unnamed')}
                    
                    # Check for textures
                    for tex_attr in ['baseColorTexture', 'image', 'diffuse']:
                        if getattr(mat, tex_attr, None) is not None:
                            info['has_textures'] = True
                            mat_info['has_texture'] = True
                            break
                    
                    info['materials'].append(mat_info)
            
            # Estimate model type based on proportions and structure
            if info['size']:
                w, h, d = info['size']['width'], info['size']['height'], info['size']['depth']
                
                # Humanoid: taller than wide, has skeleton or named bones
                if h > w * 1.5 and h > d * 1.5:
                    if info['has_skeleton'] or len(info['skeleton_bones']) > 0:
                        info['estimated_type'] = 'humanoid_rigged'
                    else:
                        info['estimated_type'] = 'humanoid_static'
                # Check mesh names for hints
                elif any('character' in name.lower() or 'body' in name.lower() or 'face' in name.lower() 
                        for name in info['mesh_names']):
                    info['estimated_type'] = 'character'
                # Roughly cubic = object
                elif abs(w - h) < w * 0.3 and abs(h - d) < h * 0.3:
                    info['estimated_type'] = 'object'
                else:
                    info['estimated_type'] = 'model'
            
            print(f"[Avatar] Model analysis: {info['estimated_type']}, "
                  f"{'skeleton with ' + str(len(info['skeleton_bones'])) + ' bones' if info['has_skeleton'] else 'no skeleton'}, "
                  f"{len(info['materials'])} materials")
            
        except Exception as e:
            print(f"[Avatar] Model analysis error: {e}")
        
        return info
    
    def get_model_metadata(self) -> dict:
        """Get the analyzed model metadata for AI use."""
        return getattr(self, '_model_metadata', {})
    
    def reset_view(self):
        """Reset camera to default position."""
        self.rotation_x = self._default_rotation_x
        self.rotation_y = self._default_rotation_y
        self.zoom = self._default_zoom
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()
    
    def auto_orient_model(self) -> tuple[float, float, float]:
        """Automatically detect and fix model orientation.
        
        Returns (pitch, yaw, roll) adjustments in degrees needed to make
        the model upright AND facing the camera. Uses:
        1. Bounding box analysis for up-axis
        2. Normal analysis for front/back detection
        3. Mesh name heuristics (face, eye, front markers)
        """
        if self.vertices is None or len(self.vertices) == 0:
            return (0.0, 0.0, 0.0)
        
        try:
            # Get bounding box dimensions
            mins = self.vertices.min(axis=0)
            maxs = self.vertices.max(axis=0)
            dims = maxs - mins  # [x_size, y_size, z_size]
            
            x_size, y_size, z_size = dims[0], dims[1], dims[2]
            
            pitch, yaw, roll = 0.0, 0.0, 0.0
            
            # === STEP 1: Detect up-axis (pitch/roll correction) ===
            
            # Standard Y-up (most common, no rotation needed)
            if y_size >= x_size and y_size >= z_size:
                print(f"[Avatar] Auto-orient: Model appears Y-up (correct), dims={dims}")
            
            # Z-up (Blender, some game engines) - rotate 90° around X
            elif z_size > y_size and z_size > x_size:
                print(f"[Avatar] Auto-orient: Model appears Z-up, rotating -90° X")
                pitch = -90.0
            
            # X-up (unusual but possible) - rotate 90° around Z
            elif x_size > y_size and x_size > z_size:
                print(f"[Avatar] Auto-orient: Model appears X-up, rotating 90° Z")
                roll = 90.0
            
            # If model is lying flat (wider than tall)
            horizontal_size = max(x_size, z_size)
            if horizontal_size > y_size * 1.5:
                if z_size > x_size:
                    print(f"[Avatar] Auto-orient: Model appears lying forward, rotating -90° X")
                    pitch = -90.0
                else:
                    print(f"[Avatar] Auto-orient: Model appears lying sideways, rotating 90° Z")
                    roll = 90.0
            
            # === STEP 2: Detect front/back facing (yaw correction) ===
            yaw = self._detect_facing_direction()
            
            if yaw != 0:
                print(f"[Avatar] Auto-orient: Detected backward facing, rotating {yaw}° Y")
            
            return (pitch, yaw, roll)
            
        except Exception as e:
            print(f"[Avatar] Auto-orient error: {e}")
            return (0.0, 0.0, 0.0)
    
    def _detect_facing_direction(self) -> float:
        """Detect if model is facing backward and needs 180° yaw rotation.
        
        Uses multiple heuristics:
        1. Mesh name analysis (look for "face", "eye", "front" keywords)
        2. Normal distribution analysis (where do most normals point?)
        3. Vertex density in upper region (faces have more detail)
        
        Returns: yaw adjustment in degrees (0 or 180)
        """
        yaw_correction = 0.0
        confidence_score = 0
        
        try:
            # === Heuristic 1: Mesh name analysis ===
            front_indicators = ['face', 'eye', 'nose', 'mouth', 'front', 'head', 'visor', 'lens']
            back_indicators = ['back', 'spine', 'rear', 'tail', 'behind']
            
            mesh_names = getattr(self, '_mesh_names', []) or []
            if hasattr(self, '_model_metadata') and self._model_metadata:
                mesh_names = self._model_metadata.get('mesh_names', [])
            
            front_meshes = []
            back_meshes = []
            
            for name in mesh_names:
                name_lower = str(name).lower()
                if any(indicator in name_lower for indicator in front_indicators):
                    front_meshes.append(name)
                if any(indicator in name_lower for indicator in back_indicators):
                    back_meshes.append(name)
            
            # If we find "face" or "eye" meshes, check their centroid position
            if front_meshes:
                print(f"[Avatar] Front-detection: Found front-indicator meshes: {front_meshes}")
                confidence_score += 1
            
            # === Heuristic 2: Normal distribution analysis ===
            if self.normals is not None and len(self.normals) > 0:
                # Normalize normals
                norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                normalized_normals = self.normals / norms
                
                # Focus on upper half of model (more likely to be face/front)
                if self.vertices is not None:
                    y_center = self.vertices[:, 1].mean()
                    upper_mask = self.vertices[:, 1] > y_center
                    
                    if upper_mask.sum() > 10:  # Need enough vertices
                        upper_normals = normalized_normals[upper_mask]
                        
                        # Average normal direction in upper region
                        avg_normal = upper_normals.mean(axis=0)
                        
                        # For most character models:
                        # - If facing viewer (+Z toward camera), average Z normal is positive
                        # - If facing away (-Z toward camera), average Z normal is negative
                        avg_z = avg_normal[2] if len(avg_normal) > 2 else 0
                        
                        print(f"[Avatar] Front-detection: Upper region avg normal Z = {avg_z:.3f}")
                        
                        # If most normals point backward (negative Z), model is facing away
                        if avg_z < -0.1:  # Threshold: significant negative Z
                            confidence_score += 2  # Normals are strong indicator
                            print(f"[Avatar] Front-detection: Normals suggest model faces -Z (backward)")
            
            # === Heuristic 3: Vertex density in upper front ===
            # Character faces have more vertices than backs of heads
            if self.vertices is not None and len(self.vertices) > 100:
                # Split into front (+Z) and back (-Z) halves
                z_center = self.vertices[:, 2].mean()
                y_center = self.vertices[:, 1].mean()
                
                # Count vertices in upper-front vs upper-back
                upper_front = ((self.vertices[:, 1] > y_center) & (self.vertices[:, 2] > z_center)).sum()
                upper_back = ((self.vertices[:, 1] > y_center) & (self.vertices[:, 2] <= z_center)).sum()
                
                if upper_front > 0 and upper_back > 0:
                    ratio = upper_back / upper_front
                    print(f"[Avatar] Front-detection: Upper back/front vertex ratio = {ratio:.2f}")
                    
                    # If there's significantly more detail in the "back" half, model is facing away
                    if ratio > 1.5:  # 50% more vertices in back half
                        confidence_score += 1
                        print(f"[Avatar] Front-detection: More detail in back suggests backward facing")
            
            # === Decision ===
            if confidence_score >= 2:
                print(f"[Avatar] Front-detection: Confidence {confidence_score}/4 - applying 180° yaw")
                yaw_correction = 180.0
            else:
                print(f"[Avatar] Front-detection: Confidence {confidence_score}/4 - no yaw correction")
            
        except Exception as e:
            print(f"[Avatar] Front-detection error: {e}")
        
        return yaw_correction
    
    def apply_auto_orientation(self):
        """Detect and apply automatic orientation correction."""
        pitch, yaw, roll = self.auto_orient_model()
        if pitch != 0 or yaw != 0 or roll != 0:
            self.model_pitch = np.radians(pitch)
            self.model_yaw = np.radians(yaw)
            self.model_roll = np.radians(roll)
            self.update()
            return True
        return False
    
    def load_saved_orientation(self, model_path: str = None) -> bool:
        """Load saved orientation for this model from JSON file."""
        import json
        import math
        
        path = model_path or self._model_path
        if not path:
            return False
        
        settings_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "avatar" / "model_orientations.json"
        if not settings_path.exists():
            return False
        
        try:
            with open(settings_path) as f:
                orientations = json.load(f)
            
            model_key = Path(path).name
            if model_key in orientations:
                data = orientations[model_key]
                self.model_pitch = math.radians(data.get('pitch', 0))
                self.model_yaw = math.radians(data.get('yaw', 0))
                self.model_roll = math.radians(data.get('roll', 0))
                print(f"[Avatar] Loaded saved orientation for {model_key}: pitch={data.get('pitch', 0)}°, yaw={data.get('yaw', 0)}°, roll={data.get('roll', 0)}°")
                self.update()
                return True
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Avatar] Could not load orientation: {e}")
        
        return False
    
    def reset_all(self):
        """Reset everything including display settings."""
        self.reset_view()
        self.auto_rotate = False
        self.wireframe_mode = False
        self.show_grid = True
        self.ambient_strength = 0.15
        self.light_intensity = 1.0
        self.model_color = [0.75, 0.75, 0.82]
        if self._rotate_timer:
            self._rotate_timer.stop()
        self.update()
    
    def start_auto_rotate(self):
        """Start auto-rotation."""
        self.auto_rotate = True
        if self._rotate_timer is None:
            self._rotate_timer = QTimer()
            self._rotate_timer.timeout.connect(self._do_auto_rotate)
        self._rotate_timer.start(16)  # ~60fps
    
    def stop_auto_rotate(self):
        """Stop auto-rotation."""
        self.auto_rotate = False
        if self._rotate_timer:
            self._rotate_timer.stop()
    
    def _do_auto_rotate(self):
        """Auto-rotate step."""
        if self.auto_rotate:
            self.rotation_y += self.auto_rotate_speed
            self.update()
    
    def initializeGL(self):
        """Initialize OpenGL with Sketchfab-style settings."""
        if not HAS_OPENGL or GL is None:
            return
        
        try:
            # Transparent or dark background
            if self.transparent_bg:
                GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            else:
                GL.glClearColor(0.08, 0.08, 0.12, 1.0)
            
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_LIGHTING)
            GL.glEnable(GL.GL_LIGHT0)
            GL.glEnable(GL.GL_LIGHT1)  # Rim/fill light
            GL.glEnable(GL.GL_COLOR_MATERIAL)
            GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
            
            # Enable smooth shading
            GL.glShadeModel(GL.GL_SMOOTH)
            
            # Enable blending for transparency
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            
            self._update_lighting()
            
            # Anti-aliasing (may not be supported on all systems)
            try:
                GL.glEnable(GL.GL_LINE_SMOOTH)
                GL.glEnable(GL.GL_POLYGON_SMOOTH)
                GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
                GL.glHint(GL.GL_POLYGON_SMOOTH_HINT, GL.GL_NICEST)
            except Exception:
                pass  # Smoothing not supported
                
            self._gl_initialized = True
        except Exception as e:
            print(f"OpenGL init error (may still work): {e}")
            self._gl_initialized = False
    
    def _update_lighting(self):
        """Update lighting based on current settings."""
        if not HAS_OPENGL or GL is None:
            return
        
        intensity = self.light_intensity
        ambient = self.ambient_strength
        
        # Key light (warm, from top-right-front)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [2.0, 3.0, 2.0, 0.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [intensity, intensity * 0.95, intensity * 0.9, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, [ambient, ambient, ambient * 1.2, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
        # Fill/rim light (cool, from bottom-left-back)
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_POSITION, [-2.0, -1.0, -2.0, 0.0])
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE, [intensity * 0.3, intensity * 0.35, intensity * 0.5, 1.0])
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
        
        # Material properties
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
        GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 30.0)
        
    def resizeGL(self, w, h):
        """Handle resize."""
        if not HAS_OPENGL or GL is None or GLU is None:
            return
        GL.glViewport(0, 0, w, h)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        aspect = w / h if h > 0 else 1
        GLU.gluPerspective(35, aspect, 0.1, 100.0)  # Narrower FOV for less distortion
        GL.glMatrixMode(GL.GL_MODELVIEW)
        
    def paintGL(self):
        """Render with Sketchfab-style visuals."""
        if not HAS_OPENGL or GL is None:
            return
        
        try:
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            
            # Draw gradient background (skip if transparent)
            if not self.transparent_bg:
                self._draw_gradient_background()
            
            # Draw grid floor (if enabled and not transparent)
            if self.show_grid and not self.transparent_bg:
                self._draw_grid()
            
            GL.glLoadIdentity()
            
            # Camera
            GL.glTranslatef(self.pan_x, self.pan_y, -self.zoom)
            GL.glRotatef(self.rotation_x, 1, 0, 0)
            GL.glRotatef(self.rotation_y, 0, 1, 0)
            
            # Apply model orientation (user-adjustable)
            import math
            GL.glRotatef(math.degrees(self.model_pitch), 1, 0, 0)  # Pitch (X axis)
            GL.glRotatef(math.degrees(self.model_yaw), 0, 1, 0)    # Yaw (Y axis)
            GL.glRotatef(math.degrees(self.model_roll), 0, 0, 1)   # Roll (Z axis)
            
            if self.is_loading:
                # Draw loading indicator
                GL.glDisable(GL.GL_LIGHTING)
                GL.glColor3f(0.5, 0.5, 0.6)
                GL.glEnable(GL.GL_LIGHTING)
                return
            
            if self.vertices is not None and self.faces is not None:
                # Wireframe mode
                if self.wireframe_mode:
                    GL.glDisable(GL.GL_LIGHTING)
                    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
                    GL.glColor3f(0.4, 0.6, 0.9)  # Blue wireframe
                else:
                    GL.glEnable(GL.GL_LIGHTING)
                    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
                
                # Use vertex colors if available, otherwise model_color
                if self.colors is not None and not self.wireframe_mode:
                    GL.glEnableClientState(GL.GL_COLOR_ARRAY)
                    GL.glColorPointer(3, GL.GL_FLOAT, 0, self.colors)
                else:
                    GL.glColor3f(*self.model_color)
                
                GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
                GL.glVertexPointer(3, GL.GL_FLOAT, 0, self.vertices)
                
                if self.normals is not None:
                    GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
                    GL.glNormalPointer(GL.GL_FLOAT, 0, self.normals)
                
                GL.glDrawElements(GL.GL_TRIANGLES, len(self.faces) * 3, GL.GL_UNSIGNED_INT, self.faces)
                
                GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
                if self.normals is not None:
                    GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
                if self.colors is not None and not self.wireframe_mode:
                    GL.glDisableClientState(GL.GL_COLOR_ARRAY)
                    
                # Reset polygon mode
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        except Exception as e:
            # OpenGL error - may happen on some systems
            if not getattr(self, '_paint_error_logged', False):
                print(f"OpenGL paint error: {e}")
                self._paint_error_logged = True
    
    def _draw_gradient_background(self):
        """Draw Sketchfab-style gradient background."""
        if not HAS_OPENGL or GL is None:
            return
        
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_LIGHTING)
        
        GL.glBegin(GL.GL_QUADS)
        # Top - darker
        GL.glColor3f(0.06, 0.06, 0.10)
        GL.glVertex2f(-1, 1)
        GL.glVertex2f(1, 1)
        # Bottom - slightly lighter
        GL.glColor3f(0.12, 0.12, 0.18)
        GL.glVertex2f(1, -1)
        GL.glVertex2f(-1, -1)
        GL.glEnd()
        
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)
        
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()
    
    def _draw_grid(self):
        """Draw Sketchfab-style grid floor."""
        if not HAS_OPENGL or GL is None:
            return
        
        GL.glDisable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        
        GL.glPushMatrix()
        GL.glTranslatef(self.pan_x, self.pan_y, -self.zoom)
        GL.glRotatef(self.rotation_x, 1, 0, 0)
        GL.glRotatef(self.rotation_y, 0, 1, 0)
        
        # Grid at y = -0.8
        grid_y = -0.8
        grid_size = 3.0
        grid_step = 0.25
        
        GL.glBegin(GL.GL_LINES)
        
        # Draw grid lines with fade
        steps = int(grid_size / grid_step)
        for i in range(-steps, steps + 1):
            x = i * grid_step
            # Fade based on distance from center
            dist = abs(i) / steps
            alpha = max(0.05, 0.2 * (1 - dist))
            
            GL.glColor4f(0.3, 0.35, 0.45, alpha)
            GL.glVertex3f(x, grid_y, -grid_size)
            GL.glVertex3f(x, grid_y, grid_size)
            
            GL.glVertex3f(-grid_size, grid_y, x)
            GL.glVertex3f(grid_size, grid_y, x)
        
        GL.glEnd()
        
        GL.glPopMatrix()
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_LIGHTING)
    
    def mousePressEvent(self, event):
        """Start drag or pan. Left-click to rotate, Right-click to orbit (Sketchfab style), Shift+Left to pan."""
        # If mouse interaction disabled (overlay mode), ignore
        if getattr(self, 'disable_mouse_interaction', False):
            event.ignore()
            return
        self.last_pos = event.pos()
        # Right-click OR left-click rotates (Sketchfab lets you orbit with right-click)
        if event.button() == Qt_LeftButton:
            self.is_panning = event.modifiers() == Qt.ShiftModifier if hasattr(Qt, 'ShiftModifier') else False
            # Don't change cursor - keep normal arrow
        elif event.button() == Qt.RightButton if hasattr(Qt, 'RightButton') else 0x00000002:
            # Right-click = orbit (rotation) - Sketchfab style
            self.is_panning = False
            # Don't change cursor - keep normal arrow
        event.accept()
        
    def mouseMoveEvent(self, event):
        """Rotate or pan on drag. Works with both left and right mouse buttons."""
        # If mouse interaction disabled (overlay mode), ignore
        if getattr(self, 'disable_mouse_interaction', False):
            event.ignore()
            return
        if self.last_pos is not None:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            
            # Check which buttons are pressed
            buttons = event.buttons()
            right_button = Qt.RightButton if hasattr(Qt, 'RightButton') else 0x00000002
            
            if self.is_panning:
                # Pan (Shift+Left click)
                self.pan_x += dx * 0.005
                self.pan_y -= dy * 0.005
            elif (buttons & right_button) or (buttons & Qt_LeftButton):
                # Rotate - full 360 degree rotation allowed (left OR right click)
                self.rotation_y += dx * 0.5
                self.rotation_x += dy * 0.5
            
            self.last_pos = event.pos()
            self.update()
        event.accept()
            
    def mouseReleaseEvent(self, event):
        """End drag."""
        # If mouse interaction disabled (overlay mode), ignore
        if getattr(self, 'disable_mouse_interaction', False):
            event.ignore()
            return
        self.last_pos = None
        self.is_panning = False
        self.setCursor(QCursor(Qt_ArrowCursor))  # Keep normal cursor
        event.accept()
        
    def wheelEvent(self, event):
        """Zoom with scroll wheel."""
        # If mouse interaction disabled (overlay mode), ignore
        if getattr(self, 'disable_mouse_interaction', False):
            event.ignore()
            return
        delta = event.angleDelta().y() / 120
        self.zoom = max(1.0, min(15.0, self.zoom - delta * 0.3))
        self.update()
        event.accept()
        
    def mouseDoubleClickEvent(self, event):
        """Reset view on double click."""
        self.reset_view()
        event.accept()


class AvatarOverlayWindow(QWidget):
    """Transparent overlay window for desktop avatar display.
    
    Features:
    - Drag anywhere to move
    - Right-click for menu (expressions, size, close)
    - Drag from edges/corners to resize (when enabled)
    - Always on top of other windows
    - Border wraps TIGHTLY around the avatar image
    - Blue border shows when resize mode is ON
    - Touch reactions: tap, double-tap, hold, pet (repeated taps)
    """
    
    closed = pyqtSignal()
    
    # Touch reaction signal: (touch_type, global_pos)
    # touch_type: 'tap', 'double_tap', 'hold', 'pet' (repeated taps)
    touched = pyqtSignal(str, object)
    
    # Touch detection constants
    HOLD_THRESHOLD_MS = 500  # Milliseconds to hold for 'hold' touch type
    
    def __init__(self):
        super().__init__(None)
        
        # Transparent, always-on-top, no taskbar, but accept mouse input
        self.setWindowFlags(
            Qt_FramelessWindowHint |
            Qt_WindowStaysOnTopHint
        )
        self.setAttribute(Qt_WA_TranslucentBackground, True)
        # Important: Make sure we receive mouse events
        self.setAttribute(Qt.WA_NoSystemBackground, True) if hasattr(Qt, 'WA_NoSystemBackground') else None
        
        # IMPORTANT: Don't set WA_TransparentForMouseEvents - we want clicks!
        # Ensure the window can receive focus and input
        self.setFocusPolicy(Qt.StrongFocus if hasattr(Qt, 'StrongFocus') else 0x0b)
        
        # _size is the TARGET maximum dimension for scaling (not window size!)
        self._size = 300
        self._last_saved_size = 300  # Track last saved size to avoid redundant saves
        # DON'T call setFixedSize here - let set_avatar do it based on actual image
        self.move(100, 100)
        
        self.pixmap = None
        self._original_pixmap = None
        self._drag_pos = None
        self._resize_enabled = False  # Default OFF - user must enable via right-click
        self._reposition_enabled = True  # Default ON - allow dragging to move
        
        # Current avatar path (for per-avatar settings)
        self._avatar_path = None
        
        # Resize state for edge-drag resizing
        self._resize_edge = None  # Which edge is being dragged
        self._resize_start_pos = None
        self._resize_start_size = None
        self._resize_start_geo = None  # Starting geometry for position adjustments
        self._edge_margin = 20  # Pixels from edge to trigger resize (larger for easier grabbing)
        
        # Rotation state for Shift+drag rotation
        self._rotation_angle = 0.0  # Current rotation in degrees
        self._rotate_start_x = None  # Starting X for rotation drag
        
        # Eye tracking state
        self._eye_tracking_enabled = False
        self._eye_offset_x = 0.0  # Eye offset from center (-1 to 1)
        self._eye_offset_y = 0.0
        self._eye_timer = None  # Timer for smooth tracking
        
        # Enable mouse tracking for resize cursor feedback
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt_ArrowCursor))
        
        # Auto-save timer for size changes (saves every 2 seconds if changed)
        from PyQt5.QtCore import QTimer
        self._save_timer = QTimer(self)
        self._save_timer.timeout.connect(self._auto_save_size)
        self._save_timer.start(2000)  # Check every 2 seconds
        
        # Pixel-based hit testing - transparent areas pass through clicks
        self._use_pixel_hit_test = True
        self._is_dragging = False
        
        # Touch detection state
        self._touch_press_time = 0
        self._touch_press_pos = None
        self._touch_moved = False
        self._last_tap_time = 0
        self._tap_count = 0
        
        # Hold detection timer
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._on_hold_detected)
        
        # Pet detection timer (resets tap count after delay)
        self._pet_timer = QTimer(self)
        self._pet_timer.setSingleShot(True)
        self._pet_timer.timeout.connect(self._on_pet_timeout)
        
        # Connect touch signal to write events for AI
        self.touched.connect(self._on_touched)
        
        # File watcher for hot-swap (reload avatar when file changes)
        from PyQt5.QtCore import QFileSystemWatcher
        self._file_watcher = QFileSystemWatcher(self)
        self._file_watcher.fileChanged.connect(self._on_avatar_file_changed)
        self._hotswap_enabled = True  # Enable by default
        self._hotswap_debounce_timer = QTimer(self)
        self._hotswap_debounce_timer.setSingleShot(True)
        self._hotswap_debounce_timer.timeout.connect(self._do_hotswap_reload)
        self._pending_hotswap_path = None
        
        # Crossfade state
        self._crossfade_active = False
        self._crossfade_old_pixmap = None
        self._crossfade_progress = 1.0  # 0.0 = old, 1.0 = new
        self._crossfade_timer = QTimer(self)
        self._crossfade_timer.timeout.connect(self._update_crossfade)
        
        # Make window layered for transparency (Windows)
        import sys
        if sys.platform == 'win32':
            try:
                import ctypes
                hwnd = int(self.winId())
                GWL_EXSTYLE = -20
                WS_EX_LAYERED = 0x80000
                user32 = ctypes.windll.user32
                current_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                user32.SetWindowLongW(hwnd, GWL_EXSTYLE, current_style | WS_EX_LAYERED)
            except Exception:
                pass
    
    def set_eye_tracking(self, enabled: bool):
        """Enable or disable eye tracking (follow cursor)."""
        self._eye_tracking_enabled = enabled
        
        if enabled:
            # Start eye tracking timer
            if self._eye_timer is None:
                self._eye_timer = QTimer(self)
                self._eye_timer.timeout.connect(self._update_eye_tracking)
            self._eye_timer.start(50)  # Update 20 times per second
        else:
            # Stop and reset
            if self._eye_timer:
                self._eye_timer.stop()
            self._eye_offset_x = 0.0
            self._eye_offset_y = 0.0
            self.update()
    
    def _on_hold_detected(self):
        """Called when user holds on avatar long enough."""
        if self._touch_press_pos and not self._touch_moved:
            self.touched.emit('hold', self._touch_press_pos)
    
    def _on_pet_timeout(self):
        """Called after delay to reset tap count."""
        if self._tap_count >= 3:
            # Already emitted 'pet' on third tap
            pass
        self._tap_count = 0
    
    def _on_touched(self, touch_type: str, global_pos):
        """Handle touch event - write to file for AI to read."""
        try:
            from ....avatar.persistence import write_touch_event_for_ai
            write_touch_event_for_ai(touch_type, region='avatar')
        except Exception as e:
            print(f"[Touch] Failed to write touch event: {e}")
    
    def _check_tap_type(self, global_pos):
        """Determine tap type based on timing."""
        import time
        current_time = time.time()
        
        # Check for double tap (two taps within 400ms)
        if current_time - self._last_tap_time < 0.4:
            self._tap_count += 1
            if self._tap_count == 2:
                self.touched.emit('double_tap', global_pos)
            elif self._tap_count >= 3:
                # Repeated taps = petting (headpats!)
                self.touched.emit('pet', global_pos)
        else:
            # Single tap
            self._tap_count = 1
            self.touched.emit('tap', global_pos)
        
        self._last_tap_time = current_time
        
        # Reset pet timer
        self._pet_timer.stop()
        self._pet_timer.start(600)  # Reset tap count after 600ms of no taps

    def _update_eye_tracking(self):
        """Update eye position based on cursor location."""
        if not self._eye_tracking_enabled:
            return
        
        # Get global cursor position
        cursor_pos = QCursor.pos()
        
        # Get avatar center position (global)
        avatar_center = self.mapToGlobal(QPoint(self.width() // 2, self.height() // 2))
        
        # Calculate direction to cursor
        dx = cursor_pos.x() - avatar_center.x()
        dy = cursor_pos.y() - avatar_center.y()
        
        # Normalize to -1 to 1 range (with max distance of 500 pixels)
        max_dist = 500.0
        self._eye_offset_x = max(-1.0, min(1.0, dx / max_dist))
        self._eye_offset_y = max(-1.0, min(1.0, dy / max_dist))
        
        # Trigger repaint to show the shift
        self.update()
    
    def set_avatar_path(self, path: str):
        """Set the current avatar path and load per-avatar settings."""
        # Stop watching old file
        if self._avatar_path and hasattr(self, '_file_watcher'):
            old_files = self._file_watcher.files()
            if old_files:
                self._file_watcher.removePaths(old_files)
        
        self._avatar_path = path
        
        # Start watching new file for hot-swap
        if path and hasattr(self, '_file_watcher') and self._hotswap_enabled:
            from pathlib import Path
            if Path(path).exists():
                self._file_watcher.addPath(path)
        
        # Load per-avatar size and position
        try:
            from ....avatar.persistence import load_avatar_settings
            settings = load_avatar_settings()
            self._size = settings.get_size_for_avatar(path)
            self._last_saved_size = self._size  # Track loaded size
            # DON'T setFixedSize here - _update_scaled_pixmap will handle it
            x, y = settings.get_position_for_avatar(path)
            self.move(x, y)
            self._update_scaled_pixmap()
        except Exception:
            pass
    
    def set_hotswap_enabled(self, enabled: bool):
        """Enable or disable avatar hot-swap (auto-reload on file change)."""
        self._hotswap_enabled = enabled
        if not enabled:
            # Stop watching files
            if hasattr(self, '_file_watcher'):
                old_files = self._file_watcher.files()
                if old_files:
                    self._file_watcher.removePaths(old_files)
        elif self._avatar_path:
            # Start watching current file
            from pathlib import Path
            if Path(self._avatar_path).exists():
                self._file_watcher.addPath(self._avatar_path)
    
    def _on_avatar_file_changed(self, path: str):
        """Handle avatar file change - debounce and reload."""
        if not self._hotswap_enabled:
            return
        
        # Debounce rapid file changes (like during save)
        self._pending_hotswap_path = path
        self._hotswap_debounce_timer.stop()
        self._hotswap_debounce_timer.start(300)  # Wait 300ms for file to settle
    
    def _do_hotswap_reload(self):
        """Actually reload the avatar after debounce."""
        path = self._pending_hotswap_path
        if not path:
            return
        
        self._pending_hotswap_path = None
        
        from pathlib import Path
        if not Path(path).exists():
            return
        
        print(f"[Avatar] Hot-swapping avatar: {path}")
        
        # Load new pixmap
        try:
            new_pixmap = QPixmap(path)
            if new_pixmap.isNull():
                return
            
            # Start crossfade if enabled
            if hasattr(self, '_crossfade_timer'):
                self._start_crossfade(new_pixmap)
            else:
                # Immediate swap
                self._original_pixmap = new_pixmap
                self._update_scaled_pixmap()
            
            # Re-add to watcher (file changes remove the watch)
            if hasattr(self, '_file_watcher') and self._hotswap_enabled:
                self._file_watcher.addPath(path)
                
        except Exception as e:
            print(f"[Avatar] Hot-swap failed: {e}")
    
    def _start_crossfade(self, new_pixmap: QPixmap):
        """Start crossfade animation from current to new avatar."""
        # Store old pixmap for blending
        if self.pixmap and not self.pixmap.isNull():
            self._crossfade_old_pixmap = self.pixmap.copy()
        else:
            self._crossfade_old_pixmap = None
        
        # Set new avatar
        self._original_pixmap = new_pixmap
        self._update_scaled_pixmap()
        
        # Start crossfade animation
        self._crossfade_active = True
        self._crossfade_progress = 0.0
        self._crossfade_timer.start(16)  # ~60 FPS
    
    def _update_crossfade(self):
        """Update crossfade animation progress."""
        if not self._crossfade_active:
            self._crossfade_timer.stop()
            return
        
        # Advance progress
        self._crossfade_progress += 0.05  # 20 frames total (~320ms)
        
        if self._crossfade_progress >= 1.0:
            # Crossfade complete
            self._crossfade_progress = 1.0
            self._crossfade_active = False
            self._crossfade_old_pixmap = None
            self._crossfade_timer.stop()
        
        self.update()  # Trigger repaint
        
    def set_avatar(self, pixmap: QPixmap):
        """Set avatar image and resize window to wrap tightly around it."""
        self._original_pixmap = pixmap
        self._update_scaled_pixmap()
    
    def _get_edge_at_pos(self, pos):
        """Get which edge the mouse is near (for resize cursor).
        Only returns edge if resize is ENABLED."""
        if not getattr(self, '_resize_enabled', False):
            return None  # Resize disabled - no edge detection
        
        margin = self._edge_margin
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        
        edges = []
        if x < margin:
            edges.append('left')
        elif x > w - margin:
            edges.append('right')
        if y < margin:
            edges.append('top')
        elif y > h - margin:
            edges.append('bottom')
        
        if edges:
            return '-'.join(edges)
        return None
        
    def _update_scaled_pixmap(self):
        """Scale avatar to fit within _size, then resize window to TIGHTLY wrap the result.
        This ALWAYS wraps tightly, regardless of resize mode."""
        if not self._original_pixmap or self._original_pixmap.isNull():
            # No image - set a default small size
            self.setFixedSize(100, 100)
            return
            
        # Border space around the image (small - just for the border line)
        border_margin = 8
        
        # Scale image to fit within _size (this is the max dimension)
        max_dim = max(50, self._size - border_margin)
        scaled = self._original_pixmap.scaled(
            max_dim, max_dim,
            Qt_KeepAspectRatio, Qt_SmoothTransformation
        )
        
        # Apply rotation if any
        rotation = getattr(self, '_rotation_angle', 0.0)
        if rotation != 0:
            from PyQt5.QtGui import QTransform
            transform = QTransform().rotate(rotation)
            self.pixmap = scaled.transformed(transform, Qt_SmoothTransformation)
        else:
            self.pixmap = scaled
        
        if not self.pixmap or self.pixmap.isNull():
            self.setFixedSize(100, 100)
            return
            
        # Calculate exact window size to wrap TIGHTLY around the image
        img_w = self.pixmap.width()
        img_h = self.pixmap.height()
        win_w = img_w + border_margin
        win_h = img_h + border_margin
        
        # CRITICAL: Force the window to this EXACT size
        # Clear all size constraints first
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        # Set geometry and fixed size
        self.resize(win_w, win_h)
        self.setFixedSize(win_w, win_h)
        
        # Force a repaint
        self.update()
        self.repaint()
        
    def paintEvent(self, a0):
        """Draw avatar with border that always wraps tightly around the image."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        if self.pixmap:
            # Center pixmap in window (should be nearly edge-to-edge with tight wrap)
            x = (self.width() - self.pixmap.width()) // 2
            y = (self.height() - self.pixmap.height()) // 2
            
            # Apply eye tracking offset (subtle shift in gaze direction)
            eye_shift_x = 0
            eye_shift_y = 0
            if getattr(self, '_eye_tracking_enabled', False):
                # Max shift is 10 pixels in each direction for subtle effect
                eye_shift_x = int(getattr(self, '_eye_offset_x', 0) * 10)
                eye_shift_y = int(getattr(self, '_eye_offset_y', 0) * 5)  # Less vertical
            
            # Draw a subtle circular background/glow FIRST (behind everything)
            painter.setPen(Qt_NoPen)
            painter.setBrush(QColor(30, 30, 46, 80))  # Semi-transparent dark
            painter.drawEllipse(x - 3, y - 3, self.pixmap.width() + 6, self.pixmap.height() + 6)
            
            # Handle crossfade during hot-swap
            if getattr(self, '_crossfade_active', False) and self._crossfade_old_pixmap:
                progress = getattr(self, '_crossfade_progress', 1.0)
                
                # Draw old pixmap with fading opacity
                old_opacity = 1.0 - progress
                if old_opacity > 0:
                    painter.setOpacity(old_opacity)
                    old_x = (self.width() - self._crossfade_old_pixmap.width()) // 2
                    old_y = (self.height() - self._crossfade_old_pixmap.height()) // 2
                    painter.drawPixmap(old_x + eye_shift_x, old_y + eye_shift_y, self._crossfade_old_pixmap)
                
                # Draw new pixmap with increasing opacity
                painter.setOpacity(progress)
                painter.drawPixmap(x + eye_shift_x, y + eye_shift_y, self.pixmap)
                painter.setOpacity(1.0)  # Reset
            else:
                # Normal draw (no crossfade)
                painter.drawPixmap(x + eye_shift_x, y + eye_shift_y, self.pixmap)
            
            # Draw border ALWAYS tight around avatar - color indicates resize mode
            pen = painter.pen()
            if getattr(self, '_resize_enabled', False):
                # Blue border when resize is enabled (you can resize)
                pen.setColor(QColor("#3498db"))  # Blue
                pen.setWidth(3)
            else:
                # Green border when resize is disabled (fixed size)
                pen.setColor(QColor("#2ecc71"))  # Green
                pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt_NoBrush)
            # Draw border tightly around the avatar image
            painter.drawRoundedRect(x - 2, y - 2, self.pixmap.width() + 4, self.pixmap.height() + 4, 6, 6)
        else:
            # Draw placeholder circle
            painter.setPen(QColor("#6c7086"))
            painter.setBrush(QColor(30, 30, 46, 150))
            size = min(self.width(), self.height()) - 20
            painter.drawEllipse(10, 10, size, size)
            painter.drawText(self.rect(), Qt_AlignCenter, "?")
        
    def mouseMoveEvent(self, a0):  # type: ignore
        """Handle drag to move, resize, or rotate."""
        try:
            global_pos = a0.globalPosition().toPoint()
            local_pos = a0.position().toPoint()
        except AttributeError:
            global_pos = a0.globalPos()
            local_pos = a0.pos()
        
        # Check if mouse moved significantly (for tap detection)
        if self._touch_press_pos:
            delta = global_pos - self._touch_press_pos
            if abs(delta.x()) > 5 or abs(delta.y()) > 5:
                self._touch_moved = True
                self._hold_timer.stop()  # Cancel hold detection if dragging
        
        # If rotating (Shift+drag)
        if self._rotate_start_x is not None and a0.buttons() == Qt_LeftButton:
            delta_x = global_pos.x() - self._rotate_start_x
            self._rotation_angle = (self._rotation_angle + delta_x * 0.5) % 360
            self._rotate_start_x = global_pos.x()
            self._update_scaled_pixmap()  # Re-render with rotation
            self.update()
            a0.accept()
            return
        
        # If resizing
        if self._resize_edge and a0.buttons() == Qt_LeftButton:
            delta = global_pos - self._resize_start_pos
            delta = global_pos - self._resize_start_pos
            new_size = self._resize_start_size
            new_geo = self._resize_start_geo
            
            if 'right' in self._resize_edge or 'bottom' in self._resize_edge:
                # Resize from right/bottom - increase size (no max limit)
                change = max(delta.x(), delta.y())
                new_size = max(50, self._resize_start_size + change)
            elif 'left' in self._resize_edge or 'top' in self._resize_edge:
                # Resize from left/top - decrease size and move (no max limit)
                change = max(-delta.x(), -delta.y())
                new_size = max(50, self._resize_start_size + change)
                # Adjust position to keep bottom-right corner fixed
                size_diff = new_size - self._resize_start_size
                self.move(new_geo.x() - size_diff, new_geo.y() - size_diff)
            
            self._size = new_size
            # DON'T call setFixedSize here - let _update_scaled_pixmap handle window sizing
            self._update_scaled_pixmap()
            a0.accept()
            return
        
        # If dragging (only if reposition enabled)
        if self._drag_pos is not None and a0.buttons() == Qt_LeftButton and getattr(self, '_reposition_enabled', True):
            new_pos = global_pos - self._drag_pos
            
            # Keep avatar on virtual desktop (all monitors combined) - can drag across monitors
            # but can't go completely off-screen
            try:
                from PyQt5.QtWidgets import QDesktopWidget
                desktop = QDesktopWidget()
                # Get combined geometry of all screens
                virtual_geo = desktop.geometry()  # Virtual desktop = all monitors
                min_visible = 50
                new_x = max(min_visible - self._size, min(virtual_geo.right() - min_visible, new_pos.x()))
                new_y = max(min_visible - self._size, min(virtual_geo.bottom() - min_visible, new_pos.y()))
                new_pos.setX(new_x)
                new_pos.setY(new_y)
            except Exception:
                pass  # If virtual desktop fails, allow free movement
            
            self.move(new_pos)
            a0.accept()
            return
        
        # Update cursor based on position (only when resize enabled)
        edge = self._get_edge_at_pos(local_pos)
        if edge:
            # Set resize cursor
            if 'left' in edge or 'right' in edge:
                self.setCursor(QCursor(Qt.SizeHorCursor if hasattr(Qt, 'SizeHorCursor') else 0))
            if 'top' in edge or 'bottom' in edge:
                self.setCursor(QCursor(Qt.SizeVerCursor if hasattr(Qt, 'SizeVerCursor') else 0))
            if ('top' in edge and 'left' in edge) or ('bottom' in edge and 'right' in edge):
                self.setCursor(QCursor(Qt.SizeFDiagCursor if hasattr(Qt, 'SizeFDiagCursor') else 0))
            if ('top' in edge and 'right' in edge) or ('bottom' in edge and 'left' in edge):
                self.setCursor(QCursor(Qt.SizeBDiagCursor if hasattr(Qt, 'SizeBDiagCursor') else 0))
        else:
            self.setCursor(QCursor(Qt_ArrowCursor))
        a0.accept()
            
    def mouseReleaseEvent(self, a0):  # type: ignore
        """End drag or resize, and detect touch type."""
        # Stop hold timer
        self._hold_timer.stop()
        
        # Check for tap (didn't move much, wasn't a long hold)
        if not self._touch_moved and self._touch_press_pos:
            import time as time_module
            try:
                global_pos = a0.globalPosition().toPoint()
            except AttributeError:
                global_pos = a0.globalPos()
            
            press_duration = time_module.time() - self._touch_press_time
            
            # If it wasn't a hold (timer would have fired), it's a tap
            if press_duration < (self.HOLD_THRESHOLD_MS / 1000.0):
                self._check_tap_type(global_pos)
        
        # Reset touch state
        self._touch_press_pos = None
        self._touch_moved = False
        
        # Save position if we were dragging (per-avatar)
        if self._drag_pos is not None:
            try:
                from ....avatar.persistence import (
                    get_persistence,
                    write_avatar_state_for_ai,
                )
                pos = self.pos()
                persistence = get_persistence()
                settings = persistence.load()
                if self._avatar_path:
                    settings.set_position_for_avatar(self._avatar_path, pos.x(), pos.y())
                settings.screen_position = (pos.x(), pos.y())  # Also save as default
                persistence.save(settings)
                write_avatar_state_for_ai()  # Update AI awareness
            except Exception:
                pass
        # Save size if we were resizing (per-avatar)
        if self._resize_edge is not None:
            try:
                from ....avatar.persistence import (
                    get_persistence,
                    write_avatar_state_for_ai,
                )
                persistence = get_persistence()
                settings = persistence.load()
                if self._avatar_path:
                    settings.set_size_for_avatar(self._avatar_path, self._size)
                settings.overlay_size = self._size  # Also save as default
                persistence.save(settings)
                write_avatar_state_for_ai()
            except Exception:
                pass
        # Save rotation if we were rotating
        if self._rotate_start_x is not None:
            try:
                from ....avatar.persistence import save_avatar_settings
                save_avatar_settings(overlay_rotation=self._rotation_angle)
            except Exception:
                pass
        self._drag_pos = None
        self._resize_edge = None
        self._rotate_start_x = None
        self._is_dragging = False
        self.setCursor(QCursor(Qt_ArrowCursor))
        a0.accept()
    
    def showEvent(self, a0):
        """Restore saved position when shown."""
        super().showEvent(a0)
        # Position is loaded per-avatar in set_avatar_path, so only use default if no avatar path
        if not self._avatar_path:
            try:
                from ....avatar.persistence import load_position
                x, y = load_position()
                if x is not None and y is not None:
                    self.move(x, y)
            except Exception:
                pass
        
        # Register with fullscreen controller for visibility management
        try:
            from ....core.fullscreen_mode import get_fullscreen_controller
            controller = get_fullscreen_controller()
            controller.register_element('avatar_overlay', self, category='avatar')
        except Exception:
            pass
    
    def _auto_save_size(self):
        """Auto-save size if it changed (called by timer)."""
        if not hasattr(self, '_last_saved_size'):
            self._last_saved_size = self._size
            return
        
        if self._size != self._last_saved_size:
            try:
                from ....avatar.persistence import (
                    get_persistence,
                    write_avatar_state_for_ai,
                )
                persistence = get_persistence()
                settings = persistence.load()
                if self._avatar_path:
                    settings.set_size_for_avatar(self._avatar_path, self._size)
                settings.overlay_size = self._size
                persistence.save(settings)
                self._last_saved_size = self._size
                write_avatar_state_for_ai()
            except Exception:
                pass
    
    def hideEvent(self, a0):
        """Save position and size when hidden."""
        super().hideEvent(a0)
        try:
            from ....avatar.persistence import (
                save_avatar_settings,
                save_position,
                write_avatar_state_for_ai,
            )
            pos = self.pos()
            save_position(pos.x(), pos.y())
            save_avatar_settings(overlay_size=self._size)
            self._last_saved_size = self._size  # Update last saved
            write_avatar_state_for_ai()
        except Exception:
            pass
        
    def keyPressEvent(self, a0):  # type: ignore
        """ESC to close."""
        if a0.key() == Qt.Key_Escape if hasattr(Qt, 'Key_Escape') else 0x01000000:
            self.hide()
            self.closed.emit()
        
    def wheelEvent(self, a0):  # type: ignore
        """Scroll wheel to resize when resize is enabled."""
        if getattr(self, '_resize_enabled', False):
            # Get scroll delta
            try:
                delta = a0.angleDelta().y()
            except AttributeError:
                delta = a0.delta()
            
            # Resize based on scroll direction (no limits)
            if delta > 0:
                new_size = self._size + 20  # Scroll up = bigger
            else:
                new_size = max(50, self._size - 20)   # Scroll down = smaller (min 50)
            
            self._size = new_size
            # DON'T call setFixedSize - let _update_scaled_pixmap handle window sizing
            self._update_scaled_pixmap()
            
            # Save size
            try:
                from ....avatar.persistence import (
                    save_avatar_settings,
                    write_avatar_state_for_ai,
                )
                save_avatar_settings(overlay_size=self._size)
                write_avatar_state_for_ai()
            except Exception:
                pass
            
            a0.accept()
        else:
            a0.ignore()
        
    def contextMenuEvent(self, a0):  # type: ignore
        """Right-click to show options menu."""
        from PyQt5.QtWidgets import QMenu
        
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
        
        # Atlas-style gestures (Portal 2) - sends to AI
        gestures_menu = menu.addMenu("Gestures")
        gesture_actions = [
            ("Wave", "wave"),
            ("High Five", "highfive"),
            ("Hug", "hug"),
            ("Dance", "dance"),
            ("Laugh", "laugh"),
            ("Tease", "tease"),
            ("Rock Paper Scissors", "rps"),
        ]
        for label, gesture in gesture_actions:
            action = gestures_menu.addAction(label)
            action.triggered.connect(lambda checked, g=gesture: self._request_gesture(g))
        
        menu.addSeparator()
        
        # Resize & Rotate toggle - default is OFF
        resize_text = "Disable Resize/Rotate" if getattr(self, '_resize_enabled', False) else "Enable Resize/Rotate"
        resize_action = menu.addAction(resize_text)
        resize_action.setToolTip("Enable edge-drag resize and Shift+drag rotate")
        resize_action.triggered.connect(self._toggle_resize)
        
        # Size input action
        size_action = menu.addAction(f"Set Size... ({self._size}px)")
        size_action.triggered.connect(self._show_size_dialog)
        
        menu.addSeparator()
        
        # Center on screen (moved from double-click)
        center_action = menu.addAction("Center on Screen")
        center_action.triggered.connect(self._center_on_screen)
        
        menu.addSeparator()
        
        # Close
        close_action = menu.addAction("Hide Avatar")
        close_action.triggered.connect(self._close_avatar)
        
        menu.exec_(a0.globalPos())
    
    def _request_gesture(self, gesture: str):
        """Request a gesture from the AI - AI decides how to react."""
        # Send gesture request to AI via conversation system
        try:
            from ....memory import ConversationManager
            conv = ConversationManager()
            # Add as a system message so AI knows user requested this
            conv.add_message("system", f"[User requested gesture: {gesture}]")
        except Exception:
            pass
    
    def _toggle_resize(self):
        """Toggle resize/rotate mode and update border visibility."""
        self._resize_enabled = not getattr(self, '_resize_enabled', False)
        self.update()  # Repaint to show/hide border
        # Save setting
        try:
            from ....avatar.persistence import (
                save_avatar_settings,
                write_avatar_state_for_ai,
            )
            save_avatar_settings(resize_enabled=self._resize_enabled)
            write_avatar_state_for_ai()
        except Exception:
            pass
    
    def _center_on_screen(self):
        """Center the avatar on the current screen."""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geo = screen.availableGeometry()
            self.move(
                screen_geo.center().x() - self._size // 2,
                screen_geo.center().y() - self._size // 2
            )
    
    def _close_avatar(self):
        """Close the avatar."""
        self.hide()
        self.closed.emit()
    
    def _show_size_dialog(self):
        """Show dialog to set avatar size."""
        from PyQt5.QtWidgets import QInputDialog
        size, ok = QInputDialog.getInt(
            self, "Set Avatar Size", "Size (pixels):",
            self._size, 50, 2000, 10
        )
        if ok:
            self._size = size
            # DON'T call setFixedSize - let _update_scaled_pixmap handle window sizing
            self._update_scaled_pixmap()
            # Save size (per-avatar)
            try:
                from ....avatar.persistence import (
                    get_persistence,
                    write_avatar_state_for_ai,
                )
                persistence = get_persistence()
                settings = persistence.load()
                if self._avatar_path:
                    settings.set_size_for_avatar(self._avatar_path, self._size)
                settings.overlay_size = self._size  # Also save as default
                persistence.save(settings)
                write_avatar_state_for_ai()
            except Exception:
                pass
    
    def mouseDoubleClickEvent(self, a0):  # type: ignore
        """Double-click does nothing - use right-click menu for Center on Screen."""
        a0.accept()  # Consume the event
    
    def nativeEvent(self, eventType, message):
        """Handle Windows native events for per-pixel hit testing.
        
        WM_NCHITTEST is called by Windows to determine what part of the window
        the mouse is over. By returning HTTRANSPARENT for transparent pixels,
        clicks pass through to windows behind.
        
        This only runs when Windows needs to know - zero overhead otherwise.
        """
        import sys
        if sys.platform != 'win32':
            return super().nativeEvent(eventType, message)
        
        try:
            import ctypes
            from ctypes import wintypes

            # Check if this is WM_NCHITTEST (0x0084)
            WM_NCHITTEST = 0x0084
            HTTRANSPARENT = -1
            HTCLIENT = 1
            
            # Get message ID - handle different PyQt versions
            msg = ctypes.cast(int(message), ctypes.POINTER(wintypes.MSG)).contents
            
            if msg.message == WM_NCHITTEST:
                # Don't do hit testing while dragging
                if getattr(self, '_is_dragging', False) or self._drag_pos is not None:
                    return super().nativeEvent(eventType, message)
                
                # Get mouse position from lParam
                x = msg.lParam & 0xFFFF
                y = (msg.lParam >> 16) & 0xFFFF
                
                # Handle signed coordinates (can be negative on multi-monitor)
                if x > 32767:
                    x -= 65536
                if y > 32767:
                    y -= 65536
                
                # Convert screen coords to widget coords
                local_pos = self.mapFromGlobal(QPoint(x, y))
                
                # Check if pixel is opaque (part of the avatar)
                if not self._is_pixel_opaque(local_pos.x(), local_pos.y()):
                    # Transparent pixel - let click pass through
                    return True, HTTRANSPARENT
                
                # Opaque pixel - handle normally (HTCLIENT = inside window)
                return True, HTCLIENT
                
        except Exception:
            pass
        
        return super().nativeEvent(eventType, message)
    
    def _is_pixel_opaque(self, x: int, y: int, threshold: int = 10) -> bool:
        """Check if the pixel at (x, y) is opaque (part of the avatar).
        
        Args:
            x, y: Position relative to widget
            threshold: Alpha value below which pixel is considered transparent
            
        Returns:
            True if pixel is opaque (should handle click), False if transparent
        """
        if not self.pixmap or self.pixmap.isNull():
            return False  # No image = transparent
        
        # Get the offset where pixmap is drawn (centered in window)
        pixmap_x = (self.width() - self.pixmap.width()) // 2
        pixmap_y = (self.height() - self.pixmap.height()) // 2
        
        # Convert to pixmap coordinates
        px = x - pixmap_x
        py = y - pixmap_y
        
        # Bounds check - outside pixmap is transparent
        if px < 0 or px >= self.pixmap.width() or py < 0 or py >= self.pixmap.height():
            return False
        
        # Get pixel color from pixmap - check alpha
        img = self.pixmap.toImage()
        if img.isNull():
            return False
        
        pixel = img.pixelColor(px, py)
        return pixel.alpha() > threshold
    
    def mousePressEvent(self, a0):  # type: ignore
        """Start drag to move, resize, rotate (Shift+drag), or detect touch."""
        import time as time_module
        if a0.button() == Qt_LeftButton:
            try:
                global_pos = a0.globalPosition().toPoint()
                local_pos = a0.position().toPoint()
            except AttributeError:
                global_pos = a0.globalPos()
                local_pos = a0.pos()
            
            # Start touch tracking for tap/hold detection
            self._touch_press_time = time_module.time()
            self._touch_press_pos = global_pos
            self._touch_moved = False
            
            # Mark as dragging for nativeEvent
            self._is_dragging = True
            
            # Check for Shift+drag (rotation mode) - only when resize/rotate is enabled
            modifiers = a0.modifiers()
            shift_held = bool(modifiers & (Qt.ShiftModifier if hasattr(Qt, 'ShiftModifier') else 0x02000000))
            
            if shift_held and getattr(self, '_resize_enabled', False):
                # Start rotation drag (only when enabled)
                self._rotate_start_x = global_pos.x()
                self.setCursor(QCursor(Qt.SizeHorCursor if hasattr(Qt, 'SizeHorCursor') else 0))
            else:
                # Check if we're on an edge (for resize) - _get_edge_at_pos already checks _resize_enabled
                edge = self._get_edge_at_pos(local_pos)
                if edge:
                    self._resize_edge = edge
                    self._resize_start_pos = global_pos
                    self._resize_start_size = self._size
                    self._resize_start_geo = self.geometry()
                elif getattr(self, '_reposition_enabled', True):
                    # Normal drag (only if reposition is enabled) - no hand cursor
                    self._drag_pos = global_pos - self.pos()
                    # Keep arrow cursor, no grabbing hand
                    
                    # Start hold timer for touch detection
                    self._hold_timer.start(self.HOLD_THRESHOLD_MS)
        a0.accept()


class DragBarWidget(QWidget):
    """A draggable bar widget that sits on top of the avatar for moving/resizing.
    
    Has two modes:
    - Visible: Shows a solid bar with grip lines
    - Ghost: Nearly invisible (just a subtle outline) but still functional
    """
    
    drag_started = pyqtSignal(object)  # Emits global position
    drag_moved = pyqtSignal(object)   # Emits global position
    drag_ended = pyqtSignal()
    position_changed = pyqtSignal(int, int)  # Emits x, y when bar is repositioned within parent
    context_menu_requested = pyqtSignal(object)  # Emits global position for context menu
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_pos = None
        self._resize_edge = None
        self._resize_start_pos = None
        self._resize_start_size = None
        self._bar_drag_pos = None  # For dragging the bar itself within the window
        self._ghost_mode = False  # When True, bar is nearly invisible
        self._reposition_mode = False  # When True, dragging moves bar not avatar
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt_ArrowCursor))
        self.setMinimumSize(20, 10)  # Allow smaller sizes
        self.setContextMenuPolicy(Qt.CustomContextMenu if hasattr(Qt, 'CustomContextMenu') else 3)
        
    def set_ghost_mode(self, ghost: bool):
        """Set ghost mode - when True, bar is nearly invisible but still functional."""
        self._ghost_mode = ghost
        self.update()
        
    def is_ghost_mode(self) -> bool:
        return self._ghost_mode
    
    def set_reposition_mode(self, reposition: bool):
        """Set reposition mode - when True, dragging moves bar instead of avatar."""
        self._reposition_mode = reposition
        self.update()
    
    def is_reposition_mode(self) -> bool:
        return self._reposition_mode
    
    def contextMenuEvent(self, event):
        """Forward right-click to parent for context menu."""
        self.context_menu_requested.emit(event.globalPos())
        event.accept()
        
    def paintEvent(self, event):
        """Draw the drag bar."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        w, h = self.width(), self.height()
        
        if self._ghost_mode:
            # Ghost mode: very subtle, just a faint outline
            painter.setBrush(QColor(255, 255, 255, 15))  # Nearly invisible
            painter.setPen(QColor(255, 255, 255, 40))
            painter.drawRoundedRect(1, 1, w-2, h-2, 5, 5)
            # Tiny grip hint in center
            painter.setPen(QColor(255, 255, 255, 60))
            center_x, center_y = w // 2, h // 2
            painter.drawLine(center_x - 8, center_y, center_x + 8, center_y)
        else:
            # Visible mode: solid bar
            painter.setBrush(QColor(30, 30, 46, 200))
            painter.setPen(QColor(100, 100, 120, 180))
            painter.drawRoundedRect(0, 0, w, h, 5, 5)
            
            # Resize handles on edges
            handle_size = 6
            painter.setBrush(QColor(137, 180, 250, 180))
            painter.setPen(QColor(200, 200, 220, 200))
            # Right edge handle
            painter.drawRect(w - handle_size, h//2 - handle_size//2, handle_size, handle_size)
            # Bottom edge handle
            painter.drawRect(w//2 - handle_size//2, h - handle_size, handle_size, handle_size)
            
            # Grip lines in center
            painter.setPen(QColor(200, 200, 220, 150))
            center_x = w // 2
            center_y = h // 2
            for i in range(3):
                y = center_y - 4 + i * 4
                painter.drawLine(center_x - 15, y, center_x + 15, y)
    
    def _get_edge(self, pos):
        """Check if mouse is on an edge for resizing."""
        if self._ghost_mode:
            return None  # No resize in ghost mode
        margin = 8
        w, h = self.width(), self.height()
        if pos.x() >= w - margin:
            return 'right'
        elif pos.y() >= h - margin:
            return 'bottom'
        return None
    
    def mousePressEvent(self, event):
        if event.button() == Qt_LeftButton:
            edge = self._get_edge(event.pos())
            if edge:
                self._resize_edge = edge
                self._resize_start_pos = event.pos()
                self._resize_start_size = self.size()
            elif event.modifiers() & Qt_ShiftModifier:
                # Shift+drag repositions the bar within the window
                self._bar_drag_pos = event.pos()
            elif self._reposition_mode:
                # Reposition mode: drag moves bar, not avatar
                self._bar_drag_pos = event.pos()
            else:
                # Regular drag moves the whole window
                self._drag_pos = event.globalPos() - self.parent().pos()
                self.drag_started.emit(event.globalPos())
            event.accept()
    
    def mouseMoveEvent(self, event):
        if self._resize_edge:
            delta = event.pos() - self._resize_start_pos
            if self._resize_edge == 'right':
                new_w = max(20, min(300, self._resize_start_size.width() + delta.x()))
                self.setFixedWidth(new_w)
            elif self._resize_edge == 'bottom':
                new_h = max(10, min(100, self._resize_start_size.height() + delta.y()))
                self.setFixedHeight(new_h)
            event.accept()
        elif self._bar_drag_pos is not None:
            # Reposition bar within parent
            delta = event.pos() - self._bar_drag_pos
            new_x = self.x() + delta.x()
            new_y = self.y() + delta.y()
            # Clamp to parent bounds
            parent = self.parent()
            if parent:
                new_x = max(0, min(parent.width() - self.width(), new_x))
                new_y = max(0, min(parent.height() - self.height(), new_y))
            self.move(new_x, new_y)
            self.position_changed.emit(new_x, new_y)
            event.accept()
        elif self._drag_pos is not None:
            self.drag_moved.emit(event.globalPos())
            event.accept()
        else:
            # Update cursor based on position
            edge = self._get_edge(event.pos())
            if edge == 'right':
                self.setCursor(QCursor(Qt_SizeHorCursor))
            elif edge == 'bottom':
                self.setCursor(QCursor(Qt_SizeVerCursor))
            else:
                self.setCursor(QCursor(Qt_ArrowCursor))
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt_LeftButton:
            if self._drag_pos is not None:
                self.drag_ended.emit()
            self._drag_pos = None
            self._bar_drag_pos = None
            self._resize_edge = None
            self._resize_start_pos = None
            self._resize_start_size = None
            self.setCursor(QCursor(Qt_ArrowCursor))
            event.accept()


class FloatingDragBar(QWidget):
    """A separate top-level window that floats above the avatar for interaction.
    
    Behavior modes:
    - VISIBLE mode: Shows bar, dragging moves the bar itself (repositions on avatar)
    - GHOST mode: Nearly invisible bar, dragging moves the avatar window
    - ANCHORED mode: Fully invisible, auto-centers on avatar, dragging moves avatar
    
    Right-click always shows context menu.
    Edges can resize the bar (visible mode only).
    """
    
    drag_started = pyqtSignal(object)
    drag_moved = pyqtSignal(object)
    drag_ended = pyqtSignal()
    position_changed = pyqtSignal(int, int)
    context_menu_requested = pyqtSignal(object)
    
    def __init__(self, avatar_window):
        # Separate top-level window, not a child
        super().__init__(None)
        self._avatar = avatar_window
        
        # Frameless, always on top, tool window (no taskbar)
        self.setWindowFlags(
            Qt_FramelessWindowHint |
            Qt_WindowStaysOnTopHint |
            Qt_Tool
        )
        self.setAttribute(Qt_WA_TranslucentBackground, True)
        
        self._width = 100
        self._height = 25
        self.setFixedSize(self._width, self._height)
        
        # Relative position within avatar bounds
        self._rel_x = 5
        self._rel_y = 5
        
        self._drag_pos = None
        self._ghost_mode = False
        self._anchored_mode = False  # New: fully invisible, auto-centered
        self._anchor_offset_x = 0.5  # 0.0=left, 0.5=center, 1.0=right
        self._anchor_offset_y = 0.5  # 0.0=top, 0.5=center, 1.0=bottom
        self._bar_drag_pos = None
        self._resize_edge = None
        self._resize_start_pos = None
        self._resize_start_size = None
        
        self.setMouseTracking(True)
        # Use arrow cursor, not hand - subtle change
        self.setCursor(QCursor(Qt_ArrowCursor))
    
    def set_relative_position(self, x: int, y: int):
        """Set position relative to avatar window."""
        self._rel_x = x
        self._rel_y = y
        if not self._anchored_mode:
            self._update_position()
    
    def _update_position(self):
        """Update screen position based on avatar window position."""
        if self._avatar:
            avatar_pos = self._avatar.pos()
            avatar_w = self._avatar.width()
            avatar_h = self._avatar.height()
            
            if self._anchored_mode:
                # Anchored mode: scale drag area with avatar size (60% of avatar)
                self._width = max(100, int(avatar_w * 0.6))
                self._height = max(100, int(avatar_h * 0.6))
                self.setFixedSize(self._width, self._height)
                
                # Calculate position based on anchor offset (0.5 = center)
                anchor_x = int(avatar_w * self._anchor_offset_x - self._width / 2)
                anchor_y = int(avatar_h * self._anchor_offset_y - self._height / 2)
                # Clamp to avatar bounds
                anchor_x = max(0, min(avatar_w - self._width, anchor_x))
                anchor_y = max(0, min(avatar_h - self._height, anchor_y))
                self.move(avatar_pos.x() + anchor_x, avatar_pos.y() + anchor_y)
            else:
                self.move(avatar_pos.x() + self._rel_x, avatar_pos.y() + self._rel_y)
    
    def set_ghost_mode(self, ghost: bool):
        self._ghost_mode = ghost
        if ghost:
            self._anchored_mode = False
        self.update()
    
    def set_anchored_mode(self, anchored: bool, anchor_x: float = 0.5, anchor_y: float = 0.5):
        """Enable anchored mode - fully invisible, auto-positioned drag area.
        
        Args:
            anchored: Enable/disable anchored mode
            anchor_x: Horizontal anchor (0.0=left, 0.5=center, 1.0=right)
            anchor_y: Vertical anchor (0.0=top, 0.5=center, 1.0=bottom)
        """
        self._anchored_mode = anchored
        self._anchor_offset_x = max(0.0, min(1.0, anchor_x))
        self._anchor_offset_y = max(0.0, min(1.0, anchor_y))
        if anchored:
            self._ghost_mode = False
        self._update_position()  # This will now set the size based on avatar
        self.update()
    
    def is_anchored_mode(self) -> bool:
        return self._anchored_mode
    
    def set_anchor_point(self, x: float, y: float):
        """Set the anchor point (0.0-1.0 range for both axes)."""
        self._anchor_offset_x = max(0.0, min(1.0, x))
        self._anchor_offset_y = max(0.0, min(1.0, y))
        self._update_position()
    
    def is_ghost_mode(self) -> bool:
        return self._ghost_mode
    
    def isVisible(self):
        return super().isVisible()
    
    def setVisible(self, visible: bool):
        super().setVisible(visible)
    
    def geometry(self):
        """Return geometry in avatar-relative coordinates for hit testing."""
        from PyQt5.QtCore import QRect
        return QRect(self._rel_x, self._rel_y, self._width, self._height)
    
    def contextMenuEvent(self, event):
        self.context_menu_requested.emit(event.globalPos())
        event.accept()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        w, h = self.width(), self.height()
        
        if self._anchored_mode:
            # ANCHORED MODE: Draw a nearly invisible but solid hit area
            # Alpha=1 is enough to capture mouse events but appears invisible
            painter.setBrush(QColor(128, 128, 128, 1))  # Almost invisible
            painter.setPen(Qt_NoPen)
            painter.drawRect(0, 0, w, h)
        elif self._ghost_mode:
            painter.setBrush(QColor(255, 255, 255, 15))
            painter.setPen(QColor(255, 255, 255, 40))
            painter.drawRoundedRect(1, 1, w-2, h-2, 5, 5)
            painter.setPen(QColor(255, 255, 255, 60))
            cx, cy = w // 2, h // 2
            painter.drawLine(cx - 8, cy, cx + 8, cy)
        else:
            painter.setBrush(QColor(30, 30, 46, 220))
            painter.setPen(QColor(100, 100, 120, 200))
            painter.drawRoundedRect(0, 0, w, h, 5, 5)
            
            # Resize handles
            hs = 6
            painter.setBrush(QColor(137, 180, 250, 180))
            painter.setPen(QColor(200, 200, 220, 200))
            painter.drawRect(w - hs, h//2 - hs//2, hs, hs)
            painter.drawRect(w//2 - hs//2, h - hs, hs, hs)
            
            # Grip lines
            painter.setPen(QColor(200, 200, 220, 150))
            cx, cy = w // 2, h // 2
            for i in range(3):
                y = cy - 4 + i * 4
                painter.drawLine(cx - 15, y, cx + 15, y)
    
    def _get_edge(self, pos):
        if self._ghost_mode or self._anchored_mode:
            return None  # No resize edges in ghost or anchored mode
        margin = 8
        w, h = self.width(), self.height()
        if pos.x() >= w - margin:
            return 'right'
        elif pos.y() >= h - margin:
            return 'bottom'
        return None
    
    def mousePressEvent(self, event):
        if event.button() == Qt_LeftButton:
            edge = self._get_edge(event.pos())
            if edge and not self._anchored_mode:
                # Resize from edges (not in anchored mode)
                self._resize_edge = edge
                self._resize_start_pos = event.pos()
                self._resize_start_size = self.size()
            elif self._ghost_mode or self._anchored_mode:
                # GHOST/ANCHORED MODE: drag moves the avatar
                self._drag_pos = event.globalPos() - self._avatar.pos()
                self.drag_started.emit(event.globalPos())
            else:
                # VISIBLE MODE: drag moves the bar itself
                self._bar_drag_pos = event.pos()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if self._resize_edge and self._resize_start_pos:
            delta = event.pos() - self._resize_start_pos
            new_w = self._resize_start_size.width()
            new_h = self._resize_start_size.height()
            if self._resize_edge == 'right':
                new_w = max(50, self._resize_start_size.width() + delta.x())
            elif self._resize_edge == 'bottom':
                new_h = max(15, self._resize_start_size.height() + delta.y())
            self._width = new_w
            self._height = new_h
            self.setFixedSize(new_w, new_h)
        elif self._bar_drag_pos is not None:
            # Moving bar within avatar bounds
            delta = event.pos() - self._bar_drag_pos
            new_x = max(0, min(self._avatar.width() - self.width(), self._rel_x + delta.x()))
            new_y = max(0, min(self._avatar.height() - self.height(), self._rel_y + delta.y()))
            self._rel_x = new_x
            self._rel_y = new_y
            self._update_position()
            self.position_changed.emit(new_x, new_y)
        elif self._drag_pos is not None:
            # Moving avatar window (ghost mode)
            self.drag_moved.emit(event.globalPos())
        else:
            # Update cursor based on position
            edge = self._get_edge(event.pos())
            if edge == 'right':
                self.setCursor(QCursor(Qt_SizeHorCursor))
            elif edge == 'bottom':
                self.setCursor(QCursor(Qt_SizeVerCursor))
            else:
                self.setCursor(QCursor(Qt_ArrowCursor))
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt_LeftButton:
            if self._drag_pos is not None:
                self.drag_ended.emit()
            self._drag_pos = None
            self._bar_drag_pos = None
            self._resize_edge = None
            self._resize_start_pos = None
            self._resize_start_size = None
            self.setCursor(QCursor(Qt_ArrowCursor))
            event.accept()
    
    def wheelEvent(self, event):
        """Forward scroll wheel to avatar for resizing."""
        if self._avatar and hasattr(self._avatar, '_set_size'):
            try:
                delta = event.angleDelta().y()
            except AttributeError:
                delta = getattr(event, 'delta', lambda: 0)()
            
            # Resize based on scroll direction
            current_size = getattr(self._avatar, '_size', 300)
            if delta > 0:
                new_size = current_size + 20  # Scroll up = bigger
            else:
                new_size = max(50, current_size - 20)  # Scroll down = smaller
            
            self._avatar._set_size(new_size)
            event.accept()
        else:
            event.ignore()


class AvatarHitLayer(QWidget):
    """Resizable hit area centered on avatar with visible border.
    
    This is a separate top-level window that:
    - Shows a subtle dashed border so you can see it
    - Drag edges to resize the hit area
    - Left-drag center to move the avatar
    - Scroll wheel resizes the avatar
    - Right-click shows context menu
    - Rest of avatar is click-through!
    """
    
    drag_started = pyqtSignal(object)
    drag_moved = pyqtSignal(object)
    drag_ended = pyqtSignal()
    context_menu_requested = pyqtSignal(object)
    
    EDGE_MARGIN = 12  # Pixels from edge to trigger resize
    MIN_RATIO = 0.15  # Minimum hit area ratio
    
    def __init__(self, avatar_window):
        super().__init__(None)  # Top-level window, not a child
        self._avatar = avatar_window
        
        # Frameless, always on top, tool window (no taskbar entry)
        self.setWindowFlags(
            Qt_FramelessWindowHint |
            Qt_WindowStaysOnTopHint |
            Qt_Tool
        )
        self.setAttribute(Qt_WA_TranslucentBackground, True)
        
        self._drag_pos = None
        self._resize_edge = None
        self._resize_start_pos = None
        self._resize_start_ratios = None
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt_ArrowCursor))
        
        # Edge ratios (0.0 = at avatar edge, 1.0 = at center)
        # These are proportional so they scale with avatar size
        self._ratio_left = 0.12    # 12% inward from left edge
        self._ratio_right = 0.12   # 12% inward from right edge
        self._ratio_top = 0.12     # 12% inward from top edge
        self._ratio_bottom = 0.12  # 12% inward from bottom edge
        
        # Visibility and lock options
        self._show_border = True   # Show the dashed border
        self._resize_locked = False  # Lock USER resize (AI can still control avatar)
        
        # Auto-track mode - follows model bounds automatically
        self._auto_track = False  # When True, hit area follows rendered model bounds
        self._auto_track_timer = QTimer()
        self._auto_track_timer.timeout.connect(self._update_auto_track)
        self._auto_track_padding = 0.05  # 5% padding around model bounds
        
        # Load saved settings
        self._load_settings()
        
        # Sync with avatar immediately
        self._sync_with_avatar()
    
    def set_auto_track(self, enabled: bool):
        """Enable or disable auto-tracking of model bounds."""
        self._auto_track = enabled
        if enabled:
            self._auto_track_timer.start(50)  # Update at 20fps
        else:
            self._auto_track_timer.stop()
        self.update()
        try:
            from ....avatar.persistence import save_avatar_settings
            save_avatar_settings(hit_auto_track=enabled)
        except Exception:
            pass
    
    def _update_auto_track(self):
        """Update hit area to match model's rendered bounds."""
        if not self._auto_track or not self._avatar:
            return
        
        # Get model bounds from the GL widget
        gl_widget = getattr(self._avatar, '_gl_widget', None)
        if not gl_widget:
            return
        
        # Try to get projected model bounds
        bounds = self._get_model_screen_bounds(gl_widget)
        if bounds:
            x, y, w, h = bounds
            avatar_pos = self._avatar.pos()
            
            # Add padding
            pad = int(min(w, h) * self._auto_track_padding)
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(self._avatar.width() - x, w + pad * 2)
            h = min(self._avatar.height() - y, h + pad * 2)
            
            # Update hit layer position and size
            self.setFixedSize(max(40, w), max(40, h))
            self.move(avatar_pos.x() + x, avatar_pos.y() + y)
    
    def _get_model_screen_bounds(self, gl_widget):
        """Get the 2D screen bounds of the rendered 3D model."""
        try:
            # Check if model has computed bounds info
            if hasattr(gl_widget, 'model_bounds') and gl_widget.model_bounds:
                return gl_widget.model_bounds
            
            # Try to get from mesh data
            if hasattr(gl_widget, 'mesh') and gl_widget.mesh:
                mesh = gl_widget.mesh
                if hasattr(mesh, 'bounds') and mesh.bounds is not None:
                    # Get mesh bounds and convert to screen space
                    # This is a simplified approach - full projection would be better
                    avatar_size = self._avatar.width()
                    
                    # Use a fixed ratio based on typical model centering
                    # Models are usually centered, so bounds are roughly proportional
                    cx, cy = avatar_size // 2, avatar_size // 2
                    
                    # Estimate visible bounds (models typically fill 60-80% of viewport)
                    scale = 0.7  # Assume model fills ~70% of viewport
                    half_w = int(avatar_size * scale / 2)
                    half_h = int(avatar_size * scale / 2)
                    
                    return (cx - half_w, cy - half_h, half_w * 2, half_h * 2)
            
            return None
        except Exception:
            return None
    
    def set_border_visible(self, visible: bool):
        """Show or hide the border."""
        self._show_border = visible
        self.update()
        try:
            from ....avatar.persistence import save_avatar_settings
            save_avatar_settings(hit_border_visible=visible)
        except Exception:
            pass
    
    def set_resize_locked(self, locked: bool):
        """Lock or unlock resize."""
        self._resize_locked = locked
        self.update()
        try:
            from ....avatar.persistence import save_avatar_settings
            save_avatar_settings(hit_resize_locked=locked)
        except Exception:
            pass
    
    def _load_settings(self):
        """Load saved settings."""
        try:
            from ....avatar.persistence import load_avatar_settings
            settings = load_avatar_settings()
            self._ratio_left = getattr(settings, 'hit_ratio_left', 0.12)
            self._ratio_right = getattr(settings, 'hit_ratio_right', 0.12)
            self._ratio_top = getattr(settings, 'hit_ratio_top', 0.12)
            self._ratio_bottom = getattr(settings, 'hit_ratio_bottom', 0.12)
            self._show_border = getattr(settings, 'hit_border_visible', True)
            self._resize_locked = getattr(settings, 'hit_resize_locked', False)
            # Auto-track mode
            auto_track = getattr(settings, 'hit_auto_track', False)
            if auto_track:
                self.set_auto_track(True)
        except Exception:
            pass
    
    def _save_settings(self):
        """Save settings to persistence."""
        try:
            from ....avatar.persistence import save_avatar_settings
            save_avatar_settings(
                hit_ratio_left=self._ratio_left,
                hit_ratio_right=self._ratio_right,
                hit_ratio_top=self._ratio_top,
                hit_ratio_bottom=self._ratio_bottom
            )
        except Exception:
            pass
    
    def set_hit_ratio(self, ratio: float):
        """Set the hit area ratio (0.0-1.0) - applies uniformly to all sides."""
        # Convert overall ratio to edge ratios
        # ratio=1.0 means full coverage (edge ratios = 0)
        # ratio=0.5 means 50% coverage (edge ratios = 0.25 each side)
        edge_ratio = (1.0 - max(self.MIN_RATIO, min(1.0, ratio))) / 2
        self._ratio_left = edge_ratio
        self._ratio_right = edge_ratio
        self._ratio_top = edge_ratio
        self._ratio_bottom = edge_ratio
        self._sync_with_avatar()
        self._save_settings()
    
    def get_hit_ratio(self) -> float:
        """Get approximate overall hit area ratio."""
        avg_edge = (self._ratio_left + self._ratio_right + self._ratio_top + self._ratio_bottom) / 4
        return max(self.MIN_RATIO, 1.0 - avg_edge * 2)
    
    def _sync_with_avatar(self):
        """Sync size and position with avatar window using proportional ratios."""
        if self._avatar:
            avatar_size = self._avatar.width()
            avatar_pos = self._avatar.pos()
            
            # Calculate hit area bounds using ratios (scales with avatar)
            left_offset = int(avatar_size * self._ratio_left)
            right_offset = int(avatar_size * self._ratio_right)
            top_offset = int(avatar_size * self._ratio_top)
            bottom_offset = int(avatar_size * self._ratio_bottom)
            
            # Calculate position and size
            x = avatar_pos.x() + left_offset
            y = avatar_pos.y() + top_offset
            w = avatar_size - left_offset - right_offset
            h = avatar_size - top_offset - bottom_offset
            
            # Ensure minimum size
            min_size = int(avatar_size * self.MIN_RATIO)
            w = max(min_size, w)
            h = max(min_size, h)
            
            self.setFixedSize(w, h)
            self.move(x, y)
    
    def _get_edge_at_pos(self, pos):
        """Check if mouse is near an edge or midpoint for resizing."""
        margin = self.EDGE_MARGIN
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        mid_x = w // 2
        mid_y = h // 2
        
        # Check corners first (diagonal resize)
        if x < margin and y < margin:
            return 'top-left'
        if x > w - margin and y < margin:
            return 'top-right'
        if x < margin and y > h - margin:
            return 'bottom-left'
        if x > w - margin and y > h - margin:
            return 'bottom-right'
        
        # Check edge midpoints (single edge resize)
        if abs(x - mid_x) < margin * 2 and y < margin:
            return 'top'
        if abs(x - mid_x) < margin * 2 and y > h - margin:
            return 'bottom'
        if x < margin and abs(y - mid_y) < margin * 2:
            return 'left'
        if x > w - margin and abs(y - mid_y) < margin * 2:
            return 'right'
        
        # Check full edges
        if x < margin:
            return 'left'
        if x > w - margin:
            return 'right'
        if y < margin:
            return 'top'
        if y > h - margin:
            return 'bottom'
        return None
    
    def paintEvent(self, event):
        """Draw border and handles only if visible."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        w, h = self.width(), self.height()
        mid_x, mid_y = w // 2, h // 2
        
        # Only draw anything if border is visible
        if not self._show_border:
            # Completely invisible - just a subtle fill for mouse capture
            painter.setBrush(QColor(0, 0, 0, 1))  # Nearly invisible
            painter.setPen(Qt_NoPen)
            painter.drawRect(0, 0, w, h)
            return
        
        # Different fill color for auto-track mode (green tint)
        if self._auto_track:
            painter.setBrush(QColor(100, 200, 150, 15))  # Green tint for auto-track
        else:
            painter.setBrush(QColor(100, 150, 255, 12))  # Blue tint for manual
        painter.setPen(Qt_NoPen)
        painter.drawRoundedRect(2, 2, w-4, h-4, 6, 6)
        
        # Dashed border - green for auto-track, blue for manual
        if self._auto_track:
            pen = QPen(QColor(100, 200, 150, 160))  # Green border
        else:
            pen = QPen(QColor(137, 180, 250, 140))  # Blue border
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine if hasattr(Qt, 'DashLine') else 2)
        painter.setPen(pen)
        painter.setBrush(Qt_NoBrush)
        painter.drawRoundedRect(2, 2, w-4, h-4, 6, 6)
        
        # Draw handles if not locked and not in auto-track mode
        if not self._resize_locked and not self._auto_track:
            handle_size = 8
            half_handle = handle_size // 2
            painter.setBrush(QColor(137, 180, 250, 200))
            painter.setPen(QColor(220, 220, 240, 220))
            
            # Four corners
            painter.drawRect(0, 0, handle_size, handle_size)
            painter.drawRect(w - handle_size, 0, handle_size, handle_size)
            painter.drawRect(0, h - handle_size, handle_size, handle_size)
            painter.drawRect(w - handle_size, h - handle_size, handle_size, handle_size)
            
            # Four edge midpoints
            painter.drawRect(mid_x - half_handle, 0, handle_size, handle_size)  # Top center
            painter.drawRect(mid_x - half_handle, h - handle_size, handle_size, handle_size)  # Bottom center
            painter.drawRect(0, mid_y - half_handle, handle_size, handle_size)  # Left center
            painter.drawRect(w - handle_size, mid_y - half_handle, handle_size, handle_size)  # Right center
    
    def mousePressEvent(self, event):
        if event.button() == Qt_LeftButton:
            edge = self._get_edge_at_pos(event.pos())
            # Only allow resize if not locked
            if edge and not self._resize_locked:
                # Start resize - store current ratios
                self._resize_edge = edge
                self._resize_start_pos = event.globalPos()
                self._resize_start_ratios = {
                    'left': self._ratio_left,
                    'right': self._ratio_right,
                    'top': self._ratio_top,
                    'bottom': self._ratio_bottom
                }
            else:
                # Start drag (move avatar)
                self._drag_pos = event.globalPos() - self._avatar.pos()
                self.drag_started.emit(event.globalPos())
            event.accept()
        elif event.button() == Qt_RightButton:
            self.context_menu_requested.emit(event.globalPos())
            event.accept()
    
    def mouseMoveEvent(self, event):
        if self._resize_edge and self._resize_start_pos and self._resize_start_ratios:
            # Independent edge resize using ratios
            delta = event.globalPos() - self._resize_start_pos
            edge = self._resize_edge
            avatar_size = self._avatar.width()
            
            # Convert pixel delta to ratio delta
            ratio_delta_x = delta.x() / avatar_size if avatar_size > 0 else 0
            ratio_delta_y = delta.y() / avatar_size if avatar_size > 0 else 0
            max_ratio = 0.45  # Maximum 45% inward from each edge
            
            # Update only the specific edge(s) being dragged
            if 'left' in edge:
                # Dragging left edge: positive delta.x = edge moves right = more inward
                new_ratio = self._resize_start_ratios['left'] + ratio_delta_x
                self._ratio_left = max(0, min(max_ratio, new_ratio))
            if 'right' in edge:
                # Dragging right edge: positive delta.x = edge moves right = less inward
                new_ratio = self._resize_start_ratios['right'] - ratio_delta_x
                self._ratio_right = max(0, min(max_ratio, new_ratio))
            if 'top' in edge:
                # Dragging top edge: positive delta.y = edge moves down = more inward
                new_ratio = self._resize_start_ratios['top'] + ratio_delta_y
                self._ratio_top = max(0, min(max_ratio, new_ratio))
            if 'bottom' in edge:
                # Dragging bottom edge: positive delta.y = edge moves down = less inward
                new_ratio = self._resize_start_ratios['bottom'] - ratio_delta_y
                self._ratio_bottom = max(0, min(max_ratio, new_ratio))
            
            self._sync_with_avatar()
            
        elif self._drag_pos is not None:
            self.drag_moved.emit(event.globalPos())
        else:
            # Update cursor based on position (only show resize cursors if not locked)
            edge = self._get_edge_at_pos(event.pos())
            if self._resize_locked or not edge:
                self.setCursor(QCursor(Qt_ArrowCursor))
            elif edge in ['left', 'right']:
                self.setCursor(QCursor(Qt_SizeHorCursor))
            elif edge in ['top', 'bottom']:
                self.setCursor(QCursor(Qt_SizeVerCursor))
            elif edge == 'top-left' or edge == 'bottom-right':
                self.setCursor(QCursor(Qt_SizeFDiagCursor))
            elif edge == 'top-right' or edge == 'bottom-left':
                self.setCursor(QCursor(Qt_SizeBDiagCursor))
            else:
                self.setCursor(QCursor(Qt_ArrowCursor))
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt_LeftButton:
            if self._drag_pos is not None:
                self.drag_ended.emit()
            elif self._resize_edge:
                # Save ratios when resize ends
                self._save_settings()
            self._drag_pos = None
            self._resize_edge = None
            self._resize_start_pos = None
            self._resize_start_ratios = None
            self.setCursor(QCursor(Qt_ArrowCursor))
            event.accept()
    
    def wheelEvent(self, event):
        """Scroll wheel resizes the avatar."""
        if self._avatar and hasattr(self._avatar, '_set_size'):
            try:
                delta = event.angleDelta().y()
            except AttributeError:
                delta = getattr(event, 'delta', lambda: 0)()
            
            current_size = getattr(self._avatar, '_size', 300)
            if delta > 0:
                new_size = current_size + 20  # Scroll up = bigger
            else:
                new_size = max(50, current_size - 20)  # Scroll down = smaller
            
            self._avatar._set_size(new_size)
            event.accept()
        else:
            event.ignore()


class BoneHitRegion(QWidget):
    """Individual hit region for a single bone.
    
    Users can resize each bone's hit area to match their avatar's shape.
    The region follows the bone when the AI animates the avatar.
    Left-drag center to move avatar, right-click for menu.
    Double-click or hold to trigger touch reactions.
    """
    
    EDGE_MARGIN = 8  # Pixels from edge for resize handles
    HOLD_THRESHOLD_MS = 500  # How long to hold for "hold" touch type
    
    # Signals for avatar dragging
    drag_started = pyqtSignal(object)
    drag_moved = pyqtSignal(object)
    drag_ended = pyqtSignal()
    context_menu_requested = pyqtSignal(object)
    
    # Touch reaction signal: (region_name, touch_type, global_pos)
    # touch_type: 'tap', 'hold', 'double_tap', 'pet' (repeated taps)
    touched = pyqtSignal(str, str, object)
    
    def __init__(self, bone_name: str, parent_manager):
        super().__init__(None)  # Top-level window
        self._bone_name = bone_name
        self._manager = parent_manager
        
        # Frameless, always on top, tool window
        self.setWindowFlags(
            Qt_FramelessWindowHint |
            Qt_WindowStaysOnTopHint |
            Qt_Tool
        )
        self.setAttribute(Qt_WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        
        # Size and position relative to bone center (in pixels)
        self._width = 40
        self._height = 40
        self._offset_x = 0  # Offset from bone center
        self._offset_y = 0
        
        # Screen position (updated by manager)
        self._screen_x = 0
        self._screen_y = 0
        
        # Interaction state
        self._drag_pos = None  # For moving avatar
        self._offset_drag_pos = None  # For adjusting bone offset (Shift+drag)
        self._resize_edge = None
        self._resize_start = None
        
        # Touch detection state
        self._touch_press_time = 0
        self._touch_press_pos = None
        self._touch_moved = False
        self._last_tap_time = 0
        self._tap_count = 0
        
        # Hold detection timer
        self._hold_timer = QTimer()
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._on_hold_detected)
        
        # Pet detection timer (resets tap count after delay)
        self._pet_timer = QTimer()
        self._pet_timer.setSingleShot(True)
        self._pet_timer.timeout.connect(self._on_pet_timeout)
        
        # Settings (from manager)
        self._show_border = True
        self._resize_locked = False
        
        self.setFixedSize(self._width, self._height)
        self.setCursor(QCursor(Qt_ArrowCursor))
        
        # Set tooltip to show bone name on hover
        self.setToolTip(self._bone_name)
    
    @property
    def bone_name(self) -> str:
        return self._bone_name
    
    def get_config(self) -> dict:
        """Get serializable config for this region."""
        return {
            'width': self._width,
            'height': self._height,
            'offset_x': self._offset_x,
            'offset_y': self._offset_y,
        }
    
    def set_config(self, config: dict):
        """Apply saved config."""
        self._width = config.get('width', 40)
        self._height = config.get('height', 40)
        self._offset_x = config.get('offset_x', 0)
        self._offset_y = config.get('offset_y', 0)
        self.setFixedSize(self._width, self._height)
    
    def update_position(self, screen_x: int, screen_y: int, width: int = None, height: int = None):
        """Update position and optionally size based on body region.
        
        Args:
            screen_x: Center X position on screen
            screen_y: Center Y position on screen  
            width: Optional new width (for body regions that scale with avatar)
            height: Optional new height
        """
        # Update size if provided
        if width is not None and height is not None:
            self._width = max(20, width)
            self._height = max(20, height)
            self.setFixedSize(self._width, self._height)
        
        # Position centered on the given coordinates
        self._screen_x = screen_x + self._offset_x - self._width // 2
        self._screen_y = screen_y + self._offset_y - self._height // 2
        self.move(self._screen_x, self._screen_y)
    
    def _get_edge_at_pos(self, pos):
        """Check if mouse is near an edge for resizing."""
        margin = self.EDGE_MARGIN
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        
        # Check corners
        if x < margin and y < margin:
            return 'top-left'
        if x > w - margin and y < margin:
            return 'top-right'
        if x < margin and y > h - margin:
            return 'bottom-left'
        if x > w - margin and y > h - margin:
            return 'bottom-right'
        
        # Check edges
        if x < margin:
            return 'left'
        if x > w - margin:
            return 'right'
        if y < margin:
            return 'top'
        if y > h - margin:
            return 'bottom'
        return None
    
    def set_border_visible(self, visible: bool):
        """Show or hide the border."""
        self._show_border = visible
        self.update()
    
    def set_resize_locked(self, locked: bool):
        """Lock or unlock resize."""
        self._resize_locked = locked
        self.update()
    
    def _on_hold_detected(self):
        """Called when user holds on region long enough."""
        if self._touch_press_pos and not self._touch_moved:
            self.touched.emit(self._bone_name, 'hold', self._touch_press_pos)
    
    def _on_pet_timeout(self):
        """Called after delay to check for petting (repeated taps)."""
        if self._tap_count >= 3:
            # Extended petting detected
            pass  # Already emitted 'pet' on third tap
        self._tap_count = 0
    
    def _check_tap_type(self, global_pos):
        """Determine tap type based on timing."""
        import time
        current_time = time.time()
        
        # Check for double tap (two taps within 400ms)
        if current_time - self._last_tap_time < 0.4:
            self._tap_count += 1
            if self._tap_count == 2:
                self.touched.emit(self._bone_name, 'double_tap', global_pos)
            elif self._tap_count >= 3:
                # Repeated taps = petting
                self.touched.emit(self._bone_name, 'pet', global_pos)
        else:
            # Single tap
            self._tap_count = 1
            self.touched.emit(self._bone_name, 'tap', global_pos)
        
        self._last_tap_time = current_time
        
        # Reset pet timer
        self._pet_timer.stop()
        self._pet_timer.start(600)  # Reset tap count after 600ms of no taps

    def paintEvent(self, event):
        """Draw the bone region with handles."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        w, h = self.width(), self.height()
        
        # If border hidden, just draw minimal fill for mouse capture
        if not self._show_border:
            painter.setBrush(QColor(0, 0, 0, 1))  # Nearly invisible
            painter.setPen(Qt_NoPen)
            painter.drawRect(0, 0, w, h)
            return
        
        # Semi-transparent fill
        painter.setBrush(QColor(200, 150, 100, 30))
        painter.setPen(Qt_NoPen)
        painter.drawRoundedRect(2, 2, w-4, h-4, 4, 4)
        
        # Orange dashed border for bone regions
        pen = QPen(QColor(255, 180, 100, 180))
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine if hasattr(Qt, 'DashLine') else 2)
        painter.setPen(pen)
        painter.setBrush(Qt_NoBrush)
        painter.drawRoundedRect(2, 2, w-4, h-4, 4, 4)
        
        # Draw handles at corners only if resize not locked
        if not self._resize_locked:
            handle_size = 6
            painter.setBrush(QColor(255, 180, 100, 220))
            painter.setPen(QColor(255, 220, 180))
            
            painter.drawRect(0, 0, handle_size, handle_size)
            painter.drawRect(w - handle_size, 0, handle_size, handle_size)
            painter.drawRect(0, h - handle_size, handle_size, handle_size)
            painter.drawRect(w - handle_size, h - handle_size, handle_size, handle_size)
    
    def mousePressEvent(self, event):
        import time as time_module
        if event.button() == Qt_LeftButton:
            edge = self._get_edge_at_pos(event.pos())
            # Check for Shift modifier - Shift+drag adjusts bone offset
            shift_held = event.modifiers() & Qt_ShiftModifier
            
            # Start touch tracking for tap/hold detection
            self._touch_press_time = time_module.time()
            self._touch_press_pos = event.globalPos()
            self._touch_moved = False
            
            if edge and not self._resize_locked:
                # Resize the bone region
                self._resize_edge = edge
                self._resize_start = {
                    'pos': event.globalPos(),
                    'width': self._width,
                    'height': self._height,
                    'offset_x': self._offset_x,
                    'offset_y': self._offset_y,
                    'x': self.x(),  # Store initial window position
                    'y': self.y(),
                }
            elif shift_held:
                # Shift+drag to adjust bone region offset
                self._offset_drag_pos = event.globalPos()
            else:
                # Normal drag moves the avatar
                if self._manager and self._manager._avatar:
                    self._drag_pos = event.globalPos() - self._manager._avatar.pos()
                    self.drag_started.emit(event.globalPos())
                
                # Start hold timer for touch detection
                self._hold_timer.start(self.HOLD_THRESHOLD_MS)
            event.accept()
        elif event.button() == Qt_RightButton:
            # Pass context menu up to manager
            self.context_menu_requested.emit(event.globalPos())
            event.accept()
    
    def mouseMoveEvent(self, event):
        # Check if mouse moved significantly (for tap detection)
        if self._touch_press_pos:
            delta = event.globalPos() - self._touch_press_pos
            if abs(delta.x()) > 5 or abs(delta.y()) > 5:
                self._touch_moved = True
                self._hold_timer.stop()  # Cancel hold detection if dragging
        
        if self._resize_edge and self._resize_start:
            delta = event.globalPos() - self._resize_start['pos']
            edge = self._resize_edge
            
            # Calculate new size based on edge being dragged
            # Each edge only moves that one side - not both
            new_w = self._resize_start['width']
            new_h = self._resize_start['height']
            new_x = self._resize_start.get('x', self.x())
            new_y = self._resize_start.get('y', self.y())
            
            # Right edge: increase width, left side stays fixed
            if 'right' in edge:
                new_w = max(20, self._resize_start['width'] + delta.x())
            
            # Left edge: increase width AND move window left
            if 'left' in edge:
                new_w = max(20, self._resize_start['width'] - delta.x())
                new_x = self._resize_start.get('x', self.x()) + delta.x()
            
            # Bottom edge: increase height, top side stays fixed
            if 'bottom' in edge:
                new_h = max(20, self._resize_start['height'] + delta.y())
            
            # Top edge: increase height AND move window up
            if 'top' in edge:
                new_h = max(20, self._resize_start['height'] - delta.y())
                new_y = self._resize_start.get('y', self.y()) + delta.y()
            
            self._width = new_w
            self._height = new_h
            self.setFixedSize(new_w, new_h)
            self.move(int(new_x), int(new_y))
            
        elif self._offset_drag_pos is not None:
            # Shift+drag: adjust bone region offset
            delta = event.globalPos() - self._offset_drag_pos
            self._offset_x += delta.x()
            self._offset_y += delta.y()
            self._offset_drag_pos = event.globalPos()
            self.move(self.x() + delta.x(), self.y() + delta.y())
            
        elif self._drag_pos is not None:
            # Normal drag: move the avatar
            self.drag_moved.emit(event.globalPos())
        else:
            # Update cursor based on position and lock state
            edge = self._get_edge_at_pos(event.pos())
            if self._resize_locked or not edge:
                self.setCursor(QCursor(Qt_ArrowCursor))
            elif edge in ['left', 'right']:
                self.setCursor(QCursor(Qt_SizeHorCursor))
            elif edge in ['top', 'bottom']:
                self.setCursor(QCursor(Qt_SizeVerCursor))
            elif edge in ['top-left', 'bottom-right']:
                self.setCursor(QCursor(Qt_SizeFDiagCursor))
            elif edge in ['top-right', 'bottom-left']:
                self.setCursor(QCursor(Qt_SizeBDiagCursor))
            else:
                self.setCursor(QCursor(Qt_ArrowCursor))
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt_LeftButton:
            # Stop hold timer
            self._hold_timer.stop()
            
            # Check for tap (didn't move much, wasn't a long hold)
            if not self._touch_moved and self._touch_press_pos:
                import time as time_module
                press_duration = time_module.time() - self._touch_press_time
                
                # If it wasn't a hold (timer would have fired), it's a tap
                if press_duration < (self.HOLD_THRESHOLD_MS / 1000.0):
                    self._check_tap_type(event.globalPos())
            
            # Reset touch state
            self._touch_press_pos = None
            self._touch_moved = False
            
            if self._drag_pos is not None:
                self.drag_ended.emit()
            self._drag_pos = None
            self._offset_drag_pos = None
            self._resize_edge = None
            self._resize_start = None
            self.setCursor(QCursor(Qt_ArrowCursor))
            # Save config through manager
            if self._manager:
                self._manager._save_bone_configs()
            event.accept()
    
    def wheelEvent(self, event):
        """Scroll wheel resizes the avatar."""
        if self._manager and self._manager._avatar and hasattr(self._manager._avatar, '_set_size'):
            try:
                delta = event.angleDelta().y()
            except AttributeError:
                delta = getattr(event, 'delta', lambda: 0)()
            
            current_size = getattr(self._manager._avatar, '_size', 300)
            if delta > 0:
                new_size = current_size + 20  # Scroll up = bigger
            else:
                new_size = max(50, current_size - 20)  # Scroll down = smaller
            
            self._manager._avatar._set_size(new_size)
            event.accept()
        else:
            event.ignore()


class ResizeHandle(QWidget):
    """A draggable resize handle for the avatar overlay.
    
    Appears as a small bar at the bottom-right corner when enabled.
    Drag it to resize the avatar - much more intuitive than scroll wheel.
    """
    
    resize_requested = pyqtSignal(int)  # Emits new size
    
    def __init__(self, avatar_window):
        super().__init__(None)  # Top-level window
        
        self._avatar = avatar_window
        self._drag_start = None
        self._start_size = None
        
        # Window setup - small handle at corner
        self.setWindowFlags(
            Qt_FramelessWindowHint |
            Qt_WindowStaysOnTopHint |
            Qt_Tool
        )
        self.setAttribute(Qt_WA_TranslucentBackground, True)
        
        self._handle_width = 60
        self._handle_height = 16
        self.setFixedSize(self._handle_width, self._handle_height)
        
        self.setCursor(QCursor(Qt_SizeFDiagCursor))  # Diagonal resize cursor
        
    def paintEvent(self, event):
        """Draw the resize handle."""
        from PyQt5.QtGui import QBrush, QColor, QPainter, QPen
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw handle background
        bg_color = QColor(69, 71, 90, 200)  # Semi-transparent dark
        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(QColor(137, 180, 250), 1))
        painter.drawRoundedRect(0, 0, self.width() - 1, self.height() - 1, 4, 4)
        
        # Draw resize grip lines (like a standard resize handle)
        grip_color = QColor(137, 180, 250)
        painter.setPen(QPen(grip_color, 2))
        
        # Three diagonal lines
        for i in range(3):
            offset = 6 + i * 6
            painter.drawLine(self.width() - offset, self.height() - 4,
                           self.width() - 4, self.height() - offset)
        
        painter.end()
    
    def mousePressEvent(self, event):
        if event.button() == Qt_LeftButton:
            self._drag_start = event.globalPos()
            self._start_size = self._avatar._size if self._avatar else 250
            event.accept()
    
    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            delta = event.globalPos() - self._drag_start
            # Use the larger of X or Y movement
            change = max(delta.x(), delta.y())
            new_size = max(50, self._start_size + change)  # Only minimum limit, no max cap
            
            if self._avatar:
                self._avatar._set_size(new_size, keep_center=False)
                self.update_position()
            
            event.accept()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt_LeftButton:
            self._drag_start = None
            self._start_size = None
            event.accept()
    
    def update_position(self):
        """Position handle at bottom-right corner of avatar."""
        if self._avatar:
            avatar_pos = self._avatar.pos()
            avatar_size = self._avatar._size
            
            # Position at bottom-right corner, slightly inside
            x = avatar_pos.x() + avatar_size - self._handle_width - 5
            y = avatar_pos.y() + avatar_size - self._handle_height - 5
            self.move(x, y)


class BoneHitManager(QObject):
    """Manages body region hit areas for an avatar.
    
    Uses 6 body regions instead of individual bones for simplicity:
    - Head (includes neck)
    - Torso (chest, spine, hips)
    - Left Arm (shoulder, arm, elbow, hand)
    - Right Arm (shoulder, arm, elbow, hand)
    - Left Leg (thigh, knee, foot)
    - Right Leg (thigh, knee, foot)
    
    This works with any model regardless of how many bones it has.
    """
    
    # Signal emitted when avatar is touched: (region_name, touch_type, global_pos)
    # region_name: 'head', 'torso', 'arm_left', 'arm_right', 'leg_left', 'leg_right'
    # touch_type: 'tap', 'double_tap', 'hold', 'pet' (repeated taps)
    region_touched = pyqtSignal(str, str, object)
    
    # Body regions with their screen positions (relative 0-1)
    # Each region covers a larger area than individual bones
    BODY_REGIONS = {
        'head': {
            'position': (0.5, 0.12),
            'size': (0.35, 0.20),  # width, height as ratio of avatar size
            'bones': ['head', 'neck', 'skull', 'face', 'jaw'],
        },
        'torso': {
            'position': (0.5, 0.42),
            'size': (0.45, 0.35),
            'bones': ['chest', 'spine', 'hips', 'pelvis', 'torso', 'body', 'root'],
        },
        'arm_left': {
            'position': (0.15, 0.40),
            'size': (0.25, 0.40),
            'bones': ['shoulder_l', 'arm_l', 'elbow_l', 'hand_l', 'finger', 'left_arm', 'l_arm'],
        },
        'arm_right': {
            'position': (0.85, 0.40),
            'size': (0.25, 0.40),
            'bones': ['shoulder_r', 'arm_r', 'elbow_r', 'hand_r', 'finger', 'right_arm', 'r_arm'],
        },
        'leg_left': {
            'position': (0.35, 0.78),
            'size': (0.25, 0.40),
            'bones': ['leg_l', 'knee_l', 'foot_l', 'thigh_l', 'shin_l', 'left_leg', 'l_leg'],
        },
        'leg_right': {
            'position': (0.65, 0.78),
            'size': (0.25, 0.40),
            'bones': ['leg_r', 'knee_r', 'foot_r', 'thigh_r', 'shin_r', 'right_leg', 'r_leg'],
        },
    }
    
    def __init__(self, avatar_window):
        super().__init__()
        self._avatar = avatar_window
        self._regions: dict[str, BoneHitRegion] = {}
        self._visible = False
        self._edit_mode = False
        
        # Settings (propagated to regions)
        self._show_border = True
        self._resize_locked = False
        
        # Update timer for tracking bone positions
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_positions)
        
        # Load saved configs and settings
        self._bone_configs = self._load_bone_configs()
        self._load_settings()
    
    def _load_settings(self):
        """Load saved border/lock settings."""
        try:
            from ....avatar.persistence import load_avatar_settings
            settings = load_avatar_settings()
            self._show_border = getattr(settings, 'hit_border_visible', True)
            self._resize_locked = getattr(settings, 'hit_resize_locked', False)
        except Exception:
            pass
    
    def set_border_visible(self, visible: bool):
        """Show or hide borders on all bone regions."""
        self._show_border = visible
        for region in self._regions.values():
            region.set_border_visible(visible)
        try:
            from ....avatar.persistence import save_avatar_settings
            save_avatar_settings(hit_border_visible=visible)
        except Exception:
            pass
    
    def set_resize_locked(self, locked: bool):
        """Lock or unlock resize on all bone regions."""
        self._resize_locked = locked
        for region in self._regions.values():
            region.set_resize_locked(locked)
        try:
            from ....avatar.persistence import save_avatar_settings
            save_avatar_settings(hit_resize_locked=locked)
        except Exception:
            pass
    
    def setup_bones(self, bone_names: list):
        """Create hit regions for the given bone names."""
        # Clear existing regions
        self.clear()
        
        for bone_name in bone_names:
            region = BoneHitRegion(bone_name, self)
            
            # Apply saved config if available
            if bone_name in self._bone_configs:
                region.set_config(self._bone_configs[bone_name])
            
            # Apply current settings
            region.set_border_visible(self._show_border)
            region.set_resize_locked(self._resize_locked)
            
            # Connect signals to avatar window for dragging
            if self._avatar:
                region.drag_started.connect(self._avatar._on_drag_bar_start)
                region.drag_moved.connect(self._avatar._on_drag_bar_move)
                region.drag_ended.connect(self._avatar._on_drag_bar_end)
                region.context_menu_requested.connect(self._avatar._show_context_menu_at)
            
            # Connect touch signal for AI interaction
            region.touched.connect(self.region_touched.emit)
            
            self._regions[bone_name] = region
        
        # Update positions immediately
        self._update_positions()
    
    def setup_from_gl_widget(self, gl_widget):
        """Setup body regions - always uses 6 standard regions regardless of model bones."""
        # Always use 6 body regions for consistent, manageable hitbox coverage
        # This works with any model - 3 bones or 59 bones
        self.setup_bones(list(self.BODY_REGIONS.keys()))
        print(f"[BoneHitManager] Using 6 body region hitboxes")
    
    def show(self):
        """Show all bone regions."""
        self._visible = True
        for region in self._regions.values():
            region.show()
        self._update_timer.start(50)  # 20fps updates
    
    def hide(self):
        """Hide all bone regions."""
        self._visible = False
        self._update_timer.stop()
        for region in self._regions.values():
            region.hide()
    
    def set_edit_mode(self, enabled: bool):
        """Enable or disable edit mode (allows resizing)."""
        self._edit_mode = enabled
        # Could add visual indicator for edit mode
    
    def clear(self):
        """Remove all bone regions."""
        self._update_timer.stop()
        for region in self._regions.values():
            region.close()
        self._regions.clear()
    
    def _update_positions(self):
        """Update all body region positions based on avatar state."""
        if not self._avatar or not self._visible:
            return
        
        avatar_pos = self._avatar.pos()
        avatar_size = self._avatar.width()
        
        # Get GL widget for rotation info
        gl_widget = getattr(self._avatar, '_gl_widget', None)
        model_yaw = 0.0
        if gl_widget:
            model_yaw = getattr(gl_widget, 'model_yaw', 0.0)
        
        for region_name, region in self._regions.items():
            # Get body region config
            region_config = self.BODY_REGIONS.get(region_name, {})
            pos = region_config.get('position', (0.5, 0.5))
            size = region_config.get('size', (0.2, 0.2))
            
            # Apply model yaw rotation to x position
            import math
            rel_x, rel_y = pos
            if model_yaw != 0:
                # Rotate x around center (0.5)
                cx = rel_x - 0.5
                rotated_x = cx * math.cos(model_yaw)
                rel_x = rotated_x + 0.5
            
            # Calculate screen position
            screen_x = avatar_pos.x() + int(rel_x * avatar_size)
            screen_y = avatar_pos.y() + int(rel_y * avatar_size)
            
            # Calculate region size based on avatar size
            region_w = int(size[0] * avatar_size)
            region_h = int(size[1] * avatar_size)
            
            region.update_position(screen_x, screen_y, region_w, region_h)
    
    def _show_bone_context_menu(self, region: BoneHitRegion, global_pos):
        """Show context menu for a bone region."""
        menu = QMenu()
        
        # Reset size action
        reset_action = menu.addAction("Reset Size")
        reset_action.triggered.connect(lambda: self._reset_region(region))
        
        # Hide this bone
        hide_action = menu.addAction("Hide This Bone")
        hide_action.triggered.connect(lambda: region.hide())
        
        menu.addSeparator()
        
        # Hide all bones
        hide_all_action = menu.addAction("Hide All Bones")
        hide_all_action.triggered.connect(self.hide)
        
        menu.exec_(global_pos)
    
    def _reset_region(self, region: BoneHitRegion):
        """Reset a region to default size."""
        region._width = 40
        region._height = 40
        region._offset_x = 0
        region._offset_y = 0
        region.setFixedSize(40, 40)
        self._update_positions()
        self._save_bone_configs()
    
    def _load_bone_configs(self) -> dict:
        """Load saved bone region configs."""
        try:
            from ....avatar.persistence import load_avatar_settings
            settings = load_avatar_settings()
            return getattr(settings, 'bone_hit_configs', {})
        except Exception:
            return {}
    
    def _save_bone_configs(self):
        """Save all bone region configs."""
        configs = {}
        for bone_name, region in self._regions.items():
            configs[bone_name] = region.get_config()
        
        try:
            from ....avatar.persistence import save_avatar_settings
            save_avatar_settings(bone_hit_configs=configs)
        except Exception:
            pass


class Avatar3DOverlayWindow(QWidget):
    """Advanced 3D overlay for desktop avatar display.
    
    Features:
    - Drag to move anywhere (rotation is DISABLED for desktop mode)
    - Right-click menu for options
    - Drag from edges/corners to resize (when enabled)
    - Adaptive animation system that works with ANY model
    - AI can control: position, size, emotion, speaking, gestures
    - Lip sync (adapts to model capabilities)
    - Blue border shows when resize mode is ON
    """
    
    closed = pyqtSignal()
    
    def __init__(self):
        super().__init__(None)
        
        # Transparent, always-on-top, frameless
        self.setWindowFlags(
            Qt_FramelessWindowHint |
            Qt_WindowStaysOnTopHint |
            Qt_Tool  # Don't show in taskbar
        )
        self.setAttribute(Qt_WA_TranslucentBackground, True)
        # WheelFocus so we can receive scroll wheel events
        self.setFocusPolicy(Qt.WheelFocus if hasattr(Qt, 'WheelFocus') else 0x0f)
        
        self._size = 250
        self.setFixedSize(self._size, self._size)
        self._resize_enabled = False  # Controlled from Avatar tab
        self._reposition_enabled = True  # Controlled from Avatar tab, default ON
        
        # NOTE: Edge-drag resize removed - use scroll wheel or size dialog instead
        
        # Start at bottom right
        screen = QApplication.primaryScreen()
        if screen:
            screen_geo = screen.availableGeometry()
            self.move(screen_geo.right() - self._size - 50, screen_geo.bottom() - self._size - 50)
        else:
            self.move(100, 100)
        
        # Click-through enabled flag - actual setup done after GL widget created
        self._click_through_enabled = True
        
        self._model_path = None
        self._use_circular_mask = False  # Disabled - can cause visual artifacts
        
        # Eye tracking state
        self._eye_tracking_enabled = False
        self._eye_timer = None
        
        # Drag bar - ALWAYS functional for dragging, can be made visible for resize/reposition
        self._show_control_bar = True  # DEFAULT: Show the drag bar so users can see it
        self._control_bar_height = 25  # Height of control bar
        self._control_bar_width = 100  # Width of control bar (smaller default)
        self._control_bar_x = 5  # Control bar X position (relative to avatar)
        self._control_bar_y = 5  # Control bar Y position (relative to avatar)
        # NOTE: Drag bar state is handled by DragBarWidget, not these variables
        
        # Adaptive Animator - works with ANY model
        self._animator = None
        try:
            from ....avatar.adaptive_animator import AdaptiveAnimator
            self._animator = AdaptiveAnimator()
            self._animator.on_transform_update(self._on_animator_transform)
        except ImportError as e:
            print(f"[Avatar3DOverlay] AdaptiveAnimator not available: {e}")
        
        # Idle animation state (DISABLED by default - looks weird on humanoid models)
        self._idle_animation_enabled = False  # Only enable for simple/abstract models
        self._idle_timer = QTimer()
        self._idle_timer.timeout.connect(self._do_idle_animation)
        self._idle_phase = 0.0  # Animation phase (radians)
        self._idle_bob_amount = 0.005  # Very subtle if enabled
        self._idle_sway_amount = 0.1  # Very subtle if enabled
        self._idle_speed = 0.03  # Slower
        self._base_pan_y = 0.0  # Base camera Y position
        self._base_rotation_y = 0.0  # Base camera Y rotation
        
        # Model info (for AI awareness)
        self._model_info = {
            'vertices': 0,
            'faces': 0,
            'has_colors': False,
            'bounds': None,
            'center': None,
        }
        
        # AI command watcher
        self._command_timer = QTimer()
        self._command_timer.timeout.connect(self._check_ai_commands)
        self._command_timer.start(500)
        self._last_command_time = 0
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Container for the 3D widget
        self._gl_container = QWidget()
        self._gl_container.setFixedSize(self._size, self._size)
        self._gl_container.setStyleSheet("background: transparent;")
        
        gl_layout = QVBoxLayout(self._gl_container)
        gl_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create 3D widget with transparent background
        if HAS_OPENGL and HAS_TRIMESH:
            self._gl_widget = OpenGL3DWidget(self._gl_container, transparent_bg=True)
            self._gl_widget.setFixedSize(self._size, self._size)
            # Disable GL widget's own mouse handling - parent handles interaction
            self._gl_widget.disable_mouse_interaction = True
            # Pass wheel events to parent for resizing
            self._gl_widget.wheelEvent = lambda e: self.wheelEvent(e)
            gl_layout.addWidget(self._gl_widget)
            # Apply visual mask (circular shape)
            self._apply_circular_mask()
        else:
            self._gl_widget = None
            placeholder = QLabel("3D not available")
            placeholder.setStyleSheet("color: white; background: rgba(30,30,46,150); border-radius: 50%;")
            placeholder.setAlignment(Qt_AlignCenter)
            gl_layout.addWidget(placeholder)
        
        main_layout.addWidget(self._gl_container)
        
        # Legacy references for compatibility
        self._bone_hit_manager = None  # Not used - pixel hit testing instead
        self._hit_layer = None
        self._drag_bar = None
        
        # Drag state
        self._drag_bar_offset = None
        self._is_dragging = False
        
        # Initialize pixel-based hit testing (only avatar pixels are clickable)
        self._make_click_through()
    
    def set_eye_tracking(self, enabled: bool):
        """Enable or disable eye tracking (follow cursor) for 3D avatar."""
        self._eye_tracking_enabled = enabled
        
        if enabled:
            # Start eye tracking timer for 3D camera rotation
            if self._eye_timer is None:
                self._eye_timer = QTimer(self)
                self._eye_timer.timeout.connect(self._update_eye_tracking_3d)
            self._eye_timer.start(50)  # 20 FPS updates
        else:
            # Stop and reset camera to default view
            if self._eye_timer:
                self._eye_timer.stop()
            if self._gl_widget:
                # Reset camera rotation to default
                self._gl_widget.rotation_x = 0
                self._gl_widget.rotation_y = 0
                self._gl_widget.update()
    
    def _update_eye_tracking_3d(self):
        """Update 3D camera rotation to follow cursor."""
        if not self._eye_tracking_enabled or not self._gl_widget:
            return
        
        # Get global cursor position
        cursor_pos = QCursor.pos()
        
        # Get avatar center position (global)
        avatar_center = self.mapToGlobal(QPoint(self.width() // 2, self.height() // 2))
        
        # Calculate direction to cursor
        dx = cursor_pos.x() - avatar_center.x()
        dy = cursor_pos.y() - avatar_center.y()
        
        # Convert to rotation angles (limited range for subtle effect)
        # Max rotation of 15 degrees in each direction
        max_angle = 15.0
        max_dist = 400.0
        
        target_rot_y = max(-max_angle, min(max_angle, (dx / max_dist) * max_angle))
        target_rot_x = max(-max_angle, min(max_angle, (dy / max_dist) * max_angle))
        
        # Smooth interpolation to target
        lerp_speed = 0.15
        current_x = getattr(self._gl_widget, 'rotation_x', 0)
        current_y = getattr(self._gl_widget, 'rotation_y', 0)
        
        self._gl_widget.rotation_x = current_x + (target_rot_x - current_x) * lerp_speed
        self._gl_widget.rotation_y = current_y + (target_rot_y - current_y) * lerp_speed
        self._gl_widget.update()
    
    def _update_drag_bar_position(self):
        """Update position tracking (for compatibility)."""
        pass  # No longer needed - avatar window handles everything
    
    def moveEvent(self, event):
        """When avatar moves, save position."""
        super().moveEvent(event)
    
    def resizeEvent(self, event):
        """When avatar resizes."""
        super().resizeEvent(event)
    
    def _make_click_through(self):
        """Set up click-through for transparent areas.
        
        Uses WS_EX_TRANSPARENT to make window click-through by default,
        then we selectively handle clicks on opaque (avatar) pixels.
        """
        self._use_pixel_hit_test = True
        self._hit_test_image = None
        self._is_dragging = False
        
        # Timer to update hit test image periodically
        self._hit_test_timer = QTimer()
        self._hit_test_timer.timeout.connect(self._update_hit_test_image)
        self._hit_test_timer.start(200)
        
        # Make window click-through using Windows extended styles
        import sys
        if sys.platform == 'win32':
            try:
                import ctypes
                hwnd = int(self.winId())
                GWL_EXSTYLE = -20
                WS_EX_LAYERED = 0x80000
                WS_EX_TRANSPARENT = 0x20
                user32 = ctypes.windll.user32
                current_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                # Add both LAYERED and TRANSPARENT
                user32.SetWindowLongW(hwnd, GWL_EXSTYLE, current_style | WS_EX_LAYERED | WS_EX_TRANSPARENT)
            except Exception as e:
                print(f"[Avatar] Could not set window style: {e}")
        
        # Start mouse tracking to detect when mouse is over avatar
        self._start_mouse_tracking()
    
    def _start_mouse_tracking(self):
        """Start tracking mouse position to enable/disable click-through dynamically."""
        self._mouse_track_timer = QTimer()
        self._mouse_track_timer.timeout.connect(self._check_mouse_over_avatar)
        self._mouse_track_timer.start(16)  # ~60 FPS for responsive feel
    
    def _check_mouse_over_avatar(self):
        """Check if mouse is over an opaque pixel and toggle click-through accordingly."""
        import sys
        if sys.platform != 'win32':
            return
        
        try:
            import ctypes

            # Get current mouse position
            cursor_pos = QCursor.pos()
            local_pos = self.mapFromGlobal(cursor_pos)
            
            # Check if mouse is within our window bounds
            if not self.rect().contains(local_pos):
                return
            
            # Check if pixel is opaque (part of avatar)
            is_opaque = self._is_pixel_opaque(local_pos.x(), local_pos.y())
            
            # Toggle WS_EX_TRANSPARENT based on pixel opacity
            hwnd = int(self.winId())
            GWL_EXSTYLE = -20
            WS_EX_TRANSPARENT = 0x20
            user32 = ctypes.windll.user32
            
            current_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            has_transparent = bool(current_style & WS_EX_TRANSPARENT)
            
            if is_opaque and has_transparent:
                # Mouse over avatar - remove transparent flag to receive clicks
                user32.SetWindowLongW(hwnd, GWL_EXSTYLE, current_style & ~WS_EX_TRANSPARENT)
            elif not is_opaque and not has_transparent:
                # Mouse over background - add transparent flag to pass clicks through
                user32.SetWindowLongW(hwnd, GWL_EXSTYLE, current_style | WS_EX_TRANSPARENT)
                
        except Exception:
            pass
    
    def _update_hit_test_image(self):
        """Capture the current GL frame for pixel-based hit testing."""
        if not self._gl_widget:
            return
        try:
            self._hit_test_image = self._gl_widget.grabFramebuffer()
        except Exception:
            pass
    
    def _is_pixel_opaque(self, x: int, y: int, threshold: int = 30) -> bool:
        """Check if the pixel at (x, y) is opaque (part of the avatar).
        
        Uses a combination of alpha and RGB brightness to detect the avatar mesh.
        The background is cleared to (0,0,0,0) so any pixel with color is the model.
        
        Args:
            x, y: Position relative to widget
            threshold: Brightness threshold below which pixel is considered transparent
            
        Returns:
            True if pixel is opaque (should handle click), False if transparent
        """
        if self._hit_test_image is None:
            # No image yet - try to grab one now
            self._update_hit_test_image()
            if self._hit_test_image is None:
                return False  # Still no image = transparent
        
        # Scale coordinates if image size differs from widget size
        img_w = self._hit_test_image.width()
        img_h = self._hit_test_image.height()
        wid_w = self.width()
        wid_h = self.height()
        
        if img_w <= 0 or img_h <= 0:
            return False
        
        if wid_w > 0 and wid_h > 0:
            scaled_x = int(x * img_w / wid_w)
            scaled_y = int(y * img_h / wid_h)
        else:
            scaled_x, scaled_y = x, y
        
        # Bounds check
        if scaled_x < 0 or scaled_x >= img_w or scaled_y < 0 or scaled_y >= img_h:
            return False
        
        # Get pixel color
        pixel = self._hit_test_image.pixelColor(scaled_x, scaled_y)
        
        # Check alpha first (if framebuffer preserves it)
        if pixel.alpha() > threshold:
            return True
        
        # Fallback: check if pixel has ANY color (background is black)
        # This handles cases where alpha channel isn't preserved properly
        brightness = pixel.red() + pixel.green() + pixel.blue()
        return brightness > threshold
    
    def _apply_click_through(self):
        """No-op - using dynamic click-through instead."""
        pass
    
    def _apply_click_through_windows(self):
        """No-op - using dynamic click-through instead."""
        pass
    
    def _apply_click_through_linux(self):
        """Linux: Make avatar window click-through using X11 input shape."""
        try:
            import ctypes

            from PyQt5.QtX11Extras import QX11Info
            
            x11 = ctypes.CDLL('libX11.so.6')
            xext = ctypes.CDLL('libXext.so.6')
            
            display = QX11Info.display()
            window_id = int(self.winId())
            
            # Set empty input shape - window receives no input
            # ShapeInput = 2, ShapeSet = 0
            ShapeInput = 2
            ShapeSet = 0
            
            # Empty region = no input
            XShapeCombineRectangles = xext.XShapeCombineRectangles
            XShapeCombineRectangles(display, window_id, ShapeInput, 0, 0, None, 0, ShapeSet, 0)
            x11.XFlush(display)
            
            print("[Avatar] Linux click-through enabled (X11 input shape)")
            
        except ImportError:
            print("[Avatar] X11 extras not available for Linux click-through")
        except Exception as e:
            print(f"[Avatar] Linux click-through setup failed: {e}")
    
    def _on_drag_bar_start(self, global_pos):
        """Called when drag bar drag starts."""
        self._drag_bar_offset = global_pos - self.pos()
    
    def _on_drag_bar_move(self, global_pos):
        """Called when drag bar is being dragged."""
        if self._drag_bar_offset is not None:
            new_pos = global_pos - self._drag_bar_offset
            # Constrain to screen
            try:
                from PyQt5.QtWidgets import QDesktopWidget
                desktop = QDesktopWidget()
                virtual_geo = desktop.geometry()
                min_visible = 50
                new_x = max(min_visible - self._size, min(virtual_geo.right() - min_visible, new_pos.x()))
                new_y = max(min_visible - self._size, min(virtual_geo.bottom() - min_visible, new_pos.y()))
                new_pos.setX(new_x)
                new_pos.setY(new_y)
            except Exception:
                pass
            self.move(new_pos)
    
    def _on_drag_bar_end(self):
        """Called when drag bar drag ends."""
        self._drag_bar_offset = None
        # Save position
        try:
            from ....avatar.persistence import save_position, write_avatar_state_for_ai
            pos = self.pos()
            save_position(pos.x(), pos.y())
            write_avatar_state_for_ai()
        except Exception:
            pass
    
    def _on_drag_bar_repositioned(self, new_x: int, new_y: int):
        """Called when drag bar is repositioned within the window (Shift+drag)."""
        self._control_bar_x = new_x
        self._control_bar_y = new_y
        # Update window mask to include new drag bar position
        self._apply_circular_mask()
        # Save position
        try:
            from ....avatar.persistence import (
                save_avatar_settings,
                write_avatar_state_for_ai,
            )
            save_avatar_settings(control_bar_x=new_x, control_bar_y=new_y)
            write_avatar_state_for_ai()
        except Exception:
            pass

    def _on_animator_transform(self, pos_offset, rot_offset, scale):
        """Callback from AdaptiveAnimator for transform updates."""
        if not self._gl_widget:
            return
        
        # Apply position offset (Y = vertical bob)
        self._gl_widget.pan_y = self._base_pan_y + pos_offset[1]
        
        # Apply rotation offset (Y = sway)
        self._gl_widget.rotation_y = self._base_rotation_y + rot_offset[1]
        
        # Apply scale (for speaking pulse effect)
        # Note: OpenGL widget doesn't directly support scale, but we can adjust zoom
        # self._gl_widget.zoom = self._base_zoom / scale
        
        self._gl_widget.update()
    
    def _apply_circular_mask(self):
        """Apply window mask for visual shape only.
        
        This controls the VISIBLE shape of the window, NOT click-through behavior.
        Click-through is handled separately by _apply_click_through().
        
        Options:
        - Circular mask: avatar appears as circle
        - No mask: avatar appears as square/rectangle
        """
        from PyQt5.QtCore import QRectF
        from PyQt5.QtGui import QPainterPath, QRegion

        # Clear any existing mask first
        self.clearMask()
        
        # If circular mask is disabled, leave window as rectangle
        if not self._use_circular_mask:
            return
        
        # Create circular path for the avatar
        path = QPainterPath()
        padding = 5
        path.addEllipse(QRectF(padding, padding, self._size - padding*2, self._size - padding*2))
        circle_region = QRegion(path.toFillPolygon().toPolygon())
        
        # Get drag bar region to include in visual mask
        if hasattr(self, '_drag_bar') and self._drag_bar:
            bar_x = self._drag_bar.x()
            bar_y = self._drag_bar.y()
            bar_w = self._drag_bar.width()
            bar_h = self._drag_bar.height()
        else:
            bar_x = getattr(self, '_control_bar_x', 5)
            bar_y = getattr(self, '_control_bar_y', 5)
            bar_w = getattr(self, '_control_bar_width', 100)
            bar_h = getattr(self, '_control_bar_height', 25)
        
        # Add drag bar region with margin
        drag_bar_region = QRegion(bar_x - 2, bar_y - 2, bar_w + 4, bar_h + 4)
        
        # Combine circle + drag bar for visual mask
        window_region = circle_region.united(drag_bar_region)
        self.setMask(window_region)
    
    def load_model(self, path: str) -> bool:
        """Load a 3D model into the overlay."""
        self._model_path = path
        if self._gl_widget:
            result = self._gl_widget.load_model(path)
            self._apply_circular_mask()
            
            # Capture model info for AI awareness
            if result and self._gl_widget.vertices is not None:
                import numpy as np
                self._model_info = {
                    'vertices': len(self._gl_widget.vertices),
                    'faces': len(self._gl_widget.faces) if self._gl_widget.faces is not None else 0,
                    'has_colors': self._gl_widget.colors is not None and len(np.unique(self._gl_widget.colors, axis=0)) > 1,
                    'bounds': {
                        'min': self._gl_widget.vertices.min(axis=0).tolist(),
                        'max': self._gl_widget.vertices.max(axis=0).tolist(),
                    },
                    'center': self._gl_widget.vertices.mean(axis=0).tolist(),
                    'size': (self._gl_widget.vertices.max(axis=0) - self._gl_widget.vertices.min(axis=0)).tolist(),
                }
                
                # Add detailed model metadata from analysis
                metadata = self._gl_widget.get_model_metadata()
                if metadata:
                    self._model_info['mesh_count'] = metadata.get('mesh_count', 0)
                    self._model_info['mesh_names'] = metadata.get('mesh_names', [])
                    self._model_info['has_skeleton'] = metadata.get('has_skeleton', False)
                    self._model_info['skeleton_bones'] = metadata.get('skeleton_bones', [])
                    self._model_info['estimated_type'] = metadata.get('estimated_type', 'unknown')
                    self._model_info['has_textures'] = metadata.get('has_textures', False)
                    self._model_info['materials'] = metadata.get('materials', [])
                    
                    # Connect bone controller if model has skeleton
                    if self._model_info['has_skeleton'] and self._model_info['skeleton_bones']:
                        try:
                            from ....avatar import get_avatar
                            from ....avatar.bone_control import get_bone_controller

                            # Get avatar controller first
                            avatar_controller = get_avatar()
                            if avatar_controller is None:
                                print("[Avatar] Warning: Avatar controller not initialized, bone control unavailable")
                            else:
                                # Link bone controller to avatar controller
                                bone_controller = get_bone_controller(avatar_controller=avatar_controller)
                                bone_controller.set_avatar_bones(self._model_info['skeleton_bones'])
                                
                                # Write bone info for AI to read
                                bone_controller.write_info_for_ai()
                                
                                print(f"[Avatar] Bone controller initialized with {len(self._model_info['skeleton_bones'])} bones")
                                print(f"[Avatar] Detected bones: {', '.join(self._model_info['skeleton_bones'][:5])}"
                                      f"{'...' if len(self._model_info['skeleton_bones']) > 5 else ''}")
                                print(f"[Avatar] Bone animation is now PRIMARY control (priority 100)")
                        except ImportError as e:
                            print(f"[Avatar] Bone controller module not available: {e}")
                        except Exception as e:
                            print(f"[Avatar] Could not initialize bone controller: {e}")
                            import traceback
                            traceback.print_exc()
                
                # Initialize AdaptiveAnimator with model capabilities
                if self._animator:
                    self._animator.set_model_info(self._model_info)
                    self._animator.start()
                    # Write capabilities for AI
                    caps = self._animator.get_capabilities_for_ai()
                    caps_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "avatar" / "capabilities.json"
                    caps_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(caps_path, 'w') as f:
                        json.dump(caps, f, indent=2)
                    print(f"[Avatar3DOverlay] Model capabilities written to {caps_path}")
                
                # Save base camera positions
                self._base_pan_y = self._gl_widget.pan_y
                self._base_rotation_y = self._gl_widget.rotation_y
                
                # Don't start idle animation - it looks weird on humanoid models
                # The AdaptiveAnimator handles proper animations via bone/blendshape
                
                # DON'T show bone hitboxes anymore - using pixel-based hit testing instead
                # The avatar window itself is now clickable (only on visible pixels)
                if hasattr(self, '_bone_hit_manager') and self._bone_hit_manager:
                    self._bone_hit_manager.hide()  # Hide blocky hitboxes
            
            return result
        return False
    
    def _start_idle_animation(self):
        """Start the idle bobbing/swaying animation."""
        if not self._idle_timer.isActive():
            self._idle_phase = 0.0
            self._idle_timer.start(33)  # ~30fps for smooth animation
    
    def _stop_idle_animation(self):
        """Stop the idle animation and reset to base position."""
        self._idle_timer.stop()
        if self._gl_widget:
            self._gl_widget.pan_y = self._base_pan_y
            self._gl_widget.rotation_y = self._base_rotation_y
            self._gl_widget.update()
    
    def _do_idle_animation(self):
        """Perform one frame of idle animation."""
        import math
        
        if not self._gl_widget:
            return
        
        self._idle_phase += self._idle_speed
        
        # Gentle bobbing (vertical movement)
        bob_offset = math.sin(self._idle_phase) * self._idle_bob_amount
        self._gl_widget.pan_y = self._base_pan_y + bob_offset
        
        # Gentle swaying (subtle rotation)
        sway_offset = math.sin(self._idle_phase * 0.7) * self._idle_sway_amount
        self._gl_widget.rotation_y = self._base_rotation_y + sway_offset
        
        self._gl_widget.update()
    
    def set_idle_animation(self, enabled: bool):
        """Enable or disable idle animation."""
        self._idle_animation_enabled = enabled
        if enabled:
            self._start_idle_animation()
        else:
            self._stop_idle_animation()
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model for AI awareness."""
        return self._model_info.copy()
    
    # NOTE: _get_edge_at_pos and _do_resize removed - DragBarWidget handles interaction
    # NOTE: eventFilter removed - DragBarWidget handles all interaction now
    
    def showEvent(self, a0):
        """Restore saved position when shown."""
        super().showEvent(a0)
        try:
            from ....avatar.persistence import load_position
            x, y = load_position()
            if x is not None and y is not None:
                self.move(x, y)
        except Exception:
            pass
    
    def paintEvent(self, event):
        """Draw visual indicators."""
        super().paintEvent(event)
        # No border drawing - clean transparent overlay

    def mousePressEvent(self, event):
        """Handle mouse press - avatar is clickable, background passes through."""
        if event.button() == Qt_LeftButton:
            # Only start drag if reposition is enabled
            if getattr(self, '_reposition_enabled', True):
                self._drag_bar_offset = event.globalPos() - self.pos()
                self._is_dragging = True
                self.setCursor(QCursor(Qt_ClosedHandCursor))
            event.accept()
        elif event.button() == Qt_RightButton:
            # Show context menu
            self._show_context_menu_at(event.globalPos())
            event.accept()
        else:
            event.ignore()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging."""
        if self._drag_bar_offset is not None:
            # Dragging avatar
            new_pos = event.globalPos() - self._drag_bar_offset
            # Constrain to screen
            try:
                from PyQt5.QtWidgets import QDesktopWidget
                desktop = QDesktopWidget()
                virtual_geo = desktop.geometry()
                min_visible = 50
                new_x = max(min_visible - self._size, min(virtual_geo.right() - min_visible, new_pos.x()))
                new_y = max(min_visible - self._size, min(virtual_geo.bottom() - min_visible, new_pos.y()))
                new_pos.setX(new_x)
                new_pos.setY(new_y)
            except Exception:
                pass
            self.move(new_pos)
            event.accept()
        else:
            self.setCursor(QCursor(Qt_OpenHandCursor))
            event.ignore()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt_LeftButton and self._drag_bar_offset is not None:
            self._drag_bar_offset = None
            self._is_dragging = False
            self.setCursor(QCursor(Qt_OpenHandCursor))
            # Save position
            try:
                from ....avatar.persistence import (
                    save_position,
                    write_avatar_state_for_ai,
                )
                pos = self.pos()
                save_position(pos.x(), pos.y())
                write_avatar_state_for_ai()
            except Exception:
                pass
            event.accept()
        else:
            event.ignore()

    def contextMenuEvent(self, event):
        """Handle right-click context menu."""
        self._show_context_menu_at(event.globalPos())
        event.accept()
    
    def wheelEvent(self, event):
        """Scroll wheel to resize avatar - only when resize is enabled from Avatar tab."""
        # Only resize if enabled from Avatar tab checkbox
        if not getattr(self, '_resize_enabled', False):
            event.ignore()
            return
        
        try:
            delta = event.angleDelta().y()
        except AttributeError:
            delta = getattr(event, 'delta', lambda: 0)()
        
        step = 15
        if delta > 0:
            new_size = self._size + step  # Scroll up = bigger
        else:
            new_size = max(50, self._size - step)  # Scroll down = smaller, min 50
        
        self._set_size(new_size, keep_center=True)
        event.accept()
    
    def _show_context_menu_at(self, global_pos):
        """Show context menu at the given global position."""
        from PyQt5.QtWidgets import QMenu
        
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
        
        # Gestures submenu
        gestures_menu = menu.addMenu("Gestures")
        gesture_actions = [
            ("Wave", "wave"),
            ("High Five", "highfive"),
            ("Hug", "hug"),
            ("Dance", "dance"),
            ("Laugh", "laugh"),
            ("Tease", "tease"),
            ("Rock Paper Scissors", "rps"),
        ]
        for label, gesture in gesture_actions:
            action = gestures_menu.addAction(label)
            action.triggered.connect(lambda checked, g=gesture: self._request_gesture(g))
        
        menu.addSeparator()
        
        # Resize toggle
        resize_text = "Disable Resize" if getattr(self, '_resize_enabled', False) else "Enable Resize"
        resize_action = menu.addAction(resize_text)
        resize_action.triggered.connect(self._toggle_resize)
        
        # Reposition toggle
        reposition_text = "Disable Reposition" if getattr(self, '_reposition_enabled', True) else "Enable Reposition"
        reposition_action = menu.addAction(reposition_text)
        reposition_action.triggered.connect(self._toggle_reposition)
        
        menu.addSeparator()
        
        # Center on screen
        center_action = menu.addAction("Center on Screen")
        center_action.triggered.connect(self._center_on_screen)
        
        menu.addSeparator()
        
        # Hide avatar
        close_action = menu.addAction("Hide Avatar")
        close_action.triggered.connect(self._close)
        
        menu.exec_(global_pos)
    
    def _request_gesture(self, gesture: str):
        """Request a gesture from the AI - AI decides how to react."""
        # Send gesture request to AI via conversation system
        try:
            from ....memory import ConversationManager
            conv = ConversationManager()
            # Add as a system message so AI knows user requested this
            conv.add_message("system", f"[User requested gesture: {gesture}]")
        except Exception:
            pass
    
    def _toggle_hit_border(self):
        """Toggle drag area border visibility."""
        if hasattr(self, '_hit_layer') and self._hit_layer:
            current = getattr(self._hit_layer, '_show_border', True)
            self._hit_layer.set_border_visible(not current)
    
    def _toggle_hit_lock(self):
        """Toggle drag area resize lock."""
        if hasattr(self, '_hit_layer') and self._hit_layer:
            current = getattr(self._hit_layer, '_resize_locked', False)
            self._hit_layer.set_resize_locked(not current)
            self._hit_layer.update()  # Redraw to hide/show handles
    
    def _reset_drag_area(self):
        """Reset drag area to default centered size."""
        if hasattr(self, '_hit_layer') and self._hit_layer:
            self._hit_layer._ratio_left = 0.12
            self._hit_layer._ratio_right = 0.12
    def _toggle_hit_border(self):
        """Toggle bone region border visibility."""
        if hasattr(self, '_bone_hit_manager') and self._bone_hit_manager:
            current = self._bone_hit_manager._show_border
            self._bone_hit_manager.set_border_visible(not current)
    
    def _toggle_hit_lock(self):
        """Toggle bone region resize lock."""
        if hasattr(self, '_bone_hit_manager') and self._bone_hit_manager:
            current = self._bone_hit_manager._resize_locked
            self._bone_hit_manager.set_resize_locked(not current)
    
    def _reset_bone_regions(self):
        """Reset all bone regions to default sizes."""
        if hasattr(self, '_bone_hit_manager') and self._bone_hit_manager:
            for region in self._bone_hit_manager._regions.values():
                self._bone_hit_manager._reset_region(region)
    
    def _toggle_resize(self):
        """Toggle resize mode - controls scroll wheel resizing."""
        self._resize_enabled = not getattr(self, '_resize_enabled', False)
        self.update()
        # Save setting
        try:
            from ....avatar.persistence import (
                save_avatar_settings,
                write_avatar_state_for_ai,
            )
            save_avatar_settings(resize_enabled=self._resize_enabled)
            write_avatar_state_for_ai()
        except Exception as e:
            print(f"[Avatar] Error saving resize state: {e}")
    
    def _toggle_reposition(self):
        """Toggle reposition mode - controls drag to move."""
        self._reposition_enabled = not getattr(self, '_reposition_enabled', True)
        self.update()
        # Save setting
        try:
            from ....avatar.persistence import (
                save_avatar_settings,
                write_avatar_state_for_ai,
            )
            save_avatar_settings(reposition_enabled=self._reposition_enabled)
            write_avatar_state_for_ai()
        except Exception as e:
            print(f"[Avatar] Error saving reposition state: {e}")
    
    def _toggle_control_bar(self):
        """Toggle hit layer visibility (mostly for debugging - it's invisible anyway)."""
        self._show_control_bar = not getattr(self, '_show_control_bar', False)
        # Show/hide the hit layer
        if hasattr(self, '_hit_layer'):
            self._hit_layer.setVisible(self._show_control_bar)
        self._apply_circular_mask()
        self.update()
        # Save setting
        try:
            from ....avatar.persistence import (
                save_avatar_settings,
                write_avatar_state_for_ai,
            )
            save_avatar_settings(show_control_bar_3d=self._show_control_bar)
            write_avatar_state_for_ai()
        except Exception:
            pass
    
    def _toggle_ghost_mode(self):
        """No-op - hit layer is always invisible."""
        pass  # Hit layer is always invisible, no ghost mode needed
    
    def _toggle_anchored_mode(self):
        """No-op - hit layer is always active and covers entire avatar."""
        pass  # Hit layer is automatic
    
    def _set_anchor_position(self, x: float, y: float):
        """No-op - hit layer covers entire avatar, no positioning needed."""
        pass  # Hit layer is automatic
    
    def _toggle_reposition_mode(self):
        """No-op - hit layer is automatic, no reposition mode needed."""
        pass  # Hit layer is automatic
    
    def _set_bar_size(self, width, height):
        """No-op - hit layer automatically matches avatar size."""
        pass  # Hit layer is automatic
    
    def _show_bar_size_dialog(self):
        """No-op - hit layer automatically matches avatar size."""
        pass  # Hit layer is automatic

    def _toggle_click_through_mode(self):
        """Enable click-through mode - clicks pass through avatar."""
        self._click_through_mode = True
        self.update()
        try:
            from ....avatar.persistence import (
                save_avatar_settings,
                write_avatar_state_for_ai,
            )
            save_avatar_settings(click_through_mode_3d=self._click_through_mode)
            write_avatar_state_for_ai()
        except Exception:
            pass
    
    def _set_always_interactive(self):
        """Set always interactive mode - drag anywhere on avatar."""
        self._click_through_mode = False
        self.update()
        try:
            from ....avatar.persistence import (
                save_avatar_settings,
                write_avatar_state_for_ai,
            )
            save_avatar_settings(click_through_mode_3d=self._click_through_mode)
            write_avatar_state_for_ai()
        except Exception:
            pass
    
    def _center_on_screen(self):
        """Center the avatar on the current screen."""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geo = screen.availableGeometry()
            self.move(
                screen_geo.center().x() - self._size // 2,
                screen_geo.center().y() - self._size // 2
            )
    
    def _show_size_dialog(self):
        """Show dialog to set avatar size."""
        from PyQt5.QtWidgets import QInputDialog
        size, ok = QInputDialog.getInt(
            self, "Set Avatar Size", "Size (pixels):",
            self._size, 50, 2000, 10
        )
        if ok:
            self._size = size
            self.setFixedSize(self._size, self._size)
            self._gl_container.setFixedSize(self._size, self._size)
            if self._gl_widget:
                self._gl_widget.setFixedSize(self._size, self._size)
                self._apply_circular_mask()
            # Update hit layer position
            if hasattr(self, '_hit_layer') and self._hit_layer:
                self._hit_layer._sync_with_avatar()
            # Save size
            try:
                from ....avatar.persistence import (
                    save_avatar_settings,
                    write_avatar_state_for_ai,
                )
                save_avatar_settings(overlay_3d_size=self._size)
                write_avatar_state_for_ai()
            except Exception:
                pass
    
    def _set_size(self, size: int, keep_center: bool = True, smooth: bool = False):
        """Set avatar size programmatically (used by AI commands and scroll wheel).
        
        Args:
            size: New size in pixels
            keep_center: If True, keep avatar centered at same position
            smooth: If True, use smaller increments for smoother resize
        
        Uses debounced saving to prevent jitter from rapid resize events.
        """
        size = max(50, size)  # Only minimum limit, no maximum cap
        
        # Skip if size hasn't changed
        if size == self._size:
            return
        
        # Calculate center before resize
        old_center_x = self.x() + self._size // 2
        old_center_y = self.y() + self._size // 2
        
        self._size = size
        
        # Batch all size changes together to reduce jitter
        self.setFixedSize(self._size, self._size)
        self._gl_container.setFixedSize(self._size, self._size)
        if self._gl_widget:
            self._gl_widget.setFixedSize(self._size, self._size)
        
        # Keep centered at same position
        if keep_center:
            new_x = old_center_x - self._size // 2
            new_y = old_center_y - self._size // 2
            self.move(new_x, new_y)
        
        # Apply mask after position is set (not during resize)
        if self._gl_widget:
            # Debounce mask application to reduce flicker
            if not hasattr(self, '_mask_timer'):
                self._mask_timer = QTimer()
                self._mask_timer.setSingleShot(True)
                self._mask_timer.timeout.connect(self._apply_circular_mask)
            self._mask_timer.start(50)  # Apply mask 50ms after last resize
        
        # Update drag bar position
        self._update_drag_bar_position()
        
        # Debounce save to prevent excessive disk writes during rapid resize
        if not hasattr(self, '_save_timer'):
            self._save_timer = QTimer()
            self._save_timer.setSingleShot(True)
            self._save_timer.timeout.connect(self._save_size_debounced)
        self._save_timer.start(300)  # Save 300ms after last resize
    
    def _save_size_debounced(self):
        """Save size after debounce delay."""
        try:
            from ....avatar.persistence import (
                save_avatar_settings,
                write_avatar_state_for_ai,
            )
            save_avatar_settings(overlay_3d_size=self._size)
            write_avatar_state_for_ai()
        except Exception:
            pass

    def _check_ai_commands(self):
        """Check for AI commands and execute them."""
        import json
        from datetime import datetime
        
        command_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "avatar" / "ai_command.json"
        if not command_path.exists():
            return
        
        try:
            with open(command_path) as f:
                cmd = json.load(f)
            
            timestamp = cmd.get("timestamp", 0)
            if timestamp <= self._last_command_time:
                return
            
            self._last_command_time = timestamp
            action = cmd.get("action", "").lower()
            value = cmd.get("value", "")
            
            # Log the command to shared activity log
            _log_avatar_activity(action, value)
            
            if action == "show":
                self.show()
            elif action == "hide":
                self.hide()
                self.closed.emit()
            elif action == "move":
                try:
                    parts = value.split(",")
                    if len(parts) >= 2:
                        x, y = int(parts[0].strip()), int(parts[1].strip())
                        self.move(x, y)
                except ValueError:
                    pass
            elif action == "resize":
                try:
                    size = int(value)
                    self._set_size(size)
                except ValueError:
                    pass
            elif action == "orientation":
                value_lower = value.lower()
                if value_lower == "front":
                    self._set_view_angle(0, 0)
                elif value_lower == "back":
                    self._set_view_angle(0, 180)
                elif value_lower == "left":
                    self._set_view_angle(0, -90)
                elif value_lower == "right":
                    self._set_view_angle(0, 90)
                else:
                    try:
                        parts = value.split(",")
                        if len(parts) >= 2:
                            rx, ry = float(parts[0].strip()), float(parts[1].strip())
                            self._set_view_angle(rx, ry)
                    except ValueError:
                        pass
            
            # === NEW: AdaptiveAnimator commands ===
            elif action == "speak":
                # Make avatar speak (lip sync or visual pulse)
                if self._animator:
                    duration = float(cmd.get("duration", 0)) if cmd.get("duration") else None
                    self._animator.speak(value, duration)
                else:
                    # Fallback: just set speaking state visually
                    self._do_speaking_pulse()
            
            elif action == "emotion":
                # Set emotion/expression
                if self._animator:
                    self._animator.set_emotion(value)
            
            elif action == "nod":
                # Nod gesture (yes)
                if self._animator:
                    intensity = float(value) if value else 1.0
                    self._animator.nod(intensity)
                else:
                    self._do_nod_fallback()
            
            elif action == "shake":
                # Shake gesture (no)
                if self._animator:
                    intensity = float(value) if value else 1.0
                    self._animator.shake(intensity)
                else:
                    self._do_shake_fallback()
            
            elif action == "wave":
                # Wave gesture
                if self._animator:
                    self._animator.wave()
            
            elif action == "blink":
                # Blink
                if self._animator:
                    self._animator.blink()
            
            elif action == "look_at":
                # Look at position
                if self._animator:
                    try:
                        parts = value.split(",")
                        if len(parts) >= 2:
                            x, y = float(parts[0].strip()), float(parts[1].strip())
                            z = float(parts[2].strip()) if len(parts) > 2 else 0.0
                            self._animator.look_at(x, y, z)
                    except ValueError:
                        pass
            
            elif action == "get_capabilities":
                # Write avatar capabilities for AI to read
                caps_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "avatar" / "capabilities.json"
                caps_path.parent.mkdir(parents=True, exist_ok=True)
                if self._animator:
                    caps = self._animator.get_capabilities_for_ai()
                else:
                    caps = {'strategy': 'TRANSFORM', 'can_lip_sync': False, 'can_emote': False}
                caps['model_info'] = self._model_info
                with open(caps_path, 'w') as f:
                    json.dump(caps, f, indent=2)
            
            # Legacy commands
            elif action == "animate":
                value_lower = value.lower()
                if value_lower in ("true", "on", "start", "yes", "1"):
                    self.set_idle_animation(True)
                elif value_lower in ("false", "off", "stop", "no", "0"):
                    self.set_idle_animation(False)
            elif action == "animation_speed":
                try:
                    speed = float(value)
                    if speed > 1:
                        speed = speed / 100.0
                    self._idle_speed = max(0.01, min(0.2, speed * 0.2))
                except ValueError:
                    pass
            elif action == "get_model_info":
                info_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "avatar" / "model_info.json"
                info_path.parent.mkdir(parents=True, exist_ok=True)
                with open(info_path, 'w') as f:
                    json.dump(self._model_info, f, indent=2)
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    
    def _do_speaking_pulse(self):
        """Fallback speaking indicator - disabled for humanoid models."""
        # Don't do weird pulsing on human-shaped avatars
        # Proper lip sync should be handled by AdaptiveAnimator if model supports it
        pass
    
    def _do_nod_fallback(self):
        """Fallback nod animation."""
        if not self._gl_widget:
            return
        # Simple up-down via camera
        import threading
        def animate():
            import time
            original_x = self._gl_widget.rotation_x
            for i in range(10):
                self._gl_widget.rotation_x = original_x + math.sin(i * 0.6) * 15
                self._gl_widget.update()
                time.sleep(0.04)
            self._gl_widget.rotation_x = original_x
            self._gl_widget.update()
        threading.Thread(target=animate, daemon=True).start()
    
    def _do_shake_fallback(self):
        """Fallback shake animation."""
        if not self._gl_widget:
            return
        import threading
        def animate():
            import time
            original_y = self._gl_widget.rotation_y
            for i in range(12):
                self._gl_widget.rotation_y = original_y + math.sin(i * 1.0) * 20 * (1 - i/12)
                self._gl_widget.update()
                time.sleep(0.04)
            self._gl_widget.rotation_y = original_y
            self._gl_widget.update()
        threading.Thread(target=animate, daemon=True).start()
    
    def _close(self):
        self._command_timer.stop()
        self._idle_timer.stop()
        if hasattr(self, '_key_check_timer'):
            self._key_check_timer.stop()
        if self._animator:
            self._animator.stop()
        # Close the floating drag bar
        if hasattr(self, '_drag_bar') and self._drag_bar:
            self._drag_bar.close()
        # Close bone hit regions
        if hasattr(self, '_bone_hit_manager') and self._bone_hit_manager:
            self._bone_hit_manager.clear()
        # Close resize handle
        if hasattr(self, '_resize_handle') and self._resize_handle:
            self._resize_handle.hide()
        self.hide()
        self.closed.emit()
    
    def closeEvent(self, event):
        self._command_timer.stop()
        self._idle_timer.stop()
        if hasattr(self, '_key_check_timer'):
            self._key_check_timer.stop()
        if self._animator:
            self._animator.stop()
        # Close the floating drag bar
        if hasattr(self, '_drag_bar') and self._drag_bar:
            self._drag_bar.close()
        # Close bone hit regions
        if hasattr(self, '_bone_hit_manager') and self._bone_hit_manager:
            self._bone_hit_manager.clear()
        super().closeEvent(event)


class AvatarPreviewWidget(QFrame):
    """2D image preview with parallax 2.5D effect support."""
    
    expression_changed = pyqtSignal(str)  # Signal when expression changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.original_pixmap = None
        self._svg_mode = False
        self._current_svg = None
        
        # Parallax 2.5D settings
        self._parallax_enabled = False
        self._parallax_layers = []  # List of (pixmap, depth) tuples, depth 0-1 (0=back, 1=front)
        self._parallax_offset_x = 0.0  # -1 to 1, controlled by mouse or AI
        self._parallax_offset_y = 0.0
        self._parallax_intensity = 15  # Max pixel offset for parallax
        self._mouse_tracking_parallax = False
        
        self.setMinimumSize(250, 250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)  # Enable mouse tracking for parallax
        # No border in stylesheet - we draw it in paintEvent to wrap tight around image
        self.setStyleSheet("""
            QFrame {
                background: #1e1e2e;
            }
        """)
    
    def set_parallax_enabled(self, enabled: bool):
        """Enable/disable parallax effect."""
        self._parallax_enabled = enabled
        self.update()
    
    def set_parallax_layers(self, layers: list):
        """Set parallax layers. Each layer is (QPixmap, depth 0-1)."""
        self._parallax_layers = layers
        self.update()
    
    def set_parallax_offset(self, x: float, y: float):
        """Set parallax offset (-1 to 1). Called by AI or mouse tracking."""
        self._parallax_offset_x = max(-1, min(1, x))
        self._parallax_offset_y = max(-1, min(1, y))
        self.update()
    
    def set_mouse_tracking_parallax(self, enabled: bool):
        """Enable mouse-based parallax (eyes follow cursor)."""
        self._mouse_tracking_parallax = enabled
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Track mouse for parallax effect."""
        if self._mouse_tracking_parallax and self._parallax_enabled:
            # Calculate offset based on mouse position relative to center
            center_x = self.width() / 2
            center_y = self.height() / 2
            offset_x = (event.x() - center_x) / center_x
            offset_y = (event.y() - center_y) / center_y
            self.set_parallax_offset(offset_x, offset_y)
        super().mouseMoveEvent(event)
        
    def set_avatar(self, pixmap: QPixmap):
        """Set avatar image."""
        self.original_pixmap = pixmap
        self._svg_mode = False
        self._update_display()
    
    def set_svg_sprite(self, svg_data: str):
        """Set avatar from SVG data."""
        self._svg_mode = True
        self._current_svg = svg_data
        
        # Use minimum size of 200 if widget not yet sized
        size = min(self.width(), self.height()) - 20
        if size <= 0:
            size = 200
        
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt_transparent if isinstance(Qt_transparent, QColor) else QColor(0, 0, 0, 0))
        
        if HAS_SVG and QSvgRenderer is not None:
            # Convert SVG to pixmap using Qt SVG renderer
            renderer = QSvgRenderer(QByteArray(svg_data.encode('utf-8')))
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
        else:
            # Fallback: Create a simple drawn avatar without SVG support
            # Parse basic colors from SVG if possible
            primary = "#89b4fa"  # Default blue
            accent = "#cdd6f4"   # Default light
            secondary = "#f5c2e7"  # Default pink
            
            # Try to extract colors from SVG
            import re
            for pattern, var in [
                (r'fill="\{primary_color\}"', 'primary'),
                (r'fill="\{accent_color\}"', 'accent'),
                (r'fill="\{secondary_color\}"', 'secondary'),
            ]:
                pass  # Colors are placeholders in template
            
            # If SVG has actual color values
            primary_match = re.search(r'fill="(#[0-9a-fA-F]{6})"', svg_data)
            if primary_match:
                primary = primary_match.group(1)
            
            # Draw fallback avatar
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing, True)
            
            # Background circle
            painter.setPen(Qt_NoPen)
            painter.setBrush(QColor(primary))
            painter.setOpacity(0.2)
            painter.drawEllipse(size//2 - size*4//10, size//2 - size*4//10, size*8//10, size*8//10)
            
            # Main head circle
            painter.setOpacity(1.0)
            painter.setBrush(QColor(primary))
            painter.drawEllipse(size//2 - size*3//10, size//2 - size*3//10, size*6//10, size*6//10)
            
            # Eyes
            painter.setBrush(QColor(accent))
            eye_y = size//2 - size//20
            eye_size = size//12
            painter.drawEllipse(size//2 - size//6 - eye_size//2, eye_y - eye_size//2, eye_size, int(eye_size * 1.3))
            painter.drawEllipse(size//2 + size//6 - eye_size//2, eye_y - eye_size//2, eye_size, int(eye_size * 1.3))
            
            # Pupils
            pupil_size = eye_size // 2
            painter.setBrush(QColor("#1e1e2e"))
            painter.drawEllipse(size//2 - size//6 - pupil_size//2, eye_y, pupil_size, pupil_size)
            painter.drawEllipse(size//2 + size//6 - pupil_size//2, eye_y, pupil_size, pupil_size)
            
            # Mouth (neutral)
            painter.setPen(QPen(QColor(accent), 2))
            painter.setBrush(Qt_NoBrush)
            from PyQt5.QtCore import QRect
            mouth_rect = QRect(size//2 - size//8, size//2 + size//12, size//4, size//10)
            painter.drawArc(mouth_rect, 0, -180 * 16)  # Bottom half of arc
            
            painter.end()
            print("SVG support not available - using QPainter fallback avatar")
        
        self.original_pixmap = pixmap
        self.pixmap = pixmap
        self.update()
        
    def _update_display(self):
        """Scale pixmap to fit."""
        if self.original_pixmap:
            size = min(self.width(), self.height()) - 20
            if size > 0:
                self.pixmap = self.original_pixmap.scaled(
                    size, size, Qt_KeepAspectRatio, Qt_SmoothTransformation
                )
        self.update()
        
    def paintEvent(self, a0):
        """Draw avatar with border wrapped tightly around the image."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        # Parallax mode: render layers with depth-based offsets
        if self._parallax_enabled and self._parallax_layers:
            base_x = (self.width() - self.pixmap.width()) // 2 if self.pixmap else self.width() // 2
            base_y = (self.height() - self.pixmap.height()) // 2 if self.pixmap else self.height() // 2
            
            # Draw subtle background
            painter.setPen(Qt_NoPen)
            painter.setBrush(QColor(30, 30, 46, 80))
            if self.pixmap:
                painter.drawRoundedRect(base_x - 5, base_y - 5, 
                    self.pixmap.width() + 10, self.pixmap.height() + 10, 12, 12)
            
            # Draw each layer with parallax offset based on depth
            for layer_pixmap, depth in self._parallax_layers:
                # Deeper layers (depth closer to 0) move MORE (parallax effect)
                # Front layers (depth closer to 1) move LESS
                parallax_factor = 1.0 - depth  # Invert so background moves more
                offset_x = int(self._parallax_offset_x * self._parallax_intensity * parallax_factor)
                offset_y = int(self._parallax_offset_y * self._parallax_intensity * parallax_factor)
                
                # Scale layer to fit
                size = min(self.width(), self.height()) - 20
                if size > 0:
                    scaled = layer_pixmap.scaled(size, size, Qt_KeepAspectRatio, Qt_SmoothTransformation)
                    lx = (self.width() - scaled.width()) // 2 + offset_x
                    ly = (self.height() - scaled.height()) // 2 + offset_y
                    painter.drawPixmap(lx, ly, scaled)
            
            # Draw border
            if self.pixmap:
                pen = painter.pen()
                pen.setColor(QColor("#45475a"))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.setBrush(Qt_NoBrush)
                painter.drawRoundedRect(base_x - 3, base_y - 3, 
                    self.pixmap.width() + 6, self.pixmap.height() + 6, 10, 10)
        elif self.pixmap:
            x = (self.width() - self.pixmap.width()) // 2
            y = (self.height() - self.pixmap.height()) // 2
            
            # Draw a subtle background/glow behind the avatar
            painter.setPen(Qt_NoPen)
            painter.setBrush(QColor(30, 30, 46, 80))
            painter.drawRoundedRect(x - 5, y - 5, self.pixmap.width() + 10, self.pixmap.height() + 10, 12, 12)
            
            # Draw the avatar
            painter.drawPixmap(x, y, self.pixmap)
            
            # Draw border TIGHT around the avatar image
            pen = painter.pen()
            pen.setColor(QColor("#45475a"))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt_NoBrush)
            painter.drawRoundedRect(x - 3, y - 3, self.pixmap.width() + 6, self.pixmap.height() + 6, 10, 10)
        else:
            painter.setPen(QColor("#6c7086"))
            painter.drawText(self.rect(), Qt_AlignCenter, 
                           "No avatar loaded\n\nClick 'Load Avatar' to select")
            
    def resizeEvent(self, a0):
        """Update on resize."""
        # Re-render SVG at new size if in SVG mode
        if self._svg_mode and self._current_svg:
            self.set_svg_sprite(self._current_svg)
        else:
            self._update_display()
        super().resizeEvent(a0)


def create_avatar_subtab(parent):
    """Create the avatar display sub-tab."""
    widget = QWidget()
    main_layout = QHBoxLayout()  # Changed to horizontal for side panel
    main_layout.setSpacing(8)
    
    # Check if avatar module is enabled
    avatar_module_enabled = _is_avatar_module_enabled()
    
    # Left side - Preview and basic controls
    left_panel = QVBoxLayout()
    
    # Header with format help
    header = QLabel("Avatar Display")
    header.setObjectName("header")
    left_panel.addWidget(header)
    
    # Format help section - collapsible info box
    format_help = QLabel(
        "2D: PNG, JPG, GIF | 3D: GLB (best), GLTF, OBJ, FBX\n"
        "GLB = single file with textures | GLTF = folder with separate files\n"
        "Free avatars: VRoid Hub, Mixamo, Ready Player Me, Sketchfab"
    )
    format_help.setStyleSheet(
        "color: #89b4fa; font-size: 10px; padding: 4px 8px; "
        "background: #1e1e2e; border: 1px solid #45475a; border-radius: 4px;"
    )
    format_help.setWordWrap(True)
    format_help.setToolTip(
        "GLB vs GLTF:\\n"
        "- GLB: Single binary file, includes all textures. RECOMMENDED - just drag and drop!\\n"
        "- GLTF: JSON file + separate texture images. Need to keep all files together.\\n\\n"
        "Best sources for free avatars:\\n"
        "- VRoid Hub (anime style, VRM format - convert to GLB)\\n"
        "- Ready Player Me (realistic, free GLB export)\\n"
        "- Mixamo (animated characters)\\n"
        "- Sketchfab (search for 'avatar' with downloadable filter)"
    )
    left_panel.addWidget(format_help)
    
    # Get avatar controller
    avatar = get_avatar()
    
    # Module status message (shown when module is off)
    parent.module_status_label = QLabel(
        "Avatar module is disabled.\nGo to the Modules tab to enable it."
    )
    parent.module_status_label.setStyleSheet(
        "color: #fab387; font-size: 12px; padding: 10px; "
        "background: #313244; border-radius: 8px;"
    )
    parent.module_status_label.setWordWrap(True)
    parent.module_status_label.setVisible(not avatar_module_enabled)
    left_panel.addWidget(parent.module_status_label)
    
    # Top controls
    top_row = QHBoxLayout()
    
    parent.avatar_enabled_checkbox = QCheckBox("Enable Avatar")
    parent.avatar_enabled_checkbox.setChecked(avatar.is_enabled)
    parent.avatar_enabled_checkbox.toggled.connect(lambda c: _toggle_avatar(parent, c))
    top_row.addWidget(parent.avatar_enabled_checkbox)
    
    parent.show_overlay_btn = QPushButton("Run")
    parent.show_overlay_btn.setCheckable(True)
    parent.show_overlay_btn.setStyleSheet("""
        QPushButton {
            background: #a6e3a1;
            color: #1e1e2e;
            font-weight: bold;
            padding: 6px 12px;
            border-radius: 6px;
        }
        QPushButton:hover {
            background: #94e2d5;
        }
        QPushButton:checked {
            background: #f38ba8;
        }
        QPushButton:checked:hover {
            background: #eba0ac;
        }
    """)
    parent.show_overlay_btn.clicked.connect(lambda: _toggle_overlay(parent))
    top_row.addWidget(parent.show_overlay_btn)
    
    top_row.addStretch()
    left_panel.addLayout(top_row)
    
    # Settings row: Auto-load on startup + Auto-run + Manual resize toggle
    settings_row = QHBoxLayout()
    
    parent.avatar_auto_load_checkbox = QCheckBox("Auto-load on startup")
    parent.avatar_auto_load_checkbox.setToolTip("Automatically load the last used avatar when Enigma AI Engine starts")
    parent.avatar_auto_load_checkbox.toggled.connect(lambda c: _toggle_auto_load(parent, c))
    settings_row.addWidget(parent.avatar_auto_load_checkbox)
    
    parent.avatar_auto_run_checkbox = QCheckBox("Auto-show pop-out")
    parent.avatar_auto_run_checkbox.setToolTip("Automatically show the desktop pop-out overlay when Enigma AI Engine starts")
    parent.avatar_auto_run_checkbox.toggled.connect(lambda c: _toggle_auto_run(parent, c))
    settings_row.addWidget(parent.avatar_auto_run_checkbox)
    
    # Reset Position button - moves avatar back to visible area
    parent.reset_position_btn = QPushButton("Reset Position")
    parent.reset_position_btn.setToolTip("Reset avatar position to center of screen (use if avatar went off-screen)")
    parent.reset_position_btn.setStyleSheet("""
        QPushButton {
            background: #f38ba8;
            color: #1e1e2e;
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background: #eba0ac;
        }
    """)
    parent.reset_position_btn.clicked.connect(lambda: _reset_avatar_position(parent))
    settings_row.addWidget(parent.reset_position_btn)
    
    settings_row.addStretch()
    left_panel.addLayout(settings_row)
    
    # 3D rendering toggle (only if libraries available)
    if HAS_OPENGL and HAS_TRIMESH:
        render_row = QHBoxLayout()
        parent.use_3d_render_checkbox = QCheckBox("Enable 3D Rendering")
        parent.use_3d_render_checkbox.setToolTip(
            "Turn ON to view 3D models (GLB, GLTF, OBJ, FBX).\n"
            "Required for 3D avatars to display properly."
        )
        parent.use_3d_render_checkbox.setStyleSheet("""
            QCheckBox {
                color: #89b4fa;
                font-weight: bold;
                padding: 4px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #89b4fa;
                border-radius: 3px;
                background: transparent;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #a6e3a1;
                border-radius: 3px;
                background: #a6e3a1;
            }
        """)
        # Don't set checked yet - widgets don't exist
        parent.use_3d_render_checkbox.toggled.connect(lambda c: _toggle_3d_render(parent, c))
        render_row.addWidget(parent.use_3d_render_checkbox)
        render_row.addStretch()
        left_panel.addLayout(render_row)
    else:
        # Show message that 3D libraries are missing
        parent.use_3d_render_checkbox = None
        missing_3d_label = QLabel("3D avatars need: pip install trimesh PyOpenGL")
        missing_3d_label.setStyleSheet(
            "color: #fab387; font-size: 10px; padding: 2px;"
        )
        left_panel.addWidget(missing_3d_label)
    
    # Preview widgets (stacked - 2D and 3D)
    parent.avatar_preview_2d = AvatarPreviewWidget()
    left_panel.addWidget(parent.avatar_preview_2d, stretch=1)
    
    if HAS_OPENGL and HAS_TRIMESH:
        parent.avatar_preview_3d = OpenGL3DWidget()
        parent.avatar_preview_3d.setVisible(False)  # Start hidden, will show when checkbox is set
        left_panel.addWidget(parent.avatar_preview_3d, stretch=1)
    else:
        parent.avatar_preview_3d = None
    
    # 3D viewer controls (Sketchfab-style)
    viewer_controls = QHBoxLayout()
    viewer_controls.addStretch()
    
    parent.reset_view_btn = QPushButton("Reset View")
    parent.reset_view_btn.setToolTip("Reset camera (or double-click)")
    parent.reset_view_btn.clicked.connect(lambda: _reset_view(parent))
    parent.reset_view_btn.setVisible(False)
    parent.reset_view_btn.setStyleSheet("""
        QPushButton {
            background: #2d2d3d;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 4px 8px;
        }
        QPushButton:hover {
            background: #45475a;
        }
    """)
    viewer_controls.addWidget(parent.reset_view_btn)
    
    viewer_controls.addStretch()
    left_panel.addLayout(viewer_controls)
    
    # Avatar selector
    select_row = QHBoxLayout()
    select_row.addWidget(QLabel("Avatar:"))
    parent.avatar_combo = NoScrollComboBox()
    parent.avatar_combo.setToolTip("Select an avatar from your collection")
    parent.avatar_combo.setMinimumWidth(200)
    parent.avatar_combo.currentIndexChanged.connect(lambda: _on_avatar_selected(parent))
    select_row.addWidget(parent.avatar_combo, stretch=1)
    
    btn_refresh = QPushButton("Refresh")
    btn_refresh.setMinimumWidth(80)
    btn_refresh.setToolTip("Refresh selected avatar and list")
    btn_refresh.clicked.connect(lambda: _refresh_avatar(parent))
    select_row.addWidget(btn_refresh)
    left_panel.addLayout(select_row)
    
    # Load and Apply buttons
    btn_row2 = QHBoxLayout()
    parent.load_btn = QPushButton("Load Avatar")
    parent.load_btn.setToolTip(
        "Load an avatar file from your computer.\\n\\n"
        "2D Images: PNG, JPG, GIF, BMP, WEBP\\n"
        "3D Models: GLB, GLTF, OBJ, FBX, DAE\\n\\n"
        "Free 3D avatars: VRoid Hub, Mixamo, Sketchfab"
    )
    parent.load_btn.clicked.connect(lambda: _load_avatar_file(parent))
    btn_row2.addWidget(parent.load_btn)
    
    parent.apply_btn = QPushButton("Apply Avatar")
    parent.apply_btn.setToolTip("Make the selected avatar your active avatar")
    parent.apply_btn.clicked.connect(lambda: _apply_avatar(parent))
    parent.apply_btn.setStyleSheet("background: #89b4fa; color: #1e1e2e; font-weight: bold;")
    btn_row2.addWidget(parent.apply_btn)
    left_panel.addLayout(btn_row2)
    
    # Status
    parent.avatar_status = QLabel("No avatar loaded")
    parent.avatar_status.setStyleSheet("color: #6c7086; font-style: italic;")
    left_panel.addWidget(parent.avatar_status)
    
    main_layout.addLayout(left_panel, stretch=2)
    
    # Right side - Customization Controls
    right_panel = QVBoxLayout()
    
    # === Quick Actions ===
    actions_group = QGroupBox("Quick Actions")
    actions_layout = QVBoxLayout()
    
# Auto-design from personality (only for built-in sprites)
    parent.auto_design_btn = QPushButton("AI Auto-Design (Sprites)")
    parent.auto_design_btn.setToolTip("Randomly generate a 2D sprite style. For 3D models, use the 3D tab instead.")
    parent.auto_design_btn.clicked.connect(lambda: _auto_design_avatar(parent))
    actions_layout.addWidget(parent.auto_design_btn)

    # Export sprite button (only for built-in sprites)
    parent.export_btn = QPushButton("Export Current Sprite")
    parent.export_btn.setToolTip("Export the built-in 2D sprite to SVG/PNG file. Only works with sprites, not 3D models.")
    parent.export_btn.clicked.connect(lambda: _export_sprite(parent))
    actions_layout.addWidget(parent.export_btn)
    
    actions_group.setLayout(actions_layout)
    right_panel.addWidget(actions_group)
    
    # === Parallax 2.5D Effect (for 2D images) ===
    parallax_group = QGroupBox("2.5D Parallax Effect (NEW!)")
    parallax_group.setStyleSheet("""
        QGroupBox {
            border: 2px solid #f5c2e7;
            border-radius: 8px;
            margin-top: 8px;
            padding-top: 8px;
            font-weight: bold;
        }
        QGroupBox::title {
            color: #f5c2e7;
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
    """)
    parallax_group.setToolTip(
        "Make 2D images look 3D by splitting into layers that move at different speeds.\\n"
        "Works with any PNG/JPG image - AI can auto-generate layers!"
    )
    parallax_layout = QVBoxLayout()
    parallax_layout.setSpacing(6)
    
    # Enable parallax
    parent.parallax_check = QCheckBox("Enable Parallax Effect")
    parent.parallax_check.setChecked(False)
    parent.parallax_check.setToolTip("Enable 2.5D depth effect for 2D avatars")
    parent.parallax_check.stateChanged.connect(
        lambda state: _toggle_parallax(parent, state == 2)
    )
    parallax_layout.addWidget(parent.parallax_check)
    
    # Mouse tracking for parallax
    parent.parallax_mouse_check = QCheckBox("Follow Cursor")
    parent.parallax_mouse_check.setChecked(True)
    parent.parallax_mouse_check.setToolTip("Layers react to mouse position (like eyes following you)")
    parent.parallax_mouse_check.setEnabled(False)  # Enabled when parallax is on
    parent.parallax_mouse_check.stateChanged.connect(
        lambda state: _toggle_parallax_mouse(parent, state == 2)
    )
    parallax_layout.addWidget(parent.parallax_mouse_check)
    
    # Intensity slider
    intensity_row = QHBoxLayout()
    intensity_row.addWidget(QLabel("Depth:"))
    parent.parallax_intensity_slider = QSlider(Qt.Horizontal)
    parent.parallax_intensity_slider.setRange(5, 40)
    parent.parallax_intensity_slider.setValue(15)
    parent.parallax_intensity_slider.setToolTip("How much the layers move (5=subtle, 40=dramatic)")
    parent.parallax_intensity_slider.setEnabled(False)
    parent.parallax_intensity_slider.valueChanged.connect(
        lambda v: _set_parallax_intensity(parent, v)
    )
    intensity_row.addWidget(parent.parallax_intensity_slider)
    parent.parallax_intensity_label = QLabel("15")
    intensity_row.addWidget(parent.parallax_intensity_label)
    parallax_layout.addLayout(intensity_row)
    
    # AI Generate Layers button
    parent.ai_generate_layers_btn = QPushButton("AI: Generate Layers")
    parent.ai_generate_layers_btn.setToolTip(
        "Use AI to automatically split your 2D image into depth layers.\\n"
        "Creates: background, body, face, eyes/accessories"
    )
    parent.ai_generate_layers_btn.setStyleSheet(
        "background: #f5c2e7; color: #1e1e2e; font-weight: bold;"
    )
    parent.ai_generate_layers_btn.clicked.connect(lambda: _ai_generate_parallax_layers(parent))
    parallax_layout.addWidget(parent.ai_generate_layers_btn)
    
    # Load custom layers button
    parent.load_layers_btn = QPushButton("Load Layer Files")
    parent.load_layers_btn.setToolTip(
        "Load your own layer files (e.g., avatar_bg.png, avatar_body.png, avatar_face.png).\\n"
        "Name files with _bg, _body, _face, _eyes suffixes for auto-depth ordering."
    )
    parent.load_layers_btn.clicked.connect(lambda: _load_parallax_layers(parent))
    parallax_layout.addWidget(parent.load_layers_btn)
    
    parallax_group.setLayout(parallax_layout)
    right_panel.addWidget(parallax_group)
    
    # === Behavior Controls ===
    behavior_group = QGroupBox("Behavior Controls")
    behavior_layout = QVBoxLayout()
    behavior_layout.setSpacing(8)
    
    # Auto expressions from AI text
    parent.auto_avatar_check = QCheckBox("Auto Expressions (from AI text)")
    parent.auto_avatar_check.setChecked(True)
    parent.auto_avatar_check.setToolTip("Avatar changes expression based on AI responses")
    parent.auto_avatar_check.stateChanged.connect(
        lambda state: setattr(parent, 'auto_avatar_enabled', state == 2)
    )
    parent.auto_avatar_enabled = True
    behavior_layout.addWidget(parent.auto_avatar_check)
    
    # Eye tracking - follow mouse cursor
    parent.eye_tracking_check = QCheckBox("Eye Tracking (look at cursor)")
    parent.eye_tracking_check.setChecked(False)
    parent.eye_tracking_check.setToolTip("Avatar eyes/head follows your mouse cursor")
    parent.eye_tracking_check.stateChanged.connect(
        lambda state: _toggle_eye_tracking(parent, state == 2)
    )
    parent.eye_tracking_enabled = False
    behavior_layout.addWidget(parent.eye_tracking_check)
    
    # Autonomous mode (react to screen)
    parent.avatar_autonomous_check = QCheckBox("Autonomous Mode (react to screen)")
    parent.avatar_autonomous_check.setChecked(False)
    parent.avatar_autonomous_check.setToolTip("Avatar watches screen and reacts to content")
    parent.avatar_autonomous_check.stateChanged.connect(
        lambda state: _toggle_avatar_autonomous(parent, state)
    )
    behavior_layout.addWidget(parent.avatar_autonomous_check)
    
    # Activity level slider
    activity_row = QHBoxLayout()
    activity_row.addWidget(QLabel("Activity:"))
    parent.avatar_activity_slider = QSlider(Qt.Horizontal)
    parent.avatar_activity_slider.setRange(1, 10)
    parent.avatar_activity_slider.setValue(5)
    parent.avatar_activity_slider.setToolTip("How active the avatar is (1=calm, 10=energetic)")
    activity_row.addWidget(parent.avatar_activity_slider)
    parent.activity_value_label = QLabel("5")
    parent.avatar_activity_slider.valueChanged.connect(
        lambda v: parent.activity_value_label.setText(str(v))
    )
    activity_row.addWidget(parent.activity_value_label)
    behavior_layout.addLayout(activity_row)
    
    # Behavior status
    parent.avatar_behavior_status = QLabel("Mode: Manual")
    parent.avatar_behavior_status.setStyleSheet("color: #bac2de; font-style: italic;")
    behavior_layout.addWidget(parent.avatar_behavior_status)
    
    behavior_group.setLayout(behavior_layout)
    right_panel.addWidget(behavior_group)

    # === Avatar Gallery ===
    gallery_group = QGroupBox("Avatar Gallery")
    gallery_layout = QVBoxLayout()
    
    # Browse avatars button
    parent.browse_avatars_btn = QPushButton("Browse Avatars...")
    parent.browse_avatars_btn.setToolTip("Browse and select from installed avatars")
    parent.browse_avatars_btn.clicked.connect(lambda: _browse_avatars(parent))
    gallery_layout.addWidget(parent.browse_avatars_btn)
    
    # Import avatar button (single file)
    parent.import_avatar_btn = QPushButton("Import Avatar...")
    parent.import_avatar_btn.setToolTip("Import a new avatar from files or .forgeavatar bundle")
    parent.import_avatar_btn.clicked.connect(lambda: _import_avatar(parent))
    gallery_layout.addWidget(parent.import_avatar_btn)
    
    # Import multiple files button
    parent.import_multiple_btn = QPushButton("Import Multiple Files...")
    parent.import_multiple_btn.setToolTip("Select and import multiple avatar files at once (images or 3D models)")
    parent.import_multiple_btn.clicked.connect(lambda: _import_multiple_avatars(parent))
    gallery_layout.addWidget(parent.import_multiple_btn)
    
    # Import from Downloads button
    parent.import_downloads_btn = QPushButton("Import from Downloads")
    parent.import_downloads_btn.setToolTip("Quick import avatars from your Downloads folder")
    parent.import_downloads_btn.setStyleSheet("background: #89b4fa; color: #1e1e2e; font-weight: bold;")
    parent.import_downloads_btn.clicked.connect(lambda: _import_from_downloads(parent))
    gallery_layout.addWidget(parent.import_downloads_btn)
    
    # Import & Extract ZIP button  
    parent.import_zip_btn = QPushButton("Import ZIP/Archive...")
    parent.import_zip_btn.setToolTip("Import and extract avatar ZIP files (e.g., downloaded glTF models)")
    parent.import_zip_btn.setStyleSheet("background: #f5c2e7; color: #1e1e2e; font-weight: bold;")
    parent.import_zip_btn.clicked.connect(lambda: _import_zip_archive(parent))
    gallery_layout.addWidget(parent.import_zip_btn)
    
    # Generate samples button
    parent.generate_samples_btn = QPushButton("Generate Samples")
    parent.generate_samples_btn.setToolTip("Generate sample avatars to get started")
    parent.generate_samples_btn.clicked.connect(lambda: _generate_sample_avatars(parent))
    gallery_layout.addWidget(parent.generate_samples_btn)
    
    gallery_group.setLayout(gallery_layout)
    right_panel.addWidget(gallery_group)
    
    # === 3D Viewer Settings (Sketchfab-style) ===
    if HAS_OPENGL and HAS_TRIMESH:
        viewer_group = QGroupBox("3D Viewer Settings")
        viewer_layout = QVBoxLayout()
        
        # Display options row
        display_row = QHBoxLayout()
        parent.wireframe_checkbox = QCheckBox("Wireframe")
        parent.wireframe_checkbox.toggled.connect(lambda c: _set_wireframe(parent, c))
        display_row.addWidget(parent.wireframe_checkbox)
        
        parent.show_grid_checkbox = QCheckBox("Grid")
        parent.show_grid_checkbox.setChecked(True)
        parent.show_grid_checkbox.toggled.connect(lambda c: _set_show_grid(parent, c))
        display_row.addWidget(parent.show_grid_checkbox)
        display_row.addStretch()
        viewer_layout.addLayout(display_row)
        
        # === Facing Direction Indicator ===
        facing_group = QHBoxLayout()
        facing_label = QLabel("Facing:")
        facing_label.setStyleSheet("font-weight: bold;")
        facing_group.addWidget(facing_label)
        
        parent.facing_direction_label = QLabel("Front (0 deg)")
        parent.facing_direction_label.setStyleSheet("""
            background: #313244;
            padding: 4px 12px;
            border-radius: 4px;
            color: #89b4fa;
            font-weight: bold;
        """)
        parent.facing_direction_label.setToolTip("Current facing direction (Yaw rotation)")
        facing_group.addWidget(parent.facing_direction_label)
        facing_group.addStretch()
        viewer_layout.addLayout(facing_group)
        
        # === Orientation Controls ===
        orient_label = QLabel("Model Orientation (fix sideways models):")
        orient_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        viewer_layout.addWidget(orient_label)
        
        # X rotation (pitch)
        pitch_row = QHBoxLayout()
        pitch_row.addWidget(QLabel("Pitch (X):"))
        parent.pitch_slider = QSlider(Qt.Horizontal if hasattr(Qt, 'Horizontal') else 0x01)
        parent.pitch_slider.setRange(-180, 180)
        parent.pitch_slider.setValue(0)
        parent.pitch_slider.valueChanged.connect(lambda v: _set_model_orientation(parent, 'pitch', v))
        pitch_row.addWidget(parent.pitch_slider)
        parent.pitch_label = QLabel("0")
        parent.pitch_label.setFixedWidth(30)
        pitch_row.addWidget(parent.pitch_label)
        viewer_layout.addLayout(pitch_row)
        
        # Y rotation (yaw)
        yaw_row = QHBoxLayout()
        yaw_row.addWidget(QLabel("Yaw (Y):"))
        parent.yaw_slider = QSlider(Qt.Horizontal if hasattr(Qt, 'Horizontal') else 0x01)
        parent.yaw_slider.setRange(-180, 180)
        parent.yaw_slider.setValue(0)
        parent.yaw_slider.valueChanged.connect(lambda v: _set_model_orientation(parent, 'yaw', v))
        yaw_row.addWidget(parent.yaw_slider)
        parent.yaw_label = QLabel("0")
        parent.yaw_label.setFixedWidth(30)
        yaw_row.addWidget(parent.yaw_label)
        viewer_layout.addLayout(yaw_row)
        
        # Z rotation (roll)
        roll_row = QHBoxLayout()
        roll_row.addWidget(QLabel("Roll (Z):"))
        parent.roll_slider = QSlider(Qt.Horizontal if hasattr(Qt, 'Horizontal') else 0x01)
        parent.roll_slider.setRange(-180, 180)
        parent.roll_slider.setValue(0)
        parent.roll_slider.valueChanged.connect(lambda v: _set_model_orientation(parent, 'roll', v))
        roll_row.addWidget(parent.roll_slider)
        parent.roll_label = QLabel("0")
        parent.roll_label.setFixedWidth(30)
        roll_row.addWidget(parent.roll_label)
        viewer_layout.addLayout(roll_row)
        
        # Simple view buttons row (combined)
        view_row = QHBoxLayout()
        view_front = QPushButton("Front")
        view_front.clicked.connect(lambda: _set_camera_view(parent, 0, 0))
        view_row.addWidget(view_front)
        
        view_back = QPushButton("Back")
        view_back.clicked.connect(lambda: _set_camera_view(parent, 0, 180))
        view_row.addWidget(view_back)
        
        view_left = QPushButton("Left")
        view_left.clicked.connect(lambda: _set_camera_view(parent, 0, -90))
        view_row.addWidget(view_left)
        
        view_right = QPushButton("Right")
        view_right.clicked.connect(lambda: _set_camera_view(parent, 0, 90))
        view_row.addWidget(view_right)
        viewer_layout.addLayout(view_row)
        
        # Quick orientation buttons row (simplified)
        quick_orient_row = QHBoxLayout()
        
        reset_orient_btn = QPushButton("Reset")
        reset_orient_btn.setToolTip("Reset orientation to 0")
        reset_orient_btn.clicked.connect(lambda: _reset_orientation(parent))
        quick_orient_row.addWidget(reset_orient_btn)
        
        auto_orient_btn = QPushButton("Auto-Orient")
        auto_orient_btn.setToolTip("Auto-detect and fix model orientation")
        auto_orient_btn.clicked.connect(lambda: _auto_orient_model(parent))
        quick_orient_row.addWidget(auto_orient_btn)
        
        viewer_layout.addLayout(quick_orient_row)
        
        # Hint about controls on popup
        rotate_hint = QLabel("Tip: Popup - Right-click to Enable Resize/Rotate, then Shift+drag")
        rotate_hint.setStyleSheet("color: #6c7086; font-size: 10px;")
        viewer_layout.addWidget(rotate_hint)
        
        # Save orientation button
        parent.save_orientation_btn = QPushButton("Save Orientation for This Model")
        parent.save_orientation_btn.clicked.connect(lambda: _save_model_orientation(parent))
        parent.save_orientation_btn.setToolTip("Save orientation so it loads automatically next time")
        parent.save_orientation_btn.setStyleSheet("background: #a6e3a1; color: #1e1e2e; font-weight: bold;")
        viewer_layout.addWidget(parent.save_orientation_btn)
        
        # === Reset Options (combined into viewer group) ===
        reset_label = QLabel("Reset Options:")
        reset_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        viewer_layout.addWidget(reset_label)
        
        # Reset buttons row
        reset_row = QHBoxLayout()
        
        parent.reset_preview_btn = QPushButton("Reset Preview")
        parent.reset_preview_btn.setToolTip("Reset 3D preview camera and settings")
        parent.reset_preview_btn.clicked.connect(lambda: _reset_preview(parent))
        reset_row.addWidget(parent.reset_preview_btn)
        
        parent.reset_overlay_btn = QPushButton("Reset Overlay")
        parent.reset_overlay_btn.setToolTip("Reset desktop avatar position and size")
        parent.reset_overlay_btn.clicked.connect(lambda: _reset_overlay(parent))
        reset_row.addWidget(parent.reset_overlay_btn)
        
        parent.reset_all_btn = QPushButton("Reset All")
        parent.reset_all_btn.setToolTip("Reset all avatar settings to default")
        parent.reset_all_btn.clicked.connect(lambda: _reset_all_avatar(parent))
        parent.reset_all_btn.setStyleSheet("background: #f38ba8; color: #1e1e2e;")
        reset_row.addWidget(parent.reset_all_btn)
        
        viewer_layout.addLayout(reset_row)
        
        viewer_group.setLayout(viewer_layout)
        right_panel.addWidget(viewer_group)
    
    # AI Activity Log - shows what the AI is doing with the avatar
    activity_group = QGroupBox("AI Activity")
    activity_layout = QVBoxLayout()
    
    parent.avatar_activity_log = QTextEdit()
    parent.avatar_activity_log.setReadOnly(True)
    parent.avatar_activity_log.setMaximumHeight(100)
    parent.avatar_activity_log.setStyleSheet("""
        QTextEdit {
            background: #1e1e2e;
            border: 1px solid #45475a;
            border-radius: 4px;
            font-family: monospace;
            font-size: 11px;
            color: #a6e3a1;
        }
    """)
    parent.avatar_activity_log.setPlaceholderText("AI avatar commands will appear here...")
    activity_layout.addWidget(parent.avatar_activity_log)
    
    clear_log_btn = QPushButton("Clear Log")
    clear_log_btn.clicked.connect(lambda: parent.avatar_activity_log.clear())
    activity_layout.addWidget(clear_log_btn)
    
    activity_group.setLayout(activity_layout)
    right_panel.addWidget(activity_group)
    
    right_panel.addStretch()
    
    # Info
    info = QLabel("Desktop avatar: Drag window to move | Right-click for gestures | Double-click to center")
    info.setStyleSheet("color: #6c7086; font-size: 10px;")
    info.setWordWrap(True)
    right_panel.addWidget(info)
    
    main_layout.addLayout(right_panel, stretch=1)
    
    widget.setLayout(main_layout)
    
    # Initialize state
    parent._avatar_controller = avatar
    parent._overlay = None
    parent._overlay_3d = None  # 3D transparent overlay
    parent._current_path = None
    parent._is_3d_model = False
    parent._using_3d_render = HAS_OPENGL and HAS_TRIMESH  # True if 3D available
    parent.avatar_expressions = {}
    parent.current_expression = "neutral"
    parent._current_colors = {
        "primary": "#6366f1",
        "secondary": "#8b5cf6", 
        "accent": "#10b981"
    }
    parent._using_builtin_sprite = False
    parent._avatar_module_enabled = avatar_module_enabled
    parent._avatar_auto_load = False
    parent._avatar_resize_enabled = False  # Default OFF - user must enable via right-click menu
    
    # Create directories
    AVATAR_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    AVATAR_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    AVATAR_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize 3D view state - NOW set the checkbox (widgets exist)
    if HAS_OPENGL and HAS_TRIMESH and parent.use_3d_render_checkbox:
        # This triggers _toggle_3d_render which shows the 3D view
        parent.use_3d_render_checkbox.setChecked(True)
    
    # Load list
    _refresh_list(parent)
    
    # Restore saved avatar settings
    _restore_avatar_settings(parent)
    
    # Set up file watcher timer to auto-refresh when files change
    parent._avatar_file_watcher = QTimer()
    parent._avatar_file_watcher.timeout.connect(lambda: _check_for_new_files(parent))
    parent._avatar_file_watcher.start(3000)  # Check every 3 seconds
    parent._last_file_count = parent.avatar_combo.count()
    
    # Set up AI customization polling
    parent._ai_customize_watcher = QTimer()
    parent._ai_customize_watcher.timeout.connect(lambda: _poll_ai_customizations(parent))
    parent._ai_customize_watcher.start(1000)  # Check every 1 second
    parent._last_ai_customize_time = 0.0
    
    # Set up AI activity log polling
    parent._activity_log_watcher = QTimer()
    parent._activity_log_watcher.timeout.connect(lambda: _update_activity_log(parent))
    parent._activity_log_watcher.start(500)  # Check every 0.5 seconds
    parent._last_activity_log_content = ""
    
    # Register for expression change callbacks from AI chat
    def _on_expression_change(old_expr, new_expr):
        """Called when AI chat triggers an expression change."""
        try:
            # Update expression on overlay if it's showing (from_callback=True prevents loop)
            _set_expression(parent, new_expr, from_callback=True)
        except Exception:
            pass
    
    avatar.on("expression", _on_expression_change)
    parent._expression_callback = _on_expression_change  # Keep reference
    
    # Show default sprite on initialization (unless auto-loading)
    if not getattr(parent, '_avatar_auto_loaded', False):
        parent._using_builtin_sprite = True
        _show_default_preview(parent)
    
    return widget


def _restore_avatar_settings(parent):
    """Restore saved avatar settings from gui_settings.json and avatar_settings.json."""
    try:
        from pathlib import Path

        from ....config import CONFIG
        settings_path = Path(CONFIG["data_dir"]) / "gui_settings.json"
        if not settings_path.exists():
            return
        
        with open(settings_path) as f:
            settings = json.load(f)
        
        # Restore auto-load setting
        parent._avatar_auto_load = settings.get("avatar_auto_load", False)
        if hasattr(parent, 'avatar_auto_load_checkbox'):
            parent.avatar_auto_load_checkbox.setChecked(parent._avatar_auto_load)
        
        # Restore auto-run setting
        parent._avatar_auto_run = settings.get("avatar_auto_run", False)
        if hasattr(parent, 'avatar_auto_run_checkbox'):
            parent.avatar_auto_run_checkbox.setChecked(parent._avatar_auto_run)
        
        # Restore resize enabled setting (default OFF now)
        parent._avatar_resize_enabled = settings.get("avatar_resize_enabled", False)
        
        # Restore saved overlay sizes from avatar persistence (primary) or gui settings (fallback)
        # Avatar persistence is more reliable as it's updated when resizing
        try:
            from ...avatar.persistence import load_avatar_settings
            avatar_settings = load_avatar_settings()
            saved_2d_size = avatar_settings.overlay_size
            saved_3d_size = avatar_settings.overlay_3d_size
        except Exception:
            saved_2d_size = settings.get("avatar_overlay_size", 300)
            saved_3d_size = settings.get("avatar_overlay_3d_size", 250)
        
        parent._saved_overlay_size = max(100, saved_2d_size if isinstance(saved_2d_size, (int, float)) else 300)
        parent._saved_overlay_3d_size = max(100, saved_3d_size if isinstance(saved_3d_size, (int, float)) else 250)
        
        # Restore last avatar selection (always restore the selection)
        last_avatar_index = settings.get("last_avatar_index", 0)
        last_avatar_path = settings.get("last_avatar_path", "")
        selection_restored = False
        
        # First try to restore by path (more reliable)
        if last_avatar_path:
            for i in range(parent.avatar_combo.count()):
                data = parent.avatar_combo.itemData(i)
                if data and len(data) > 1 and str(data[1]) == last_avatar_path:
                    parent.avatar_combo.blockSignals(True)
                    parent.avatar_combo.setCurrentIndex(i)
                    parent.avatar_combo.blockSignals(False)
                    selection_restored = True
                    break
        
        # Fallback to index if path didn't work
        if not selection_restored and last_avatar_index > 0 and last_avatar_index < parent.avatar_combo.count():
            parent.avatar_combo.blockSignals(True)
            parent.avatar_combo.setCurrentIndex(last_avatar_index)
            parent.avatar_combo.blockSignals(False)
            selection_restored = True
        
        if selection_restored:
            parent.avatar_status.setText("Restored last avatar")
            parent.avatar_status.setStyleSheet("color: #89b4fa;")
            # Store the path for later
            data = parent.avatar_combo.currentData()
            if data and len(data) > 1:
                parent._current_path = data[1]
            
            # ALWAYS load the avatar preview when tab opens (not just when auto-load desktop is on)
            # This shows the selected avatar in the preview pane
            QTimer.singleShot(300, lambda: _load_avatar_preview(parent))
        
        # Auto-load avatar TO DESKTOP if setting is enabled
        if parent._avatar_auto_load and selection_restored:
            parent._avatar_auto_loaded = True
            # Apply the avatar after a brief delay (let UI settle)
            QTimer.singleShot(500, lambda: _apply_avatar(parent))
            parent.avatar_status.setText("Auto-loading avatar...")
            parent.avatar_status.setStyleSheet("color: #f9e2af;")
            
            # Auto-RUN the overlay if that setting is also enabled
            if getattr(parent, '_avatar_auto_run', False):
                def auto_run_overlay():
                    # Check if we have a valid avatar loaded
                    pixmap = parent.avatar_preview_2d.original_pixmap if hasattr(parent.avatar_preview_2d, 'original_pixmap') else None
                    is_3d = getattr(parent, '_is_3d_model', False) and getattr(parent, '_using_3d_render', False)
                    has_path = bool(getattr(parent, '_current_path', None))
                    
                    print(f"[Avatar Auto-Run] Checking: pixmap={pixmap is not None}, is_3d={is_3d}, has_path={has_path}, builtin={getattr(parent, '_using_builtin_sprite', True)}")
                    
                    if is_3d and has_path:
                        # 3D model ready
                        if not parent.show_overlay_btn.isChecked():
                            parent.show_overlay_btn.setChecked(True)
                            _toggle_overlay(parent)
                            parent.avatar_status.setText("3D Avatar auto-started on desktop!")
                            parent.avatar_status.setStyleSheet("color: #a6e3a1;")
                    elif pixmap and not getattr(parent, '_using_builtin_sprite', True):
                        # 2D avatar ready
                        if not parent.show_overlay_btn.isChecked():
                            parent.show_overlay_btn.setChecked(True)
                            _toggle_overlay(parent)
                            parent.avatar_status.setText("Avatar auto-started on desktop!")
                            parent.avatar_status.setStyleSheet("color: #a6e3a1;")
                    else:
                        # Avatar not ready yet, try again after a delay
                        parent.avatar_status.setText("Waiting for avatar to load...")
                        parent.avatar_status.setStyleSheet("color: #f9e2af;")
                        QTimer.singleShot(1000, auto_run_overlay)  # Retry
                
                # Run after apply completes - give extra time for loading
                QTimer.singleShot(1500, auto_run_overlay)
        elif parent._avatar_auto_load and not selection_restored:
            parent.avatar_status.setText("Auto-load enabled but no avatar found. Add avatars to data/avatar/")
            parent.avatar_status.setStyleSheet("color: #fab387;")
    
    except Exception as e:
        print(f"[Avatar] Could not restore gui settings: {e}")
    
    # Also restore display settings from avatar_settings.json
    try:
        from ....avatar.persistence import load_avatar_settings
        avatar_settings = load_avatar_settings()
        
        # Restore display settings to 3D widget
        if hasattr(parent, 'avatar_preview_3d') and parent.avatar_preview_3d:
            parent.avatar_preview_3d.wireframe_mode = avatar_settings.wireframe_mode
            parent.avatar_preview_3d.show_grid = avatar_settings.show_grid
            parent.avatar_preview_3d.light_intensity = avatar_settings.light_intensity
            parent.avatar_preview_3d.ambient_strength = avatar_settings.ambient_strength
        
        # Update UI controls to match
        if hasattr(parent, 'wireframe_checkbox'):
            parent.wireframe_checkbox.setChecked(avatar_settings.wireframe_mode)
        if hasattr(parent, 'show_grid_checkbox'):
            parent.show_grid_checkbox.setChecked(avatar_settings.show_grid)
        if hasattr(parent, 'light_slider'):
            parent.light_slider.setValue(int(avatar_settings.light_intensity * 50))
        if hasattr(parent, 'ambient_slider'):
            parent.ambient_slider.setValue(int(avatar_settings.ambient_strength * 200))
            
        # Restore resize enabled from avatar persistence
        parent._avatar_resize_enabled = avatar_settings.resize_enabled
        
    except Exception as e:
        print(f"[Avatar] Could not restore avatar settings: {e}")


def _toggle_auto_load(parent, enabled: bool):
    """Toggle auto-load avatar on startup setting."""
    parent._avatar_auto_load = enabled
    if enabled:
        parent.avatar_status.setText("Avatar will auto-load on startup")
        parent.avatar_status.setStyleSheet("color: #a6e3a1;")
    else:
        parent.avatar_status.setText("Avatar auto-load disabled")
        parent.avatar_status.setStyleSheet("color: #6c7086;")
        # If auto-load is off, also disable auto-run
        if hasattr(parent, 'avatar_auto_run_checkbox'):
            parent.avatar_auto_run_checkbox.setChecked(False)
    
    # Save setting immediately
    _save_auto_settings(parent)


def _toggle_auto_run(parent, enabled: bool):
    """Toggle auto-run avatar overlay on startup setting."""
    parent._avatar_auto_run = enabled
    if enabled:
        # Auto-run requires auto-load to be on
        if not getattr(parent, '_avatar_auto_load', False):
            parent.avatar_auto_load_checkbox.setChecked(True)
        parent.avatar_status.setText("Avatar will auto-run on desktop at startup")
        parent.avatar_status.setStyleSheet("color: #a6e3a1;")
    else:
        parent.avatar_status.setText("Avatar auto-run disabled")
        parent.avatar_status.setStyleSheet("color: #6c7086;")
    
    # Save setting immediately
    _save_auto_settings(parent)


def _save_auto_settings(parent):
    """Save avatar auto-load/auto-run settings to gui_settings.json."""
    try:
        import json
        from pathlib import Path

        from ....config import CONFIG
        
        settings_path = Path(CONFIG.get("data_dir", "data")) / "gui_settings.json"
        
        # Load existing settings
        settings = {}
        if settings_path.exists():
            try:
                with open(settings_path) as f:
                    settings = json.load(f)
            except Exception:
                pass
        
        # Update auto settings
        settings["avatar_auto_load"] = getattr(parent, '_avatar_auto_load', False)
        settings["avatar_auto_run"] = getattr(parent, '_avatar_auto_run', False)
        
        # Save back
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
            
    except Exception as e:
        print(f"[Avatar] Could not save auto settings: {e}")


def _toggle_resize_enabled(parent, enabled: bool):
    """Toggle manual resize for avatar popup."""
    parent._avatar_resize_enabled = enabled
    
    # Update 2D overlay if it exists - set _resize_enabled directly
    if parent._overlay:
        parent._overlay._resize_enabled = enabled
        parent._overlay.update()  # Trigger repaint to show/hide border
    
    # Update 3D overlay if it exists - set _resize_enabled directly
    if parent._overlay_3d:
        parent._overlay_3d._resize_enabled = enabled
        parent._overlay_3d.update()  # Trigger repaint to show/hide border
    
    # Save setting to persistence
    try:
        from ....avatar.persistence import (
            save_avatar_settings,
            write_avatar_state_for_ai,
        )
        save_avatar_settings(resize_enabled=enabled)
        write_avatar_state_for_ai()
    except Exception as e:
        print(f"[Avatar] Could not save resize setting: {e}")
    
    if enabled:
        parent.avatar_status.setText("Popup resize enabled (drag edges to resize)")
        parent.avatar_status.setStyleSheet("color: #a6e3a1;")
    else:
        parent.avatar_status.setText("Popup resize disabled")
        parent.avatar_status.setStyleSheet("color: #6c7086;")


def _toggle_reposition_enabled(parent, enabled: bool):
    """Toggle reposition mode for avatar popup."""
    parent._avatar_reposition_enabled = enabled
    
    # Update 2D overlay if it exists
    if parent._overlay:
        parent._overlay._reposition_enabled = enabled
    
    # Update 3D overlay if it exists
    if parent._overlay_3d:
        parent._overlay_3d._reposition_enabled = enabled
    
    # Save setting to persistence
    try:
        from ....avatar.persistence import (
            save_avatar_settings,
            write_avatar_state_for_ai,
        )
        save_avatar_settings(reposition_enabled=enabled)
        write_avatar_state_for_ai()
    except Exception as e:
        print(f"[Avatar] Could not save reposition setting: {e}")
    
    if enabled:
        parent.avatar_status.setText("Reposition enabled (drag avatar to move)")
        parent.avatar_status.setStyleSheet("color: #a6e3a1;")
    else:
        parent.avatar_status.setText("Reposition disabled (avatar locked in place)")
        parent.avatar_status.setStyleSheet("color: #6c7086;")


def _reset_avatar_position(parent):
    """Reset avatar overlay position to center of primary screen."""
    try:
        from PyQt5.QtCore import QPoint
        from PyQt5.QtWidgets import QApplication

        # Get primary screen geometry
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            center_x = geo.x() + (geo.width() // 2) - 150  # Offset for avatar size
            center_y = geo.y() + (geo.height() // 2) - 150
        else:
            center_x, center_y = 400, 300  # Fallback
        
        # Reset 2D overlay position
        if parent._overlay:
            parent._overlay.move(center_x, center_y)
            parent.avatar_status.setText(f"Avatar moved to center ({center_x}, {center_y})")
            parent.avatar_status.setStyleSheet("color: #a6e3a1;")
        
        # Reset 3D overlay position
        if parent._overlay_3d:
            parent._overlay_3d.move(center_x, center_y)
            parent.avatar_status.setText(f"Avatar moved to center ({center_x}, {center_y})")
            parent.avatar_status.setStyleSheet("color: #a6e3a1;")
        
        # Save the new position to persistence
        try:
            from ....avatar.persistence import (
                save_avatar_settings,
                save_position,
                write_avatar_state_for_ai,
            )
            save_position(center_x, center_y)
            
            # Also clear per-avatar positions so it doesn't get overridden
            current_path = getattr(parent, '_current_path', None)
            if current_path:
                from ....avatar.persistence import get_persistence
                persistence = get_persistence()
                settings = persistence.load()
                # Remove saved position for current avatar so it uses the new default
                if str(current_path) in settings.per_avatar_positions:
                    del settings.per_avatar_positions[str(current_path)]
                    persistence.save(settings)
            
            write_avatar_state_for_ai()
        except Exception as e:
            print(f"[Avatar] Could not save reset position: {e}")
        
        # If no overlay is visible yet, still save the position for next time
        if not parent._overlay and not parent._overlay_3d:
            try:
                from ....avatar.persistence import save_position
                save_position(center_x, center_y)
                parent.avatar_status.setText(f"Position reset to center - run avatar to see it")
                parent.avatar_status.setStyleSheet("color: #89b4fa;")
            except Exception:
                pass
                
    except Exception as e:
        parent.avatar_status.setText(f"Reset position error: {e}")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")


def _overlay_wheel_resize(overlay, event):
    """Handle wheel event for 2D overlay resize."""
    delta = event.angleDelta().y()
    if delta > 0:
        overlay._size = min(500, overlay._size + 20)
    else:
        overlay._size = max(100, overlay._size - 20)
    # DON'T call setFixedSize - let _update_scaled_pixmap handle window sizing
    overlay._update_scaled_pixmap()
    event.accept()


def _overlay_3d_wheel_resize(overlay, event):
    """Handle wheel event for 3D overlay resize."""
    delta = event.angleDelta().y()
    if delta > 0:
        overlay._size = min(500, overlay._size + 25)
    else:
        overlay._size = max(100, overlay._size - 25)
    overlay.setFixedSize(overlay._size, overlay._size)
    overlay._gl_container.setFixedSize(overlay._size, overlay._size)
    if overlay._gl_widget:
        overlay._gl_widget.setFixedSize(overlay._size, overlay._size)
        overlay._apply_circular_mask()
    event.accept()


def _check_for_new_files(parent):
    """Check if new files were added and refresh if so."""
    try:
        # Count current files
        count = 0
        if AVATAR_CONFIG_DIR.exists():
            count += len(list(AVATAR_CONFIG_DIR.glob("*.json")))
        if AVATAR_IMAGES_DIR.exists():
            count += len([f for f in AVATAR_IMAGES_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])
        if AVATAR_MODELS_DIR.exists():
            # Count direct files
            count += len([f for f in AVATAR_MODELS_DIR.iterdir() if f.is_file() and f.suffix.lower() in MODEL_3D_EXTENSIONS])
            # Count subdirectories with models
            for subdir in AVATAR_MODELS_DIR.iterdir():
                if subdir.is_dir():
                    if (subdir / "scene.gltf").exists() or (subdir / "scene.glb").exists():
                        count += 1
                    else:
                        count += len([f for f in subdir.glob("*") if f.suffix.lower() in MODEL_3D_EXTENSIONS])
        
        # If count changed, refresh (preserve current selection)
        expected = getattr(parent, '_last_file_count', 0) - 1  # Minus the "-- Select --" item
        if count != expected:
            _refresh_list(parent, preserve_selection=True)
            parent._last_file_count = parent.avatar_combo.count()
            parent.avatar_status.setText(f"Found {count} avatars (auto-refreshed)")
            parent.avatar_status.setStyleSheet("color: #a6e3a1;")
    except Exception:
        pass  # Silently ignore errors in background check


def _update_activity_log(parent):
    """Update the activity log display with any new AI commands."""
    try:
        if not hasattr(parent, 'avatar_activity_log'):
            return
        
        if not AVATAR_ACTIVITY_LOG.exists():
            return
        
        content = AVATAR_ACTIVITY_LOG.read_text()
        
        # Only update if content changed
        if content != parent._last_activity_log_content:
            parent._last_activity_log_content = content
            parent.avatar_activity_log.setPlainText(content)
            # Scroll to bottom
            scrollbar = parent.avatar_activity_log.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    except Exception:
        pass


def _poll_ai_customizations(parent):
    """Check for AI-requested avatar customizations."""
    try:
        # Check the customization file
        from pathlib import Path
        settings_path = Path(__file__).parent.parent.parent.parent.parent / "information" / "avatar" / "customization.json"
        
        if not settings_path.exists():
            return
        
        # Read settings
        settings = json.loads(settings_path.read_text())
        last_updated = settings.get("_last_updated", 0)
        
        # Skip if no new changes
        if last_updated <= parent._last_ai_customize_time:
            return
        
        parent._last_ai_customize_time = last_updated
        
        # Apply customizations
        for setting, value in settings.items():
            if setting.startswith("_"):
                continue  # Skip metadata
                
            try:
                _apply_ai_customization(parent, setting, value)
            except Exception as e:
                print(f"[Avatar AI] Error applying {setting}={value}: {e}")
        
        # Clear the file after processing
        settings_path.write_text(json.dumps({"_processed": True}))
        
    except Exception:
        pass  # Silently ignore errors in background check


def _apply_ai_customization(parent, setting: str, value: str):
    """Apply a single AI customization to the avatar."""
    # Get the 3D widget if available
    widget_3d = getattr(parent, 'avatar_preview_3d', None)
    overlay = getattr(parent, '_overlay', None)
    
    setting = setting.lower()
    value_lower = value.lower() if isinstance(value, str) else value
    
    # Parse boolean values
    def parse_bool(v):
        if isinstance(v, bool):
            return v
        return v.lower() in ('true', '1', 'yes', 'on')
    
    # Parse numeric values (0-100 -> 0.0-1.0)
    def parse_float(v, min_val=0.0, max_val=1.0):
        try:
            n = float(v)
            if n > 1.0:  # Assume 0-100 scale
                n = n / 100.0
            return max(min_val, min(max_val, n))
        except (ValueError, TypeError):
            return 0.5
    
    # Parse hex color
    def parse_color(v):
        if isinstance(v, str) and v.startswith('#'):
            try:
                r = int(v[1:3], 16) / 255.0
                g = int(v[3:5], 16) / 255.0
                b = int(v[5:7], 16) / 255.0
                return [r, g, b]
            except (ValueError, IndexError):
                pass
        return None
    
    # Apply the setting
    if setting == "wireframe":
        if widget_3d:
            widget_3d.wireframe_mode = parse_bool(value)
            widget_3d.update()
        # Update checkbox if exists
        checkbox = getattr(parent, 'wireframe_checkbox', None)
        if checkbox:
            checkbox.setChecked(parse_bool(value))
            
    elif setting == "show_grid":
        if widget_3d:
            widget_3d.show_grid = parse_bool(value)
            widget_3d.update()
        checkbox = getattr(parent, 'grid_checkbox', None)
        if checkbox:
            checkbox.setChecked(parse_bool(value))
            
    elif setting == "light_intensity":
        val = parse_float(value)
        if widget_3d:
            widget_3d.light_intensity = val * 2.0  # 0-2 range
            widget_3d._update_lighting()
            widget_3d.update()
        slider = getattr(parent, 'light_slider', None)
        if slider:
            slider.setValue(int(val * 100))
            
    elif setting == "ambient_strength":
        val = parse_float(value)
        if widget_3d:
            widget_3d.ambient_strength = val * 0.5  # 0-0.5 range
            widget_3d._update_lighting()
            widget_3d.update()
        slider = getattr(parent, 'ambient_slider', None)
        if slider:
            slider.setValue(int(val * 100))
            
    elif setting == "rotate_speed":
        val = parse_float(value)
        if widget_3d:
            widget_3d.auto_rotate_speed = val * 2.0  # 0-2 range
        slider = getattr(parent, 'rotate_speed_slider', None)
        if slider:
            slider.setValue(int(val * 100))
            
    elif setting == "auto_rotate":
        if widget_3d:
            if parse_bool(value):
                widget_3d.start_auto_rotate()
            else:
                widget_3d.stop_auto_rotate()
        checkbox = getattr(parent, 'auto_rotate_checkbox', None)
        if checkbox:
            checkbox.setChecked(parse_bool(value))
            
    elif setting == "primary_color":
        color = parse_color(value)
        if color:
            # Update 2D sprite colors
            parent._current_colors["primary"] = value
            _update_sprite_colors(parent)
            # Also set 3D model color
            if widget_3d:
                widget_3d.model_color = color
                widget_3d.update()
                
    elif setting == "secondary_color":
        color = parse_color(value)
        if color:
            parent._current_colors["secondary"] = value
            _update_sprite_colors(parent)
            
    elif setting == "accent_color":
        color = parse_color(value)
        if color:
            parent._current_colors["accent"] = value
            _update_sprite_colors(parent)
            
    elif setting == "reset":
        # Reset everything
        if widget_3d:
            widget_3d.reset_all()
        # Reset colors to defaults
        parent._current_colors = {
            "primary": "#6366f1",
            "secondary": "#4f46e5",
            "accent": "#818cf8"
        }
        _update_sprite_colors(parent)
    
    elif setting == "expression":
        # Set expression (AI can control expressions)
        _set_expression(parent, value)
        # Also update overlay if visible
        if overlay:
            overlay._change_expression(value) if hasattr(overlay, '_change_expression') else None
    
    elif setting == "wave":
        # Trigger wave animation on overlay
        if overlay and hasattr(overlay, '_wave'):
            overlay._wave()
    
    elif setting == "nod":
        # Trigger nod animation
        pass  # Implement when animation system ready
    
    elif setting == "show_overlay":
        # Show/hide desktop overlay
        if parse_bool(value):
            if parent.show_overlay_btn:
                parent.show_overlay_btn.setChecked(True)
                _toggle_overlay(parent)
        else:
            if parent.show_overlay_btn:
                parent.show_overlay_btn.setChecked(False)
                _toggle_overlay(parent)
        
    print(f"[Avatar AI] Applied: {setting} = {value}")


def _update_sprite_colors(parent):
    """Update the 2D sprite with current colors."""
    try:
        expr = getattr(parent, 'current_expression', 'neutral')
        svg_data = generate_sprite(
            expr,
            parent._current_colors["primary"],
            parent._current_colors["secondary"],
            parent._current_colors["accent"]
        )
        parent.avatar_preview_2d.set_svg_sprite(svg_data)
    except Exception:
        pass


def _is_avatar_module_enabled() -> bool:
    """Check if avatar module is enabled in ModuleManager.
    
    NOTE: We default to True because the avatar tab provides its own
    functionality for previewing/customizing avatars. The module check
    is only relevant for integration with the main chat system.
    """
    # Always return True - avatar tab features work standalone
    return True


def _show_default_preview(parent):
    """Show a default preview sprite in the preview area."""
    svg_data = generate_sprite(
        "neutral",
        parent._current_colors["primary"],
        parent._current_colors["secondary"],
        parent._current_colors["accent"]
    )
    parent.avatar_preview_2d.set_svg_sprite(svg_data)
    parent._using_builtin_sprite = True


def _test_random_expression(parent):
    """Cycle through expressions for preview testing."""
    # Instead of random, cycle through expressions in order for testing
    expressions = list(SPRITE_TEMPLATES.keys()) if SPRITE_TEMPLATES else []
    if not expressions:
        return
    
    # Get current expression index and move to next
    current = getattr(parent, 'current_expression', 'neutral')
    try:
        current_idx = expressions.index(current)
        next_idx = (current_idx + 1) % len(expressions)
    except ValueError:
        next_idx = 0
    
    expr = expressions[next_idx]
    parent.current_expression = expr
    if hasattr(parent, 'expression_label'):
        parent.expression_label.setText(f"Current: {expr}")
    
    svg_data = generate_sprite(
        expr,
        parent._current_colors["primary"],
        parent._current_colors["secondary"],
        parent._current_colors["accent"]
    )
    parent.avatar_preview_2d.set_svg_sprite(svg_data)
    
    # Also update overlay if visible (pass original, overlay handles scaling)
    if parent._overlay and parent._overlay.isVisible():
        pixmap = parent.avatar_preview_2d.original_pixmap
        if pixmap:
            parent._overlay.set_avatar(pixmap)


def _set_expression(parent, expression: str, from_callback: bool = False):
    """Set avatar expression and update preview.
    
    Args:
        parent: The avatar display widget
        expression: Expression name (neutral, happy, sad, etc.)
        from_callback: If True, don't call controller again (prevents loop)
    """
    parent.current_expression = expression
    if hasattr(parent, 'expression_label'):
        parent.expression_label.setText(f"Current: {expression}")
    
    # Only notify controller if this wasn't triggered by controller callback
    if not from_callback:
        parent._avatar_controller.set_expression(expression)
    
    # Update preview with new expression sprite
    if parent._using_builtin_sprite or not parent._current_path:
        svg_data = generate_sprite(
            expression,
            parent._current_colors["primary"],
            parent._current_colors["secondary"],
            parent._current_colors["accent"]
        )
        parent.avatar_preview_2d.set_svg_sprite(svg_data)
        
        # Update overlay too (pass original, overlay handles scaling)
        if parent._overlay and parent._overlay.isVisible():
            pixmap = parent.avatar_preview_2d.original_pixmap
            if pixmap:
                parent._overlay.set_avatar(pixmap)
    else:
        # Using custom avatar - try to find expression-specific image
        _update_avatar_for_expression(parent, expression)
    
    parent.avatar_status.setText(f"Expression: {expression}")
    parent.avatar_status.setStyleSheet("color: #a6e3a1;")


def _update_avatar_for_expression(parent, expression: str):
    """Update custom avatar to show specific expression if available."""
    if not parent._current_path:
        return
    
    from pathlib import Path
    base_path = Path(parent._current_path)
    
    # Check for expression-specific image in same folder or emotions subfolder
    search_paths = []
    if base_path.is_file():
        folder = base_path.parent
        search_paths = [
            folder / f"{expression}.png",
            folder / "emotions" / f"{expression}.png",
            folder / f"{expression}.jpg",
            folder / f"{expression}.gif",
        ]
    elif base_path.is_dir():
        search_paths = [
            base_path / f"{expression}.png",
            base_path / "emotions" / f"{expression}.png",
            base_path / f"{expression}.jpg",
        ]
    
    # Try to find expression image
    for expr_path in search_paths:
        if expr_path.exists():
            pixmap = QPixmap(str(expr_path))
            if not pixmap.isNull():
                parent.avatar_preview_2d.set_avatar(pixmap)
                
                # Update overlay (pass original, overlay handles scaling)
                if parent._overlay and parent._overlay.isVisible():
                    parent._overlay.set_avatar(pixmap)
                return
    
    # No expression-specific image found - keep current
    
    parent.avatar_status.setText(f"Expression: {expression}")
    parent.avatar_status.setStyleSheet("color: #a6e3a1;")


def _use_builtin_sprite(parent):
    """Switch to built-in sprite system."""
    parent._using_builtin_sprite = True
    parent._current_path = None
    
    # Generate default sprite
    svg_data = generate_sprite(
        parent.current_expression,
        parent._current_colors["primary"],
        parent._current_colors["secondary"],
        parent._current_colors["accent"]
    )
    parent.avatar_preview_2d.set_svg_sprite(svg_data)
    
    parent.avatar_status.setText("Using built-in sprite")
    parent.avatar_status.setStyleSheet("color: #a6e3a1;")


def _export_sprite(parent):
    """Export current sprite to file."""
    if not parent._using_builtin_sprite:
        parent.avatar_status.setText("Use built-in sprite first to export")
        parent.avatar_status.setStyleSheet("color: #fab387;")
        return
    
    path, _ = QFileDialog.getSaveFileName(
        parent,
        "Export Sprite",
        str(AVATAR_IMAGES_DIR / f"avatar_{parent.current_expression}.svg"),
        "SVG Files (*.svg);;PNG Files (*.png)"
    )
    
    if path:
        from ....avatar.renderers.default_sprites import save_sprite
        save_sprite(
            parent.current_expression,
            path,
            parent._current_colors["primary"],
            parent._current_colors["secondary"],
            parent._current_colors["accent"]
        )
        parent.avatar_status.setText(f"Exported to: {Path(path).name}")
        parent.avatar_status.setStyleSheet("color: #a6e3a1;")


def _toggle_avatar(parent, enabled):
    """Toggle avatar."""
    if enabled:
        parent._avatar_controller.enable()
        
        # Set initial appearance with current colors
        from ....avatar.avatar_identity import AvatarAppearance
        appearance = AvatarAppearance(
            primary_color=parent._current_colors["primary"],
            secondary_color=parent._current_colors["secondary"],
            accent_color=parent._current_colors["accent"],
            default_expression=parent.current_expression
        )
        parent._avatar_controller.set_appearance(appearance)
        
        parent.avatar_status.setText("Avatar enabled")
        parent.avatar_status.setStyleSheet("color: #a6e3a1;")
    else:
        parent._avatar_controller.disable()
        parent.avatar_status.setText("Avatar disabled")
        parent.avatar_status.setStyleSheet("color: #6c7086;")


# ============================================================================
# Parallax 2.5D Effect Functions
# ============================================================================

def _toggle_parallax(parent, enabled: bool):
    """Toggle parallax effect for 2D avatars."""
    parent.avatar_preview_2d.set_parallax_enabled(enabled)
    
    # Enable/disable related controls
    parent.parallax_mouse_check.setEnabled(enabled)
    parent.parallax_intensity_slider.setEnabled(enabled)
    
    if enabled:
        # Check if we have layers
        if not parent.avatar_preview_2d._parallax_layers:
            parent.avatar_status.setText("Parallax ON - Click 'AI: Generate Layers' or 'Load Layers'")
            parent.avatar_status.setStyleSheet("color: #f5c2e7;")
        else:
            parent.avatar_status.setText(f"Parallax ON - {len(parent.avatar_preview_2d._parallax_layers)} layers active")
            parent.avatar_status.setStyleSheet("color: #a6e3a1;")
        
        # Enable mouse tracking by default
        if parent.parallax_mouse_check.isChecked():
            parent.avatar_preview_2d.set_mouse_tracking_parallax(True)
    else:
        parent.avatar_preview_2d.set_mouse_tracking_parallax(False)
        parent.avatar_status.setText("Parallax disabled")
        parent.avatar_status.setStyleSheet("color: #6c7086;")


def _toggle_parallax_mouse(parent, enabled: bool):
    """Toggle mouse-based parallax tracking."""
    parent.avatar_preview_2d.set_mouse_tracking_parallax(enabled)


def _set_parallax_intensity(parent, value: int):
    """Set parallax effect intensity."""
    parent.avatar_preview_2d._parallax_intensity = value
    parent.parallax_intensity_label.setText(str(value))
    parent.avatar_preview_2d.update()


def _load_parallax_layers(parent):
    """Load custom parallax layer files."""
    paths, _ = QFileDialog.getOpenFileNames(
        parent,
        "Select Layer Images (ordered back to front)",
        str(AVATAR_IMAGES_DIR),
        "Images (*.png *.jpg *.jpeg *.webp);;All Files (*)"
    )
    
    if not paths:
        return
    
    layers = []
    num_layers = len(paths)
    
    for i, path_str in enumerate(paths):
        path = Path(path_str)
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            # Auto-assign depth based on filename or order
            name = path.stem.lower()
            if '_bg' in name or '_back' in name or 'background' in name:
                depth = 0.0
            elif '_body' in name or 'body' in name:
                depth = 0.3
            elif '_face' in name or 'face' in name or 'head' in name:
                depth = 0.6
            elif '_eyes' in name or 'eyes' in name or '_front' in name:
                depth = 0.9
            else:
                # Distribute evenly based on position
                depth = i / max(1, num_layers - 1)
            
            layers.append((pixmap, depth))
    
    if layers:
        # Sort by depth (back to front)
        layers.sort(key=lambda x: x[1])
        parent.avatar_preview_2d.set_parallax_layers(layers)
        
        # Auto-enable parallax
        parent.parallax_check.setChecked(True)
        
        parent.avatar_status.setText(f"Loaded {len(layers)} parallax layers")
        parent.avatar_status.setStyleSheet("color: #a6e3a1;")
    else:
        parent.avatar_status.setText("No valid images loaded")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")


def _ai_generate_parallax_layers(parent):
    """Use AI to generate depth layers from a single 2D image."""
    if not parent.original_pixmap and not parent.avatar_preview_2d.original_pixmap:
        parent.avatar_status.setText("Load an image first!")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")
        return
    
    parent.avatar_status.setText("Generating layers... (this may take a moment)")
    parent.avatar_status.setStyleSheet("color: #89dceb;")
    QApplication.processEvents()
    
    try:
        # Get the original pixmap
        pixmap = parent.avatar_preview_2d.original_pixmap
        if not pixmap:
            parent.avatar_status.setText("No image to process!")
            parent.avatar_status.setStyleSheet("color: #f38ba8;")
            return
        
        # Convert QPixmap to numpy array for processing
        img = pixmap.toImage()
        width, height = img.width(), img.height()
        
        # Try using depth estimation model if available
        layers = _generate_layers_from_depth(parent, img, width, height)
        
        if layers:
            parent.avatar_preview_2d.set_parallax_layers(layers)
            parent.parallax_check.setChecked(True)
            parent.avatar_status.setText(f"Generated {len(layers)} depth layers!")
            parent.avatar_status.setStyleSheet("color: #a6e3a1;")
        else:
            # Fallback: create simple layered version
            layers = _generate_simple_layers(parent, pixmap)
            parent.avatar_preview_2d.set_parallax_layers(layers)
            parent.parallax_check.setChecked(True)
            parent.avatar_status.setText("Generated simple layers (install torch for better AI)")
            parent.avatar_status.setStyleSheet("color: #fab387;")
            
    except Exception as e:
        logger.error(f"Error generating parallax layers: {e}")
        parent.avatar_status.setText(f"Error: {str(e)[:50]}")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")


def _generate_layers_from_depth(parent, qimage, width, height):
    """Generate layers using AI depth estimation."""
    try:
        # Try to use depth estimation
        import numpy as np
        from PIL import Image
        
        # Convert QImage to PIL Image
        qimage = qimage.convertToFormat(QImage.Format_RGBA8888)
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.array(ptr).reshape(height, width, 4)
        pil_img = Image.fromarray(arr[:, :, :3])  # RGB only
        
        # Try different depth estimation methods
        depth_map = None
        
        # Method 1: Try transformers depth estimation
        try:
            from transformers import pipeline
            depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
            result = depth_estimator(pil_img)
            depth_map = np.array(result['depth'])
            print("[Parallax] Using transformers depth estimation")
        except Exception:
            pass
        
        # Method 2: Try MiDaS directly
        if depth_map is None:
            try:
                import torch
                midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
                midas.eval()
                
                transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
                input_batch = transform(pil_img)
                
                with torch.no_grad():
                    prediction = midas(input_batch)
                    depth_map = prediction.squeeze().cpu().numpy()
                print("[Parallax] Using MiDaS depth estimation")
            except Exception:
                pass
        
        if depth_map is None:
            return None
        
        # Normalize depth map to 0-1
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        # Create 4 layers based on depth ranges
        layers = []
        depth_ranges = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        
        for i, (d_min, d_max) in enumerate(depth_ranges):
            # Create mask for this depth range
            mask = ((depth_map >= d_min) & (depth_map < d_max)).astype(np.uint8) * 255
            
            # Apply mask to original image
            layer_arr = arr.copy()
            layer_arr[:, :, 3] = mask  # Set alpha channel
            
            # Convert back to QPixmap
            layer_img = QImage(layer_arr.data, width, height, width * 4, QImage.Format_RGBA8888)
            layer_pixmap = QPixmap.fromImage(layer_img.copy())
            
            depth_value = (d_min + d_max) / 2
            layers.append((layer_pixmap, depth_value))
        
        return layers
        
    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"Depth estimation failed: {e}")
        return None


def _generate_simple_layers(parent, pixmap):
    """Generate simple parallax layers without AI (center-weighted)."""
    # Create 3 simple layers: edge (back), middle, center (front)
    img = pixmap.toImage()
    width, height = img.width(), img.height()
    
    layers = []
    
    try:
        import numpy as np
        
        # Convert to numpy
        img = img.convertToFormat(QImage.Format_RGBA8888)
        ptr = img.bits()
        ptr.setsize(height * width * 4)
        arr = np.array(ptr).reshape(height, width, 4).copy()
        
        # Create distance-from-center map
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width / 2, height / 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        distance = distance / max_dist  # Normalize to 0-1
        
        # Layer 1: Outer edges (background, depth=0)
        mask1 = (distance > 0.6).astype(np.uint8) * 255
        layer1 = arr.copy()
        layer1[:, :, 3] = np.minimum(layer1[:, :, 3], mask1)
        l1_img = QImage(layer1.data, width, height, width * 4, QImage.Format_RGBA8888)
        layers.append((QPixmap.fromImage(l1_img.copy()), 0.0))
        
        # Layer 2: Middle ring (body, depth=0.5)
        mask2 = ((distance > 0.3) & (distance <= 0.6)).astype(np.uint8) * 255
        layer2 = arr.copy()
        layer2[:, :, 3] = np.minimum(layer2[:, :, 3], mask2)
        l2_img = QImage(layer2.data, width, height, width * 4, QImage.Format_RGBA8888)
        layers.append((QPixmap.fromImage(l2_img.copy()), 0.5))
        
        # Layer 3: Center (face/focus, depth=1.0)
        mask3 = (distance <= 0.3).astype(np.uint8) * 255
        layer3 = arr.copy()
        layer3[:, :, 3] = np.minimum(layer3[:, :, 3], mask3)
        l3_img = QImage(layer3.data, width, height, width * 4, QImage.Format_RGBA8888)
        layers.append((QPixmap.fromImage(l3_img.copy()), 1.0))
        
    except ImportError:
        # No numpy - just use the original image as a single layer
        layers.append((pixmap, 0.5))
    
    return layers


def _toggle_eye_tracking(parent, enabled: bool):
    """Toggle avatar eye tracking - look at mouse cursor."""
    parent.eye_tracking_enabled = enabled
    
    # Enable/disable on overlays
    if hasattr(parent, '_overlay') and parent._overlay:
        parent._overlay.set_eye_tracking(enabled)
    if hasattr(parent, '_overlay_3d') and parent._overlay_3d:
        parent._overlay_3d.set_eye_tracking(enabled)
    
    if enabled:
        parent.avatar_status.setText("Eye tracking enabled - avatar will follow your cursor")
        parent.avatar_status.setStyleSheet("color: #89b4fa;")
    else:
        parent.avatar_status.setText("Eye tracking disabled")
        parent.avatar_status.setStyleSheet("color: #6c7086;")


def _toggle_avatar_autonomous(parent, state):
    """Toggle avatar autonomous mode (react to screen content)."""
    enabled = state == 2  # Qt.Checked
    
    try:
        from ....avatar import get_avatar
        from ....avatar.autonomous import get_autonomous_avatar
        
        avatar = get_avatar()
        autonomous = get_autonomous_avatar(avatar)
        
        if enabled:
            avatar.enable()
            autonomous.start()
            if hasattr(parent, 'avatar_behavior_status'):
                parent.avatar_behavior_status.setText("Mode: Autonomous (watching screen)")
                parent.avatar_behavior_status.setStyleSheet("color: #22c55e; font-weight: bold;")
        else:
            autonomous.stop()
            if hasattr(parent, 'avatar_behavior_status'):
                parent.avatar_behavior_status.setText("Mode: Manual")
                parent.avatar_behavior_status.setStyleSheet("color: #bac2de; font-style: italic;")
    except Exception as e:
        if hasattr(parent, 'avatar_behavior_status'):
            parent.avatar_behavior_status.setText(f"Error: {e}")
            parent.avatar_behavior_status.setStyleSheet("color: #ef4444;")


def _toggle_overlay(parent):
    """Toggle desktop overlay (2D or 3D based on current model)."""
    # Check if module is enabled
    if not getattr(parent, '_avatar_module_enabled', True):
        parent.show_overlay_btn.setChecked(False)
        parent.avatar_status.setText("Enable avatar module in Modules tab first")
        parent.avatar_status.setStyleSheet("color: #fab387;")
        return
    
    is_3d = getattr(parent, '_is_3d_model', False) and getattr(parent, '_using_3d_render', False)
    
    if parent.show_overlay_btn.isChecked():
        # Create or show overlay
        if is_3d and HAS_OPENGL and HAS_TRIMESH:
            # Use 3D overlay
            if parent._overlay_3d is None:
                parent._overlay_3d = Avatar3DOverlayWindow()
                parent._overlay_3d.closed.connect(lambda: _on_overlay_closed(parent))
                
                # Load size from avatar persistence (primary) or fallback
                try:
                    from ....avatar.persistence import load_avatar_settings
                    avatar_settings = load_avatar_settings()
                    saved_size = avatar_settings.overlay_3d_size
                    
                    # Also restore control bar settings
                    parent._overlay_3d._control_bar_x = avatar_settings.control_bar_x
                    parent._overlay_3d._control_bar_y = avatar_settings.control_bar_y
                    parent._overlay_3d._control_bar_width = avatar_settings.control_bar_width
                    parent._overlay_3d._control_bar_height = avatar_settings.control_bar_height
                    parent._overlay_3d._show_control_bar = avatar_settings.show_control_bar_3d
                    parent._overlay_3d._click_through_mode = avatar_settings.click_through_mode_3d
                    
                    # Restore hit area ratio
                    if hasattr(parent._overlay_3d, '_hit_layer') and parent._overlay_3d._hit_layer:
                        hit_ratio = getattr(avatar_settings, 'hit_area_ratio', 0.4)
                        parent._overlay_3d._hit_layer.set_hit_ratio(hit_ratio)
                    
                    # Sync drag bar widget with loaded settings (legacy - may not be used)
                    if hasattr(parent._overlay_3d, '_drag_bar'):
                        parent._overlay_3d._drag_bar.setVisible(avatar_settings.show_control_bar_3d)
                except Exception:
                    saved_size = getattr(parent, '_saved_overlay_3d_size', 250)
                
                # Ensure minimum size 100, no max limit
                saved_size = max(100, saved_size)
                parent._overlay_3d._size = saved_size
                parent._overlay_3d.setFixedSize(saved_size, saved_size)
                parent._overlay_3d._gl_container.setFixedSize(saved_size, saved_size)
                if parent._overlay_3d._gl_widget:
                    parent._overlay_3d._gl_widget.setFixedSize(saved_size, saved_size)
                
                # Restore position from persistence (same as 2D overlay)
                try:
                    from ....avatar.persistence import load_position
                    x, y = load_position()
                    if x >= 0 and y >= 0:
                        parent._overlay_3d.move(x, y)
                except Exception:
                    pass
                
                # Reapply mask after loading settings (includes drag bar region)
                parent._overlay_3d._apply_circular_mask()
            
            # ALWAYS set resize to OFF by default when showing overlay
            # This overrides any saved setting - user can enable via right-click if wanted
            parent._overlay_3d._resize_enabled = False
            
            # Load the model into 3D overlay
            if parent._current_path:
                parent._overlay_3d.load_model(str(parent._current_path))
                parent._overlay_3d.show()
                parent._overlay_3d.raise_()
                parent.show_overlay_btn.setText("Stop")
                parent.avatar_status.setText("3D avatar on desktop! Drag the bar to move, right-click for options.")
                parent.avatar_status.setStyleSheet("color: #a6e3a1;")
            else:
                parent.show_overlay_btn.setChecked(False)
                parent.avatar_status.setText("No 3D model loaded")
                parent.avatar_status.setStyleSheet("color: #fab387;")
        else:
            # Use 2D overlay
            if parent._overlay is None:
                parent._overlay = AvatarOverlayWindow()
                parent._overlay.closed.connect(lambda: _on_overlay_closed(parent))
            
            # Set avatar path FIRST so per-avatar settings load correctly
            if parent._current_path:
                parent._overlay.set_avatar_path(str(parent._current_path))
            else:
                # Fallback: load default size from persistence
                try:
                    from ....avatar.persistence import (
                        load_avatar_settings,
                        load_position,
                    )
                    avatar_settings = load_avatar_settings()
                    saved_size = avatar_settings.overlay_size
                    # Allow any reasonable size (50-2000)
                    if saved_size < 50:
                        saved_size = 300
                    elif saved_size > 2000:
                        saved_size = 2000
                    parent._overlay._size = saved_size
                    
                    # Restore position
                    x, y = load_position()
                    if x >= 0 and y >= 0:
                        parent._overlay.move(x, y)
                except Exception:
                    pass
            
            # ALWAYS set resize to OFF by default when showing overlay
            # This overrides any saved setting
            parent._overlay._resize_enabled = False
            
            # Only show avatar if one is selected - don't show default/test sprite
            pixmap = parent.avatar_preview_2d.original_pixmap
            if not pixmap or getattr(parent, '_using_builtin_sprite', True):
                # No real avatar loaded - require user to select one
                parent.show_overlay_btn.setChecked(False)
                parent.avatar_status.setText("Select an avatar first! Use 'Load Avatar' or pick from dropdown.")
                parent.avatar_status.setStyleSheet("color: #fab387;")
                return
            
            if pixmap:
                # Pass the ORIGINAL pixmap - set_avatar will scale and resize window to wrap tightly
                parent._overlay.set_avatar(pixmap)
                parent._overlay.show()
                parent._overlay.raise_()
                parent.show_overlay_btn.setText("Stop")
                parent.avatar_status.setText("Avatar on desktop! Drag to move, right-click for gestures.")
                parent.avatar_status.setStyleSheet("color: #a6e3a1;")
            else:
                parent.show_overlay_btn.setChecked(False)
                parent.avatar_status.setText("Could not create avatar sprite")
                parent.avatar_status.setStyleSheet("color: #f38ba8;")
    else:
        # Hide overlays
        if parent._overlay:
            parent._overlay.hide()
        if parent._overlay_3d:
            parent._overlay_3d.hide()
        parent.show_overlay_btn.setText("Run")
        parent.avatar_status.setText("Avatar stopped")
        parent.avatar_status.setStyleSheet("color: #6c7086;")


def _on_overlay_closed(parent):
    """Handle overlay closed."""
    parent.show_overlay_btn.setChecked(False)
    parent.show_overlay_btn.setText("Run")


def _toggle_3d_render(parent, enabled):
    """Toggle between 2D preview and 3D rendering."""
    parent._using_3d_render = enabled
    
    if enabled and parent.avatar_preview_3d:
        parent.avatar_preview_2d.setVisible(False)
        parent.avatar_preview_3d.setVisible(True)
        parent.reset_view_btn.setVisible(True)
        
        # Load model into 3D viewer if we have a 3D model
        if parent._is_3d_model and parent._current_path:
            parent.avatar_preview_3d.load_model(str(parent._current_path))
    else:
        parent.avatar_preview_2d.setVisible(True)
        if parent.avatar_preview_3d:
            parent.avatar_preview_3d.setVisible(False)
        parent.reset_view_btn.setVisible(False)


def _reset_view(parent):
    """Reset 3D view."""
    if parent.avatar_preview_3d:
        parent.avatar_preview_3d.reset_view()


def _set_wireframe(parent, enabled: bool):
    """Toggle wireframe mode."""
    if parent.avatar_preview_3d:
        parent.avatar_preview_3d.wireframe_mode = enabled
        parent.avatar_preview_3d.update()
    # Save setting
    try:
        from ....avatar.persistence import save_avatar_settings
        save_avatar_settings(wireframe_mode=enabled)
    except Exception:
        pass


def _set_show_grid(parent, enabled: bool):
    """Toggle grid floor."""
    if parent.avatar_preview_3d:
        parent.avatar_preview_3d.show_grid = enabled
        parent.avatar_preview_3d.update()
    # Save setting
    try:
        from ....avatar.persistence import save_avatar_settings
        save_avatar_settings(show_grid=enabled)
    except Exception:
        pass


def _set_lighting(parent, intensity: float):
    """Set lighting intensity."""
    if parent.avatar_preview_3d:
        parent.avatar_preview_3d.light_intensity = intensity
        parent.avatar_preview_3d._update_lighting()
        parent.avatar_preview_3d.update()
    # Save setting
    try:
        from ....avatar.persistence import save_avatar_settings
        save_avatar_settings(light_intensity=intensity)
    except Exception:
        pass


def _set_ambient(parent, strength: float):
    """Set ambient light strength."""
    if parent.avatar_preview_3d:
        parent.avatar_preview_3d.ambient_strength = strength
        parent.avatar_preview_3d._update_lighting()
        parent.avatar_preview_3d.update()
    # Save setting
    try:
        from ....avatar.persistence import save_avatar_settings
        save_avatar_settings(ambient_strength=strength)
    except Exception:
        pass


def _set_rotate_speed(parent, speed: float):
    """Set auto-rotate speed."""
    if parent.avatar_preview_3d:
        parent.avatar_preview_3d.auto_rotate_speed = speed


def _set_model_orientation(parent, axis: str, value: float):
    """Set model orientation for a specific axis."""
    if parent.avatar_preview_3d:
        # Convert slider value to radians
        import math
        radians = math.radians(value)
        
        if axis == 'pitch':
            parent.avatar_preview_3d.model_pitch = radians
            if hasattr(parent, 'pitch_label'):
                parent.pitch_label.setText(str(int(value)))
        elif axis == 'yaw':
            parent.avatar_preview_3d.model_yaw = radians
            if hasattr(parent, 'yaw_label'):
                parent.yaw_label.setText(str(int(value)))
            # Update facing direction indicator
            _update_facing_direction_label(parent, int(value))
        elif axis == 'roll':
            parent.avatar_preview_3d.model_roll = radians
            if hasattr(parent, 'roll_label'):
                parent.roll_label.setText(str(int(value)))
        
        parent.avatar_preview_3d.update()


def _update_facing_direction_label(parent, yaw_degrees: int):
    """Update the facing direction label based on yaw rotation."""
    if not hasattr(parent, 'facing_direction_label'):
        return
    
    # Normalize to 0-360 range
    yaw = yaw_degrees % 360
    if yaw < 0:
        yaw += 360
    
    # Determine direction name based on yaw angle
    if yaw >= 337.5 or yaw < 22.5:
        direction = "Front"
    elif 22.5 <= yaw < 67.5:
        direction = "Front-Right"
    elif 67.5 <= yaw < 112.5:
        direction = "Right"
    elif 112.5 <= yaw < 157.5:
        direction = "Back-Right"
    elif 157.5 <= yaw < 202.5:
        direction = "Back"
    elif 202.5 <= yaw < 247.5:
        direction = "Back-Left"
    elif 247.5 <= yaw < 292.5:
        direction = "Left"
    else:  # 292.5 <= yaw < 337.5
        direction = "Front-Left"
    
    parent.facing_direction_label.setText(f"{direction} ({yaw_degrees} deg)")


def _quick_rotate(parent, axis: str, degrees: int):
    """Quick rotate by specified degrees."""
    if axis == 'pitch' and hasattr(parent, 'pitch_slider'):
        current = parent.pitch_slider.value()
        new_val = (current + degrees) % 360
        if new_val > 180:
            new_val -= 360
        parent.pitch_slider.setValue(new_val)
    elif axis == 'yaw' and hasattr(parent, 'yaw_slider'):
        current = parent.yaw_slider.value()
        new_val = (current + degrees) % 360
        if new_val > 180:
            new_val -= 360
        parent.yaw_slider.setValue(new_val)
    elif axis == 'roll' and hasattr(parent, 'roll_slider'):
        current = parent.roll_slider.value()
        new_val = (current + degrees) % 360
        if new_val > 180:
            new_val -= 360
        parent.roll_slider.setValue(new_val)


def _reset_orientation(parent):
    """Reset all orientation sliders to 0."""
    if hasattr(parent, 'pitch_slider'):
        parent.pitch_slider.setValue(0)
    if hasattr(parent, 'yaw_slider'):
        parent.yaw_slider.setValue(0)
    if hasattr(parent, 'roll_slider'):
        parent.roll_slider.setValue(0)
    # Update facing direction indicator
    _update_facing_direction_label(parent, 0)


def _auto_design_avatar(parent):
    """Let AI design avatar based on personality (runs in background thread)."""
    import threading
    
    parent.avatar_status.setText("AI is designing avatar...")
    parent.avatar_status.setStyleSheet("color: #89b4fa;")
    parent.auto_design_btn.setEnabled(False)
    
    def do_design():
        try:
            # Try to get the avatar controller
            avatar = get_avatar()
            if avatar and hasattr(avatar, 'auto_design'):
                appearance = avatar.auto_design()
                
                # Update UI on main thread
                from PyQt5.QtCore import QTimer
                def update_ui():
                    if appearance:
                        parent.avatar_status.setText(f"AI designed avatar: {appearance.style if hasattr(appearance, 'style') else 'Custom'}")
                        parent.avatar_status.setStyleSheet("color: #a6e3a1;")
                        _refresh_avatar(parent)
                    else:
                        parent.avatar_status.setText("AI design complete - refresh to see changes")
                        parent.avatar_status.setStyleSheet("color: #a6e3a1;")
                    parent.auto_design_btn.setEnabled(True)
                
                QTimer.singleShot(0, update_ui)
            else:
                # Fallback: Use AI to pick a style based on personality
                try:
                    from ....core.inference import EnigmaEngine
                    engine = EnigmaEngine.get_instance()
                    styles = list(SPRITE_TEMPLATES.keys()) if SPRITE_TEMPLATES else ['default']
                    
                    if engine and engine.model:
                        prompt = f"Pick ONE avatar style from this list that best fits an AI assistant: {', '.join(styles)}. Reply with ONLY the style name, nothing else."
                        response = engine.generate(prompt, max_gen=20, temperature=0.7)
                        chosen_style = response.strip().lower()
                        # Validate it's a real style
                        if chosen_style not in styles:
                            chosen_style = styles[0]  # Default to first if invalid
                    else:
                        chosen_style = styles[0]  # No AI, use first style
                except Exception:
                    styles = list(SPRITE_TEMPLATES.keys()) if SPRITE_TEMPLATES else ['default']
                    chosen_style = styles[0]  # Fallback to first style
                
                # Generate sprite with random style
                sprite_path = generate_sprite(chosen_style)
                
                from PyQt5.QtCore import QTimer
                def update_ui():
                    if sprite_path:
                        parent.avatar_status.setText(f"Generated AI style: {chosen_style}")
                        parent.avatar_status.setStyleSheet("color: #a6e3a1;")
                        _refresh_avatar(parent)
                    else:
                        parent.avatar_status.setText("Could not generate avatar")
                        parent.avatar_status.setStyleSheet("color: #f38ba8;")
                    parent.auto_design_btn.setEnabled(True)
                
                QTimer.singleShot(0, update_ui)
                
        except Exception as e:
            from PyQt5.QtCore import QTimer
            def show_error():
                parent.avatar_status.setText(f"Auto-design error: {str(e)[:50]}")
                parent.avatar_status.setStyleSheet("color: #f38ba8;")
                parent.auto_design_btn.setEnabled(True)
            QTimer.singleShot(0, show_error)
    
    thread = threading.Thread(target=do_design, daemon=True)
    thread.start()


def _set_camera_view(parent, rotation_x: float, rotation_y: float):
    """Set camera view angle (for View Presets)."""
    if parent.avatar_preview_3d:
        parent.avatar_preview_3d.rotation_x = rotation_x
        parent.avatar_preview_3d.rotation_y = rotation_y
        parent.avatar_preview_3d.update()


def _auto_orient_model(parent):
    """Automatically detect and fix model orientation."""
    if not parent.avatar_preview_3d:
        parent.avatar_status.setText("No model loaded")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")
        return
    
    # Use the widget's auto-orient function
    pitch, yaw, roll = parent.avatar_preview_3d.auto_orient_model()
    
    if pitch == 0 and yaw == 0 and roll == 0:
        parent.avatar_status.setText("Model appears correctly oriented")
        parent.avatar_status.setStyleSheet("color: #a6e3a1;")
        return
    
    # Apply the detected orientation
    if hasattr(parent, 'pitch_slider'):
        parent.pitch_slider.setValue(int(pitch))
    if hasattr(parent, 'yaw_slider'):
        parent.yaw_slider.setValue(int(yaw))
    if hasattr(parent, 'roll_slider'):
        parent.roll_slider.setValue(int(roll))
    
    # Update facing direction indicator
    _update_facing_direction_label(parent, int(yaw))
    
    parent.avatar_status.setText(f"Auto-oriented: X={pitch}deg, Y={yaw}deg, Z={roll}deg")
    parent.avatar_status.setStyleSheet("color: #89b4fa;")


def _save_model_orientation(parent):
    """Save model orientation to settings file."""
    import json
    import math
    from pathlib import Path
    
    if not parent.avatar_preview_3d:
        parent.avatar_status.setText("No model loaded")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")
        return
    
    # Get current orientation values
    pitch = getattr(parent.avatar_preview_3d, 'model_pitch', 0.0)
    yaw = getattr(parent.avatar_preview_3d, 'model_yaw', 0.0)
    roll = getattr(parent.avatar_preview_3d, 'model_roll', 0.0)
    
    # Get current model path for identification
    model_path = getattr(parent.avatar_preview_3d, '_model_path', None)
    if not model_path:
        parent.avatar_status.setText("No model path found")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")
        return
    
    # Load or create settings file
    settings_path = Path(CONFIG.get("data_dir", "data")) / "avatar" / "model_orientations.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    
    orientations = {}
    if settings_path.exists():
        try:
            with open(settings_path) as f:
                orientations = json.load(f)
        except (json.JSONDecodeError, OSError):
            orientations = {}
    
    # Save orientation for this model (use filename as key)
    model_key = Path(model_path).name
    orientations[model_key] = {
        'pitch': math.degrees(pitch),
        'yaw': math.degrees(yaw),
        'roll': math.degrees(roll)
    }
    
    # Write back
    with open(settings_path, 'w') as f:
        json.dump(orientations, f, indent=2)
    
    parent.avatar_status.setText(f"Orientation saved for {model_key}")
    parent.avatar_status.setStyleSheet("color: #a6e3a1;")


def _load_model_orientation(parent, model_path: str):
    """Load saved orientation for a model."""
    import json
    import math
    from pathlib import Path
    
    settings_path = Path(CONFIG.get("data_dir", "data")) / "avatar" / "model_orientations.json"
    if not settings_path.exists():
        return None
    
    try:
        with open(settings_path) as f:
            orientations = json.load(f)
        
        model_key = Path(model_path).name
        if model_key in orientations:
            data = orientations[model_key]
            return {
                'pitch': math.radians(data.get('pitch', 0)),
                'yaw': math.radians(data.get('yaw', 0)),
                'roll': math.radians(data.get('roll', 0))
            }
    except (json.JSONDecodeError, OSError):
        pass
    
    return None


def _reset_preview(parent):
    """Reset 3D preview to defaults."""
    if parent.avatar_preview_3d:
        parent.avatar_preview_3d.reset_all()
        
        # Reset UI controls
        if hasattr(parent, 'wireframe_checkbox'):
            parent.wireframe_checkbox.setChecked(False)
        if hasattr(parent, 'show_grid_checkbox'):
            parent.show_grid_checkbox.setChecked(True)
        if hasattr(parent, 'light_slider'):
            parent.light_slider.setValue(100)
        if hasattr(parent, 'ambient_slider'):
            parent.ambient_slider.setValue(15)
    
    parent.avatar_status.setText("Preview reset to defaults")
    parent.avatar_status.setStyleSheet("color: #a6e3a1;")


def _reset_overlay(parent):
    """Reset desktop overlay to defaults."""
    if parent._overlay and parent._overlay.isVisible():
        parent._overlay.move(100, 100)
        parent._overlay._size = 300
        # DON'T call setFixedSize - let _update_scaled_pixmap handle window size
        parent._overlay._update_scaled_pixmap()
        
        parent.avatar_status.setText("Desktop overlay reset")
        parent.avatar_status.setStyleSheet("color: #a6e3a1;")
    else:
        parent.avatar_status.setText("No overlay active")
        parent.avatar_status.setStyleSheet("color: #fab387;")


def _reset_all_avatar(parent):
    """Reset all avatar settings to defaults."""
    # Reset 3D preview
    _reset_preview(parent)
    
    # Reset overlay
    if parent._overlay:
        parent._overlay.move(100, 100)
        parent._overlay._size = 300
        # DON'T call setFixedSize - let _update_scaled_pixmap handle window size
        parent._overlay._update_scaled_pixmap()
    
    # Reset colors
    parent._current_colors = {
        "primary": "#6366f1",
        "secondary": "#8b5cf6",
        "accent": "#10b981"
    }
    if hasattr(parent, 'primary_color_btn'):
        parent.primary_color_btn.setStyleSheet("background: #6366f1; color: white;")
    if hasattr(parent, 'secondary_color_btn'):
        parent.secondary_color_btn.setStyleSheet("background: #8b5cf6; color: white;")
    if hasattr(parent, 'accent_color_btn'):
        parent.accent_color_btn.setStyleSheet("background: #10b981; color: white;")
    if hasattr(parent, 'color_preset_combo'):
        parent.color_preset_combo.setCurrentText("Default")
    
    # Reset expression
    parent.current_expression = "neutral"
    
    parent.avatar_status.setText("All avatar settings reset to defaults")
    parent.avatar_status.setStyleSheet("color: #a6e3a1;")


def _refresh_avatar(parent):
    """Refresh the selected avatar (reload from file) and update the list."""
    # Remember current selection
    current_data = parent.avatar_combo.currentData()
    current_index = parent.avatar_combo.currentIndex()
    
    # Refresh the list
    _refresh_list(parent)
    
    # Restore selection if it still exists
    if current_data:
        file_type, path_str = current_data
        for i in range(parent.avatar_combo.count()):
            data = parent.avatar_combo.itemData(i)
            if data and data[1] == path_str:
                parent.avatar_combo.setCurrentIndex(i)
                # Trigger reload of the avatar
                _on_avatar_selected(parent)
                parent.avatar_status.setText(f"Refreshed: {Path(path_str).name}")
                parent.avatar_status.setStyleSheet("color: #a6e3a1;")
                return
    
    # If no selection or selection not found, just show list refreshed
    parent.avatar_status.setText("Avatar list refreshed")
    parent.avatar_status.setStyleSheet("color: #a6e3a1;")


def _refresh_list(parent, preserve_selection=False):
    """Refresh avatar list - scans all subdirectories too."""
    # Remember current selection if preserving
    previous_data = parent.avatar_combo.currentData() if preserve_selection else None
    
    parent.avatar_combo.blockSignals(True)
    parent.avatar_combo.clear()
    parent.avatar_combo.addItem("-- Select Avatar --", None)
    
    # System config files to SKIP (these are NOT avatars)
    SYSTEM_CONFIG_FILES = {
        "avatar_settings.json",
        "avatar_registry.json", 
        "ai_avatar_state.json",
        "settings.json",
        "registry.json",
        "state.json",
        "config.json",
    }
    
    # JSON avatar configs (but skip system files)
    if AVATAR_CONFIG_DIR.exists():
        for f in sorted(AVATAR_CONFIG_DIR.glob("*.json")):
            # Skip system config files
            if f.name.lower() in SYSTEM_CONFIG_FILES:
                continue
            cfg = _load_json(f)
            # Only add if it looks like an avatar config (has image_path or model_path)
            if not cfg.get("image_path") and not cfg.get("model_path") and not cfg.get("type"):
                continue
            is_3d = cfg.get("type") == "3d" or "model_path" in cfg
            suffix = " (3D)" if is_3d else ""
            parent.avatar_combo.addItem(f"{f.stem}{suffix}", ("config", str(f)))
    
    # Direct images
    if AVATAR_IMAGES_DIR.exists():
        for f in sorted(AVATAR_IMAGES_DIR.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                parent.avatar_combo.addItem(f.name, ("image", str(f)))
    
    # 3D models - scan direct files AND subdirectories
    if AVATAR_MODELS_DIR.exists():
        # Direct model files
        for f in sorted(AVATAR_MODELS_DIR.iterdir()):
            if f.is_file() and f.suffix.lower() in MODEL_3D_EXTENSIONS:
                parent.avatar_combo.addItem(f"{f.name} (3D)", ("model", str(f)))
        
        # Subdirectories containing models (e.g., glados/, rurune/)
        for subdir in sorted(AVATAR_MODELS_DIR.iterdir()):
            if subdir.is_dir():
                # Look for scene.gltf or scene.glb first (common format)
                scene_gltf = subdir / "scene.gltf"
                scene_glb = subdir / "scene.glb"
                
                if scene_gltf.exists():
                    parent.avatar_combo.addItem(f"{subdir.name} (3D)", ("model", str(scene_gltf)))
                elif scene_glb.exists():
                    parent.avatar_combo.addItem(f"{subdir.name} (3D)", ("model", str(scene_glb)))
                else:
                    # Look for any model file in subdirectory
                    for f in sorted(subdir.glob("*")):
                        if f.suffix.lower() in MODEL_3D_EXTENSIONS:
                            parent.avatar_combo.addItem(f"{subdir.name}/{f.name} (3D)", ("model", str(f)))
                            break  # Only add first model found
    
    # Update status
    count = parent.avatar_combo.count() - 1  # Exclude "-- Select --"
    
    # Restore selection if found
    if previous_data:
        for i in range(parent.avatar_combo.count()):
            data = parent.avatar_combo.itemData(i)
            if data and data[1] == previous_data[1]:  # Compare file path
                parent.avatar_combo.setCurrentIndex(i)
                break
    
    parent.avatar_combo.blockSignals(False)
    
    parent.avatar_status.setText(f"Found {count} avatars")
    parent.avatar_status.setStyleSheet("color: #a6e3a1;" if count > 0 else "color: #6c7086;")


def _load_json(path: Path) -> dict:
    """Load JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.debug(f"Failed to load JSON from {path}: {e}")
        return {}


def _load_avatar_preview(parent):
    """Load the currently selected avatar into the preview pane (called on tab open)."""
    # Simply call _on_avatar_selected which handles all preview logic
    _on_avatar_selected(parent)


def _on_avatar_selected(parent):
    """Handle avatar selection from dropdown - show preview."""
    data = parent.avatar_combo.currentData()
    if not data:
        return
    
    file_type, path_str = data
    path = Path(path_str)
    
    if not path.exists():
        parent.avatar_status.setText(f"File not found: {path.name}")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")
        return
    
    parent._current_path = path
    
    # Determine what kind of file
    if file_type == "config":
        cfg = _load_json(path)
        if cfg.get("type") == "3d" or "model_path" in cfg:
            model_path = cfg.get("model_path", "")
            full_path = path.parent / model_path if not Path(model_path).is_absolute() else Path(model_path)
            if full_path.exists():
                parent._current_path = full_path
                parent._is_3d_model = True
                _preview_3d_model(parent, full_path)
        elif "image" in cfg:
            img_path = cfg["image"]
            full_path = path.parent / img_path if not Path(img_path).is_absolute() else Path(img_path)
            if full_path.exists():
                parent._current_path = full_path
                parent._is_3d_model = False
                _preview_image(parent, full_path)
            if "expressions" in cfg:
                parent.avatar_expressions = cfg["expressions"]
    elif file_type == "image":
        parent._is_3d_model = False
        _preview_image(parent, path)
    elif file_type == "model":
        parent._is_3d_model = True
        _preview_3d_model(parent, path)
    
    parent.avatar_status.setText(f"Selected: {path.name} - Click 'Apply Avatar' to load")
    parent.avatar_status.setStyleSheet("color: #fab387;")


def _preview_image(parent, path: Path):
    """Preview a 2D image."""
    pixmap = QPixmap(str(path))
    if not pixmap.isNull():
        parent.avatar_preview_2d.set_avatar(pixmap)
        
        # Enable 3D checkbox option only for 3D models
        if parent.use_3d_render_checkbox:
            parent.use_3d_render_checkbox.setEnabled(False)
            parent.use_3d_render_checkbox.setChecked(False)
        
        # Suggest parallax effect for 2D images
        parent.avatar_status.setText(
            f"Loaded: {path.name} - Try '2.5D Parallax' for depth effect!"
        )
        parent.avatar_status.setStyleSheet("color: #f5c2e7;")


def _preview_3d_model(parent, path: Path):
    """Preview a 3D model - auto-enable 3D rendering and load into viewer."""
    # Auto-enable 3D rendering when loading a 3D model
    if parent.use_3d_render_checkbox and HAS_OPENGL and HAS_TRIMESH:
        if not parent.use_3d_render_checkbox.isChecked():
            parent.use_3d_render_checkbox.setChecked(True)  # This triggers _toggle_3d_render
        
        # Load directly into 3D viewer
        if parent.avatar_preview_3d:
            try:
                parent.avatar_preview_3d.load_model(str(path))
                parent.avatar_status.setText(f"Loaded 3D model: {path.name}")
                parent.avatar_status.setStyleSheet("color: #a6e3a1;")
                return
            except Exception as e:
                print(f"Error loading 3D model into viewer: {e}")
    
    # Fallback - create a preview thumbnail using trimesh
    if HAS_TRIMESH:
        try:
            scene = trimesh.load(str(path))
            
            # Render to image
            if hasattr(scene, 'geometry') and scene.geometry:
                # Get scene with all geometry
                png_data = scene.save_image(resolution=[256, 256])
                if png_data:
                    img = QImage()
                    img.loadFromData(png_data)
                    pixmap = QPixmap.fromImage(img)
                    parent.avatar_preview_2d.set_avatar(pixmap)
                    return
            elif hasattr(scene, 'vertices'):
                # Single mesh - create scene and render
                render_scene = trimesh.Scene(scene)
                png_data = render_scene.save_image(resolution=[256, 256])
                if png_data:
                    img = QImage()
                    img.loadFromData(png_data)
                    pixmap = QPixmap.fromImage(img)
                    parent.avatar_preview_2d.set_avatar(pixmap)
                    return
        except Exception as e:
            print(f"Error rendering 3D preview: {e}")
    
    # Fallback - create info card
    _create_model_info_card(parent, path)


def _create_model_info_card(parent, path: Path):
    """Create an info card pixmap for 3D model with clear instructions."""
    size = 256
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor("#1e1e2e"))
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    
    # Model icon
    painter.setPen(QColor("#89b4fa"))
    font = painter.font()
    font.setPointSize(36)
    painter.setFont(font)
    painter.drawText(0, 30, size, 50, Qt_AlignCenter, "[3D]")
    
    # "3D Model Loaded" label
    font.setPointSize(12)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor("#a6e3a1"))
    painter.drawText(0, 80, size, 25, Qt_AlignCenter, "3D Model Loaded!")
    
    # File name
    font.setPointSize(10)
    font.setBold(False)
    painter.setFont(font)
    painter.setPen(QColor("#cdd6f4"))
    name = path.name
    if len(name) > 25:
        name = name[:22] + "..."
    painter.drawText(0, 110, size, 20, Qt_AlignCenter, name)
    
    # File size
    size_kb = path.stat().st_size / 1024
    painter.setPen(QColor("#6c7086"))
    if size_kb > 1024:
        size_str = f"{size_kb/1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"
    painter.drawText(0, 135, size, 20, Qt_AlignCenter, size_str)
    
    # Clear instructions based on state
    font.setPointSize(9)
    font.setBold(True)
    painter.setFont(font)
    
    if not HAS_OPENGL or not HAS_TRIMESH:
        # Libraries missing
        painter.setPen(QColor("#f38ba8"))
        painter.drawText(0, 165, size, 20, Qt_AlignCenter, "Missing 3D Libraries!")
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor("#fab387"))
        painter.drawText(0, 185, size, 60, Qt_AlignCenter, 
            "Run in terminal:\npip install trimesh PyOpenGL")
    elif parent.use_3d_render_checkbox and not parent.use_3d_render_checkbox.isChecked():
        # Checkbox exists but unchecked - give VERY clear location
        painter.setPen(QColor("#a6e3a1"))
        painter.drawText(0, 160, size, 20, Qt_AlignCenter, "To view this 3D model:")
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor("#89b4fa"))
        # Draw arrow pointing up
        painter.drawText(0, 178, size, 20, Qt_AlignCenter, "^ ^ ^ ^ ^")
        painter.setPen(QColor("#fab387"))
        painter.drawText(0, 198, size, 50, Qt_AlignCenter,
            "Check the BLUE checkbox\n'Enable 3D Rendering'\nright above this preview")
    else:
        # Should be showing but failed
        painter.setPen(QColor("#fab387"))
        painter.drawText(0, 175, size, 50, Qt_AlignCenter,
            "Click 'Apply Avatar'\nto load the model")
    
    painter.end()
    parent.avatar_preview_2d.set_avatar(pixmap)
    
    # Auto-enable 3D rendering if checkbox exists and model is 3D
    if parent.use_3d_render_checkbox and not parent.use_3d_render_checkbox.isChecked():
        # Schedule auto-enable with a delay so UI is ready
        QTimer.singleShot(100, lambda: _auto_enable_3d_for_model(parent, path))


def _auto_enable_3d_for_model(parent, path: Path):
    """Auto-enable 3D rendering when a 3D model is loaded."""
    try:
        if parent.use_3d_render_checkbox and not parent.use_3d_render_checkbox.isChecked():
            # Enable the checkbox - this triggers _toggle_3d_render
            parent.use_3d_render_checkbox.setChecked(True)
            
            # Update status to let user know what happened
            parent.avatar_status.setText(f"3D enabled for: {path.name}")
            parent.avatar_status.setStyleSheet("color: #a6e3a1;")
            
            # Load the model into the 3D viewer
            if parent.avatar_preview_3d and hasattr(parent, '_current_path'):
                try:
                    parent.avatar_preview_3d.load_model(str(path))
                except Exception as e:
                    logger.warning(f"Could not load 3D model: {e}")
    except Exception as e:
        logger.warning(f"Auto-enable 3D failed: {e}")


def _apply_avatar(parent):
    """Apply the selected avatar - fully load it."""
    if not parent._current_path or not parent._current_path.exists():
        parent.avatar_status.setText("Select an avatar first!")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")
        return
    
    path = parent._current_path
    
    # Save current avatar to persistence for AI awareness
    try:
        from ....avatar.persistence import (
            save_avatar_settings,
            write_avatar_state_for_ai,
        )
        save_avatar_settings(current_avatar=str(path))
        write_avatar_state_for_ai()
    except Exception as e:
        print(f"[Avatar] Could not save current avatar: {e}")
    
    # Load into the backend controller
    if parent._is_3d_model:
        parent._avatar_controller.load_model(str(path))
        parent.avatar_status.setText(f"Loaded 3D: {path.name}")
        
        # If 3D rendering is enabled, load into GL widget
        if parent._using_3d_render and parent.avatar_preview_3d:
            parent.avatar_preview_3d.load_model(str(path))
            
            # Load saved orientation for this model
            saved_orientation = _load_model_orientation(parent, str(path))
            if saved_orientation:
                parent.avatar_preview_3d.model_pitch = saved_orientation['pitch']
                parent.avatar_preview_3d.model_yaw = saved_orientation['yaw']
                parent.avatar_preview_3d.model_roll = saved_orientation['roll']
                
                # Update sliders to match
                import math
                if hasattr(parent, 'pitch_slider'):
                    parent.pitch_slider.setValue(int(math.degrees(saved_orientation['pitch'])))
                if hasattr(parent, 'yaw_slider'):
                    yaw_deg = int(math.degrees(saved_orientation['yaw']))
                    parent.yaw_slider.setValue(yaw_deg)
                    # Update facing direction indicator
                    _update_facing_direction_label(parent, yaw_deg)
                if hasattr(parent, 'roll_slider'):
                    parent.roll_slider.setValue(int(math.degrees(saved_orientation['roll'])))
                
                parent.avatar_preview_3d.update()
    else:
        # Set the model path in the controller's config (not appearance which doesn't exist)
        parent._avatar_controller.config.model_path = str(path)
        parent.avatar_status.setText(f"Loaded: {path.name}")
    
    parent.avatar_status.setStyleSheet("color: #a6e3a1;")
    
    # Update overlay if visible (pass original, overlay handles scaling)
    if parent._overlay and parent._overlay.isVisible():
        # Update path first so per-avatar settings are used
        if parent._current_path:
            parent._overlay.set_avatar_path(str(parent._current_path))
        pixmap = parent.avatar_preview_2d.original_pixmap
        if pixmap:
            parent._overlay.set_avatar(pixmap)


def _load_avatar_file(parent):
    """Open file dialog to load avatar."""
    all_exts = " ".join(f"*{ext}" for ext in ALL_AVATAR_EXTENSIONS)
    img_exts = " ".join(f"*{ext}" for ext in IMAGE_EXTENSIONS)
    model_exts = " ".join(f"*{ext}" for ext in MODEL_3D_EXTENSIONS)
    
    path, _ = QFileDialog.getOpenFileName(
        parent,
        "Load Avatar",
        str(AVATAR_CONFIG_DIR),
        f"All Avatars ({all_exts});;Images ({img_exts});;3D Models ({model_exts});;All Files (*)"
    )
    
    if not path:
        return
    
    path = Path(path)
    parent._current_path = path
    
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        parent._is_3d_model = False
        _preview_image(parent, path)
    elif path.suffix.lower() in MODEL_3D_EXTENSIONS:
        parent._is_3d_model = True
        _preview_3d_model(parent, path)
    elif path.suffix.lower() == ".json":
        # Add to combo and trigger selection
        parent.avatar_combo.addItem(f"[cfg] {path.stem}", ("config", str(path)))
        parent.avatar_combo.setCurrentIndex(parent.avatar_combo.count() - 1)
        return
    
    parent.avatar_status.setText(f"Selected: {path.name} - Click 'Apply Avatar' to load")
    parent.avatar_status.setStyleSheet("color: #fab387;")


def set_avatar_expression(parent, expression: str):
    """Set avatar expression (called by AI)."""
    if not hasattr(parent, '_avatar_controller'):
        return
    
    parent._avatar_controller.set_expression(expression)
    parent.current_expression = expression
    
    if expression in parent.avatar_expressions:
        img_path = parent.avatar_expressions[expression]
        if not Path(img_path).is_absolute():
            img_path = AVATAR_CONFIG_DIR / img_path
        path = Path(img_path)
        if path.exists():
            _preview_image(parent, path)
            _apply_avatar(parent)


def load_avatar_config(config_path: Path) -> dict:
    """Load avatar config (compatibility)."""
    return _load_json(config_path)


# ============================================================================
# Avatar Gallery / Import Functions
# ============================================================================

def _browse_avatars(parent):
    """Open the avatar picker dialog."""
    try:
        from ....avatar.avatar_dialogs import AvatarPickerDialog
        
        picker = AvatarPickerDialog(parent)
        picker.avatar_selected.connect(lambda info: _apply_selected_avatar(parent, info))
        picker.exec_()
    except ImportError as e:
        parent.avatar_status.setText(f"Gallery not available: {e}")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")
    except Exception as e:
        parent.avatar_status.setText(f"Error: {e}")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")


def _import_avatar(parent):
    """Open the avatar import wizard."""
    try:
        from ....avatar.avatar_dialogs import AvatarImportWizard
        
        wizard = AvatarImportWizard(parent)
        if wizard.exec_():
            parent.avatar_status.setText("Avatar imported successfully!")
            parent.avatar_status.setStyleSheet("color: #a6e3a1;")
            # Refresh combo box
            _refresh_avatar(parent)
    except ImportError as e:
        parent.avatar_status.setText(f"Import wizard not available: {e}")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")
    except Exception as e:
        parent.avatar_status.setText(f"Error: {e}")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")


def _import_multiple_avatars(parent):
    """Import multiple avatar files at once."""
    all_exts = " ".join(f"*{ext}" for ext in ALL_AVATAR_EXTENSIONS)
    img_exts = " ".join(f"*{ext}" for ext in IMAGE_EXTENSIONS)
    model_exts = " ".join(f"*{ext}" for ext in MODEL_3D_EXTENSIONS)
    
    paths, _ = QFileDialog.getOpenFileNames(
        parent,
        "Import Multiple Avatars",
        str(Path.home() / "Downloads"),
        f"All Avatars ({all_exts});;Images ({img_exts});;3D Models ({model_exts});;All Files (*)"
    )
    
    if not paths:
        return
    
    imported = 0
    for path_str in paths:
        path = Path(path_str)
        try:
            # Copy to appropriate directory
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                dest = AVATAR_IMAGES_DIR / path.name
            else:
                dest = AVATAR_MODELS_DIR / path.name
            
            # Don't overwrite existing files
            if dest.exists():
                base = dest.stem
                suffix = dest.suffix
                counter = 1
                while dest.exists():
                    dest = dest.parent / f"{base}_{counter}{suffix}"
                    counter += 1
            
            import shutil
            shutil.copy2(path, dest)
            imported += 1
        except Exception as e:
            logger.error(f"Failed to import {path.name}: {e}")
    
    if imported > 0:
        parent.avatar_status.setText(f"Imported {imported} avatar(s)!")
        parent.avatar_status.setStyleSheet("color: #a6e3a1;")
        _refresh_avatar(parent)
    else:
        parent.avatar_status.setText("No avatars imported")
        parent.avatar_status.setStyleSheet("color: #fab387;")


def _import_from_downloads(parent):
    """Quick import from Downloads folder - shows files available there."""
    downloads_path = Path.home() / "Downloads"
    
    if not downloads_path.exists():
        parent.avatar_status.setText("Downloads folder not found")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")
        return
    
    # Find avatar files in Downloads
    avatar_files = []
    zip_files = []
    
    for ext in ALL_AVATAR_EXTENSIONS:
        avatar_files.extend(downloads_path.glob(f"*{ext}"))
    
    # Also find ZIP files that might contain avatars
    zip_files.extend(downloads_path.glob("*.zip"))
    
    if not avatar_files and not zip_files:
        parent.avatar_status.setText("No avatar files found in Downloads")
        parent.avatar_status.setStyleSheet("color: #fab387;")
        # Open file dialog to Downloads anyway
        _import_multiple_avatars(parent)
        return
    
    # Show dialog to select which files to import
    from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QListWidget, QListWidgetItem
    
    dialog = QDialog(parent)
    dialog.setWindowTitle("Import from Downloads")
    dialog.setMinimumSize(400, 300)
    layout = QVBoxLayout(dialog)
    
    label = QLabel(f"Found {len(avatar_files)} avatar file(s) and {len(zip_files)} ZIP file(s) in Downloads:")
    layout.addWidget(label)
    
    file_list = QListWidget()
    file_list.setSelectionMode(QListWidget.MultiSelection)
    
    for path in avatar_files:
        item = QListWidgetItem(f"[Avatar] {path.name}")
        item.setData(Qt.UserRole, ("avatar", str(path)))
        file_list.addItem(item)
        item.setSelected(True)  # Select all by default
    
    for path in zip_files:
        item = QListWidgetItem(f"[ZIP] {path.name}")
        item.setData(Qt.UserRole, ("zip", str(path)))
        file_list.addItem(item)
        item.setSelected(True)
    
    layout.addWidget(file_list)
    
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    
    if dialog.exec_() != QDialog.Accepted:
        return
    
    # Import selected files
    imported = 0
    extracted = 0
    
    for item in file_list.selectedItems():
        file_type, file_path = item.data(Qt.UserRole)
        path = Path(file_path)
        
        try:
            if file_type == "avatar":
                # Copy avatar file
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    dest = AVATAR_IMAGES_DIR / path.name
                else:
                    dest = AVATAR_MODELS_DIR / path.name
                
                if not dest.exists():
                    import shutil
                    shutil.copy2(path, dest)
                    imported += 1
                else:
                    # File already exists
                    imported += 1
                    
            elif file_type == "zip":
                # Extract ZIP file
                count = _extract_avatar_zip(path, AVATAR_MODELS_DIR)
                extracted += count
                
        except Exception as e:
            logger.error(f"Failed to import {path.name}: {e}")
    
    status_parts = []
    if imported > 0:
        status_parts.append(f"{imported} file(s) imported")
    if extracted > 0:
        status_parts.append(f"{extracted} file(s) extracted from ZIPs")
    
    if status_parts:
        parent.avatar_status.setText(", ".join(status_parts) + "!")
        parent.avatar_status.setStyleSheet("color: #a6e3a1;")
        _refresh_avatar(parent)
    else:
        parent.avatar_status.setText("No files imported")
        parent.avatar_status.setStyleSheet("color: #fab387;")


def _import_zip_archive(parent):
    """Import and extract avatar ZIP/archive files."""
    paths, _ = QFileDialog.getOpenFileNames(
        parent,
        "Import Avatar Archives",
        str(Path.home() / "Downloads"),
        "Archives (*.zip *.7z *.tar.gz *.tar);;ZIP Files (*.zip);;All Files (*)"
    )
    
    if not paths:
        return
    
    total_extracted = 0
    
    for path_str in paths:
        path = Path(path_str)
        try:
            count = _extract_avatar_zip(path, AVATAR_MODELS_DIR)
            total_extracted += count
        except Exception as e:
            logger.error(f"Failed to extract {path.name}: {e}")
            parent.avatar_status.setText(f"Error extracting {path.name}: {e}")
            parent.avatar_status.setStyleSheet("color: #f38ba8;")
    
    if total_extracted > 0:
        parent.avatar_status.setText(f"Extracted {total_extracted} avatar file(s)!")
        parent.avatar_status.setStyleSheet("color: #a6e3a1;")
        _refresh_avatar(parent)
    else:
        parent.avatar_status.setText("No avatar files found in archive(s)")
        parent.avatar_status.setStyleSheet("color: #fab387;")


def _extract_avatar_zip(zip_path: Path, dest_dir: Path) -> int:
    """
    Extract avatar files from a ZIP archive.
    
    Returns the number of avatar files extracted.
    """
    import zipfile
    import shutil
    
    extracted_count = 0
    
    if not zip_path.suffix.lower() == '.zip':
        logger.warning(f"Only ZIP files supported, got: {zip_path.suffix}")
        return 0
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get list of files in the archive
            namelist = zf.namelist()
            
            # Find avatar files (images and 3D models)
            avatar_files = []
            for name in namelist:
                name_lower = name.lower()
                # Skip directories
                if name.endswith('/'):
                    continue
                # Skip macOS resource forks
                if '__MACOSX' in name or name.startswith('.'):
                    continue
                
                # Check if it's an avatar file
                for ext in ALL_AVATAR_EXTENSIONS:
                    if name_lower.endswith(ext):
                        avatar_files.append(name)
                        break
            
            if not avatar_files:
                logger.info(f"No avatar files found in {zip_path.name}")
                return 0
            
            # Create a subfolder for this archive (to keep related files together)
            archive_name = zip_path.stem
            archive_dest = dest_dir / archive_name
            archive_dest.mkdir(parents=True, exist_ok=True)
            
            # Extract avatar files
            for name in avatar_files:
                # Get just the filename, strip any directory paths
                filename = Path(name).name
                dest_path = archive_dest / filename
                
                # Extract the file
                with zf.open(name) as src:
                    with open(dest_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                
                extracted_count += 1
                logger.info(f"Extracted: {filename} -> {dest_path}")
            
            # Also extract any texture files that might be associated with 3D models
            texture_exts = {'.png', '.jpg', '.jpeg', '.tga', '.bmp', '.dds'}
            for name in namelist:
                name_lower = name.lower()
                if name.endswith('/') or '__MACOSX' in name:
                    continue
                
                for ext in texture_exts:
                    if name_lower.endswith(ext) and name not in avatar_files:
                        # This is a texture, extract it too
                        filename = Path(name).name
                        dest_path = archive_dest / filename
                        
                        with zf.open(name) as src:
                            with open(dest_path, 'wb') as dst:
                                shutil.copyfileobj(src, dst)
                        
                        logger.info(f"Extracted texture: {filename}")
                        break
            
            # If only one avatar file, also copy it to the main models dir for easy access
            if extracted_count == 1:
                first_file = archive_dest / Path(avatar_files[0]).name
                if first_file.exists():
                    main_dest = dest_dir / first_file.name
                    if not main_dest.exists():
                        shutil.copy2(first_file, main_dest)
                        logger.info(f"Also copied to: {main_dest}")
    
    except zipfile.BadZipFile:
        logger.error(f"Invalid ZIP file: {zip_path}")
        raise ValueError(f"Invalid ZIP file: {zip_path.name}")
    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {e}")
        raise
    
    return extracted_count


def _generate_sample_avatars(parent):
    """Generate sample avatars."""
    try:
        from ....avatar.sample_avatars import generate_sample_avatars
        
        parent.avatar_status.setText("Generating samples...")
        parent.avatar_status.setStyleSheet("color: #89dceb;")
        QApplication.processEvents()
        
        avatars = generate_sample_avatars()
        
        parent.avatar_status.setText(f"Generated {len(avatars)} sample avatars!")
        parent.avatar_status.setStyleSheet("color: #a6e3a1;")
        
        # Refresh combo
        _refresh_avatar(parent)
        
    except ImportError as e:
        parent.avatar_status.setText(f"Sample generator not available: {e}")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")
    except Exception as e:
        parent.avatar_status.setText(f"Error generating: {e}")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")


def _apply_selected_avatar(parent, avatar_info: dict):
    """Apply an avatar from the picker."""
    path = Path(avatar_info.get("path", ""))
    if not path.exists():
        parent.avatar_status.setText("Avatar path not found")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")
        return
    
    # Check for neutral/default expression
    for img_name in ["neutral.png", "base.png", "default.png"]:
        img_path = path / img_name
        if img_path.exists():
            parent._current_path = img_path
            parent._is_3d_model = False
            _preview_image(parent, img_path)
            _apply_avatar(parent)
            parent.avatar_status.setText(f"Loaded: {avatar_info.get('name', path.name)}")
            parent.avatar_status.setStyleSheet("color: #a6e3a1;")
            return
    
    # Check for any image
    for ext in IMAGE_EXTENSIONS:
        for img_path in path.glob(f"*{ext}"):
            parent._current_path = img_path
            parent._is_3d_model = False
            _preview_image(parent, img_path)
            _apply_avatar(parent)
            parent.avatar_status.setText(f"Loaded: {avatar_info.get('name', path.name)}")
            parent.avatar_status.setStyleSheet("color: #a6e3a1;")
            return
    
    parent.avatar_status.setText("No image found in avatar")
    parent.avatar_status.setStyleSheet("color: #fab387;")

