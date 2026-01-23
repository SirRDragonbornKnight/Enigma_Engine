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
"""
# type: ignore[attr-defined]
# PyQt5 type stubs are incomplete; runtime works correctly

from pathlib import Path
from typing import Optional, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QComboBox, QCheckBox, QFrame, QSizePolicy,
    QApplication, QOpenGLWidget, QMessageBox, QGroupBox,
    QSlider, QColorDialog, QGridLayout, QScrollArea, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer, QPoint, pyqtSignal, QSize, QByteArray
from PyQt5.QtGui import QPixmap, QPainter, QColor, QCursor, QImage, QMouseEvent, QWheelEvent

# Optional SVG support - not all PyQt5 installs have it
try:
    from PyQt5.QtSvg import QSvgWidget, QSvgRenderer
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
Qt_LeftButton: Any = getattr(Qt, 'LeftButton', 0x00000001)
Qt_KeepAspectRatio: Any = getattr(Qt, 'KeepAspectRatio', 1)
Qt_SmoothTransformation: Any = getattr(Qt, 'SmoothTransformation', 1)
Qt_AlignCenter: Any = getattr(Qt, 'AlignCenter', 0x0084)
Qt_transparent: Any = getattr(Qt, 'transparent', QColor(0, 0, 0, 0))
Qt_NoPen: Any = getattr(Qt, 'NoPen', 0)
Qt_NoBrush: Any = getattr(Qt, 'NoBrush', 0)
Qt_OpenHandCursor: Any = getattr(Qt, 'OpenHandCursor', 17)
Qt_ClosedHandCursor: Any = getattr(Qt, 'ClosedHandCursor', 18)
Qt_ArrowCursor: Any = getattr(Qt, 'ArrowCursor', 0)
import json
import os
import time

from ....config import CONFIG
from ....avatar import get_avatar, AvatarState
from ....avatar.renderers.default_sprites import generate_sprite, SPRITE_TEMPLATES
from ....avatar.customizer import AvatarCustomizer

# Try importing 3D libraries
HAS_TRIMESH = False
HAS_OPENGL = False
trimesh = None
np = None

try:
    import trimesh as _trimesh
    import numpy as _np
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
            with open(settings_path, 'r') as f:
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
        except (json.JSONDecodeError, IOError) as e:
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
        self.last_pos = None
        self.is_panning = False
        self.setCursor(QCursor(Qt_ArrowCursor))  # Keep normal cursor
        event.accept()
        
    def wheelEvent(self, event):
        """Zoom with scroll wheel."""
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
    - Blue border shows when resize mode is ON
    """
    
    closed = pyqtSignal()
    
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
        
        self._size = 300
        self.setFixedSize(self._size, self._size)
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
        
        # Enable mouse tracking for resize cursor feedback
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt_ArrowCursor))
    
    def set_avatar_path(self, path: str):
        """Set the current avatar path and load per-avatar settings."""
        self._avatar_path = path
        # Load per-avatar size and position
        try:
            from ....avatar.persistence import load_avatar_settings
            settings = load_avatar_settings()
            self._size = settings.get_size_for_avatar(path)
            self.setFixedSize(self._size, self._size)
            x, y = settings.get_position_for_avatar(path)
            self.move(x, y)
            self._update_scaled_pixmap()
        except Exception:
            pass
        
    def set_avatar(self, pixmap: QPixmap):
        """Set avatar image."""
        self._original_pixmap = pixmap
        self._update_scaled_pixmap()
    
    def _get_edge_at_pos(self, pos):
        """Get which edge the mouse is near (for resize cursor)."""
        if not getattr(self, '_resize_enabled', False):
            return None
        
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
        """Update scaled pixmap to match current size and rotation, then resize window to wrap tightly."""
        if self._original_pixmap and not self._original_pixmap.isNull():
            # Scale to fit within the requested size (with small margin for border)
            border_margin = 8  # Space for the border
            max_dim = max(50, self._size - border_margin)  # Ensure positive
            
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
            
            # TIGHT WRAP: Resize window to exactly fit pixmap (plus border margin)
            if self.pixmap and not self.pixmap.isNull():
                new_width = self.pixmap.width() + border_margin
                new_height = self.pixmap.height() + border_margin
                # Force resize window to wrap tightly - use setMinimum/Maximum to allow any size
                self.setMinimumSize(1, 1)  # Allow shrinking
                self.setMaximumSize(16777215, 16777215)  # Allow growing (Qt max)
                self.resize(new_width, new_height)
                self.setFixedSize(new_width, new_height)  # Lock at this size
        self.update()
        
    def paintEvent(self, a0):
        """Draw avatar with border that always wraps tightly around the image."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        if self.pixmap:
            # Center pixmap in window (should be nearly edge-to-edge with tight wrap)
            x = (self.width() - self.pixmap.width()) // 2
            y = (self.height() - self.pixmap.height()) // 2
            
            # Draw a subtle circular background/glow FIRST (behind everything)
            painter.setPen(Qt_NoPen)
            painter.setBrush(QColor(30, 30, 46, 80))  # Semi-transparent dark
            painter.drawEllipse(x - 3, y - 3, self.pixmap.width() + 6, self.pixmap.height() + 6)
            
            # Draw the avatar
            painter.drawPixmap(x, y, self.pixmap)
            
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
        
    def mousePressEvent(self, a0):  # type: ignore
        """Start drag to move, resize, or rotate (Shift+drag when enabled)."""
        if a0.button() == Qt_LeftButton:
            try:
                global_pos = a0.globalPosition().toPoint()
                local_pos = a0.position().toPoint()
            except AttributeError:
                global_pos = a0.globalPos()
                local_pos = a0.pos()
            
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
        a0.accept()
            
    def mouseMoveEvent(self, a0):  # type: ignore
        """Handle drag to move, resize, or rotate."""
        try:
            global_pos = a0.globalPosition().toPoint()
            local_pos = a0.position().toPoint()
        except AttributeError:
            global_pos = a0.globalPos()
            local_pos = a0.pos()
        
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
            self.setFixedSize(self._size, self._size)
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
        """End drag or resize."""
        # Save position if we were dragging (per-avatar)
        if self._drag_pos is not None:
            try:
                from ....avatar.persistence import get_persistence, write_avatar_state_for_ai
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
                from ....avatar.persistence import get_persistence, write_avatar_state_for_ai
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
    
    def hideEvent(self, a0):
        """Save position and size when hidden."""
        super().hideEvent(a0)
        try:
            from ....avatar.persistence import save_avatar_settings, save_position, write_avatar_state_for_ai
            pos = self.pos()
            save_position(pos.x(), pos.y())
            save_avatar_settings(overlay_size=self._size)
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
            self.setFixedSize(self._size, self._size)
            self._update_scaled_pixmap()
            
            # Save size
            try:
                from ....avatar.persistence import save_avatar_settings, write_avatar_state_for_ai
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
            from ....avatar.persistence import save_avatar_settings, write_avatar_state_for_ai
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
            self.setFixedSize(self._size, self._size)
            self._update_scaled_pixmap()
            # Save size (per-avatar)
            try:
                from ....avatar.persistence import get_persistence, write_avatar_state_for_ai
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
        self.setFocusPolicy(Qt.NoFocus if hasattr(Qt, 'NoFocus') else 0x00)
        
        self._size = 250
        self.setFixedSize(self._size, self._size)
        self._resize_enabled = False  # Default OFF - user must enable via right-click
        
        # Edge-drag resize state
        self._resize_edge = None  # Which edge is being dragged
        self._resize_start_pos = None
        self._resize_start_size = None
        self._resize_start_geo = None
        self._edge_margin = 20  # Pixels from edge to trigger resize (larger for easier grabbing)
        
        # Rotation state for Shift+drag manual rotation
        self._rotate_start_x = None
        self._manual_yaw_offset = 0.0  # Manual yaw adjustment from user
        
        # Start at bottom right
        screen = QApplication.primaryScreen()
        if screen:
            screen_geo = screen.availableGeometry()
            self.move(screen_geo.right() - self._size - 50, screen_geo.bottom() - self._size - 50)
        else:
            self.move(100, 100)
        
        self._drag_pos = None
        self._model_path = None
        self._use_circular_mask = True
        
        # Adaptive Animator - works with ANY model
        self._animator = None
        try:
            from ....avatar.adaptive_animator import AdaptiveAnimator
            self._animator = AdaptiveAnimator()
            self._animator.on_transform_update(self._on_animator_transform)
        except ImportError as e:
            print(f"[Avatar3DOverlay] AdaptiveAnimator not available: {e}")
        
        # Idle animation state (fallback if no animator)
        self._idle_animation_enabled = True
        self._idle_timer = QTimer()
        self._idle_timer.timeout.connect(self._do_idle_animation)
        self._idle_phase = 0.0  # Animation phase (radians)
        self._idle_bob_amount = 0.02  # How much to bob up/down (fraction of zoom)
        self._idle_sway_amount = 0.3  # How much to sway left/right (degrees)
        self._idle_speed = 0.05  # Animation speed
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
            gl_layout.addWidget(self._gl_widget)
            self._apply_circular_mask()
        else:
            self._gl_widget = None
            placeholder = QLabel("3D not available")
            placeholder.setStyleSheet("color: white; background: rgba(30,30,46,150); border-radius: 50%;")
            placeholder.setAlignment(Qt_AlignCenter)
            gl_layout.addWidget(placeholder)
        
        main_layout.addWidget(self._gl_container)
        
        # Install event filter for dragging - this intercepts ALL mouse events
        if self._gl_widget:
            self._gl_widget.installEventFilter(self)
        
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt_OpenHandCursor))
    
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
        """Apply a circular mask."""
        if not self._use_circular_mask:
            return
        
        from PyQt5.QtGui import QRegion, QPainterPath
        from PyQt5.QtCore import QRectF
        
        path = QPainterPath()
        padding = 5
        path.addEllipse(QRectF(padding, padding, self._size - padding*2, self._size - padding*2))
        
        region = QRegion(path.toFillPolygon().toPolygon())
        if self._gl_widget:
            self._gl_widget.setMask(region)
    
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
                            from ....avatar.bone_control import get_bone_controller
                            from ....avatar import get_avatar
                            
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
                
                # Save base camera positions for idle animation
                self._base_pan_y = self._gl_widget.pan_y
                self._base_rotation_y = self._gl_widget.rotation_y
                
                # Start idle animation (fallback if no animator)
                if self._idle_animation_enabled and not self._animator:
                    self._start_idle_animation()
            
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
    
    def _get_edge_at_pos(self, pos):
        """Get which edge the mouse is near (for resize cursor)."""
        if not getattr(self, '_resize_enabled', False):
            return None
        
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
    
    def _do_resize(self, global_pos):
        """Perform edge-drag resize."""
        if not self._resize_edge or not self._resize_start_pos:
            return
        
        delta = global_pos - self._resize_start_pos
        new_size = self._resize_start_size
        
        if 'right' in self._resize_edge or 'bottom' in self._resize_edge:
            # Resize from right/bottom - increase size
            change = max(delta.x(), delta.y())
            new_size = max(100, min(500, self._resize_start_size + change))
        elif 'left' in self._resize_edge or 'top' in self._resize_edge:
            # Resize from left/top - decrease size and move
            change = max(-delta.x(), -delta.y())
            new_size = max(100, min(500, self._resize_start_size + change))
            # Adjust position to keep bottom-right corner fixed
            if self._resize_start_geo:
                size_diff = new_size - self._resize_start_size
                self.move(self._resize_start_geo.x() - size_diff, self._resize_start_geo.y() - size_diff)
        
        self._size = new_size
        self.setFixedSize(self._size, self._size)
        self._gl_container.setFixedSize(self._size, self._size)
        if self._gl_widget:
            self._gl_widget.setFixedSize(self._size, self._size)
            self._apply_circular_mask()
    
    def eventFilter(self, obj, event):
        """Handle mouse events for dragging and edge-resize - blocks rotation in desktop mode."""
        if obj == self._gl_widget:
            event_type = event.type()
            
            try:
                global_pos = event.globalPosition().toPoint()
                local_pos = event.position().toPoint()
            except AttributeError:
                global_pos = event.globalPos() if hasattr(event, 'globalPos') else None
                local_pos = event.pos() if hasattr(event, 'pos') else None
            
            # Block ALL mouse events that would cause rotation
            if event_type == event.MouseButtonPress:
                if event.button() == Qt_LeftButton and local_pos and global_pos:
                    # Check for Shift+drag (manual rotation mode) - only when resize/rotate enabled
                    modifiers = event.modifiers()
                    shift_held = bool(modifiers & (Qt.ShiftModifier if hasattr(Qt, 'ShiftModifier') else 0x02000000))
                    
                    if shift_held and getattr(self, '_resize_enabled', False):
                        # Start rotation drag (only when enabled)
                        self._rotate_start_x = global_pos.x()
                        self.setCursor(QCursor(Qt.SizeHorCursor if hasattr(Qt, 'SizeHorCursor') else 0))
                    else:
                        # Check if we're on an edge (for resize) - _get_edge_at_pos checks _resize_enabled
                        edge = self._get_edge_at_pos(local_pos)
                        if edge:
                            self._resize_edge = edge
                            self._resize_start_pos = global_pos
                            self._resize_start_size = self._size
                            self._resize_start_geo = self.geometry()
                        else:
                            # Normal drag
                            self._drag_pos = global_pos - self.pos()
                            self.setCursor(QCursor(Qt_ClosedHandCursor))
                return True  # Block event from reaching GL widget
                
            elif event_type == event.MouseMove:
                if global_pos:
                    # If rotating (Shift+drag)
                    if self._rotate_start_x is not None:
                        delta_x = global_pos.x() - self._rotate_start_x
                        self._manual_yaw_offset = (self._manual_yaw_offset + delta_x * 0.5) % 360
                        self._rotate_start_x = global_pos.x()
                        # Apply rotation to the 3D model
                        if self._gl_widget:
                            import math
                            self._gl_widget.model_yaw = math.radians(self._manual_yaw_offset)
                            self._gl_widget.update()
                        return True
                    
                    # If resizing
                    if self._resize_edge:
                        self._do_resize(global_pos)
                        return True
                    
                    # If dragging
                    if self._drag_pos is not None:
                        new_pos = global_pos - self._drag_pos
                        
                        # Keep avatar on virtual desktop (all monitors) - can drag across monitors
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
                            pass  # If virtual desktop fails, allow free movement
                        
                        self.move(new_pos)
                        return True
                    
                    # Update cursor based on position (only when resize enabled)
                    if local_pos:
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
                            self.setCursor(QCursor(Qt_OpenHandCursor))
                return True  # Block event from reaching GL widget
                
            elif event_type == event.MouseButtonRelease:
                if event.button() == Qt_LeftButton:
                    # Save position when drag ends
                    if self._drag_pos is not None:
                        try:
                            from ....avatar.persistence import save_position, write_avatar_state_for_ai
                            pos = self.pos()
                            save_position(pos.x(), pos.y())
                            write_avatar_state_for_ai()
                        except Exception:
                            pass
                    # Save size when resize ends
                    if self._resize_edge is not None:
                        try:
                            from ....avatar.persistence import save_avatar_settings, write_avatar_state_for_ai
                            save_avatar_settings(overlay_3d_size=self._size)
                            write_avatar_state_for_ai()
                        except Exception:
                            pass
                    # Save rotation when rotation drag ends
                    if self._rotate_start_x is not None:
                        try:
                            from ....avatar.persistence import save_avatar_settings
                            save_avatar_settings(overlay_3d_yaw=self._manual_yaw_offset)
                        except Exception:
                            pass
                    self._drag_pos = None
                    self._resize_edge = None
                    self._rotate_start_x = None
                    self.setCursor(QCursor(Qt_OpenHandCursor))
                return True  # Block event from reaching GL widget
            
            elif event_type == event.MouseButtonDblClick:
                # Double-click does nothing - use right-click menu for Center on Screen
                return True  # Block event
            
            elif event_type == event.Wheel:
                # Scroll wheel is disabled - use edge drag for resizing
                return True
                
        return super().eventFilter(obj, event)
    
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
    
    def wheelEvent(self, event):
        """Scroll wheel to resize when resize is enabled."""
        if getattr(self, '_resize_enabled', False):
            # Get scroll delta
            try:
                delta = event.angleDelta().y()
            except AttributeError:
                delta = event.delta()
            
            # Resize based on scroll direction (no limits)
            if delta > 0:
                new_size = self._size + 20  # Scroll up = bigger
            else:
                new_size = max(50, self._size - 20)   # Scroll down = smaller (min 50)
            
            self._size = new_size
            self.setFixedSize(self._size, self._size)
            self._gl_container.setFixedSize(self._size, self._size)
            if self._gl_widget:
                self._gl_widget.setFixedSize(self._size, self._size)
                self._apply_circular_mask()
            
            # Save size
            try:
                from ....avatar.persistence import save_avatar_settings, write_avatar_state_for_ai
                save_avatar_settings(overlay_3d_size=self._size)
                write_avatar_state_for_ai()
            except Exception:
                pass
            
            event.accept()
        else:
            event.ignore()
    
    def paintEvent(self, event):
        """Draw thin border when resize mode is enabled."""
        super().paintEvent(event)
        if getattr(self, '_resize_enabled', False):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            pen = painter.pen()
            pen.setColor(QColor("#3498db"))  # Blue border
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt_NoBrush)
            painter.drawRoundedRect(2, 2, self.width() - 4, self.height() - 4, 10, 10)
    
    def contextMenuEvent(self, event):
        """Right-click menu."""
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
        
        # Hide avatar
        close_action = menu.addAction("Hide Avatar")
        close_action.triggered.connect(self._close)
        
        menu.exec_(event.globalPos())
    
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
            from ....avatar.persistence import save_avatar_settings, write_avatar_state_for_ai
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
            # Save size
            try:
                from ....avatar.persistence import save_avatar_settings, write_avatar_state_for_ai
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
            with open(command_path, 'r') as f:
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
        except (json.JSONDecodeError, IOError, ValueError):
            pass
    
    def _do_speaking_pulse(self):
        """Fallback speaking animation when no animator."""
        # Simple scale pulse handled by idle animation
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
        if self._animator:
            self._animator.stop()
        self.hide()
        self.closed.emit()
    
    def closeEvent(self, event):
        self._command_timer.stop()
        self._idle_timer.stop()
        if self._animator:
            self._animator.stop()
        super().closeEvent(event)


class AvatarPreviewWidget(QFrame):
    """2D image preview with drag-to-rotate for 3D simulation."""
    
    expression_changed = pyqtSignal(str)  # Signal when expression changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.original_pixmap = None
        self._svg_mode = False
        self._current_svg = None
        
        self.setMinimumSize(250, 250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # No border in stylesheet - we draw it in paintEvent to wrap tight around image
        self.setStyleSheet("""
            QFrame {
                background: #1e1e2e;
            }
        """)
        
    def set_avatar(self, pixmap: QPixmap):
        """Set avatar image."""
        self.original_pixmap = pixmap
        self._svg_mode = False
        self._update_display()
    
    def set_svg_sprite(self, svg_data: str):
        """Set avatar from SVG data."""
        if not HAS_SVG or QSvgRenderer is None:
            print("SVG support not available - using fallback")
            return
        
        self._svg_mode = True
        self._current_svg = svg_data
        
        # Convert SVG to pixmap for display
        renderer = QSvgRenderer(QByteArray(svg_data.encode('utf-8')))
        
        # Use minimum size of 200 if widget not yet sized
        size = min(self.width(), self.height()) - 20
        if size <= 0:
            size = 200
        
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt_transparent if isinstance(Qt_transparent, QColor) else QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        
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
        
        if self.pixmap:
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
    
    # Header
    header = QLabel("Avatar Display")
    header.setObjectName("header")
    left_panel.addWidget(header)
    
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
    parent.avatar_auto_load_checkbox.setToolTip("Automatically load the last used avatar when ForgeAI starts")
    parent.avatar_auto_load_checkbox.toggled.connect(lambda c: _toggle_auto_load(parent, c))
    settings_row.addWidget(parent.avatar_auto_load_checkbox)
    
    parent.avatar_auto_run_checkbox = QCheckBox("Auto-show pop-out")
    parent.avatar_auto_run_checkbox.setToolTip("Automatically show the desktop pop-out overlay when ForgeAI starts")
    parent.avatar_auto_run_checkbox.toggled.connect(lambda c: _toggle_auto_run(parent, c))
    settings_row.addWidget(parent.avatar_auto_run_checkbox)
    
    parent.avatar_resize_checkbox = QCheckBox("Allow popup resize")
    parent.avatar_resize_checkbox.setToolTip("Allow manual resizing of the desktop avatar popup window (drag from edges to resize)")
    parent.avatar_resize_checkbox.setChecked(False)  # Default OFF - less intrusive
    parent.avatar_resize_checkbox.toggled.connect(lambda c: _toggle_resize_enabled(parent, c))
    settings_row.addWidget(parent.avatar_resize_checkbox)
    
    parent.avatar_reposition_checkbox = QCheckBox("Allow reposition")
    parent.avatar_reposition_checkbox.setToolTip("Allow dragging the avatar to reposition it (saved per avatar)")
    parent.avatar_reposition_checkbox.setChecked(True)  # Default ON - natural behavior
    parent.avatar_reposition_checkbox.toggled.connect(lambda c: _toggle_reposition_enabled(parent, c))
    settings_row.addWidget(parent.avatar_reposition_checkbox)
    
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
        # Don't set checked yet - widgets don't exist
        parent.use_3d_render_checkbox.toggled.connect(lambda c: _toggle_3d_render(parent, c))
        render_row.addWidget(parent.use_3d_render_checkbox)
        render_row.addStretch()
        left_panel.addLayout(render_row)
    else:
        parent.use_3d_render_checkbox = None
    
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
    parent.avatar_combo = QComboBox()
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
    parent.load_btn.clicked.connect(lambda: _load_avatar_file(parent))
    btn_row2.addWidget(parent.load_btn)
    
    parent.apply_btn = QPushButton("Apply Avatar")
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
    
    # Auto-design from personality
    parent.auto_design_btn = QPushButton("AI Auto-Design")
    parent.auto_design_btn.setToolTip("Let AI design avatar based on its personality")
    parent.auto_design_btn.clicked.connect(lambda: _auto_design_avatar(parent))
    actions_layout.addWidget(parent.auto_design_btn)
    
    # Export sprite button
    parent.export_btn = QPushButton("Export Current Sprite")
    parent.export_btn.clicked.connect(lambda: _export_sprite(parent))
    actions_layout.addWidget(parent.export_btn)
    
    actions_group.setLayout(actions_layout)
    right_panel.addWidget(actions_group)
    
    # === Avatar Gallery ===
    gallery_group = QGroupBox("Avatar Gallery")
    gallery_layout = QVBoxLayout()
    
    # Browse avatars button
    parent.browse_avatars_btn = QPushButton("Browse Avatars...")
    parent.browse_avatars_btn.setToolTip("Browse and select from installed avatars")
    parent.browse_avatars_btn.clicked.connect(lambda: _browse_avatars(parent))
    gallery_layout.addWidget(parent.browse_avatars_btn)
    
    # Import avatar button
    parent.import_avatar_btn = QPushButton("Import Avatar...")
    parent.import_avatar_btn.setToolTip("Import a new avatar from files or .forgeavatar bundle")
    parent.import_avatar_btn.clicked.connect(lambda: _import_avatar(parent))
    gallery_layout.addWidget(parent.import_avatar_btn)
    
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
    parent._avatar_resize_enabled = True
    
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
        
        with open(settings_path, 'r') as f:
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
        if hasattr(parent, 'avatar_resize_checkbox'):
            parent.avatar_resize_checkbox.setChecked(parent._avatar_resize_enabled)
        
        # Restore saved overlay sizes for later use when overlay is created
        # Validate sizes are within bounds (100-500) to handle corrupted settings
        saved_2d_size = settings.get("avatar_overlay_size", 300)
        saved_3d_size = settings.get("avatar_overlay_3d_size", 250)
        parent._saved_overlay_size = max(100, min(500, saved_2d_size if isinstance(saved_2d_size, (int, float)) else 300))
        parent._saved_overlay_3d_size = max(100, min(500, saved_3d_size if isinstance(saved_3d_size, (int, float)) else 250))
        
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
                    # Simulate clicking the Run button
                    if not parent.show_overlay_btn.isChecked():
                        parent.show_overlay_btn.setChecked(True)
                        _toggle_overlay(parent)
                        parent.avatar_status.setText("Avatar auto-started on desktop!")
                        parent.avatar_status.setStyleSheet("color: #a6e3a1;")
                # Run after apply completes
                QTimer.singleShot(1000, auto_run_overlay)
    
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
        if hasattr(parent, 'avatar_resize_checkbox'):
            parent.avatar_resize_checkbox.setChecked(avatar_settings.resize_enabled)
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
        from ....avatar.persistence import save_avatar_settings, write_avatar_state_for_ai
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
        from ....avatar.persistence import save_avatar_settings, write_avatar_state_for_ai
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
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QPoint
        
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
            from ....avatar.persistence import save_position, save_avatar_settings, write_avatar_state_for_ai
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
    overlay.setFixedSize(overlay._size, overlay._size)
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
        
    except Exception as e:
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
        except:
            return 0.5
    
    # Parse hex color
    def parse_color(v):
        if isinstance(v, str) and v.startswith('#'):
            try:
                r = int(v[1:3], 16) / 255.0
                g = int(v[3:5], 16) / 255.0
                b = int(v[5:7], 16) / 255.0
                return [r, g, b]
            except:
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
    """Test a random expression in the preview."""
    import random
    expressions = list(SPRITE_TEMPLATES.keys())
    if expressions:
        expr = random.choice(expressions)
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
                # Apply saved size
                saved_size = getattr(parent, '_saved_overlay_3d_size', 250)
                parent._overlay_3d._size = saved_size
                parent._overlay_3d.setFixedSize(saved_size, saved_size)
            
            # Sync resize enabled state to overlay (default OFF)
            parent._overlay_3d._resize_enabled = getattr(parent, '_avatar_resize_enabled', False)
            
            # Load the model into 3D overlay
            if parent._current_path:
                parent._overlay_3d.load_model(str(parent._current_path))
                parent._overlay_3d.show()
                parent._overlay_3d.raise_()
                parent.show_overlay_btn.setText("Stop")
                parent.avatar_status.setText("3D avatar on desktop! Drag window to move, right-click for gestures.")
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
                
                # Load size from avatar persistence (most reliable) or gui settings
                try:
                    from ....avatar.persistence import load_avatar_settings
                    avatar_settings = load_avatar_settings()
                    saved_size = avatar_settings.overlay_size
                    if saved_size < 100 or saved_size > 500:
                        saved_size = 300
                except Exception:
                    saved_size = getattr(parent, '_saved_overlay_size', 300)
                
                parent._overlay._size = saved_size
                parent._overlay.setFixedSize(saved_size, saved_size)
                
                # Also restore position from persistence
                try:
                    from ....avatar.persistence import load_position
                    x, y = load_position()
                    if x >= 0 and y >= 0:
                        parent._overlay.move(x, y)
                except Exception:
                    pass
            
            # Sync resize enabled state to overlay (default OFF)
            parent._overlay._resize_enabled = getattr(parent, '_avatar_resize_enabled', False)
            
            # Only show avatar if one is selected - don't show default/test sprite
            pixmap = parent.avatar_preview_2d.original_pixmap
            if not pixmap or getattr(parent, '_using_builtin_sprite', True):
                # No real avatar loaded - require user to select one
                parent.show_overlay_btn.setChecked(False)
                parent.avatar_status.setText("Select an avatar first! Use 'Load Avatar' or pick from dropdown.")
                parent.avatar_status.setStyleSheet("color: #fab387;")
                return
            
            if pixmap:
                # Pass the ORIGINAL pixmap - set_avatar and _update_scaled_pixmap will handle scaling
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
        elif axis == 'roll':
            parent.avatar_preview_3d.model_roll = radians
            if hasattr(parent, 'roll_label'):
                parent.roll_label.setText(str(int(value)))
        
        parent.avatar_preview_3d.update()


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
    
    parent.avatar_status.setText(f"Auto-oriented: X={pitch}deg, Y={yaw}deg, Z={roll}deg")
    parent.avatar_status.setStyleSheet("color: #89b4fa;")


def _save_model_orientation(parent):
    """Save model orientation to settings file."""
    import json
    from pathlib import Path
    import math
    
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
    settings_path = Path("data/avatar/model_orientations.json")
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    
    orientations = {}
    if settings_path.exists():
        try:
            with open(settings_path, 'r') as f:
                orientations = json.load(f)
        except (json.JSONDecodeError, IOError):
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
    from pathlib import Path
    import math
    
    settings_path = Path("data/avatar/model_orientations.json")
    if not settings_path.exists():
        return None
    
    try:
        with open(settings_path, 'r') as f:
            orientations = json.load(f)
        
        model_key = Path(model_path).name
        if model_key in orientations:
            data = orientations[model_key]
            return {
                'pitch': math.radians(data.get('pitch', 0)),
                'yaw': math.radians(data.get('yaw', 0)),
                'roll': math.radians(data.get('roll', 0))
            }
    except (json.JSONDecodeError, IOError):
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
        parent._overlay.setFixedSize(300, 300)
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
        parent._overlay.setFixedSize(300, 300)
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
    
    # JSON configs
    if AVATAR_CONFIG_DIR.exists():
        for f in sorted(AVATAR_CONFIG_DIR.glob("*.json")):
            cfg = _load_json(f)
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
        with open(path, 'r') as f:
            return json.load(f)
    except:
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
    """Create an info card pixmap for 3D model."""
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
    painter.drawText(0, 40, size, 60, Qt_AlignCenter, "[3D]")
    
    # "3D Model" label
    font.setPointSize(14)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor("#cdd6f4"))
    painter.drawText(0, 100, size, 25, Qt_AlignCenter, "3D Model")
    
    # File name
    font.setPointSize(10)
    font.setBold(False)
    painter.setFont(font)
    painter.setPen(QColor("#a6e3a1"))
    name = path.name
    if len(name) > 25:
        name = name[:22] + "..."
    painter.drawText(0, 130, size, 20, Qt_AlignCenter, name)
    
    # File size
    size_kb = path.stat().st_size / 1024
    painter.setPen(QColor("#6c7086"))
    if size_kb > 1024:
        size_str = f"{size_kb/1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"
    painter.drawText(0, 155, size, 20, Qt_AlignCenter, size_str)
    
    # Instructions
    font.setPointSize(9)
    painter.setFont(font)
    painter.setPen(QColor("#fab387"))
    painter.drawText(0, 200, size, 40, Qt_AlignCenter, "Enable '3D Rendering'\nfor full preview")
    
    painter.end()
    parent.avatar_preview_2d.set_avatar(pixmap)


def _apply_avatar(parent):
    """Apply the selected avatar - fully load it."""
    if not parent._current_path or not parent._current_path.exists():
        parent.avatar_status.setText("Select an avatar first!")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")
        return
    
    path = parent._current_path
    
    # Save current avatar to persistence for AI awareness
    try:
        from ....avatar.persistence import save_avatar_settings, write_avatar_state_for_ai
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
                    parent.yaw_slider.setValue(int(math.degrees(saved_orientation['yaw'])))
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

