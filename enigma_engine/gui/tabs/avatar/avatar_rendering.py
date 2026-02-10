"""
Avatar 3D Rendering Module

Contains OpenGL3DWidget for rendering 3D models with:
- Sketchfab-style orbit controls
- Texture/material support  
- Auto-orientation detection
- VRM avatar support
- Model metadata analysis

Extracted from avatar_display.py for maintainability.
"""
# type: ignore[attr-defined]

import logging
import math
from pathlib import Path
from typing import Any, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QOpenGLWidget

logger = logging.getLogger(__name__)

# Qt flag compatibility
Qt_ArrowCursor: Any = getattr(Qt, 'ArrowCursor', 0)
Qt_LeftButton: Any = getattr(Qt, 'LeftButton', 0x00000001)

# Optional 3D libraries
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

try:
    import OpenGL.GL as GL
    import OpenGL.GLU as GLU
    HAS_OPENGL = True
except ImportError:
    GL = None  # type: ignore
    GLU = None  # type: ignore


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
    - VRM avatar support
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
            
            # Store mesh names for front/back detection
            self._mesh_names = mesh_names_list
            
            # Check if we actually got colors
            if self.colors is not None:
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
                self.apply_auto_orientation()
            
            # Log info
            has_real_colors = self.colors is not None and len(np.unique(self.colors, axis=0)) > 1
            print(f"[Avatar] Loaded {Path(path).name}: {len(self.vertices)} vertices, "
                  f"{len(self.faces)} faces, colors: {'yes (varied)' if has_real_colors else 'no/uniform'}")
            
            # Store model metadata for AI awareness
            self._model_metadata = self._analyze_model_structure(scene, meshes)
            
            # VRM-specific loading for anime-style models
            if Path(path).suffix.lower() == '.vrm':
                self._load_vrm_data(path)
            
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
                if len(np.unique(colors, axis=0)) > 1:
                    print(f"[Avatar] Got vertex colors from mesh")
                    return colors
        
        # Method 2: TextureVisuals - sample from texture using UVs
        if hasattr(visual, 'kind') and visual.kind == 'texture':
            try:
                uv = None
                if hasattr(visual, 'uv') and visual.uv is not None:
                    uv = visual.uv
                
                material = getattr(visual, 'material', None)
                img = None
                
                if material:
                    for attr in ['baseColorTexture', 'image', 'diffuse']:
                        tex = getattr(material, attr, None)
                        if tex is not None:
                            img = tex
                            break
                
                if img is not None and uv is not None and len(uv) == num_verts:
                    if not isinstance(img, np.ndarray):
                        img = np.array(img)
                    
                    h, w = img.shape[:2]
                    
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
            
            texture_dirs = [
                model_dir / "textures",
                model_dir / "texture", 
                model_dir / "tex",
                model_dir,
            ]
            
            for subdir in model_dir.iterdir():
                if subdir.is_dir() and subdir.name.isdigit():
                    texture_dirs.append(subdir)
            
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
            
            tex_img = None
            tex_name = None
            
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
            
            all_colors = []
            for mesh in meshes:
                if not hasattr(mesh, 'vertices'):
                    continue
                    
                num_verts = len(mesh.vertices)
                
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                    uv = mesh.visual.uv
                    if len(uv) == num_verts:
                        u = np.clip(uv[:, 0] % 1.0, 0, 0.9999)
                        v = np.clip((1.0 - uv[:, 1]) % 1.0, 0, 0.9999)
                        
                        px = (u * w).astype(int)
                        py = (v * h).astype(int)
                        
                        mesh_colors = img_array[py, px, :3].astype(np.float32) / 255.0
                        all_colors.append(mesh_colors)
                        continue
                
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
        """Analyze the 3D model structure for AI awareness."""
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
            
            for mesh in meshes:
                name = getattr(mesh, 'name', None) or getattr(mesh.metadata, 'name', None) if hasattr(mesh, 'metadata') else None
                if name:
                    info['mesh_names'].append(name)
            
            if hasattr(scene, 'graph') and scene.graph is not None:
                try:
                    graph = scene.graph
                    if hasattr(graph, 'nodes_geometry'):
                        for node_name in graph.nodes:
                            node_name_lower = str(node_name).lower()
                            if any(bone_keyword in node_name_lower for bone_keyword in [
                                'bone', 'joint', 'skeleton', 'armature', 'spine', 'hip', 'head',
                                'arm', 'leg', 'hand', 'foot', 'finger', 'neck', 'shoulder', 'elbow',
                                'knee', 'ankle', 'wrist', 'pelvis', 'chest', 'root'
                            ]):
                                info['skeleton_bones'].append(str(node_name))
                                info['has_skeleton'] = True
                except Exception as e:
                    print(f"[Avatar] Skeleton detection error: {e}")
            
            for mesh in meshes:
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                    mat = mesh.visual.material
                    mat_info = {'name': getattr(mat, 'name', 'unnamed')}
                    
                    for tex_attr in ['baseColorTexture', 'image', 'diffuse']:
                        if getattr(mat, tex_attr, None) is not None:
                            info['has_textures'] = True
                            mat_info['has_texture'] = True
                            break
                    
                    info['materials'].append(mat_info)
            
            if info['size']:
                w, h, d = info['size']['width'], info['size']['height'], info['size']['depth']
                
                if h > w * 1.5 and h > d * 1.5:
                    if info['has_skeleton'] or len(info['skeleton_bones']) > 0:
                        info['estimated_type'] = 'humanoid_rigged'
                    else:
                        info['estimated_type'] = 'humanoid_static'
                elif any('character' in name.lower() or 'body' in name.lower() or 'face' in name.lower() 
                        for name in info['mesh_names']):
                    info['estimated_type'] = 'character'
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
    
    def _load_vrm_data(self, path: str):
        """Load VRM-specific data (expressions, humanoid bones, metadata)."""
        try:
            from ....avatar.vrm_support import VRMLoader, VRMModel
            
            loader = VRMLoader()
            vrm_model: VRMModel = loader.load(path)
            
            self._vrm_model = vrm_model
            
            if hasattr(self, '_model_metadata'):
                self._model_metadata['is_vrm'] = True
                self._model_metadata['vrm_version'] = vrm_model.version.value
                self._model_metadata['estimated_type'] = 'vrm_humanoid'
                
                if vrm_model.meta:
                    self._model_metadata['vrm_meta'] = {
                        'name': vrm_model.meta.name or vrm_model.meta.title,
                        'author': vrm_model.meta.author,
                        'version': vrm_model.meta.version,
                    }
                
                self._model_metadata['vrm_bones'] = len(vrm_model.humanoid.bones)
                self._model_metadata['has_skeleton'] = True
                self._model_metadata['skeleton_bones'] = list(vrm_model.humanoid.bones.keys())
                self._model_metadata['vrm_expressions'] = list(vrm_model.expressions.keys())
            
            self._vrm_expression_map = {}
            emotion_mapping = {
                'happy': ['happy', 'joy', 'smile'],
                'sad': ['sad', 'sorrow'],
                'angry': ['angry', 'anger'],
                'surprised': ['surprised', 'surprise'],
                'neutral': ['neutral', 'relaxed'],
                'blink': ['blink', 'blinkLeft', 'blinkRight'],
                'aa': ['aa', 'a'],
                'ih': ['ih', 'i'],
                'ou': ['ou', 'u'],
                'ee': ['ee', 'e'],
                'oh': ['oh', 'o'],
            }
            
            for emotion, vrm_names in emotion_mapping.items():
                for vrm_name in vrm_names:
                    if vrm_name in vrm_model.expressions:
                        self._vrm_expression_map[emotion] = vrm_name
                        break
            
            print(f"[Avatar] VRM loaded: {vrm_model.meta.name or Path(path).stem}")
            print(f"[Avatar] VRM expressions: {list(vrm_model.expressions.keys())}")
            print(f"[Avatar] VRM bones: {len(vrm_model.humanoid.bones)}")
            
        except ImportError as e:
            print(f"[Avatar] VRM support not available: {e}")
        except Exception as e:
            print(f"[Avatar] VRM parsing error: {e}")
            import traceback
            traceback.print_exc()
    
    def get_vrm_model(self) -> Optional[Any]:
        """Get the loaded VRM model data if available."""
        return getattr(self, '_vrm_model', None)
    
    def set_vrm_expression(self, expression: str, weight: float = 1.0) -> bool:
        """Set a VRM expression by name or emotion."""
        vrm_model = self.get_vrm_model()
        if not vrm_model:
            return False
        
        expr_map = getattr(self, '_vrm_expression_map', {})
        vrm_expr_name = expr_map.get(expression, expression)
        
        if vrm_expr_name in vrm_model.expressions:
            if not hasattr(self, '_vrm_expression_weights'):
                self._vrm_expression_weights = {}
            self._vrm_expression_weights[vrm_expr_name] = weight
            print(f"[Avatar] VRM expression set: {vrm_expr_name} = {weight}")
            return True
        
        return False

    def reset_view(self):
        """Reset camera to default position."""
        self.rotation_x = self._default_rotation_x
        self.rotation_y = self._default_rotation_y
        self.zoom = self._default_zoom
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()
    
    def auto_orient_model(self) -> tuple:
        """Automatically detect and fix model orientation."""
        if self.vertices is None or len(self.vertices) == 0:
            return (0.0, 0.0, 0.0)
        
        try:
            mins = self.vertices.min(axis=0)
            maxs = self.vertices.max(axis=0)
            dims = maxs - mins
            
            x_size, y_size, z_size = dims[0], dims[1], dims[2]
            
            pitch, yaw, roll = 0.0, 0.0, 0.0
            
            if y_size >= x_size and y_size >= z_size:
                print(f"[Avatar] Auto-orient: Model appears Y-up (correct), dims={dims}")
            elif z_size > y_size and z_size > x_size:
                print(f"[Avatar] Auto-orient: Model appears Z-up, rotating -90 deg X")
                pitch = -90.0
            elif x_size > y_size and x_size > z_size:
                print(f"[Avatar] Auto-orient: Model appears X-up, rotating 90 deg Z")
                roll = 90.0
            
            horizontal_size = max(x_size, z_size)
            if horizontal_size > y_size * 1.5:
                if z_size > x_size:
                    print(f"[Avatar] Auto-orient: Model appears lying forward, rotating -90 deg X")
                    pitch = -90.0
                else:
                    print(f"[Avatar] Auto-orient: Model appears lying sideways, rotating 90 deg Z")
                    roll = 90.0
            
            yaw = self._detect_facing_direction()
            
            if yaw != 0:
                print(f"[Avatar] Auto-orient: Detected backward facing, rotating {yaw} deg Y")
            
            return (pitch, yaw, roll)
            
        except Exception as e:
            print(f"[Avatar] Auto-orient error: {e}")
            return (0.0, 0.0, 0.0)
    
    def _detect_facing_direction(self) -> float:
        """Detect if model is facing backward and needs 180 deg yaw rotation."""
        yaw_correction = 0.0
        confidence_score = 0
        
        try:
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
            
            if front_meshes:
                print(f"[Avatar] Front-detection: Found front-indicator meshes: {front_meshes}")
                confidence_score += 1
            
            if self.normals is not None and len(self.normals) > 0:
                norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
                norms[norms == 0] = 1
                normalized_normals = self.normals / norms
                
                if self.vertices is not None:
                    y_center = self.vertices[:, 1].mean()
                    upper_mask = self.vertices[:, 1] > y_center
                    
                    if upper_mask.sum() > 10:
                        upper_normals = normalized_normals[upper_mask]
                        avg_normal = upper_normals.mean(axis=0)
                        avg_z = avg_normal[2] if len(avg_normal) > 2 else 0
                        
                        print(f"[Avatar] Front-detection: Upper region avg normal Z = {avg_z:.3f}")
                        
                        if avg_z < -0.1:
                            confidence_score += 2
                            print(f"[Avatar] Front-detection: Normals suggest model faces -Z (backward)")
            
            if self.vertices is not None and len(self.vertices) > 100:
                z_center = self.vertices[:, 2].mean()
                y_center = self.vertices[:, 1].mean()
                
                upper_front = ((self.vertices[:, 1] > y_center) & (self.vertices[:, 2] > z_center)).sum()
                upper_back = ((self.vertices[:, 1] > y_center) & (self.vertices[:, 2] <= z_center)).sum()
                
                if upper_front > 0 and upper_back > 0:
                    ratio = upper_back / upper_front
                    print(f"[Avatar] Front-detection: Upper back/front vertex ratio = {ratio:.2f}")
                    
                    if ratio > 1.5:
                        confidence_score += 1
                        print(f"[Avatar] Front-detection: More detail in back suggests backward facing")
            
            if confidence_score >= 2:
                print(f"[Avatar] Front-detection: Confidence {confidence_score}/4 - applying 180 deg yaw")
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
                print(f"[Avatar] Loaded saved orientation for {model_key}")
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
        self._rotate_timer.start(16)
    
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
            if self.transparent_bg:
                GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            else:
                GL.glClearColor(0.08, 0.08, 0.12, 1.0)
            
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_LIGHTING)
            GL.glEnable(GL.GL_LIGHT0)
            GL.glEnable(GL.GL_LIGHT1)
            GL.glEnable(GL.GL_COLOR_MATERIAL)
            GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
            GL.glShadeModel(GL.GL_SMOOTH)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            
            self._update_lighting()
            
            try:
                GL.glEnable(GL.GL_LINE_SMOOTH)
                GL.glEnable(GL.GL_POLYGON_SMOOTH)
                GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
                GL.glHint(GL.GL_POLYGON_SMOOTH_HINT, GL.GL_NICEST)
            except Exception:
                pass
                
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
        
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [2.0, 3.0, 2.0, 0.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [intensity, intensity * 0.95, intensity * 0.9, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, [ambient, ambient, ambient * 1.2, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_POSITION, [-2.0, -1.0, -2.0, 0.0])
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE, [intensity * 0.3, intensity * 0.35, intensity * 0.5, 1.0])
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
        
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
        GLU.gluPerspective(35, aspect, 0.1, 100.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        
    def paintGL(self):
        """Render with Sketchfab-style visuals."""
        if not HAS_OPENGL or GL is None:
            return
        
        try:
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            
            if not self.transparent_bg:
                self._draw_gradient_background()
            
            if self.show_grid and not self.transparent_bg:
                self._draw_grid()
            
            GL.glLoadIdentity()
            
            GL.glTranslatef(self.pan_x, self.pan_y, -self.zoom)
            GL.glRotatef(self.rotation_x, 1, 0, 0)
            GL.glRotatef(self.rotation_y, 0, 1, 0)
            
            GL.glRotatef(math.degrees(self.model_pitch), 1, 0, 0)
            GL.glRotatef(math.degrees(self.model_yaw), 0, 1, 0)
            GL.glRotatef(math.degrees(self.model_roll), 0, 0, 1)
            
            if self.is_loading:
                GL.glDisable(GL.GL_LIGHTING)
                GL.glColor3f(0.5, 0.5, 0.6)
                GL.glEnable(GL.GL_LIGHTING)
                return
            
            if self.vertices is not None and self.faces is not None:
                if self.wireframe_mode:
                    GL.glDisable(GL.GL_LIGHTING)
                    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
                    GL.glColor3f(0.4, 0.6, 0.9)
                else:
                    GL.glEnable(GL.GL_LIGHTING)
                    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
                
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
                    
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        except Exception as e:
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
        GL.glColor3f(0.06, 0.06, 0.10)
        GL.glVertex2f(-1, 1)
        GL.glVertex2f(1, 1)
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
        
        grid_y = -0.8
        grid_size = 3.0
        grid_step = 0.25
        
        GL.glBegin(GL.GL_LINES)
        
        steps = int(grid_size / grid_step)
        for i in range(-steps, steps + 1):
            x = i * grid_step
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
        """Start drag or pan."""
        if getattr(self, 'disable_mouse_interaction', False):
            event.ignore()
            return
        self.last_pos = event.pos()
        if event.button() == Qt_LeftButton:
            self.is_panning = event.modifiers() == Qt.ShiftModifier if hasattr(Qt, 'ShiftModifier') else False
        elif event.button() == Qt.RightButton if hasattr(Qt, 'RightButton') else 0x00000002:
            self.is_panning = False
        event.accept()
        
    def mouseMoveEvent(self, event):
        """Rotate or pan on drag."""
        if getattr(self, 'disable_mouse_interaction', False):
            event.ignore()
            return
        if self.last_pos is not None:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            
            buttons = event.buttons()
            right_button = Qt.RightButton if hasattr(Qt, 'RightButton') else 0x00000002
            
            if self.is_panning:
                self.pan_x += dx * 0.005
                self.pan_y -= dy * 0.005
            elif (buttons & right_button) or (buttons & Qt_LeftButton):
                self.rotation_y += dx * 0.5
                self.rotation_x += dy * 0.5
            
            self.last_pos = event.pos()
            self.update()
        event.accept()
            
    def mouseReleaseEvent(self, event):
        """End drag."""
        if getattr(self, 'disable_mouse_interaction', False):
            event.ignore()
            return
        self.last_pos = None
        self.is_panning = False
        self.setCursor(QCursor(Qt_ArrowCursor))
        event.accept()
        
    def wheelEvent(self, event):
        """Zoom with scroll wheel."""
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


# Export constants for other modules
__all__ = [
    'OpenGL3DWidget',
    'HAS_TRIMESH',
    'HAS_OPENGL',
]
