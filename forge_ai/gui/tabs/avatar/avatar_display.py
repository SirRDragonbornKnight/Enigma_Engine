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
    QSlider, QColorDialog, QGridLayout, QScrollArea
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
Qt_OpenHandCursor: Any = getattr(Qt, 'OpenHandCursor', 17)
Qt_ClosedHandCursor: Any = getattr(Qt, 'ClosedHandCursor', 18)
Qt_ArrowCursor: Any = getattr(Qt, 'ArrowCursor', 0)
import json
import os

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
        self.mesh = None
        self.vertices = None
        self.faces = None
        self.normals = None
        self.colors = None
        self.texture_colors = None  # Per-vertex colors from texture
        
        # Camera
        self.rotation_x = 20.0  # Slight tilt for better view
        self.rotation_y = 45.0  # Start at 45 degrees
        self.zoom = 3.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        # Default camera settings (for reset)
        self._default_rotation_x = 20.0
        self._default_rotation_y = 45.0
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
        
        self.setMinimumSize(250, 250)
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt_OpenHandCursor))
        
    def load_model(self, path: str) -> bool:
        """Load a 3D model file with texture support."""
        if not HAS_TRIMESH or trimesh is None or np is None:
            return False
        
        self.is_loading = True
        self.model_name = Path(path).stem
        self._model_path = path
        self.update()
        
        try:
            # Load with texture resolver for GLTF files
            scene = trimesh.load(str(path))
            
            # Collect all meshes with their colors
            all_vertices = []
            all_faces = []
            all_normals = []
            all_colors = []
            vertex_offset = 0
            
            # Get list of meshes (handle both Scene and single Mesh)
            if hasattr(scene, 'geometry') and scene.geometry:
                meshes = list(scene.geometry.values())
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
            
            # Check if we actually got colors
            if self.colors is not None:
                # Check if colors are all the same (default gray)
                unique_colors = np.unique(self.colors, axis=0)
                if len(unique_colors) <= 1:
                    print("[Avatar] Colors appear to be uniform - trying texture loading...")
                    self.colors = self._load_textures_from_files(path, meshes)
            
            # Center and scale the mesh
            centroid = self.vertices.mean(axis=0)
            self.vertices -= centroid
            max_extent = max(self.vertices.max(axis=0) - self.vertices.min(axis=0))
            if max_extent > 0:
                scale = 1.5 / max_extent
                self.vertices *= scale
            
            self.is_loading = False
            self.reset_view()
            
            # Log info
            print(f"[Avatar] Loaded {Path(path).name}: {len(self.vertices)} vertices, "
                  f"{len(self.faces)} faces, colors: {'yes' if self.colors is not None else 'no'}")
            
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
        
        # Method 1: Direct vertex colors
        if hasattr(visual, 'vertex_colors') and visual.vertex_colors is not None:
            vc = visual.vertex_colors
            if len(vc) == num_verts and not np.all(vc[:, :3] == vc[0, :3]):
                return np.array(vc[:, :3] / 255.0, dtype=np.float32)
        
        # Method 2: Try to_color() to bake textures
        if hasattr(visual, 'to_color'):
            try:
                color_visual = visual.to_color()
                if hasattr(color_visual, 'vertex_colors') and color_visual.vertex_colors is not None:
                    vc = color_visual.vertex_colors
                    if len(vc) == num_verts:
                        return np.array(vc[:, :3] / 255.0, dtype=np.float32)
            except Exception:
                pass
        
        # Method 3: Sample from texture image using UVs
        if hasattr(visual, 'uv') and visual.uv is not None:
            try:
                material = getattr(visual, 'material', None)
                if material:
                    # Get base color texture
                    img = None
                    if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                        img = material.baseColorTexture
                    elif hasattr(material, 'image') and material.image is not None:
                        img = material.image
                    
                    if img is not None:
                        from PIL import Image
                        uv = visual.uv
                        img_array = np.array(img)
                        h, w = img_array.shape[:2]
                        
                        # Handle UV coordinates
                        u = np.clip(uv[:, 0] % 1.0, 0, 1)
                        v = np.clip((1 - uv[:, 1]) % 1.0, 0, 1)
                        
                        px = (u * (w - 1)).astype(int)
                        py = (v * (h - 1)).astype(int)
                        
                        if img_array.ndim == 3:
                            if img_array.shape[2] >= 3:
                                return img_array[py, px, :3].astype(np.float32) / 255.0
                        else:
                            gray = img_array[py, px].astype(np.float32) / 255.0
                            return np.stack([gray, gray, gray], axis=-1)
            except Exception as e:
                print(f"[Avatar] UV texture sampling failed: {e}")
        
        # Method 4: Material base color
        if hasattr(visual, 'material'):
            material = visual.material
            if hasattr(material, 'main_color') and material.main_color is not None:
                color = np.array(material.main_color[:3]) / 255.0
                return np.tile(color, (num_verts, 1)).astype(np.float32)
            elif hasattr(material, 'baseColorFactor') and material.baseColorFactor is not None:
                color = np.array(material.baseColorFactor[:3])
                return np.tile(color, (num_verts, 1)).astype(np.float32)
        
        return default_color
    
    def _load_textures_from_files(self, model_path, meshes):
        """Try to load textures directly from files in the model directory."""
        try:
            from PIL import Image
            model_dir = Path(model_path).parent
            
            # Look for texture directory
            texture_dirs = [
                model_dir / "textures",
                model_dir / "texture",
                model_dir,
            ]
            
            texture_files = {}
            for tex_dir in texture_dirs:
                if tex_dir.exists():
                    for f in tex_dir.iterdir():
                        if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                            # Prioritize baseColor textures
                            name = f.stem.lower()
                            if 'basecolor' in name or 'diffuse' in name or 'albedo' in name:
                                texture_files[name] = f
            
            if not texture_files:
                print(f"[Avatar] No texture files found in {model_dir}")
                return None
            
            print(f"[Avatar] Found {len(texture_files)} texture files")
            
            # For now, just load the first baseColor texture and apply it globally
            # This is a fallback when trimesh fails to apply textures
            for name, tex_path in texture_files.items():
                try:
                    img = Image.open(tex_path).convert('RGB')
                    # Sample average color from texture
                    img_small = img.resize((32, 32))
                    img_array = np.array(img_small) / 255.0
                    avg_color = img_array.mean(axis=(0, 1))
                    
                    print(f"[Avatar] Using texture {tex_path.name}, avg color: {avg_color}")
                    
                    # Apply this color to all vertices
                    return np.tile(avg_color, (len(self.vertices), 1)).astype(np.float32)
                except Exception as e:
                    print(f"[Avatar] Failed to load {tex_path}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            print(f"[Avatar] Texture file loading failed: {e}")
            return None
    
    def reset_view(self):
        """Reset camera to default position."""
        self.rotation_x = self._default_rotation_x
        self.rotation_y = self._default_rotation_y
        self.zoom = self._default_zoom
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()
    
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
        """Start drag or pan."""
        self.last_pos = event.pos()
        if event.button() == Qt_LeftButton:
            self.is_panning = event.modifiers() == Qt.ShiftModifier if hasattr(Qt, 'ShiftModifier') else False
            self.setCursor(QCursor(Qt_ClosedHandCursor))
        event.accept()
        
    def mouseMoveEvent(self, event):
        """Rotate or pan on drag."""
        if self.last_pos is not None:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            
            if self.is_panning:
                # Pan
                self.pan_x += dx * 0.005
                self.pan_y -= dy * 0.005
            else:
                # Rotate
                self.rotation_y += dx * 0.5
                self.rotation_x += dy * 0.5
                # Clamp vertical rotation
                self.rotation_x = max(-90, min(90, self.rotation_x))
            
            self.last_pos = event.pos()
            self.update()
        event.accept()
            
    def mouseReleaseEvent(self, event):
        """End drag."""
        self.last_pos = None
        self.is_panning = False
        self.setCursor(QCursor(Qt_OpenHandCursor))
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
    - Scroll wheel to resize
    - Always on top of other windows
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
        
        # Enable mouse tracking for visual cursor feedback
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt_OpenHandCursor))
        
    def set_avatar(self, pixmap: QPixmap):
        """Set avatar image."""
        self._original_pixmap = pixmap
        self._update_scaled_pixmap()
        
    def _update_scaled_pixmap(self):
        """Update scaled pixmap to match current size."""
        if self._original_pixmap:
            self.pixmap = self._original_pixmap.scaled(
                self._size - 20, self._size - 20,
                Qt_KeepAspectRatio, Qt_SmoothTransformation
            )
        self.update()
        
    def paintEvent(self, a0):
        """Draw avatar with subtle shadow."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        if self.pixmap:
            x = (self.width() - self.pixmap.width()) // 2
            y = (self.height() - self.pixmap.height()) // 2
            
            # Draw a subtle circular background/glow
            painter.setPen(Qt_NoPen)
            painter.setBrush(QColor(30, 30, 46, 80))  # Semi-transparent dark
            painter.drawEllipse(x - 5, y - 5, self.pixmap.width() + 10, self.pixmap.height() + 10)
            
            # Draw the avatar
            painter.drawPixmap(x, y, self.pixmap)
        else:
            # Draw placeholder circle
            painter.setPen(QColor("#6c7086"))
            painter.setBrush(QColor(30, 30, 46, 150))
            size = min(self.width(), self.height()) - 20
            painter.drawEllipse(10, 10, size, size)
            painter.drawText(self.rect(), Qt_AlignCenter, "?")
        
    def mousePressEvent(self, a0):  # type: ignore
        """Start drag to move."""
        if a0.button() == Qt_LeftButton:
            self._drag_pos = a0.globalPos() - self.pos()
            self.setCursor(QCursor(Qt_ClosedHandCursor))
        a0.accept()  # Always accept to ensure we get events
            
    def mouseMoveEvent(self, a0):  # type: ignore
        """Drag to move window."""
        if self._drag_pos is not None and a0.buttons() == Qt_LeftButton:
            self.move(a0.globalPos() - self._drag_pos)
        a0.accept()
            
    def mouseReleaseEvent(self, a0):  # type: ignore
        """End drag."""
        self._drag_pos = None
        self.setCursor(QCursor(Qt_OpenHandCursor))
        a0.accept()
        
    def keyPressEvent(self, a0):  # type: ignore
        """ESC to close."""
        if a0.key() == Qt.Key_Escape if hasattr(Qt, 'Key_Escape') else 0x01000000:
            self.hide()
            self.closed.emit()
        
    def wheelEvent(self, a0):  # type: ignore
        """Scroll to resize."""
        delta = a0.angleDelta().y()
        if delta > 0:
            self._size = min(500, self._size + 20)
        else:
            self._size = max(100, self._size - 20)
        
        self.setFixedSize(self._size, self._size)
        self._update_scaled_pixmap()
        a0.accept()
        
    def contextMenuEvent(self, a0):  # type: ignore
        """Right-click to show options menu."""
        from PyQt5.QtWidgets import QMenu, QAction
        
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
        
        # Expression submenu
        expr_menu = menu.addMenu("😊 Expression")
        expressions = ["idle", "happy", "sad", "thinking", "surprised", "excited", "angry", "love", "sleeping", "winking"]
        for expr in expressions:
            action = expr_menu.addAction(expr.title())
            action.triggered.connect(lambda checked, e=expr: self._change_expression(e))
        
        menu.addSeparator()
        
        # Size options
        size_menu = menu.addMenu("📐 Size")
        for size in [150, 200, 300, 400, 500]:
            action = size_menu.addAction(f"{size}px")
            action.triggered.connect(lambda checked, s=size: self._set_size(s))
        
        menu.addSeparator()
        
        # Reset position
        reset_pos = menu.addAction("🏠 Reset Position")
        reset_pos.triggered.connect(lambda: self.move(100, 100))
        
        # Reset size
        reset_size = menu.addAction("↩️ Reset Size")
        reset_size.triggered.connect(lambda: self._set_size(300))
        
        menu.addSeparator()
        
        # Close
        close_action = menu.addAction("❌ Close Avatar")
        close_action.triggered.connect(self._close_avatar)
        
        menu.exec_(a0.globalPos())
        
    def _change_expression(self, expression: str):
        """Change avatar expression."""
        try:
            svg_data = generate_sprite(
                expression,
                "#6366f1",  # Default colors
                "#8b5cf6",
                "#10b981"
            )
            # Convert SVG to pixmap
            if HAS_SVG and QSvgRenderer is not None:
                renderer = QSvgRenderer(QByteArray(svg_data.encode('utf-8')))
                pixmap = QPixmap(280, 280)
                pixmap.fill(QColor(0, 0, 0, 0))
                painter = QPainter(pixmap)
                renderer.render(painter)
                painter.end()
                self.set_avatar(pixmap)
            else:
                print("SVG support not available")
        except Exception as e:
            print(f"Error changing expression: {e}")
            
    def _set_size(self, size: int):
        """Set avatar size."""
        self._size = size
        self.setFixedSize(self._size, self._size)
        self._update_scaled_pixmap()
        
    def _close_avatar(self):
        """Close the avatar."""
        self.hide()
        self.closed.emit()
        
    def mouseDoubleClickEvent(self, a0):  # type: ignore
        """Double-click to reset size."""
        self._size = 300
        self.setFixedSize(self._size, self._size)
        self._update_scaled_pixmap()


class Avatar3DOverlayWindow(QWidget):
    """Transparent 3D overlay window for desktop avatar display.
    
    Features:
    - Drag handle at top to move (not the whole window)
    - 3D model can be rotated/zoomed normally
    - Circular mask that wraps around the avatar
    - Right-click menu for options
    """
    
    closed = pyqtSignal()
    
    def __init__(self):
        super().__init__(None)
        
        # Transparent, always-on-top, frameless
        self.setWindowFlags(
            Qt_FramelessWindowHint |
            Qt_WindowStaysOnTopHint
        )
        self.setAttribute(Qt_WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.StrongFocus if hasattr(Qt, 'StrongFocus') else 0x0b)
        
        self._size = 300
        self._drag_handle_height = 24
        self.setFixedSize(self._size, self._size + self._drag_handle_height)
        self.move(100, 100)
        
        self._drag_pos = None
        self._model_path = None
        self._use_circular_mask = True
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Drag handle bar at top
        self._drag_handle = QWidget()
        self._drag_handle.setFixedHeight(self._drag_handle_height)
        self._drag_handle.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(99, 102, 241, 180),
                    stop:0.5 rgba(139, 92, 246, 200),
                    stop:1 rgba(99, 102, 241, 180));
                border-radius: 12px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }
        """)
        self._drag_handle.setCursor(QCursor(Qt_OpenHandCursor))
        self._drag_handle.setToolTip("Drag to move • Right-click for menu • Scroll to resize")
        
        # Add grip dots to drag handle
        handle_layout = QHBoxLayout(self._drag_handle)
        handle_layout.setContentsMargins(0, 0, 0, 0)
        grip_label = QLabel("⋮⋮⋮")
        grip_label.setStyleSheet("color: rgba(255,255,255,0.7); font-size: 10px; background: transparent;")
        grip_label.setAlignment(Qt_AlignCenter)
        handle_layout.addWidget(grip_label)
        
        main_layout.addWidget(self._drag_handle)
        
        # Container for the 3D widget with circular mask
        self._gl_container = QWidget()
        self._gl_container.setFixedSize(self._size, self._size)
        self._gl_container.setStyleSheet("background: transparent;")
        
        gl_layout = QVBoxLayout(self._gl_container)
        gl_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create 3D widget with transparent background
        if HAS_OPENGL and HAS_TRIMESH:
            self._gl_widget = OpenGL3DWidget(self._gl_container, transparent_bg=True)
            self._gl_widget.setFixedSize(self._size, self._size)
            self._gl_widget.start_auto_rotate()
            gl_layout.addWidget(self._gl_widget)
            
            # Apply circular mask to the GL widget
            self._apply_circular_mask()
        else:
            self._gl_widget = None
            placeholder = QLabel("3D not available")
            placeholder.setStyleSheet("color: white; background: rgba(30,30,46,150); border-radius: 50%;")
            placeholder.setAlignment(Qt_AlignCenter)
            gl_layout.addWidget(placeholder)
        
        main_layout.addWidget(self._gl_container)
        
        # Install event filter on drag handle for dragging
        self._drag_handle.installEventFilter(self)
        
        self.setMouseTracking(True)
    
    def _apply_circular_mask(self):
        """Apply a circular/elliptical mask to wrap around the avatar."""
        if not self._use_circular_mask:
            return
        
        from PyQt5.QtGui import QRegion, QBitmap, QPainterPath
        from PyQt5.QtCore import QRectF
        
        # Create circular region for the GL widget
        path = QPainterPath()
        # Slightly smaller circle with padding
        padding = 10
        path.addEllipse(QRectF(padding, padding, self._size - padding*2, self._size - padding*2))
        
        region = QRegion(path.toFillPolygon().toPolygon())
        if self._gl_widget:
            self._gl_widget.setMask(region)
    
    def load_model(self, path: str) -> bool:
        """Load a 3D model into the overlay."""
        self._model_path = path
        if self._gl_widget:
            result = self._gl_widget.load_model(path)
            # Re-apply mask after loading
            self._apply_circular_mask()
            return result
        return False
    
    def eventFilter(self, obj, event):
        """Handle drag events on the drag handle."""
        if obj == self._drag_handle:
            if event.type() == event.MouseButtonPress and event.button() == Qt_LeftButton:
                self._drag_pos = event.globalPos() - self.pos()
                self._drag_handle.setCursor(QCursor(Qt_ClosedHandCursor))
                return True
            elif event.type() == event.MouseMove and self._drag_pos is not None:
                self.move(event.globalPos() - self._drag_pos)
                return True
            elif event.type() == event.MouseButtonRelease:
                self._drag_pos = None
                self._drag_handle.setCursor(QCursor(Qt_OpenHandCursor))
                return True
        return super().eventFilter(obj, event)
    
    def wheelEvent(self, event):
        """Scroll on drag handle to resize."""
        # Only resize if scrolling over drag handle area
        if event.pos().y() <= self._drag_handle_height:
            delta = event.angleDelta().y()
            if delta > 0:
                self._size = min(600, self._size + 30)
            else:
                self._size = max(150, self._size - 30)
            
            self.setFixedSize(self._size, self._size + self._drag_handle_height)
            self._gl_container.setFixedSize(self._size, self._size)
            if self._gl_widget:
                self._gl_widget.setFixedSize(self._size, self._size)
                self._apply_circular_mask()
            event.accept()
        else:
            # Pass to GL widget for zoom
            if self._gl_widget:
                self._gl_widget.wheelEvent(event)
    
    def keyPressEvent(self, event):
        """ESC to close, R to toggle rotation."""
        key = event.key()
        if key == (Qt.Key_Escape if hasattr(Qt, 'Key_Escape') else 0x01000000):
            self.hide()
            self.closed.emit()
        elif key == (Qt.Key_R if hasattr(Qt, 'Key_R') else 0x52):
            if self._gl_widget:
                if self._gl_widget.auto_rotate:
                    self._gl_widget.stop_auto_rotate()
                else:
                    self._gl_widget.start_auto_rotate()
        elif key == (Qt.Key_W if hasattr(Qt, 'Key_W') else 0x57):
            if self._gl_widget:
                self._gl_widget.wireframe_mode = not self._gl_widget.wireframe_mode
                self._gl_widget.update()
        elif key == (Qt.Key_M if hasattr(Qt, 'Key_M') else 0x4D):
            # Toggle mask
            self._use_circular_mask = not self._use_circular_mask
            if self._use_circular_mask:
                self._apply_circular_mask()
            elif self._gl_widget:
                self._gl_widget.clearMask()
    
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
        
        # Toggle rotation
        rotate_text = "⏸️ Stop Rotation" if (self._gl_widget and self._gl_widget.auto_rotate) else "🔄 Start Rotation"
        rotate_action = menu.addAction(f"{rotate_text} (R)")
        rotate_action.triggered.connect(lambda: self._toggle_rotation())
        
        # Toggle wireframe
        wireframe_action = menu.addAction("🔲 Toggle Wireframe (W)")
        wireframe_action.triggered.connect(lambda: self._toggle_wireframe())
        
        # Toggle circular mask
        mask_text = "⬜ Square Mode" if self._use_circular_mask else "⚫ Circular Mode"
        mask_action = menu.addAction(f"{mask_text} (M)")
        mask_action.triggered.connect(lambda: self._toggle_mask())
        
        menu.addSeparator()
        
        # Size options
        size_menu = menu.addMenu("📐 Size")
        for size in [200, 300, 400, 500, 600]:
            action = size_menu.addAction(f"{size}px")
            action.triggered.connect(lambda checked, s=size: self._set_size(s))
        
        menu.addSeparator()
        
        # Reset view
        reset_action = menu.addAction("↩️ Reset View")
        reset_action.triggered.connect(lambda: self._reset_view())
        
        # Close
        close_action = menu.addAction("❌ Hide Avatar")
        close_action.triggered.connect(self._close)
        
        menu.exec_(event.globalPos())
    
    def _toggle_rotation(self):
        if self._gl_widget:
            if self._gl_widget.auto_rotate:
                self._gl_widget.stop_auto_rotate()
            else:
                self._gl_widget.start_auto_rotate()
    
    def _toggle_wireframe(self):
        if self._gl_widget:
            self._gl_widget.wireframe_mode = not self._gl_widget.wireframe_mode
            self._gl_widget.update()
    
    def _toggle_mask(self):
        self._use_circular_mask = not self._use_circular_mask
        if self._use_circular_mask:
            self._apply_circular_mask()
        elif self._gl_widget:
            self._gl_widget.clearMask()
    
    def _set_size(self, size: int):
        self._size = size
        self.setFixedSize(self._size, self._size + self._drag_handle_height)
        self._gl_container.setFixedSize(self._size, self._size)
        if self._gl_widget:
            self._gl_widget.setFixedSize(self._size, self._size)
            self._apply_circular_mask()
    
    def _reset_view(self):
        if self._gl_widget:
            self._gl_widget.reset_view()
        self._set_size(300)
    
    def _close(self):
        self.hide()
        self.closed.emit()


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
        self.setStyleSheet("""
            QFrame {
                border: 2px solid #45475a;
                border-radius: 12px;
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
        """Draw avatar."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        if self.pixmap:
            x = (self.width() - self.pixmap.width()) // 2
            y = (self.height() - self.pixmap.height()) // 2
            painter.drawPixmap(x, y, self.pixmap)
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
    
    parent.show_overlay_btn = QPushButton("Show on Desktop")
    parent.show_overlay_btn.setCheckable(True)
    parent.show_overlay_btn.clicked.connect(lambda: _toggle_overlay(parent))
    top_row.addWidget(parent.show_overlay_btn)
    
    top_row.addStretch()
    left_panel.addLayout(top_row)
    
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
    
    parent.auto_rotate_btn = QPushButton("🔄 Auto-Rotate")
    parent.auto_rotate_btn.setCheckable(True)
    parent.auto_rotate_btn.setToolTip("Toggle auto-rotation")
    parent.auto_rotate_btn.clicked.connect(lambda: _toggle_auto_rotate(parent))
    parent.auto_rotate_btn.setVisible(False)
    parent.auto_rotate_btn.setStyleSheet("""
        QPushButton {
            background: #2d2d3d;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 4px 8px;
        }
        QPushButton:checked {
            background: #45475a;
            border-color: #89b4fa;
        }
    """)
    viewer_controls.addWidget(parent.auto_rotate_btn)
    
    parent.reset_view_btn = QPushButton("🏠 Reset")
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
    btn_refresh.setFixedWidth(60)
    btn_refresh.setToolTip("Refresh list")
    btn_refresh.clicked.connect(lambda: _refresh_list(parent))
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
    
    # === Built-in 2D Avatar Option ===
    builtin_group = QGroupBox("Simple 2D Avatar")
    builtin_layout = QVBoxLayout()
    
    builtin_info = QLabel("Use a simple animated 2D avatar instead of loading a file. Good for quick setup.")
    builtin_info.setStyleSheet("color: #6c7086; font-size: 10px;")
    builtin_info.setWordWrap(True)
    builtin_layout.addWidget(builtin_info)
    
    parent.use_builtin_btn = QPushButton("Use Built-in 2D Avatar")
    parent.use_builtin_btn.clicked.connect(lambda: _use_builtin_sprite(parent))
    parent.use_builtin_btn.setEnabled(avatar_module_enabled)
    builtin_layout.addWidget(parent.use_builtin_btn)
    
    builtin_group.setLayout(builtin_layout)
    right_panel.addWidget(builtin_group)
    
    # === Color Customization ===
    color_group = QGroupBox("Colors")
    color_layout = QVBoxLayout()
    
    # Color preset combo
    preset_row = QHBoxLayout()
    preset_row.addWidget(QLabel("Preset:"))
    parent.color_preset_combo = QComboBox()
    parent.color_preset_combo.addItems([
        "Default", "Warm", "Cool", "Nature", "Sunset", 
        "Ocean", "Fire", "Dark", "Pastel"
    ])
    parent.color_preset_combo.currentTextChanged.connect(
        lambda preset: _apply_color_preset(parent, preset.lower())
    )
    preset_row.addWidget(parent.color_preset_combo, stretch=1)
    color_layout.addLayout(preset_row)
    
    # Individual color pickers
    color_btn_row = QHBoxLayout()
    
    parent.primary_color_btn = QPushButton("Primary")
    parent.primary_color_btn.setStyleSheet("background: #6366f1; color: white;")
    parent.primary_color_btn.clicked.connect(lambda: _pick_color(parent, "primary"))
    color_btn_row.addWidget(parent.primary_color_btn)
    
    parent.secondary_color_btn = QPushButton("Secondary")
    parent.secondary_color_btn.setStyleSheet("background: #8b5cf6; color: white;")
    parent.secondary_color_btn.clicked.connect(lambda: _pick_color(parent, "secondary"))
    color_btn_row.addWidget(parent.secondary_color_btn)
    
    parent.accent_color_btn = QPushButton("Accent")
    parent.accent_color_btn.setStyleSheet("background: #10b981; color: white;")
    parent.accent_color_btn.clicked.connect(lambda: _pick_color(parent, "accent"))
    color_btn_row.addWidget(parent.accent_color_btn)
    
    color_layout.addLayout(color_btn_row)
    color_group.setLayout(color_layout)
    right_panel.addWidget(color_group)
    
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
    
    # === 3D Viewer Settings (Sketchfab-style) ===
    if HAS_OPENGL and HAS_TRIMESH:
        viewer_group = QGroupBox("3D Viewer Settings")
        viewer_layout = QVBoxLayout()
        
        # Wireframe toggle
        parent.wireframe_checkbox = QCheckBox("Wireframe Mode")
        parent.wireframe_checkbox.toggled.connect(lambda c: _set_wireframe(parent, c))
        viewer_layout.addWidget(parent.wireframe_checkbox)
        
        # Show grid toggle
        parent.show_grid_checkbox = QCheckBox("Show Grid Floor")
        parent.show_grid_checkbox.setChecked(True)
        parent.show_grid_checkbox.toggled.connect(lambda c: _set_show_grid(parent, c))
        viewer_layout.addWidget(parent.show_grid_checkbox)
        
        # Lighting controls
        light_row = QHBoxLayout()
        light_row.addWidget(QLabel("Lighting:"))
        parent.light_slider = QSlider(Qt.Horizontal if hasattr(Qt, 'Horizontal') else 0x01)
        parent.light_slider.setRange(0, 200)
        parent.light_slider.setValue(100)
        parent.light_slider.valueChanged.connect(lambda v: _set_lighting(parent, v / 100.0))
        light_row.addWidget(parent.light_slider)
        viewer_layout.addLayout(light_row)
        
        # Ambient strength
        ambient_row = QHBoxLayout()
        ambient_row.addWidget(QLabel("Ambient:"))
        parent.ambient_slider = QSlider(Qt.Horizontal if hasattr(Qt, 'Horizontal') else 0x01)
        parent.ambient_slider.setRange(0, 100)
        parent.ambient_slider.setValue(15)
        parent.ambient_slider.valueChanged.connect(lambda v: _set_ambient(parent, v / 100.0))
        ambient_row.addWidget(parent.ambient_slider)
        viewer_layout.addLayout(ambient_row)
        
        # Auto-rotate speed
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Rotate Speed:"))
        parent.rotate_speed_slider = QSlider(Qt.Horizontal if hasattr(Qt, 'Horizontal') else 0x01)
        parent.rotate_speed_slider.setRange(1, 30)
        parent.rotate_speed_slider.setValue(5)
        parent.rotate_speed_slider.valueChanged.connect(lambda v: _set_rotate_speed(parent, v / 10.0))
        speed_row.addWidget(parent.rotate_speed_slider)
        viewer_layout.addLayout(speed_row)
        
        viewer_group.setLayout(viewer_layout)
        right_panel.addWidget(viewer_group)
    
    # === Reset Buttons ===
    reset_group = QGroupBox("Reset Options")
    reset_layout = QVBoxLayout()
    
    # Reset preview (camera, settings)
    parent.reset_preview_btn = QPushButton("🔄 Reset Preview")
    parent.reset_preview_btn.setToolTip("Reset 3D preview camera and settings")
    parent.reset_preview_btn.clicked.connect(lambda: _reset_preview(parent))
    reset_layout.addWidget(parent.reset_preview_btn)
    
    # Reset desktop overlay
    parent.reset_overlay_btn = QPushButton("🖥️ Reset Desktop Overlay")
    parent.reset_overlay_btn.setToolTip("Reset desktop avatar position and size")
    parent.reset_overlay_btn.clicked.connect(lambda: _reset_overlay(parent))
    reset_layout.addWidget(parent.reset_overlay_btn)
    
    # Reset all (both preview and overlay)
    parent.reset_all_btn = QPushButton("↩️ Reset Everything")
    parent.reset_all_btn.setToolTip("Reset all avatar settings to default")
    parent.reset_all_btn.clicked.connect(lambda: _reset_all_avatar(parent))
    parent.reset_all_btn.setStyleSheet("background: #f38ba8; color: #1e1e2e;")
    reset_layout.addWidget(parent.reset_all_btn)
    
    reset_group.setLayout(reset_layout)
    right_panel.addWidget(reset_group)
    
    right_panel.addStretch()
    
    # Info
    info = QLabel("Desktop avatar: Drag to move • Scroll to resize • Right-click to hide • Double-click to reset size")
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
    
    # Show default sprite on initialization
    parent._using_builtin_sprite = True
    _show_default_preview(parent)
    
    return widget


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
        
        # If count changed, refresh
        expected = getattr(parent, '_last_file_count', 0) - 1  # Minus the "-- Select --" item
        if count != expected:
            _refresh_list(parent)
            parent._last_file_count = parent.avatar_combo.count()
            parent.avatar_status.setText(f"Found {count} avatars (auto-refreshed)")
            parent.avatar_status.setStyleSheet("color: #a6e3a1;")
    except Exception:
        pass  # Silently ignore errors in background check


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
        
        # Also update overlay if visible
        if parent._overlay and parent._overlay.isVisible():
            pixmap = parent.avatar_preview_2d.original_pixmap
            if pixmap:
                scaled = pixmap.scaled(280, 280, Qt_KeepAspectRatio, Qt_SmoothTransformation)
                parent._overlay.set_avatar(scaled)


def _set_expression(parent, expression: str):
    """Set avatar expression and update preview."""
    parent.current_expression = expression
    if hasattr(parent, 'expression_label'):
        parent.expression_label.setText(f"Current: {expression}")
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
        
        # Update overlay too
        if parent._overlay and parent._overlay.isVisible():
            pixmap = parent.avatar_preview_2d.original_pixmap
            if pixmap:
                scaled = pixmap.scaled(280, 280, Qt_KeepAspectRatio, Qt_SmoothTransformation)
                parent._overlay.set_avatar(scaled)
    
    parent.avatar_status.setText(f"Expression: {expression}")
    parent.avatar_status.setStyleSheet("color: #a6e3a1;")


def _apply_color_preset(parent, preset: str):
    """Apply a color preset."""
    presets = {
        "default": {"primary": "#6366f1", "secondary": "#8b5cf6", "accent": "#10b981"},
        "warm": {"primary": "#f59e0b", "secondary": "#ef4444", "accent": "#fbbf24"},
        "cool": {"primary": "#3b82f6", "secondary": "#06b6d4", "accent": "#8b5cf6"},
        "nature": {"primary": "#10b981", "secondary": "#22c55e", "accent": "#84cc16"},
        "sunset": {"primary": "#f59e0b", "secondary": "#ec4899", "accent": "#8b5cf6"},
        "ocean": {"primary": "#06b6d4", "secondary": "#0ea5e9", "accent": "#3b82f6"},
        "fire": {"primary": "#ef4444", "secondary": "#f59e0b", "accent": "#fbbf24"},
        "dark": {"primary": "#1e293b", "secondary": "#475569", "accent": "#64748b"},
        "pastel": {"primary": "#a78bfa", "secondary": "#f0abfc", "accent": "#fbcfe8"},
    }
    
    if preset in presets:
        colors = presets[preset]
        parent._current_colors = colors.copy()
        
        # Update color buttons
        parent.primary_color_btn.setStyleSheet(f"background: {colors['primary']}; color: white;")
        parent.secondary_color_btn.setStyleSheet(f"background: {colors['secondary']}; color: white;")
        parent.accent_color_btn.setStyleSheet(f"background: {colors['accent']}; color: white;")
        
        # Update preview
        if parent._using_builtin_sprite:
            _set_expression(parent, parent.current_expression)


def _pick_color(parent, color_type: str):
    """Open color picker for specified color type."""
    current = parent._current_colors.get(color_type, "#ffffff")
    color = QColorDialog.getColor(QColor(current), parent, f"Pick {color_type.title()} Color")
    
    if color.isValid():
        hex_color = color.name()
        parent._current_colors[color_type] = hex_color
        
        # Update button style
        btn_map = {
            "primary": parent.primary_color_btn,
            "secondary": parent.secondary_color_btn,
            "accent": parent.accent_color_btn
        }
        if color_type in btn_map:
            btn_map[color_type].setStyleSheet(f"background: {hex_color}; color: white;")
        
        # Update preview
        if parent._using_builtin_sprite:
            _set_expression(parent, parent.current_expression)


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


def _auto_design_avatar(parent):
    """Let AI auto-design avatar based on personality."""
    try:
        appearance = parent._avatar_controller.auto_design()
        
        if appearance:
            # Update colors
            parent._current_colors = {
                "primary": appearance.primary_color,
                "secondary": appearance.secondary_color,
                "accent": appearance.accent_color
            }
            
            # Update UI
            parent.primary_color_btn.setStyleSheet(f"background: {appearance.primary_color}; color: white;")
            parent.secondary_color_btn.setStyleSheet(f"background: {appearance.secondary_color}; color: white;")
            parent.accent_color_btn.setStyleSheet(f"background: {appearance.accent_color}; color: white;")
            
            # Use built-in sprite with AI colors
            parent._using_builtin_sprite = True
            _set_expression(parent, appearance.default_expression)
            
            explanation = parent._avatar_controller.explain_appearance()
            parent.avatar_status.setText(f"AI designed: {explanation[:50]}...")
            parent.avatar_status.setStyleSheet("color: #a6e3a1;")
        else:
            parent.avatar_status.setText("Link personality first (in Training tab)")
            parent.avatar_status.setStyleSheet("color: #fab387;")
    except Exception as e:
        parent.avatar_status.setText(f"Auto-design failed: {str(e)[:30]}")
        parent.avatar_status.setStyleSheet("color: #f38ba8;")


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
            
            # Load the model into 3D overlay
            if parent._current_path:
                parent._overlay_3d.load_model(str(parent._current_path))
                parent._overlay_3d.show()
                parent._overlay_3d.raise_()
                parent.show_overlay_btn.setText("Hide from Desktop")
                parent.avatar_status.setText("3D avatar on desktop! Drag to move, R to rotate, right-click for menu.")
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
            
            # Get current pixmap, or generate default if none
            pixmap = parent.avatar_preview_2d.original_pixmap
            if not pixmap:
                _show_default_preview(parent)
                pixmap = parent.avatar_preview_2d.original_pixmap
            
            if pixmap:
                scaled = pixmap.scaled(280, 280, Qt_KeepAspectRatio, Qt_SmoothTransformation)
                parent._overlay.set_avatar(scaled)
                parent._overlay.show()
                parent._overlay.raise_()
                parent.show_overlay_btn.setText("Hide from Desktop")
                parent.avatar_status.setText("Avatar on desktop! Drag to move, scroll to resize, right-click for menu.")
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
        parent.show_overlay_btn.setText("Show on Desktop")
        parent.avatar_status.setText("Avatar hidden from desktop")
        parent.avatar_status.setStyleSheet("color: #6c7086;")


def _on_overlay_closed(parent):
    """Handle overlay closed."""
    parent.show_overlay_btn.setChecked(False)
    parent.show_overlay_btn.setText("Show on Desktop")


def _toggle_3d_render(parent, enabled):
    """Toggle between 2D preview and 3D rendering."""
    parent._using_3d_render = enabled
    
    if enabled and parent.avatar_preview_3d:
        parent.avatar_preview_2d.setVisible(False)
        parent.avatar_preview_3d.setVisible(True)
        parent.reset_view_btn.setVisible(True)
        parent.auto_rotate_btn.setVisible(True)
        
        # Load model into 3D viewer if we have a 3D model
        if parent._is_3d_model and parent._current_path:
            parent.avatar_preview_3d.load_model(str(parent._current_path))
    else:
        parent.avatar_preview_2d.setVisible(True)
        if parent.avatar_preview_3d:
            parent.avatar_preview_3d.setVisible(False)
        parent.reset_view_btn.setVisible(False)
        parent.auto_rotate_btn.setVisible(False)
        parent.auto_rotate_btn.setVisible(False)


def _reset_view(parent):
    """Reset 3D view."""
    if parent.avatar_preview_3d:
        parent.avatar_preview_3d.reset_view()


def _toggle_auto_rotate(parent):
    """Toggle auto-rotation on 3D preview."""
    if parent.avatar_preview_3d:
        if parent.auto_rotate_btn.isChecked():
            parent.avatar_preview_3d.start_auto_rotate()
        else:
            parent.avatar_preview_3d.stop_auto_rotate()


def _set_wireframe(parent, enabled: bool):
    """Toggle wireframe mode."""
    if parent.avatar_preview_3d:
        parent.avatar_preview_3d.wireframe_mode = enabled
        parent.avatar_preview_3d.update()


def _set_show_grid(parent, enabled: bool):
    """Toggle grid floor."""
    if parent.avatar_preview_3d:
        parent.avatar_preview_3d.show_grid = enabled
        parent.avatar_preview_3d.update()


def _set_lighting(parent, intensity: float):
    """Set lighting intensity."""
    if parent.avatar_preview_3d:
        parent.avatar_preview_3d.light_intensity = intensity
        parent.avatar_preview_3d._update_lighting()
        parent.avatar_preview_3d.update()


def _set_ambient(parent, strength: float):
    """Set ambient light strength."""
    if parent.avatar_preview_3d:
        parent.avatar_preview_3d.ambient_strength = strength
        parent.avatar_preview_3d._update_lighting()
        parent.avatar_preview_3d.update()


def _set_rotate_speed(parent, speed: float):
    """Set auto-rotate speed."""
    if parent.avatar_preview_3d:
        parent.avatar_preview_3d.auto_rotate_speed = speed


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
        if hasattr(parent, 'rotate_speed_slider'):
            parent.rotate_speed_slider.setValue(5)
        if hasattr(parent, 'auto_rotate_btn'):
            parent.auto_rotate_btn.setChecked(False)
    
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


def _refresh_list(parent):
    """Refresh avatar list - scans all subdirectories too."""
    parent.avatar_combo.clear()
    parent.avatar_combo.addItem("-- Select Avatar --", None)
    
    # JSON configs
    if AVATAR_CONFIG_DIR.exists():
        for f in sorted(AVATAR_CONFIG_DIR.glob("*.json")):
            cfg = _load_json(f)
            is_3d = cfg.get("type") == "3d" or "model_path" in cfg
            icon = "🎮" if is_3d else "🖼️"
            parent.avatar_combo.addItem(f"{icon} {f.stem}", ("config", str(f)))
    
    # Direct images
    if AVATAR_IMAGES_DIR.exists():
        for f in sorted(AVATAR_IMAGES_DIR.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                parent.avatar_combo.addItem(f"🖼️ {f.name}", ("image", str(f)))
    
    # 3D models - scan direct files AND subdirectories
    if AVATAR_MODELS_DIR.exists():
        # Direct model files
        for f in sorted(AVATAR_MODELS_DIR.iterdir()):
            if f.is_file() and f.suffix.lower() in MODEL_3D_EXTENSIONS:
                parent.avatar_combo.addItem(f"🎮 {f.name}", ("model", str(f)))
        
        # Subdirectories containing models (e.g., glados/, rurune/)
        for subdir in sorted(AVATAR_MODELS_DIR.iterdir()):
            if subdir.is_dir():
                # Look for scene.gltf or scene.glb first (common format)
                scene_gltf = subdir / "scene.gltf"
                scene_glb = subdir / "scene.glb"
                
                if scene_gltf.exists():
                    parent.avatar_combo.addItem(f"🎮 {subdir.name}", ("model", str(scene_gltf)))
                elif scene_glb.exists():
                    parent.avatar_combo.addItem(f"🎮 {subdir.name}", ("model", str(scene_glb)))
                else:
                    # Look for any model file in subdirectory
                    for f in sorted(subdir.glob("*")):
                        if f.suffix.lower() in MODEL_3D_EXTENSIONS:
                            parent.avatar_combo.addItem(f"🎮 {subdir.name}/{f.name}", ("model", str(f)))
                            break  # Only add first model found
    
    # Update status
    count = parent.avatar_combo.count() - 1  # Exclude "-- Select --"
    parent.avatar_status.setText(f"Found {count} avatars")
    parent.avatar_status.setStyleSheet("color: #a6e3a1;" if count > 0 else "color: #6c7086;")


def _load_json(path: Path) -> dict:
    """Load JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return {}


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
    painter.drawText(0, 40, size, 60, Qt_AlignCenter, "📦")
    
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
    
    # Load into the backend controller
    if parent._is_3d_model:
        parent._avatar_controller.load_model(str(path))
        parent.avatar_status.setText(f"✓ Loaded 3D: {path.name}")
        
        # If 3D rendering is enabled, load into GL widget
        if parent._using_3d_render and parent.avatar_preview_3d:
            parent.avatar_preview_3d.load_model(str(path))
    else:
        parent._avatar_controller.appearance.model_path = str(path)
        parent.avatar_status.setText(f"✓ Loaded: {path.name}")
    
    parent.avatar_status.setStyleSheet("color: #a6e3a1;")
    
    # Update overlay if visible
    if parent._overlay and parent._overlay.isVisible():
        pixmap = parent.avatar_preview_2d.original_pixmap
        if pixmap:
            scaled = pixmap.scaled(280, 280, Qt_KeepAspectRatio, Qt_SmoothTransformation)
            parent._overlay.set_avatar(scaled)


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
        parent.avatar_combo.addItem(f"📄 {path.stem}", ("config", str(path)))
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
