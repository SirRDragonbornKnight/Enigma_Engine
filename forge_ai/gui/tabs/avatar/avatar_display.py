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
from PyQt5.QtSvg import QSvgWidget

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
HAS_PYRENDER = False
trimesh = None
np = None
pyrender = None

try:
    import trimesh as _trimesh
    import numpy as _np
    trimesh = _trimesh
    np = _np
    HAS_TRIMESH = True
except ImportError:
    pass

try:
    import pyrender as _pyrender
    from PIL import Image as PILImage
    pyrender = _pyrender
    HAS_PYRENDER = True
except ImportError:
    PILImage = None  # type: ignore
    pass

# OpenGL imports with explicit names to avoid wildcard import issues
try:
    import OpenGL.GL as GL
    import OpenGL.GLU as GLU
    HAS_OPENGL = True
except ImportError:
    GL = None  # type: ignore
    GLU = None  # type: ignore

# Try importing USD support
HAS_USD = False
try:
    from pxr import Usd, UsdGeom
    HAS_USD = True
except ImportError:
    pass

# Supported file extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
MODEL_3D_EXTENSIONS = {'.glb', '.gltf', '.obj', '.fbx', '.dae', '.usd', '.usda', '.usdc', '.usdz'}
ALL_AVATAR_EXTENSIONS = IMAGE_EXTENSIONS | MODEL_3D_EXTENSIONS

# Avatar directories
AVATAR_CONFIG_DIR = Path(CONFIG["data_dir"]) / "avatar"
AVATAR_MODELS_DIR = AVATAR_CONFIG_DIR / "models"
AVATAR_IMAGES_DIR = AVATAR_CONFIG_DIR / "images"


def convert_usd_to_glb(usd_path: Path) -> Optional[Path]:
    """Convert USD file to GLB format for compatibility."""
    if not HAS_USD or not HAS_TRIMESH:
        return None
    
    try:
        from pxr import Usd, UsdGeom, Gf
        import numpy as np
        
        # Output path
        glb_path = usd_path.parent / f"{usd_path.stem}_converted.glb"
        
        # If already converted, return cached version
        if glb_path.exists():
            return glb_path
        
        # Open USD stage
        stage = Usd.Stage.Open(str(usd_path))
        if not stage:
            return None
        
        # Collect all mesh data
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                
                # Get points
                points_attr = mesh.GetPointsAttr()
                if points_attr:
                    points = points_attr.Get()
                    if points:
                        vertices = np.array([[p[0], p[1], p[2]] for p in points])
                        all_vertices.append(vertices)
                        
                        # Get face indices
                        face_counts = mesh.GetFaceVertexCountsAttr().Get()
                        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
                        
                        if face_counts and face_indices:
                            idx = 0
                            for count in face_counts:
                                if count == 3:
                                    face = [face_indices[idx + i] + vertex_offset for i in range(3)]
                                    all_faces.append(face)
                                elif count == 4:
                                    # Triangulate quad
                                    face1 = [face_indices[idx] + vertex_offset, 
                                            face_indices[idx + 1] + vertex_offset, 
                                            face_indices[idx + 2] + vertex_offset]
                                    face2 = [face_indices[idx] + vertex_offset, 
                                            face_indices[idx + 2] + vertex_offset, 
                                            face_indices[idx + 3] + vertex_offset]
                                    all_faces.extend([face1, face2])
                                idx += count
                        
                        vertex_offset += len(vertices)
        
        if not all_vertices:
            return None
        
        # Combine all meshes
        combined_vertices = np.vstack(all_vertices)
        combined_faces = np.array(all_faces)
        
        # Create trimesh
        mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)
        
        # Export to GLB
        mesh.export(str(glb_path), file_type='glb')
        
        return glb_path
        
    except Exception as e:
        print(f"USD conversion error: {e}")
        return None


class OpenGL3DWidget(QOpenGLWidget):
    """OpenGL widget for rendering 3D models."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh = None
        self.vertices = None
        self.faces = None
        self.normals = None
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = 2.0
        self.last_pos = None
        self.setMinimumSize(200, 200)
        
    def load_model(self, path: str) -> bool:
        """Load a 3D model file."""
        if not HAS_TRIMESH or trimesh is None or np is None:
            return False
        try:
            model_path = Path(path)
            
            # Check if this is a USD file that needs conversion
            if model_path.suffix.lower() in {'.usd', '.usda', '.usdc', '.usdz'}:
                converted = convert_usd_to_glb(model_path)
                if converted:
                    path = str(converted)
                else:
                    print(f"Could not convert USD file: {model_path}")
                    return False
            
            scene = trimesh.load(str(path))  # type: ignore[union-attr]
            # Convert scene to single mesh if needed
            if hasattr(scene, 'geometry'):
                meshes = list(scene.geometry.values())  # type: ignore[union-attr]
                if meshes:
                    self.mesh = trimesh.util.concatenate(meshes)  # type: ignore[union-attr]
                else:
                    return False
            else:
                self.mesh = scene
            
            # Center and scale the mesh
            self.mesh.vertices -= self.mesh.centroid  # type: ignore[union-attr]
            scale = 1.0 / max(self.mesh.extents)  # type: ignore[union-attr]
            self.mesh.vertices *= scale  # type: ignore[union-attr]
            
            self.vertices = np.array(self.mesh.vertices, dtype=np.float32)  # type: ignore[union-attr]
            self.faces = np.array(self.mesh.faces, dtype=np.uint32)  # type: ignore[union-attr]
            if hasattr(self.mesh, 'vertex_normals'):
                self.normals = np.array(self.mesh.vertex_normals, dtype=np.float32)  # type: ignore[union-attr]
            else:
                self.normals = None
            
            self.update()
            return True
        except Exception as e:
            print(f"Error loading 3D model: {e}")
            return False
    
    def reset_view(self):
        """Reset rotation to default."""
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = 2.0
        self.update()
    
    def initializeGL(self):
        """Initialize OpenGL settings."""
        if not HAS_OPENGL or GL is None:
            return
        GL.glClearColor(0.12, 0.12, 0.18, 1.0)  # Dark background
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
        
        # Light setup
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [1, 1, 1, 0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [1, 1, 1, 1])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        
    def resizeGL(self, w, h):
        """Handle resize."""
        if not HAS_OPENGL or GL is None or GLU is None:
            return
        GL.glViewport(0, 0, w, h)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        aspect = w / h if h > 0 else 1
        GLU.gluPerspective(45, aspect, 0.1, 100.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        
    def paintGL(self):
        """Render the 3D model."""
        if not HAS_OPENGL or GL is None:
            return
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # type: ignore[operator]
        GL.glLoadIdentity()
        
        # Camera position
        GL.glTranslatef(0, 0, -self.zoom)
        GL.glRotatef(self.rotation_x, 1, 0, 0)
        GL.glRotatef(self.rotation_y, 0, 1, 0)
        
        if self.vertices is not None and self.faces is not None:
            GL.glColor3f(0.7, 0.7, 0.8)  # Light gray-blue
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glVertexPointer(3, GL.GL_FLOAT, 0, self.vertices)
            
            if self.normals is not None:
                GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
                GL.glNormalPointer(GL.GL_FLOAT, 0, self.normals)
            
            GL.glDrawElements(GL.GL_TRIANGLES, len(self.faces) * 3, GL.GL_UNSIGNED_INT, self.faces)
            
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            if self.normals is not None:
                GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
    
    def mousePressEvent(self, a0):  # type: ignore
        """Start drag."""
        self.last_pos = a0.pos()
        
    def mouseMoveEvent(self, a0):  # type: ignore
        """Rotate on drag."""
        if self.last_pos is not None:
            dx = a0.x() - self.last_pos.x()
            dy = a0.y() - self.last_pos.y()
            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
            self.last_pos = a0.pos()
            self.update()
            
    def mouseReleaseEvent(self, a0):  # type: ignore
        """End drag."""
        self.last_pos = None
        
    def wheelEvent(self, a0):  # type: ignore
        """Zoom with scroll wheel."""
        delta = a0.angleDelta().y() / 120
        self.zoom = max(0.5, min(10.0, self.zoom - delta * 0.2))
        self.update()
        
    def mouseDoubleClickEvent(self, a0):  # type: ignore
        """Reset view on double click."""
        self.reset_view()


class AvatarOverlayWindow(QWidget):
    """Transparent overlay window for desktop avatar display.
    
    Features:
    - Drag anywhere to move
    - Right-click to hide
    - Scroll wheel to resize
    - Always on top of other windows
    - Background color control (AI can change)
    - Quick avatar switching
    """
    
    closed = pyqtSignal()
    avatar_changed = pyqtSignal(str)  # Signal when avatar changes (path)
    
    def __init__(self, parent_tab=None):
        super().__init__(None)
        self._parent_tab = parent_tab  # Reference to main avatar tab for syncing
        
        # Transparent, always-on-top, no taskbar, but accept mouse input
        self.setWindowFlags(
            Qt_FramelessWindowHint |
            Qt_WindowStaysOnTopHint |
            Qt_Tool
        )
        self.setAttribute(Qt_WA_TranslucentBackground, True)
        # Important: Make sure we receive mouse events
        self.setAttribute(Qt.WA_NoSystemBackground, True) if hasattr(Qt, 'WA_NoSystemBackground') else None
        
        self._size = 300
        self.setFixedSize(self._size, self._size)
        self.move(100, 100)
        
        self.pixmap = None
        self._original_pixmap = None
        self._drag_pos = None
        
        # Background settings (AI controllable)
        self._bg_color = QColor(0, 0, 0, 0)  # Start transparent/clear
        self._bg_shape = "none"  # none, circle, square, glow
        
        # Available avatars cache (for quick menu)
        self._available_avatars = []
        
        # Enable mouse tracking for visual cursor feedback
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt_OpenHandCursor))
        
    def set_avatar(self, pixmap: QPixmap):
        """Set avatar image."""
        self._original_pixmap = pixmap
        self._update_scaled_pixmap()
    
    def set_background(self, color: QColor = None, shape: str = "none"):
        """Set avatar background (AI controllable).
        
        Args:
            color: Background color (None = transparent)
            shape: 'none', 'circle', 'square', 'glow'
        """
        self._bg_color = color if color else QColor(0, 0, 0, 0)
        self._bg_shape = shape
        self.update()
    
    def set_available_avatars(self, avatars: list):
        """Set list of available avatars for quick menu."""
        self._available_avatars = avatars
        
    def _update_scaled_pixmap(self):
        """Update scaled pixmap to match current size."""
        if self._original_pixmap:
            self.pixmap = self._original_pixmap.scaled(
                self._size - 20, self._size - 20,
                Qt_KeepAspectRatio, Qt_SmoothTransformation
            )
        self.update()
        
    def paintEvent(self, a0):
        """Draw avatar with background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        if self.pixmap:
            x = (self.width() - self.pixmap.width()) // 2
            y = (self.height() - self.pixmap.height()) // 2
            
            # Draw background based on shape
            painter.setPen(Qt_NoPen)
            if self._bg_shape == "circle":
                painter.setBrush(self._bg_color)
                painter.drawEllipse(x - 5, y - 5, self.pixmap.width() + 10, self.pixmap.height() + 10)
            elif self._bg_shape == "square":
                painter.setBrush(self._bg_color)
                painter.drawRoundedRect(x - 5, y - 5, self.pixmap.width() + 10, self.pixmap.height() + 10, 10, 10)
            elif self._bg_shape == "glow":
                # Glow effect - semi-transparent with blur simulation
                glow_color = QColor(self._bg_color)
                for i in range(3):
                    glow_color.setAlpha(max(0, self._bg_color.alpha() - i * 30))
                    painter.setBrush(glow_color)
                    offset = (3 - i) * 5
                    painter.drawEllipse(x - offset, y - offset, 
                                       self.pixmap.width() + offset * 2, 
                                       self.pixmap.height() + offset * 2)
            # else: "none" - no background
            
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
            a0.accept()
            
    def mouseMoveEvent(self, a0):  # type: ignore
        """Drag to move window."""
        if self._drag_pos is not None and a0.buttons() == Qt_LeftButton:
            self.move(a0.globalPos() - self._drag_pos)
            a0.accept()
            
    def mouseReleaseEvent(self, a0):  # type: ignore
        """End drag."""
        self._drag_pos = None
        self.setCursor(QCursor(Qt_OpenHandCursor))
        
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
        
        # === Quick Avatar Switch ===
        if self._available_avatars:
            avatar_menu = menu.addMenu("🔄 Switch Avatar")
            for name, data in self._available_avatars[:15]:  # Limit to 15
                action = avatar_menu.addAction(name)
                action.triggered.connect(lambda checked, d=data: self._quick_switch_avatar(d))
            if len(self._available_avatars) > 15:
                avatar_menu.addSeparator()
                more_action = avatar_menu.addAction("📂 Browse more...")
                more_action.triggered.connect(self._change_avatar_file)
        else:
            change_avatar = menu.addAction("📂 Change Avatar...")
            change_avatar.triggered.connect(self._change_avatar_file)
        
        menu.addSeparator()
        
        # === Background Options ===
        bg_menu = menu.addMenu("🎨 Background")
        
        # Clear/transparent (default)
        clear_action = bg_menu.addAction("✨ Clear (Transparent)")
        clear_action.triggered.connect(lambda: self.set_background(None, "none"))
        
        bg_menu.addSeparator()
        bg_menu.addAction("── Shapes ──").setEnabled(False)
        
        # Shape options with default dark color
        circle_action = bg_menu.addAction("⚫ Dark Circle")
        circle_action.triggered.connect(lambda: self.set_background(QColor(30, 30, 46, 180), "circle"))
        
        glow_action = bg_menu.addAction("💫 Glow Effect")  
        glow_action.triggered.connect(lambda: self.set_background(QColor(99, 102, 241, 100), "glow"))
        
        square_action = bg_menu.addAction("⬛ Rounded Square")
        square_action.triggered.connect(lambda: self.set_background(QColor(30, 30, 46, 180), "square"))
        
        bg_menu.addSeparator()
        bg_menu.addAction("── Colors ──").setEnabled(False)
        
        # Color presets
        colors = [
            ("🔵 Blue Glow", QColor(59, 130, 246, 100), "glow"),
            ("🟣 Purple Glow", QColor(139, 92, 246, 100), "glow"),
            ("🟢 Green Glow", QColor(16, 185, 129, 100), "glow"),
            ("🔴 Red Glow", QColor(239, 68, 68, 100), "glow"),
            ("🟡 Gold Glow", QColor(245, 158, 11, 100), "glow"),
        ]
        for name, color, shape in colors:
            action = bg_menu.addAction(name)
            action.triggered.connect(lambda checked, c=color, s=shape: self.set_background(c, s))
        
        menu.addSeparator()
        
        # Style submenu (Blob vs Anime)
        style_menu = menu.addMenu("😊 Expression")
        blob_action = style_menu.addAction("Blob (Default)")
        blob_action.triggered.connect(lambda: self._change_expression("idle"))
        anime_action = style_menu.addAction("Anime Girl")
        anime_action.triggered.connect(lambda: self._change_expression("anime_idle"))
        
        style_menu.addSeparator()
        
        # Quick expressions
        quick_expr = ["happy", "thinking", "surprised", "love", "sleeping"]
        for expr in quick_expr:
            action = style_menu.addAction(f"  {expr.title()}")
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
        
        menu.addSeparator()
        
        # Close
        close_action = menu.addAction("❌ Close Avatar")
        close_action.triggered.connect(self._close_avatar)
        
        menu.exec_(a0.globalPos())
    
    def _quick_switch_avatar(self, data):
        """Quick switch to a different avatar."""
        if not data:
            return
        
        file_type, path_str = data
        path = Path(path_str)
        
        if not path.exists():
            return
        
        # Emit signal so parent tab can update
        self.avatar_changed.emit(path_str)
        
        # For images, load directly
        if file_type == "image" or path.suffix.lower() in IMAGE_EXTENSIONS:
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                self.set_avatar(pixmap)
        elif file_type == "model" or path.suffix.lower() in MODEL_3D_EXTENSIONS:
            # For 3D models, render a preview and use that
            if self._parent_tab:
                # Trigger render in parent tab and get the pixmap
                self._parent_tab._current_path = path
                self._parent_tab._is_3d_model = True
                _preview_3d_model(self._parent_tab, path)
                # Get the rendered pixmap
                pixmap = self._parent_tab.avatar_preview_2d.original_pixmap
                if pixmap:
                    self.set_avatar(pixmap)
        
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
            from PyQt5.QtSvg import QSvgRenderer
            from PyQt5.QtCore import QByteArray
            renderer = QSvgRenderer(QByteArray(svg_data.encode('utf-8')))
            pixmap = QPixmap(280, 280)
            pixmap.fill(QColor(0, 0, 0, 0))
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            self.set_avatar(pixmap)
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
    
    def _change_avatar_file(self):
        """Open file dialog to change avatar."""
        from PyQt5.QtWidgets import QFileDialog
        
        # Build filter string
        img_exts = " ".join(f"*{ext}" for ext in IMAGE_EXTENSIONS)
        model_exts = " ".join(f"*{ext}" for ext in MODEL_3D_EXTENSIONS)
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Avatar",
            str(AVATAR_MODELS_DIR),
            f"All Avatar Files ({img_exts} {model_exts});;Images ({img_exts});;3D Models ({model_exts})"
        )
        
        if file_path:
            path = Path(file_path)
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                # Load image directly
                pixmap = QPixmap(str(path))
                if not pixmap.isNull():
                    self.set_avatar(pixmap)
            elif path.suffix.lower() in MODEL_3D_EXTENSIONS:
                # For 3D models, show a message (would need full 3D rendering setup)
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(
                    self, "3D Model",
                    f"3D model loaded: {path.name}\n\n"
                    "For full 3D rendering, use the Avatar tab in the main GUI."
                )
        
    def mouseDoubleClickEvent(self, a0):  # type: ignore
        """Double-click to reset size."""
        self._size = 300
        self.setFixedSize(self._size, self._size)
        self._update_scaled_pixmap()


class LoadingOverlay(QWidget):
    """Semi-transparent loading overlay with progress bar."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt_WA_TranslucentBackground, True)
        self.setVisible(False)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt_AlignCenter)
        
        # Loading label
        self.label = QLabel("Loading avatar...")
        self.label.setStyleSheet("color: #cdd6f4; font-size: 14px; font-weight: bold;")
        self.label.setAlignment(Qt_AlignCenter)
        layout.addWidget(self.label)
        
        # Progress bar
        from PyQt5.QtWidgets import QProgressBar
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(0)  # Indeterminate
        self.progress.setFixedWidth(200)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #45475a;
                border-radius: 5px;
                background: #1e1e2e;
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #89b4fa, stop:1 #cba6f7);
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress)
        
    def paintEvent(self, a0):
        """Draw semi-transparent background."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 46, 200))
        
    def show_loading(self, text: str = "Loading avatar..."):
        """Show loading overlay."""
        self.label.setText(text)
        self.setVisible(True)
        self.raise_()
        QApplication.processEvents()
        
    def hide_loading(self):
        """Hide loading overlay."""
        self.setVisible(False)


class AvatarPreviewWidget(QFrame):
    """2D image preview with drag-to-rotate for 3D models."""
    
    expression_changed = pyqtSignal(str)  # Signal when expression changes
    rotation_changed = pyqtSignal(float, float)  # Signal when rotation changes (rx, ry)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.original_pixmap = None
        self._svg_mode = False
        self._current_svg = None
        
        # 3D rotation state
        self._is_3d = False
        self._rotation_x = 0.0  # Pitch (up/down)
        self._rotation_y = 0.0  # Yaw (left/right)
        self._last_mouse_pos = None
        self._model_path = None
        self._cached_mesh = None
        
        # Loading overlay
        self._loading_overlay = None
        
        self.setMinimumSize(250, 250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QFrame {
                border: 2px solid #45475a;
                border-radius: 12px;
                background: #1e1e2e;
            }
        """)
        self.setMouseTracking(True)
        
    def set_avatar(self, pixmap: QPixmap):
        """Set avatar image."""
        self.original_pixmap = pixmap
        self._svg_mode = False
        self._is_3d = False
        self._update_display()
    
    def set_3d_model(self, path: Path, mesh=None):
        """Set 3D model for rotatable preview."""
        self._is_3d = True
        self._model_path = path
        self._cached_mesh = mesh
        self._rotation_x = 0.0
        self._rotation_y = 0.0
        self.setCursor(QCursor(Qt_OpenHandCursor))
        
    def get_rotation(self) -> tuple:
        """Get current rotation angles."""
        return (self._rotation_x, self._rotation_y)
    
    def set_rotation(self, rx: float, ry: float):
        """Set rotation angles and re-render."""
        self._rotation_x = rx
        self._rotation_y = ry
        if self._is_3d and self._model_path:
            self._render_3d_at_rotation()
    
    def _render_3d_at_rotation(self):
        """Render 3D model at current rotation."""
        if not HAS_PYRENDER or not HAS_TRIMESH or not self._model_path:
            return
        
        try:
            # Use cached mesh if available
            if self._cached_mesh is not None:
                mesh = self._cached_mesh
            else:
                scene = trimesh.load(str(self._model_path))
                if hasattr(scene, 'geometry') and scene.geometry:
                    meshes = list(scene.geometry.values())
                    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
                elif hasattr(scene, 'vertices'):
                    mesh = scene
                else:
                    return
                
                # Center and scale
                mesh.vertices -= mesh.centroid
                if max(mesh.extents) > 0:
                    mesh.vertices *= 1.0 / max(mesh.extents)
                self._cached_mesh = mesh
            
            # Build rotation matrix
            rx = np.radians(self._rotation_x)
            ry = np.radians(self._rotation_y)
            
            # Rotation matrices
            rot_x = np.array([
                [1, 0, 0, 0],
                [0, np.cos(rx), -np.sin(rx), 0],
                [0, np.sin(rx), np.cos(rx), 0],
                [0, 0, 0, 1]
            ], dtype=float)
            
            rot_y = np.array([
                [np.cos(ry), 0, np.sin(ry), 0],
                [0, 1, 0, 0],
                [-np.sin(ry), 0, np.cos(ry), 0],
                [0, 0, 0, 1]
            ], dtype=float)
            
            model_pose = rot_y @ rot_x
            
            # Create scene
            pr_scene = pyrender.Scene(bg_color=[30/255, 30/255, 46/255, 1.0])
            pr_mesh = pyrender.Mesh.from_trimesh(mesh)
            pr_scene.add(pr_mesh, pose=model_pose)
            
            # Camera
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            camera_pose = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0.2],
                [0, 0, 1, 2.0],
                [0, 0, 0, 1]
            ], dtype=float)
            pr_scene.add(camera, pose=camera_pose)
            
            # Lights
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            pr_scene.add(light, pose=camera_pose)
            
            fill_pose = np.array([
                [0.7, 0, 0.7, 1.5],
                [0, 1, 0, 0.5],
                [-0.7, 0, 0.7, 1.5],
                [0, 0, 0, 1]
            ], dtype=float)
            fill_light = pyrender.DirectionalLight(color=[0.8, 0.85, 1.0], intensity=1.0)
            pr_scene.add(fill_light, pose=fill_pose)
            
            # Render
            r = pyrender.OffscreenRenderer(256, 256)
            color, depth = r.render(pr_scene)
            r.delete()
            
            # Convert to pixmap
            import io
            img = PILImage.fromarray(color)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            
            qimg = QImage()
            qimg.loadFromData(buffer.read())
            pixmap = QPixmap.fromImage(qimg)
            
            self.original_pixmap = pixmap
            self._svg_mode = False
            self._update_display()
            
            # Emit rotation changed signal
            self.rotation_changed.emit(self._rotation_x, self._rotation_y)
            
        except Exception as e:
            print(f"Error rendering rotated 3D: {e}")
    
    def mousePressEvent(self, a0):
        """Start drag for rotation."""
        if self._is_3d:
            self._last_mouse_pos = a0.pos()
            self.setCursor(QCursor(Qt_ClosedHandCursor))
        
    def mouseMoveEvent(self, a0):
        """Rotate on drag."""
        if self._is_3d and self._last_mouse_pos is not None:
            dx = a0.x() - self._last_mouse_pos.x()
            dy = a0.y() - self._last_mouse_pos.y()
            
            self._rotation_y += dx * 0.5
            self._rotation_x += dy * 0.5
            
            # Clamp pitch
            self._rotation_x = max(-90, min(90, self._rotation_x))
            
            self._last_mouse_pos = a0.pos()
            self._render_3d_at_rotation()
            
    def mouseReleaseEvent(self, a0):
        """End drag."""
        if self._is_3d:
            self._last_mouse_pos = None
            self.setCursor(QCursor(Qt_OpenHandCursor))
    
    def mouseDoubleClickEvent(self, a0):
        """Reset rotation on double-click."""
        if self._is_3d:
            self._rotation_x = 0.0
            self._rotation_y = 0.0
            self._render_3d_at_rotation()
    
    def set_svg_sprite(self, svg_data: str):
        """Set avatar from SVG data."""
        self._svg_mode = True
        self._current_svg = svg_data
        
        # Convert SVG to pixmap for display
        from PyQt5.QtSvg import QSvgRenderer
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
        """Draw avatar with rotation hint for 3D models."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        if self.pixmap:
            x = (self.width() - self.pixmap.width()) // 2
            y = (self.height() - self.pixmap.height()) // 2
            painter.drawPixmap(x, y, self.pixmap)
            
            # Show rotation hint for 3D models
            if self._is_3d:
                painter.setPen(QColor("#6c7086"))
                font = painter.font()
                font.setPointSize(9)
                painter.setFont(font)
                hint = f"Drag to rotate • Double-click to reset"
                if self._rotation_x != 0 or self._rotation_y != 0:
                    hint = f"↻ {self._rotation_y:.0f}° / {self._rotation_x:.0f}° • Double-click to reset"
                painter.drawText(5, self.height() - 8, hint)
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
    parent.avatar_enabled_checkbox.setEnabled(avatar_module_enabled)
    top_row.addWidget(parent.avatar_enabled_checkbox)
    
    parent.show_overlay_btn = QPushButton("Show on Desktop")
    parent.show_overlay_btn.setCheckable(True)
    parent.show_overlay_btn.clicked.connect(lambda: _toggle_overlay(parent))
    parent.show_overlay_btn.setEnabled(avatar_module_enabled)
    top_row.addWidget(parent.show_overlay_btn)
    
    top_row.addStretch()
    left_panel.addLayout(top_row)
    
    # 3D rendering toggle (only if libraries available)
    if HAS_OPENGL and HAS_TRIMESH:
        render_row = QHBoxLayout()
        parent.use_3d_render_checkbox = QCheckBox("Enable 3D Rendering (uses more resources)")
        parent.use_3d_render_checkbox.setChecked(False)
        parent.use_3d_render_checkbox.toggled.connect(lambda c: _toggle_3d_render(parent, c))
        render_row.addWidget(parent.use_3d_render_checkbox)
        render_row.addStretch()
        left_panel.addLayout(render_row)
    else:
        parent.use_3d_render_checkbox = None
    
    # Preview widgets (stacked - 2D and 3D)
    parent.avatar_preview_2d = AvatarPreviewWidget()
    left_panel.addWidget(parent.avatar_preview_2d, stretch=1)
    
    # Add loading overlay on top of preview
    parent._loading_overlay = LoadingOverlay(parent.avatar_preview_2d)
    
    if HAS_OPENGL and HAS_TRIMESH:
        parent.avatar_preview_3d = OpenGL3DWidget()
        parent.avatar_preview_3d.setVisible(False)
        left_panel.addWidget(parent.avatar_preview_3d, stretch=1)
    else:
        parent.avatar_preview_3d = None
    
    # Home/reset button
    btn_row = QHBoxLayout()
    btn_row.addStretch()
    parent.reset_view_btn = QPushButton("Reset View")
    parent.reset_view_btn.clicked.connect(lambda: _reset_view(parent))
    parent.reset_view_btn.setVisible(False)
    btn_row.addWidget(parent.reset_view_btn)
    btn_row.addStretch()
    left_panel.addLayout(btn_row)
    
    # Avatar selector
    select_row = QHBoxLayout()
    select_row.addWidget(QLabel("Avatar:"))
    parent.avatar_combo = QComboBox()
    parent.avatar_combo.setMinimumWidth(200)
    parent.avatar_combo.currentIndexChanged.connect(lambda: _on_avatar_selected(parent))
    parent.avatar_combo.setEnabled(avatar_module_enabled)
    select_row.addWidget(parent.avatar_combo, stretch=1)
    
    btn_refresh = QPushButton("↻")
    btn_refresh.setFixedWidth(30)
    btn_refresh.setToolTip("Refresh avatar list")
    btn_refresh.setStyleSheet("font-size: 14px;")
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
    parent.use_builtin_btn.clicked.connect(lambda: _use_builtin_avatar(parent))
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
    parent.color_preset_combo.setEnabled(avatar_module_enabled)
    preset_row.addWidget(parent.color_preset_combo, stretch=1)
    color_layout.addLayout(preset_row)
    
    # Individual color pickers
    color_btn_row = QHBoxLayout()
    
    parent.primary_color_btn = QPushButton("Primary")
    parent.primary_color_btn.setStyleSheet("background: #6366f1; color: white;")
    parent.primary_color_btn.clicked.connect(lambda: _pick_color(parent, "primary"))
    parent.primary_color_btn.setEnabled(avatar_module_enabled)
    color_btn_row.addWidget(parent.primary_color_btn)
    
    parent.secondary_color_btn = QPushButton("Secondary")
    parent.secondary_color_btn.setStyleSheet("background: #8b5cf6; color: white;")
    parent.secondary_color_btn.clicked.connect(lambda: _pick_color(parent, "secondary"))
    parent.secondary_color_btn.setEnabled(avatar_module_enabled)
    color_btn_row.addWidget(parent.secondary_color_btn)
    
    parent.accent_color_btn = QPushButton("Accent")
    parent.accent_color_btn.setStyleSheet("background: #10b981; color: white;")
    parent.accent_color_btn.clicked.connect(lambda: _pick_color(parent, "accent"))
    parent.accent_color_btn.setEnabled(avatar_module_enabled)
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
    parent.auto_design_btn.setEnabled(avatar_module_enabled)
    actions_layout.addWidget(parent.auto_design_btn)
    
    # Export sprite button
    parent.export_btn = QPushButton("Export Current Sprite")
    parent.export_btn.clicked.connect(lambda: _export_sprite(parent))
    parent.export_btn.setEnabled(avatar_module_enabled)
    actions_layout.addWidget(parent.export_btn)
    
    actions_group.setLayout(actions_layout)
    right_panel.addWidget(actions_group)
    
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
    parent._current_path = None
    parent._is_3d_model = False
    parent._using_3d_render = False
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
    
    # Load list
    _refresh_list(parent)
    
    # Start with no avatar loaded - user can choose
    parent._using_builtin_sprite = False
    _show_default_preview(parent)
    
    return widget


def _is_avatar_module_enabled() -> bool:
    """Check if avatar module is enabled in ModuleManager."""
    # Avatar is always available - it's a built-in feature
    # No external dependencies required
    return True


def _show_default_preview(parent):
    """Show placeholder text in the preview area (no avatar loaded)."""
    # Clear the preview - don't show built-in sprite by default
    parent.avatar_preview_2d.pixmap = None
    parent.avatar_preview_2d.original_pixmap = None
    parent.avatar_preview_2d._is_3d = False
    parent.avatar_preview_2d.update()
    parent._using_builtin_sprite = False


def _use_builtin_avatar(parent):
    """Use the simple built-in 2D avatar."""
    svg_data = generate_sprite(
        "neutral",
        parent._current_colors["primary"],
        parent._current_colors["secondary"],
        parent._current_colors["accent"]
    )
    parent.avatar_preview_2d.set_svg_sprite(svg_data)
    parent._using_builtin_sprite = True
    parent._current_path = None
    parent._is_3d_model = False
    parent.avatar_status.setText("Using built-in 2D avatar")
    parent.avatar_status.setStyleSheet("color: #a6e3a1;")


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
    """Toggle desktop overlay."""
    # Check if module is enabled
    if not getattr(parent, '_avatar_module_enabled', True):
        parent.show_overlay_btn.setChecked(False)
        parent.avatar_status.setText("Enable avatar module in Modules tab first")
        parent.avatar_status.setStyleSheet("color: #fab387;")
        return
    
    if parent._overlay is None:
        parent._overlay = AvatarOverlayWindow(parent_tab=parent)
        parent._overlay.closed.connect(lambda: _on_overlay_closed(parent))
        parent._overlay.avatar_changed.connect(lambda path: _on_overlay_avatar_changed(parent, path))
        
        # Connect preview rotation changes to sync with overlay
        parent.avatar_preview_2d.rotation_changed.connect(
            lambda rx, ry: _sync_rotation_to_overlay(parent, rx, ry)
        )
    
    if parent.show_overlay_btn.isChecked():
        # Build avatar list for quick switching
        avatar_list = _get_avatar_list_for_menu(parent)
        parent._overlay.set_available_avatars(avatar_list)
        
        # Get current pixmap from preview (syncs the view)
        pixmap = parent.avatar_preview_2d.original_pixmap
        if not pixmap:
            # Generate default sprite
            _show_default_preview(parent)
            pixmap = parent.avatar_preview_2d.original_pixmap
        
        if pixmap:
            scaled = pixmap.scaled(280, 280, Qt_KeepAspectRatio, Qt_SmoothTransformation)
            parent._overlay.set_avatar(scaled)
            parent._overlay.show()
            parent._overlay.raise_()  # Bring to front
            parent.show_overlay_btn.setText("Hide from Desktop")
            parent.avatar_status.setText("Avatar on desktop! Right-click for options.")
            parent.avatar_status.setStyleSheet("color: #a6e3a1;")
        else:
            parent.show_overlay_btn.setChecked(False)
            parent.avatar_status.setText("Could not create avatar sprite")
            parent.avatar_status.setStyleSheet("color: #f38ba8;")
    else:
        parent._overlay.hide()
        parent.show_overlay_btn.setText("Show on Desktop")
        parent.avatar_status.setText("Avatar hidden from desktop")
        parent.avatar_status.setStyleSheet("color: #6c7086;")


def _sync_rotation_to_overlay(parent, rx: float, ry: float):
    """Sync rotation from preview to overlay when visible."""
    if parent._overlay and parent._overlay.isVisible():
        # Get the updated pixmap from preview
        pixmap = parent.avatar_preview_2d.original_pixmap
        if pixmap:
            scaled = pixmap.scaled(280, 280, Qt_KeepAspectRatio, Qt_SmoothTransformation)
            parent._overlay.set_avatar(scaled)


def _get_avatar_list_for_menu(parent) -> list:
    """Get list of avatars for quick-switch menu."""
    avatars = []
    
    # Direct images
    if AVATAR_IMAGES_DIR.exists():
        for f in sorted(AVATAR_IMAGES_DIR.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                avatars.append((f"🖼️ {f.stem}", ("image", str(f))))
    
    # 3D models (direct files)
    if AVATAR_MODELS_DIR.exists():
        for f in sorted(AVATAR_MODELS_DIR.iterdir()):
            if f.suffix.lower() in MODEL_3D_EXTENSIONS:
                avatars.append((f"🎮 {f.stem}", ("model", str(f))))
        
        # Subdirectories with scene files
        for folder in sorted(AVATAR_MODELS_DIR.iterdir()):
            if folder.is_dir():
                for scene_name in ["scene_converted.glb", "scene.gltf", "scene.glb"]:
                    scene_file = folder / scene_name
                    if scene_file.exists():
                        display_name = folder.name.replace("_", " ").replace("-", " ").title()
                        if len(display_name) > 25:
                            display_name = display_name[:22] + "..."
                        avatars.append((f"🎮 {display_name}", ("model", str(scene_file))))
                        break
    
    return avatars


def _on_overlay_avatar_changed(parent, path_str: str):
    """Handle avatar change from overlay."""
    path = Path(path_str)
    if not path.exists():
        return
    
    # Update parent state
    parent._current_path = path
    parent._is_3d_model = path.suffix.lower() in MODEL_3D_EXTENSIONS
    
    # Find and select in combo box
    for i in range(parent.avatar_combo.count()):
        data = parent.avatar_combo.itemData(i)
        if data and data[1] == path_str:
            parent.avatar_combo.setCurrentIndex(i)
            break
    
    parent.avatar_status.setText(f"Switched to: {path.stem}")
    parent.avatar_status.setStyleSheet("color: #a6e3a1;")


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


def _refresh_list(parent):
    """Refresh avatar list."""
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
    
    # 3D models (direct files)
    if AVATAR_MODELS_DIR.exists():
        for f in sorted(AVATAR_MODELS_DIR.iterdir()):
            if f.suffix.lower() in MODEL_3D_EXTENSIONS:
                parent.avatar_combo.addItem(f"🎮 {f.name}", ("model", str(f)))
        
        # Also scan subdirectories for GLTF folders (scene.gltf inside)
        for folder in sorted(AVATAR_MODELS_DIR.iterdir()):
            if folder.is_dir():
                # Check for common model file names in order of preference
                scene_converted = folder / "scene_converted.glb"  # Our USD conversions
                scene_gltf = folder / "scene.gltf"
                scene_glb = folder / "scene.glb"
                
                model_file = None
                if scene_converted.exists():
                    model_file = scene_converted
                elif scene_gltf.exists():
                    model_file = scene_gltf
                elif scene_glb.exists():
                    model_file = scene_glb
                else:
                    # Look for any supported model file in the folder
                    for ext in ['.glb', '.gltf', '.obj', '.fbx', '.dae']:  # Prefer these over USD
                        matches = list(folder.glob(f"*{ext}"))
                        if matches:
                            model_file = matches[0]
                            break
                    
                    # If still nothing, check for USD files and try to convert
                    if not model_file:
                        for ext in ['.usdc', '.usd', '.usda', '.usdz']:
                            matches = list(folder.glob(f"*{ext}"))
                            if matches:
                                model_file = matches[0]
                                break
                
                if model_file:
                    # Use folder name as display name, clean it up
                    display_name = folder.name.replace("_", " ").replace("-", " ").title()
                    parent.avatar_combo.addItem(f"🎮 {display_name}", ("model", str(model_file)))


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


def _show_loading(parent, text: str = "Loading..."):
    """Show loading overlay on the preview widget."""
    if hasattr(parent, '_loading_overlay') and parent._loading_overlay:
        parent._loading_overlay.show_loading(text)
        # Process events to show immediately
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()


def _hide_loading(parent):
    """Hide loading overlay."""
    if hasattr(parent, '_loading_overlay') and parent._loading_overlay:
        parent._loading_overlay.hide_loading()


def _preview_image(parent, path: Path):
    """Preview a 2D image."""
    # Show loading
    _show_loading(parent, "Loading image...")
    
    pixmap = QPixmap(str(path))
    if not pixmap.isNull():
        parent.avatar_preview_2d.set_avatar(pixmap)
        
        # Enable 3D checkbox option only for 3D models
        if parent.use_3d_render_checkbox:
            parent.use_3d_render_checkbox.setEnabled(False)
            parent.use_3d_render_checkbox.setChecked(False)
    
    _hide_loading(parent)


def _preview_3d_model(parent, path: Path):
    """Preview a 3D model - render a thumbnail using pyrender or trimesh."""
    # Show loading indicator
    _show_loading(parent, "Loading 3D model...")
    
    # Enable 3D rendering option
    if parent.use_3d_render_checkbox:
        parent.use_3d_render_checkbox.setEnabled(True)
    
    # Try to render using pyrender (preferred - works with pyglet 2.x)
    if HAS_PYRENDER and HAS_TRIMESH and np is not None:
        try:
            _show_loading(parent, "Parsing model geometry...")
            scene = trimesh.load(str(path))
            
            # Get mesh(es) from scene
            if hasattr(scene, 'geometry') and scene.geometry:
                meshes = list(scene.geometry.values())
                if meshes:
                    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
                else:
                    _hide_loading(parent)
                    _create_model_info_card(parent, path)
                    return
            elif hasattr(scene, 'vertices'):
                mesh = scene
            else:
                _hide_loading(parent)
                _create_model_info_card(parent, path)
                return
            
            _show_loading(parent, "Preparing model for preview...")
            
            # Center and scale the mesh
            mesh.vertices -= mesh.centroid
            if max(mesh.extents) > 0:
                scale = 1.0 / max(mesh.extents)
                mesh.vertices *= scale
            
            # Enable rotation mode on preview widget - pass the mesh for interactive rotation
            parent.avatar_preview_2d.set_3d_model(str(path), mesh)
            
            _show_loading(parent, "Rendering preview...")
            
            # Create pyrender scene with dark background
            pr_scene = pyrender.Scene(bg_color=[30/255, 30/255, 46/255, 1.0])
            
            # Add mesh - try to preserve colors/materials
            pr_mesh = pyrender.Mesh.from_trimesh(mesh)
            pr_scene.add(pr_mesh)
            
            # Add camera
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            # Position camera to view the model from front-ish angle
            camera_pose = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0.2],  # Slightly above
                [0, 0, 1, 2.0],  # Distance back
                [0, 0, 0, 1]
            ], dtype=float)
            pr_scene.add(camera, pose=camera_pose)
            
            # Add lights for better visibility
            # Main directional light
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            pr_scene.add(light, pose=camera_pose)
            
            # Fill light from side
            fill_pose = np.array([
                [0.7, 0, 0.7, 1.5],
                [0, 1, 0, 0.5],
                [-0.7, 0, 0.7, 1.5],
                [0, 0, 0, 1]
            ], dtype=float)
            fill_light = pyrender.DirectionalLight(color=[0.8, 0.85, 1.0], intensity=1.0)
            pr_scene.add(fill_light, pose=fill_pose)
            
            # Render offscreen
            try:
                r = pyrender.OffscreenRenderer(256, 256)
                color, depth = r.render(pr_scene)
                r.delete()
                
                # Convert to QPixmap using PIL
                import io
                img = PILImage.fromarray(color)
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                
                qimg = QImage()
                qimg.loadFromData(buffer.read())
                pixmap = QPixmap.fromImage(qimg)
                parent.avatar_preview_2d.set_avatar(pixmap)
                _hide_loading(parent)
                return
            except Exception as e:
                print(f"Pyrender offscreen render error: {e}")
        except Exception as e:
            print(f"Error loading 3D model with pyrender: {e}")
    
    _hide_loading(parent)
    
    # Fallback: try trimesh's save_image (requires pyglet<2)
    if HAS_TRIMESH:
        _show_loading(parent, "Trying alternative renderer...")
        try:
            scene = trimesh.load(str(path))
            
            # Render to image
            if hasattr(scene, 'geometry') and scene.geometry:
                png_data = scene.save_image(resolution=[256, 256])
                if png_data:
                    img = QImage()
                    img.loadFromData(png_data)
                    pixmap = QPixmap.fromImage(img)
                    parent.avatar_preview_2d.set_avatar(pixmap)
                    _hide_loading(parent)
                    return
            elif hasattr(scene, 'vertices'):
                render_scene = trimesh.Scene(scene)
                png_data = render_scene.save_image(resolution=[256, 256])
                if png_data:
                    img = QImage()
                    img.loadFromData(png_data)
                    pixmap = QPixmap.fromImage(img)
                    parent.avatar_preview_2d.set_avatar(pixmap)
                    _hide_loading(parent)
                    return
        except ImportError:
            # pyglet<2 not available
            pass
        except Exception as e:
            print(f"Error rendering 3D preview with trimesh: {e}")
    
    _hide_loading(parent)
    
    # Final fallback - create info card
    _create_model_info_card(parent, path)


def _create_model_info_card(parent, path: Path):
    """Create an info card pixmap for 3D model - show friendly message."""
    from PyQt5.QtGui import QPolygon
    
    size = 256
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor("#1e1e2e"))
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    
    # Draw a 3D cube icon in the center
    painter.setPen(QColor("#89b4fa"))
    painter.setBrush(QColor("#313244"))
    
    # Simple 3D box shape
    cx, cy = size // 2, size // 2 - 20
    box_size = 60
    offset = 20
    
    # Front face
    painter.drawRect(cx - box_size//2, cy - box_size//2, box_size, box_size)
    
    # Top face (parallelogram)
    top_poly = QPolygon([
        QPoint(cx - box_size//2, cy - box_size//2),
        QPoint(cx - box_size//2 + offset, cy - box_size//2 - offset),
        QPoint(cx + box_size//2 + offset, cy - box_size//2 - offset),
        QPoint(cx + box_size//2, cy - box_size//2),
    ])
    painter.drawPolygon(top_poly)
    
    # Right face
    right_poly = QPolygon([
        QPoint(cx + box_size//2, cy - box_size//2),
        QPoint(cx + box_size//2 + offset, cy - box_size//2 - offset),
        QPoint(cx + box_size//2 + offset, cy + box_size//2 - offset),
        QPoint(cx + box_size//2, cy + box_size//2),
    ])
    painter.drawPolygon(right_poly)
    
    # File name at bottom
    font = painter.font()
    font.setPointSize(10)
    painter.setFont(font)
    painter.setPen(QColor("#cdd6f4"))
    name = path.stem  # Just the name without extension
    if len(name) > 20:
        name = name[:17] + "..."
    painter.drawText(0, size - 50, size, 20, Qt_AlignCenter, name)
    
    # "3D Model" label
    painter.setPen(QColor("#6c7086"))
    font.setPointSize(9)
    painter.setFont(font)
    painter.drawText(0, size - 30, size, 20, Qt_AlignCenter, "(3D Model - Enable 3D Rendering to preview)")
    
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


def sync_preview_to_overlay(parent):
    """Sync the current preview to the desktop overlay."""
    if parent._overlay and parent._overlay.isVisible():
        pixmap = parent.avatar_preview_2d.original_pixmap
        if pixmap:
            scaled = pixmap.scaled(280, 280, Qt_KeepAspectRatio, Qt_SmoothTransformation)
            parent._overlay.set_avatar(scaled)


# === AI Control Functions ===
# These can be called by the AI to control the avatar

def ai_set_avatar_background(parent, color: str = None, shape: str = "none"):
    """
    AI can set avatar background.
    
    Args:
        parent: The avatar tab parent widget
        color: Hex color like "#FF0000" or None for transparent
        shape: 'none' (transparent), 'circle', 'square', 'glow'
    
    Example:
        ai_set_avatar_background(avatar_tab, "#6366f1", "glow")  # Blue glow
        ai_set_avatar_background(avatar_tab, None, "none")  # Clear/transparent
    """
    if not parent._overlay:
        return
    
    if color:
        qcolor = QColor(color)
        qcolor.setAlpha(100)  # Semi-transparent
    else:
        qcolor = None
    
    parent._overlay.set_background(qcolor, shape)


def ai_switch_avatar(parent, avatar_name: str):
    """
    AI can switch to a different avatar by name.
    
    Args:
        parent: The avatar tab parent widget
        avatar_name: Partial name match (case insensitive)
    
    Example:
        ai_switch_avatar(avatar_tab, "freddy")  # Switches to Glamrock Freddy
        ai_switch_avatar(avatar_tab, "robot")  # Switches to Robot.glb
    """
    name_lower = avatar_name.lower()
    
    # Search in combo box
    for i in range(parent.avatar_combo.count()):
        item_text = parent.avatar_combo.itemText(i).lower()
        if name_lower in item_text:
            parent.avatar_combo.setCurrentIndex(i)
            _apply_avatar(parent)
            return True
    
    return False


def ai_set_expression(parent, expression: str):
    """
    AI can change avatar expression.
    
    Args:
        parent: The avatar tab parent widget
        expression: Expression name (happy, sad, thinking, surprised, etc.)
    """
    _set_expression(parent, expression)
    sync_preview_to_overlay(parent)


def ai_show_avatar(parent, show: bool = True):
    """
    AI can show/hide the desktop avatar.
    
    Args:
        parent: The avatar tab parent widget
        show: True to show, False to hide
    """
    if show and not parent.show_overlay_btn.isChecked():
        parent.show_overlay_btn.setChecked(True)
        _toggle_overlay(parent)
    elif not show and parent.show_overlay_btn.isChecked():
        parent.show_overlay_btn.setChecked(False)
        _toggle_overlay(parent)


def get_ai_avatar_controls():
    """
    Get dict of AI-callable avatar control functions.
    
    Returns:
        Dict of function name -> function
    """
    return {
        "set_background": ai_set_avatar_background,
        "switch_avatar": ai_switch_avatar,
        "set_expression": ai_set_expression,
        "show_avatar": ai_show_avatar,
    }
