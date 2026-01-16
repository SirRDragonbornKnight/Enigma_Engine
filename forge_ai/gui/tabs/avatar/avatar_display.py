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
    """
    
    closed = pyqtSignal()
    
    def __init__(self):
        super().__init__(None)
        
        # Transparent, always-on-top, no taskbar
        self.setWindowFlags(
            Qt_FramelessWindowHint |
            Qt_WindowStaysOnTopHint |
            Qt_Tool
        )
        self.setAttribute(Qt_WA_TranslucentBackground, True)
        
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
        
    def mouseDoubleClickEvent(self, a0):  # type: ignore
        """Double-click to reset size."""
        self._size = 300
        self.setFixedSize(self._size, self._size)
        self._update_scaled_pixmap()


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
    parent.load_btn.setEnabled(avatar_module_enabled)
    btn_row2.addWidget(parent.load_btn)
    
    parent.apply_btn = QPushButton("Apply Avatar")
    parent.apply_btn.clicked.connect(lambda: _apply_avatar(parent))
    parent.apply_btn.setStyleSheet("background: #45475a;")
    parent.apply_btn.setEnabled(avatar_module_enabled)
    btn_row2.addWidget(parent.apply_btn)
    left_panel.addLayout(btn_row2)
    
    # Status
    parent.avatar_status = QLabel("No avatar loaded")
    parent.avatar_status.setStyleSheet("color: #6c7086; font-style: italic;")
    left_panel.addWidget(parent.avatar_status)
    
    main_layout.addLayout(left_panel, stretch=2)
    
    # Right side - Customization Controls
    right_panel = QVBoxLayout()
    
    # === Expression Preview (read-only) ===
    expression_group = QGroupBox("Expression Preview")
    expression_layout = QVBoxLayout()
    
    parent.expression_label = QLabel("Current: neutral")
    parent.expression_label.setStyleSheet("color: #cdd6f4; font-size: 11px;")
    expression_layout.addWidget(parent.expression_label)
    
    expression_info = QLabel("Expressions change automatically based on AI mood and conversation.")
    expression_info.setStyleSheet("color: #6c7086; font-size: 10px;")
    expression_info.setWordWrap(True)
    expression_layout.addWidget(expression_info)
    
    parent.test_expression_btn = QPushButton("Test Random Expression")
    parent.test_expression_btn.clicked.connect(lambda: _test_random_expression(parent))
    parent.test_expression_btn.setEnabled(avatar_module_enabled)
    expression_layout.addWidget(parent.test_expression_btn)
    
    expression_group.setLayout(expression_layout)
    right_panel.addWidget(expression_group)
    
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
    
    # Show default sprite on initialization
    parent._using_builtin_sprite = True
    _show_default_preview(parent)
    
    return widget


def _is_avatar_module_enabled() -> bool:
    """Check if avatar module is enabled in ModuleManager."""
    try:
        from ....modules import get_manager
        manager = get_manager()
        if manager:
            return manager.is_loaded('avatar')
    except Exception:
        pass
    return True  # Default to enabled if can't check


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
    """Toggle desktop overlay."""
    # Check if module is enabled
    if not getattr(parent, '_avatar_module_enabled', True):
        parent.show_overlay_btn.setChecked(False)
        parent.avatar_status.setText("Enable avatar module in Modules tab first")
        parent.avatar_status.setStyleSheet("color: #fab387;")
        return
    
    if parent._overlay is None:
        parent._overlay = AvatarOverlayWindow()
        parent._overlay.closed.connect(lambda: _on_overlay_closed(parent))
    
    if parent.show_overlay_btn.isChecked():
        # Get current pixmap, or generate default if none
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
            parent.avatar_status.setText("Avatar shown on desktop! Drag to move, right-click to hide.")
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
    
    # 3D models
    if AVATAR_MODELS_DIR.exists():
        for f in sorted(AVATAR_MODELS_DIR.iterdir()):
            if f.suffix.lower() in MODEL_3D_EXTENSIONS:
                parent.avatar_combo.addItem(f"🎮 {f.name}", ("model", str(f)))


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
    """Preview a 3D model - render a thumbnail."""
    # Enable 3D rendering option
    if parent.use_3d_render_checkbox:
        parent.use_3d_render_checkbox.setEnabled(True)
    
    # Create a preview thumbnail using trimesh
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
