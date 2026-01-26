"""Vision tab for ForgeAI GUI - screen capture and camera support."""

import os
import subprocess
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QPlainTextEdit, QGroupBox, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

from ...config import CONFIG

# Images directory
IMAGES_DIR = Path(CONFIG["data_dir"]) / "images"


def create_vision_tab(parent):
    """Create the screen vision tab with camera support."""
    w = QWidget()
    layout = QVBoxLayout()
    
    # Header
    header = QLabel("Screen Vision & Camera")
    header.setObjectName("header")
    layout.addWidget(header)
    
    # Preview area - flexible height
    parent.vision_preview = QLabel("Vision not started")
    parent.vision_preview.setMinimumHeight(150)  # Reduced for smaller screens
    parent.vision_preview.setAlignment(Qt.AlignCenter)
    parent.vision_preview.setStyleSheet("border: 1px solid #45475a; border-radius: 4px; background: #313244;")
    parent.vision_preview.setTextInteractionFlags(Qt.TextSelectableByMouse)
    layout.addWidget(parent.vision_preview, stretch=1)  # Allow to grow/shrink
    
    # Source selection group
    source_group = QGroupBox("Capture Source")
    source_layout = QHBoxLayout()
    
    parent.btn_capture_screen = QPushButton("Screen")
    parent.btn_capture_screen.setToolTip("Capture from screen")
    parent.btn_capture_screen.clicked.connect(parent._capture_screen)
    source_layout.addWidget(parent.btn_capture_screen)
    
    parent.btn_capture_camera = QPushButton("Camera")
    parent.btn_capture_camera.setToolTip("Capture from webcam/camera")
    parent.btn_capture_camera.clicked.connect(parent._capture_camera)
    source_layout.addWidget(parent.btn_capture_camera)
    
    parent.btn_load_image = QPushButton("Load Image")
    parent.btn_load_image.setToolTip("Load image from file")
    parent.btn_load_image.clicked.connect(parent._load_vision_image)
    source_layout.addWidget(parent.btn_load_image)
    
    parent.btn_clear_image = QPushButton("Clear")
    parent.btn_clear_image.setToolTip("Clear current image")
    parent.btn_clear_image.clicked.connect(lambda: _clear_vision_image(parent))
    source_layout.addWidget(parent.btn_clear_image)
    
    source_layout.addStretch()
    source_group.setLayout(source_layout)
    layout.addWidget(source_group)
    
    # Control row
    ctrl_layout = QHBoxLayout()
    
    parent.btn_start_watching = QPushButton("Start Auto-Watch")
    parent.btn_start_watching.setCheckable(True)
    parent.btn_start_watching.clicked.connect(parent._toggle_screen_watching)
    ctrl_layout.addWidget(parent.btn_start_watching)
    
    parent.btn_analyze = QPushButton("Analyze Image")
    parent.btn_analyze.setToolTip("Have AI analyze current image")
    parent.btn_analyze.clicked.connect(parent._analyze_vision_image)
    ctrl_layout.addWidget(parent.btn_analyze)
    
    ctrl_layout.addStretch()
    layout.addLayout(ctrl_layout)
    
    # Interval setting
    interval_layout = QHBoxLayout()
    interval_layout.addWidget(QLabel("Auto-watch interval:"))
    parent.vision_interval_spin = QSpinBox()
    parent.vision_interval_spin.setRange(1, 60)
    parent.vision_interval_spin.setValue(5)
    parent.vision_interval_spin.setSuffix(" sec")
    parent.vision_interval_spin.setMinimumWidth(80)
    interval_layout.addWidget(parent.vision_interval_spin)
    interval_layout.addStretch()
    layout.addLayout(interval_layout)
    
    # OCR/Analysis output
    analysis_label = QLabel("AI Analysis / OCR Text:")
    layout.addWidget(analysis_label)
    
    parent.vision_text = QPlainTextEdit()
    parent.vision_text.setReadOnly(True)
    parent.vision_text.setPlaceholderText("What the AI sees will appear here...")
    parent.vision_text.setMaximumHeight(150)
    parent.vision_text.setTextInteractionFlags(
        Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
    )
    layout.addWidget(parent.vision_text)
    
    # Saved Images group
    images_group = QGroupBox("Saved Images")
    images_layout = QVBoxLayout()
    
    parent.vision_image_list = QListWidget()
    parent.vision_image_list.setMaximumHeight(100)
    parent.vision_image_list.itemDoubleClicked.connect(lambda item: _load_saved_image(parent, item))
    images_layout.addWidget(parent.vision_image_list)
    
    img_btn_layout = QHBoxLayout()
    btn_refresh_images = QPushButton("Refresh")
    btn_refresh_images.clicked.connect(lambda: _refresh_saved_images(parent))
    img_btn_layout.addWidget(btn_refresh_images)
    
    btn_open_folder = QPushButton("Open Folder")
    btn_open_folder.clicked.connect(lambda: _open_images_folder(parent))
    img_btn_layout.addWidget(btn_open_folder)
    
    btn_save_current = QPushButton("Save Current")
    btn_save_current.clicked.connect(lambda: _save_current_image(parent))
    img_btn_layout.addWidget(btn_save_current)
    
    img_btn_layout.addStretch()
    images_layout.addLayout(img_btn_layout)
    images_group.setLayout(images_layout)
    layout.addWidget(images_group)
    
    # Timer for continuous watching
    parent.vision_timer = QTimer()
    parent.vision_timer.timeout.connect(parent._do_continuous_capture)
    
    # Store current image path
    parent.current_vision_image = None
    
    # Initialize images directory and list
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    _refresh_saved_images(parent)
    
    layout.addStretch()
    w.setLayout(layout)
    return w


def _clear_vision_image(parent):
    """Clear the current image from the preview."""
    parent.vision_preview.clear()
    parent.vision_preview.setText("Vision not started")
    parent.current_vision_image = None
    parent.vision_text.clear()


def _refresh_saved_images(parent):
    """Refresh the list of saved images."""
    parent.vision_image_list.clear()
    
    if IMAGES_DIR.exists():
        for img_file in sorted(IMAGES_DIR.glob("*"), reverse=True):
            if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
                item = QListWidgetItem(img_file.name)
                item.setData(Qt.UserRole, str(img_file))
                parent.vision_image_list.addItem(item)


def _load_saved_image(parent, item):
    """Load a saved image into the preview."""
    img_path = item.data(Qt.UserRole)
    if img_path and Path(img_path).exists():
        pixmap = QPixmap(img_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                parent.vision_preview.width() - 10,
                parent.vision_preview.height() - 10,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            parent.vision_preview.setPixmap(scaled)
            parent.current_vision_image = img_path


def _open_images_folder(parent):
    """Open the images folder in the system file explorer."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    from .output_helpers import open_folder
    open_folder(IMAGES_DIR)


def _save_current_image(parent):
    """Save the current vision image to the images folder."""
    if not parent.current_vision_image:
        return
    
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_path = IMAGES_DIR / f"capture_{timestamp}.png"
    
    try:
        # If current image is a path, copy it
        if Path(parent.current_vision_image).exists():
            import shutil
            shutil.copy2(parent.current_vision_image, dest_path)
        else:
            # Save from pixmap
            pixmap = parent.vision_preview.pixmap()
            if pixmap:
                pixmap.save(str(dest_path))
        
        _refresh_saved_images(parent)
    except Exception:
        pass  # Silent fail
