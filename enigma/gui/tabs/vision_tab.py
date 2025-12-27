"""Vision tab for Enigma Engine GUI - screen capture and camera support."""

from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QPlainTextEdit, QCheckBox, QFileDialog, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage


def create_vision_tab(parent):
    """Create the screen vision tab with camera support."""
    w = QWidget()
    layout = QVBoxLayout()
    
    # Header
    header = QLabel("Screen Vision & Camera")
    header.setObjectName("header")
    layout.addWidget(header)
    
    # Preview area
    parent.vision_preview = QLabel("Vision not started")
    parent.vision_preview.setMinimumHeight(250)
    parent.vision_preview.setAlignment(Qt.AlignCenter)
    parent.vision_preview.setStyleSheet("border: 1px solid #45475a; border-radius: 4px; background: #313244;")
    parent.vision_preview.setTextInteractionFlags(Qt.TextSelectableByMouse)
    layout.addWidget(parent.vision_preview)
    
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
    
    # Timer for continuous watching
    parent.vision_timer = QTimer()
    parent.vision_timer.timeout.connect(parent._do_continuous_capture)
    
    # Store current image path
    parent.current_vision_image = None
    
    layout.addStretch()
    w.setLayout(layout)
    return w
