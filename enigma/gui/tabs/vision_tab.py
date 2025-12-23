"""Vision tab for Enigma Engine GUI."""

from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QPlainTextEdit, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer


def create_vision_tab(parent):
    """Create the screen vision tab."""
    w = QWidget()
    layout = QVBoxLayout()
    
    # Header
    header = QLabel("Screen Vision")
    header.setObjectName("header")
    layout.addWidget(header)
    
    # Screen preview
    parent.vision_preview = QLabel("Vision not started")
    parent.vision_preview.setMinimumHeight(250)
    parent.vision_preview.setAlignment(Qt.AlignCenter)
    parent.vision_preview.setStyleSheet("border: 1px solid #45475a; border-radius: 4px; background: #313244;")
    layout.addWidget(parent.vision_preview)
    
    # Control row
    ctrl_layout = QHBoxLayout()
    
    parent.btn_start_watching = QPushButton("Start Watching")
    parent.btn_start_watching.setCheckable(True)
    parent.btn_start_watching.clicked.connect(parent._toggle_screen_watching)
    ctrl_layout.addWidget(parent.btn_start_watching)
    
    parent.btn_single_capture = QPushButton("Capture Once")
    parent.btn_single_capture.clicked.connect(parent._do_single_capture)
    ctrl_layout.addWidget(parent.btn_single_capture)
    
    ctrl_layout.addStretch()
    layout.addLayout(ctrl_layout)
    
    # Interval setting
    interval_layout = QHBoxLayout()
    interval_layout.addWidget(QLabel("Update every:"))
    parent.vision_interval_spin = QSpinBox()
    parent.vision_interval_spin.setRange(1, 60)
    parent.vision_interval_spin.setValue(5)
    parent.vision_interval_spin.setSuffix(" sec")
    parent.vision_interval_spin.setMinimumWidth(80)
    interval_layout.addWidget(parent.vision_interval_spin)
    interval_layout.addStretch()
    layout.addLayout(interval_layout)
    
    # Analysis output
    analysis_label = QLabel("AI Analysis:")
    layout.addWidget(analysis_label)
    
    parent.vision_text = QPlainTextEdit()
    parent.vision_text.setReadOnly(True)
    parent.vision_text.setPlaceholderText("What the AI sees...")
    parent.vision_text.setMaximumHeight(120)
    layout.addWidget(parent.vision_text)
    
    # Timer for continuous watching
    parent.vision_timer = QTimer()
    parent.vision_timer.timeout.connect(parent._do_continuous_capture)
    
    layout.addStretch()
    w.setLayout(layout)
    return w
