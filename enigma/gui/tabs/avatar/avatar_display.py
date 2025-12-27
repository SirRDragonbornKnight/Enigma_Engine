"""
Avatar Display Module

Displays the AI's visual avatar representation.
The same AI that learns and responds controls this avatar.
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import json

from ....config import CONFIG


# Avatar config directory
AVATAR_CONFIG_DIR = Path(CONFIG["data_dir"]) / "avatar"


def create_avatar_subtab(parent):
    """
    Create the avatar display sub-tab.
    
    The AI being trained is the one controlling this avatar.
    When the AI responds, it can choose expressions/states.
    """
    widget = QWidget()
    layout = QVBoxLayout()
    
    # Avatar display - large centered image
    parent.avatar_image_label = QLabel()
    parent.avatar_image_label.setMinimumSize(350, 350)
    parent.avatar_image_label.setAlignment(Qt.AlignCenter)
    parent.avatar_image_label.setStyleSheet("""
        border: 2px solid #45475a; 
        border-radius: 12px; 
        background: #1e1e2e;
    """)
    parent.avatar_image_label.setScaledContents(False)
    parent.avatar_image_label.setText("No avatar loaded\n\nSelect a config file")
    layout.addWidget(parent.avatar_image_label, stretch=1, alignment=Qt.AlignCenter)
    
    # Config file selector
    config_layout = QHBoxLayout()
    config_layout.addWidget(QLabel("Avatar Config:"))
    parent.avatar_config_combo = QComboBox()
    parent.avatar_config_combo.currentIndexChanged.connect(
        lambda idx: _on_avatar_config_changed(parent, idx)
    )
    config_layout.addWidget(parent.avatar_config_combo, stretch=1)
    
    btn_refresh = QPushButton("[R]")
    btn_refresh.setFixedWidth(40)
    btn_refresh.setToolTip("Refresh config list")
    btn_refresh.clicked.connect(lambda: _refresh_avatar_configs(parent))
    config_layout.addWidget(btn_refresh)
    
    btn_browse = QPushButton("Browse...")
    btn_browse.clicked.connect(lambda: _browse_avatar_config(parent))
    config_layout.addWidget(btn_browse)
    layout.addLayout(config_layout)
    
    # Quick image load
    btn_load_image = QPushButton("Load Avatar Image Directly")
    btn_load_image.clicked.connect(parent._load_avatar_image)
    layout.addWidget(btn_load_image)
    
    # Status
    parent.avatar_status_label = QLabel("No avatar loaded")
    parent.avatar_status_label.setStyleSheet("color: #6c7086; font-style: italic;")
    layout.addWidget(parent.avatar_status_label)
    
    # Info label
    info = QLabel("[i] The AI you train controls this avatar. When the AI responds,\n"
                  "    it can change expressions based on its learned behavior.")
    info.setStyleSheet("color: #6c7086; font-size: 10px;")
    layout.addWidget(info)
    
    widget.setLayout(layout)
    
    # Initialize
    parent.avatar_expressions = {}
    parent.current_expression = "neutral"
    AVATAR_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load configs
    _refresh_avatar_configs(parent)
    
    return widget


def load_avatar_config(config_path: Path) -> dict:
    """Load an avatar configuration file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        pass  # Silently fail, return empty dict
        return {}


def _refresh_avatar_configs(parent):
    """Refresh the list of available avatar configs."""
    parent.avatar_config_combo.clear()
    parent.avatar_config_combo.addItem("-- Select Config File --", None)
    
    # Scan for .json config files in avatar directory
    if AVATAR_CONFIG_DIR.exists():
        for config_file in sorted(AVATAR_CONFIG_DIR.glob("*.json")):
            parent.avatar_config_combo.addItem(config_file.stem, str(config_file))
    
    # Also check model-specific avatar configs
    model_name = getattr(parent, 'current_model_name', CONFIG.get("default_model", "default"))
    model_avatar_dir = Path(CONFIG["models_dir"]) / model_name / "avatar"
    if model_avatar_dir.exists():
        for config_file in sorted(model_avatar_dir.glob("*.json")):
            parent.avatar_config_combo.addItem(
                f"{config_file.stem} (model)", str(config_file)
            )


def _on_avatar_config_changed(parent, index):
    """Handle avatar config selection."""
    config_path = parent.avatar_config_combo.currentData()
    if not config_path:
        return
    
    config = load_avatar_config(Path(config_path))
    if not config:
        parent.avatar_status_label.setText("Failed to load config")
        return
    
    # Load avatar image if specified
    if "image" in config:
        img_path = Path(config["image"])
        if not img_path.is_absolute():
            img_path = Path(config_path).parent / img_path
        
        if img_path.exists():
            pixmap = QPixmap(str(img_path))
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                parent.avatar_image_label.setPixmap(scaled)
                parent.avatar_status_label.setText(f"Loaded: {Path(config_path).stem}")
    
    # Load expressions if available
    if "expressions" in config:
        parent.avatar_expressions = config["expressions"]


def _browse_avatar_config(parent):
    """Browse for an avatar config file."""
    path, _ = QFileDialog.getOpenFileName(
        parent, "Select Avatar Config",
        str(AVATAR_CONFIG_DIR),
        "JSON Files (*.json);;All Files (*)"
    )
    if path:
        # Add to combo if not already there
        for i in range(parent.avatar_config_combo.count()):
            if parent.avatar_config_combo.itemData(i) == path:
                parent.avatar_config_combo.setCurrentIndex(i)
                return
        
        parent.avatar_config_combo.addItem(Path(path).stem, path)
        parent.avatar_config_combo.setCurrentIndex(parent.avatar_config_combo.count() - 1)


def create_sample_avatar_config():
    """Create a sample avatar config file."""
    sample = {
        "name": "Default Avatar",
        "image": "avatar.png",
        "expressions": {
            "neutral": "avatar.png",
            "happy": "avatar_happy.png",
            "thinking": "avatar_thinking.png",
            "confused": "avatar_confused.png"
        },
        "description": "Sample avatar configuration"
    }
    
    sample_path = AVATAR_CONFIG_DIR / "sample_avatar.json"
    if not sample_path.exists():
        AVATAR_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(sample_path, 'w') as f:
            json.dump(sample, f, indent=2)
    
    return sample_path
