"""
Avatar Management Panel

Enhanced avatar management with:
- Avatar gallery browsing
- Import wizard integration  
- Bundle format support
- Sample avatar generation
"""

from pathlib import Path
from typing import Any

try:
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import (
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QVBoxLayout,
        QWidget,
    )
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from ....config import CONFIG


class AvatarGalleryItem(QFrame):
    """A clickable avatar item for the gallery."""
    
    clicked = pyqtSignal(dict)
    
    def __init__(self, avatar_info: dict[str, Any], parent=None):
        super().__init__(parent)
        self.avatar_info = avatar_info
        self._setup_ui()
        self.setCursor(Qt.PointingHandCursor)
    
    def _setup_ui(self):
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setFixedSize(120, 150)
        self.setStyleSheet("""
            AvatarGalleryItem {
                background-color: white;
                border-radius: 6px;
                border: 1px solid #ddd;
            }
            AvatarGalleryItem:hover {
                border: 2px solid #4a9eff;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Thumbnail
        self.thumb = QLabel()
        self.thumb.setFixedSize(80, 80)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet("border: 1px solid #eee; border-radius: 4px;")
        layout.addWidget(self.thumb, alignment=Qt.AlignCenter)
        
        self._load_thumbnail()
        
        # Name
        name = QLabel(self.avatar_info.get("name", "Unknown")[:15])
        name.setAlignment(Qt.AlignCenter)
        name.setStyleSheet("font-size: 11px; font-weight: bold;")
        layout.addWidget(name)
        
        # Type
        avatar_type = self.avatar_info.get("type", "").lower()
        type_label = QLabel(avatar_type[:10])
        type_label.setAlignment(Qt.AlignCenter)
        type_label.setStyleSheet("font-size: 10px; color: #bac2de;")
        layout.addWidget(type_label)
    
    def _load_thumbnail(self):
        path = Path(self.avatar_info.get("path", ""))
        for img_name in ["neutral.png", "base.png", "default.png"]:
            img_path = path / img_name
            if img_path.exists():
                pixmap = QPixmap(str(img_path))
                if not pixmap.isNull():
                    scaled = pixmap.scaled(70, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.thumb.setPixmap(scaled)
                    return
        
        self.thumb.setText("?")
        self.thumb.setStyleSheet("border: 1px solid #eee; color: #ccc; font-size: 24px;")
    
    def mousePressEvent(self, event):
        self.clicked.emit(self.avatar_info)


class AvatarManagementPanel(QWidget):
    """
    Panel for avatar management within the existing avatar display tab.
    
    Provides:
    - Gallery view of installed avatars
    - Sample generation
    - Import wizard access
    """
    
    avatar_selected = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._load_avatars()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Actions row
        actions = QHBoxLayout()
        
        import_btn = QPushButton("Import Avatar...")
        import_btn.clicked.connect(self._show_import_wizard)
        actions.addWidget(import_btn)
        
        generate_btn = QPushButton("Generate Samples")
        generate_btn.clicked.connect(self._generate_samples)
        actions.addWidget(generate_btn)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_avatars)
        actions.addWidget(refresh_btn)
        
        actions.addStretch()
        layout.addLayout(actions)
        
        # Gallery scroll area
        gallery_group = QGroupBox("Available Avatars")
        gallery_layout = QVBoxLayout(gallery_group)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(200)
        
        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        self.gallery_layout.setSpacing(10)
        scroll.setWidget(self.gallery_widget)
        
        gallery_layout.addWidget(scroll)
        layout.addWidget(gallery_group)
    
    def _load_avatars(self):
        """Load avatars into the gallery."""
        # Clear existing
        while self.gallery_layout.count():
            item = self.gallery_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        avatars = []
        
        # Check installed avatars
        try:
            from ....avatar.avatar_bundle import list_installed_avatars
            avatars.extend(list_installed_avatars())
        except Exception:
            pass
        
        # Check sample avatars
        samples_dir = Path(CONFIG["data_dir"]) / "avatar" / "samples"
        if samples_dir.exists():
            for sample_path in samples_dir.iterdir():
                if sample_path.is_dir():
                    manifest_path = sample_path / "manifest.json"
                    if manifest_path.exists():
                        try:
                            import json
                            manifest = json.loads(manifest_path.read_text())
                            avatars.append({
                                "name": manifest.get("name", sample_path.name),
                                "path": str(sample_path),
                                "type": manifest.get("avatar_type", "ABSTRACT"),
                                "author": manifest.get("author", "Enigma AI Engine"),
                            })
                        except Exception:
                            pass
        
        # Populate gallery
        cols = 4
        for i, avatar in enumerate(avatars):
            item = AvatarGalleryItem(avatar)
            item.clicked.connect(self._on_avatar_clicked)
            self.gallery_layout.addWidget(item, i // cols, i % cols)
        
        # Stretch at end
        self.gallery_layout.setRowStretch(len(avatars) // cols + 1, 1)
        
        if not avatars:
            no_avatars = QLabel("No avatars found.\nClick 'Generate Samples' to create some!")
            no_avatars.setAlignment(Qt.AlignCenter)
            no_avatars.setStyleSheet("color: #bac2de; padding: 20px;")
            self.gallery_layout.addWidget(no_avatars, 0, 0, 1, cols)
    
    def _on_avatar_clicked(self, avatar_info: dict):
        """Handle avatar selection."""
        self.avatar_selected.emit(avatar_info)
    
    def _show_import_wizard(self):
        """Show the import wizard."""
        try:
            from ....avatar.avatar_dialogs import AvatarImportWizard
            
            wizard = AvatarImportWizard(self)
            if wizard.exec_():
                self._load_avatars()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open wizard: {e}")
    
    def _generate_samples(self):
        """Generate sample avatars."""
        try:
            from ....avatar.sample_avatars import generate_sample_avatars
            
            avatars = generate_sample_avatars()
            self._load_avatars()
            
            QMessageBox.information(
                self, "Success",
                f"Generated {len(avatars)} sample avatars!\n"
                "Click one to use it."
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to generate: {e}")


def create_management_panel(parent=None) -> AvatarManagementPanel:
    """Create an avatar management panel."""
    return AvatarManagementPanel(parent)
