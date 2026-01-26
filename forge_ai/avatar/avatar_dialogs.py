"""
Avatar Import Wizard and Picker Dialog

Provides easy-to-use dialogs for:
- Importing new avatars (drag & drop, file picker)
- Browsing and selecting installed avatars
- Previewing avatars before selection
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any

try:
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QFileDialog, QLineEdit,
        QGroupBox, QScrollArea, QWidget,
        QFrame, QTextEdit, QProgressBar, QMessageBox,
        QTabWidget, QListWidget, QListWidgetItem, QStackedWidget
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QMimeData, QSize, QTimer
    from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QIcon, QPainter, QColor
    from ..gui.tabs.shared_components import NoScrollComboBox
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from .avatar_bundle import (
    AvatarBundle, AvatarManifest, AvatarBundleCreator,
    install_avatar_bundle, list_installed_avatars
)


class DropZone(QLabel):
    """
    A drop zone widget for drag-and-drop file imports.
    """
    
    file_dropped = pyqtSignal(str)  # Emits file path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 150)
        self._set_default_style()
        self.setText("Drop avatar image or .forgeavatar file here\n\n"
                    "Supports: PNG, GIF, JPG, .forgeavatar")
    
    def _set_default_style(self):
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #888;
                border-radius: 10px;
                background-color: #f0f0f0;
                color: #666;
                font-size: 14px;
                padding: 20px;
            }
        """)
    
    def _set_hover_style(self):
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #4a9eff;
                border-radius: 10px;
                background-color: #e8f4ff;
                color: #4a9eff;
                font-size: 14px;
                padding: 20px;
            }
        """)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_hover_style()
    
    def dragLeaveEvent(self, event):
        self._set_default_style()
    
    def dropEvent(self, event: QDropEvent):
        self._set_default_style()
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.file_dropped.emit(file_path)


class AvatarPreview(QWidget):
    """
    Widget that shows avatar preview with emotion buttons.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._current_bundle: Optional[AvatarBundle] = None
        self._current_path: Optional[str] = None
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Preview image
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(200, 200)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 10px;
                background-color: #f8f8f8;
            }
        """)
        layout.addWidget(self.preview_label, alignment=Qt.AlignCenter)
        
        # Emotion buttons
        emotions_group = QGroupBox("Preview Emotions")
        emotions_layout = QGridLayout(emotions_group)
        
        self.emotion_buttons = {}
        emotions = ["neutral", "happy", "sad", "surprised", 
                   "thinking", "confused", "angry", "excited"]
        
        for i, emotion in enumerate(emotions):
            btn = QPushButton(emotion.capitalize())
            btn.setFixedWidth(80)
            btn.clicked.connect(lambda checked, e=emotion: self._show_emotion(e))
            emotions_layout.addWidget(btn, i // 4, i % 4)
            self.emotion_buttons[emotion] = btn
        
        layout.addWidget(emotions_group)
    
    def load_from_path(self, path: str):
        """Load preview from a file or directory path."""
        path = Path(path)
        self._current_path = str(path)
        
        if path.suffix == '.forgeavatar':
            self._current_bundle = AvatarBundle.load(str(path))
            self._show_bundle_preview()
        elif path.is_dir():
            # Check for manifest
            manifest_path = path / "manifest.json"
            if manifest_path.exists():
                import json
                self._current_bundle = AvatarBundle()
                self._current_bundle.manifest = AvatarManifest.from_dict(
                    json.loads(manifest_path.read_text())
                )
                # Load files from directory
                for file in path.iterdir():
                    if file.is_file():
                        self._current_bundle._files[file.name] = file.read_bytes()
                self._show_bundle_preview()
            else:
                # Try to create bundle from folder
                self._current_bundle = AvatarBundleCreator.from_emotion_folder(str(path))
                self._show_bundle_preview()
        elif path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
            # Single image
            self._show_image(str(path))
            self._current_bundle = None
    
    def _show_bundle_preview(self):
        """Show preview from loaded bundle."""
        if not self._current_bundle:
            return
        
        # Show base or neutral image
        base_data = self._current_bundle.get_base_image()
        if base_data:
            pixmap = QPixmap()
            pixmap.loadFromData(base_data)
            self._show_pixmap(pixmap)
        
        # Enable/disable emotion buttons based on availability
        available_emotions = self._current_bundle.list_emotions()
        for emotion, btn in self.emotion_buttons.items():
            btn.setEnabled(emotion in available_emotions or emotion == "neutral")
    
    def _show_emotion(self, emotion: str):
        """Show specific emotion preview."""
        if not self._current_bundle:
            return
        
        data = self._current_bundle.get_emotion_image(emotion)
        if data:
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            self._show_pixmap(pixmap)
        elif emotion == "neutral":
            data = self._current_bundle.get_base_image()
            if data:
                pixmap = QPixmap()
                pixmap.loadFromData(data)
                self._show_pixmap(pixmap)
    
    def _show_image(self, path: str):
        """Show image from file path."""
        pixmap = QPixmap(path)
        self._show_pixmap(pixmap)
    
    def _show_pixmap(self, pixmap: QPixmap):
        """Display a pixmap in the preview."""
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.preview_label.size() - QSize(20, 20),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled)


class AvatarImportWizard(QDialog):
    """
    Wizard dialog for importing new avatars.
    
    Supports:
    - Drag & drop single image
    - Drag & drop folder with emotions
    - Drag & drop .forgeavatar bundle
    - File picker for any of the above
    """
    
    avatar_imported = pyqtSignal(str)  # Emits installed path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Avatar")
        self.setMinimumSize(600, 500)
        self._setup_ui()
        self._imported_path: Optional[str] = None
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Import New Avatar")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Main content in horizontal layout
        content = QHBoxLayout()
        
        # Left side - Import options
        left_side = QVBoxLayout()
        
        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self._on_file_dropped)
        left_side.addWidget(self.drop_zone)
        
        # Or divider
        or_label = QLabel("— OR —")
        or_label.setAlignment(Qt.AlignCenter)
        or_label.setStyleSheet("color: #888; margin: 10px;")
        left_side.addWidget(or_label)
        
        # File picker buttons
        buttons_layout = QHBoxLayout()
        
        btn_file = QPushButton("Select Image")
        btn_file.clicked.connect(self._select_image)
        buttons_layout.addWidget(btn_file)
        
        btn_folder = QPushButton("Select Folder")
        btn_folder.clicked.connect(self._select_folder)
        buttons_layout.addWidget(btn_folder)
        
        btn_bundle = QPushButton("Select .forgeavatar")
        btn_bundle.clicked.connect(self._select_bundle)
        buttons_layout.addWidget(btn_bundle)
        
        left_side.addLayout(buttons_layout)
        
        # Avatar settings
        settings_group = QGroupBox("Avatar Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("My Avatar")
        name_layout.addWidget(self.name_edit)
        settings_layout.addLayout(name_layout)
        
        # Type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = NoScrollComboBox()
        self.type_combo.setToolTip("Select the avatar type category")
        self.type_combo.addItems(["Human", "Animal", "Robot", "Fantasy", "Abstract", "Custom"])
        type_layout.addWidget(self.type_combo)
        settings_layout.addLayout(type_layout)
        
        # Author
        author_layout = QHBoxLayout()
        author_layout.addWidget(QLabel("Author:"))
        self.author_edit = QLineEdit()
        self.author_edit.setPlaceholderText("Your name (optional)")
        author_layout.addWidget(self.author_edit)
        settings_layout.addLayout(author_layout)
        
        left_side.addWidget(settings_group)
        left_side.addStretch()
        
        content.addLayout(left_side)
        
        # Right side - Preview
        right_side = QVBoxLayout()
        
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview = AvatarPreview()
        preview_layout.addWidget(self.preview)
        
        right_side.addWidget(preview_group)
        
        content.addLayout(right_side)
        layout.addLayout(content)
        
        # Bottom buttons
        buttons = QHBoxLayout()
        buttons.addStretch()
        
        self.import_btn = QPushButton("Import Avatar")
        self.import_btn.setEnabled(False)
        self.import_btn.clicked.connect(self._do_import)
        buttons.addWidget(self.import_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        
        layout.addLayout(buttons)
    
    def _on_file_dropped(self, path: str):
        """Handle dropped file."""
        self._load_source(path)
    
    def _select_image(self):
        """Open file picker for single image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Avatar Image",
            "", "Images (*.png *.jpg *.jpeg *.gif)"
        )
        if path:
            self._load_source(path)
    
    def _select_folder(self):
        """Open folder picker for emotion set."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Avatar Folder"
        )
        if path:
            self._load_source(path)
    
    def _select_bundle(self):
        """Open file picker for .forgeavatar bundle."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Avatar Bundle",
            "", "Avatar Bundle (*.forgeavatar)"
        )
        if path:
            self._load_source(path)
    
    def _load_source(self, path: str):
        """Load and preview an avatar source."""
        self._source_path = path
        self.preview.load_from_path(path)
        
        # Auto-fill name from filename
        name = Path(path).stem
        if not self.name_edit.text():
            self.name_edit.setText(name.replace("_", " ").title())
        
        self.import_btn.setEnabled(True)
        self.drop_zone.setText(f"Selected: {Path(path).name}")
    
    def _do_import(self):
        """Import the avatar."""
        if not hasattr(self, '_source_path'):
            return
        
        try:
            source = Path(self._source_path)
            name = self.name_edit.text() or "Custom Avatar"
            avatar_type = self.type_combo.currentText().upper()
            author = self.author_edit.text() or "Unknown"
            
            # Create bundle based on source type
            if source.suffix == '.forgeavatar':
                bundle = AvatarBundle.load(str(source))
                bundle.manifest.name = name
                bundle.manifest.author = author
            elif source.is_dir():
                bundle = AvatarBundleCreator.from_emotion_folder(
                    str(source), name, avatar_type
                )
                bundle.manifest.author = author
            else:
                bundle = AvatarBundleCreator.from_single_image(
                    str(source), name, avatar_type
                )
                bundle.manifest.author = author
            
            # Install bundle
            from ..config import CONFIG
            avatars_dir = Path(CONFIG["data_dir"]) / "avatar" / "installed"
            avatars_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as bundle then install
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.forgeavatar', delete=False) as f:
                temp_path = f.name
            
            bundle.save(temp_path)
            installed_path = install_avatar_bundle(temp_path, str(avatars_dir))
            os.unlink(temp_path)
            
            self._imported_path = str(installed_path)
            self.avatar_imported.emit(str(installed_path))
            
            QMessageBox.information(
                self, "Success",
                f"Avatar '{name}' imported successfully!"
            )
            self.accept()
            
        except Exception as e:
            QMessageBox.warning(
                self, "Import Failed",
                f"Failed to import avatar: {str(e)}"
            )
    
    def get_imported_path(self) -> Optional[str]:
        """Get the path where avatar was installed."""
        return self._imported_path


class AvatarCard(QFrame):
    """
    A card widget displaying an avatar with preview and info.
    """
    
    clicked = pyqtSignal(dict)  # Emits avatar info dict
    
    def __init__(self, avatar_info: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.avatar_info = avatar_info
        self._setup_ui()
        # No special cursor - use default
    
    def _setup_ui(self):
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("""
            AvatarCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #ddd;
            }
            AvatarCard:hover {
                border: 2px solid #4a9eff;
                background-color: #f8fbff;
            }
        """)
        self.setFixedSize(140, 180)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Thumbnail
        self.thumb = QLabel()
        self.thumb.setFixedSize(100, 100)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet("border: 1px solid #eee; border-radius: 4px;")
        layout.addWidget(self.thumb, alignment=Qt.AlignCenter)
        
        # Load thumbnail
        self._load_thumbnail()
        
        # Name
        name = QLabel(self.avatar_info.get("name", "Unknown"))
        name.setAlignment(Qt.AlignCenter)
        name.setStyleSheet("font-weight: bold;")
        name.setWordWrap(True)
        layout.addWidget(name)
        
        # Type badge
        avatar_type = self.avatar_info.get("type", "CUSTOM")
        type_label = QLabel(avatar_type.lower())
        type_label.setAlignment(Qt.AlignCenter)
        type_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(type_label)
    
    def _load_thumbnail(self):
        """Load the thumbnail image."""
        path = Path(self.avatar_info.get("path", ""))
        
        # Try to find base image
        for img_name in ["neutral.png", "base.png", "default.png"]:
            img_path = path / img_name
            if img_path.exists():
                pixmap = QPixmap(str(img_path))
                if not pixmap.isNull():
                    scaled = pixmap.scaled(
                        90, 90,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.thumb.setPixmap(scaled)
                    return
        
        # Fallback - placeholder
        pixmap = QPixmap(90, 90)
        pixmap.fill(QColor(200, 200, 200))
        painter = QPainter(pixmap)
        painter.setPen(QColor(150, 150, 150))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "No\nPreview")
        painter.end()
        self.thumb.setPixmap(pixmap)
    
    def mousePressEvent(self, event):
        self.clicked.emit(self.avatar_info)


class AvatarPickerDialog(QDialog):
    """
    Dialog for browsing and selecting installed avatars.
    """
    
    avatar_selected = pyqtSignal(dict)  # Emits selected avatar info
    
    def __init__(self, parent=None, avatars_dir: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Select Avatar")
        self.setMinimumSize(700, 500)
        self._avatars_dir = avatars_dir
        self._selected_avatar: Optional[Dict] = None
        self._setup_ui()
        self._load_avatars()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header with tabs
        self.tabs = QTabWidget()
        
        # Installed avatars tab
        installed_widget = QWidget()
        installed_layout = QVBoxLayout(installed_widget)
        
        # Search/filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search avatars...")
        self.search_edit.textChanged.connect(self._filter_avatars)
        filter_layout.addWidget(self.search_edit)
        
        self.type_filter = NoScrollComboBox()
        self.type_filter.setToolTip("Filter avatars by type")
        self.type_filter.addItems(["All Types", "Human", "Animal", "Robot", "Fantasy", "Abstract"])
        self.type_filter.currentTextChanged.connect(self._filter_avatars)
        filter_layout.addWidget(self.type_filter)
        
        installed_layout.addLayout(filter_layout)
        
        # Avatar grid in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        scroll.setWidget(self.grid_widget)
        
        installed_layout.addWidget(scroll)
        
        self.tabs.addTab(installed_widget, "Installed Avatars")
        
        # Sample avatars tab
        samples_widget = QWidget()
        samples_layout = QVBoxLayout(samples_widget)
        
        samples_info = QLabel(
            "Sample avatars are generated programmatically.\n"
            "Click 'Generate Samples' to create them."
        )
        samples_info.setAlignment(Qt.AlignCenter)
        samples_layout.addWidget(samples_info)
        
        generate_btn = QPushButton("Generate Sample Avatars")
        generate_btn.clicked.connect(self._generate_samples)
        samples_layout.addWidget(generate_btn, alignment=Qt.AlignCenter)
        
        self.samples_scroll = QScrollArea()
        self.samples_scroll.setWidgetResizable(True)
        self.samples_grid = QWidget()
        self.samples_grid_layout = QGridLayout(self.samples_grid)
        self.samples_scroll.setWidget(self.samples_grid)
        samples_layout.addWidget(self.samples_scroll)
        
        self.tabs.addTab(samples_widget, "Sample Avatars")
        
        layout.addWidget(self.tabs)
        
        # Preview panel
        preview_group = QGroupBox("Selected Avatar")
        preview_layout = QHBoxLayout(preview_group)
        
        self.selected_preview = AvatarPreview()
        preview_layout.addWidget(self.selected_preview)
        
        info_layout = QVBoxLayout()
        self.info_name = QLabel("")
        self.info_name.setStyleSheet("font-size: 16px; font-weight: bold;")
        info_layout.addWidget(self.info_name)
        
        self.info_details = QLabel("")
        self.info_details.setWordWrap(True)
        info_layout.addWidget(self.info_details)
        
        info_layout.addStretch()
        preview_layout.addLayout(info_layout)
        
        layout.addWidget(preview_group)
        
        # Bottom buttons
        buttons = QHBoxLayout()
        
        import_btn = QPushButton("Import New...")
        import_btn.clicked.connect(self._show_import_wizard)
        buttons.addWidget(import_btn)
        
        buttons.addStretch()
        
        self.select_btn = QPushButton("Select Avatar")
        self.select_btn.setEnabled(False)
        self.select_btn.clicked.connect(self._confirm_selection)
        buttons.addWidget(self.select_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        
        layout.addLayout(buttons)
    
    def _load_avatars(self):
        """Load installed avatars into the grid."""
        # Clear existing
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Get avatars
        avatars = list_installed_avatars(self._avatars_dir)
        self._all_avatars = avatars
        
        # Populate grid
        self._populate_grid(avatars)
    
    def _populate_grid(self, avatars: List[Dict], layout: Optional[QGridLayout] = None):
        """Populate a grid with avatar cards."""
        if layout is None:
            layout = self.grid_layout
        
        cols = 4
        for i, avatar in enumerate(avatars):
            card = AvatarCard(avatar)
            card.clicked.connect(self._on_avatar_clicked)
            layout.addWidget(card, i // cols, i % cols)
        
        # Add spacer at end
        layout.setRowStretch(len(avatars) // cols + 1, 1)
    
    def _filter_avatars(self):
        """Filter avatars based on search and type."""
        search = self.search_edit.text().lower()
        type_filter = self.type_filter.currentText()
        
        filtered = []
        for avatar in self._all_avatars:
            # Search filter
            if search:
                name = avatar.get("name", "").lower()
                desc = avatar.get("description", "").lower()
                if search not in name and search not in desc:
                    continue
            
            # Type filter
            if type_filter != "All Types":
                if avatar.get("type", "").upper() != type_filter.upper():
                    continue
            
            filtered.append(avatar)
        
        # Clear and repopulate
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self._populate_grid(filtered)
    
    def _on_avatar_clicked(self, avatar_info: Dict):
        """Handle avatar card click."""
        self._selected_avatar = avatar_info
        
        # Update preview
        self.selected_preview.load_from_path(avatar_info.get("path", ""))
        
        # Update info
        self.info_name.setText(avatar_info.get("name", "Unknown"))
        
        details = []
        details.append(f"Type: {avatar_info.get('type', 'Unknown')}")
        details.append(f"Author: {avatar_info.get('author', 'Unknown')}")
        if avatar_info.get("description"):
            details.append(f"\n{avatar_info['description']}")
        self.info_details.setText("\n".join(details))
        
        self.select_btn.setEnabled(True)
    
    def _show_import_wizard(self):
        """Show the import wizard."""
        wizard = AvatarImportWizard(self)
        if wizard.exec_() == QDialog.Accepted:
            self._load_avatars()
    
    def _generate_samples(self):
        """Generate sample avatars."""
        try:
            from .sample_avatars import generate_sample_avatars
            from ..config import CONFIG
            
            output_dir = Path(CONFIG["data_dir"]) / "avatar" / "samples"
            avatars = generate_sample_avatars(str(output_dir))
            
            # Add samples to grid
            samples = list_installed_avatars(str(output_dir))
            
            while self.samples_grid_layout.count():
                item = self.samples_grid_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            self._populate_grid(samples, self.samples_grid_layout)
            
            QMessageBox.information(
                self, "Success",
                f"Generated {len(avatars)} sample avatar sets!"
            )
            
        except Exception as e:
            QMessageBox.warning(
                self, "Error",
                f"Failed to generate samples: {str(e)}"
            )
    
    def _confirm_selection(self):
        """Confirm avatar selection."""
        if self._selected_avatar:
            self.avatar_selected.emit(self._selected_avatar)
            self.accept()
    
    def get_selected_avatar(self) -> Optional[Dict]:
        """Get the selected avatar info."""
        return self._selected_avatar


def show_avatar_import_wizard(parent=None) -> Optional[str]:
    """
    Show the avatar import wizard and return installed path.
    
    Returns:
        Path to installed avatar, or None if cancelled
    """
    if not HAS_PYQT:
        raise RuntimeError("PyQt5 required for avatar import wizard")
    
    wizard = AvatarImportWizard(parent)
    if wizard.exec_() == QDialog.Accepted:
        return wizard.get_imported_path()
    return None


def show_avatar_picker(parent=None) -> Optional[Dict]:
    """
    Show the avatar picker dialog and return selected avatar.
    
    Returns:
        Avatar info dict, or None if cancelled
    """
    if not HAS_PYQT:
        raise RuntimeError("PyQt5 required for avatar picker")
    
    picker = AvatarPickerDialog(parent)
    if picker.exec_() == QDialog.Accepted:
        return picker.get_selected_avatar()
    return None
