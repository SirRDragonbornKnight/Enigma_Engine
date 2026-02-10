"""
Bundle Manager Tab - Browse, load, save, export, and clone AI bundles

AI bundles (.enigma-bundle) package together trained models, personas,
and configurations for easy sharing and deployment.

Usage:
    from enigma_engine.gui.tabs.bundle_manager_tab import BundleManagerTab
    
    tab = BundleManagerTab(parent_window)
    # Add to main window tabs
"""

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...config import CONFIG

logger = logging.getLogger(__name__)

# Bundle directory
BUNDLES_DIR = Path(CONFIG.get("models_dir", "models")) / "bundles"

# =============================================================================
# STYLES
# =============================================================================

CARD_STYLE = """
QFrame#bundle_card {
    background: #313244;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 12px;
}
QFrame#bundle_card:hover {
    border-color: #89b4fa;
}
"""

ACTION_BTN_STYLE = """
QPushButton {
    background: #45475a;
    color: #cdd6f4;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background: #585b70;
}
QPushButton:pressed {
    background: #6c7086;
}
"""

PRIMARY_BTN_STYLE = """
QPushButton {
    background: #a6e3a1;
    color: #1e1e2e;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background: #94e2d5;
}
QPushButton:disabled {
    background: #45475a;
    color: #6c7086;
}
"""

DANGER_BTN_STYLE = """
QPushButton {
    background: #f38ba8;
    color: #1e1e2e;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background: #eba0ac;
}
"""

SECONDARY_BTN_STYLE = """
QPushButton {
    background: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background: #74c7ec;
}
"""


# =============================================================================
# BUNDLE INFO DATACLASS
# =============================================================================

@dataclass
class BundleDisplayInfo:
    """Information for displaying a bundle in the UI."""
    path: Path
    name: str
    version: str
    description: str
    author: str
    created: str
    positions: List[str]
    tools: List[str]
    persona_name: str
    personality_mode: str
    quality_score: float = 0.0


# =============================================================================
# WORKER THREAD FOR BUNDLE OPERATIONS
# =============================================================================

class BundleWorker(QThread):
    """Worker thread for bundle operations."""
    finished = pyqtSignal(object)  # Result data
    error = pyqtSignal(str)  # Error message
    progress = pyqtSignal(str)  # Progress message
    
    def __init__(self, operation: str, **kwargs):
        super().__init__()
        self.operation = operation
        self.kwargs = kwargs
    
    def run(self):
        try:
            if self.operation == "list":
                result = self._list_bundles()
            elif self.operation == "load":
                result = self._load_bundle()
            elif self.operation == "export":
                result = self._export_bundle()
            elif self.operation == "clone":
                result = self._clone_bundle()
            elif self.operation == "delete":
                result = self._delete_bundle()
            else:
                raise ValueError(f"Unknown operation: {self.operation}")
            
            self.finished.emit(result)
        except Exception as e:
            logger.exception(f"Bundle operation failed: {e}")
            self.error.emit(str(e))
    
    def _list_bundles(self) -> List[BundleDisplayInfo]:
        """List all available bundles."""
        from ...core.trainer_ai import get_trainer_ai
        
        trainer = get_trainer_ai()
        bundles_dir = self.kwargs.get("bundles_dir", BUNDLES_DIR)
        raw_bundles = trainer.list_bundles(bundles_dir)
        
        result = []
        for bundle_data in raw_bundles:
            bundle_path = Path(bundle_data["path"])
            manifest_path = bundle_path / "manifest.json"
            
            # Load full manifest for detailed info
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                
                persona = manifest.get("persona", {})
                metadata = manifest.get("metadata", {})
                
                result.append(BundleDisplayInfo(
                    path=bundle_path,
                    name=manifest.get("name", "Unknown"),
                    version=manifest.get("version", "1.0.0"),
                    description=manifest.get("description", ""),
                    author=manifest.get("author", "Anonymous"),
                    created=manifest.get("created", ""),
                    positions=manifest.get("capabilities", {}).get("positions_trained", []),
                    tools=manifest.get("capabilities", {}).get("tools_enabled", []),
                    persona_name=persona.get("name", "AI Assistant"),
                    personality_mode=manifest.get("personality_mode", "system_prompt"),
                    quality_score=metadata.get("quality_score", 0.0),
                ))
            except Exception as e:
                logger.warning(f"Failed to load bundle details: {e}")
        
        return result
    
    def _load_bundle(self) -> Dict[str, Any]:
        """Load a bundle and return config for activation."""
        from ...core.trainer_ai import get_trainer_ai
        
        bundle_path = Path(self.kwargs["bundle_path"])
        trainer = get_trainer_ai()
        
        self.progress.emit(f"Loading bundle from {bundle_path}...")
        spec = trainer.load_bundle(bundle_path)
        
        # Prepare load result
        return {
            "spec": spec,
            "bundle_path": bundle_path,
            "models": spec.models,
            "persona_name": spec.persona_name,
            "system_prompt": spec.system_prompt,
            "personality": spec.personality,
            "tools_enabled": spec.tools_enabled,
        }
    
    def _export_bundle(self) -> Path:
        """Export a bundle to a specified location."""
        source_path = Path(self.kwargs["source_path"])
        dest_path = Path(self.kwargs["dest_path"])
        
        self.progress.emit(f"Exporting bundle to {dest_path}...")
        
        # Create a zip file or copy directory
        if dest_path.suffix == ".zip":
            import zipfile
            with zipfile.ZipFile(dest_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in source_path.rglob("*"):
                    if file.is_file():
                        arcname = file.relative_to(source_path)
                        zipf.write(file, arcname)
        else:
            # Copy as directory
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(source_path, dest_path)
        
        return dest_path
    
    def _clone_bundle(self) -> Path:
        """Clone a bundle with a new name."""
        source_path = Path(self.kwargs["source_path"])
        new_name = self.kwargs["new_name"]
        
        self.progress.emit(f"Cloning bundle as '{new_name}'...")
        
        # Create destination path
        dest_path = BUNDLES_DIR / new_name.lower().replace(" ", "_")
        
        # Copy bundle directory
        if dest_path.exists():
            raise ValueError(f"Bundle '{new_name}' already exists")
        
        shutil.copytree(source_path, dest_path)
        
        # Update manifest with new name
        manifest_path = dest_path / "manifest.json"
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        manifest["name"] = new_name
        manifest["created"] = datetime.now().isoformat()
        manifest["version"] = "1.0.0"  # Reset version for clone
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        # Update README
        readme_path = dest_path / "README.md"
        if readme_path.exists():
            content = readme_path.read_text()
            # Replace first heading
            lines = content.split('\n')
            if lines and lines[0].startswith('# '):
                lines[0] = f"# {new_name}"
                readme_path.write_text('\n'.join(lines))
        
        return dest_path
    
    def _delete_bundle(self) -> bool:
        """Delete a bundle."""
        bundle_path = Path(self.kwargs["bundle_path"])
        
        self.progress.emit(f"Deleting bundle at {bundle_path}...")
        
        if bundle_path.exists():
            shutil.rmtree(bundle_path)
            return True
        return False


# =============================================================================
# BUNDLE CARD WIDGET
# =============================================================================

class BundleCard(QFrame):
    """Card widget displaying a single bundle."""
    
    load_requested = pyqtSignal(Path)
    export_requested = pyqtSignal(Path)
    clone_requested = pyqtSignal(Path, str)
    delete_requested = pyqtSignal(Path)
    
    def __init__(self, info: BundleDisplayInfo, parent=None):
        super().__init__(parent)
        self.info = info
        self._setup_ui()
    
    def _setup_ui(self):
        self.setObjectName("bundle_card")
        self.setStyleSheet(CARD_STYLE)
        self.setMinimumWidth(280)
        self.setMaximumWidth(350)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Header with name and version
        header_layout = QHBoxLayout()
        
        name_label = QLabel(self.info.name)
        name_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        name_label.setStyleSheet("color: #cdd6f4;")
        header_layout.addWidget(name_label)
        
        version_label = QLabel(f"v{self.info.version}")
        version_label.setStyleSheet("color: #6c7086; font-size: 10px;")
        header_layout.addWidget(version_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Description
        if self.info.description:
            desc_label = QLabel(self.info.description[:100] + ("..." if len(self.info.description) > 100 else ""))
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
            layout.addWidget(desc_label)
        
        # Persona and author info
        info_layout = QHBoxLayout()
        
        persona_label = QLabel(f"Persona: {self.info.persona_name}")
        persona_label.setStyleSheet("color: #89b4fa; font-size: 10px;")
        info_layout.addWidget(persona_label)
        
        author_label = QLabel(f"by {self.info.author}")
        author_label.setStyleSheet("color: #6c7086; font-size: 10px;")
        info_layout.addWidget(author_label)
        info_layout.addStretch()
        
        layout.addLayout(info_layout)
        
        # Capabilities badges
        caps_layout = QHBoxLayout()
        caps_layout.setSpacing(4)
        
        for position in self.info.positions[:4]:  # Show max 4
            badge = QLabel(position)
            badge.setStyleSheet("""
                background: #45475a;
                color: #cdd6f4;
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 9px;
            """)
            caps_layout.addWidget(badge)
        
        if len(self.info.positions) > 4:
            more_badge = QLabel(f"+{len(self.info.positions) - 4}")
            more_badge.setStyleSheet("""
                background: #313244;
                color: #6c7086;
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 9px;
            """)
            caps_layout.addWidget(more_badge)
        
        caps_layout.addStretch()
        layout.addLayout(caps_layout)
        
        # Quality score if available
        if self.info.quality_score > 0:
            score_label = QLabel(f"Quality: {self.info.quality_score:.1f}/10")
            score_label.setStyleSheet("color: #a6e3a1; font-size: 10px;")
            layout.addWidget(score_label)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)
        
        load_btn = QPushButton("Load")
        load_btn.setStyleSheet(PRIMARY_BTN_STYLE)
        load_btn.clicked.connect(lambda: self.load_requested.emit(self.info.path))
        btn_layout.addWidget(load_btn)
        
        clone_btn = QPushButton("Clone")
        clone_btn.setStyleSheet(SECONDARY_BTN_STYLE)
        clone_btn.clicked.connect(self._request_clone)
        btn_layout.addWidget(clone_btn)
        
        export_btn = QPushButton("Export")
        export_btn.setStyleSheet(ACTION_BTN_STYLE)
        export_btn.clicked.connect(lambda: self.export_requested.emit(self.info.path))
        btn_layout.addWidget(export_btn)
        
        delete_btn = QPushButton("X")
        delete_btn.setStyleSheet(DANGER_BTN_STYLE)
        delete_btn.setFixedWidth(30)
        delete_btn.setToolTip("Delete this bundle")
        delete_btn.clicked.connect(lambda: self.delete_requested.emit(self.info.path))
        btn_layout.addWidget(delete_btn)
        
        layout.addLayout(btn_layout)
    
    def _request_clone(self):
        """Show dialog to get new name for clone."""
        from PyQt5.QtWidgets import QInputDialog
        
        new_name, ok = QInputDialog.getText(
            self,
            "Clone Bundle",
            "Enter name for the cloned bundle:",
            QLineEdit.Normal,
            f"{self.info.name} (Copy)"
        )
        
        if ok and new_name:
            self.clone_requested.emit(self.info.path, new_name)


# =============================================================================
# CREATE BUNDLE DIALOG
# =============================================================================

class CreateBundleDialog(QDialog):
    """Dialog for creating a new bundle from current AI configuration."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create AI Bundle")
        self.setMinimumWidth(500)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Info text
        info_label = QLabel(
            "Create a shareable AI bundle from your current configuration. "
            "This packages your trained models, persona, and settings together."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #a6adc8; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Form
        form_layout = QFormLayout()
        form_layout.setSpacing(10)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("My Custom AI")
        form_layout.addRow("Bundle Name:", self.name_input)
        
        self.version_input = QLineEdit("1.0.0")
        form_layout.addRow("Version:", self.version_input)
        
        self.author_input = QLineEdit()
        self.author_input.setPlaceholderText("Your name (optional)")
        form_layout.addRow("Author:", self.author_input)
        
        self.desc_input = QPlainTextEdit()
        self.desc_input.setMaximumHeight(80)
        self.desc_input.setPlaceholderText("Describe what this AI does...")
        form_layout.addRow("Description:", self.desc_input)
        
        layout.addLayout(form_layout)
        
        # Models to include
        models_group = QGroupBox("Models to Include")
        models_layout = QVBoxLayout(models_group)
        
        self.model_list = QListWidget()
        self.model_list.setMaximumHeight(120)
        self._populate_available_models()
        models_layout.addWidget(self.model_list)
        
        layout.addWidget(models_group)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _populate_available_models(self):
        """Populate list with available trained models."""
        models_dir = Path(CONFIG.get("models_dir", "models"))
        
        # Look for .pth files
        if models_dir.exists():
            for pth_file in models_dir.glob("**/*.pth"):
                # Skip files in bundles directory
                if "bundles" in str(pth_file):
                    continue
                
                item = QListWidgetItem(pth_file.stem)
                item.setData(Qt.UserRole, str(pth_file))
                item.setCheckState(Qt.Checked)
                self.model_list.addItem(item)
        
        if self.model_list.count() == 0:
            item = QListWidgetItem("No trained models found")
            item.setFlags(Qt.NoItemFlags)
            self.model_list.addItem(item)
    
    def get_bundle_config(self) -> Dict[str, Any]:
        """Get the bundle configuration from dialog inputs."""
        # Collect selected models
        model_paths = {}
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            if item.checkState() == Qt.Checked:
                path = item.data(Qt.UserRole)
                if path:
                    # Use filename stem as position name
                    name = Path(path).stem
                    model_paths[name] = path
        
        return {
            "name": self.name_input.text() or "Unnamed Bundle",
            "version": self.version_input.text() or "1.0.0",
            "author": self.author_input.text() or "Anonymous",
            "description": self.desc_input.toPlainText(),
            "model_paths": model_paths,
        }


# =============================================================================
# MAIN BUNDLE MANAGER TAB
# =============================================================================

class BundleManagerTab(QWidget):
    """Tab for managing AI bundles."""
    
    # Signal emitted when a bundle is loaded
    bundle_loaded = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.bundles: List[BundleDisplayInfo] = []
        self.worker: Optional[BundleWorker] = None
        
        self._setup_ui()
        
        # Load bundles on startup (slight delay to not block UI)
        QTimer.singleShot(500, self.refresh_bundles)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("AI Bundle Manager")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #cdd6f4;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Action buttons
        create_btn = QPushButton("Create Bundle")
        create_btn.setStyleSheet(PRIMARY_BTN_STYLE)
        create_btn.clicked.connect(self._show_create_dialog)
        header_layout.addWidget(create_btn)
        
        import_btn = QPushButton("Import Bundle")
        import_btn.setStyleSheet(SECONDARY_BTN_STYLE)
        import_btn.clicked.connect(self._import_bundle)
        header_layout.addWidget(import_btn)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet(ACTION_BTN_STYLE)
        refresh_btn.clicked.connect(self.refresh_bundles)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Description
        desc = QLabel(
            "AI bundles package trained models, personas, and configurations together "
            "for easy sharing and deployment. Load a bundle to instantly switch your AI's personality and capabilities."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a6adc8; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Status/progress
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #89b4fa;")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)  # Indeterminate
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # Bundles grid in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        self.bundles_container = QWidget()
        self.bundles_layout = QGridLayout(self.bundles_container)
        self.bundles_layout.setSpacing(16)
        self.bundles_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        scroll.setWidget(self.bundles_container)
        layout.addWidget(scroll, 1)
        
        # Empty state message
        self.empty_label = QLabel(
            "No bundles found.\n\n"
            "Create your first bundle using 'Create Bundle' or import one with 'Import Bundle'."
        )
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color: #6c7086; font-size: 14px;")
        self.empty_label.hide()
        layout.addWidget(self.empty_label)
    
    def refresh_bundles(self):
        """Refresh the list of available bundles."""
        self._show_loading("Loading bundles...")
        
        self.worker = BundleWorker("list", bundles_dir=BUNDLES_DIR)
        self.worker.finished.connect(self._on_bundles_loaded)
        self.worker.error.connect(self._on_error)
        self.worker.start()
    
    def _on_bundles_loaded(self, bundles: List[BundleDisplayInfo]):
        """Handle loaded bundles."""
        self._hide_loading()
        self.bundles = bundles
        self._populate_bundles()
    
    def _populate_bundles(self):
        """Populate the bundles grid."""
        # Clear existing cards
        while self.bundles_layout.count():
            item = self.bundles_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not self.bundles:
            self.empty_label.show()
            return
        
        self.empty_label.hide()
        
        # Add cards in a grid (3 columns)
        cols = 3
        for i, bundle_info in enumerate(self.bundles):
            card = BundleCard(bundle_info, self)
            card.load_requested.connect(self._load_bundle)
            card.export_requested.connect(self._export_bundle)
            card.clone_requested.connect(self._clone_bundle)
            card.delete_requested.connect(self._delete_bundle)
            
            row = i // cols
            col = i % cols
            self.bundles_layout.addWidget(card, row, col)
    
    def _show_create_dialog(self):
        """Show dialog to create a new bundle."""
        dialog = CreateBundleDialog(self)
        
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.get_bundle_config()
            self._create_bundle(config)
    
    def _create_bundle(self, config: Dict[str, Any]):
        """Create a new bundle with the given configuration."""
        from ...core.trainer_ai import get_trainer_ai
        
        try:
            trainer = get_trainer_ai()
            
            # Get persona info from parent window if available
            persona_name = "AI Assistant"
            system_prompt = ""
            personality = ""
            tools_enabled = []
            
            if self.parent_window:
                # Try to get current persona
                if hasattr(self.parent_window, 'get_current_persona'):
                    persona = self.parent_window.get_current_persona()
                    if persona:
                        persona_name = persona.get("name", persona_name)
                        system_prompt = persona.get("system_prompt", "")
                        personality = persona.get("personality", "")
                
                # Try to get enabled tools
                if hasattr(self.parent_window, 'get_enabled_tools'):
                    tools_enabled = self.parent_window.get_enabled_tools()
            
            # Convert model paths
            model_paths = {k: Path(v) for k, v in config["model_paths"].items()}
            
            bundle_path = trainer.create_bundle(
                name=config["name"],
                description=config["description"],
                model_paths=model_paths,
                persona_name=persona_name,
                personality=personality,
                system_prompt=system_prompt,
                tools_enabled=tools_enabled,
                output_dir=BUNDLES_DIR,
            )
            
            self.status_label.setText(f"Created bundle: {config['name']}")
            self.status_label.setStyleSheet("color: #a6e3a1;")
            
            # Refresh list
            QTimer.singleShot(500, self.refresh_bundles)
            
        except Exception as e:
            logger.exception("Failed to create bundle")
            QMessageBox.critical(self, "Error", f"Failed to create bundle: {e}")
    
    def _load_bundle(self, bundle_path: Path):
        """Load a bundle and apply its configuration."""
        self._show_loading(f"Loading bundle...")
        
        self.worker = BundleWorker("load", bundle_path=bundle_path)
        self.worker.finished.connect(self._on_bundle_loaded)
        self.worker.error.connect(self._on_error)
        self.worker.progress.connect(self._on_progress)
        self.worker.start()
    
    def _on_bundle_loaded(self, result: Dict[str, Any]):
        """Handle bundle loaded."""
        self._hide_loading()
        
        spec = result["spec"]
        bundle_path = result["bundle_path"]
        
        # Show success message
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Bundle Loaded")
        msg.setText(f"Successfully loaded bundle: {spec.name}")
        msg.setInformativeText(
            f"Persona: {spec.persona_name}\n"
            f"Capabilities: {', '.join(spec.positions_trained)}\n"
            f"Tools: {', '.join(spec.tools_enabled) if spec.tools_enabled else 'None'}"
        )
        
        # Add option to apply persona
        msg.setStandardButtons(QMessageBox.Ok)
        apply_btn = msg.addButton("Apply Persona", QMessageBox.ActionRole)
        
        msg.exec_()
        
        if msg.clickedButton() == apply_btn:
            self._apply_bundle_persona(spec, bundle_path)
        
        # Emit signal for other components
        self.bundle_loaded.emit(result)
        
        self.status_label.setText(f"Loaded: {spec.name}")
        self.status_label.setStyleSheet("color: #a6e3a1;")
    
    def _apply_bundle_persona(self, spec, bundle_path: Path):
        """Apply the bundle's persona to the current session."""
        if not self.parent_window:
            return
        
        # Try to create/update persona
        try:
            from ...core.persona import PersonaManager
            
            pm = PersonaManager()
            
            # Create persona from bundle
            persona_data = {
                "name": spec.persona_name,
                "personality": spec.personality,
                "system_prompt": spec.system_prompt,
                "speech_patterns": spec.speech_patterns,
                "source_bundle": str(bundle_path),
            }
            
            pm.create_persona(spec.persona_name, persona_data)
            
            # Activate persona if parent supports it
            if hasattr(self.parent_window, 'set_active_persona'):
                self.parent_window.set_active_persona(spec.persona_name)
            
            self.status_label.setText(f"Applied persona: {spec.persona_name}")
            
        except Exception as e:
            logger.warning(f"Failed to apply persona: {e}")
    
    def _export_bundle(self, bundle_path: Path):
        """Export a bundle to a file."""
        # Ask for destination
        dest_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Bundle",
            str(bundle_path.name) + ".zip",
            "Zip Files (*.zip);;All Files (*)"
        )
        
        if not dest_path:
            return
        
        self._show_loading("Exporting bundle...")
        
        self.worker = BundleWorker(
            "export",
            source_path=bundle_path,
            dest_path=dest_path
        )
        self.worker.finished.connect(self._on_export_complete)
        self.worker.error.connect(self._on_error)
        self.worker.start()
    
    def _on_export_complete(self, dest_path: Path):
        """Handle export complete."""
        self._hide_loading()
        self.status_label.setText(f"Exported to: {dest_path}")
        self.status_label.setStyleSheet("color: #a6e3a1;")
        
        QMessageBox.information(
            self,
            "Export Complete",
            f"Bundle exported to:\n{dest_path}"
        )
    
    def _clone_bundle(self, bundle_path: Path, new_name: str):
        """Clone a bundle with a new name."""
        self._show_loading(f"Cloning as '{new_name}'...")
        
        self.worker = BundleWorker(
            "clone",
            source_path=bundle_path,
            new_name=new_name
        )
        self.worker.finished.connect(self._on_clone_complete)
        self.worker.error.connect(self._on_error)
        self.worker.start()
    
    def _on_clone_complete(self, new_path: Path):
        """Handle clone complete."""
        self._hide_loading()
        self.status_label.setText(f"Cloned bundle to: {new_path.name}")
        self.status_label.setStyleSheet("color: #a6e3a1;")
        
        # Refresh list
        self.refresh_bundles()
    
    def _delete_bundle(self, bundle_path: Path):
        """Delete a bundle after confirmation."""
        # Find bundle name
        bundle_name = bundle_path.name
        for b in self.bundles:
            if b.path == bundle_path:
                bundle_name = b.name
                break
        
        reply = QMessageBox.question(
            self,
            "Delete Bundle",
            f"Are you sure you want to delete '{bundle_name}'?\n\n"
            "This will permanently remove the bundle and all its models.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        self._show_loading("Deleting bundle...")
        
        self.worker = BundleWorker("delete", bundle_path=bundle_path)
        self.worker.finished.connect(self._on_delete_complete)
        self.worker.error.connect(self._on_error)
        self.worker.start()
    
    def _on_delete_complete(self, success: bool):
        """Handle delete complete."""
        self._hide_loading()
        
        if success:
            self.status_label.setText("Bundle deleted")
            self.status_label.setStyleSheet("color: #f9e2af;")
            self.refresh_bundles()
        else:
            self.status_label.setText("Failed to delete bundle")
            self.status_label.setStyleSheet("color: #f38ba8;")
    
    def _import_bundle(self):
        """Import a bundle from a file or directory."""
        # Ask for source
        source_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Bundle",
            "",
            "Zip Files (*.zip);;All Files (*)"
        )
        
        if not source_path:
            return
        
        source_path = Path(source_path)
        
        try:
            self._show_loading("Importing bundle...")
            
            if source_path.suffix == ".zip":
                import zipfile
                
                # Extract to bundles directory
                with zipfile.ZipFile(source_path, 'r') as zipf:
                    # Get the root folder name from the zip
                    names = zipf.namelist()
                    if names:
                        root = names[0].split('/')[0]
                        dest_path = BUNDLES_DIR / root
                        
                        # Extract
                        BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
                        zipf.extractall(BUNDLES_DIR)
            else:
                # Copy directory
                dest_path = BUNDLES_DIR / source_path.name
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(source_path, dest_path)
            
            self._hide_loading()
            self.status_label.setText("Bundle imported successfully")
            self.status_label.setStyleSheet("color: #a6e3a1;")
            
            # Refresh list
            self.refresh_bundles()
            
        except Exception as e:
            self._hide_loading()
            logger.exception("Failed to import bundle")
            QMessageBox.critical(self, "Error", f"Failed to import bundle: {e}")
    
    def _show_loading(self, message: str):
        """Show loading state."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #89b4fa;")
        self.progress_bar.show()
    
    def _hide_loading(self):
        """Hide loading state."""
        self.progress_bar.hide()
    
    def _on_progress(self, message: str):
        """Handle progress update."""
        self.status_label.setText(message)
    
    def _on_error(self, error: str):
        """Handle error."""
        self._hide_loading()
        self.status_label.setText(f"Error: {error}")
        self.status_label.setStyleSheet("color: #f38ba8;")
        
        QMessageBox.critical(self, "Error", error)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_bundle_manager_tab(parent=None) -> BundleManagerTab:
    """Create and return a BundleManagerTab instance."""
    return BundleManagerTab(parent)
