"""
Plugin Marketplace Browser

Browse, search, and install community plugins for Enigma AI Engine.
Displays available plugins from a registry with ratings, descriptions,
and one-click installation.

Usage:
    from enigma_engine.gui.tabs.marketplace_tab import MarketplaceTab
    
    tab = MarketplaceTab()
    # Add to main window tabs
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# Plugin directory
PLUGINS_DIR = Path(__file__).parent.parent.parent.parent / "plugins"

# Styles
CARD_STYLE = """
QFrame#plugin_card {
    background: #313244;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 12px;
}
QFrame#plugin_card:hover {
    border-color: #89b4fa;
}
"""

INSTALL_BTN_STYLE = """
QPushButton#install_btn {
    background: #a6e3a1;
    color: #1e1e2e;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}
QPushButton#install_btn:hover {
    background: #94e2d5;
}
QPushButton#install_btn:disabled {
    background: #45475a;
    color: #6c7086;
}
QPushButton#installed_btn {
    background: #45475a;
    color: #a6adc8;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
}
"""


@dataclass
class PluginInfo:
    """Information about a plugin."""
    id: str
    name: str
    description: str
    version: str
    author: str
    category: str
    downloads: int = 0
    rating: float = 0.0
    tags: list[str] = None
    requires: list[str] = None
    installed: bool = False
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.requires is None:
            self.requires = []


# Sample plugin registry (would be fetched from server in production)
SAMPLE_PLUGINS = [
    PluginInfo(
        id="voice-clone-xtts",
        name="XTTS Voice Cloning",
        description="Clone any voice with just 6 seconds of audio using Coqui XTTS v2. "
                   "Supports 17 languages with natural-sounding output.",
        version="1.2.0",
        author="Enigma AI Engine Community",
        category="Voice",
        downloads=12500,
        rating=4.7,
        tags=["voice", "tts", "cloning", "multilingual"],
        requires=["torch>=2.0", "TTS>=0.20"]
    ),
    PluginInfo(
        id="image-upscale",
        name="AI Image Upscaler",
        description="Upscale images 4x using Real-ESRGAN. Works great for "
                   "enhancing AI-generated images or old photos.",
        version="2.0.1",
        author="ImageMaster",
        category="Image",
        downloads=8900,
        rating=4.5,
        tags=["image", "upscale", "enhancement"],
        requires=["realesrgan>=0.3"]
    ),
    PluginInfo(
        id="code-autocomplete",
        name="Code Autocomplete Pro",
        description="Intelligent code completion using local models. "
                   "Supports Python, JavaScript, TypeScript, and more.",
        version="1.0.3",
        author="CodeCraft",
        category="Code",
        downloads=15600,
        rating=4.8,
        tags=["code", "autocomplete", "productivity"],
        requires=["transformers>=4.30"]
    ),
    PluginInfo(
        id="document-qa",
        name="Document Q&A",
        description="Chat with your documents! Upload PDFs, Word docs, or text files "
                   "and ask questions about their content.",
        version="1.5.0",
        author="DocAI",
        category="Tools",
        downloads=7200,
        rating=4.4,
        tags=["documents", "qa", "rag", "search"],
        requires=["langchain>=0.1", "pypdf>=3.0"]
    ),
    PluginInfo(
        id="stable-diffusion-xl",
        name="SDXL Integration",
        description="Generate stunning images with Stable Diffusion XL. "
                   "Includes multiple samplers and LoRA support.",
        version="1.1.0",
        author="ArtForge",
        category="Image",
        downloads=22000,
        rating=4.9,
        tags=["image", "generation", "sdxl", "art"],
        requires=["diffusers>=0.24", "accelerate>=0.24"]
    ),
    PluginInfo(
        id="video-generation",
        name="Video Gen Suite",
        description="Create videos from text prompts using AnimateDiff and "
                   "Stable Video Diffusion models.",
        version="0.9.2",
        author="VideoForge",
        category="Video",
        downloads=5400,
        rating=4.2,
        tags=["video", "generation", "animation"],
        requires=["diffusers>=0.24", "animatediff>=0.2"]
    ),
    PluginInfo(
        id="speech-recognition",
        name="Whisper Plus",
        description="Enhanced speech recognition with speaker diarization, "
                   "timestamps, and 98+ language support.",
        version="2.1.0",
        author="AudioMaster",
        category="Voice",
        downloads=9800,
        rating=4.6,
        tags=["voice", "stt", "transcription", "whisper"],
        requires=["openai-whisper>=20231117"]
    ),
    PluginInfo(
        id="memory-graph",
        name="Knowledge Graph Memory",
        description="Store and query knowledge as a graph. Better long-term "
                   "memory with entity relationships.",
        version="1.0.0",
        author="MemoryLabs",
        category="Memory",
        downloads=3200,
        rating=4.3,
        tags=["memory", "knowledge", "graph", "rag"],
        requires=["networkx>=3.0", "sentence-transformers>=2.2"]
    ),
]

CATEGORIES = ["All", "Voice", "Image", "Video", "Code", "Tools", "Memory"]


class PluginCard(QFrame):
    """Widget displaying a single plugin."""
    
    install_clicked = pyqtSignal(str)  # plugin_id
    
    def __init__(self, plugin: PluginInfo, parent=None):
        super().__init__(parent)
        self.plugin = plugin
        self.setObjectName("plugin_card")
        self.setStyleSheet(CARD_STYLE + INSTALL_BTN_STYLE)
        self.setMinimumHeight(140)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Header: Name + Version
        header = QHBoxLayout()
        
        name_label = QLabel(self.plugin.name)
        name_font = name_label.font()
        name_font.setPointSize(12)
        name_font.setBold(True)
        name_label.setFont(name_font)
        name_label.setStyleSheet("color: #cdd6f4;")
        header.addWidget(name_label)
        
        version_label = QLabel(f"v{self.plugin.version}")
        version_label.setStyleSheet("color: #6c7086; font-size: 10px;")
        header.addWidget(version_label)
        
        header.addStretch()
        
        # Category badge
        category_label = QLabel(self.plugin.category)
        category_label.setStyleSheet(
            "background: #45475a; color: #89b4fa; padding: 2px 8px; "
            "border-radius: 4px; font-size: 10px;"
        )
        header.addWidget(category_label)
        
        layout.addLayout(header)
        
        # Description
        desc_label = QLabel(self.plugin.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        desc_label.setMaximumHeight(50)
        layout.addWidget(desc_label)
        
        # Footer: Author, Stats, Install button
        footer = QHBoxLayout()
        
        author_label = QLabel(f"by {self.plugin.author}")
        author_label.setStyleSheet("color: #6c7086; font-size: 10px;")
        footer.addWidget(author_label)
        
        # Rating
        stars = int(self.plugin.rating)
        rating_text = "*" * stars + f" {self.plugin.rating:.1f}"
        rating_label = QLabel(rating_text)
        rating_label.setStyleSheet("color: #f9e2af; font-size: 10px;")
        footer.addWidget(rating_label)
        
        # Downloads
        downloads_text = self._format_downloads(self.plugin.downloads)
        downloads_label = QLabel(downloads_text)
        downloads_label.setStyleSheet("color: #6c7086; font-size: 10px;")
        footer.addWidget(downloads_label)
        
        footer.addStretch()
        
        # Install button
        if self.plugin.installed:
            self._install_btn = QPushButton("Installed")
            self._install_btn.setObjectName("installed_btn")
            self._install_btn.setEnabled(False)
        else:
            self._install_btn = QPushButton("Install")
            self._install_btn.setObjectName("install_btn")
            self._install_btn.clicked.connect(
                lambda: self.install_clicked.emit(self.plugin.id)
            )
        
        footer.addWidget(self._install_btn)
        
        layout.addLayout(footer)
    
    def _format_downloads(self, count: int) -> str:
        if count >= 1000:
            return f"{count/1000:.1f}K downloads"
        return f"{count} downloads"
    
    def mark_installed(self):
        """Mark this plugin as installed."""
        self.plugin.installed = True
        self._install_btn.setText("Installed")
        self._install_btn.setObjectName("installed_btn")
        self._install_btn.setEnabled(False)
        self._install_btn.setStyleSheet(INSTALL_BTN_STYLE)


class InstallWorker(QThread):
    """Background worker for installing plugins."""
    
    progress = pyqtSignal(int, str)  # percent, message
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, plugin: PluginInfo, parent=None):
        super().__init__(parent)
        self.plugin = plugin
    
    def run(self):
        """Simulate plugin installation."""
        import time
        
        try:
            self.progress.emit(10, "Checking dependencies...")
            time.sleep(0.5)
            
            self.progress.emit(30, "Downloading plugin...")
            time.sleep(1)
            
            self.progress.emit(60, "Installing dependencies...")
            time.sleep(0.8)
            
            self.progress.emit(80, "Configuring plugin...")
            time.sleep(0.3)
            
            # Create plugin directory
            plugin_dir = PLUGINS_DIR / self.plugin.id
            plugin_dir.mkdir(parents=True, exist_ok=True)
            
            # Write plugin info
            info_file = plugin_dir / "plugin.json"
            with open(info_file, 'w') as f:
                json.dump({
                    "id": self.plugin.id,
                    "name": self.plugin.name,
                    "version": self.plugin.version,
                    "installed_at": str(Path.ctime(plugin_dir)),
                }, f, indent=2)
            
            self.progress.emit(100, "Complete!")
            self.finished.emit(True, f"Successfully installed {self.plugin.name}")
            
        except Exception as e:
            self.finished.emit(False, f"Installation failed: {str(e)}")


class MarketplaceTab(QWidget):
    """Plugin marketplace browser tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._plugins: list[PluginInfo] = []
        self._cards: dict[str, PluginCard] = {}
        self._worker: Optional[InstallWorker] = None
        self._setup_ui()
        self._load_plugins()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Header
        header = QLabel("Plugin Marketplace")
        header_font = header.font()
        header_font.setPointSize(18)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setStyleSheet("color: #89b4fa;")
        layout.addWidget(header)
        
        subtitle = QLabel("Browse and install community plugins to extend Enigma AI Engine")
        subtitle.setStyleSheet("color: #a6adc8;")
        layout.addWidget(subtitle)
        
        # Search and filter bar
        filter_layout = QHBoxLayout()
        
        # Search box
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Search plugins...")
        self._search_box.setStyleSheet(
            "background: #313244; color: #cdd6f4; border: 1px solid #45475a; "
            "border-radius: 4px; padding: 8px;"
        )
        self._search_box.textChanged.connect(self._filter_plugins)
        filter_layout.addWidget(self._search_box, stretch=2)
        
        # Category filter
        self._category_combo = QComboBox()
        self._category_combo.addItems(CATEGORIES)
        self._category_combo.setStyleSheet(
            "background: #313244; color: #cdd6f4; border: 1px solid #45475a; "
            "border-radius: 4px; padding: 6px;"
        )
        self._category_combo.currentTextChanged.connect(self._filter_plugins)
        filter_layout.addWidget(self._category_combo)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet(
            "background: #45475a; color: #cdd6f4; border: none; "
            "border-radius: 4px; padding: 8px 16px;"
        )
        refresh_btn.clicked.connect(self._load_plugins)
        filter_layout.addWidget(refresh_btn)
        
        layout.addLayout(filter_layout)
        
        # Progress bar (hidden by default)
        self._progress_bar = QProgressBar()
        self._progress_bar.setStyleSheet(
            "QProgressBar { background: #313244; border: 1px solid #45475a; "
            "border-radius: 4px; height: 20px; }"
            "QProgressBar::chunk { background: #a6e3a1; border-radius: 3px; }"
        )
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)
        
        self._progress_label = QLabel()
        self._progress_label.setStyleSheet("color: #a6adc8;")
        self._progress_label.hide()
        layout.addWidget(self._progress_label)
        
        # Scroll area for plugin cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setContentsMargins(0, 0, 8, 0)
        self._cards_layout.setSpacing(12)
        self._cards_layout.addStretch()
        
        scroll.setWidget(self._cards_container)
        layout.addWidget(scroll)
    
    def _load_plugins(self):
        """Load plugins from registry."""
        self._plugins = SAMPLE_PLUGINS.copy()
        
        # Check which are installed
        if PLUGINS_DIR.exists():
            for plugin in self._plugins:
                plugin_dir = PLUGINS_DIR / plugin.id
                plugin.installed = plugin_dir.exists()
        
        self._display_plugins(self._plugins)
    
    def _display_plugins(self, plugins: list[PluginInfo]):
        """Display plugin cards."""
        # Clear existing cards
        for card in self._cards.values():
            card.deleteLater()
        self._cards.clear()
        
        # Remove stretch
        item = self._cards_layout.takeAt(self._cards_layout.count() - 1)
        if item:
            del item
        
        # Add new cards
        for plugin in plugins:
            card = PluginCard(plugin, self)
            card.install_clicked.connect(self._install_plugin)
            self._cards[plugin.id] = card
            self._cards_layout.addWidget(card)
        
        # Re-add stretch
        self._cards_layout.addStretch()
    
    def _filter_plugins(self):
        """Filter plugins based on search and category."""
        search_text = self._search_box.text().lower()
        category = self._category_combo.currentText()
        
        filtered = []
        for plugin in self._plugins:
            # Category filter
            if category != "All" and plugin.category != category:
                continue
            
            # Search filter
            if search_text:
                searchable = f"{plugin.name} {plugin.description} {' '.join(plugin.tags)}".lower()
                if search_text not in searchable:
                    continue
            
            filtered.append(plugin)
        
        self._display_plugins(filtered)
    
    def _install_plugin(self, plugin_id: str):
        """Start plugin installation."""
        plugin = next((p for p in self._plugins if p.id == plugin_id), None)
        if not plugin:
            return
        
        # Show progress
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self._progress_label.setText("Starting installation...")
        self._progress_label.show()
        
        # Disable all install buttons
        for card in self._cards.values():
            card._install_btn.setEnabled(False)
        
        # Start worker
        self._worker = InstallWorker(plugin, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_install_finished)
        self._worker.start()
    
    def _on_progress(self, percent: int, message: str):
        """Update progress display."""
        self._progress_bar.setValue(percent)
        self._progress_label.setText(message)
    
    def _on_install_finished(self, success: bool, message: str):
        """Handle installation completion."""
        self._progress_bar.hide()
        self._progress_label.hide()
        
        # Re-enable install buttons
        for card in self._cards.values():
            if not card.plugin.installed:
                card._install_btn.setEnabled(True)
        
        if success:
            # Mark as installed
            plugin_id = self._worker.plugin.id
            if plugin_id in self._cards:
                self._cards[plugin_id].mark_installed()
            
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Installation Failed", message)
        
        self._worker = None
