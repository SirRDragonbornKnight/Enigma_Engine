"""
Embeddings Tab - Generate and compare text embeddings.

Providers:
  - LOCAL: sentence-transformers (all-MiniLM-L6-v2 or other models)
  - OPENAI: OpenAI embedding API (requires API key)
"""

import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QComboBox, QTextEdit, QProgressBar,
        QMessageBox, QGroupBox, QListWidget, QListWidgetItem,
        QSplitter, QLineEdit, QTableWidget, QTableWidgetItem,
        QHeaderView
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from ...config import CONFIG

# Output directory
OUTPUT_DIR = Path(CONFIG.get("outputs_dir", "outputs")) / "embeddings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Embedding Implementations
# =============================================================================

class LocalEmbedding:
    """Local embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
    
    def load(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.is_loaded = True
            return True
        except ImportError:
            print("Install: pip install sentence-transformers")
            return False
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            return False
    
    def unload(self):
        if self.model:
            del self.model
            self.model = None
        self.is_loaded = False
    
    def embed(self, text: str) -> Dict[str, Any]:
        """Generate embedding for a single text."""
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}
        
        try:
            start = time.time()
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            return {
                "success": True,
                "embedding": embedding.tolist(),
                "dimensions": len(embedding),
                "duration": time.time() - start
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def embed_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Generate embeddings for multiple texts."""
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}
        
        try:
            start = time.time()
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            return {
                "success": True,
                "embeddings": [e.tolist() for e in embeddings],
                "dimensions": len(embeddings[0]) if len(embeddings) > 0 else 0,
                "count": len(embeddings),
                "duration": time.time() - start
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        """Calculate cosine similarity between two texts."""
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}
        
        try:
            start = time.time()
            embeddings = self.model.encode([text1, text2], convert_to_numpy=True)
            
            # Cosine similarity
            import numpy as np
            norm1 = np.linalg.norm(embeddings[0])
            norm2 = np.linalg.norm(embeddings[1])
            if norm1 == 0 or norm2 == 0:
                cos_sim = 0.0  # Handle zero vectors
            else:
                cos_sim = np.dot(embeddings[0], embeddings[1]) / (norm1 * norm2)
            
            return {
                "success": True,
                "similarity": float(cos_sim),
                "duration": time.time() - start
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class OpenAIEmbedding:
    """OpenAI embedding API (CLOUD - requires API key)."""
    
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = None
        self.is_loaded = False
    
    def load(self) -> bool:
        if not self.api_key:
            print("OpenAI requires OPENAI_API_KEY")
            return False
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self.is_loaded = True
            return True
        except ImportError:
            print("Install: pip install openai")
            return False
        except Exception as e:
            print(f"Failed to load OpenAI: {e}")
            return False
    
    def unload(self):
        self.client = None
        self.is_loaded = False
    
    def embed(self, text: str) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded or missing API key"}
        
        try:
            start = time.time()
            
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            return {
                "success": True,
                "embedding": embedding,
                "dimensions": len(embedding),
                "duration": time.time() - start
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def embed_batch(self, texts: List[str]) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded or missing API key"}
        
        try:
            start = time.time()
            
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = [d.embedding for d in response.data]
            
            return {
                "success": True,
                "embeddings": embeddings,
                "dimensions": len(embeddings[0]) if embeddings else 0,
                "count": len(embeddings),
                "duration": time.time() - start
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded or missing API key"}
        
        try:
            import numpy as np
            start = time.time()
            
            response = self.client.embeddings.create(
                model=self.model,
                input=[text1, text2]
            )
            
            emb1 = response.data[0].embedding
            emb2 = response.data[1].embedding
            
            # Cosine similarity
            emb1 = np.array(emb1)
            emb2 = np.array(emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                cos_sim = 0.0  # Handle zero vectors
            else:
                cos_sim = np.dot(emb1, emb2) / (norm1 * norm2)
            
            return {
                "success": True,
                "similarity": float(cos_sim),
                "duration": time.time() - start
            }
        except ImportError:
            return {"success": False, "error": "NumPy required for similarity"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# GUI Components
# =============================================================================

_providers = {
    'local': None,
    'openai': None,
}


def get_provider(name: str):
    global _providers
    
    if name == 'local' and _providers['local'] is None:
        _providers['local'] = LocalEmbedding()
    elif name == 'openai' and _providers['openai'] is None:
        _providers['openai'] = OpenAIEmbedding()
    
    return _providers.get(name)


class EmbeddingWorker(QThread):
    """Background worker for embedding operations."""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    
    def __init__(self, operation, provider_name, **kwargs):
        super().__init__()
        self.operation = operation  # 'embed', 'similarity', 'batch'
        self.provider_name = provider_name
        self.kwargs = kwargs
    
    def run(self):
        try:
            self.progress.emit(10)
            
            provider = get_provider(self.provider_name)
            if provider is None:
                self.finished.emit({"success": False, "error": "Unknown provider"})
                return
            
            if not provider.is_loaded:
                self.progress.emit(20)
                if not provider.load():
                    self.finished.emit({"success": False, "error": "Failed to load provider"})
                    return
            
            self.progress.emit(50)
            
            if self.operation == 'embed':
                result = provider.embed(self.kwargs.get('text', ''))
            elif self.operation == 'similarity':
                result = provider.similarity(
                    self.kwargs.get('text1', ''),
                    self.kwargs.get('text2', '')
                )
            elif self.operation == 'batch':
                result = provider.embed_batch(self.kwargs.get('texts', []))
            else:
                result = {"success": False, "error": f"Unknown operation: {self.operation}"}
            
            self.progress.emit(100)
            self.finished.emit(result)
            
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)})


class EmbeddingsTab(QWidget):
    """Tab for text embeddings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.worker = None
        self.stored_embeddings = {}  # name -> embedding vector
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Text Embeddings")
        header.setFont(QFont('Arial', 14, QFont.Bold))
        header.setStyleSheet("color: #1abc9c;")
        layout.addWidget(header)
        
        # Provider selection
        provider_group = QGroupBox("Provider")
        provider_layout = QHBoxLayout()
        
        self.provider_combo = QComboBox()
        self.provider_combo.addItems([
            'Local (sentence-transformers)',
            'OpenAI (Cloud)'
        ])
        provider_layout.addWidget(self.provider_combo)
        
        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self._load_provider)
        provider_layout.addWidget(self.load_btn)
        
        provider_layout.addStretch()
        provider_group.setLayout(provider_layout)
        layout.addWidget(provider_group)
        
        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Input
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Single embedding
        single_group = QGroupBox("Generate Embedding")
        single_layout = QVBoxLayout()
        
        self.text_input = QTextEdit()
        self.text_input.setMaximumHeight(80)
        self.text_input.setPlaceholderText("Enter text to embed...")
        single_layout.addWidget(self.text_input)
        
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Save as:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("optional name")
        name_row.addWidget(self.name_input)
        single_layout.addLayout(name_row)
        
        self.embed_btn = QPushButton("Generate Embedding")
        self.embed_btn.setStyleSheet("background-color: #1abc9c; font-weight: bold;")
        self.embed_btn.clicked.connect(self._generate_embedding)
        single_layout.addWidget(self.embed_btn)
        
        single_group.setLayout(single_layout)
        left_layout.addWidget(single_group)
        
        # Similarity
        sim_group = QGroupBox("Compare Similarity")
        sim_layout = QVBoxLayout()
        
        self.text1_input = QTextEdit()
        self.text1_input.setMaximumHeight(60)
        self.text1_input.setPlaceholderText("Text 1...")
        sim_layout.addWidget(self.text1_input)
        
        self.text2_input = QTextEdit()
        self.text2_input.setMaximumHeight(60)
        self.text2_input.setPlaceholderText("Text 2...")
        sim_layout.addWidget(self.text2_input)
        
        self.similarity_btn = QPushButton("Calculate Similarity")
        self.similarity_btn.clicked.connect(self._calculate_similarity)
        sim_layout.addWidget(self.similarity_btn)
        
        sim_group.setLayout(sim_layout)
        left_layout.addWidget(sim_group)
        
        left_layout.addStretch()
        splitter.addWidget(left_panel)
        
        # Right panel - Stored embeddings
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        stored_group = QGroupBox("Stored Embeddings")
        stored_layout = QVBoxLayout()
        
        self.stored_table = QTableWidget()
        self.stored_table.setColumnCount(3)
        self.stored_table.setHorizontalHeaderLabels(["Name", "Dimensions", "Preview"])
        self.stored_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stored_layout.addWidget(self.stored_table)
        
        btn_row = QHBoxLayout()
        
        self.export_btn = QPushButton("Export All")
        self.export_btn.clicked.connect(self._export_embeddings)
        btn_row.addWidget(self.export_btn)
        
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self._clear_embeddings)
        btn_row.addWidget(self.clear_btn)
        
        btn_row.addStretch()
        stored_layout.addLayout(btn_row)
        
        stored_group.setLayout(stored_layout)
        right_layout.addWidget(stored_group)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 400])
        
        layout.addWidget(splitter)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Status
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Info
        info_label = QLabel(
            "💡 Embeddings convert text into numerical vectors for semantic search.\n"
            "Local uses sentence-transformers (384 dims). OpenAI uses text-embedding-3-small (1536 dims)."
        )
        info_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(info_label)
    
    def _get_provider_name(self) -> str:
        text = self.provider_combo.currentText()
        if 'Local' in text:
            return 'local'
        elif 'OpenAI' in text:
            return 'openai'
        return 'local'
    
    def _load_provider(self):
        provider_name = self._get_provider_name()
        provider = get_provider(provider_name)
        
        if provider and not provider.is_loaded:
            self.status_label.setText(f"Loading {provider_name}...")
            self.load_btn.setEnabled(False)
            
            from PyQt5.QtCore import QTimer
            def do_load():
                success = provider.load()
                if success:
                    self.status_label.setText(f"{provider_name} loaded!")
                else:
                    self.status_label.setText(f"Failed to load {provider_name}")
                self.load_btn.setEnabled(True)
            
            QTimer.singleShot(100, do_load)
    
    def _generate_embedding(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Text", "Please enter text to embed")
            return
        
        provider_name = self._get_provider_name()
        name = self.name_input.text().strip() or f"emb_{len(self.stored_embeddings) + 1}"
        
        self.embed_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Generating embedding...")
        
        self.worker = EmbeddingWorker('embed', provider_name, text=text)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(lambda r: self._on_embed_complete(r, name, text))
        self.worker.start()
    
    def _on_embed_complete(self, result: dict, name: str, text: str):
        self.embed_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        if result.get("success"):
            embedding = result.get("embedding", [])
            dims = result.get("dimensions", 0)
            duration = result.get("duration", 0)
            
            # Store embedding
            self.stored_embeddings[name] = {
                "text": text,
                "embedding": embedding
            }
            self._update_table()
            
            self.status_label.setText(
                f"Generated {dims}-dim embedding in {duration:.3f}s"
            )
        else:
            error = result.get("error", "Unknown error")
            self.status_label.setText(f"Error: {error}")
    
    def _calculate_similarity(self):
        text1 = self.text1_input.toPlainText().strip()
        text2 = self.text2_input.toPlainText().strip()
        
        if not text1 or not text2:
            QMessageBox.warning(self, "Missing Text", "Please enter both texts")
            return
        
        provider_name = self._get_provider_name()
        
        self.similarity_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Calculating similarity...")
        
        self.worker = EmbeddingWorker('similarity', provider_name, text1=text1, text2=text2)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_similarity_complete)
        self.worker.start()
    
    def _on_similarity_complete(self, result: dict):
        self.similarity_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        if result.get("success"):
            similarity = result.get("similarity", 0)
            duration = result.get("duration", 0)
            
            # Show as percentage
            sim_percent = similarity * 100
            
            # Color based on similarity
            if sim_percent >= 80:
                color = "#27ae60"  # green
            elif sim_percent >= 50:
                color = "#f39c12"  # orange
            else:
                color = "#e74c3c"  # red
            
            self.status_label.setText(
                f"<span style='color: {color}; font-weight: bold;'>"
                f"Similarity: {sim_percent:.1f}%</span> (calculated in {duration:.3f}s)"
            )
        else:
            error = result.get("error", "Unknown error")
            self.status_label.setText(f"Error: {error}")
    
    def _update_table(self):
        self.stored_table.setRowCount(len(self.stored_embeddings))
        
        for row, (name, data) in enumerate(self.stored_embeddings.items()):
            embedding = data.get("embedding", [])
            
            self.stored_table.setItem(row, 0, QTableWidgetItem(name))
            self.stored_table.setItem(row, 1, QTableWidgetItem(str(len(embedding))))
            
            # Preview first few values
            preview = str(embedding[:3])[:-1] + "..." if len(embedding) > 3 else str(embedding)
            self.stored_table.setItem(row, 2, QTableWidgetItem(preview))
    
    def _export_embeddings(self):
        if not self.stored_embeddings:
            QMessageBox.warning(self, "Nothing to Export", "No embeddings stored")
            return
        
        timestamp = int(time.time())
        filename = f"embeddings_{timestamp}.json"
        filepath = OUTPUT_DIR / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.stored_embeddings, f, indent=2)
        
        self.status_label.setText(f"Exported to: {filepath}")
    
    def _clear_embeddings(self):
        if not self.stored_embeddings:
            return
        
        reply = QMessageBox.question(
            self, "Clear All",
            "Clear all stored embeddings?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.stored_embeddings.clear()
            self._update_table()
            self.status_label.setText("Cleared all embeddings")


def create_embeddings_tab(parent) -> QWidget:
    """Factory function for creating the embeddings tab."""
    if not HAS_PYQT:
        raise ImportError("PyQt5 is required for the Embeddings Tab")
    return EmbeddingsTab(parent)
