"""
Embeddings Tab - Generate and compare text embeddings.

Providers:
  - LOCAL: sentence-transformers (or built-in fallback)
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
        QPushButton, QTextEdit, QProgressBar,
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
from .shared_components import NoScrollComboBox, disable_scroll_on_combos

# Output directory
OUTPUT_DIR = Path(CONFIG.get("outputs_dir", "outputs")) / "embeddings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Embedding Implementations
# =============================================================================

class LocalEmbedding:
    """Local embeddings using sentence-transformers with built-in fallback."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
        self._using_builtin = False
        self._builtin_emb = None
    
    def load(self) -> bool:
        # Try sentence-transformers first
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.is_loaded = True
            self._using_builtin = False
            return True
        except ImportError:
            pass
        except Exception as e:
            print(f"sentence-transformers failed: {e}")
        
        # Fall back to built-in embeddings
        try:
            from ...builtin import BuiltinEmbeddings
            self._builtin_emb = BuiltinEmbeddings()
            if self._builtin_emb.load():
                self.is_loaded = True
                self._using_builtin = True
                print("Using built-in embeddings (TF-IDF based)")
                return True
        except Exception as e:
            print(f"Built-in embeddings failed: {e}")
        
        print("No embeddings available. Install sentence-transformers for better quality.")
        return False
    
    def unload(self):
        if self.model:
            del self.model
            self.model = None
        if self._builtin_emb:
            self._builtin_emb.unload()
            self._builtin_emb = None
        self.is_loaded = False
        self._using_builtin = False
    
    def embed(self, text: str) -> Dict[str, Any]:
        """Generate embedding for a single text."""
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}
        
        if self._using_builtin:
            return self._builtin_emb.embed(text) if self._builtin_emb else {"success": False, "error": "No embeddings"}
        
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
        
        if self._using_builtin:
            return self._builtin_emb.embed_batch(texts) if self._builtin_emb else {"success": False, "error": "No embeddings"}
        
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
        
        # Register references on parent window for chat integration
        if parent:
            parent.embed_text = self.text_input
            parent.embed_tab = self
            parent._generate_embeddings = self._generate_embedding
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)
        
        # === OUTPUT AT TOP ===
        # Results display - takes most space
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setPlaceholderText("Embedding results and similarity scores will appear here...")
        self.results_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e2e;
                border: 1px solid #313244;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', monospace;
            }
        """)
        layout.addWidget(self.results_display, stretch=1)
        
        # Stored embeddings table (compact)
        self.stored_table = QTableWidget()
        self.stored_table.setColumnCount(3)
        self.stored_table.setHorizontalHeaderLabels(["Name", "Dims", "Preview"])
        self.stored_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stored_table.setMaximumHeight(120)
        layout.addWidget(self.stored_table)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMaximumHeight(8)
        layout.addWidget(self.progress)
        
        # Status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #89b4fa;")
        layout.addWidget(self.status_label)
        
        # === CONTROLS ===
        # Provider row
        provider_row = QHBoxLayout()
        provider_row.addWidget(QLabel("Provider:"))
        self.provider_combo = NoScrollComboBox()
        self.provider_combo.addItems(['Local (sentence-transformers)', 'OpenAI (Cloud)'])
        self.provider_combo.setMaximumWidth(200)
        provider_row.addWidget(self.provider_combo)
        self.load_btn = QPushButton("Load")
        self.load_btn.setMaximumWidth(60)
        self.load_btn.clicked.connect(self._load_provider)
        provider_row.addWidget(self.load_btn)
        provider_row.addStretch()
        layout.addLayout(provider_row)
        
        # Single embedding row
        embed_row = QHBoxLayout()
        embed_row.addWidget(QLabel("Text:"))
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter text to embed...")
        embed_row.addWidget(self.text_input, stretch=1)
        embed_row.addWidget(QLabel("Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("optional")
        self.name_input.setMaximumWidth(100)
        embed_row.addWidget(self.name_input)
        self.embed_btn = QPushButton("Embed")
        self.embed_btn.setStyleSheet("background-color: #1abc9c; font-weight: bold;")
        self.embed_btn.clicked.connect(self._generate_embedding)
        embed_row.addWidget(self.embed_btn)
        layout.addLayout(embed_row)
        
        # Similarity row
        sim_row = QHBoxLayout()
        sim_row.addWidget(QLabel("Compare:"))
        self.text1_input = QLineEdit()
        self.text1_input.setPlaceholderText("Text 1...")
        sim_row.addWidget(self.text1_input, stretch=1)
        self.text2_input = QLineEdit()
        self.text2_input.setPlaceholderText("Text 2...")
        sim_row.addWidget(self.text2_input, stretch=1)
        self.similarity_btn = QPushButton("Similarity")
        self.similarity_btn.clicked.connect(self._calculate_similarity)
        sim_row.addWidget(self.similarity_btn)
        layout.addLayout(sim_row)
        
        # Export/Clear row
        btn_row = QHBoxLayout()
        self.export_btn = QPushButton("Export All")
        self.export_btn.clicked.connect(self._export_embeddings)
        btn_row.addWidget(self.export_btn)
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self._clear_embeddings)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
    
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
        text = self.text_input.text().strip()
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
            
            # Show in results display
            preview = str(embedding[:5])[:-1] + "..." if len(embedding) > 5 else str(embedding)
            self.results_display.append(
                f"<p style='color: #a6e3a1;'><b>{name}</b>: {dims} dimensions ({duration:.3f}s)</p>"
                f"<p style='color: #6c7086;'>\"{text[:50]}{'...' if len(text) > 50 else ''}\"</p>"
                f"<p style='color: #89b4fa; font-family: monospace;'>{preview}</p><hr>"
            )
            
            self.status_label.setText(
                f"Generated {dims}-dim embedding in {duration:.3f}s"
            )
        else:
            error = result.get("error", "Unknown error")
            self.results_display.append(f"<p style='color: #f38ba8;'>Error: {error}</p>")
            self.status_label.setText(f"Error: {error}")
    
    def _calculate_similarity(self):
        text1 = self.text1_input.text().strip()
        text2 = self.text2_input.text().strip()
        
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
                color = "#a6e3a1"  # green
            elif sim_percent >= 50:
                color = "#f9e2af"  # yellow
            else:
                color = "#f38ba8"  # red
            
            # Show in results display
            self.results_display.append(
                f"<p style='color: {color}; font-size: 12px;'><b>Similarity: {sim_percent:.1f}%</b></p>"
                f"<p style='color: #bac2de;'>Calculated in {duration:.3f}s</p><hr>"
            )
            
            self.status_label.setText(f"Similarity: {sim_percent:.1f}%")
        else:
            error = result.get("error", "Unknown error")
            self.results_display.append(f"<p style='color: #f38ba8;'>Error: {error}</p>")
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
