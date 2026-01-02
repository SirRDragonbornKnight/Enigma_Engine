"""
Embeddings Tab - Generate text embeddings for semantic search.
"""

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
        QPushButton, QFrame, QComboBox, QTextEdit, QProgressBar,
        QMessageBox, QGroupBox, QListWidget, QListWidgetItem,
        QLineEdit, QPlainTextEdit, QSpinBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from pathlib import Path
from ...config import CONFIG


# Provider colors for UI badges
PROVIDER_COLORS = {
    'LOCAL': '#27ae60',
    'OPENAI': '#10a37f',
    'HUGGINGFACE': '#ffcc00',
}


class EmbeddingWorker(QThread):
    """Background worker for embedding generation."""
    finished = pyqtSignal(list)  # Embedding vector
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, text, provider, parent=None):
        super().__init__(parent)
        self.text = text
        self.provider = provider
    
    def run(self):
        try:
            self.progress.emit(10)
            
            # Try to use module manager if available
            try:
                from ...modules.manager import ModuleManager
                manager = ModuleManager()
                
                if self.provider == 'LOCAL':
                    if manager.is_loaded('embedding_local'):
                        module = manager.get_module('embedding_local')
                        self.progress.emit(50)
                        result = module.embed(self.text)
                        self.progress.emit(100)
                        self.finished.emit(result)
                        return
                else:
                    if manager.is_loaded('embedding_api'):
                        module = manager.get_module('embedding_api')
                        self.progress.emit(50)
                        result = module.embed(self.text)
                        self.progress.emit(100)
                        self.finished.emit(result)
                        return
            except ImportError:
                pass
            
            # Try sentence-transformers fallback
            try:
                from sentence_transformers import SentenceTransformer
                self.progress.emit(30)
                model = SentenceTransformer('all-MiniLM-L6-v2')
                self.progress.emit(70)
                embedding = model.encode(self.text).tolist()
                self.progress.emit(100)
                self.finished.emit(embedding)
                return
            except ImportError:
                pass
            
            # Mock for demo
            self.progress.emit(100)
            self.finished.emit([0.0] * 384)  # Fake embedding
            
        except Exception as e:
            self.error.emit(str(e))


class SearchWorker(QThread):
    """Background worker for semantic search."""
    finished = pyqtSignal(list)  # Search results
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, query, documents, top_k, parent=None):
        super().__init__(parent)
        self.query = query
        self.documents = documents
        self.top_k = top_k
    
    def run(self):
        try:
            self.progress.emit(10)
            
            # Try sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer, util
                import torch
                
                self.progress.emit(30)
                model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Encode query and documents
                self.progress.emit(50)
                query_embedding = model.encode(self.query, convert_to_tensor=True)
                doc_embeddings = model.encode(self.documents, convert_to_tensor=True)
                
                # Calculate similarity
                self.progress.emit(80)
                cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
                top_results = torch.topk(cos_scores, k=min(self.top_k, len(self.documents)))
                
                results = []
                for score, idx in zip(top_results[0], top_results[1]):
                    results.append({
                        'document': self.documents[idx],
                        'score': float(score),
                        'index': int(idx)
                    })
                
                self.progress.emit(100)
                self.finished.emit(results)
                return
            except ImportError:
                pass
            
            # Mock for demo - simple keyword matching
            results = []
            query_lower = self.query.lower()
            for i, doc in enumerate(self.documents):
                if query_lower in doc.lower():
                    results.append({
                        'document': doc,
                        'score': 0.8,
                        'index': i
                    })
            
            self.progress.emit(100)
            self.finished.emit(results[:self.top_k])
            
        except Exception as e:
            self.error.emit(str(e))


class ProviderCard(QFrame):
    """Card displaying a single embedding provider."""
    
    def __init__(self, name: str, info: dict, parent=None):
        super().__init__(parent)
        self.provider_name = name
        self.provider_info = info
        self.setup_ui()
    
    def setup_ui(self):
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(1)
        self.setMaximumHeight(90)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        
        # Header
        header = QHBoxLayout()
        
        name_label = QLabel(self.provider_info.get('name', self.provider_name))
        name_label.setFont(QFont('Arial', 10, QFont.Bold))
        header.addWidget(name_label)
        
        header.addStretch()
        
        # Provider badge
        provider = self.provider_info.get('provider', 'UNKNOWN')
        color = PROVIDER_COLORS.get(provider, '#666')
        provider_label = QLabel(provider)
        provider_label.setStyleSheet(
            f"background-color: {color}; color: white; "
            f"padding: 2px 6px; border-radius: 3px; font-size: 9px;"
        )
        header.addWidget(provider_label)
        
        layout.addLayout(header)
        
        # Description
        desc = self.provider_info.get('description', 'No description')
        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #888; font-size: 9px;")
        layout.addWidget(desc_label)


class EmbeddingsTab(QWidget):
    """Tab for embeddings and semantic search."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.embed_worker = None
        self.search_worker = None
        self.documents = []
        self.setup_ui()
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        
        # Left: Provider list and document store
        left = QWidget()
        left.setMaximumWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        type_label = QLabel("Embedding Providers")
        type_label.setFont(QFont('Arial', 12, QFont.Bold))
        type_label.setStyleSheet("color: #f39c12;")
        left_layout.addWidget(type_label)
        
        # Provider cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMaximumHeight(200)
        
        cards_widget = QWidget()
        cards_layout = QVBoxLayout(cards_widget)
        cards_layout.setSpacing(6)
        
        providers = {
            'sentence_transformers': {
                'name': 'Sentence Transformers',
                'description': 'Local embeddings. Fast and free.',
                'requirements': ['sentence-transformers'],
                'provider': 'LOCAL',
            },
            'openai_embedding': {
                'name': 'OpenAI Embeddings',
                'description': 'High quality embeddings via API.',
                'requirements': ['openai'],
                'provider': 'OPENAI',
                'needs_api_key': True,
            },
        }
        
        for name, info in providers.items():
            card = ProviderCard(name, info)
            cards_layout.addWidget(card)
        
        cards_layout.addStretch()
        scroll.setWidget(cards_widget)
        left_layout.addWidget(scroll)
        
        # Document store
        doc_group = QGroupBox("Document Store")
        doc_layout = QVBoxLayout(doc_group)
        
        self.doc_list = QListWidget()
        self.doc_list.setMaximumHeight(150)
        doc_layout.addWidget(self.doc_list)
        
        doc_btn_row = QHBoxLayout()
        add_doc_btn = QPushButton("Add")
        add_doc_btn.clicked.connect(self._add_document)
        doc_btn_row.addWidget(add_doc_btn)
        
        remove_doc_btn = QPushButton("Remove")
        remove_doc_btn.clicked.connect(self._remove_document)
        doc_btn_row.addWidget(remove_doc_btn)
        
        clear_docs_btn = QPushButton("Clear")
        clear_docs_btn.clicked.connect(self._clear_documents)
        doc_btn_row.addWidget(clear_docs_btn)
        
        doc_layout.addLayout(doc_btn_row)
        
        self.doc_count_label = QLabel("0 documents")
        self.doc_count_label.setStyleSheet("color: #888; font-size: 10px;")
        doc_layout.addWidget(self.doc_count_label)
        
        left_layout.addWidget(doc_group)
        left_layout.addStretch()
        
        layout.addWidget(left)
        
        # Right: Search and embedding panels
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        # Provider selection
        provider_row = QHBoxLayout()
        provider_row.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(['Sentence Transformers (Local)', 'OpenAI'])
        provider_row.addWidget(self.provider_combo)
        provider_row.addStretch()
        right_layout.addLayout(provider_row)
        
        # Search section
        search_group = QGroupBox("Semantic Search")
        search_layout = QVBoxLayout(search_group)
        
        query_row = QHBoxLayout()
        query_row.addWidget(QLabel("Query:"))
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter search query...")
        self.query_input.returnPressed.connect(self._search)
        query_row.addWidget(self.query_input)
        search_layout.addLayout(query_row)
        
        search_opts = QHBoxLayout()
        search_opts.addWidget(QLabel("Top K:"))
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 20)
        self.top_k_spin.setValue(5)
        search_opts.addWidget(self.top_k_spin)
        
        self.search_btn = QPushButton("Search")
        self.search_btn.setStyleSheet("background-color: #f39c12; font-weight: bold;")
        self.search_btn.clicked.connect(self._search)
        search_opts.addWidget(self.search_btn)
        search_opts.addStretch()
        search_layout.addLayout(search_opts)
        
        right_layout.addWidget(search_group)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        right_layout.addWidget(self.progress)
        
        # Results
        results_label = QLabel("Search Results:")
        right_layout.addWidget(results_label)
        
        self.results_list = QListWidget()
        self.results_list.setMinimumHeight(150)
        right_layout.addWidget(self.results_list, stretch=1)
        
        # Single text embedding section
        embed_group = QGroupBox("Single Text Embedding")
        embed_layout = QVBoxLayout(embed_group)
        
        self.embed_input = QTextEdit()
        self.embed_input.setMaximumHeight(60)
        self.embed_input.setPlaceholderText("Enter text to embed...")
        embed_layout.addWidget(self.embed_input)
        
        embed_btn_row = QHBoxLayout()
        self.embed_btn = QPushButton("Generate Embedding")
        self.embed_btn.clicked.connect(self._generate_embedding)
        embed_btn_row.addWidget(self.embed_btn)
        embed_btn_row.addStretch()
        embed_layout.addLayout(embed_btn_row)
        
        self.embed_result = QLabel("Embedding dimensions: -")
        self.embed_result.setStyleSheet("color: #888; font-size: 10px;")
        embed_layout.addWidget(self.embed_result)
        
        right_layout.addWidget(embed_group)
        
        # Info
        info_label = QLabel(
            "Install sentence-transformers for local embeddings:\n"
            "pip install sentence-transformers"
        )
        info_label.setStyleSheet("color: #666; font-size: 9px;")
        right_layout.addWidget(info_label)
        
        layout.addWidget(right, stretch=1)
    
    def _add_document(self):
        """Add a document to the store."""
        from PyQt5.QtWidgets import QInputDialog
        text, ok = QInputDialog.getMultiLineText(
            self, "Add Document", "Enter document text:"
        )
        if ok and text.strip():
            self.documents.append(text.strip())
            item = QListWidgetItem(text[:50] + "..." if len(text) > 50 else text)
            item.setToolTip(text)
            self.doc_list.addItem(item)
            self._update_doc_count()
    
    def _remove_document(self):
        """Remove selected document."""
        row = self.doc_list.currentRow()
        if row >= 0:
            self.doc_list.takeItem(row)
            del self.documents[row]
            self._update_doc_count()
    
    def _clear_documents(self):
        """Clear all documents."""
        self.doc_list.clear()
        self.documents.clear()
        self._update_doc_count()
    
    def _update_doc_count(self):
        """Update document count label."""
        count = len(self.documents)
        self.doc_count_label.setText(f"{count} document{'s' if count != 1 else ''}")
    
    def _search(self):
        """Perform semantic search."""
        query = self.query_input.text().strip()
        if not query:
            QMessageBox.warning(self, "No Query", "Please enter a search query")
            return
        
        if not self.documents:
            QMessageBox.warning(self, "No Documents", "Please add documents first")
            return
        
        self.search_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.results_list.clear()
        
        self.search_worker = SearchWorker(
            query,
            self.documents,
            self.top_k_spin.value()
        )
        self.search_worker.progress.connect(self.progress.setValue)
        self.search_worker.finished.connect(self._on_search_complete)
        self.search_worker.error.connect(self._on_search_error)
        self.search_worker.start()
    
    def _on_search_complete(self, results: list):
        """Handle search completion."""
        self.search_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        self.results_list.clear()
        for r in results:
            doc = r['document']
            score = r['score']
            preview = doc[:60] + "..." if len(doc) > 60 else doc
            item = QListWidgetItem(f"[{score:.3f}] {preview}")
            item.setToolTip(doc)
            self.results_list.addItem(item)
        
        if not results:
            self.results_list.addItem("No matching documents found")
    
    def _on_search_error(self, error: str):
        """Handle search error."""
        self.search_btn.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.warning(self, "Search Failed", f"Error: {error}")
    
    def _generate_embedding(self):
        """Generate embedding for single text."""
        text = self.embed_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Text", "Please enter text to embed")
            return
        
        # Determine provider
        provider_text = self.provider_combo.currentText()
        if 'Local' in provider_text:
            provider = 'LOCAL'
        else:
            provider = 'API'
        
        self.embed_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        self.embed_worker = EmbeddingWorker(text, provider)
        self.embed_worker.progress.connect(self.progress.setValue)
        self.embed_worker.finished.connect(self._on_embedding_complete)
        self.embed_worker.error.connect(self._on_embedding_error)
        self.embed_worker.start()
    
    def _on_embedding_complete(self, embedding: list):
        """Handle embedding completion."""
        self.embed_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        dims = len(embedding)
        preview = str(embedding[:5])[:-1] + ", ..."
        self.embed_result.setText(f"Embedding dimensions: {dims}\nPreview: {preview}")
    
    def _on_embedding_error(self, error: str):
        """Handle embedding error."""
        self.embed_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.embed_result.setText(f"Error: {error}")
        QMessageBox.warning(self, "Embedding Failed", f"Error: {error}")


def create_embeddings_tab(parent) -> QWidget:
    """Factory function for creating the embeddings tab."""
    return EmbeddingsTab(parent)


if not HAS_PYQT:
    class EmbeddingsTab:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyQt5 is required for the Embeddings Tab")
    
    def create_embeddings_tab(parent):
        raise ImportError("PyQt5 is required for the Embeddings Tab")
