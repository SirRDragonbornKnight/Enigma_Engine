"""
Embedding Generation for ForgeAI Memory System
Supports local sentence-transformers and API-based embeddings.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np

from .vector_db import VectorDBInterface

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using various backends."""
    
    def __init__(self, model: str = "local", device: str = "auto"):
        """
        Initialize embedding generator.
        
        Args:
            model: Model to use ('local', 'openai', or specific model name)
            device: Device for local models ('cpu', 'cuda', 'auto')
        """
        self.model = model
        self.device = device
        self._model_instance = None
        self._dimension = None
        
        # Initialize model
        if model == "local":
            self._init_local_model()
        elif model.startswith("openai"):
            self._init_openai()
        else:
            # Try to load as sentence-transformers model
            self._init_local_model(model)
    
    def _init_local_model(self, model_name: Optional[str] = None):
        """Initialize local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use default model if not specified
            if model_name is None:
                model_name = "all-MiniLM-L6-v2"  # Fast and good quality
            
            # Set device
            if self.device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            self._model_instance = SentenceTransformer(model_name, device=device)
            self._dimension = self._model_instance.get_sentence_embedding_dimension()
            
            logger.info(f"Loaded sentence-transformers model: {model_name} (dim={self._dimension})")
            
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
            self._init_fallback()
    
    def _init_openai(self):
        """Initialize OpenAI embeddings."""
        try:
            import openai
            import os
            
            # Get API key from environment
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set, falling back to local")
                self._init_fallback()
                return
            
            self._model_instance = "openai"
            self._dimension = 1536  # OpenAI ada-002 dimension
            
            logger.info("Initialized OpenAI embeddings")
            
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai")
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize fallback embedding (hash-based)."""
        logger.warning("Using fallback hash-based embeddings (not suitable for production)")
        self._model_instance = "fallback"
        self._dimension = 128  # Default fallback dimension
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text:
            return np.zeros(self.dimension)
        
        # Local model (sentence-transformers)
        if hasattr(self._model_instance, 'encode') and self._model_instance != "fallback":
            try:
                embedding = self._model_instance.encode(text, convert_to_numpy=True)
                return embedding.astype('float32')
            except TypeError:
                # Fallback if convert_to_numpy not supported
                embedding = self._model_instance.encode(text)
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                return embedding.astype('float32')
        
        # OpenAI
        elif self._model_instance == "openai":
            return self._embed_openai(text)
        
        # Fallback
        else:
            return self._embed_fallback(text)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings (num_texts x dimension)
        """
        if not texts:
            return np.array([])
        
        # Local model
        if hasattr(self._model_instance, 'encode'):
            embeddings = self._model_instance.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True
            )
            return embeddings.astype('float32')
        
        # OpenAI (process in batches to avoid rate limits)
        elif self._model_instance == "openai":
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                for text in batch:
                    embeddings.append(self._embed_openai(text))
            return np.array(embeddings)
        
        # Fallback
        else:
            return np.array([self._embed_fallback(text) for text in texts])
    
    def _embed_openai(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        try:
            import openai
            
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            
            embedding = response['data'][0]['embedding']
            return np.array(embedding, dtype='float32')
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            return self._embed_fallback(text)
    
    def _embed_fallback(self, text: str) -> np.ndarray:
        """Generate fallback embedding using hash."""
        # Use SHA-256 hash
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        # Convert to float array
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype('float32')
        
        # Pad or truncate to desired dimension
        if len(embedding) < self._dimension:
            embedding = np.pad(embedding, (0, self._dimension - len(embedding)))
        else:
            embedding = embedding[:self._dimension]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


class AutoEmbeddingVectorDB:
    """Vector DB wrapper that auto-generates embeddings."""
    
    def __init__(
        self,
        vector_db: VectorDBInterface,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """
        Initialize auto-embedding vector DB.
        
        Args:
            vector_db: Underlying vector database
            embedding_generator: Embedding generator (creates default if None)
        """
        self.vector_db = vector_db
        
        if embedding_generator is None:
            # Create default generator
            self.embedder = EmbeddingGenerator(model="local")
            
            # Verify dimension matches
            if self.embedder.dimension != vector_db.dim:
                logger.warning(
                    f"Embedding dimension ({self.embedder.dimension}) doesn't match "
                    f"vector DB dimension ({vector_db.dim}). Using hash-based fallback."
                )
                self.embedder._dimension = vector_db.dim
        else:
            self.embedder = embedding_generator
            
            # Verify dimensions match
            if self.embedder.dimension != vector_db.dim:
                raise ValueError(
                    f"Embedding dimension ({self.embedder.dimension}) must match "
                    f"vector DB dimension ({vector_db.dim})"
                )
    
    def add_text(
        self,
        text: str,
        id_: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add text with auto-generated embedding.
        
        Args:
            text: Text to add
            id_: ID for the text
            metadata: Optional metadata
        """
        # Generate embedding
        embedding = self.embedder.embed(text)
        
        # Add to vector DB
        metadata = metadata or {}
        metadata['content'] = text  # Store text in metadata
        
        self.vector_db.add(embedding, [id_], [metadata])
    
    def add_texts(
        self,
        texts: List[str],
        ids: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Add multiple texts with auto-generated embeddings.
        
        Args:
            texts: Texts to add
            ids: IDs for the texts
            metadata: Optional metadata for each text
        """
        if len(texts) != len(ids):
            raise ValueError("Number of texts must match number of IDs")
        
        # Generate embeddings in batch
        embeddings = self.embedder.embed_batch(texts)
        
        # Store text in metadata
        if metadata is None:
            metadata = [{} for _ in texts]
        
        for i, text in enumerate(texts):
            metadata[i]['content'] = text
        
        # Add to vector DB
        self.vector_db.add(embeddings, ids, metadata)
    
    def search_text(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search using text query (auto-embeds).
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of (id, score, metadata) tuples
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query)
        
        # Search vector DB
        return self.vector_db.search(query_embedding, top_k)
    
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
        self.vector_db.delete(ids)
    
    def save(self, path) -> None:
        """Save vector DB to disk."""
        self.vector_db.save(path)
    
    def load(self, path) -> None:
        """Load vector DB from disk."""
        self.vector_db.load(path)
    
    def count(self) -> int:
        """Get number of vectors."""
        return self.vector_db.count()
    
    @property
    def dim(self) -> int:
        """Get vector dimension."""
        return self.vector_db.dim
