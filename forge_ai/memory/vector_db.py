"""
================================================================================
🔍 VECTOR DATABASE - SEMANTIC SEARCH ENGINE
================================================================================

Find memories by MEANING, not just keywords! Converts text to vectors
and finds similar items using mathematical distance.

📍 FILE: forge_ai/memory/vector_db.py
🏷️ TYPE: Vector Storage & Similarity Search
🎯 MAIN CLASSES: SimpleVectorDB, FAISSVectorDB, VectorDBInterface

┌─────────────────────────────────────────────────────────────────────────────┐
│  SEMANTIC SEARCH FLOW:                                                      │
│                                                                             │
│  Search: "feline pets"                                                     │
│     │                                                                       │
│     ▼                                                                       │
│  [Embedding] → [0.2, 0.8, 0.1, ...]  (convert to vector)                   │
│     │                                                                       │
│     ▼                                                                       │
│  [VectorDB Search] → Find similar vectors (cosine similarity)              │
│     │                                                                       │
│     ▼                                                                       │
│  Results: "Tell me about cats" (similar meaning!)                          │
└─────────────────────────────────────────────────────────────────────────────┘

📊 AVAILABLE BACKENDS:
    • SimpleVectorDB  - Pure Python, no dependencies (fallback)
    • FAISSVectorDB   - Facebook AI's fast similarity search
    • PineconeVectorDB - Cloud-based (requires API key)

🔗 CONNECTED FILES:
    → USES:      forge_ai/memory/embeddings.py (convert text to vectors)
    ← USED BY:   forge_ai/memory/manager.py (ConversationManager)
    ← USED BY:   forge_ai/memory/rag.py (retrieval-augmented generation)

📖 SEE ALSO:
    • forge_ai/memory/manager.py    - Uses this for conversation search
    • forge_ai/memory/embeddings.py - Converts text to vectors
    • forge_ai/memory/search.py     - High-level search interface
"""
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# 📐 VECTOR DATABASE INTERFACE
# =============================================================================

class VectorDBInterface(ABC):
    """
    Abstract interface for vector databases.
    
    📖 WHAT IS A VECTOR DATABASE?
    A special database that stores "vectors" (lists of numbers) and can
    quickly find similar vectors using mathematical distance measures.
    
    📐 WHY USE VECTORS?
    Text like "I love cats" gets converted to numbers like [0.2, 0.8, ...].
    Similar text gets similar numbers, so we can find related content
    by finding vectors that are "close together" in mathematical space.
    
    📐 IMPLEMENTATIONS:
    - SimpleVectorDB: Pure Python, no dependencies, good for small data
    - FAISSVectorDB: Facebook's library, fast, good for large data
    - PineconeVectorDB: Cloud service, scales to millions of vectors
    
    🔗 CONNECTS TO:
      ← Implemented by SimpleVectorDB, FAISSVectorDB, PineconeVectorDB
      → Used by ConversationManager for semantic search
    """
    
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """
        Add vectors with IDs and optional metadata.
        
        Args:
            vectors: Array of shape (n, dim) containing n vectors
            ids: List of unique identifiers for each vector
            metadata: Optional list of dicts with extra info for each vector
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: The vector to search for
            top_k: Number of results to return
            
        Returns:
            List of (id, similarity_score, metadata) tuples, sorted by score
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save index to disk for persistence."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load index from disk."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get number of vectors in the database."""
        pass


# =============================================================================
# ⚡ FAISS VECTOR DATABASE (Fast, Production-Ready)
# =============================================================================

class FAISSVectorDB(VectorDBInterface):
    """
    FAISS-based vector database (fast, local, production-ready).
    
    📖 WHAT IS FAISS?
    Facebook AI Similarity Search - a library for efficient similarity
    search. Can handle millions of vectors with sub-millisecond queries!
    
    📐 INDEX TYPES:
    - Flat: Exact search, slow for large data, best accuracy
    - IVFFlat: Clusters data, fast approximate search
    - HNSW: Graph-based, very fast, good accuracy
    
    📐 WHEN TO USE:
    - Large datasets (>10,000 vectors)
    - Need fast search (<10ms)
    - Running locally (not cloud)
    """
    
    def __init__(self, dim: int, index_type: str = "Flat"):
        """
        Initialize FAISS vector database.
        
        Args:
            dim: Dimension of vectors (e.g., 128, 512, 768)
            index_type: Type of index ('Flat', 'IVFFlat', 'HNSW')
        """
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu (or faiss-gpu)"
            )
        
        self.dim = dim
        self.index_type = index_type
        
        # ─────────────────────────────────────────────────────────────────────
        # CREATE INDEX based on type
        # Different index types trade off speed vs accuracy
        # ─────────────────────────────────────────────────────────────────────
        if index_type == "Flat":
            # Exact L2 distance search - slow but accurate
            self.index = faiss.IndexFlatL2(dim)
        elif index_type == "IVFFlat":
            # Inverted file index - clusters vectors for faster search
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, 100)  # 100 clusters
        elif index_type == "HNSW":
            # Hierarchical Navigable Small World - graph-based, very fast
            self.index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors per node
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # ─────────────────────────────────────────────────────────────────────
        # METADATA STORAGE
        # FAISS only stores vectors, so we track IDs and metadata separately
        # ─────────────────────────────────────────────────────────────────────
        self.id_map = {}    # Internal index → string ID
        self.metadata = {}  # ID → metadata dict
        self.counter = 0    # Next internal index
    
    def add(self, vectors: np.ndarray, ids: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Add vectors to the index."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        vectors = vectors.astype('float32')
        
        # Train index if needed (for IVF)
        if self.index_type == "IVFFlat" and not self.index.is_trained:
            self.index.train(vectors)
        
        # Add to index
        self.index.add(vectors)
        
        # Store mappings
        for i, id_ in enumerate(ids):
            self.id_map[self.counter] = id_
            if metadata:
                self.metadata[id_] = metadata[i]
            self.counter += 1
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for similar vectors."""
        if self.index.ntotal == 0:
            return []
        
        query_vector = query_vector.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.id_map:
                id_ = self.id_map[idx]
                meta = self.metadata.get(id_, {})
                results.append((id_, float(dist), meta))
        
        return results
    
    def delete(self, ids: List[str]) -> None:
        """Delete vectors (FAISS doesn't support deletion, so we mark as deleted)."""
        for id_ in ids:
            # Remove from mappings
            if id_ in self.metadata:
                del self.metadata[id_]
            # Note: FAISS doesn't support true deletion, would need rebuild
            logger.warning("FAISS doesn't support efficient deletion. Consider rebuilding index.")
    
    def save(self, path: Path) -> None:
        """Save index to disk."""
        import pickle
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        self.faiss.write_index(self.index, str(path))
        
        # Save metadata
        meta_path = path.with_suffix('.meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'id_map': self.id_map,
                'metadata': self.metadata,
                'counter': self.counter,
                'dim': self.dim,
                'index_type': self.index_type
            }, f)
    
    def load(self, path: Path) -> None:
        """Load index from disk."""
        import pickle
        
        # Load FAISS index
        self.index = self.faiss.read_index(str(path))
        
        # Load metadata
        meta_path = path.with_suffix('.meta.pkl')
        with open(meta_path, 'rb') as f:
            data = pickle.load(f)
            self.id_map = data['id_map']
            self.metadata = data['metadata']
            self.counter = data['counter']
            self.dim = data['dim']
            self.index_type = data['index_type']
    
    def count(self) -> int:
        """Get number of vectors."""
        return self.index.ntotal


class PineconeVectorDB(VectorDBInterface):
    """Pinecone-based vector database (cloud, managed, scalable)."""
    
    def __init__(self, dim: int, api_key: str, environment: str, index_name: str = "forge-memory"):
        """
        Initialize Pinecone vector database.
        
        Args:
            dim: Dimension of vectors
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., 'us-west1-gcp')
            index_name: Name of the index
        """
        try:
            import pinecone
            self.pinecone = pinecone
        except ImportError:
            raise ImportError("Pinecone not installed. Install with: pip install pinecone-client")
        
        self.dim = dim
        self.index_name = index_name
        
        # Initialize Pinecone
        self.pinecone.init(api_key=api_key, environment=environment)
        
        # Create or get index
        if index_name not in self.pinecone.list_indexes():
            self.pinecone.create_index(index_name, dimension=dim, metric="cosine")
        
        self.index = self.pinecone.Index(index_name)
    
    def add(self, vectors: np.ndarray, ids: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Add vectors to Pinecone."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        # Prepare upsert data
        to_upsert = []
        for i, (vec, id_) in enumerate(zip(vectors, ids)):
            item = (id_, vec.tolist())
            if metadata:
                item = (id_, vec.tolist(), metadata[i])
            to_upsert.append(item)
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(to_upsert), batch_size):
            batch = to_upsert[i:i+batch_size]
            self.index.upsert(vectors=batch)
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search in Pinecone."""
        results = self.index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        output = []
        for match in results.matches:
            output.append((match.id, match.score, match.metadata or {}))
        
        return output
    
    def delete(self, ids: List[str]) -> None:
        """Delete vectors from Pinecone."""
        self.index.delete(ids=ids)
    
    def save(self, path: Path) -> None:
        """Pinecone is cloud-based, no local save needed."""
        logger.info("Pinecone index is cloud-based, no local save required")
    
    def load(self, path: Path) -> None:
        """Pinecone is cloud-based, no local load needed."""
        logger.info("Pinecone index is cloud-based, no local load required")
    
    def count(self) -> int:
        """Get number of vectors."""
        stats = self.index.describe_index_stats()
        return stats.total_vector_count


# =============================================================================
# 🟢 SIMPLE VECTOR DATABASE (Pure Python Fallback)
# =============================================================================

class SimpleVectorDB(VectorDBInterface):
    """
    Simple in-memory vector database (fallback, no dependencies).
    
    📖 WHAT THIS IS:
    A pure Python implementation of a vector database.
    No fancy libraries needed - just numpy!
    
    📐 HOW IT WORKS:
    
    ┌──────────────────────────────────────────────────────────────────────┐
    │  ADDING VECTORS:                                                     │
    │    add([0.2, 0.8, ...], "message_123")                              │
    │                ↓                                                     │
    │    vectors: [ [0.2, 0.8, ...], [0.1, 0.9, ...], ... ]              │
    │    ids:     [ "message_123",   "message_456",   ... ]              │
    │                                                                      │
    │  SEARCHING:                                                          │
    │    search([0.21, 0.79, ...])  # Query vector                        │
    │                ↓                                                     │
    │    1. Compute cosine similarity with ALL stored vectors             │
    │    2. Sort by similarity (highest first)                            │
    │    3. Return top K results                                          │
    │                                                                      │
    │  COSINE SIMILARITY:                                                  │
    │    similarity = (A · B) / (|A| × |B|)                               │
    │    Range: -1 (opposite) to 1 (identical)                            │
    └──────────────────────────────────────────────────────────────────────┘
    
    📐 WHEN TO USE:
    - Small datasets (<10,000 vectors)
    - Don't want to install FAISS
    - Quick prototyping
    
    📐 LIMITATIONS:
    - Slow for large datasets (O(n) search)
    - All vectors must fit in memory
    """
    
    def __init__(self, dim: int):
        """
        Initialize simple vector database.
        
        Args:
            dim: Dimension of vectors (all vectors must have this size)
        """
        self.dim = dim
        self.vectors = []   # List of numpy arrays
        self.ids = []       # List of string IDs (same order as vectors)
        self.metadata = []  # List of metadata dicts (same order as vectors)
    
    def add(self, vectors, ids, metadata: Optional[List[Dict]] = None) -> None:
        """
        Add vectors to the database.
        
        📖 SUPPORTS TWO APIS:
        - Legacy: add(single_vector, single_id)
        - New:    add(array_of_vectors, list_of_ids)
        
        Args:
            vectors: Vector or array of vectors
            ids: Single ID (legacy) or list of IDs (new API)
            metadata: Optional metadata list
        """
        # ─────────────────────────────────────────────────────────────────────
        # CONVERT TO NUMPY: Handle various input types
        # ─────────────────────────────────────────────────────────────────────
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        
        # ─────────────────────────────────────────────────────────────────────
        # BACKWARD COMPATIBILITY: Handle single ID (legacy API)
        # ─────────────────────────────────────────────────────────────────────
        if isinstance(ids, str):
            ids = [ids]
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
        
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        # ─────────────────────────────────────────────────────────────────────
        # ADD EACH VECTOR: Store with its ID and metadata
        # ─────────────────────────────────────────────────────────────────────
        for i, vec in enumerate(vectors):
            # Validate dimension
            if vec.shape[0] != self.dim:
                raise ValueError(f"Expected dim {self.dim}, got {vec.shape[0]}")
            
            self.vectors.append(vec.astype(float))
            self.ids.append(ids[i])
            self.metadata.append(metadata[i] if metadata else {})
    
    def search(self, query_vector: np.ndarray, top_k: int = 5, topk: int = None) -> List[Tuple[str, float, Dict]]:
        """
        Search using cosine similarity.
        
        📖 COSINE SIMILARITY EXPLAINED:
        Measures the angle between two vectors, ignoring magnitude.
        
        📐 FORMULA:
        similarity = (A · B) / (|A| × |B|)
        
        Where:
        - A · B = dot product (sum of element-wise multiplication)
        - |A| = magnitude (sqrt of sum of squares)
        
        📐 INTERPRETATION:
        - 1.0 = vectors point same direction (same meaning)
        - 0.0 = vectors are perpendicular (unrelated)
        - -1.0 = vectors point opposite (opposite meaning)
        
        Args:
            query_vector: Vector to search for
            top_k: Number of results (new API)
            topk: Number of results (legacy API, deprecated)
            
        Returns:
            List of (id, similarity_score, metadata) tuples, best first
        """
        # Support legacy topk parameter
        if topk is not None:
            top_k = topk
        
        if not self.vectors:
            return []
        
        # ─────────────────────────────────────────────────────────────────────
        # CONVERT QUERY: Ensure numpy array with correct dtype
        # ─────────────────────────────────────────────────────────────────────
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector)
        
        # ─────────────────────────────────────────────────────────────────────
        # COMPUTE COSINE SIMILARITY with all stored vectors
        # This is O(n) - checks every vector in the database
        # ─────────────────────────────────────────────────────────────────────
        vectors_array = np.stack(self.vectors, axis=0)  # Shape: (n, dim)
        query = query_vector.astype(float)               # Shape: (dim,)
        
        # Dot product: how much vectors "align"
        dots = vectors_array @ query  # Shape: (n,)
        
        # Normalize by magnitudes to get cosine similarity
        norms = np.linalg.norm(vectors_array, axis=1) * np.linalg.norm(query)
        similarities = dots / (norms + 1e-9)  # Add epsilon to avoid division by zero
        
        # ─────────────────────────────────────────────────────────────────────
        # GET TOP K: Sort by similarity (descending) and take top k
        # ─────────────────────────────────────────────────────────────────────
        indices = np.argsort(-similarities)[:top_k]  # Negative for descending
        
        results = []
        for idx in indices:
            results.append((self.ids[idx], float(similarities[idx]), self.metadata[idx]))
        
        return results
    
    def delete(self, ids: List[str]) -> None:
        """
        Delete vectors by IDs.
        
        📖 NOTE:
        This modifies lists in-place. For very large databases,
        consider marking as deleted and periodically compacting.
        """
        # Find indices to remove
        to_remove = []
        for i, id_ in enumerate(self.ids):
            if id_ in ids:
                to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for i in reversed(to_remove):
            del self.vectors[i]
            del self.ids[i]
            del self.metadata[i]
    
    def save(self, path: Path) -> None:
        """
        Save database to disk using pickle.
        
        📖 FILE FORMAT:
        {
            'vectors': list of numpy arrays,
            'ids': list of string IDs,
            'metadata': list of dicts,
            'dim': vector dimension
        }
        """
        import pickle
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'vectors': self.vectors,
                'ids': self.ids,
                'metadata': self.metadata,
                'dim': self.dim
            }, f)
    
    def load(self, path: Path) -> None:
        """Load database from disk."""
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectors = data['vectors']
            self.ids = data['ids']
            self.metadata = data['metadata']
            self.dim = data['dim']
    
    def count(self) -> int:
        """Get number of vectors in the database."""
        return len(self.vectors)


# =============================================================================
# 🏭 FACTORY FUNCTION
# =============================================================================

def create_vector_db(
    dim: int,
    backend: str = "simple",
    **kwargs
) -> VectorDBInterface:
    """
    Factory function to create vector database.
    
    📖 USE THIS to create the right database for your needs:
    
    📐 BACKEND OPTIONS:
    - "simple": Pure Python, no dependencies, good for small data
    - "faiss": Fast local search, good for medium/large data
    - "pinecone": Cloud service, scales to millions
    
    📐 EXAMPLE:
        # Simple (default, no setup needed)
        db = create_vector_db(128)
        
        # FAISS (fast, local)
        db = create_vector_db(128, backend="faiss", index_type="HNSW")
        
        # Pinecone (cloud, needs API key)
        db = create_vector_db(128, backend="pinecone", 
                              api_key="...", environment="us-west1-gcp")
    
    Args:
        dim: Dimension of vectors
        backend: Type of backend ('simple', 'faiss', 'pinecone')
        **kwargs: Backend-specific arguments
        
    Returns:
        VectorDBInterface instance
    """
    if backend == "faiss":
        index_type = kwargs.get('index_type', 'Flat')
        return FAISSVectorDB(dim, index_type)
    elif backend == "pinecone":
        api_key = kwargs.get('api_key')
        environment = kwargs.get('environment')
        index_name = kwargs.get('index_name', 'forge-memory')
        if not api_key or not environment:
            raise ValueError("Pinecone requires 'api_key' and 'environment'")
        return PineconeVectorDB(dim, api_key, environment, index_name)
    elif backend == "simple":
        return SimpleVectorDB(dim)
    else:
        raise ValueError(f"Unknown backend: {backend}")
