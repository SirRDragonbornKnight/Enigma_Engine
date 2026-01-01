"""
Advanced Vector Database Support for Enigma Engine
Supports FAISS (local), Pinecone (cloud), and simple in-memory fallback.
"""
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorDBInterface(ABC):
    """Abstract interface for vector databases."""
    
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Add vectors with IDs and optional metadata."""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for similar vectors. Returns list of (id, score, metadata)."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save index to disk."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load index from disk."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get number of vectors in the database."""
        pass


class FAISSVectorDB(VectorDBInterface):
    """FAISS-based vector database (fast, local, production-ready)."""
    
    def __init__(self, dim: int, index_type: str = "Flat"):
        """
        Initialize FAISS vector database.
        
        Args:
            dim: Dimension of vectors
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
        
        # Create index based on type
        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(dim)
        elif index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, 100)  # 100 clusters
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.id_map = {}  # Map from internal index to string ID
        self.metadata = {}  # Store metadata
        self.counter = 0
    
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
    
    def __init__(self, dim: int, api_key: str, environment: str, index_name: str = "enigma-memory"):
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


class SimpleVectorDB(VectorDBInterface):
    """Simple in-memory vector database (fallback, no dependencies)."""
    
    def __init__(self, dim: int):
        """Initialize simple vector database."""
        self.dim = dim
        self.vectors = []
        self.ids = []
        self.metadata = []
    
    def add(self, vectors: np.ndarray, ids: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Add vectors."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        for i, vec in enumerate(vectors):
            if vec.shape[0] != self.dim:
                raise ValueError(f"Expected dim {self.dim}, got {vec.shape[0]}")
            self.vectors.append(vec.astype(float))
            self.ids.append(ids[i])
            self.metadata.append(metadata[i] if metadata else {})
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search using cosine similarity."""
        if not self.vectors:
            return []
        
        vectors_array = np.stack(self.vectors, axis=0)
        query = query_vector.astype(float)
        
        # Cosine similarity
        dots = vectors_array @ query
        norms = np.linalg.norm(vectors_array, axis=1) * np.linalg.norm(query)
        similarities = dots / (norms + 1e-9)
        
        # Get top k
        indices = np.argsort(-similarities)[:top_k]
        
        results = []
        for idx in indices:
            results.append((self.ids[idx], float(similarities[idx]), self.metadata[idx]))
        
        return results
    
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
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
        """Save to disk."""
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
        """Load from disk."""
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectors = data['vectors']
            self.ids = data['ids']
            self.metadata = data['metadata']
            self.dim = data['dim']
    
    def count(self) -> int:
        """Get number of vectors."""
        return len(self.vectors)


def create_vector_db(
    dim: int,
    backend: str = "simple",
    **kwargs
) -> VectorDBInterface:
    """
    Factory function to create vector database.
    
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
        index_name = kwargs.get('index_name', 'enigma-memory')
        if not api_key or not environment:
            raise ValueError("Pinecone requires 'api_key' and 'environment'")
        return PineconeVectorDB(dim, api_key, environment, index_name)
    elif backend == "simple":
        return SimpleVectorDB(dim)
    else:
        raise ValueError(f"Unknown backend: {backend}")
