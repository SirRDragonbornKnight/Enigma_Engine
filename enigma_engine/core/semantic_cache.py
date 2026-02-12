"""
Semantic Caching System

Cache responses based on semantic similarity of queries.
Uses embedding-based similarity for intelligent cache hits.

FILE: enigma_engine/core/semantic_cache.py
TYPE: Core/Optimization
MAIN CLASSES: SemanticCache, SemanticIndex, SimilarityMatcher
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CachedItem:
    """A semantically cached item."""
    key: str
    query: str
    response: Any
    embedding: np.ndarray
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    
    # Metadata
    query_tokens: int = 0
    response_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def touch(self) -> None:
        """Mark as recently accessed."""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheConfig:
    """Semantic cache configuration."""
    # Similarity threshold (0.0 - 1.0)
    similarity_threshold: float = 0.92
    
    # Cache size limits
    max_items: int = 10000
    max_memory_mb: float = 500.0
    
    # Embedding
    embedding_dim: int = 384
    normalize_embeddings: bool = True
    
    # Index
    use_faiss: bool = True
    index_type: str = "flat"  # flat, ivf, hnsw
    nlist: int = 100  # For IVF index
    nprobe: int = 10  # For IVF search
    
    # Eviction
    ttl_seconds: Optional[float] = 3600  # 1 hour
    eviction_check_interval: float = 60.0
    
    # Persistence
    persist_path: Optional[Path] = None


class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
    
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        raise NotImplementedError
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts."""
        return np.array([self.embed(t) for t in texts])


class HashEmbedding(EmbeddingProvider):
    """Simple hash-based embeddings (for testing)."""
    
    def embed(self, text: str) -> np.ndarray:
        """Create deterministic embedding from hash."""
        # Use SHA256 and expand to embedding dim
        h = hashlib.sha256(text.lower().encode()).digest()
        
        # Expand hash to embedding size
        expanded = []
        for i in range(0, self.dim, 32):
            h = hashlib.sha256(h).digest()
            expanded.extend([b / 255.0 - 0.5 for b in h])
        
        emb = np.array(expanded[:self.dim], dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-8)


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Sentence-transformers based embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384) -> None:
        super().__init__(dim)
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.warning("sentence-transformers not available, using hash embeddings")
                return None
        return self._model
    
    def embed(self, text: str) -> np.ndarray:
        """Embed text using sentence transformer."""
        if self.model is None:
            # Fallback to hash
            return HashEmbedding(self.dim).embed(text)
        
        return self.model.encode(text, normalize_embeddings=True)
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed batch of texts."""
        if self.model is None:
            return HashEmbedding(self.dim).embed_batch(texts)
        
        return self.model.encode(texts, normalize_embeddings=True)


class SemanticIndex:
    """Index for efficient similarity search."""
    
    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self._embeddings: list[np.ndarray] = []
        self._keys: list[str] = []
        self._faiss_index = None
        self._lock = threading.Lock()
        
        self._init_faiss()
    
    def _init_faiss(self) -> None:
        """Initialize FAISS index if available."""
        if not self.config.use_faiss:
            return
        
        try:
            import faiss
            
            dim = self.config.embedding_dim
            
            if self.config.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dim)
                self._faiss_index = faiss.IndexIVFFlat(
                    quantizer, dim, self.config.nlist,
                    faiss.METRIC_INNER_PRODUCT
                )
            elif self.config.index_type == "hnsw":
                self._faiss_index = faiss.IndexHNSWFlat(
                    dim, 32, faiss.METRIC_INNER_PRODUCT
                )
            else:
                self._faiss_index = faiss.IndexFlatIP(dim)
            
            logger.info(f"Initialized FAISS index: {self.config.index_type}")
        
        except ImportError:
            logger.warning("FAISS not available, using brute-force search")
            self._faiss_index = None
    
    def add(self, key: str, embedding: np.ndarray) -> None:
        """Add embedding to index."""
        with self._lock:
            if self.config.normalize_embeddings:
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            self._embeddings.append(embedding)
            self._keys.append(key)
            
            if self._faiss_index is not None:
                pass

                # Check if index needs training
                if hasattr(self._faiss_index, 'is_trained') and not self._faiss_index.is_trained:
                    if len(self._embeddings) >= self.config.nlist:
                        embeddings_array = np.array(self._embeddings).astype(np.float32)
                        self._faiss_index.train(embeddings_array)
                        self._faiss_index.add(embeddings_array)
                elif self._faiss_index.is_trained if hasattr(self._faiss_index, 'is_trained') else True:
                    self._faiss_index.add(embedding.reshape(1, -1).astype(np.float32))
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query vector
            k: Number of results
        
        Returns:
            List of (key, similarity) tuples
        """
        if not self._embeddings:
            return []
        
        if self.config.normalize_embeddings:
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        with self._lock:
            if self._faiss_index is not None and self._faiss_index.ntotal > 0:
                pass
                
                if hasattr(self._faiss_index, 'nprobe'):
                    self._faiss_index.nprobe = self.config.nprobe
                
                query = query_embedding.reshape(1, -1).astype(np.float32)
                scores, indices = self._faiss_index.search(query, min(k, len(self._keys)))
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and idx < len(self._keys):
                        results.append((self._keys[idx], float(score)))
                return results
            
            else:
                # Brute-force search
                embeddings_array = np.array(self._embeddings)
                similarities = np.dot(embeddings_array, query_embedding)
                
                top_k = np.argsort(similarities)[-k:][::-1]
                return [(self._keys[i], float(similarities[i])) for i in top_k]
    
    def remove(self, key: str) -> None:
        """Remove embedding from index."""
        with self._lock:
            if key in self._keys:
                idx = self._keys.index(key)
                self._keys.pop(idx)
                self._embeddings.pop(idx)
                
                # Rebuild FAISS index (no direct removal support)
                if self._faiss_index is not None and self._embeddings:
                    self._rebuild_faiss()
    
    def _rebuild_faiss(self) -> None:
        """Rebuild FAISS index from scratch."""
        if not self._embeddings or self._faiss_index is None:
            return
        
        
        self._faiss_index.reset()
        
        embeddings_array = np.array(self._embeddings).astype(np.float32)
        
        if hasattr(self._faiss_index, 'train'):
            self._faiss_index.train(embeddings_array)
        
        self._faiss_index.add(embeddings_array)
    
    def size(self) -> int:
        """Get number of indexed items."""
        return len(self._keys)


class SemanticCache:
    """
    Cache responses based on semantic similarity.
    
    Caches LLM responses and returns cached results for
    semantically similar queries, reducing API calls and latency.
    """
    
    def __init__(
        self,
        config: CacheConfig = None,
        embedding_provider: EmbeddingProvider = None
    ) -> None:
        self.config = config or CacheConfig()
        
        # Embedding provider
        self.embedding_provider = embedding_provider or SentenceTransformerEmbedding(
            dim=self.config.embedding_dim
        )
        
        # Storage
        self._cache: dict[str, CachedItem] = {}
        self._index = SemanticIndex(self.config)
        self._lock = threading.Lock()
        
        # Stats
        self._stats = {
            "hits": 0,
            "misses": 0,
            "semantic_hits": 0,
            "exact_hits": 0,
            "evictions": 0
        }
        
        # Background cleanup
        self._running = False
        self._cleanup_thread: Optional[threading.Thread] = None
    
    def get(self, query: str) -> tuple[Optional[Any], float]:
        """
        Get cached response for query.
        
        Args:
            query: Query string
        
        Returns:
            (response, similarity) or (None, 0.0) if not found
        """
        # Check exact match first
        key = self._make_key(query)
        
        with self._lock:
            if key in self._cache:
                item = self._cache[key]
                if not self._is_expired(item):
                    item.touch()
                    self._stats["hits"] += 1
                    self._stats["exact_hits"] += 1
                    return item.response, 1.0
        
        # Semantic search
        embedding = self.embedding_provider.embed(query)
        results = self._index.search(embedding, k=1)
        
        if results:
            best_key, similarity = results[0]
            
            if similarity >= self.config.similarity_threshold:
                with self._lock:
                    if best_key in self._cache:
                        item = self._cache[best_key]
                        if not self._is_expired(item):
                            item.touch()
                            self._stats["hits"] += 1
                            self._stats["semantic_hits"] += 1
                            return item.response, similarity
        
        self._stats["misses"] += 1
        return None, 0.0
    
    def set(
        self,
        query: str,
        response: Any,
        metadata: dict[str, Any] = None
    ) -> None:
        """
        Cache a query-response pair.
        
        Args:
            query: Query string
            response: Response to cache
            metadata: Optional metadata
        """
        # Ensure space
        self._ensure_capacity()
        
        # Create embedding
        embedding = self.embedding_provider.embed(query)
        key = self._make_key(query)
        
        item = CachedItem(
            key=key,
            query=query,
            response=response,
            embedding=embedding,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._cache[key] = item
            self._index.add(key, embedding)
    
    def invalidate(self, query: str) -> None:
        """Invalidate a cached query."""
        key = self._make_key(query)
        
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._index.remove(key)
    
    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
            self._index = SemanticIndex(self.config)
    
    def _make_key(self, query: str) -> str:
        """Create cache key from query."""
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()
    
    def _is_expired(self, item: CachedItem) -> bool:
        """Check if item is expired."""
        if self.config.ttl_seconds is None:
            return False
        return time.time() - item.created_at > self.config.ttl_seconds
    
    def _ensure_capacity(self) -> None:
        """Ensure cache has capacity for new items."""
        with self._lock:
            while len(self._cache) >= self.config.max_items:
                self._evict_one()
    
    def _evict_one(self) -> None:
        """Evict one item (LRU)."""
        if not self._cache:
            return
        
        # Find least recently accessed
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].accessed_at
        )
        
        del self._cache[oldest_key]
        self._index.remove(oldest_key)
        self._stats["evictions"] += 1
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        
        return {
            **self._stats,
            "size": len(self._cache),
            "hit_rate": hit_rate,
            "semantic_hit_rate": self._stats["semantic_hits"] / self._stats["hits"]
            if self._stats["hits"] > 0 else 0.0
        }
    
    def start_cleanup(self) -> None:
        """Start background cleanup thread."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def stop_cleanup(self) -> None:
        """Stop background cleanup."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2.0)
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._running:
            time.sleep(self.config.eviction_check_interval)
            self._cleanup_expired()
    
    def _cleanup_expired(self) -> None:
        """Remove expired items."""
        with self._lock:
            expired = [
                key for key, item in self._cache.items()
                if self._is_expired(item)
            ]
            for key in expired:
                del self._cache[key]
                self._index.remove(key)
    
    def save(self, path: Path = None) -> None:
        """Save cache to disk."""
        path = path or self.config.persist_path
        if not path:
            return
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "items": [
                {
                    "key": item.key,
                    "query": item.query,
                    "response": item.response,
                    "embedding": item.embedding.tolist(),
                    "created_at": item.created_at,
                    "metadata": item.metadata
                }
                for item in self._cache.values()
            ],
            "stats": self._stats
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: Path = None) -> None:
        """Load cache from disk."""
        path = path or self.config.persist_path
        if not path or not path.exists():
            return
        
        with open(path) as f:
            data = json.load(f)
        
        for item_data in data.get("items", []):
            item = CachedItem(
                key=item_data["key"],
                query=item_data["query"],
                response=item_data["response"],
                embedding=np.array(item_data["embedding"]),
                created_at=item_data["created_at"],
                metadata=item_data.get("metadata", {})
            )
            
            self._cache[item.key] = item
            self._index.add(item.key, item.embedding)
        
        self._stats = data.get("stats", self._stats)


class CachedLLM:
    """Wrapper that adds semantic caching to an LLM."""
    
    def __init__(
        self,
        llm: Callable[[str], str],
        cache: SemanticCache = None
    ) -> None:
        self.llm = llm
        self.cache = cache or SemanticCache()
    
    def __call__(self, query: str, **kwargs) -> str:
        """Execute query with caching."""
        # Try cache
        cached, similarity = self.cache.get(query)
        if cached is not None:
            return cached
        
        # Call LLM
        response = self.llm(query, **kwargs)
        
        # Cache result
        self.cache.set(query, response)
        
        return response


# Global instance
_cache: Optional[SemanticCache] = None


def get_semantic_cache(config: CacheConfig = None) -> SemanticCache:
    """Get or create global semantic cache."""
    global _cache
    if _cache is None:
        _cache = SemanticCache(config)
    return _cache
