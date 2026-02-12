"""
Token Embedding Explorer

Interactive exploration and analysis of token embeddings
including clustering, similarity search, and semantic analysis.

FILE: enigma_engine/core/token_explorer.py
TYPE: Core/Analysis
MAIN CLASSES: TokenExplorer, EmbeddingClusterer, SemanticAnalyzer
"""

import logging
from dataclasses import dataclass, field
from typing import Any

try:
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TokenInfo:
    """Information about a token."""
    index: int
    token: str
    embedding_norm: float
    cluster_id: int = -1
    frequency: int = 0
    similar_tokens: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class ClusterInfo:
    """Information about an embedding cluster."""
    cluster_id: int
    size: int
    center: Any  # np.ndarray
    tokens: list[int]
    representative_token: int
    coherence: float  # Avg similarity within cluster


if HAS_NUMPY:
    
    class TokenExplorer:
        """
        Explore and analyze token embeddings.
        """
        
        def __init__(
            self,
            embeddings: np.ndarray,
            vocab: dict[str, int] = None,
            tokenizer: Any = None
        ):
            """
            Initialize token explorer.
            
            Args:
                embeddings: Token embeddings [vocab_size, embed_dim]
                vocab: Token to index mapping
                tokenizer: Optional tokenizer for decoding
            """
            self.embeddings = embeddings
            self.vocab = vocab or {}
            self.idx_to_token = {v: k for k, v in self.vocab.items()}
            self.tokenizer = tokenizer
            
            # Precompute norms for efficiency
            self._norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self._normalized = embeddings / (self._norms + 1e-8)
        
        @classmethod
        def from_model(
            cls,
            model: Any,
            tokenizer: Any = None
        ) -> "TokenExplorer":
            """Create from model with embedding layer."""
            if HAS_TORCH and hasattr(model, 'named_parameters'):
                for name, param in model.named_parameters():
                    if 'embed' in name.lower() and 'weight' in name.lower():
                        embeddings = param.detach().cpu().numpy()
                        vocab = {}
                        if tokenizer and hasattr(tokenizer, 'get_vocab'):
                            vocab = tokenizer.get_vocab()
                        elif tokenizer and hasattr(tokenizer, 'vocab'):
                            vocab = tokenizer.vocab
                        return cls(embeddings, vocab, tokenizer)
            
            raise ValueError("Could not find embedding layer in model")
        
        def get_token(self, idx: int) -> str:
            """Get token string from index."""
            if self.tokenizer and hasattr(self.tokenizer, 'decode'):
                try:
                    return self.tokenizer.decode([idx])
                except (KeyError, ValueError, RuntimeError):
                    pass  # Intentionally silent
            return self.idx_to_token.get(idx, f"<{idx}>")
        
        def get_index(self, token: str) -> int:
            """Get index from token string."""
            if self.tokenizer and hasattr(self.tokenizer, 'encode'):
                try:
                    indices = self.tokenizer.encode(token)
                    if indices:
                        return indices[0]
                except (ValueError, RuntimeError):
                    pass  # Intentionally silent
            return self.vocab.get(token, -1)
        
        def find_similar(
            self,
            query: Any,
            top_k: int = 10,
            exclude_query: bool = True
        ) -> list[tuple[int, str, float]]:
            """
            Find tokens most similar to query.
            
            Args:
                query: Token index, token string, or embedding vector
                top_k: Number of results
                exclude_query: Exclude query from results
            
            Returns:
                List of (index, token, similarity) tuples
            """
            # Get query embedding
            if isinstance(query, int):
                query_idx = query
                query_emb = self._normalized[query]
            elif isinstance(query, str):
                query_idx = self.get_index(query)
                if query_idx < 0:
                    raise ValueError(f"Token '{query}' not found")
                query_emb = self._normalized[query_idx]
            else:
                query_idx = -1
                query_emb = np.array(query) / (np.linalg.norm(query) + 1e-8)
            
            # Compute similarities
            similarities = self._normalized @ query_emb
            
            # Get top-k
            indices = np.argsort(similarities)[::-1]
            
            results = []
            for idx in indices:
                if exclude_query and idx == query_idx:
                    continue
                results.append((
                    int(idx),
                    self.get_token(idx),
                    float(similarities[idx])
                ))
                if len(results) >= top_k:
                    break
            
            return results
        
        def find_by_analogy(
            self,
            a: Any,
            b: Any,
            c: Any,
            top_k: int = 5
        ) -> list[tuple[int, str, float]]:
            """
            Find token d such that a:b :: c:d
            
            Args:
                a, b, c: Token indices, strings, or embeddings
                top_k: Number of results
            
            Returns:
                List of (index, token, score) tuples
            """
            def get_emb(x):
                if isinstance(x, int):
                    return self.embeddings[x]
                elif isinstance(x, str):
                    idx = self.get_index(x)
                    return self.embeddings[idx]
                return np.array(x)
            
            emb_a = get_emb(a)
            emb_b = get_emb(b)
            emb_c = get_emb(c)
            
            # d = b - a + c
            target = emb_b - emb_a + emb_c
            
            return self.find_similar(target, top_k)
        
        def get_embedding(self, query: Any) -> np.ndarray:
            """Get embedding for a token."""
            if isinstance(query, int):
                return self.embeddings[query].copy()
            elif isinstance(query, str):
                idx = self.get_index(query)
                if idx >= 0:
                    return self.embeddings[idx].copy()
            raise ValueError(f"Cannot get embedding for {query}")
        
        def analyze_token(self, query: Any, num_similar: int = 5) -> TokenInfo:
            """
            Get comprehensive information about a token.
            
            Args:
                query: Token index or string
                num_similar: Number of similar tokens to find
            
            Returns:
                TokenInfo with analysis
            """
            if isinstance(query, str):
                idx = self.get_index(query)
                token = query
            else:
                idx = query
                token = self.get_token(idx)
            
            emb_norm = float(self._norms[idx][0])
            similar = self.find_similar(idx, num_similar)
            
            return TokenInfo(
                index=idx,
                token=token,
                embedding_norm=emb_norm,
                similar_tokens=[(t, s) for _, t, s in similar]
            )
        
        def measure_distance(
            self,
            token1: Any,
            token2: Any,
            metric: str = "cosine"
        ) -> float:
            """
            Measure distance between two tokens.
            
            Args:
                token1, token2: Token indices or strings
                metric: 'cosine', 'euclidean', or 'manhattan'
            
            Returns:
                Distance value
            """
            def get_emb(x):
                if isinstance(x, int):
                    return self.embeddings[x]
                idx = self.get_index(x)
                return self.embeddings[idx]
            
            e1 = get_emb(token1)
            e2 = get_emb(token2)
            
            if metric == "cosine":
                return 1 - float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
            elif metric == "euclidean":
                return float(np.linalg.norm(e1 - e2))
            elif metric == "manhattan":
                return float(np.sum(np.abs(e1 - e2)))
            else:
                raise ValueError(f"Unknown metric: {metric}")
    
    
    class EmbeddingClusterer:
        """
        Cluster token embeddings for analysis.
        """
        
        def __init__(self, explorer: TokenExplorer):
            self.explorer = explorer
            self.labels: np.ndarray = None
            self.centers: np.ndarray = None
            self.n_clusters: int = 0
        
        def cluster_kmeans(
            self,
            n_clusters: int = 100,
            max_iter: int = 100,
            random_state: int = 42
        ) -> np.ndarray:
            """
            Cluster embeddings using K-means.
            
            Args:
                n_clusters: Number of clusters
                max_iter: Maximum iterations
                random_state: Random seed
            
            Returns:
                Cluster labels for each token
            """
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    max_iter=max_iter,
                    random_state=random_state,
                    n_init=3
                )
                self.labels = kmeans.fit_predict(self.explorer.embeddings)
                self.centers = kmeans.cluster_centers_
            except ImportError:
                # Simple K-means fallback
                self.labels, self.centers = self._simple_kmeans(
                    n_clusters, max_iter, random_state
                )
            
            self.n_clusters = n_clusters
            return self.labels
        
        def _simple_kmeans(
            self,
            n_clusters: int,
            max_iter: int,
            random_state: int
        ) -> tuple[np.ndarray, np.ndarray]:
            """Simple K-means implementation."""
            np.random.seed(random_state)
            
            # Initialize centers randomly
            indices = np.random.choice(
                len(self.explorer.embeddings),
                n_clusters,
                replace=False
            )
            centers = self.explorer.embeddings[indices].copy()
            
            for _ in range(max_iter):
                # Assign labels
                distances = np.zeros((len(self.explorer.embeddings), n_clusters))
                for i, center in enumerate(centers):
                    distances[:, i] = np.linalg.norm(
                        self.explorer.embeddings - center, axis=1
                    )
                labels = np.argmin(distances, axis=1)
                
                # Update centers
                new_centers = np.zeros_like(centers)
                for i in range(n_clusters):
                    mask = labels == i
                    if mask.any():
                        new_centers[i] = self.explorer.embeddings[mask].mean(axis=0)
                    else:
                        new_centers[i] = centers[i]
                
                # Check convergence
                if np.allclose(centers, new_centers):
                    break
                centers = new_centers
            
            return labels, centers
        
        def get_cluster_info(self, cluster_id: int) -> ClusterInfo:
            """
            Get detailed info about a cluster.
            
            Args:
                cluster_id: Cluster index
            
            Returns:
                ClusterInfo with cluster details
            """
            if self.labels is None:
                raise ValueError("Run cluster_kmeans first")
            
            mask = self.labels == cluster_id
            tokens = np.where(mask)[0].tolist()
            
            if not tokens:
                return ClusterInfo(
                    cluster_id=cluster_id,
                    size=0,
                    center=self.centers[cluster_id],
                    tokens=[],
                    representative_token=-1,
                    coherence=0.0
                )
            
            # Find representative (closest to center)
            cluster_embeddings = self.explorer.embeddings[mask]
            center = self.centers[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            rep_idx = tokens[np.argmin(distances)]
            
            # Compute coherence (avg pairwise similarity)
            if len(tokens) > 1:
                normalized = cluster_embeddings / (
                    np.linalg.norm(cluster_embeddings, axis=1, keepdims=True) + 1e-8
                )
                similarities = normalized @ normalized.T
                np.fill_diagonal(similarities, 0)
                coherence = similarities.sum() / (len(tokens) * (len(tokens) - 1))
            else:
                coherence = 1.0
            
            return ClusterInfo(
                cluster_id=cluster_id,
                size=len(tokens),
                center=center,
                tokens=tokens,
                representative_token=rep_idx,
                coherence=float(coherence)
            )
        
        def get_tokens_in_cluster(
            self,
            cluster_id: int
        ) -> list[tuple[int, str]]:
            """Get all tokens in a cluster."""
            if self.labels is None:
                raise ValueError("Run cluster_kmeans first")
            
            mask = self.labels == cluster_id
            indices = np.where(mask)[0]
            return [
                (int(idx), self.explorer.get_token(idx))
                for idx in indices
            ]
        
        def find_cluster(self, query: Any) -> int:
            """Find which cluster a token belongs to."""
            if self.labels is None:
                raise ValueError("Run cluster_kmeans first")
            
            if isinstance(query, str):
                idx = self.explorer.get_index(query)
            else:
                idx = query
            
            return int(self.labels[idx])
        
        def get_cluster_stats(self) -> dict[str, Any]:
            """Get overall clustering statistics."""
            if self.labels is None:
                raise ValueError("Run cluster_kmeans first")
            
            unique, counts = np.unique(self.labels, return_counts=True)
            
            return {
                "n_clusters": self.n_clusters,
                "cluster_sizes": dict(zip(unique.tolist(), counts.tolist())),
                "avg_cluster_size": float(np.mean(counts)),
                "min_cluster_size": int(np.min(counts)),
                "max_cluster_size": int(np.max(counts)),
                "std_cluster_size": float(np.std(counts))
            }
    
    
    class SemanticAnalyzer:
        """
        Analyze semantic relationships in embeddings.
        """
        
        def __init__(self, explorer: TokenExplorer):
            self.explorer = explorer
        
        def find_semantic_dimensions(
            self,
            pairs: list[tuple[Any, Any]],
            n_components: int = 10
        ) -> np.ndarray:
            """
            Find semantic dimensions from word pairs.
            
            For example, gender direction from (man, woman), (king, queen)
            
            Args:
                pairs: List of (word1, word2) tuples
                n_components: Number of components to extract
            
            Returns:
                Principal directions [n_components, embed_dim]
            """
            differences = []
            
            for w1, w2 in pairs:
                e1 = self.explorer.get_embedding(w1)
                e2 = self.explorer.get_embedding(w2)
                differences.append(e2 - e1)
            
            differences = np.array(differences)
            
            # PCA on differences
            centered = differences - np.mean(differences, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Sort by eigenvalue (descending)
            idx = eigenvalues.argsort()[::-1]
            
            return eigenvectors[:, idx[:n_components]].T
        
        def project_on_direction(
            self,
            tokens: list[Any],
            direction: np.ndarray
        ) -> list[tuple[str, float]]:
            """
            Project tokens onto a semantic direction.
            
            Args:
                tokens: Token indices or strings
                direction: Direction vector
            
            Returns:
                List of (token, projection) tuples sorted by projection
            """
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            results = []
            for token in tokens:
                emb = self.explorer.get_embedding(token)
                proj = float(np.dot(emb, direction))
                token_str = token if isinstance(token, str) else self.explorer.get_token(token)
                results.append((token_str, proj))
            
            results.sort(key=lambda x: x[1])
            return results
        
        def find_outliers(
            self,
            threshold: float = 2.0
        ) -> list[tuple[int, str, float]]:
            """
            Find tokens with unusual embeddings.
            
            Args:
                threshold: Number of std deviations for outlier
            
            Returns:
                List of (index, token, z_score) tuples
            """
            norms = self.explorer._norms.flatten()
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            
            outliers = []
            for idx, norm in enumerate(norms):
                z_score = (norm - mean_norm) / (std_norm + 1e-8)
                if abs(z_score) > threshold:
                    outliers.append((
                        int(idx),
                        self.explorer.get_token(idx),
                        float(z_score)
                    ))
            
            outliers.sort(key=lambda x: abs(x[2]), reverse=True)
            return outliers


    def create_explorer(
        model: Any = None,
        embeddings: np.ndarray = None,
        vocab: dict[str, int] = None,
        tokenizer: Any = None
    ) -> TokenExplorer:
        """
        Create a TokenExplorer from various sources.
        
        Args:
            model: Model with embedding layer
            embeddings: Direct embedding matrix
            vocab: Token to index mapping
            tokenizer: Tokenizer for token lookup
        
        Returns:
            TokenExplorer instance
        """
        if embeddings is not None:
            return TokenExplorer(embeddings, vocab, tokenizer)
        elif model is not None:
            return TokenExplorer.from_model(model, tokenizer)
        else:
            raise ValueError("Provide either model or embeddings")

else:
    class TokenExplorer:
        pass
    
    class EmbeddingClusterer:
        pass
    
    class SemanticAnalyzer:
        pass
    
    def create_explorer(*args, **kwargs):
        raise ImportError("NumPy required for token exploration")
