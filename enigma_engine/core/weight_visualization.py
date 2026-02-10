"""
Weight Visualization

Tools for visualizing neural network weights, biases,
and learned representations.

FILE: enigma_engine/core/weight_visualization.py
TYPE: Core/Analysis
MAIN CLASSES: WeightVisualizer, LayerVisualizer, EmbeddingVisualizer
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

try:
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualizationType(Enum):
    """Types of visualizations."""
    HEATMAP = auto()
    HISTOGRAM = auto()
    LINE = auto()
    SCATTER = auto()
    PCA = auto()
    TSNE = auto()


@dataclass
class WeightStats:
    """Statistics about weight tensor."""
    layer_name: str
    shape: tuple[int, ...]
    num_params: int
    mean: float
    std: float
    min_val: float
    max_val: float
    sparsity: float
    norm_l1: float
    norm_l2: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "shape": list(self.shape),
            "num_params": self.num_params,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "sparsity": self.sparsity,
            "norm_l1": self.norm_l1,
            "norm_l2": self.norm_l2
        }


if HAS_TORCH and HAS_NUMPY:
    
    class WeightVisualizer:
        """
        Visualize neural network weights and parameters.
        """
        
        def __init__(self, model: nn.Module):
            self.model = model
            self._weight_cache: dict[str, np.ndarray] = {}
        
        def get_all_weights(self) -> dict[str, np.ndarray]:
            """Extract all weights from model."""
            weights = {}
            
            for name, param in self.model.named_parameters():
                weights[name] = param.detach().cpu().numpy()
            
            self._weight_cache = weights
            return weights
        
        def compute_stats(self) -> list[WeightStats]:
            """Compute statistics for all weights."""
            if not self._weight_cache:
                self.get_all_weights()
            
            stats = []
            for name, weights in self._weight_cache.items():
                flat = weights.flatten()
                
                stats.append(WeightStats(
                    layer_name=name,
                    shape=weights.shape,
                    num_params=weights.size,
                    mean=float(np.mean(flat)),
                    std=float(np.std(flat)),
                    min_val=float(np.min(flat)),
                    max_val=float(np.max(flat)),
                    sparsity=float(np.mean(np.abs(flat) < 1e-6) * 100),
                    norm_l1=float(np.sum(np.abs(flat))),
                    norm_l2=float(np.sqrt(np.sum(flat ** 2)))
                ))
            
            return stats
        
        def plot_weight_distribution(
            self,
            layer_name: str = None,
            save_path: str = None,
            bins: int = 100
        ) -> Optional[Any]:
            """
            Plot weight distribution histogram.
            
            Args:
                layer_name: Specific layer (None = all)
                save_path: Path to save figure
                bins: Number of histogram bins
            
            Returns:
                Figure if matplotlib available
            """
            if not HAS_MATPLOTLIB:
                logger.warning("Matplotlib required for plotting")
                return None
            
            if not self._weight_cache:
                self.get_all_weights()
            
            if layer_name:
                weights = [self._weight_cache.get(layer_name, np.array([]))]
                names = [layer_name]
            else:
                weights = list(self._weight_cache.values())
                names = list(self._weight_cache.keys())
            
            # Combine all weights for histogram
            all_weights = np.concatenate([w.flatten() for w in weights if w.size > 0])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(all_weights, bins=bins, density=True, alpha=0.7)
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Density")
            ax.set_title(f"Weight Distribution ({len(names)} layers)")
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            
            # Add statistics
            stats_text = f"Mean: {np.mean(all_weights):.4f}\nStd: {np.std(all_weights):.4f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved weight distribution to {save_path}")
            
            return fig
        
        def plot_weight_heatmap(
            self,
            layer_name: str,
            save_path: str = None,
            max_size: int = 256
        ) -> Optional[Any]:
            """
            Plot weight matrix as heatmap.
            
            Args:
                layer_name: Layer to visualize
                save_path: Path to save figure
                max_size: Maximum dimension for visualization
            
            Returns:
                Figure if matplotlib available
            """
            if not HAS_MATPLOTLIB:
                logger.warning("Matplotlib required for plotting")
                return None
            
            if not self._weight_cache:
                self.get_all_weights()
            
            weights = self._weight_cache.get(layer_name)
            if weights is None:
                logger.error(f"Layer {layer_name} not found")
                return None
            
            # Reshape to 2D if needed
            if len(weights.shape) == 1:
                size = int(np.ceil(np.sqrt(weights.size)))
                weights = np.pad(weights, (0, size**2 - weights.size))
                weights = weights.reshape(size, size)
            elif len(weights.shape) > 2:
                weights = weights.reshape(weights.shape[0], -1)
            
            # Downsample if too large
            if weights.shape[0] > max_size or weights.shape[1] > max_size:
                factor_h = max(1, weights.shape[0] // max_size)
                factor_w = max(1, weights.shape[1] // max_size)
                weights = weights[::factor_h, ::factor_w]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(weights, aspect='auto', cmap='RdBu_r',
                          vmin=-np.abs(weights).max(),
                          vmax=np.abs(weights).max())
            ax.set_title(f"Weight Matrix: {layer_name}")
            ax.set_xlabel("Input Dimension")
            ax.set_ylabel("Output Dimension")
            plt.colorbar(im, ax=ax, label="Weight Value")
            
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved weight heatmap to {save_path}")
            
            return fig
        
        def plot_layer_norms(
            self,
            save_path: str = None
        ) -> Optional[Any]:
            """
            Plot L2 norms across layers.
            
            Args:
                save_path: Path to save figure
            
            Returns:
                Figure if matplotlib available
            """
            if not HAS_MATPLOTLIB:
                return None
            
            stats = self.compute_stats()
            
            # Sort by layer depth (assume numbered naming)
            names = [s.layer_name for s in stats]
            norms = [s.norm_l2 for s in stats]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(range(len(norms)), norms)
            ax.set_xlabel("Layer Index")
            ax.set_ylabel("L2 Norm")
            ax.set_title("Weight Norms by Layer")
            
            # Rotate labels if many layers
            if len(names) > 10:
                ax.set_xticks(range(0, len(names), max(1, len(names)//10)))
            
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
            return fig
        
        def find_dead_neurons(
            self,
            threshold: float = 1e-6
        ) -> dict[str, list[int]]:
            """
            Find neurons with near-zero weights.
            
            Args:
                threshold: Weight magnitude threshold
            
            Returns:
                Dict mapping layer name to dead neuron indices
            """
            if not self._weight_cache:
                self.get_all_weights()
            
            dead = {}
            
            for name, weights in self._weight_cache.items():
                if len(weights.shape) < 2:
                    continue
                
                # Check output neurons (rows)
                neuron_norms = np.linalg.norm(weights, axis=1)
                dead_indices = np.where(neuron_norms < threshold)[0].tolist()
                
                if dead_indices:
                    dead[name] = dead_indices
            
            return dead
        
        def compare_weights(
            self,
            other_model: nn.Module
        ) -> dict[str, dict[str, float]]:
            """
            Compare weights between two models.
            
            Args:
                other_model: Model to compare against
            
            Returns:
                Per-layer comparison metrics
            """
            if not self._weight_cache:
                self.get_all_weights()
            
            other_weights = {}
            for name, param in other_model.named_parameters():
                other_weights[name] = param.detach().cpu().numpy()
            
            comparison = {}
            
            for name in set(self._weight_cache.keys()) & set(other_weights.keys()):
                w1 = self._weight_cache[name].flatten()
                w2 = other_weights[name].flatten()
                
                if w1.shape != w2.shape:
                    continue
                
                comparison[name] = {
                    "mse": float(np.mean((w1 - w2) ** 2)),
                    "mae": float(np.mean(np.abs(w1 - w2))),
                    "cosine_similarity": float(
                        np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-8)
                    ),
                    "norm_diff": float(np.linalg.norm(w1) - np.linalg.norm(w2))
                }
            
            return comparison
    
    
    class EmbeddingVisualizer:
        """
        Visualize embedding layers and token representations.
        """
        
        def __init__(
            self,
            embeddings: np.ndarray = None,
            vocab: dict[str, int] = None
        ):
            self.embeddings = embeddings
            self.vocab = vocab or {}
            self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
        @classmethod
        def from_model(
            cls,
            model: nn.Module,
            embedding_layer_name: str = "embedding"
        ) -> "EmbeddingVisualizer":
            """
            Create visualizer from model.
            
            Args:
                model: Model with embedding layer
                embedding_layer_name: Name containing 'embedding'
            
            Returns:
                EmbeddingVisualizer instance
            """
            for name, param in model.named_parameters():
                if embedding_layer_name.lower() in name.lower():
                    embeddings = param.detach().cpu().numpy()
                    return cls(embeddings=embeddings)
            
            raise ValueError(f"Embedding layer '{embedding_layer_name}' not found")
        
        def reduce_dimensions(
            self,
            method: str = "pca",
            n_components: int = 2
        ) -> np.ndarray:
            """
            Reduce embedding dimensions for visualization.
            
            Args:
                method: 'pca' or 'tsne'
                n_components: Target dimensions (2 or 3)
            
            Returns:
                Reduced embeddings
            """
            if method == "pca":
                try:
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=n_components)
                    return reducer.fit_transform(self.embeddings)
                except ImportError:
                    # Simple PCA fallback
                    centered = self.embeddings - np.mean(self.embeddings, axis=0)
                    cov = np.cov(centered.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    idx = eigenvalues.argsort()[::-1]
                    eigenvectors = eigenvectors[:, idx]
                    return centered @ eigenvectors[:, :n_components]
            
            elif method == "tsne":
                try:
                    from sklearn.manifold import TSNE
                    reducer = TSNE(n_components=n_components, random_state=42)
                    return reducer.fit_transform(self.embeddings)
                except ImportError:
                    logger.warning("sklearn required for t-SNE, using PCA")
                    return self.reduce_dimensions("pca", n_components)
            
            else:
                raise ValueError(f"Unknown method: {method}")
        
        def plot_embeddings(
            self,
            method: str = "pca",
            highlight_tokens: list[int] = None,
            save_path: str = None
        ) -> Optional[Any]:
            """
            Plot 2D embedding visualization.
            
            Args:
                method: Reduction method ('pca' or 'tsne')
                highlight_tokens: Token indices to label
                save_path: Path to save figure
            
            Returns:
                Figure if matplotlib available
            """
            if not HAS_MATPLOTLIB:
                return None
            
            reduced = self.reduce_dimensions(method, 2)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.3, s=10)
            
            # Highlight specific tokens
            if highlight_tokens:
                for idx in highlight_tokens:
                    if idx < len(reduced):
                        ax.scatter(reduced[idx, 0], reduced[idx, 1],
                                  color='red', s=100, zorder=5)
                        token = self.idx_to_token.get(idx, str(idx))
                        ax.annotate(token, (reduced[idx, 0], reduced[idx, 1]),
                                   fontsize=8)
            
            ax.set_xlabel(f"{method.upper()} 1")
            ax.set_ylabel(f"{method.upper()} 2")
            ax.set_title(f"Token Embeddings ({method.upper()})")
            
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
            return fig
        
        def find_similar_tokens(
            self,
            token_idx: int,
            top_k: int = 10
        ) -> list[tuple[int, float]]:
            """
            Find tokens with similar embeddings.
            
            Args:
                token_idx: Query token index
                top_k: Number of similar tokens
            
            Returns:
                List of (token_idx, similarity) tuples
            """
            query = self.embeddings[token_idx]
            query_norm = np.linalg.norm(query)
            
            similarities = []
            for i, emb in enumerate(self.embeddings):
                if i == token_idx:
                    continue
                sim = np.dot(query, emb) / (query_norm * np.linalg.norm(emb) + 1e-8)
                similarities.append((i, float(sim)))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        
        def compute_analogy(
            self,
            a: int,
            b: int,
            c: int,
            top_k: int = 5
        ) -> list[tuple[int, float]]:
            """
            Compute word analogy: a is to b as c is to ?
            
            Args:
                a, b, c: Token indices
                top_k: Number of results
            
            Returns:
                List of (token_idx, score) tuples
            """
            # d = b - a + c
            target = self.embeddings[b] - self.embeddings[a] + self.embeddings[c]
            target_norm = np.linalg.norm(target)
            
            scores = []
            for i, emb in enumerate(self.embeddings):
                if i in [a, b, c]:
                    continue
                score = np.dot(target, emb) / (target_norm * np.linalg.norm(emb) + 1e-8)
                scores.append((i, float(score)))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]


    def generate_model_report(
        model: nn.Module,
        output_dir: str,
        sample_layers: int = 5
    ) -> dict[str, Any]:
        """
        Generate comprehensive weight visualization report.
        
        Args:
            model: Model to analyze
            output_dir: Directory for output files
            sample_layers: Number of layer heatmaps to generate
        
        Returns:
            Report metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        visualizer = WeightVisualizer(model)
        stats = visualizer.compute_stats()
        
        report = {
            "weight_stats": [s.to_dict() for s in stats],
            "dead_neurons": visualizer.find_dead_neurons(),
            "figures": []
        }
        
        # Generate plots
        if HAS_MATPLOTLIB:
            dist_path = output_path / "weight_distribution.png"
            visualizer.plot_weight_distribution(save_path=str(dist_path))
            report["figures"].append(str(dist_path))
            
            norm_path = output_path / "layer_norms.png"
            visualizer.plot_layer_norms(save_path=str(norm_path))
            report["figures"].append(str(norm_path))
            
            # Sample layer heatmaps
            layer_names = list(visualizer._weight_cache.keys())
            for i, name in enumerate(layer_names[:sample_layers]):
                heatmap_path = output_path / f"heatmap_{i}.png"
                visualizer.plot_weight_heatmap(name, save_path=str(heatmap_path))
                report["figures"].append(str(heatmap_path))
        
        # Save report JSON
        with open(output_path / "report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

else:
    class WeightVisualizer:
        pass
    
    class EmbeddingVisualizer:
        pass
    
    def generate_model_report(*args, **kwargs):
        raise ImportError("PyTorch and NumPy required for weight visualization")
