"""
Live Training Visualization for Enigma AI Engine

Real-time visualization of training metrics.

Features:
- Live loss curves
- Learning rate schedules
- Gradient flow visualization
- Attention heatmaps
- Token embedding projections
- GPU memory usage
- Training speed metrics

Usage:
    from enigma_engine.gui.training_viz import TrainingVisualizer
    
    viz = TrainingVisualizer()
    viz.show()
    
    # During training, call:
    viz.update_loss(step, train_loss, val_loss)
    viz.update_gradients(model)
    viz.show_attention(attention_weights, tokens)
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
        QLabel, QFrame, QSplitter, QScrollArea, QGridLayout,
        QPushButton, QSpinBox, QComboBox
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QBrush
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int = 0
    train_loss: float = 0.0
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    tokens_per_second: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    epoch: int = 0
    timestamp: float = field(default_factory=time.time)


class LossPlotWidget(QWidget if PYQT5_AVAILABLE else object):
    """Widget for plotting loss curves."""
    
    def __init__(self, parent=None, max_points: int = 1000):
        if PYQT5_AVAILABLE:
            super().__init__(parent)
        
        self._max_points = max_points
        self._train_losses: deque = deque(maxlen=max_points)
        self._val_losses: deque = deque(maxlen=max_points)
        self._steps: deque = deque(maxlen=max_points)
        
        self._padding = 40
        self._bg_color = QColor(30, 30, 30) if PYQT5_AVAILABLE else None
        self._grid_color = QColor(60, 60, 60) if PYQT5_AVAILABLE else None
        self._train_color = QColor(0, 150, 255) if PYQT5_AVAILABLE else None
        self._val_color = QColor(255, 100, 100) if PYQT5_AVAILABLE else None
        
        if PYQT5_AVAILABLE:
            self.setMinimumSize(400, 200)
    
    def add_point(self, step: int, train_loss: float, val_loss: Optional[float] = None):
        """Add a data point."""
        self._steps.append(step)
        self._train_losses.append(train_loss)
        if val_loss is not None:
            self._val_losses.append(val_loss)
        
        if PYQT5_AVAILABLE:
            self.update()
    
    def clear(self):
        """Clear all data."""
        self._train_losses.clear()
        self._val_losses.clear()
        self._steps.clear()
        if PYQT5_AVAILABLE:
            self.update()
    
    def paintEvent(self, event):
        if not PYQT5_AVAILABLE:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), self._bg_color)
        
        # Draw content area
        w = self.width() - self._padding * 2
        h = self.height() - self._padding * 2
        
        if not self._train_losses or w <= 0 or h <= 0:
            painter.setPen(QPen(QColor(100, 100, 100)))
            painter.drawText(self.rect(), Qt.AlignCenter, "No data yet")
            return
        
        # Calculate ranges
        all_losses = list(self._train_losses) + list(self._val_losses)
        min_loss = min(all_losses) * 0.9
        max_loss = max(all_losses) * 1.1
        if max_loss <= min_loss:
            max_loss = min_loss + 1
        
        min_step = min(self._steps)
        max_step = max(self._steps)
        if max_step <= min_step:
            max_step = min_step + 1
        
        # Draw grid
        painter.setPen(QPen(self._grid_color))
        for i in range(5):
            y = self._padding + int(h * i / 4)
            painter.drawLine(self._padding, y, self._padding + w, y)
            
            loss_val = max_loss - (max_loss - min_loss) * i / 4
            painter.drawText(5, y + 4, f"{loss_val:.3f}")
        
        # X axis labels
        for i in range(5):
            x = self._padding + int(w * i / 4)
            step_val = min_step + (max_step - min_step) * i / 4
            painter.drawText(x - 15, self.height() - 5, f"{int(step_val)}")
        
        # Draw train loss
        painter.setPen(QPen(self._train_color, 2))
        self._draw_line(painter, self._steps, self._train_losses, 
                       min_step, max_step, min_loss, max_loss, w, h)
        
        # Draw val loss
        if self._val_losses:
            painter.setPen(QPen(self._val_color, 2))
            self._draw_line(painter, self._steps, self._val_losses,
                           min_step, max_step, min_loss, max_loss, w, h)
        
        # Legend
        painter.setPen(QPen(self._train_color))
        painter.drawText(self._padding + 10, self._padding + 15, "Train Loss")
        if self._val_losses:
            painter.setPen(QPen(self._val_color))
            painter.drawText(self._padding + 100, self._padding + 15, "Val Loss")
    
    def _draw_line(self, painter, steps, losses, min_step, max_step, 
                   min_loss, max_loss, w, h):
        """Draw a line from data points."""
        points = []
        for step, loss in zip(steps, losses):
            x = self._padding + int((step - min_step) / (max_step - min_step) * w)
            y = self._padding + int((max_loss - loss) / (max_loss - min_loss) * h)
            points.append((x, y))
        
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])


class GradientFlowWidget(QWidget if PYQT5_AVAILABLE else object):
    """Widget for visualizing gradient flow through layers."""
    
    def __init__(self, parent=None):
        if PYQT5_AVAILABLE:
            super().__init__(parent)
        
        self._layer_gradients: Dict[str, float] = {}
        self._max_grad = 1.0
        
        if PYQT5_AVAILABLE:
            self.setMinimumSize(300, 200)
    
    def update_gradients(self, model):
        """Update gradient information from model."""
        if not TORCH_AVAILABLE or model is None:
            return
        
        self._layer_gradients = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Simplify layer names
                short_name = name.split('.')[-1] if '.' in name else name
                self._layer_gradients[short_name] = grad_norm
        
        if self._layer_gradients:
            self._max_grad = max(self._layer_gradients.values()) * 1.1
        
        if PYQT5_AVAILABLE:
            self.update()
    
    def paintEvent(self, event):
        if not PYQT5_AVAILABLE:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        if not self._layer_gradients:
            painter.setPen(QPen(QColor(100, 100, 100)))
            painter.drawText(self.rect(), Qt.AlignCenter, "No gradients")
            return
        
        # Draw bars
        layers = list(self._layer_gradients.keys())[:20]  # Limit display
        bar_height = min(20, (self.height() - 40) // len(layers))
        
        for i, layer in enumerate(layers):
            grad = self._layer_gradients[layer]
            y = 20 + i * (bar_height + 2)
            
            # Bar width proportional to gradient
            bar_width = int((self.width() - 150) * grad / max(0.001, self._max_grad))
            
            # Color based on gradient magnitude
            if grad < 0.001:
                color = QColor(100, 100, 100)  # Dead
            elif grad < 0.01:
                color = QColor(255, 200, 0)    # Low
            elif grad < 1.0:
                color = QColor(0, 200, 100)    # Good
            else:
                color = QColor(255, 50, 50)    # Exploding
            
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawRect(140, y, bar_width, bar_height - 2)
            
            # Label
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.drawText(5, y + bar_height - 4, layer[:15])


class AttentionHeatmapWidget(QWidget if PYQT5_AVAILABLE else object):
    """Widget for displaying attention heatmaps."""
    
    def __init__(self, parent=None):
        if PYQT5_AVAILABLE:
            super().__init__(parent)
        
        self._attention: Optional[np.ndarray] = None
        self._tokens: List[str] = []
        
        if PYQT5_AVAILABLE:
            self.setMinimumSize(300, 300)
    
    def set_attention(self, attention: np.ndarray, tokens: List[str]):
        """Set attention weights to display."""
        self._attention = attention
        self._tokens = tokens
        if PYQT5_AVAILABLE:
            self.update()
    
    def paintEvent(self, event):
        if not PYQT5_AVAILABLE:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        if self._attention is None or len(self._tokens) == 0:
            painter.setPen(QPen(QColor(100, 100, 100)))
            painter.drawText(self.rect(), Qt.AlignCenter, "No attention data")
            return
        
        # Calculate cell size
        margin = 50
        n = min(len(self._tokens), 20)  # Limit display
        cell_size = min(
            (self.width() - margin * 2) // n,
            (self.height() - margin * 2) // n,
            30
        )
        
        attn = self._attention[:n, :n] if self._attention.ndim == 2 else self._attention
        
        # Draw heatmap
        for i in range(n):
            for j in range(n):
                x = margin + j * cell_size
                y = margin + i * cell_size
                
                # Color based on attention weight
                weight = float(attn[i, j]) if i < attn.shape[0] and j < attn.shape[1] else 0
                r = int(255 * weight)
                b = int(255 * (1 - weight))
                color = QColor(r, 50, b)
                
                painter.setBrush(QBrush(color))
                painter.setPen(Qt.NoPen)
                painter.drawRect(x, y, cell_size - 1, cell_size - 1)
        
        # Draw token labels
        painter.setPen(QPen(QColor(200, 200, 200)))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        
        for i, token in enumerate(self._tokens[:n]):
            # Top labels
            x = margin + i * cell_size
            painter.save()
            painter.translate(x + cell_size // 2, margin - 5)
            painter.rotate(-45)
            painter.drawText(0, 0, token[:5])
            painter.restore()
            
            # Left labels
            y = margin + i * cell_size + cell_size // 2
            painter.drawText(5, y, token[:5])


class MetricsDashboard(QWidget if PYQT5_AVAILABLE else object):
    """Dashboard showing live training metrics."""
    
    def __init__(self, parent=None):
        if PYQT5_AVAILABLE:
            super().__init__(parent)
            self._setup_ui()
        
        self._metrics = TrainingMetrics()
    
    def _setup_ui(self):
        if not PYQT5_AVAILABLE:
            return
        
        layout = QGridLayout(self)
        layout.setSpacing(10)
        
        # Metric labels
        self._labels = {}
        metrics = [
            ("Step", "step"),
            ("Epoch", "epoch"),
            ("Train Loss", "train_loss"),
            ("Val Loss", "val_loss"),
            ("Learning Rate", "learning_rate"),
            ("Gradient Norm", "gradient_norm"),
            ("Tokens/s", "tokens_per_second"),
            ("GPU Memory", "gpu_memory"),
        ]
        
        for i, (label, key) in enumerate(metrics):
            row = i // 2
            col = (i % 2) * 2
            
            name_label = QLabel(f"{label}:")
            name_label.setStyleSheet("color: #888; font-size: 10pt;")
            layout.addWidget(name_label, row, col)
            
            value_label = QLabel("--")
            value_label.setStyleSheet("color: #fff; font-size: 12pt; font-weight: bold;")
            layout.addWidget(value_label, row, col + 1)
            
            self._labels[key] = value_label
    
    def update_metrics(self, metrics: TrainingMetrics):
        """Update displayed metrics."""
        if not PYQT5_AVAILABLE:
            return
        
        self._metrics = metrics
        
        self._labels["step"].setText(str(metrics.step))
        self._labels["epoch"].setText(str(metrics.epoch))
        self._labels["train_loss"].setText(f"{metrics.train_loss:.4f}")
        self._labels["val_loss"].setText(
            f"{metrics.val_loss:.4f}" if metrics.val_loss else "--"
        )
        self._labels["learning_rate"].setText(f"{metrics.learning_rate:.2e}")
        self._labels["gradient_norm"].setText(f"{metrics.gradient_norm:.4f}")
        self._labels["tokens_per_second"].setText(f"{metrics.tokens_per_second:.1f}")
        
        if metrics.gpu_memory_total > 0:
            pct = metrics.gpu_memory_used / metrics.gpu_memory_total * 100
            self._labels["gpu_memory"].setText(
                f"{metrics.gpu_memory_used:.1f}/{metrics.gpu_memory_total:.1f} GB ({pct:.0f}%)"
            )
        else:
            self._labels["gpu_memory"].setText("N/A")


class TrainingVisualizer(QWidget if PYQT5_AVAILABLE else object):
    """
    Main training visualization widget.
    
    Combines loss plots, gradient flow, attention heatmaps,
    and live metrics into a single dashboard.
    """
    
    if PYQT5_AVAILABLE:
        metrics_updated = pyqtSignal(object)
    
    def __init__(self, parent=None):
        if PYQT5_AVAILABLE:
            super().__init__(parent)
            self._setup_ui()
        
        self._update_queue = []
        self._lock = threading.Lock()
    
    def _setup_ui(self):
        if not PYQT5_AVAILABLE:
            return
        
        layout = QVBoxLayout(self)
        
        # Tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Loss tab
        loss_widget = QWidget()
        loss_layout = QVBoxLayout(loss_widget)
        
        self._metrics_dashboard = MetricsDashboard()
        loss_layout.addWidget(self._metrics_dashboard)
        
        self._loss_plot = LossPlotWidget()
        loss_layout.addWidget(self._loss_plot)
        
        tabs.addTab(loss_widget, "Loss")
        
        # Gradients tab
        self._gradient_widget = GradientFlowWidget()
        tabs.addTab(self._gradient_widget, "Gradients")
        
        # Attention tab
        self._attention_widget = AttentionHeatmapWidget()
        tabs.addTab(self._attention_widget, "Attention")
        
        # Controls
        controls = QHBoxLayout()
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_all)
        controls.addWidget(clear_btn)
        
        controls.addStretch()
        
        layout.addLayout(controls)
        
        # Update timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._process_updates)
        self._timer.start(100)  # 10 Hz
        
        self.setMinimumSize(600, 400)
        self.setWindowTitle("Training Visualization")
    
    def update_loss(
        self,
        step: int,
        train_loss: float,
        val_loss: Optional[float] = None
    ):
        """Update loss data."""
        with self._lock:
            self._update_queue.append(('loss', step, train_loss, val_loss))
    
    def update_metrics(self, metrics: TrainingMetrics):
        """Update all metrics."""
        with self._lock:
            self._update_queue.append(('metrics', metrics))
    
    def update_gradients(self, model):
        """Update gradient visualization."""
        with self._lock:
            self._update_queue.append(('gradients', model))
    
    def show_attention(
        self,
        attention_weights: np.ndarray,
        tokens: List[str]
    ):
        """Show attention heatmap."""
        with self._lock:
            self._update_queue.append(('attention', attention_weights, tokens))
    
    def _process_updates(self):
        """Process queued updates on main thread."""
        if not PYQT5_AVAILABLE:
            return
        
        with self._lock:
            updates = self._update_queue.copy()
            self._update_queue.clear()
        
        for update in updates:
            update_type = update[0]
            
            if update_type == 'loss':
                _, step, train_loss, val_loss = update
                self._loss_plot.add_point(step, train_loss, val_loss)
                
            elif update_type == 'metrics':
                _, metrics = update
                self._metrics_dashboard.update_metrics(metrics)
                if hasattr(self, 'metrics_updated'):
                    self.metrics_updated.emit(metrics)
                    
            elif update_type == 'gradients':
                _, model = update
                self._gradient_widget.update_gradients(model)
                
            elif update_type == 'attention':
                _, weights, tokens = update
                self._attention_widget.set_attention(weights, tokens)
    
    def _clear_all(self):
        """Clear all visualizations."""
        if PYQT5_AVAILABLE:
            self._loss_plot.clear()
            self._gradient_widget._layer_gradients = {}
            self._attention_widget._attention = None
            self.update()


# Training callback for easy integration
class VisualizationCallback:
    """
    Callback to integrate with training loops.
    
    Usage:
        viz = TrainingVisualizer()
        callback = VisualizationCallback(viz)
        
        # In training loop:
        callback.on_batch_end(step, loss, model)
    """
    
    def __init__(self, visualizer: TrainingVisualizer):
        self._viz = visualizer
        self._start_time = time.time()
        self._tokens_processed = 0
    
    def on_batch_end(
        self,
        step: int,
        train_loss: float,
        model=None,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        tokens: int = 0
    ):
        """Called at end of each batch."""
        self._tokens_processed += tokens
        elapsed = time.time() - self._start_time
        
        # Calculate metrics
        gradient_norm = 0.0
        if model and TORCH_AVAILABLE:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            gradient_norm = total_norm ** 0.5
        
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1e9
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        metrics = TrainingMetrics(
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=learning_rate or 0.0,
            gradient_norm=gradient_norm,
            tokens_per_second=self._tokens_processed / max(1, elapsed),
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total
        )
        
        self._viz.update_loss(step, train_loss, val_loss)
        self._viz.update_metrics(metrics)
        
        if model and step % 10 == 0:
            self._viz.update_gradients(model)
    
    def on_epoch_end(self, epoch: int):
        """Called at end of each epoch."""
        pass


def create_visualizer() -> Optional[TrainingVisualizer]:
    """Create a training visualizer if PyQt5 is available."""
    if PYQT5_AVAILABLE:
        return TrainingVisualizer()
    else:
        logger.warning("PyQt5 not available for training visualization")
        return None
