"""
Resource Monitor Widget
========================

Real-time resource usage dashboard for the GUI.
Shows CPU, GPU, memory usage and performance metrics.
Now includes per-module memory breakdown for loaded modules.
"""

import logging
import sys
import time

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# Optional imports for resource monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

# Delay torch import to avoid DLL conflicts on Windows
torch = None
HAS_TORCH = False

def _lazy_import_torch():
    """Lazily import torch to avoid DLL conflicts."""
    global torch, HAS_TORCH
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
            HAS_TORCH = True
        except (ImportError, OSError):
            HAS_TORCH = False
    return torch


class ResourceMonitor(QWidget):
    """
    Widget that displays real-time resource usage.
    Now includes per-module memory breakdown.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
        # Setup update timer for hardware metrics
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(1000)  # Update every second
        
        # Setup slower timer for module memory (less frequent)
        self.module_timer = QTimer()
        self.module_timer.timeout.connect(self.update_module_memory)
        self.module_timer.start(5000)  # Update every 5 seconds
        
        # Track metrics
        self.last_update = time.time()
        self.generation_count = 0
        self.total_tokens = 0
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Resource Monitor")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # === HARDWARE STATUS ===
        hw_group = QGroupBox("Hardware Status")
        hw_layout = QGridLayout()
        
        # CPU Usage
        self.cpu_label = QLabel("CPU:")
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setMaximum(100)
        self.cpu_bar.setTextVisible(True)
        hw_layout.addWidget(self.cpu_label, 0, 0)
        hw_layout.addWidget(self.cpu_bar, 0, 1)
        
        # Memory Usage
        self.mem_label = QLabel("Memory:")
        self.mem_bar = QProgressBar()
        self.mem_bar.setMaximum(100)
        self.mem_bar.setTextVisible(True)
        hw_layout.addWidget(self.mem_label, 1, 0)
        hw_layout.addWidget(self.mem_bar, 1, 1)
        
        # GPU Usage (if available)
        self.gpu_label = QLabel("GPU:")
        self.gpu_bar = QProgressBar()
        self.gpu_bar.setMaximum(100)
        self.gpu_bar.setTextVisible(True)
        hw_layout.addWidget(self.gpu_label, 2, 0)
        hw_layout.addWidget(self.gpu_bar, 2, 1)
        
        # VRAM Usage
        self.vram_label = QLabel("VRAM:")
        self.vram_bar = QProgressBar()
        self.vram_bar.setMaximum(100)
        self.vram_bar.setTextVisible(True)
        hw_layout.addWidget(self.vram_label, 3, 0)
        hw_layout.addWidget(self.vram_bar, 3, 1)
        
        hw_group.setLayout(hw_layout)
        layout.addWidget(hw_group)
        
        # === PERFORMANCE METRICS ===
        perf_group = QGroupBox("Performance")
        perf_layout = QVBoxLayout()
        
        self.tokens_per_sec_label = QLabel("Tokens/sec: --")
        self.avg_latency_label = QLabel("Avg Latency: --")
        self.generations_label = QLabel("Generations: 0")
        
        perf_layout.addWidget(self.tokens_per_sec_label)
        perf_layout.addWidget(self.avg_latency_label)
        perf_layout.addWidget(self.generations_label)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # === CURRENT MODE ===
        mode_group = QGroupBox("Current Mode")
        mode_layout = QVBoxLayout()
        
        self.mode_label = QLabel("Mode: Balanced")
        self.threads_label = QLabel("CPU Threads: Auto")
        self.gpu_frac_label = QLabel("GPU Memory: 50%")
        
        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.threads_label)
        mode_layout.addWidget(self.gpu_frac_label)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # === MODULE MEMORY BREAKDOWN ===
        self.module_group = QGroupBox("Module Memory Usage")
        self.module_layout = QVBoxLayout()
        
        # Header
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Module"))
        header_layout.addStretch()
        header_layout.addWidget(QLabel("RAM"))
        header_layout.addWidget(QLabel("VRAM"))
        self.module_layout.addLayout(header_layout)
        
        # Scrollable area for modules
        self.module_scroll = QScrollArea()
        self.module_scroll.setWidgetResizable(True)
        self.module_scroll.setMaximumHeight(150)
        self.module_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
        
        self.module_container = QWidget()
        self.module_container_layout = QVBoxLayout(self.module_container)
        self.module_container_layout.setContentsMargins(0, 0, 0, 0)
        self.module_container_layout.setSpacing(2)
        
        # Placeholder label
        self.no_modules_label = QLabel("No modules loaded")
        self.no_modules_label.setStyleSheet("color: #6c7086; font-style: italic;")
        self.module_container_layout.addWidget(self.no_modules_label)
        self.module_container_layout.addStretch()
        
        self.module_scroll.setWidget(self.module_container)
        self.module_layout.addWidget(self.module_scroll)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh Modules")
        refresh_btn.setMaximumWidth(120)
        refresh_btn.clicked.connect(self.update_module_memory)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background: #45475a;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                color: #cdd6f4;
                font-size: 10px;
            }
            QPushButton:hover { background: #585b70; }
        """)
        self.module_layout.addWidget(refresh_btn)
        
        self.module_group.setLayout(self.module_layout)
        layout.addWidget(self.module_group)
        
        # Track module widgets for updates
        self._module_widgets = {}
        
        layout.addStretch()
        
        # Initial update
        self.update_metrics()
        self.update_module_memory()
    
    def update_module_memory(self):
        """Update the per-module memory breakdown."""
        try:
            from ..modules import get_manager
            manager = get_manager()
            
            # Clear existing widgets
            for widget in self._module_widgets.values():
                widget.setParent(None)
                widget.deleteLater()
            self._module_widgets.clear()
            
            # Get loaded modules
            loaded_modules = manager.list_loaded() if hasattr(manager, 'list_loaded') else []
            
            if not loaded_modules:
                self.no_modules_label.show()
                return
            
            self.no_modules_label.hide()
            
            # Calculate memory per module
            for mod_id in loaded_modules:
                try:
                    mod = manager.get_module(mod_id)
                    if mod is None:
                        continue
                    
                    # Create row widget
                    row = QFrame()
                    row.setStyleSheet("""
                        QFrame {
                            background: rgba(69, 71, 90, 0.3);
                            border-radius: 3px;
                            padding: 2px;
                        }
                    """)
                    row_layout = QHBoxLayout(row)
                    row_layout.setContentsMargins(4, 2, 4, 2)
                    row_layout.setSpacing(8)
                    
                    # Module name
                    name_label = QLabel(mod_id[:15] + "..." if len(mod_id) > 15 else mod_id)
                    name_label.setToolTip(mod_id)
                    name_label.setMinimumWidth(100)
                    name_label.setStyleSheet("color: #cdd6f4; font-size: 10px;")
                    row_layout.addWidget(name_label)
                    
                    row_layout.addStretch()
                    
                    # RAM usage estimate
                    ram_mb = self._estimate_module_ram(mod)
                    ram_label = QLabel(f"{ram_mb:.0f} MB")
                    ram_label.setStyleSheet(self._get_memory_color(ram_mb, 500))
                    ram_label.setMinimumWidth(50)
                    row_layout.addWidget(ram_label)
                    
                    # VRAM usage estimate
                    vram_mb = self._estimate_module_vram(mod)
                    vram_label = QLabel(f"{vram_mb:.0f} MB" if vram_mb > 0 else "--")
                    vram_label.setStyleSheet(self._get_memory_color(vram_mb, 1000))
                    vram_label.setMinimumWidth(50)
                    row_layout.addWidget(vram_label)
                    
                    self.module_container_layout.insertWidget(
                        self.module_container_layout.count() - 1, row
                    )
                    self._module_widgets[mod_id] = row
                    
                except Exception as e:
                    logger.debug(f"Error getting memory for module {mod_id}: {e}")
                    
        except ImportError:
            # Module manager not available
            self.no_modules_label.setText("Module manager not available")
            self.no_modules_label.show()
        except Exception as e:
            logger.debug(f"Error updating module memory: {e}")
    
    def _estimate_module_ram(self, module) -> float:
        """Estimate RAM usage of a module in MB."""
        try:
            instance = module.get_interface() if hasattr(module, 'get_interface') else None
            if instance is None:
                return 0.0
            
            # Try to get size from torch models
            if HAS_TORCH:
                if hasattr(instance, 'parameters'):
                    # It's a torch model
                    param_size = sum(p.numel() * p.element_size() for p in instance.parameters())
                    return param_size / (1024 * 1024)
            
            # Rough estimate based on sys.getsizeof
            return sys.getsizeof(instance) / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _estimate_module_vram(self, module) -> float:
        """Estimate VRAM usage of a module in MB."""
        try:
            _torch = _lazy_import_torch()
            if not HAS_TORCH or _torch is None or not _torch.cuda.is_available():
                return 0.0
                
            instance = module.get_interface() if hasattr(module, 'get_interface') else None
            if instance is None:
                return 0.0
            
            # Check if model is on GPU
            if hasattr(instance, 'parameters'):
                for p in instance.parameters():
                    if p.is_cuda:
                        param_size = sum(
                            p.numel() * p.element_size() 
                            for p in instance.parameters() 
                            if p.is_cuda
                        )
                        return param_size / (1024 * 1024)
                    break  # Only check first param
            return 0.0
        except Exception:
            return 0.0
    
    def _get_memory_color(self, mb: float, threshold: float) -> str:
        """Get color style based on memory usage."""
        if mb == 0:
            return "color: #6c7086; font-size: 10px;"
        elif mb < threshold * 0.5:
            return "color: #a6e3a1; font-size: 10px;"  # Green
        elif mb < threshold:
            return "color: #f9e2af; font-size: 10px;"  # Yellow
        else:
            return "color: #f38ba8; font-size: 10px;"  # Red
    
    def update_metrics(self):
        """Update all resource metrics."""
        # CPU usage
        if HAS_PSUTIL:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_bar.setValue(int(cpu_percent))
            self.cpu_bar.setFormat(f"{cpu_percent:.1f}%")
            
            # Apply color based on usage
            if cpu_percent > 80:
                self.cpu_bar.setStyleSheet("QProgressBar::chunk { background-color: #ef4444; }")
            elif cpu_percent > 50:
                self.cpu_bar.setStyleSheet("QProgressBar::chunk { background-color: #f59e0b; }")
            else:
                self.cpu_bar.setStyleSheet("QProgressBar::chunk { background-color: #22c55e; }")
        else:
            self.cpu_bar.setValue(0)
            self.cpu_bar.setFormat("N/A")
        
        # Memory usage
        if HAS_PSUTIL:
            mem = psutil.virtual_memory()
            mem_percent = mem.percent
            self.mem_bar.setValue(int(mem_percent))
            self.mem_bar.setFormat(f"{mem_percent:.1f}% ({mem.used // (1024**3)} GB / {mem.total // (1024**3)} GB)")
            
            if mem_percent > 85:
                self.mem_bar.setStyleSheet("QProgressBar::chunk { background-color: #ef4444; }")
            elif mem_percent > 70:
                self.mem_bar.setStyleSheet("QProgressBar::chunk { background-color: #f59e0b; }")
            else:
                self.mem_bar.setStyleSheet("QProgressBar::chunk { background-color: #22c55e; }")
        else:
            self.mem_bar.setValue(0)
            self.mem_bar.setFormat("N/A")
        
        # GPU usage
        _torch = _lazy_import_torch()
        if HAS_TORCH and _torch is not None and _torch.cuda.is_available():
            # GPU utilization
            # Note: This is a simplified version - real GPU monitoring
            # would require nvidia-ml-py3 or similar
            self.gpu_bar.setValue(0)
            self.gpu_bar.setFormat("Available")
            self.gpu_bar.setStyleSheet("QProgressBar::chunk { background-color: #22c55e; }")
            
            # VRAM usage
            vram_used = _torch.cuda.memory_allocated(0) / (1024**3)
            vram_total = _torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_percent = (vram_used / vram_total * 100) if vram_total > 0 else 0
            
            self.vram_bar.setValue(int(vram_percent))
            self.vram_bar.setFormat(f"{vram_percent:.1f}% ({vram_used:.2f} GB / {vram_total:.1f} GB)")
            
            if vram_percent > 90:
                self.vram_bar.setStyleSheet("QProgressBar::chunk { background-color: #ef4444; }")
            elif vram_percent > 75:
                self.vram_bar.setStyleSheet("QProgressBar::chunk { background-color: #f59e0b; }")
            else:
                self.vram_bar.setStyleSheet("QProgressBar::chunk { background-color: #22c55e; }")
        else:
            self.gpu_bar.setValue(0)
            self.gpu_bar.setFormat("No GPU")
            self.gpu_bar.setStyleSheet("QProgressBar::chunk { background-color: #888; }")
            
            self.vram_bar.setValue(0)
            self.vram_bar.setFormat("N/A")
            self.vram_bar.setStyleSheet("QProgressBar::chunk { background-color: #888; }")
        
        # Update mode info
        try:
            from enigma_engine.core.resources import get_resource_info
            info = get_resource_info()
            
            self.mode_label.setText(f"Mode: {info['mode'].capitalize()}")
            
            threads_text = info.get('torch_threads', 'N/A')
            self.threads_label.setText(f"CPU Threads: {threads_text}")
            
            gpu_frac = int(info['gpu_memory_fraction'] * 100)
            self.gpu_frac_label.setText(f"GPU Memory Limit: {gpu_frac}%")
        except Exception:
            pass  # Intentionally silent
    
    def record_generation(self, tokens: int, latency: float):
        """
        Record a generation event for performance tracking.
        
        Args:
            tokens: Number of tokens generated
            latency: Time taken in seconds
        """
        self.generation_count += 1
        self.total_tokens += tokens
        
        # Update labels
        self.generations_label.setText(f"Generations: {self.generation_count}")
        
        if latency > 0:
            tokens_per_sec = tokens / latency
            self.tokens_per_sec_label.setText(f"Tokens/sec: {tokens_per_sec:.1f}")
        
        # Calculate average latency
        if self.generation_count > 0:
            # This is simplified - would need to track all latencies for true average
            self.avg_latency_label.setText(f"Last Latency: {latency:.2f}s")
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.generation_count = 0
        self.total_tokens = 0
        self.generations_label.setText("Generations: 0")
        self.tokens_per_sec_label.setText("Tokens/sec: --")
        self.avg_latency_label.setText("Avg Latency: --")


def create_resource_monitor_widget(parent=None):
    """
    Create a resource monitor widget.
    
    Args:
        parent: Parent widget
        
    Returns:
        ResourceMonitor instance
    """
    return ResourceMonitor(parent)


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    monitor = ResourceMonitor()
    monitor.setWindowTitle("Resource Monitor")
    monitor.resize(400, 600)
    monitor.show()
    
    sys.exit(app.exec_())
