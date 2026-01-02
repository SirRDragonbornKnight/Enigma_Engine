"""
Resource Monitor Widget
========================

Real-time resource usage dashboard for the GUI.
Shows CPU, GPU, memory usage and performance metrics.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QProgressBar, QGroupBox, QGridLayout
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
import time


class ResourceMonitor(QWidget):
    """
    Widget that displays real-time resource usage.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
        # Setup update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(1000)  # Update every second
        
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
        
        layout.addStretch()
        
        # Initial update
        self.update_metrics()
    
    def update_metrics(self):
        """Update all resource metrics."""
        # CPU usage
        try:
            import psutil
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
        except ImportError:
            self.cpu_bar.setValue(0)
            self.cpu_bar.setFormat("N/A")
        
        # Memory usage
        try:
            import psutil
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
        except ImportError:
            self.mem_bar.setValue(0)
            self.mem_bar.setFormat("N/A")
        
        # GPU usage
        try:
            import torch
            if torch.cuda.is_available():
                # GPU utilization
                # Note: This is a simplified version - real GPU monitoring
                # would require nvidia-ml-py3 or similar
                self.gpu_bar.setValue(0)
                self.gpu_bar.setFormat("Available")
                self.gpu_bar.setStyleSheet("QProgressBar::chunk { background-color: #22c55e; }")
                
                # VRAM usage
                vram_used = torch.cuda.memory_allocated(0) / (1024**3)
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
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
        except ImportError:
            self.gpu_bar.setValue(0)
            self.gpu_bar.setFormat("N/A")
            self.vram_bar.setValue(0)
            self.vram_bar.setFormat("N/A")
        
        # Update mode info
        try:
            from enigma.core.resources import get_resource_info
            info = get_resource_info()
            
            self.mode_label.setText(f"Mode: {info['mode'].capitalize()}")
            
            threads_text = info.get('torch_threads', 'N/A')
            self.threads_label.setText(f"CPU Threads: {threads_text}")
            
            gpu_frac = int(info['gpu_memory_fraction'] * 100)
            self.gpu_frac_label.setText(f"GPU Memory Limit: {gpu_frac}%")
        except Exception:
            pass
    
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
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    monitor = ResourceMonitor()
    monitor.setWindowTitle("Resource Monitor")
    monitor.resize(400, 600)
    monitor.show()
    
    sys.exit(app.exec_())
