"""
Model Scaling Tab - Interactive Model Size Explorer
===================================================

A beautiful, interactive interface for exploring and selecting model sizes.
Features:
  - Interactive pyramid visualization
  - Real-time hardware requirement calculator
  - Performance estimates
  - One-click model creation
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtGui import QPaintEvent, QMouseEvent

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea,
    QLabel, QPushButton, QFrame, QGroupBox, QMessageBox, QProgressBar,
    QSlider, QSpinBox, QStackedWidget, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QPainterPath

from .shared_components import NoScrollComboBox

# Qt enum constants
NoBrush = Qt.BrushStyle.NoBrush
FontBold = QFont.Weight.Bold
FontNormal = QFont.Weight.Normal

import math


# Model definitions with full specs
MODEL_SPECS = {
    'nano':     {'params': 1,     'dim': 128,  'layers': 4,  'heads': 4,  'ctx': 256,   'ram': 256,   'vram': 0,    'tier': 'embedded',   'emoji': '', 'desc': 'Microcontrollers, testing'},
    'micro':    {'params': 2,     'dim': 192,  'layers': 4,  'heads': 4,  'ctx': 384,   'ram': 512,   'vram': 0,    'tier': 'embedded',   'emoji': '', 'desc': 'IoT devices, ESP32'},
    'tiny':     {'params': 5,     'dim': 256,  'layers': 6,  'heads': 8,  'ctx': 512,   'ram': 1024,  'vram': 0,    'tier': 'edge',       'emoji': '', 'desc': 'Raspberry Pi 3/4'},
    'mini':     {'params': 10,    'dim': 384,  'layers': 6,  'heads': 6,  'ctx': 512,   'ram': 2048,  'vram': 0,    'tier': 'edge',       'emoji': '', 'desc': 'Mobile, tablets'},
    'small':    {'params': 27,    'dim': 512,  'layers': 8,  'heads': 8,  'ctx': 1024,  'ram': 4096,  'vram': 2048, 'tier': 'consumer',   'emoji': '', 'desc': 'Laptops, entry GPU'},
    'medium':   {'params': 85,    'dim': 768,  'layers': 12, 'heads': 12, 'ctx': 2048,  'ram': 8192,  'vram': 4096, 'tier': 'consumer',   'emoji': '', 'desc': 'Desktop, GTX 1660+'},
    'base':     {'params': 125,   'dim': 896,  'layers': 14, 'heads': 14, 'ctx': 2048,  'ram': 12288, 'vram': 6144, 'tier': 'consumer',   'emoji': '', 'desc': 'Gaming PC, RTX 3060'},
    'large':    {'params': 200,   'dim': 1024, 'layers': 16, 'heads': 16, 'ctx': 4096,  'ram': 16384, 'vram': 8192, 'tier': 'prosumer',   'emoji': '', 'desc': 'Workstation, RTX 3070+'},
    'xl':       {'params': 600,   'dim': 1536, 'layers': 24, 'heads': 24, 'ctx': 4096,  'ram': 32768, 'vram': 12288,'tier': 'prosumer',   'emoji': '', 'desc': 'High-end, RTX 3080+'},
    'xxl':      {'params': 1500,  'dim': 2048, 'layers': 32, 'heads': 32, 'ctx': 8192,  'ram': 65536, 'vram': 24576,'tier': 'server',     'emoji': '', 'desc': 'Server, RTX 4090'},
    'huge':     {'params': 3000,  'dim': 2560, 'layers': 40, 'heads': 40, 'ctx': 8192,  'ram': 131072,'vram': 49152,'tier': 'server',     'emoji': '', 'desc': 'Multi-GPU server'},
    'giant':    {'params': 7000,  'dim': 4096, 'layers': 32, 'heads': 32, 'ctx': 8192,  'ram': 262144,'vram': 81920,'tier': 'datacenter', 'emoji': '', 'desc': 'A100/H100 cluster'},
    'colossal': {'params': 13000, 'dim': 4096, 'layers': 48, 'heads': 32, 'ctx': 16384, 'ram': 524288,'vram': 163840,'tier': 'datacenter','emoji': '', 'desc': 'Distributed training'},
    'titan':    {'params': 30000, 'dim': 6144, 'layers': 48, 'heads': 48, 'ctx': 16384, 'ram': 1048576,'vram': 327680,'tier': 'ultimate', 'emoji': '', 'desc': 'Full datacenter'},
    'omega':    {'params': 70000, 'dim': 8192, 'layers': 64, 'heads': 64, 'ctx': 32768, 'ram': 2097152,'vram': 655360,'tier': 'ultimate', 'emoji': '', 'desc': 'Research frontier'},
}

MODEL_ORDER = ['nano', 'micro', 'tiny', 'mini', 'small', 'medium', 'base', 'large', 'xl', 'xxl', 'huge', 'giant', 'colossal', 'titan', 'omega']

TIER_COLORS = {
    'embedded':   ('#ff6b6b', '#ff8787'),  # Red
    'edge':       ('#ffa94d', '#ffc078'),  # Orange
    'consumer':   ('#ffd43b', '#ffe066'),  # Yellow
    'prosumer':   ('#69db7c', '#8ce99a'),  # Green
    'server':     ('#74c0fc', '#a5d8ff'),  # Blue
    'datacenter': ('#b197fc', '#d0bfff'),  # Purple
    'ultimate':   ('#63e6be', '#96f2d7'),  # Teal
}

TIER_NAMES = {
    'embedded': 'Embedded',
    'edge': 'Edge',
    'consumer': 'Consumer',
    'prosumer': 'Prosumer',
    'server': 'Server',
    'datacenter': 'Datacenter',
    'ultimate': 'Ultimate'
}


class PyramidWidget(QFrame):
    """Interactive pyramid visualization of model sizes."""
    
    model_clicked = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)  # Smaller minimum for flexibility
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow resize
        self.selected = 'small'
        self.hovered = None
        self.setMouseTracking(True)
        self.model_rects = {}
        
    def paintEvent(self, a0: Optional[QPaintEvent]):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Background gradient
        bg = QLinearGradient(0, 0, 0, h)
        bg.setColorAt(0, QColor('#0d1117'))
        bg.setColorAt(1, QColor('#161b22'))
        painter.fillRect(0, 0, w, h, bg)
        
        # Dynamic sizing based on widget size
        margin = max(20, min(50, w // 20))  # Scale margin with width
        pyramid_width = w - (margin * 2)
        title_height = 45  # Space for title at top
        bottom_margin = 15  # Small margin at bottom
        pyramid_left = margin
        
        # Group models by tier for pyramid rows
        tiers = ['embedded', 'edge', 'consumer', 'prosumer', 'server', 'datacenter', 'ultimate']
        tier_models = {t: [] for t in tiers}
        for name in MODEL_ORDER:
            tier_models[MODEL_SPECS[name]['tier']].append(name)
        
        # Calculate dynamic row heights - pyramid fills available space
        total_rows = len([t for t in tiers if tier_models[t]])
        available_height = h - title_height - bottom_margin
        base_row_height = available_height / total_rows
        
        # Draw from bottom (ultimate) to top (embedded)
        current_y = h - bottom_margin  # Start from bottom
        
        self.model_rects = {}
        
        for i, tier in enumerate(reversed(tiers)):
            models = tier_models[tier]
            if not models:
                continue
                
            row_height = base_row_height  # Use calculated height to fill space
            current_y -= row_height
            
            # Calculate width for this row (narrower at top)
            width_factor = 0.3 + (0.7 * (i / max(1, len(tiers) - 1)))
            row_width = pyramid_width * width_factor
            row_left = pyramid_left + (pyramid_width - row_width) / 2
            
            # Draw tier background
            colors = TIER_COLORS[tier]
            tier_bg = QLinearGradient(row_left, current_y, row_left + row_width, current_y)
            tier_bg.setColorAt(0, QColor(colors[0]).darker(150))
            tier_bg.setColorAt(0.5, QColor(colors[0]))
            tier_bg.setColorAt(1, QColor(colors[0]).darker(150))
            
            path = QPainterPath()
            path.addRoundedRect(row_left, current_y, row_width, row_height - 2, 6, 6)
            painter.fillPath(path, tier_bg)
            
            # Draw models in this tier
            model_width = row_width / len(models)
            for j, model in enumerate(models):
                mx = row_left + j * model_width
                rect = (mx + 2, current_y + 2, model_width - 4, row_height - 6)
                self.model_rects[model] = rect
                
                # Highlight selected/hovered
                if model == self.selected:
                    painter.setPen(QPen(QColor('#ffffff'), 3))
                    painter.setBrush(NoBrush)
                    painter.drawRoundedRect(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]), 4, 4)
                elif model == self.hovered:
                    painter.setPen(QPen(QColor('#ffffff88'), 2))
                    painter.setBrush(NoBrush)
                    painter.drawRoundedRect(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]), 4, 4)
                
                # Model name
                spec = MODEL_SPECS[model]
                painter.setPen(QPen(QColor('#ffffff')))
                font_size = max(8, int(row_height / 3))  # Scale with row height
                font = QFont('Segoe UI', font_size, FontBold if model == self.selected else FontNormal)
                painter.setFont(font)
                
                text = f"{spec['emoji']} {model.upper()}"
                text_rect = painter.fontMetrics().boundingRect(text)
                tx = mx + (model_width - text_rect.width()) / 2
                ty = current_y + row_height / 2 + text_rect.height() / 4
                painter.drawText(int(tx), int(ty), text)
        
        # Title - dynamic font size
        painter.setPen(QPen(QColor('#ffffff')))
        title_font_size = max(12, min(16, w // 40))
        painter.setFont(QFont('Segoe UI', title_font_size, FontBold))
        painter.drawText(margin, 30, "Model Size Pyramid")
        
        # Subtitle
        painter.setPen(QPen(QColor('#8b949e')))
        subtitle_font_size = max(8, min(10, w // 60))
        painter.setFont(QFont('Segoe UI', subtitle_font_size))
        subtitle_text = "Click to select"
        subtitle_width = painter.fontMetrics().boundingRect(subtitle_text).width()
        painter.drawText(w - margin - subtitle_width, 30, subtitle_text)
        
        painter.end()
        
    def mouseMoveEvent(self, a0: Optional[QMouseEvent]):
        if a0 is None:
            return
        pos = a0.pos()
        old_hover = self.hovered
        self.hovered = None
        
        for model, rect in self.model_rects.items():
            if rect[0] <= pos.x() <= rect[0] + rect[2] and rect[1] <= pos.y() <= rect[1] + rect[3]:
                self.hovered = model
                break
        
        if old_hover != self.hovered:
            self.update()
            
    def mousePressEvent(self, a0: Optional[QMouseEvent]):
        if self.hovered:
            self.selected = self.hovered
            self.model_clicked.emit(self.hovered)
            self.update()
            
    def set_selected(self, model):
        self.selected = model
        self.update()


class SpecsPanel(QFrame):
    """Panel showing specs for selected model."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
            }
        """)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Header: model name + tier
        header_row = QHBoxLayout()
        self.header = QLabel("SMALL")
        self.header.setFont(QFont('Segoe UI', 20, FontBold))
        self.header.setStyleSheet("color: #ffd43b;")
        header_row.addWidget(self.header)
        
        self.tier_badge = QLabel("Consumer")
        self.tier_badge.setStyleSheet("""
            background: #ffd43b33;
            color: #ffd43b;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        """)
        header_row.addWidget(self.tier_badge)
        header_row.addStretch()
        layout.addLayout(header_row)
        
        # Description
        self.desc = QLabel("Laptops, entry GPU")
        self.desc.setStyleSheet("color: #8b949e; font-size: 13px;")
        layout.addWidget(self.desc)
        
        layout.addSpacing(6)
        
        # Specs grid - 2 columns
        specs_grid = QGridLayout()
        specs_grid.setSpacing(6)
        
        self.params_label = QLabel("~27M")
        self.dim_label = QLabel("512")
        self.layers_label = QLabel("8")
        self.heads_label = QLabel("8")
        self.ctx_label = QLabel("1024")
        self.ram_label = QLabel("4 GB")
        self.vram_label = QLabel("2 GB")
        
        specs = [
            ("Parameters:", self.params_label),
            ("Dim:", self.dim_label),
            ("Layers:", self.layers_label),
            ("Heads:", self.heads_label),
            ("Context:", self.ctx_label),
            ("RAM:", self.ram_label),
            ("VRAM:", self.vram_label),
        ]
        
        for i, (label, value_widget) in enumerate(specs):
            row = i // 2
            col = (i % 2) * 2
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #8b949e; font-size: 12px;")
            value_widget.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 12px;")
            specs_grid.addWidget(lbl, row, col)
            specs_grid.addWidget(value_widget, row, col + 1)
            
        layout.addLayout(specs_grid)
        
        layout.addSpacing(6)
        
        # Usage bars
        bars_layout = QHBoxLayout()
        bars_layout.setSpacing(12)
        
        for bar_name in ["RAM", "VRAM"]:
            col = QVBoxLayout()
            col.setSpacing(3)
            lbl = QLabel(bar_name)
            lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
            col.addWidget(lbl)
            bar = QProgressBar()
            bar.setMaximum(100)
            bar.setValue(25 if bar_name == "RAM" else 15)
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            bar.setStyleSheet("""
                QProgressBar { background: #30363d; border-radius: 4px; }
                QProgressBar::chunk { background: #69db7c; border-radius: 4px; }
            """)
            col.addWidget(bar)
            bars_layout.addLayout(col)
            if bar_name == "RAM":
                self.ram_bar = bar
            else:
                self.vram_bar = bar
        
        layout.addLayout(bars_layout)
        
        layout.addSpacing(6)
        
        # Performance
        self.perf_label = QLabel("~50 tok/s on RTX 3060")
        self.perf_label.setStyleSheet("color: #69db7c; font-size: 12px;")
        layout.addWidget(self.perf_label)
        
        self.quality_label = QLabel("Good for: Conversations, learning")
        self.quality_label.setStyleSheet("color: #bac2de; font-size: 11px;")
        self.quality_label.setWordWrap(True)
        layout.addWidget(self.quality_label)
        
        layout.addStretch()
        
    def update_model(self, model: str):
        spec = MODEL_SPECS.get(model, MODEL_SPECS['small'])
        tier = spec['tier']
        colors = TIER_COLORS[tier]
        
        self.header.setText(f"{spec['emoji']} {model.upper()}")
        self.header.setStyleSheet(f"color: {colors[0]};")
        
        self.desc.setText(spec['desc'])
        
        self.tier_badge.setText(TIER_NAMES[tier])
        self.tier_badge.setStyleSheet(f"""
            background: {colors[0]}33;
            color: {colors[0]};
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: bold;
        """)
        
        # Architecture
        params = spec['params']
        if params >= 1000:
            self.params_label.setText(f"~{params/1000:.1f}B")
        else:
            self.params_label.setText(f"~{params}M")
        self.dim_label.setText(str(spec['dim']))
        self.layers_label.setText(str(spec['layers']))
        self.heads_label.setText(str(spec['heads']))
        self.ctx_label.setText(f"{spec['ctx']:,}")
        
        # Hardware
        ram_mb = spec['ram']
        vram_mb = spec['vram']
        
        if ram_mb >= 1024:
            self.ram_label.setText(f"{ram_mb/1024:.0f} GB")
        else:
            self.ram_label.setText(f"{ram_mb} MB")
            
        if vram_mb == 0:
            self.vram_label.setText("CPU only")
        elif vram_mb >= 1024:
            self.vram_label.setText(f"{vram_mb/1024:.0f} GB")
        else:
            self.vram_label.setText(f"{vram_mb} MB")
        
        # Bars (relative to 32GB RAM, 24GB VRAM as "normal high-end")
        self.ram_bar.setValue(min(100, int(ram_mb / 327.68)))  # 32GB = 100%
        self.vram_bar.setValue(min(100, int(vram_mb / 245.76)))  # 24GB = 100%
        
        # Color bars based on usage
        ram_pct = ram_mb / 327.68
        vram_pct = vram_mb / 245.76
        
        ram_color = '#69db7c' if ram_pct < 50 else '#ffd43b' if ram_pct < 80 else '#ff6b6b'
        vram_color = '#69db7c' if vram_pct < 50 else '#ffd43b' if vram_pct < 80 else '#ff6b6b'
        
        self.ram_bar.setStyleSheet(f"""
            QProgressBar {{ background: #30363d; border-radius: 4px; }}
            QProgressBar::chunk {{ background: {ram_color}; border-radius: 4px; }}
        """)
        self.vram_bar.setStyleSheet(f"""
            QProgressBar {{ background: #30363d; border-radius: 4px; }}
            QProgressBar::chunk {{ background: {vram_color}; border-radius: 4px; }}
        """)
        
        # Performance estimates
        perf_estimates = {
            'nano': ('~200 tok/s on CPU', 'Testing, embedded demos'),
            'micro': ('~150 tok/s on CPU', 'IoT responses, simple queries'),
            'tiny': ('~100 tok/s on Pi 4', 'Edge AI, basic chat'),
            'mini': ('~80 tok/s on mobile', 'Mobile apps, quick responses'),
            'small': ('~50 tok/s on RTX 3060', 'Conversations, learning, prototyping'),
            'medium': ('~35 tok/s on RTX 3060', 'Good conversations, reasoning'),
            'base': ('~25 tok/s on RTX 3060', 'Quality responses, code help'),
            'large': ('~15 tok/s on RTX 3080', 'High quality, complex tasks'),
            'xl': ('~8 tok/s on RTX 4090', 'Excellent quality, research'),
            'xxl': ('~4 tok/s on RTX 4090', 'Near-commercial quality'),
            'huge': ('~2 tok/s on 2x 4090', 'Production-grade quality'),
            'giant': ('~1 tok/s on A100', 'Commercial deployment'),
            'colossal': ('~0.5 tok/s on 2x A100', 'Enterprise applications'),
            'titan': ('~0.2 tok/s on cluster', 'State-of-the-art research'),
            'omega': ('Research scale', 'Frontier capabilities'),
        }
        
        perf, quality = perf_estimates.get(model, ('Unknown', 'Unknown'))
        self.perf_label.setText(perf)
        self.quality_label.setText(f"Good for: {quality}")


class ScalingTab(QWidget):
    """Complete model scaling interface - clean horizontal layout."""
    
    model_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.selected_model = 'small'
        self.setup_ui()
        
    def setup_ui(self):
        # Main vertical layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top section: Pyramid (left) + Specs (right) side by side
        top_layout = QHBoxLayout()
        top_layout.setSpacing(12)
        
        # Pyramid - half size
        self.pyramid = PyramidWidget()
        self.pyramid.model_clicked.connect(self.on_model_selected)
        top_layout.addWidget(self.pyramid, stretch=1)
        
        # Specs panel - equal size
        self.specs = SpecsPanel()
        top_layout.addWidget(self.specs, stretch=1)
        
        main_layout.addLayout(top_layout, stretch=1)
        
        # Bottom section: Action buttons in a horizontal bar
        bottom_frame = QFrame()
        bottom_frame.setStyleSheet("""
            QFrame {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
            }
        """)
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(12, 8, 12, 8)
        bottom_layout.setSpacing(12)
        
        # Quick model selector dropdown
        from .shared_components import NoScrollComboBox
        self.model_combo = NoScrollComboBox()
        self.model_combo.setFixedWidth(120)
        for model in MODEL_ORDER:
            colors = TIER_COLORS[MODEL_SPECS[model]['tier']]
            self.model_combo.addItem(model.upper())
        self.model_combo.setCurrentText('SMALL')
        self.model_combo.currentTextChanged.connect(self._on_combo_change)
        self.model_combo.setStyleSheet("""
            QComboBox {
                background: #21262d;
                color: #ffffff;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 4px 8px;
            }
        """)
        bottom_layout.addWidget(QLabel("Size:"))
        bottom_layout.addWidget(self.model_combo)
        
        bottom_layout.addStretch()
        
        # Create button
        self.create_btn = QPushButton("Create Model")
        self.create_btn.setStyleSheet("""
            QPushButton {
                background: #238636;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background: #2ea043; }
        """)
        self.create_btn.clicked.connect(self.create_model)
        bottom_layout.addWidget(self.create_btn)
        
        # Benchmark button
        self.benchmark_btn = QPushButton("Benchmark")
        self.benchmark_btn.setStyleSheet("""
            QPushButton {
                background: #30363d;
                color: #ffffff;
                border: 1px solid #484f58;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover { background: #484f58; }
        """)
        self.benchmark_btn.clicked.connect(self.run_benchmark)
        bottom_layout.addWidget(self.benchmark_btn)
        
        main_layout.addWidget(bottom_frame)
        
        # Initial update
        self.specs.update_model('small')
    
    def _on_combo_change(self, text):
        model = text.lower()
        if model in MODEL_SPECS:
            self.selected_model = model
            self.pyramid.set_selected(model)
            self.specs.update_model(model)
        
    def on_model_selected(self, model: str):
        self.selected_model = model
        self.specs.update_model(model)
        # Sync combo box
        if hasattr(self, 'model_combo'):
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentText(model.upper())
            self.model_combo.blockSignals(False)
        
    def create_model(self):
        model = self.selected_model
        spec = MODEL_SPECS[model]
        
        params_str = f"{spec['params']/1000:.1f}B" if spec['params'] >= 1000 else f"{spec['params']}M"
        ram_str = f"{spec['ram']/1024:.1f}GB" if spec['ram'] >= 1024 else f"{spec['ram']}MB"
        vram_str = f"{spec['vram']/1024:.1f}GB" if spec['vram'] >= 1024 else "CPU only" if spec['vram'] == 0 else f"{spec['vram']}MB"
        
        # Get model name from user
        from PyQt5.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self,
            f"Create {model.upper()} Model",
            f"Enter a name for your new {model.upper()} model:\n\n"
            f"Parameters: ~{params_str}\n"
            f"RAM Required: {ram_str}\n"
            f"VRAM Required: {vram_str}",
            text=f"my_{model}_model"
        )
        
        if ok and name:
            # Clean up the name
            name = name.strip().replace(' ', '_')
            
            # Check if we have access to the registry
            if self.main_window and hasattr(self.main_window, 'registry'):
                try:
                    self.main_window.registry.create_model(
                        name,
                        size=model,
                        vocab_size=32000
                    )
                    
                    # Emit signal for any listeners
                    self.model_changed.emit(model)
                    
                    # Update window title and load the model
                    self.main_window.current_model_name = name
                    self.main_window.setWindowTitle(f"ForgeAI - {name}")
                    
                    # Update model status button if exists
                    if hasattr(self.main_window, 'model_status_btn'):
                        self.main_window.model_status_btn.setText(f"Model: {name}  v")
                    
                    QMessageBox.information(
                        self,
                        "Model Created",
                        f"{model.upper()} model '{name}' created!\n\n"
                        "Next steps:\n"
                        "1. Go to the Train tab\n"
                        "2. Add your training data\n"
                        "3. Start training"
                    )
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Error Creating Model",
                        f"Failed to create model: {e}"
                    )
            else:
                QMessageBox.warning(
                    self,
                    "No Registry",
                    "Model registry not available.\n"
                    "Please restart the application."
                )
            
    def run_benchmark(self):
        """Run performance benchmark on selected model size."""
        model = self.selected_model
        
        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            "Run Benchmark",
            f"Run benchmark for {model.upper()} model?\n\n"
            "This will:\n"
            "- Test forward pass speed\n"
            "- Test generation speed\n"
            "- Check memory usage\n\n"
            "This may take a few minutes.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            import torch
            import time
            from forge_ai.core.model import create_model
            from forge_ai.core.model_config import get_model_config
            
            # Setup device
            if torch.cuda.is_available():
                device = torch.device("cuda")
                device_name = torch.cuda.get_device_name(0)
            else:
                device = torch.device("cpu")
                device_name = "CPU"
            
            # Create progress dialog
            from PyQt5.QtWidgets import QProgressDialog
            from PyQt5.QtCore import Qt
            progress = QProgressDialog("Running benchmark...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setWindowTitle("Benchmark")
            progress.show()
            
            results = []
            results.append(f"Device: {device_name}")
            results.append(f"Model: {model.upper()}")
            results.append("-" * 40)
            
            progress.setValue(10)
            progress.setLabelText("Creating model...")
            
            # Create model
            config = get_model_config(model)
            test_model = create_model(size=model, vocab_size=32000)
            test_model.to(device)
            test_model.eval()
            
            params = sum(p.numel() for p in test_model.parameters())
            results.append(f"Parameters: {params:,}")
            
            progress.setValue(30)
            progress.setLabelText("Testing forward pass...")
            
            # Benchmark forward pass
            seq_len = 128
            input_ids = torch.randint(0, 32000, (1, seq_len), device=device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = test_model(input_ids, use_cache=False)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            # Measure
            times = []
            with torch.no_grad():
                for _ in range(5):
                    start = time.perf_counter()
                    _ = test_model(input_ids, use_cache=False)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)
            
            avg_forward = sum(times) / len(times)
            tokens_per_sec = seq_len / avg_forward
            results.append(f"\nForward Pass ({seq_len} tokens):")
            results.append(f"  Time: {avg_forward*1000:.1f} ms")
            results.append(f"  Speed: {tokens_per_sec:.0f} tokens/sec")
            
            progress.setValue(60)
            progress.setLabelText("Testing generation...")
            
            # Benchmark generation
            gen_len = 20
            gen_input = torch.randint(0, 32000, (1, 10), device=device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(2):
                    _ = test_model.generate(gen_input.clone(), max_new_tokens=gen_len, use_cache=True)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            # Measure
            times = []
            with torch.no_grad():
                for _ in range(3):
                    start = time.perf_counter()
                    _ = test_model.generate(gen_input.clone(), max_new_tokens=gen_len, use_cache=True)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)
            
            avg_gen = sum(times) / len(times)
            gen_tokens_per_sec = gen_len / avg_gen
            results.append(f"\nGeneration ({gen_len} tokens):")
            results.append(f"  Time: {avg_gen*1000:.1f} ms")
            results.append(f"  Speed: {gen_tokens_per_sec:.1f} tokens/sec")
            results.append(f"  Per token: {(avg_gen/gen_len)*1000:.1f} ms")
            
            progress.setValue(80)
            progress.setLabelText("Checking memory...")
            
            # Memory usage
            results.append(f"\nMemory Usage:")
            if device.type == "cuda":
                mem_alloc = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                results.append(f"  GPU Allocated: {mem_alloc:.0f} MB")
                results.append(f"  GPU Reserved: {mem_reserved:.0f} MB")
            else:
                try:
                    import psutil
                    process = psutil.Process()
                    ram = process.memory_info().rss / 1024**2
                    results.append(f"  RAM Used: {ram:.0f} MB")
                except:
                    results.append("  RAM: Unable to measure")
            
            # Cleanup
            del test_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            progress.setValue(100)
            progress.close()
            
            # Show results
            QMessageBox.information(
                self,
                "Benchmark Results",
                "\n".join(results)
            )
            
        except Exception as e:
            progress.close() if 'progress' in dir() else None
            QMessageBox.warning(
                self,
                "Benchmark Error",
                f"Benchmark failed: {e}"
            )


def create_scaling_tab(window) -> QWidget:
    """Factory function to create scaling tab."""
    return ScalingTab(window)
