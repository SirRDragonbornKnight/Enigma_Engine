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
    """Panel showing detailed specs for selected model."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QFrame {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
            }
        """)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header with model name and emoji
        self.header = QLabel("SMALL")
        self.header.setFont(QFont('Segoe UI', 24, FontBold))
        self.header.setStyleSheet("color: #ffd43b;")
        layout.addWidget(self.header)
        
        # Description
        self.desc = QLabel("Laptops, entry GPU")
        self.desc.setStyleSheet("color: #8b949e; font-size: 12px;")
        layout.addWidget(self.desc)
        
        # Tier badge
        self.tier_badge = QLabel("Consumer Tier")
        self.tier_badge.setStyleSheet("""
            background: #ffd43b33;
            color: #ffd43b;
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: bold;
        """)
        self.tier_badge.setFixedWidth(160)
        layout.addWidget(self.tier_badge)
        
        layout.addSpacing(10)
        
        # Architecture specs
        arch_group = QGroupBox("Architecture")
        arch_group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #30363d;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
            }
        """)
        arch_layout = QGridLayout(arch_group)
        
        self.params_label = QLabel("~27M")
        self.dim_label = QLabel("512")
        self.layers_label = QLabel("8")
        self.heads_label = QLabel("8")
        self.ctx_label = QLabel("1024")
        
        specs = [
            ("Parameters:", self.params_label),
            ("Embedding Dim:", self.dim_label),
            ("Layers:", self.layers_label),
            ("Attention Heads:", self.heads_label),
            ("Context Length:", self.ctx_label),
        ]
        
        for i, (label, value_widget) in enumerate(specs):
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #8b949e;")
            value_widget.setStyleSheet("color: #ffffff; font-weight: bold;")
            arch_layout.addWidget(lbl, i, 0)
            arch_layout.addWidget(value_widget, i, 1)
            
        layout.addWidget(arch_group)
        
        # Hardware requirements
        hw_group = QGroupBox(" Hardware Requirements")
        hw_group.setStyleSheet(arch_group.styleSheet())
        hw_layout = QGridLayout(hw_group)
        
        self.ram_label = QLabel("4 GB")
        self.vram_label = QLabel("2 GB")
        
        hw_specs = [
            ("System RAM:", self.ram_label),
            ("GPU VRAM:", self.vram_label),
        ]
        
        for i, (label, value_widget) in enumerate(hw_specs):
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #8b949e;")
            value_widget.setStyleSheet("color: #ffffff; font-weight: bold;")
            hw_layout.addWidget(lbl, i, 0)
            hw_layout.addWidget(value_widget, i, 1)
            
        # RAM bar
        self.ram_bar = QProgressBar()
        self.ram_bar.setMaximum(100)
        self.ram_bar.setValue(25)
        self.ram_bar.setTextVisible(False)
        self.ram_bar.setFixedHeight(8)
        self.ram_bar.setStyleSheet("""
            QProgressBar {
                background: #30363d;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: #69db7c;
                border-radius: 4px;
            }
        """)
        hw_layout.addWidget(QLabel("RAM Usage:"), 2, 0)
        hw_layout.addWidget(self.ram_bar, 2, 1)
        
        # VRAM bar
        self.vram_bar = QProgressBar()
        self.vram_bar.setMaximum(100)
        self.vram_bar.setValue(15)
        self.vram_bar.setTextVisible(False)
        self.vram_bar.setFixedHeight(8)
        self.vram_bar.setStyleSheet(self.ram_bar.styleSheet())
        hw_layout.addWidget(QLabel("VRAM Usage:"), 3, 0)
        hw_layout.addWidget(self.vram_bar, 3, 1)
        
        layout.addWidget(hw_group)
        
        # Performance estimate
        perf_group = QGroupBox(" Performance Estimate")
        perf_group.setStyleSheet(arch_group.styleSheet())
        perf_layout = QVBoxLayout(perf_group)
        
        self.perf_label = QLabel("~50 tokens/sec on RTX 3060")
        self.perf_label.setStyleSheet("color: #69db7c; font-size: 14px;")
        perf_layout.addWidget(self.perf_label)
        
        self.quality_label = QLabel("Good for: Conversations, basic tasks, learning")
        self.quality_label.setStyleSheet("color: #8b949e; font-size: 13px;")
        self.quality_label.setWordWrap(True)
        perf_layout.addWidget(self.quality_label)
        
        layout.addWidget(perf_group)
        
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


class CompareWidget(QFrame):
    """Side-by-side model comparison with more details."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QFrame {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
            }
        """)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(6)
        
        title = QLabel("Quick Compare")
        title.setFont(QFont('Segoe UI', 14, FontBold))
        title.setStyleSheet("color: #ffffff;")
        layout.addWidget(title)
        
        # Comparison table header
        header = QHBoxLayout()
        header.setSpacing(8)
        for col in ["Model", "Params", "Context", "RAM", "Speed"]:
            lbl = QLabel(col)
            lbl.setStyleSheet("color: #8b949e; font-weight: bold; font-size: 14px;")
            lbl.setMinimumWidth(60)
            header.addWidget(lbl, 1)
        layout.addLayout(header)
        
        # More comprehensive comparison rows
        comparisons = [
            ('nano',   '1M',    '256',   '256MB',  ''),
            ('micro',  '2M',    '384',   '512MB',  ''),
            ('tiny',   '5M',    '512',   '1GB',    ''),
            ('mini',   '10M',   '512',   '2GB',    ''),
            ('small',  '27M',   '1K',    '4GB',    ''),
            ('medium', '85M',   '2K',    '8GB',    ''),
            ('base',   '125M',  '2K',    '12GB',   ''),
            ('large',  '200M',  '4K',    '16GB',   ''),
            ('xl',     '600M',  '4K',    '32GB',   ''),
            ('xxl',    '1.5B',  '8K',    '64GB',   'Slow'),
        ]
        
        self.compare_rows = []
        for model, params, ctx, ram, speed in comparisons:
            row = QHBoxLayout()
            row.setSpacing(8)
            
            colors = TIER_COLORS[MODEL_SPECS[model]['tier']]
            
            name_lbl = QLabel(model.upper())
            name_lbl.setStyleSheet(f"color: {colors[0]}; font-weight: bold; font-size: 14px;")
            name_lbl.setMinimumWidth(60)
            row.addWidget(name_lbl, 1)
            
            params_lbl = QLabel(params)
            params_lbl.setStyleSheet("color: #ffffff; font-size: 14px;")
            params_lbl.setMinimumWidth(60)
            row.addWidget(params_lbl, 1)
            
            ctx_lbl = QLabel(ctx)
            ctx_lbl.setStyleSheet("color: #8b949e; font-size: 14px;")
            ctx_lbl.setMinimumWidth(60)
            row.addWidget(ctx_lbl, 1)
            
            ram_lbl = QLabel(ram)
            ram_lbl.setStyleSheet("color: #74c0fc; font-size: 14px;")
            ram_lbl.setMinimumWidth(60)
            row.addWidget(ram_lbl, 1)
            
            speed_lbl = QLabel(speed)
            speed_lbl.setStyleSheet("font-size: 14px;")
            speed_lbl.setMinimumWidth(60)
            row.addWidget(speed_lbl, 1)
            
            layout.addLayout(row)
            self.compare_rows.append((model, row))
        
        layout.addStretch()


class ScalingTab(QWidget):
    """Complete model scaling interface."""
    
    model_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent  # Reference to EnhancedMainWindow
        self.selected_model = 'small'
        self.setup_ui()
        
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Left side - Pyramid visualization (takes more space)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        
        self.pyramid = PyramidWidget()
        self.pyramid.model_clicked.connect(self.on_model_selected)
        left_panel.addWidget(self.pyramid, stretch=2)  # Pyramid gets more space
        
        # Quick compare below pyramid (scales with window)
        self.compare = CompareWidget()
        left_panel.addWidget(self.compare, stretch=1)
        
        layout.addLayout(left_panel, stretch=2)
        
        # Right side - Specs and actions
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        
        self.specs = SpecsPanel()
        right_panel.addWidget(self.specs, stretch=1)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        self.create_btn = QPushButton("Create This Model")
        self.create_btn.setToolTip("Create a new model with selected configuration")
        self.create_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #69db7c, stop:1 #51cf66);
                color: #1e1e2e;
                border: none;
                border-radius: 8px;
                padding: 15px 25px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #8ce99a, stop:1 #69db7c);
            }
        """)
        self.create_btn.clicked.connect(self.create_model)
        btn_layout.addWidget(self.create_btn)
        
        self.benchmark_btn = QPushButton("Benchmark")
        self.benchmark_btn.setToolTip("Run performance benchmark on selected model")
        self.benchmark_btn.setStyleSheet("""
            QPushButton {
                background: #30363d;
                color: #ffffff;
                border: 1px solid #484f58;
                border-radius: 8px;
                padding: 15px 20px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #484f58;
            }
        """)
        self.benchmark_btn.clicked.connect(self.run_benchmark)
        btn_layout.addWidget(self.benchmark_btn)
        
        right_panel.addLayout(btn_layout)
        
        layout.addLayout(right_panel, stretch=1)
        
        # Initial update
        self.specs.update_model('small')
        
    def on_model_selected(self, model: str):
        self.selected_model = model
        self.specs.update_model(model)
        
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
        QMessageBox.information(
            self,
            "Benchmark",
            " Benchmark feature coming soon!\n\n"
            "This will test your hardware and\n"
            "recommend the best model size for you."
        )


def create_scaling_tab(window) -> QWidget:
    """Factory function to create scaling tab."""
    return ScalingTab(window)
