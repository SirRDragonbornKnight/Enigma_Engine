"""
Model Scaling Tab - Visualize and manage model sizes from nano to omega
"""

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea,
        QLabel, QPushButton, QFrame, QGroupBox, QMessageBox, QTextEdit
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


# Model tier colors
TIER_COLORS = {
    'embedded': '#e74c3c',    # Red - tiny/constrained
    'edge': '#e67e22',        # Orange - RPi/mobile
    'consumer': '#f1c40f',    # Yellow - typical GPU
    'prosumer': '#2ecc71',    # Green - high-end GPU
    'server': '#3498db',      # Blue - multi-GPU
    'datacenter': '#9b59b6',  # Purple - cloud
    'ultimate': '#1abc9c',    # Teal - maximum
}


class ModelScaleWidget(QFrame):
    """Visual representation of model scale."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setStyleSheet("background-color: #1e1e2e; border-radius: 10px;")
        
        self.models = [
            ('nano', 1, 'embedded'),
            ('micro', 2, 'embedded'),
            ('tiny', 5, 'edge'),
            ('mini', 10, 'edge'),
            ('small', 27, 'consumer'),
            ('medium', 85, 'consumer'),
            ('base', 125, 'consumer'),
            ('large', 200, 'prosumer'),
            ('xl', 600, 'prosumer'),
            ('xxl', 1500, 'server'),
            ('huge', 3000, 'server'),
            ('giant', 7000, 'datacenter'),
            ('colossal', 13000, 'datacenter'),
            ('titan', 30000, 'ultimate'),
            ('omega', 70000, 'ultimate'),
        ]
        
        self.selected_model = 'small'
        self.hover_model = None
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Background
        painter.fillRect(0, 0, w, h, QColor('#1e1e2e'))
        
        # Draw scale visualization
        margin = 40
        scale_width = w - 2 * margin
        bar_height = 30
        
        # Draw main scale bar
        bar_y = h // 2 - bar_height // 2
        
        # Draw tier segments
        tier_positions = [0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
        tier_names = ['Embedded', 'Edge', 'Consumer', 'Prosumer', 'Server', 'Datacenter', 'Ultimate']
        tier_color_keys = ['embedded', 'edge', 'consumer', 'prosumer', 'server', 'datacenter', 'ultimate']
        
        for i in range(len(tier_positions) - 1):
            x1 = margin + int(tier_positions[i] * scale_width)
            x2 = margin + int(tier_positions[i + 1] * scale_width)
            color = QColor(TIER_COLORS[tier_color_keys[i]])
            
            painter.fillRect(x1, bar_y, x2 - x1, bar_height, color)
            
            # Tier label
            painter.setPen(QPen(QColor('white')))
            painter.setFont(QFont('Arial', 8))
            painter.drawText(x1 + 5, bar_y + bar_height + 15, tier_names[i])
        
        # Draw model markers
        for i, (name, size, tier) in enumerate(self.models):
            # Calculate position based on log scale
            import math
            log_pos = math.log10(size) / math.log10(100000)  # Normalize to 0-1
            x = margin + int(log_pos * scale_width)
            
            # Draw marker
            marker_y = bar_y - 10
            marker_size = 8 if name == self.selected_model else 5
            
            color = QColor(TIER_COLORS[tier])
            if name == self.selected_model:
                color = QColor('#ffffff')
            elif name == self.hover_model:
                color = color.lighter(130)
            
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor('white'), 1))
            painter.drawEllipse(x - marker_size//2, marker_y - marker_size//2, marker_size, marker_size)
            
            # Draw label for selected or hovered
            if name == self.selected_model or name == self.hover_model:
                painter.setFont(QFont('Arial', 9, QFont.Bold))
                painter.drawText(x - 20, marker_y - 15, f"{name.upper()}")
                painter.setFont(QFont('Arial', 8))
                painter.drawText(x - 20, marker_y - 3, f"~{size}M params")
        
        # Title
        painter.setPen(QPen(QColor('#cdd6f4')))
        painter.setFont(QFont('Arial', 12, QFont.Bold))
        painter.drawText(margin, 25, "Model Scale Spectrum")
        
        painter.end()
    
    def set_model(self, name: str):
        self.selected_model = name
        self.update()


class ScalingTab(QWidget):
    """Tab for understanding and configuring model scaling."""
    
    model_changed = pyqtSignal(str)  # Emitted when model size changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Model Scaling")
        title.setFont(QFont('Arial', 16, QFont.Bold))
        layout.addWidget(title)
        
        subtitle = QLabel("From Raspberry Pi to Datacenter - Enigma scales with your hardware")
        subtitle.setStyleSheet("color: #888;")
        layout.addWidget(subtitle)
        
        # Visual scale widget
        self.scale_widget = ModelScaleWidget()
        layout.addWidget(self.scale_widget)
        
        # Model cards grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(10)
        
        models = [
            ('nano', '~1M', 'Embedded', 'Microcontrollers, basic responses', '128', '4', '256'),
            ('micro', '~2M', 'Embedded', 'IoT devices, simple tasks', '192', '4', '384'),
            ('tiny', '~5M', 'Edge', 'Raspberry Pi, edge devices', '256', '6', '512'),
            ('mini', '~10M', 'Edge', 'Mobile, low-power devices', '384', '6', '512'),
            ('small', '~27M', 'Consumer', 'Entry GPU, good learning', '512', '8', '1024'),
            ('medium', '~85M', 'Consumer', 'Mid-range GPU, solid results', '768', '12', '2048'),
            ('base', '~125M', 'Consumer', 'Good GPU, versatile', '896', '14', '2048'),
            ('large', '~200M', 'Prosumer', 'RTX 3080+, high quality', '1024', '16', '4096'),
            ('xl', '~600M', 'Prosumer', 'RTX 4090, excellent results', '1536', '24', '4096'),
            ('xxl', '~1.5B', 'Server', 'Multi-GPU, near-production', '2048', '32', '8192'),
            ('huge', '~3B', 'Server', 'Server GPU, production ready', '2560', '40', '8192'),
            ('giant', '~7B', 'Datacenter', 'Multi-node, commercial grade', '4096', '32', '8192'),
            ('colossal', '~13B', 'Datacenter', 'Distributed, competitive', '4096', '48', '16384'),
            ('titan', '~30B', 'Ultimate', 'Full datacenter, state-of-art', '6144', '48', '16384'),
            ('omega', '~70B+', 'Ultimate', 'Cluster, research frontier', '8192', '64', '32768'),
        ]
        
        row, col = 0, 0
        self.model_buttons = {}
        
        for name, params, tier, desc, dim, layers, seq_len in models:
            card = self._create_model_card(name, params, tier, desc, dim, layers, seq_len)
            grid_layout.addWidget(card, row, col)
            col += 1
            if col >= 3:
                col = 0
                row += 1
        
        scroll.setWidget(grid_widget)
        layout.addWidget(scroll, stretch=1)
        
        # Hardware requirements section
        hw_group = QGroupBox("Hardware Recommendations")
        hw_layout = QVBoxLayout(hw_group)
        
        self.hw_label = QTextEdit()
        self.hw_label.setReadOnly(True)
        self.hw_label.setMaximumHeight(100)
        self.hw_label.setHtml(self._get_hw_requirements('small'))
        hw_layout.addWidget(self.hw_label)
        
        layout.addWidget(hw_group)
        
        # Actions
        actions = QHBoxLayout()
        
        self.current_label = QLabel("Current Model: small (~27M params)")
        self.current_label.setFont(QFont('Arial', 10, QFont.Bold))
        actions.addWidget(self.current_label)
        
        actions.addStretch()
        
        self.apply_btn = QPushButton("Apply Model Size")
        self.apply_btn.clicked.connect(self._apply_model)
        actions.addWidget(self.apply_btn)
        
        layout.addLayout(actions)
    
    def _create_model_card(self, name: str, params: str, tier: str, desc: str, 
                          dim: str, layers: str, seq_len: str) -> QFrame:
        card = QFrame()
        card.setFrameStyle(QFrame.Box | QFrame.Raised)
        card.setLineWidth(1)
        
        tier_lower = tier.lower()
        color = TIER_COLORS.get(tier_lower, '#666')
        card.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {color};
                border-radius: 8px;
                padding: 10px;
            }}
            QFrame:hover {{
                border: 2px solid white;
                background-color: {color}33;
            }}
        """)
        
        layout = QVBoxLayout(card)
        
        # Header
        header = QHBoxLayout()
        name_label = QLabel(name.upper())
        name_label.setFont(QFont('Arial', 12, QFont.Bold))
        name_label.setStyleSheet(f"color: {color};")
        header.addWidget(name_label)
        
        params_label = QLabel(params)
        params_label.setStyleSheet("color: #888;")
        header.addWidget(params_label)
        
        layout.addLayout(header)
        
        # Description
        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 10px; color: #aaa;")
        layout.addWidget(desc_label)
        
        # Specs
        specs = QLabel(f"dim={dim} • layers={layers} • seq={seq_len}")
        specs.setStyleSheet("font-size: 9px; color: #666; font-family: monospace;")
        layout.addWidget(specs)
        
        # Select button
        btn = QPushButton("Select")
        btn.clicked.connect(lambda: self._select_model(name))
        layout.addWidget(btn)
        
        self.model_buttons[name] = btn
        
        return card
    
    def _select_model(self, name: str):
        self.scale_widget.set_model(name)
        self.current_label.setText(f"Selected: {name.upper()}")
        self.hw_label.setHtml(self._get_hw_requirements(name))
        
        # Update button states
        for model_name, btn in self.model_buttons.items():
            if model_name == name:
                btn.setStyleSheet("background-color: #2ecc71;")
                btn.setText("Selected ✓")
            else:
                btn.setStyleSheet("")
                btn.setText("Select")
    
    def _get_hw_requirements(self, model: str) -> str:
        requirements = {
            'nano': ('256MB RAM', 'Any CPU', 'None', 'Microcontrollers, Arduino'),
            'micro': ('512MB RAM', 'Any CPU', 'None', 'IoT, ESP32'),
            'tiny': ('1GB RAM', 'ARM/x86', 'None', 'Raspberry Pi 3+'),
            'mini': ('2GB RAM', 'Quad-core', 'None', 'Raspberry Pi 4, Mobile'),
            'small': ('4GB RAM', 'Modern CPU', '2GB VRAM', 'Laptop, Entry GPU'),
            'medium': ('8GB RAM', 'i5/Ryzen 5+', '4GB VRAM', 'Desktop, GTX 1650+'),
            'base': ('12GB RAM', 'i7/Ryzen 7', '6GB VRAM', 'Gaming PC, RTX 3060'),
            'large': ('16GB RAM', 'i7/Ryzen 7', '8GB VRAM', 'Workstation, RTX 3070+'),
            'xl': ('32GB RAM', 'High-end CPU', '12GB VRAM', 'Creator PC, RTX 3080+'),
            'xxl': ('64GB RAM', 'Server CPU', '24GB VRAM', 'Workstation, RTX 4090'),
            'huge': ('128GB RAM', 'Server/Multi-CPU', '48GB VRAM', 'Server, 2x RTX 4090'),
            'giant': ('256GB RAM', 'Multi-CPU', '80GB+ VRAM', 'Server, A100/H100'),
            'colossal': ('512GB RAM', 'Server Cluster', '160GB+ VRAM', 'Multi-node, 2x A100'),
            'titan': ('1TB RAM', 'HPC Cluster', '320GB+ VRAM', 'Cluster, 4+ A100/H100'),
            'omega': ('2TB+ RAM', 'Datacenter', '640GB+ VRAM', 'Full rack, 8+ H100'),
        }
        
        if model in requirements:
            ram, cpu, vram, systems = requirements[model]
            return f"""
            <table style='width:100%; color:#cdd6f4;'>
                <tr><td><b>RAM:</b></td><td>{ram}</td><td><b>CPU:</b></td><td>{cpu}</td></tr>
                <tr><td><b>VRAM:</b></td><td>{vram}</td><td><b>Typical Systems:</b></td><td>{systems}</td></tr>
            </table>
            """
        return "<p>Unknown model</p>"
    
    def _apply_model(self):
        model = self.scale_widget.selected_model
        reply = QMessageBox.question(
            self,
            "Change Model Size",
            f"Change to {model.upper()} model?\n\n"
            "This will affect memory usage and performance.\n"
            "You may need to train a new model or convert an existing one.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.model_changed.emit(model)
            QMessageBox.information(
                self, 
                "Model Changed",
                f"Model size set to {model.upper()}.\n\n"
                "Go to the Train tab to create a model with this configuration."
            )


def create_scaling_tab(window) -> QWidget:
    """Factory function to create scaling tab."""
    return ScalingTab(window)


if not HAS_PYQT:
    class ScalingTab:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyQt5 is required for the Scaling Tab")
