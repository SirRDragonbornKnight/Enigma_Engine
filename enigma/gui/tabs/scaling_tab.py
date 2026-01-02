"""
Model Scaling Tab - Visualize and manage model sizes from nano to omega
"""

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea,
        QLabel, QPushButton, QFrame, QGroupBox, QMessageBox, QTextEdit
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush, QLinearGradient
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


# Model tier colors - softer, more modern palette
TIER_COLORS = {
    'embedded': '#ff6b6b',    # Coral red
    'edge': '#ffa94d',        # Orange
    'consumer': '#ffd43b',    # Yellow
    'prosumer': '#69db7c',    # Green
    'server': '#74c0fc',      # Light blue
    'datacenter': '#b197fc',  # Purple
    'ultimate': '#63e6be',    # Teal
}

TIER_BG_COLORS = {
    'embedded': '#ff6b6b22',
    'edge': '#ffa94d22',
    'consumer': '#ffd43b22',
    'prosumer': '#69db7c22',
    'server': '#74c0fc22',
    'datacenter': '#b197fc22',
    'ultimate': '#63e6be22',
}


class ModelScaleWidget(QFrame):
    """Visual representation of model scale - improved design."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #1a1a2e);
                border-radius: 12px;
                border: 1px solid #333;
            }
        """)
        
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
        
        margin = 50
        scale_width = w - 2 * margin
        bar_height = 24
        bar_y = h // 2
        
        # Draw tier segments with rounded ends
        tier_positions = [0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
        tier_names = ['Embedded', 'Edge', 'Consumer', 'Prosumer', 'Server', 'Datacenter', 'Ultimate']
        tier_color_keys = ['embedded', 'edge', 'consumer', 'prosumer', 'server', 'datacenter', 'ultimate']
        
        for i in range(len(tier_positions) - 1):
            x1 = margin + int(tier_positions[i] * scale_width)
            x2 = margin + int(tier_positions[i + 1] * scale_width)
            color = QColor(TIER_COLORS[tier_color_keys[i]])
            
            # Draw segment
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            
            if i == 0:  # First segment - rounded left
                painter.drawRoundedRect(x1, bar_y, x2 - x1 + 5, bar_height, 6, 6)
            elif i == len(tier_positions) - 2:  # Last segment - rounded right
                painter.drawRoundedRect(x1 - 5, bar_y, x2 - x1 + 5, bar_height, 6, 6)
            else:
                painter.fillRect(x1, bar_y, x2 - x1, bar_height, color)
            
            # Tier label below
            painter.setPen(QPen(QColor('#888')))
            painter.setFont(QFont('Segoe UI', 8))
            text_width = painter.fontMetrics().horizontalAdvance(tier_names[i])
            text_x = x1 + (x2 - x1 - text_width) // 2
            painter.drawText(text_x, bar_y + bar_height + 18, tier_names[i])
        
        # Draw model markers
        for i, (name, size, tier) in enumerate(self.models):
            import math
            log_pos = math.log10(size) / math.log10(100000)
            x = margin + int(log_pos * scale_width)
            
            marker_y = bar_y - 8
            is_selected = name == self.selected_model
            marker_size = 14 if is_selected else 8
            
            color = QColor('#ffffff') if is_selected else QColor(TIER_COLORS[tier]).lighter(120)
            
            # Draw marker circle
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor('#fff'), 2 if is_selected else 1))
            painter.drawEllipse(x - marker_size//2, marker_y - marker_size//2, marker_size, marker_size)
            
            # Draw label for selected model
            if is_selected:
                painter.setPen(QPen(QColor('#fff')))
                painter.setFont(QFont('Segoe UI', 10, QFont.Bold))
                painter.drawText(x - 30, marker_y - 20, f"{name.upper()}")
                painter.setFont(QFont('Segoe UI', 8))
                painter.setPen(QPen(QColor('#aaa')))
                painter.drawText(x - 30, marker_y - 6, f"~{size}M")
        
        # Title
        painter.setPen(QPen(QColor('#fff')))
        painter.setFont(QFont('Segoe UI', 13, QFont.Bold))
        painter.drawText(margin, 28, "Model Scale Spectrum")
        
        # Subtitle
        painter.setPen(QPen(QColor('#888')))
        painter.setFont(QFont('Segoe UI', 9))
        painter.drawText(margin, 45, "Click a card below to select a model size")
        
        painter.end()
    
    def set_model(self, name: str):
        self.selected_model = name
        self.update()


class ScalingTab(QWidget):
    """Tab for understanding and configuring model scaling."""
    
    model_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Visual scale widget
        self.scale_widget = ModelScaleWidget()
        layout.addWidget(self.scale_widget)
        
        # Model cards in scrollable grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #2a2a3e;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                border-radius: 5px;
                min-height: 30px;
            }
        """)
        
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(12)
        grid_layout.setContentsMargins(4, 4, 4, 4)
        
        models = [
            ('nano', '~1M', 'Embedded', 'Microcontrollers', '128', '4', '256'),
            ('micro', '~2M', 'Embedded', 'IoT devices', '192', '4', '384'),
            ('tiny', '~5M', 'Edge', 'Raspberry Pi', '256', '6', '512'),
            ('mini', '~10M', 'Edge', 'Mobile devices', '384', '6', '512'),
            ('small', '~27M', 'Consumer', 'Entry GPU', '512', '8', '1024'),
            ('medium', '~85M', 'Consumer', 'Mid-range GPU', '768', '12', '2048'),
            ('base', '~125M', 'Consumer', 'Good GPU', '896', '14', '2048'),
            ('large', '~200M', 'Prosumer', 'RTX 3080+', '1024', '16', '4096'),
            ('xl', '~600M', 'Prosumer', 'RTX 4090', '1536', '24', '4096'),
            ('xxl', '~1.5B', 'Server', 'Multi-GPU', '2048', '32', '8192'),
            ('huge', '~3B', 'Server', 'Server GPU', '2560', '40', '8192'),
            ('giant', '~7B', 'Datacenter', 'Multi-node', '4096', '32', '8192'),
            ('colossal', '~13B', 'Datacenter', 'Distributed', '4096', '48', '16384'),
            ('titan', '~30B', 'Ultimate', 'Full datacenter', '6144', '48', '16384'),
            ('omega', '~70B+', 'Ultimate', 'Research frontier', '8192', '64', '32768'),
        ]
        
        row, col = 0, 0
        self.model_buttons = {}
        
        for name, params, tier, desc, dim, layers, seq_len in models:
            card = self._create_model_card(name, params, tier, desc, dim, layers, seq_len)
            grid_layout.addWidget(card, row, col)
            col += 1
            if col >= 5:  # 5 columns for better fit
                col = 0
                row += 1
        
        scroll.setWidget(grid_widget)
        layout.addWidget(scroll, stretch=1)
        
        # Bottom bar with selection info and apply button
        bottom_bar = QFrame()
        bottom_bar.setStyleSheet("""
            QFrame {
                background: #1e1e2e;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        bottom_layout = QHBoxLayout(bottom_bar)
        
        self.current_label = QLabel("Selected: SMALL (~27M params)")
        self.current_label.setFont(QFont('Segoe UI', 11, QFont.Bold))
        self.current_label.setStyleSheet("color: #69db7c;")
        bottom_layout.addWidget(self.current_label)
        
        # Hardware info inline
        self.hw_inline = QLabel("RAM: 4GB • VRAM: 2GB • Laptop/Entry GPU")
        self.hw_inline.setStyleSheet("color: #888; font-size: 10px;")
        bottom_layout.addWidget(self.hw_inline)
        
        bottom_layout.addStretch()
        
        self.apply_btn = QPushButton("✓ Apply Model Size")
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background: #69db7c;
                color: #1e1e2e;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #8ce99a;
            }
        """)
        self.apply_btn.clicked.connect(self._apply_model)
        bottom_layout.addWidget(self.apply_btn)
        
        layout.addWidget(bottom_bar)
    
    def _create_model_card(self, name: str, params: str, tier: str, desc: str, 
                          dim: str, layers: str, seq_len: str) -> QFrame:
        card = QFrame()
        card.setFixedSize(160, 120)
        
        tier_lower = tier.lower()
        color = TIER_COLORS.get(tier_lower, '#666')
        bg_color = TIER_BG_COLORS.get(tier_lower, '#66666622')
        
        card.setStyleSheet(f"""
            QFrame {{
                background: {bg_color};
                border: 2px solid {color}55;
                border-radius: 10px;
            }}
            QFrame:hover {{
                border: 2px solid {color};
                background: {bg_color.replace('22', '44')};
            }}
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        
        # Model name and params on same line
        header = QHBoxLayout()
        name_label = QLabel(name.upper())
        name_label.setFont(QFont('Segoe UI', 11, QFont.Bold))
        name_label.setStyleSheet(f"color: {color};")
        header.addWidget(name_label)
        
        params_label = QLabel(params)
        params_label.setStyleSheet("color: #888; font-size: 9px;")
        header.addStretch()
        header.addWidget(params_label)
        layout.addLayout(header)
        
        # Description
        desc_label = QLabel(desc)
        desc_label.setStyleSheet("color: #ccc; font-size: 9px;")
        layout.addWidget(desc_label)
        
        # Specs in smaller font
        specs = QLabel(f"d{dim} • L{layers}")
        specs.setStyleSheet("color: #666; font-size: 8px; font-family: monospace;")
        layout.addWidget(specs)
        
        layout.addStretch()
        
        # Select button
        btn = QPushButton("Select")
        btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {color};
                border: 1px solid {color};
                border-radius: 4px;
                padding: 4px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background: {color};
                color: #1e1e2e;
            }}
        """)
        btn.clicked.connect(lambda: self._select_model(name))
        layout.addWidget(btn)
        
        self.model_buttons[name] = btn
        
        return card
    
    def _select_model(self, name: str):
        self.scale_widget.set_model(name)
        
        # Get params for display
        params_map = {
            'nano': '~1M', 'micro': '~2M', 'tiny': '~5M', 'mini': '~10M',
            'small': '~27M', 'medium': '~85M', 'base': '~125M', 'large': '~200M',
            'xl': '~600M', 'xxl': '~1.5B', 'huge': '~3B', 'giant': '~7B',
            'colossal': '~13B', 'titan': '~30B', 'omega': '~70B+'
        }
        
        self.current_label.setText(f"Selected: {name.upper()} ({params_map.get(name, '?')} params)")
        self.hw_inline.setText(self._get_hw_short(name))
        
        # Update button states
        for model_name, btn in self.model_buttons.items():
            tier = self._get_tier(model_name)
            color = TIER_COLORS.get(tier, '#666')
            
            if model_name == name:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {color};
                        color: #1e1e2e;
                        border: none;
                        border-radius: 4px;
                        padding: 4px;
                        font-size: 10px;
                        font-weight: bold;
                    }}
                """)
                btn.setText("✓ Selected")
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: transparent;
                        color: {color};
                        border: 1px solid {color};
                        border-radius: 4px;
                        padding: 4px;
                        font-size: 10px;
                    }}
                    QPushButton:hover {{
                        background: {color};
                        color: #1e1e2e;
                    }}
                """)
                btn.setText("Select")
    
    def _get_tier(self, name: str) -> str:
        tier_map = {
            'nano': 'embedded', 'micro': 'embedded',
            'tiny': 'edge', 'mini': 'edge',
            'small': 'consumer', 'medium': 'consumer', 'base': 'consumer',
            'large': 'prosumer', 'xl': 'prosumer',
            'xxl': 'server', 'huge': 'server',
            'giant': 'datacenter', 'colossal': 'datacenter',
            'titan': 'ultimate', 'omega': 'ultimate'
        }
        return tier_map.get(name, 'consumer')
    
    def _get_hw_short(self, model: str) -> str:
        hw = {
            'nano': 'RAM: 256MB • No GPU • Microcontrollers',
            'micro': 'RAM: 512MB • No GPU • IoT/ESP32',
            'tiny': 'RAM: 1GB • No GPU • Raspberry Pi 3+',
            'mini': 'RAM: 2GB • No GPU • RPi 4/Mobile',
            'small': 'RAM: 4GB • VRAM: 2GB • Laptop/Entry GPU',
            'medium': 'RAM: 8GB • VRAM: 4GB • GTX 1650+',
            'base': 'RAM: 12GB • VRAM: 6GB • RTX 3060',
            'large': 'RAM: 16GB • VRAM: 8GB • RTX 3070+',
            'xl': 'RAM: 32GB • VRAM: 12GB • RTX 3080+',
            'xxl': 'RAM: 64GB • VRAM: 24GB • RTX 4090',
            'huge': 'RAM: 128GB • VRAM: 48GB • 2x RTX 4090',
            'giant': 'RAM: 256GB • VRAM: 80GB+ • A100/H100',
            'colossal': 'RAM: 512GB • VRAM: 160GB+ • 2x A100',
            'titan': 'RAM: 1TB • VRAM: 320GB+ • 4+ A100/H100',
            'omega': 'RAM: 2TB+ • VRAM: 640GB+ • 8+ H100',
        }
        return hw.get(model, 'Unknown requirements')
    
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
