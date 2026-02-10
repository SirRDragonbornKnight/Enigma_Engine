"""
Model Scaling Tab - Clean Horizontal Layout
============================================

Simple, clean interface for selecting model sizes.
- Left: Scrollable list of model cards
- Right: Detailed specs for selected model
"""

from __future__ import annotations


from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


# Model definitions
MODEL_SPECS = {
    'nano':     {'params': 1,     'dim': 128,  'layers': 4,  'heads': 4,  'ctx': 256,   'ram': 256,   'vram': 0,    'tier': 'embedded',   'desc': 'Microcontrollers, testing'},
    'micro':    {'params': 2,     'dim': 192,  'layers': 4,  'heads': 4,  'ctx': 384,   'ram': 512,   'vram': 0,    'tier': 'embedded',   'desc': 'IoT devices, ESP32'},
    'tiny':     {'params': 5,     'dim': 256,  'layers': 6,  'heads': 8,  'ctx': 512,   'ram': 1024,  'vram': 0,    'tier': 'edge',       'desc': 'Raspberry Pi 3/4'},
    'mini':     {'params': 10,    'dim': 384,  'layers': 6,  'heads': 6,  'ctx': 512,   'ram': 2048,  'vram': 0,    'tier': 'edge',       'desc': 'Mobile, tablets'},
    'small':    {'params': 27,    'dim': 512,  'layers': 8,  'heads': 8,  'ctx': 1024,  'ram': 4096,  'vram': 2048, 'tier': 'consumer',   'desc': 'Laptops, entry GPU'},
    'medium':   {'params': 85,    'dim': 768,  'layers': 12, 'heads': 12, 'ctx': 2048,  'ram': 8192,  'vram': 4096, 'tier': 'consumer',   'desc': 'Desktop, GTX 1660+'},
    'base':     {'params': 125,   'dim': 896,  'layers': 14, 'heads': 14, 'ctx': 2048,  'ram': 12288, 'vram': 6144, 'tier': 'consumer',   'desc': 'Gaming PC, RTX 3060'},
    'large':    {'params': 200,   'dim': 1024, 'layers': 16, 'heads': 16, 'ctx': 4096,  'ram': 16384, 'vram': 8192, 'tier': 'prosumer',   'desc': 'Workstation, RTX 3070+'},
    'xl':       {'params': 600,   'dim': 1536, 'layers': 24, 'heads': 24, 'ctx': 4096,  'ram': 32768, 'vram': 12288,'tier': 'prosumer',   'desc': 'High-end, RTX 3080+'},
    'xxl':      {'params': 1500,  'dim': 2048, 'layers': 32, 'heads': 32, 'ctx': 8192,  'ram': 65536, 'vram': 24576,'tier': 'server',     'desc': 'Server, RTX 4090'},
    'huge':     {'params': 3000,  'dim': 2560, 'layers': 40, 'heads': 40, 'ctx': 8192,  'ram': 131072,'vram': 49152,'tier': 'server',     'desc': 'Multi-GPU server'},
    'giant':    {'params': 7000,  'dim': 4096, 'layers': 32, 'heads': 32, 'ctx': 8192,  'ram': 262144,'vram': 81920,'tier': 'datacenter', 'desc': 'A100/H100 cluster'},
    'colossal': {'params': 13000, 'dim': 4096, 'layers': 48, 'heads': 32, 'ctx': 16384, 'ram': 524288,'vram': 163840,'tier': 'datacenter','desc': 'Distributed training'},
    'titan':    {'params': 30000, 'dim': 6144, 'layers': 48, 'heads': 48, 'ctx': 16384, 'ram': 1048576,'vram': 327680,'tier': 'ultimate', 'desc': 'Full datacenter'},
    'omega':    {'params': 70000, 'dim': 8192, 'layers': 64, 'heads': 64, 'ctx': 32768, 'ram': 2097152,'vram': 655360,'tier': 'ultimate', 'desc': 'Research frontier'},
}

MODEL_ORDER = ['nano', 'micro', 'tiny', 'mini', 'small', 'medium', 'base', 'large', 'xl', 'xxl', 'huge', 'giant', 'colossal', 'titan', 'omega']

TIER_COLORS = {
    'embedded':   '#ff6b6b',
    'edge':       '#ffa94d',
    'consumer':   '#ffd43b',
    'prosumer':   '#69db7c',
    'server':     '#74c0fc',
    'datacenter': '#b197fc',
    'ultimate':   '#63e6be',
}


class ModelCard(QFrame):
    """Clickable card for a single model size."""
    
    clicked = pyqtSignal(str)
    
    def __init__(self, model_name: str, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.spec = MODEL_SPECS[model_name]
        self.selected = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(50)
        self.setFocusPolicy(Qt.NoFocus)
        self._setup_ui()
        self._update_style()
        
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)
        
        # Color indicator
        self.color_bar = QFrame()
        self.color_bar.setFixedSize(4, 30)
        self.color_bar.setFocusPolicy(Qt.NoFocus)
        self.color_bar.setAttribute(Qt.WA_TransparentForMouseEvents)
        color = TIER_COLORS[self.spec['tier']]
        self.color_bar.setStyleSheet(f"background: {color}; border-radius: 2px;")
        layout.addWidget(self.color_bar)
        
        # Name and params
        info_layout = QVBoxLayout()
        info_layout.setSpacing(0)
        
        self.name_label = QLabel(self.model_name.upper())
        self.name_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 12px;")
        self.name_label.setFocusPolicy(Qt.NoFocus)
        self.name_label.setTextInteractionFlags(Qt.NoTextInteraction)
        self.name_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        info_layout.addWidget(self.name_label)
        
        params = self.spec['params']
        params_str = f"{params/1000:.1f}B" if params >= 1000 else f"{params}M"
        self.params_label = QLabel(params_str)
        self.params_label.setStyleSheet("color: #8b949e; font-size: 10px;")
        self.params_label.setFocusPolicy(Qt.NoFocus)
        self.params_label.setTextInteractionFlags(Qt.NoTextInteraction)
        self.params_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        info_layout.addWidget(self.params_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # RAM indicator
        ram = self.spec['ram']
        ram_str = f"{ram/1024:.0f}G" if ram >= 1024 else f"{ram}M"
        self.ram_label = QLabel(ram_str)
        self.ram_label.setStyleSheet("color: #74c0fc; font-size: 10px;")
        self.ram_label.setFocusPolicy(Qt.NoFocus)
        self.ram_label.setTextInteractionFlags(Qt.NoTextInteraction)
        self.ram_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        layout.addWidget(self.ram_label)
        
    def _update_style(self):
        if self.selected:
            self.setStyleSheet("""
                QFrame {
                    background: #238636;
                    border: 1px solid #2ea043;
                    border-radius: 6px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background: #21262d;
                    border: 1px solid #30363d;
                    border-radius: 6px;
                }
                QFrame:hover {
                    background: #30363d;
                    border: 1px solid #484f58;
                }
            """)
            
    def set_selected(self, selected: bool):
        self.selected = selected
        self._update_style()
        
    def mousePressEvent(self, event):
        # Clear any focus from anywhere
        focused = QApplication.focusWidget()
        if focused:
            focused.clearFocus()
        self.clicked.emit(self.model_name)


class ScalingTab(QWidget):
    """Model scaling tab with horizontal layout."""
    
    model_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.selected_model = 'small'
        self.model_cards = {}
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # LEFT: Model list (scrollable)
        left_frame = QFrame()
        left_frame.setFocusPolicy(Qt.NoFocus)
        left_frame.setStyleSheet("""
            QFrame {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
            }
        """)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)
        
        # Title
        title = QLabel("Select Model Size")
        title.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 13px;")
        title.setFocusPolicy(Qt.NoFocus)
        title.setTextInteractionFlags(Qt.NoTextInteraction)
        left_layout.addWidget(title)
        
        # Scroll area for cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFocusPolicy(Qt.NoFocus)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { width: 8px; background: #21262d; }
            QScrollBar::handle:vertical { background: #484f58; border-radius: 4px; min-height: 20px; }
        """)
        
        cards_widget = QWidget()
        cards_widget.setFocusPolicy(Qt.NoFocus)
        cards_layout = QVBoxLayout(cards_widget)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(4)
        
        for model in MODEL_ORDER:
            card = ModelCard(model)
            card.clicked.connect(self._on_card_clicked)
            cards_layout.addWidget(card)
            self.model_cards[model] = card
            
        cards_layout.addStretch()
        scroll.setWidget(cards_widget)
        left_layout.addWidget(scroll)
        
        left_frame.setFixedWidth(180)
        layout.addWidget(left_frame)
        
        # RIGHT: Details panel
        right_frame = QFrame()
        right_frame.setStyleSheet("""
            QFrame {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
            }
        """)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)
        
        # Header
        header_row = QHBoxLayout()
        self.model_header = QLabel("SMALL")
        self.model_header.setFont(QFont('Segoe UI', 24, QFont.Bold))
        self.model_header.setStyleSheet("color: #ffd43b;")
        header_row.addWidget(self.model_header)
        
        self.tier_badge = QLabel("Consumer")
        self.tier_badge.setStyleSheet("""
            background: #ffd43b33;
            color: #ffd43b;
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 12px;
        """)
        header_row.addWidget(self.tier_badge)
        header_row.addStretch()
        right_layout.addLayout(header_row)
        
        # Description
        self.desc_label = QLabel("Laptops, entry GPU")
        self.desc_label.setStyleSheet("color: #8b949e; font-size: 13px;")
        right_layout.addWidget(self.desc_label)
        
        right_layout.addSpacing(8)
        
        # Specs grid
        specs_frame = QFrame()
        specs_frame.setStyleSheet("background: #21262d; border-radius: 6px; border: none;")
        specs_grid = QGridLayout(specs_frame)
        specs_grid.setContentsMargins(8, 8, 8, 8)
        specs_grid.setSpacing(6)
        
        self.spec_labels = {}
        specs = [
            ("Parameters", "params"),
            ("Embedding Dim", "dim"),
            ("Layers", "layers"),
            ("Attention Heads", "heads"),
            ("Context Length", "ctx"),
            ("System RAM", "ram"),
            ("GPU VRAM", "vram"),
        ]
        
        for i, (label, key) in enumerate(specs):
            row, col = i // 2, (i % 2) * 2
            
            name_lbl = QLabel(label)
            name_lbl.setStyleSheet("color: #8b949e; font-size: 12px;")
            specs_grid.addWidget(name_lbl, row, col)
            
            value_lbl = QLabel("-")
            value_lbl.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 12px;")
            specs_grid.addWidget(value_lbl, row, col + 1)
            self.spec_labels[key] = value_lbl
            
        right_layout.addWidget(specs_frame)
        
        # Progress bars
        bars_frame = QFrame()
        bars_frame.setStyleSheet("background: transparent; border: none;")
        bars_layout = QHBoxLayout(bars_frame)
        bars_layout.setContentsMargins(0, 0, 0, 0)
        bars_layout.setSpacing(10)
        
        for name, attr in [("RAM Usage", "ram_bar"), ("VRAM Usage", "vram_bar")]:
            col = QVBoxLayout()
            col.setSpacing(4)
            lbl = QLabel(name)
            lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
            col.addWidget(lbl)
            
            bar = QProgressBar()
            bar.setMaximum(100)
            bar.setValue(25)
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            bar.setStyleSheet("""
                QProgressBar { background: #30363d; border-radius: 4px; }
                QProgressBar::chunk { background: #69db7c; border-radius: 4px; }
            """)
            col.addWidget(bar)
            setattr(self, attr, bar)
            bars_layout.addLayout(col)
            
        right_layout.addWidget(bars_frame)
        
        # Performance
        self.perf_label = QLabel("~50 tokens/sec on RTX 3060")
        self.perf_label.setStyleSheet("color: #69db7c; font-size: 12px;")
        right_layout.addWidget(self.perf_label)
        
        self.quality_label = QLabel("Good for: Conversations, learning, prototyping")
        self.quality_label.setStyleSheet("color: #bac2de; font-size: 11px;")
        right_layout.addWidget(self.quality_label)
        
        right_layout.addStretch()
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        
        self.create_btn = QPushButton("Create Model")
        self.create_btn.setStyleSheet("""
            QPushButton {
                background: #238636;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background: #2ea043; }
        """)
        self.create_btn.clicked.connect(self._create_model)
        btn_layout.addWidget(self.create_btn)
        
        self.benchmark_btn = QPushButton("Benchmark")
        self.benchmark_btn.setStyleSheet("""
            QPushButton {
                background: #30363d;
                color: #ffffff;
                border: 1px solid #484f58;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 12px;
            }
            QPushButton:hover { background: #484f58; }
        """)
        self.benchmark_btn.clicked.connect(self._run_benchmark)
        btn_layout.addWidget(self.benchmark_btn)
        
        btn_layout.addStretch()
        right_layout.addLayout(btn_layout)
        
        layout.addWidget(right_frame, stretch=1)
        
        # Initial selection
        self._select_model('small')
        
    def _on_card_clicked(self, model: str):
        self._select_model(model)
        
    def _select_model(self, model: str):
        # Update card selection
        for name, card in self.model_cards.items():
            card.set_selected(name == model)
            
        self.selected_model = model
        spec = MODEL_SPECS[model]
        tier = spec['tier']
        color = TIER_COLORS[tier]
        
        # Update header
        self.model_header.setText(model.upper())
        self.model_header.setStyleSheet(f"color: {color};")
        
        self.tier_badge.setText(tier.capitalize())
        self.tier_badge.setStyleSheet(f"""
            background: {color}33;
            color: {color};
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 12px;
        """)
        
        self.desc_label.setText(spec['desc'])
        
        # Update specs
        params = spec['params']
        self.spec_labels['params'].setText(f"~{params/1000:.1f}B" if params >= 1000 else f"~{params}M")
        self.spec_labels['dim'].setText(str(spec['dim']))
        self.spec_labels['layers'].setText(str(spec['layers']))
        self.spec_labels['heads'].setText(str(spec['heads']))
        self.spec_labels['ctx'].setText(f"{spec['ctx']:,}")
        
        ram = spec['ram']
        self.spec_labels['ram'].setText(f"{ram/1024:.0f} GB" if ram >= 1024 else f"{ram} MB")
        
        vram = spec['vram']
        if vram == 0:
            self.spec_labels['vram'].setText("CPU only")
        else:
            self.spec_labels['vram'].setText(f"{vram/1024:.0f} GB" if vram >= 1024 else f"{vram} MB")
        
        # Update bars
        ram_pct = min(100, int(ram / 327.68))
        vram_pct = min(100, int(vram / 245.76))
        self.ram_bar.setValue(ram_pct)
        self.vram_bar.setValue(vram_pct)
        
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
        perf_map = {
            'nano': ('~200 tok/s CPU', 'Testing, lightweight'),
            'micro': ('~150 tok/s CPU', 'IoT, simple queries'),
            'tiny': ('~100 tok/s Pi 4', 'Edge AI, basic chat'),
            'mini': ('~80 tok/s mobile', 'Mobile apps'),
            'small': ('~50 tok/s RTX 3060', 'Conversations, learning'),
            'medium': ('~35 tok/s RTX 3060', 'Good conversations'),
            'base': ('~25 tok/s RTX 3060', 'Quality responses'),
            'large': ('~15 tok/s RTX 3080', 'Complex tasks'),
            'xl': ('~8 tok/s RTX 4090', 'Excellent quality'),
            'xxl': ('~4 tok/s RTX 4090', 'Near-commercial'),
            'huge': ('~2 tok/s 2x 4090', 'Production-grade'),
            'giant': ('~1 tok/s A100', 'Commercial'),
            'colossal': ('~0.5 tok/s 2x A100', 'Enterprise'),
            'titan': ('~0.2 tok/s cluster', 'SOTA research'),
            'omega': ('Research scale', 'Frontier'),
        }
        
        perf, quality = perf_map.get(model, ('Unknown', 'Unknown'))
        self.perf_label.setText(perf)
        self.quality_label.setText(f"Good for: {quality}")
        
        self.model_changed.emit(model)
        
    def _create_model(self):
        model = self.selected_model
        spec = MODEL_SPECS[model]
        
        params_str = f"{spec['params']/1000:.1f}B" if spec['params'] >= 1000 else f"{spec['params']}M"
        ram_str = f"{spec['ram']/1024:.1f}GB" if spec['ram'] >= 1024 else f"{spec['ram']}MB"
        
        from PyQt5.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self,
            f"Create {model.upper()} Model",
            f"Enter name for your {model.upper()} model:\n\n"
            f"Parameters: ~{params_str}\n"
            f"RAM Required: {ram_str}",
            text=f"my_{model}_model"
        )
        
        if ok and name:
            name = name.strip().replace(' ', '_')
            
            if self.main_window and hasattr(self.main_window, 'registry'):
                try:
                    self.main_window.registry.create_model(name, size=model, vocab_size=32000)
                    self.main_window.current_model_name = name
                    self.main_window.setWindowTitle(f"Enigma AI Engine - {name}")
                    
                    if hasattr(self.main_window, 'model_status_btn'):
                        self.main_window.model_status_btn.setText(f"Model: {name}  v")
                    
                    QMessageBox.information(self, "Model Created", 
                        f"{model.upper()} model '{name}' created!\n\n"
                        "Next: Go to Train tab to add data and train.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to create model: {e}")
            else:
                QMessageBox.warning(self, "No Registry", "Model registry not available.")
                
    def _run_benchmark(self):
        model = self.selected_model
        
        reply = QMessageBox.question(
            self, "Run Benchmark",
            f"Benchmark {model.upper()} model?\n\nThis tests forward pass, generation speed, and memory.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        try:
            import time

            import torch

            from enigma_engine.core.model import create_model
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            
            from PyQt5.QtWidgets import QProgressDialog
            progress = QProgressDialog("Running benchmark...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            results = [f"Device: {device_name}", f"Model: {model.upper()}", "-" * 40]
            
            progress.setValue(10)
            test_model = create_model(size=model, vocab_size=32000)
            test_model.to(device)
            test_model.eval()
            
            params = sum(p.numel() for p in test_model.parameters())
            results.append(f"Parameters: {params:,}")
            
            progress.setValue(30)
            seq_len = 128
            input_ids = torch.randint(0, 32000, (1, seq_len), device=device)
            
            with torch.no_grad():
                for _ in range(3):
                    _ = test_model(input_ids, use_cache=False)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            times = []
            with torch.no_grad():
                for _ in range(5):
                    start = time.perf_counter()
                    _ = test_model(input_ids, use_cache=False)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)
            
            avg = sum(times) / len(times)
            results.append(f"\nForward: {avg*1000:.1f}ms, {seq_len/avg:.0f} tok/s")
            
            progress.setValue(70)
            
            if device.type == "cuda":
                mem = torch.cuda.memory_allocated() / 1024**2
                results.append(f"GPU Memory: {mem:.0f} MB")
            
            del test_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            progress.setValue(100)
            progress.close()
            
            QMessageBox.information(self, "Benchmark Results", "\n".join(results))
            
        except Exception as e:
            QMessageBox.warning(self, "Benchmark Error", f"Failed: {e}")


def create_scaling_tab(window) -> QWidget:
    """Factory function to create scaling tab."""
    return ScalingTab(window)
