"""
Model Comparison Tab - Compare responses from multiple models side by side.

Features:
  - Select 2-4 models for comparison
  - Enter a prompt and see all responses simultaneously
  - Compare latency, token counts, quality
  - Save comparison results
  - Rating system for responses
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .shared_components import NoScrollComboBox

logger = logging.getLogger(__name__)

# Comparison history directory
COMPARISON_DIR = Path.home() / ".enigma_engine" / "comparisons"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

# Styles
STYLE_GROUP = """
    QGroupBox {
        font-weight: bold;
        border: 1px solid #45475a;
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        color: #cdd6f4;
    }
"""

STYLE_RESPONSE_FRAME = """
    QFrame {
        background: #1e1e2e;
        border: 1px solid #45475a;
        border-radius: 6px;
        padding: 8px;
    }
"""

STYLE_RESPONSE_HEADER = """
    QLabel {
        color: #89b4fa;
        font-weight: bold;
        font-size: 12px;
    }
"""

STYLE_METRICS = """
    QLabel {
        color: #6c7086;
        font-size: 10px;
    }
"""

STYLE_BUTTON_PRIMARY = """
    QPushButton {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
        font-size: 11px;
    }
    QPushButton:hover { background-color: #2980b9; }
    QPushButton:pressed { background-color: #1c5980; }
    QPushButton:disabled { background-color: #45475a; color: #6c7086; }
"""

STYLE_BUTTON_SECONDARY = """
    QPushButton {
        background-color: #45475a;
        color: #cdd6f4;
        border: none;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 10px;
    }
    QPushButton:hover { background-color: #585b70; }
"""

STYLE_WINNER = """
    QFrame {
        background: rgba(166, 227, 161, 0.1);
        border: 2px solid #a6e3a1;
        border-radius: 6px;
        padding: 8px;
    }
"""


class CompareWorker(QThread):
    """Worker thread for running model comparisons."""
    
    result_ready = pyqtSignal(str, str, float, int)  # model_id, response, latency, tokens
    error = pyqtSignal(str, str)  # model_id, error_message
    all_done = pyqtSignal()
    
    def __init__(self, models: list[str], prompt: str, max_tokens: int = 256):
        super().__init__()
        self.models = models
        self.prompt = prompt
        self.max_tokens = max_tokens
        self._stop_requested = False
    
    def run(self):
        """Run comparison for all models."""
        for model_id in self.models:
            if self._stop_requested:
                break
            
            try:
                response, latency, tokens = self._run_single_model(model_id)
                self.result_ready.emit(model_id, response, latency, tokens)
            except Exception as e:
                self.error.emit(model_id, str(e))
        
        self.all_done.emit()
    
    def _run_single_model(self, model_id: str):
        """Run inference on a single model."""
        start_time = time.time()
        response = ""
        
        try:
            # Try to use the model registry to get the model
            from ...core.model_registry import ModelRegistry
            registry = ModelRegistry()
            
            # Check if model exists in registry
            if model_id in registry.list_models():
                model, config = registry.load_model(model_id)
                # HuggingFace models have .generate(), Forge models may have different APIs
                if hasattr(model, 'generate'):
                    response = model.generate(self.prompt, max_gen=self.max_tokens)  # type: ignore
                elif hasattr(model, 'chat'):
                    response = model.chat(self.prompt)  # type: ignore
                else:
                    response = f"[Model loaded but no generate method]"
            else:
                raise ValueError(f"Model {model_id} not in registry")
            
        except Exception as e:
            # Fallback: use inference engine
            try:
                from ...core.inference import EnigmaEngine
                engine = EnigmaEngine()
                response = engine.generate(self.prompt, max_gen=self.max_tokens)
            except Exception as e2:
                # Final fallback: return placeholder
                response = f"[Could not generate - {e2}]"
        
        latency = time.time() - start_time
        tokens = len(response.split())  # Rough token estimate
        
        return response, latency, tokens
    
    def stop(self):
        """Request to stop the comparison."""
        self._stop_requested = True


class ResponsePanel(QFrame):
    """Panel displaying a single model's response."""
    
    vote_clicked = pyqtSignal(str, int)  # model_id, rating (1-5)
    
    def __init__(self, model_id: str, parent=None):
        super().__init__(parent)
        self.model_id = model_id
        self.latency = 0.0
        self.tokens = 0
        self.init_ui()
    
    def init_ui(self):
        """Initialize the panel UI."""
        self.setStyleSheet(STYLE_RESPONSE_FRAME)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Header with model name
        header_layout = QHBoxLayout()
        self.model_label = QLabel(self.model_id)
        self.model_label.setStyleSheet(STYLE_RESPONSE_HEADER)
        header_layout.addWidget(self.model_label)
        header_layout.addStretch()
        
        # Loading indicator
        self.loading_label = QLabel("Loading...")
        self.loading_label.setStyleSheet("color: #f9e2af; font-style: italic;")
        header_layout.addWidget(self.loading_label)
        
        layout.addLayout(header_layout)
        
        # Response text
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.response_text.setPlaceholderText("Waiting for response...")
        self.response_text.setStyleSheet("""
            QTextEdit {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 8px;
                color: #cdd6f4;
                font-size: 11px;
            }
        """)
        self.response_text.setMinimumHeight(100)
        layout.addWidget(self.response_text, stretch=1)
        
        # Metrics row
        metrics_layout = QHBoxLayout()
        
        self.latency_label = QLabel("Latency: --")
        self.latency_label.setStyleSheet(STYLE_METRICS)
        metrics_layout.addWidget(self.latency_label)
        
        self.tokens_label = QLabel("Tokens: --")
        self.tokens_label.setStyleSheet(STYLE_METRICS)
        metrics_layout.addWidget(self.tokens_label)
        
        metrics_layout.addStretch()
        
        # Rating buttons
        self.rating_btns = []
        for i in range(1, 6):
            btn = QPushButton(str(i))
            btn.setFixedSize(24, 24)
            btn.setToolTip(f"Rate {i}/5")
            btn.setStyleSheet(STYLE_BUTTON_SECONDARY)
            btn.clicked.connect(lambda checked, r=i: self._on_rate(r))
            metrics_layout.addWidget(btn)
            self.rating_btns.append(btn)
        
        layout.addLayout(metrics_layout)
    
    def set_response(self, text: str, latency: float, tokens: int):
        """Set the response content."""
        self.response_text.setPlainText(text)
        self.latency = latency
        self.tokens = tokens
        self.latency_label.setText(f"Latency: {latency:.2f}s")
        self.tokens_label.setText(f"Tokens: ~{tokens}")
        self.loading_label.hide()
    
    def set_error(self, error: str):
        """Show an error message."""
        self.response_text.setPlainText(f"Error: {error}")
        self.response_text.setStyleSheet("""
            QTextEdit {
                background: rgba(243, 139, 168, 0.1);
                border: 1px solid #f38ba8;
                border-radius: 4px;
                padding: 8px;
                color: #f38ba8;
                font-size: 11px;
            }
        """)
        self.loading_label.hide()
    
    def mark_winner(self):
        """Mark this panel as the winner."""
        self.setStyleSheet(STYLE_WINNER)
        self.model_label.setText(f"{self.model_id} [WINNER]")
    
    def reset(self):
        """Reset the panel for a new comparison."""
        self.setStyleSheet(STYLE_RESPONSE_FRAME)
        self.model_label.setText(self.model_id)
        self.response_text.clear()
        self.response_text.setPlaceholderText("Waiting for response...")
        self.response_text.setStyleSheet("""
            QTextEdit {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 8px;
                color: #cdd6f4;
                font-size: 11px;
            }
        """)
        self.latency_label.setText("Latency: --")
        self.tokens_label.setText("Tokens: --")
        self.loading_label.setText("Loading...")
        self.loading_label.show()
        self.latency = 0.0
        self.tokens = 0
    
    def _on_rate(self, rating: int):
        """Handle rating button click."""
        # Highlight selected rating
        for i, btn in enumerate(self.rating_btns):
            if i < rating:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f9e2af;
                        color: #1e1e2e;
                        border: none;
                        border-radius: 4px;
                    }
                """)
            else:
                btn.setStyleSheet(STYLE_BUTTON_SECONDARY)
        
        self.vote_clicked.emit(self.model_id, rating)


def create_model_comparison_tab(parent) -> QWidget:
    """
    Create the model comparison tab.
    
    Args:
        parent: The main window
    
    Returns:
        QWidget: The comparison tab widget
    """
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(10)
    
    # Title
    title = QLabel("Model Comparison")
    title_font = QFont()
    title_font.setPointSize(14)
    title_font.setBold(True)
    title.setFont(title_font)
    title.setStyleSheet("color: #cdd6f4;")
    layout.addWidget(title)
    
    subtitle = QLabel("Compare responses from multiple models side by side")
    subtitle.setStyleSheet("color: #6c7086; font-size: 11px;")
    layout.addWidget(subtitle)
    
    # === MODEL SELECTION ===
    model_group = QGroupBox("Select Models to Compare")
    model_group.setStyleSheet(STYLE_GROUP)
    model_layout = QHBoxLayout()
    
    # Get available models
    available_models = _get_available_models()
    
    parent.comparison_model_combos = []
    for i in range(4):  # Support up to 4 models
        combo = NoScrollComboBox()
        combo.addItem("(None)" if i >= 2 else "(Select Model)")
        for model in available_models:
            combo.addItem(model)
        combo.setMinimumWidth(150)
        model_layout.addWidget(combo)
        parent.comparison_model_combos.append(combo)
    
    model_layout.addStretch()
    model_group.setLayout(model_layout)
    layout.addWidget(model_group)
    
    # === PROMPT INPUT ===
    prompt_group = QGroupBox("Prompt")
    prompt_group.setStyleSheet(STYLE_GROUP)
    prompt_layout = QVBoxLayout()
    
    parent.comparison_prompt = QTextEdit()
    parent.comparison_prompt.setPlaceholderText("Enter your prompt here...")
    parent.comparison_prompt.setMaximumHeight(100)
    parent.comparison_prompt.setStyleSheet("""
        QTextEdit {
            background: #313244;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 8px;
            color: #cdd6f4;
            font-size: 12px;
        }
    """)
    prompt_layout.addWidget(parent.comparison_prompt)
    
    # Options row
    options_layout = QHBoxLayout()
    
    options_layout.addWidget(QLabel("Max Tokens:"))
    parent.comparison_max_tokens = QSpinBox()
    parent.comparison_max_tokens.setRange(16, 2048)
    parent.comparison_max_tokens.setValue(256)
    parent.comparison_max_tokens.setStyleSheet("""
        QSpinBox {
            background: #313244;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 4px;
            color: #cdd6f4;
        }
    """)
    options_layout.addWidget(parent.comparison_max_tokens)
    
    options_layout.addStretch()
    
    # Compare button
    parent.btn_compare = QPushButton("Compare")
    parent.btn_compare.setStyleSheet(STYLE_BUTTON_PRIMARY)
    parent.btn_compare.setMinimumWidth(120)
    parent.btn_compare.clicked.connect(lambda: _run_comparison(parent))
    options_layout.addWidget(parent.btn_compare)
    
    # Stop button (hidden by default)
    parent.btn_stop_compare = QPushButton("Stop")
    parent.btn_stop_compare.setStyleSheet("""
        QPushButton {
            background-color: #f38ba8;
            color: #1e1e2e;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #eba0ac; }
    """)
    parent.btn_stop_compare.setMinimumWidth(80)
    parent.btn_stop_compare.clicked.connect(lambda: _stop_comparison(parent))
    parent.btn_stop_compare.hide()
    options_layout.addWidget(parent.btn_stop_compare)
    
    prompt_layout.addLayout(options_layout)
    prompt_group.setLayout(prompt_layout)
    layout.addWidget(prompt_group)
    
    # === RESPONSE PANELS ===
    parent.comparison_panels_widget = QWidget()
    parent.comparison_panels_layout = QGridLayout(parent.comparison_panels_widget)
    parent.comparison_panels_layout.setSpacing(10)
    
    # Create 4 response panels (2x2 grid)
    parent.response_panels = []
    for i in range(4):
        panel = ResponsePanel(f"Model {i+1}")
        panel.vote_clicked.connect(lambda mid, r: _on_vote(parent, mid, r))
        parent.comparison_panels_layout.addWidget(panel, i // 2, i % 2)
        parent.response_panels.append(panel)
        panel.hide()  # Hidden until comparison starts
    
    # Scroll area for responses
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(parent.comparison_panels_widget)
    scroll.setStyleSheet("QScrollArea { border: none; }")
    layout.addWidget(scroll, stretch=1)
    
    # === STATUS BAR ===
    status_layout = QHBoxLayout()
    
    parent.comparison_status = QLabel("Select models and enter a prompt to begin")
    parent.comparison_status.setStyleSheet("color: #6c7086; font-size: 11px;")
    status_layout.addWidget(parent.comparison_status)
    
    status_layout.addStretch()
    
    # Save comparison button
    parent.btn_save_comparison = QPushButton("Save Results")
    parent.btn_save_comparison.setStyleSheet(STYLE_BUTTON_SECONDARY)
    parent.btn_save_comparison.clicked.connect(lambda: _save_comparison(parent))
    parent.btn_save_comparison.setEnabled(False)
    status_layout.addWidget(parent.btn_save_comparison)
    
    layout.addLayout(status_layout)
    
    # Initialize worker reference
    parent._compare_worker = None
    parent._comparison_results = {}
    
    return tab


def _get_available_models() -> list[str]:
    """Get list of available models for comparison."""
    models = []
    
    # Try to get from model registry
    try:
        from ...core.model_registry import ModelRegistry
        registry = ModelRegistry()
        models.extend(registry.list_models())
    except Exception:
        pass
    
    # Try to get from models directory
    try:
        models_dir = Path.home() / ".enigma_engine" / "models"
        if models_dir.exists():
            for item in models_dir.iterdir():
                if item.is_dir() and (item / "config.json").exists():
                    if item.name not in models:
                        models.append(item.name)
    except Exception:
        pass
    
    # Add some default model names
    default_models = ["forge-small", "forge-medium", "forge-large", "gpt2", "gpt2-medium"]
    for m in default_models:
        if m not in models:
            models.append(m)
    
    return sorted(set(models))


def _run_comparison(parent):
    """Start the model comparison."""
    # Get selected models
    selected_models = []
    for combo in parent.comparison_model_combos:
        model = combo.currentText()
        if model and model != "(None)" and model != "(Select Model)":
            selected_models.append(model)
    
    if len(selected_models) < 2:
        QMessageBox.warning(parent, "Model Comparison", "Please select at least 2 models to compare.")
        return
    
    # Get prompt
    prompt = parent.comparison_prompt.toPlainText().strip()
    if not prompt:
        QMessageBox.warning(parent, "Model Comparison", "Please enter a prompt.")
        return
    
    # Update UI
    parent.btn_compare.setEnabled(False)
    parent.btn_stop_compare.show()
    parent.btn_save_comparison.setEnabled(False)
    parent.comparison_status.setText(f"Comparing {len(selected_models)} models...")
    
    # Setup panels
    for i, panel in enumerate(parent.response_panels):
        if i < len(selected_models):
            panel.model_id = selected_models[i]
            panel.model_label.setText(selected_models[i])
            panel.reset()
            panel.show()
        else:
            panel.hide()
    
    # Clear previous results
    parent._comparison_results = {}
    
    # Start worker
    max_tokens = parent.comparison_max_tokens.value()
    parent._compare_worker = CompareWorker(selected_models, prompt, max_tokens)
    parent._compare_worker.result_ready.connect(lambda *args: _on_result(parent, *args))
    parent._compare_worker.error.connect(lambda *args: _on_error(parent, *args))
    parent._compare_worker.all_done.connect(lambda: _on_compare_done(parent))
    parent._compare_worker.start()


def _stop_comparison(parent):
    """Stop the running comparison."""
    if parent._compare_worker:
        parent._compare_worker.stop()
        parent._compare_worker = None
    
    parent.btn_compare.setEnabled(True)
    parent.btn_stop_compare.hide()
    parent.comparison_status.setText("Comparison stopped")


def _on_result(parent, model_id: str, response: str, latency: float, tokens: int):
    """Handle a model result."""
    # Find the panel for this model
    for panel in parent.response_panels:
        if panel.model_id == model_id:
            panel.set_response(response, latency, tokens)
            break
    
    # Store result
    parent._comparison_results[model_id] = {
        "response": response,
        "latency": latency,
        "tokens": tokens
    }


def _on_error(parent, model_id: str, error: str):
    """Handle a model error."""
    for panel in parent.response_panels:
        if panel.model_id == model_id:
            panel.set_error(error)
            break


def _on_compare_done(parent):
    """Handle comparison completion."""
    parent.btn_compare.setEnabled(True)
    parent.btn_stop_compare.hide()
    parent.btn_save_comparison.setEnabled(True)
    parent._compare_worker = None
    
    # Determine winner by lowest latency
    if parent._comparison_results:
        winner = min(parent._comparison_results.keys(), 
                     key=lambda k: parent._comparison_results[k]["latency"])
        for panel in parent.response_panels:
            if panel.model_id == winner:
                panel.mark_winner()
                break
        
        parent.comparison_status.setText(
            f"Comparison complete. Fastest: {winner} "
            f"({parent._comparison_results[winner]['latency']:.2f}s)"
        )
    else:
        parent.comparison_status.setText("Comparison complete (no results)")


def _on_vote(parent, model_id: str, rating: int):
    """Handle a vote for a model response."""
    if model_id in parent._comparison_results:
        parent._comparison_results[model_id]["rating"] = rating
    parent.comparison_status.setText(f"Rated {model_id}: {rating}/5")


def _save_comparison(parent):
    """Save the comparison results."""
    if not parent._comparison_results:
        return
    
    # Create comparison record
    record = {
        "timestamp": datetime.now().isoformat(),
        "prompt": parent.comparison_prompt.toPlainText().strip(),
        "max_tokens": parent.comparison_max_tokens.value(),
        "results": parent._comparison_results
    }
    
    # Save to file
    filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = COMPARISON_DIR / filename
    
    try:
        filepath.write_text(json.dumps(record, indent=2))
        parent.comparison_status.setText(f"Saved to {filename}")
    except Exception as e:
        parent.comparison_status.setText(f"Save failed: {e}")
