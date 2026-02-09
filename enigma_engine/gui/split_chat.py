"""
Split View Chat for Enigma AI Engine

Compare two models side-by-side in real-time.

Features:
- Dual chat displays
- Synchronized input
- Response timing comparison
- Token count display
- Model switching
- Export comparison results

Usage:
    from enigma_engine.gui.split_chat import SplitChatWidget
    
    # In a PyQt5 application
    widget = SplitChatWidget()
    widget.set_models("model_a", "model_b")
    widget.show()
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
        QTextEdit, QLineEdit, QPushButton, QLabel,
        QComboBox, QFrame, QProgressBar, QGroupBox,
        QScrollArea
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
    from PyQt5.QtGui import QFont, QTextCursor
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False
    QWidget = object


@dataclass
class ChatResponse:
    """Response from a model."""
    model_name: str
    text: str
    tokens: int = 0
    time_seconds: float = 0.0
    tokens_per_second: float = 0.0


@dataclass
class ComparisonResult:
    """Result of comparing two model responses."""
    prompt: str
    response_a: ChatResponse = None
    response_b: ChatResponse = None
    timestamp: str = ""
    
    def speed_winner(self) -> Optional[str]:
        """Which model responded faster."""
        if not self.response_a or not self.response_b:
            return None
        if self.response_a.time_seconds < self.response_b.time_seconds:
            return self.response_a.model_name
        return self.response_b.model_name
    
    def length_winner(self) -> Optional[str]:
        """Which response is longer."""
        if not self.response_a or not self.response_b:
            return None
        if self.response_a.tokens > self.response_b.tokens:
            return self.response_a.model_name
        return self.response_b.model_name


class InferenceWorker(QThread):
    """Worker thread for model inference."""
    
    finished = pyqtSignal(ChatResponse)
    error = pyqtSignal(str)
    
    def __init__(self, model_name: str, prompt: str):
        super().__init__()
        self.model_name = model_name
        self.prompt = prompt
        self._engine = None
    
    def run(self):
        """Run inference."""
        start_time = time.time()
        
        try:
            # Get inference engine
            from enigma_engine.core.inference import EnigmaEngine
            
            if not hasattr(InferenceWorker, '_engines'):
                InferenceWorker._engines = {}
            
            if self.model_name not in InferenceWorker._engines:
                InferenceWorker._engines[self.model_name] = EnigmaEngine(self.model_name)
            
            engine = InferenceWorker._engines[self.model_name]
            
            # Generate
            response_text = engine.generate(self.prompt, max_gen=512)
            
            elapsed = time.time() - start_time
            
            # Estimate tokens (rough)
            tokens = len(response_text.split()) * 1.3
            
            result = ChatResponse(
                model_name=self.model_name,
                text=response_text,
                tokens=int(tokens),
                time_seconds=elapsed,
                tokens_per_second=tokens / max(elapsed, 0.001)
            )
            
            self.finished.emit(result)
            
        except Exception as e:
            logger.error(f"Inference error for {self.model_name}: {e}")
            self.error.emit(str(e))


if PYQT5_AVAILABLE:
    class ChatPanel(QFrame):
        """Single chat panel for one model."""
        
        def __init__(self, title: str = "Model", parent=None):
            super().__init__(parent)
            self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
            self._setup_ui(title)
        
        def _setup_ui(self, title: str):
            layout = QVBoxLayout(self)
            layout.setContentsMargins(8, 8, 8, 8)
            
            # Header
            header = QHBoxLayout()
            
            self.title_label = QLabel(title)
            self.title_label.setFont(QFont("", 11, QFont.Bold))
            header.addWidget(self.title_label)
            
            header.addStretch()
            
            self.status_label = QLabel("Ready")
            self.status_label.setStyleSheet("color: gray;")
            header.addWidget(self.status_label)
            
            layout.addLayout(header)
            
            # Chat display
            self.chat_display = QTextEdit()
            self.chat_display.setReadOnly(True)
            self.chat_display.setFont(QFont("Consolas", 10))
            layout.addWidget(self.chat_display)
            
            # Stats
            stats_layout = QHBoxLayout()
            
            self.tokens_label = QLabel("Tokens: 0")
            stats_layout.addWidget(self.tokens_label)
            
            self.time_label = QLabel("Time: 0.0s")
            stats_layout.addWidget(self.time_label)
            
            self.speed_label = QLabel("Speed: 0 t/s")
            stats_layout.addWidget(self.speed_label)
            
            layout.addLayout(stats_layout)
            
            # Progress
            self.progress = QProgressBar()
            self.progress.setMaximumHeight(4)
            self.progress.setTextVisible(False)
            self.progress.hide()
            layout.addWidget(self.progress)
        
        def set_title(self, title: str):
            """Set panel title."""
            self.title_label.setText(title)
        
        def add_message(self, role: str, content: str):
            """Add a message to the display."""
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.End)
            
            if role == "user":
                cursor.insertHtml(f'<p style="color: #4a9eff;"><b>You:</b></p>')
            else:
                cursor.insertHtml(f'<p style="color: #00c853;"><b>{self.title_label.text()}:</b></p>')
            
            cursor.insertHtml(f'<p>{content}</p><br>')
            
            self.chat_display.setTextCursor(cursor)
            self.chat_display.ensureCursorVisible()
        
        def update_stats(self, tokens: int, time_sec: float, speed: float):
            """Update statistics display."""
            self.tokens_label.setText(f"Tokens: {tokens}")
            self.time_label.setText(f"Time: {time_sec:.2f}s")
            self.speed_label.setText(f"Speed: {speed:.1f} t/s")
        
        def set_status(self, status: str, color: str = "gray"):
            """Set status text."""
            self.status_label.setText(status)
            self.status_label.setStyleSheet(f"color: {color};")
        
        def show_progress(self, show: bool):
            """Show/hide progress bar."""
            if show:
                self.progress.setRange(0, 0)  # Indeterminate
                self.progress.show()
            else:
                self.progress.hide()
        
        def clear(self):
            """Clear chat display."""
            self.chat_display.clear()
            self.update_stats(0, 0.0, 0.0)


    class SplitChatWidget(QWidget):
        """
        Split view for comparing two models.
        
        Shows both model responses side-by-side.
        """
        
        response_received = pyqtSignal(ChatResponse)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            
            self._model_a = ""
            self._model_b = ""
            self._worker_a: Optional[InferenceWorker] = None
            self._worker_b: Optional[InferenceWorker] = None
            self._comparisons: List[ComparisonResult] = []
            
            self._setup_ui()
        
        def _setup_ui(self):
            layout = QVBoxLayout(self)
            
            # Model selection header
            header = QHBoxLayout()
            
            # Model A selector
            header.addWidget(QLabel("Model A:"))
            self.model_a_combo = QComboBox()
            self.model_a_combo.setMinimumWidth(150)
            self.model_a_combo.currentTextChanged.connect(self._on_model_a_changed)
            header.addWidget(self.model_a_combo)
            
            header.addStretch()
            
            # Model B selector
            header.addWidget(QLabel("Model B:"))
            self.model_b_combo = QComboBox()
            self.model_b_combo.setMinimumWidth(150)
            self.model_b_combo.currentTextChanged.connect(self._on_model_b_changed)
            header.addWidget(self.model_b_combo)
            
            # Swap button
            swap_btn = QPushButton("Swap")
            swap_btn.setMaximumWidth(60)
            swap_btn.clicked.connect(self._swap_models)
            header.addWidget(swap_btn)
            
            layout.addLayout(header)
            
            # Splitter for chat panels
            self.splitter = QSplitter(Qt.Horizontal)
            
            self.panel_a = ChatPanel("Model A")
            self.panel_b = ChatPanel("Model B")
            
            self.splitter.addWidget(self.panel_a)
            self.splitter.addWidget(self.panel_b)
            self.splitter.setSizes([500, 500])
            
            layout.addWidget(self.splitter)
            
            # Input area
            input_layout = QHBoxLayout()
            
            self.input_field = QLineEdit()
            self.input_field.setPlaceholderText("Type a message to compare responses...")
            self.input_field.returnPressed.connect(self._send_message)
            input_layout.addWidget(self.input_field)
            
            self.send_btn = QPushButton("Send")
            self.send_btn.clicked.connect(self._send_message)
            input_layout.addWidget(self.send_btn)
            
            self.clear_btn = QPushButton("Clear")
            self.clear_btn.clicked.connect(self._clear_chat)
            input_layout.addWidget(self.clear_btn)
            
            layout.addLayout(input_layout)
            
            # Comparison summary
            self.summary_label = QLabel("")
            self.summary_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.summary_label)
            
            # Load available models
            self._load_models()
        
        def _load_models(self):
            """Load available models into combo boxes."""
            try:
                from enigma_engine.config import CONFIG
                
                models_dir = CONFIG.MODELS_DIR
                if models_dir.exists():
                    models = [d.name for d in models_dir.iterdir() if d.is_dir()]
                    
                    self.model_a_combo.clear()
                    self.model_b_combo.clear()
                    
                    self.model_a_combo.addItems(models)
                    self.model_b_combo.addItems(models)
                    
                    if len(models) >= 2:
                        self.model_b_combo.setCurrentIndex(1)
                        
            except Exception as e:
                logger.error(f"Failed to load models: {e}")
        
        def set_models(self, model_a: str, model_b: str):
            """Set the models to compare."""
            self._model_a = model_a
            self._model_b = model_b
            
            self.panel_a.set_title(model_a)
            self.panel_b.set_title(model_b)
            
            # Update combo boxes
            idx_a = self.model_a_combo.findText(model_a)
            idx_b = self.model_b_combo.findText(model_b)
            
            if idx_a >= 0:
                self.model_a_combo.setCurrentIndex(idx_a)
            if idx_b >= 0:
                self.model_b_combo.setCurrentIndex(idx_b)
        
        def _on_model_a_changed(self, name: str):
            self._model_a = name
            self.panel_a.set_title(name or "Model A")
        
        def _on_model_b_changed(self, name: str):
            self._model_b = name
            self.panel_b.set_title(name or "Model B")
        
        def _swap_models(self):
            """Swap model A and B."""
            idx_a = self.model_a_combo.currentIndex()
            idx_b = self.model_b_combo.currentIndex()
            
            self.model_a_combo.setCurrentIndex(idx_b)
            self.model_b_combo.setCurrentIndex(idx_a)
        
        def _send_message(self):
            """Send message to both models."""
            prompt = self.input_field.text().strip()
            if not prompt:
                return
            
            if not self._model_a or not self._model_b:
                self.summary_label.setText("Please select both models")
                return
            
            # Show user message in both panels
            self.panel_a.add_message("user", prompt)
            self.panel_b.add_message("user", prompt)
            
            self.input_field.clear()
            self.input_field.setEnabled(False)
            self.send_btn.setEnabled(False)
            
            # Start inference for both models
            self._current_comparison = ComparisonResult(prompt=prompt)
            self._responses_received = 0
            
            self.panel_a.show_progress(True)
            self.panel_a.set_status("Generating...", "#ff9800")
            
            self.panel_b.show_progress(True)
            self.panel_b.set_status("Generating...", "#ff9800")
            
            # Start workers
            self._worker_a = InferenceWorker(self._model_a, prompt)
            self._worker_a.finished.connect(lambda r: self._on_response(r, "a"))
            self._worker_a.error.connect(lambda e: self._on_error(e, "a"))
            self._worker_a.start()
            
            self._worker_b = InferenceWorker(self._model_b, prompt)
            self._worker_b.finished.connect(lambda r: self._on_response(r, "b"))
            self._worker_b.error.connect(lambda e: self._on_error(e, "b"))
            self._worker_b.start()
        
        def _on_response(self, response: ChatResponse, panel: str):
            """Handle response from a model."""
            if panel == "a":
                self.panel_a.add_message("assistant", response.text)
                self.panel_a.update_stats(response.tokens, response.time_seconds, response.tokens_per_second)
                self.panel_a.show_progress(False)
                self.panel_a.set_status("Done", "#4caf50")
                self._current_comparison.response_a = response
            else:
                self.panel_b.add_message("assistant", response.text)
                self.panel_b.update_stats(response.tokens, response.time_seconds, response.tokens_per_second)
                self.panel_b.show_progress(False)
                self.panel_b.set_status("Done", "#4caf50")
                self._current_comparison.response_b = response
            
            self._responses_received += 1
            
            if self._responses_received >= 2:
                self._comparison_complete()
        
        def _on_error(self, error: str, panel: str):
            """Handle error from a model."""
            if panel == "a":
                self.panel_a.add_message("assistant", f"Error: {error}")
                self.panel_a.show_progress(False)
                self.panel_a.set_status("Error", "#f44336")
            else:
                self.panel_b.add_message("assistant", f"Error: {error}")
                self.panel_b.show_progress(False)
                self.panel_b.set_status("Error", "#f44336")
            
            self._responses_received += 1
            
            if self._responses_received >= 2:
                self._comparison_complete()
        
        def _comparison_complete(self):
            """Called when both models have responded."""
            self.input_field.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.input_field.setFocus()
            
            # Update summary
            comp = self._current_comparison
            self._comparisons.append(comp)
            
            if comp.response_a and comp.response_b:
                speed_winner = comp.speed_winner()
                summary = f"Faster: {speed_winner} | "
                
                diff = abs(comp.response_a.time_seconds - comp.response_b.time_seconds)
                summary += f"Difference: {diff:.2f}s"
                
                self.summary_label.setText(summary)
            else:
                self.summary_label.setText("Comparison incomplete (errors occurred)")
        
        def _clear_chat(self):
            """Clear both chat panels."""
            self.panel_a.clear()
            self.panel_b.clear()
            self.summary_label.setText("")
            self._comparisons = []
        
        def export_comparisons(self, path: str):
            """Export comparison results to JSON."""
            import json
            
            data = {
                "model_a": self._model_a,
                "model_b": self._model_b,
                "comparisons": []
            }
            
            for comp in self._comparisons:
                comp_data = {
                    "prompt": comp.prompt,
                    "response_a": {
                        "text": comp.response_a.text if comp.response_a else None,
                        "tokens": comp.response_a.tokens if comp.response_a else 0,
                        "time": comp.response_a.time_seconds if comp.response_a else 0,
                    },
                    "response_b": {
                        "text": comp.response_b.text if comp.response_b else None,
                        "tokens": comp.response_b.tokens if comp.response_b else 0,
                        "time": comp.response_b.time_seconds if comp.response_b else 0,
                    },
                    "speed_winner": comp.speed_winner(),
                }
                data["comparisons"].append(comp_data)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported {len(self._comparisons)} comparisons to {path}")
        
        def get_stats_summary(self) -> Dict[str, Any]:
            """Get aggregate statistics."""
            if not self._comparisons:
                return {}
            
            a_wins = 0
            b_wins = 0
            a_total_time = 0
            b_total_time = 0
            
            for comp in self._comparisons:
                if comp.response_a and comp.response_b:
                    if comp.response_a.time_seconds < comp.response_b.time_seconds:
                        a_wins += 1
                    else:
                        b_wins += 1
                    
                    a_total_time += comp.response_a.time_seconds
                    b_total_time += comp.response_b.time_seconds
            
            n = len(self._comparisons)
            
            return {
                "total_comparisons": n,
                "model_a_wins": a_wins,
                "model_b_wins": b_wins,
                "model_a_avg_time": a_total_time / max(n, 1),
                "model_b_avg_time": b_total_time / max(n, 1),
            }

else:
    # Stub when PyQt5 not available
    class SplitChatWidget:
        def __init__(self, *args, **kwargs):
            logging.warning("PyQt5 not available for SplitChatWidget")
