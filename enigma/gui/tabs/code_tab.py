"""
Code Generation Tab - Generate code using local or cloud models.

Providers:
  - LOCAL: Uses Enigma's own model
  - OPENAI: GPT-4 (requires openai, API key)
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QComboBox, QTextEdit, QProgressBar,
        QMessageBox, QFileDialog, QGroupBox, QPlainTextEdit
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont, QFontDatabase
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from ...config import CONFIG

# Output directory
OUTPUT_DIR = Path(CONFIG.get("outputs_dir", "outputs")) / "code"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Code Generation Implementations
# =============================================================================

class EnigmaCode:
    """Use Enigma's own model for code generation."""
    
    def __init__(self, model_name: str = "sacrifice"):
        self.model_name = model_name
        self.engine = None
        self.is_loaded = False
    
    def load(self) -> bool:
        try:
            from ...core.inference import InferenceEngine
            self.engine = InferenceEngine(model_name=self.model_name)
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Failed to load Enigma model: {e}")
            return False
    
    def unload(self):
        self.engine = None
        self.is_loaded = False
    
    def generate(self, prompt: str, language: str = "python", **kwargs) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}
        
        try:
            start = time.time()
            
            # Format prompt for code generation
            code_prompt = f"Write {language} code:\n{prompt}\n\n```{language}\n"
            
            result = self.engine.generate(
                code_prompt,
                max_tokens=kwargs.get('max_tokens', 500),
                temperature=kwargs.get('temperature', 0.3),
            )
            
            return {
                "success": True,
                "code": result,
                "duration": time.time() - start
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class OpenAICode:
    """OpenAI GPT for code generation (CLOUD - requires API key)."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = None
        self.is_loaded = False
    
    def load(self) -> bool:
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.is_loaded = bool(self.api_key)
            return self.is_loaded
        except ImportError:
            print("Install: pip install openai")
            return False
    
    def unload(self):
        self.client = None
        self.is_loaded = False
    
    def generate(self, prompt: str, language: str = "python", **kwargs) -> Dict[str, Any]:
        if not self.is_loaded or not self.client:
            return {"success": False, "error": "Not loaded or missing API key"}
        
        try:
            start = time.time()
            
            system_prompt = f"You are an expert {language} programmer. Write clean, efficient, well-documented code. Return only the code, no explanations."
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get('temperature', 0.3),
                max_tokens=kwargs.get('max_tokens', 2000),
            )
            
            code = response.choices[0].message.content
            
            return {
                "success": True,
                "code": code,
                "duration": time.time() - start,
                "tokens": response.usage.total_tokens
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def explain(self, code: str) -> Dict[str, Any]:
        """Explain what code does."""
        if not self.is_loaded or not self.client:
            return {"success": False, "error": "Not loaded"}
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Explain code clearly and concisely."},
                    {"role": "user", "content": f"Explain this code:\n\n{code}"}
                ],
                temperature=0.3,
            )
            
            return {
                "success": True,
                "explanation": response.choices[0].message.content
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# GUI Components
# =============================================================================

_providers = {
    'local': None,
    'openai': None,
}


def get_provider(name: str):
    """Get or create a provider instance."""
    global _providers
    
    if name == 'local' and _providers['local'] is None:
        _providers['local'] = EnigmaCode()
    elif name == 'openai' and _providers['openai'] is None:
        _providers['openai'] = OpenAICode()
    
    return _providers.get(name)


class CodeGenerationWorker(QThread):
    """Background worker for code generation."""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    
    def __init__(self, prompt, language, provider_name, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.language = language
        self.provider_name = provider_name
    
    def run(self):
        try:
            self.progress.emit(10)
            
            provider = get_provider(self.provider_name)
            if provider is None:
                self.finished.emit({"success": False, "error": "Unknown provider"})
                return
            
            if not provider.is_loaded:
                self.progress.emit(20)
                if not provider.load():
                    self.finished.emit({"success": False, "error": "Failed to load provider"})
                    return
            
            self.progress.emit(40)
            
            result = provider.generate(self.prompt, language=self.language)
            
            self.progress.emit(100)
            self.finished.emit(result)
            
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)})


class CodeTab(QWidget):
    """Tab for code generation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.worker = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Code Generation")
        header.setFont(QFont('Arial', 14, QFont.Bold))
        header.setStyleSheet("color: #3498db;")
        layout.addWidget(header)
        
        # Provider and language selection
        settings_layout = QHBoxLayout()
        
        settings_layout.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems([
            'Local (Enigma Model)',
            'OpenAI (GPT-4) - Cloud'
        ])
        settings_layout.addWidget(self.provider_combo)
        
        settings_layout.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems([
            'python', 'javascript', 'java', 'cpp', 'go', 'rust',
            'html', 'css', 'sql', 'bash', 'typescript'
        ])
        settings_layout.addWidget(self.language_combo)
        
        settings_layout.addStretch()
        layout.addLayout(settings_layout)
        
        # Prompt input
        prompt_group = QGroupBox("What code do you need?")
        prompt_layout = QVBoxLayout()
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setMaximumHeight(100)
        self.prompt_input.setPlaceholderText(
            "Describe what you want the code to do...\n"
            "Example: A function that sorts a list of dictionaries by a specific key"
        )
        prompt_layout.addWidget(self.prompt_input)
        
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Generate Code")
        self.generate_btn.setStyleSheet("background-color: #3498db; font-weight: bold; padding: 10px;")
        self.generate_btn.clicked.connect(self._generate_code)
        btn_layout.addWidget(self.generate_btn)
        
        self.copy_btn = QPushButton("Copy to Clipboard")
        self.copy_btn.clicked.connect(self._copy_code)
        btn_layout.addWidget(self.copy_btn)
        
        self.save_btn = QPushButton("Save to File")
        self.save_btn.clicked.connect(self._save_code)
        btn_layout.addWidget(self.save_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Status
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Output
        output_group = QGroupBox("Generated Code")
        output_layout = QVBoxLayout()
        
        self.code_output = QPlainTextEdit()
        self.code_output.setReadOnly(True)
        self.code_output.setPlaceholderText("Generated code will appear here...")
        
        # Use monospace font
        font = QFont("Courier New", 10)
        font.setStyleHint(QFont.Monospace)
        self.code_output.setFont(font)
        
        output_layout.addWidget(self.code_output)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group, stretch=1)
    
    def _get_provider_name(self) -> str:
        text = self.provider_combo.currentText()
        if 'Local' in text:
            return 'local'
        elif 'OpenAI' in text:
            return 'openai'
        return 'local'
    
    def _generate_code(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "No Prompt", "Please describe what code you need")
            return
        
        provider_name = self._get_provider_name()
        language = self.language_combo.currentText()
        
        self.generate_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Generating...")
        
        self.worker = CodeGenerationWorker(prompt, language, provider_name)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_generation_complete)
        self.worker.start()
    
    def _on_generation_complete(self, result: dict):
        self.generate_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        if result.get("success"):
            code = result.get("code", "")
            duration = result.get("duration", 0)
            
            self.code_output.setPlainText(code)
            self.status_label.setText(f"Generated in {duration:.1f}s")
        else:
            error = result.get("error", "Unknown error")
            self.status_label.setText(f"Error: {error}")
            self.code_output.setPlainText(f"# Error: {error}")
    
    def _copy_code(self):
        from PyQt5.QtWidgets import QApplication
        code = self.code_output.toPlainText()
        if code:
            QApplication.clipboard().setText(code)
            self.status_label.setText("Code copied to clipboard!")
    
    def _save_code(self):
        code = self.code_output.toPlainText()
        if not code:
            return
        
        lang = self.language_combo.currentText()
        ext_map = {
            'python': '.py', 'javascript': '.js', 'java': '.java',
            'cpp': '.cpp', 'go': '.go', 'rust': '.rs',
            'html': '.html', 'css': '.css', 'sql': '.sql',
            'bash': '.sh', 'typescript': '.ts'
        }
        ext = ext_map.get(lang, '.txt')
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Code",
            str(Path.home() / f"generated_code{ext}"),
            f"{lang.title()} Files (*{ext});;All Files (*.*)"
        )
        if path:
            Path(path).write_text(code)
            QMessageBox.information(self, "Saved", f"Code saved to:\n{path}")


def create_code_tab(parent) -> QWidget:
    """Factory function for creating the code tab."""
    if not HAS_PYQT:
        raise ImportError("PyQt5 is required for the Code Tab")
    return CodeTab(parent)
