"""
Code Generation Tab - Generate code using local or cloud models.

Providers:
  - LOCAL: Uses Enigma AI Engine's own model (or built-in template fallback)
  - OPENAI: GPT-4 (requires openai, API key)
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from PyQt5.QtCore import QThread, pyqtSignal
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import (
        QCheckBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from ...config import CONFIG
from .output_helpers import open_file_in_explorer, open_folder
from .shared_components import NoScrollComboBox

# Output directory
OUTPUT_DIR = Path(CONFIG.get("outputs_dir", "outputs")) / "code"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Code Generation Implementations
# =============================================================================

class ForgeCode:
    """Use Enigma AI Engine's own model for code generation with built-in fallback."""
    
    def __init__(self, model_name: str = "small_enigma_engine"):
        self.model_name = model_name
        self.engine = None
        self.is_loaded = False
        self._using_builtin = False
        self._builtin_code = None
    
    def load(self) -> bool:
        # Try Enigma AI Engine model first
        try:
            from ...core.model_registry import ModelRegistry
            registry = ModelRegistry()
            model, config = registry.load_model(self.model_name)
            
            from ...core.inference import EnigmaEngine

            # Create engine with loaded model
            self.engine = EnigmaEngine.__new__(EnigmaEngine)
            import torch
            self.engine.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.engine.model = model
            self.engine.model.to(self.engine.device)
            self.engine.model.eval()
            
            from ...core.tokenizer import load_tokenizer
            self.engine.tokenizer = load_tokenizer()
            self.engine.use_half = False
            self.engine.enable_tools = False
            self.engine.module_manager = None
            self.engine._tool_executor = None
            
            self.is_loaded = True
            self._using_builtin = False
            return True
        except Exception as e:
            logger.debug(f"Forge model not available: {e}")
        
        # Fall back to built-in template-based code generation
        try:
            from ...builtin import BuiltinCodeGen
            self._builtin_code = BuiltinCodeGen()
            if self._builtin_code.load():
                self.is_loaded = True
                self._using_builtin = True
                logger.info("Using built-in code generator (template-based)")
                return True
        except Exception as e:
            logger.debug(f"Built-in code gen failed: {e}")
        
        return False
    
    def unload(self):
        self.engine = None
        if self._builtin_code:
            self._builtin_code.unload()
            self._builtin_code = None
        self.is_loaded = False
        self._using_builtin = False
    
    def generate(self, prompt: str, language: str = "python", **kwargs) -> dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}
        
        if self._using_builtin:
            return self._builtin_code.generate(prompt, language=language)
        
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
            logger.debug("OpenAI not available - install: pip install openai")
            return False
    
    def unload(self):
        self.client = None
        self.is_loaded = False
    
    def generate(self, prompt: str, language: str = "python", **kwargs) -> dict[str, Any]:
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
    
    def explain(self, code: str) -> dict[str, Any]:
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
        _providers['local'] = ForgeCode()
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
            
            # Check if router has code assignments - use router if configured
            try:
                from ...core.tool_router import get_router
                router = get_router()
                assignments = router.get_assignments("code")
                
                if assignments:
                    self.progress.emit(30)
                    params = {
                        "prompt": str(self.prompt).strip() if self.prompt else "",
                        "language": self.language
                    }
                    result = router.execute_tool("code", params)
                    self.progress.emit(100)
                    self.finished.emit(result)
                    return
            except Exception as router_error:
                logger.debug(f"Router fallback: {router_error}")
            
            # Direct provider fallback
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
        
        # Register references on parent window for chat integration
        if parent:
            parent.code_prompt = self.prompt_input
            parent.code_tab = self
            parent._generate_code = self._generate_code
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Code Generation")
        header.setObjectName("header")
        layout.addWidget(header)
        
        # Output at TOP
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
        
        # Progress and Status
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Provider and language selection
        settings_layout = QHBoxLayout()
        
        settings_layout.addWidget(QLabel("Provider:"))
        self.provider_combo = NoScrollComboBox()
        self.provider_combo.addItems([
            'Local (Forge Model)',
            'OpenAI (GPT-4) - Cloud'
        ])
        self.provider_combo.setToolTip("Local: Uses your Enigma AI Engine model (free, offline)\nCloud: Uses OpenAI API (requires API key, better quality)")
        settings_layout.addWidget(self.provider_combo)
        
        settings_layout.addWidget(QLabel("Language:"))
        self.language_combo = NoScrollComboBox()
        self.language_combo.addItems([
            'python', 'javascript', 'java', 'cpp', 'go', 'rust',
            'html', 'css', 'sql', 'bash', 'typescript'
        ])
        self.language_combo.setToolTip("Select the programming language for the generated code")
        settings_layout.addWidget(self.language_combo)
        
        settings_layout.addStretch()
        layout.addLayout(settings_layout)
        
        # Code Style Learning section
        style_group = QGroupBox("Code Style (Optional)")
        style_group.setCheckable(True)
        style_group.setChecked(False)
        style_layout = QVBoxLayout(style_group)
        
        style_desc = QLabel("Learn code style from your project to match your conventions")
        style_desc.setStyleSheet("color: #6c7086; font-size: 11px;")
        style_layout.addWidget(style_desc)
        
        style_row = QHBoxLayout()
        self.style_path_input = QLineEdit()
        self.style_path_input.setPlaceholderText("Path to your project folder...")
        self.style_path_input.setToolTip("Select a folder to analyze for code style patterns")
        style_row.addWidget(self.style_path_input, stretch=1)
        
        browse_style_btn = QPushButton("Browse")
        browse_style_btn.clicked.connect(self._browse_style_folder)
        style_row.addWidget(browse_style_btn)
        
        analyze_btn = QPushButton("Learn Style")
        analyze_btn.setStyleSheet("background-color: #9b59b6; color: white;")
        analyze_btn.clicked.connect(self._analyze_code_style)
        analyze_btn.setToolTip("Analyze the selected folder to learn your code style")
        style_row.addWidget(analyze_btn)
        
        style_layout.addLayout(style_row)
        
        # Style status
        self.style_status = QLabel("No style learned yet")
        self.style_status.setStyleSheet("color: #6c7086; font-style: italic;")
        style_layout.addWidget(self.style_status)
        
        layout.addWidget(style_group)
        
        # Store learned style
        self._learned_style = None
        
        # Prompt input
        prompt_group = QGroupBox("What code do you need?")
        prompt_layout = QVBoxLayout()
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setMaximumHeight(60)
        self.prompt_input.setPlaceholderText(
            "Describe what you want the code to do..."
        )
        self.prompt_input.setToolTip("Describe what you want the code to do.\nBe specific about inputs, outputs, and functionality.\nExample: 'A function that sorts a list of numbers in ascending order'")
        prompt_layout.addWidget(self.prompt_input)
        
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        # Auto-open options
        auto_layout = QHBoxLayout()
        self.auto_open_file_cb = QCheckBox("Auto-open saved file in explorer")
        self.auto_open_file_cb.setChecked(True)
        self.auto_open_file_cb.setToolTip("Automatically open the saved file location after saving")
        auto_layout.addWidget(self.auto_open_file_cb)
        
        self.use_style_cb = QCheckBox("Apply learned code style")
        self.use_style_cb.setChecked(True)
        self.use_style_cb.setToolTip("Include your learned code style in generation prompts")
        auto_layout.addWidget(self.use_style_cb)
        
        auto_layout.addStretch()
        layout.addLayout(auto_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Generate Code")
        self.generate_btn.setStyleSheet("background-color: #3498db; font-weight: bold; padding: 10px;")
        self.generate_btn.clicked.connect(self._generate_code)
        self.generate_btn.setToolTip("Generate code based on your description (Ctrl+Enter)")
        btn_layout.addWidget(self.generate_btn)
        
        self.copy_btn = QPushButton("Copy to Clipboard")
        self.copy_btn.clicked.connect(self._copy_code)
        self.copy_btn.setToolTip("Copy the generated code to your clipboard")
        btn_layout.addWidget(self.copy_btn)
        
        self.save_btn = QPushButton("Save to File")
        self.save_btn.clicked.connect(self._save_code)
        self.save_btn.setToolTip("Save the generated code to a file")
        btn_layout.addWidget(self.save_btn)
        
        self.open_folder_btn = QPushButton("Output Folder")
        self.open_folder_btn.clicked.connect(lambda: open_folder(OUTPUT_DIR))
        self.open_folder_btn.setToolTip("Open the folder where generated code files are saved")
        btn_layout.addWidget(self.open_folder_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
    
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
        
        # Include learned code style in prompt
        style_context = self.get_style_context()
        if style_context:
            prompt = f"{style_context}\n\nTask: {prompt}"
        
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
            
            # Show if specialized model was used
            model_type = result.get("model_type", "")
            if model_type:
                self.status_label.setText(f"Generated in {duration:.1f}s (using {model_type})")
            else:
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
            
            # Auto-open file in explorer
            if self.auto_open_file_cb.isChecked():
                open_file_in_explorer(path)
    
    def _browse_style_folder(self):
        """Browse for a project folder to learn code style from."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Project Folder",
            str(Path.home())
        )
        if folder:
            self.style_path_input.setText(folder)
    
    def _analyze_code_style(self):
        """Analyze the selected folder to learn code style."""
        folder = self.style_path_input.text().strip()
        if not folder or not Path(folder).exists():
            QMessageBox.warning(self, "No Folder", "Please select a valid project folder")
            return
        
        self.style_status.setText("Analyzing code style...")
        self.style_status.setStyleSheet("color: #89b4fa;")
        
        try:
            from ...tools.code_style_analyzer import get_style_analyzer
            
            analyzer = get_style_analyzer()
            style_guide = analyzer.analyze_directory(Path(folder))
            
            self._learned_style = style_guide
            
            # Show summary
            summary = (
                f"Learned: {style_guide.indent_size} {style_guide.indent_style}, "
                f"{style_guide.function_style} functions, "
                f"{'with' if style_guide.has_type_hints else 'no'} type hints"
            )
            if style_guide.framework_hints:
                summary += f", {', '.join(style_guide.framework_hints[:3])}"
            
            self.style_status.setText(f"Style learned: {summary}")
            self.style_status.setStyleSheet("color: #a6e3a1;")
            
            # Also extract project context
            try:
                from ...tools.code_style_analyzer import extract_project_context
                self._project_context = extract_project_context(folder)
            except Exception:
                self._project_context = None
            
            QMessageBox.information(
                self, "Style Learned",
                f"Code style analyzed successfully!\n\n{style_guide.to_prompt_context()}"
            )
            
        except Exception as e:
            logger.exception("Failed to analyze code style")
            self.style_status.setText(f"Error: {str(e)}")
            self.style_status.setStyleSheet("color: #f38ba8;")
    
    def get_style_context(self) -> str:
        """Get the style context string for prompts."""
        parts = []
        
        # Add project context first
        if hasattr(self, '_project_context') and self._project_context:
            ctx = self._project_context.to_prompt_context()
            if ctx:
                parts.append(ctx)
        
        # Add style guide
        if self._learned_style and hasattr(self, 'use_style_cb') and self.use_style_cb.isChecked():
            parts.append(self._learned_style.to_prompt_context())
        
        return "\n\n".join(parts)


def create_code_tab(parent) -> QWidget:
    """Factory function for creating the code tab."""
    if not HAS_PYQT:
        raise ImportError("PyQt5 is required for the Code Tab")
    return CodeTab(parent)
