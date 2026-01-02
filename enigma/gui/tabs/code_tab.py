"""
Code Generation Tab - Generate code using local or cloud models.
"""

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
        QPushButton, QFrame, QComboBox, QTextEdit, QProgressBar,
        QMessageBox, QFileDialog, QGroupBox, QPlainTextEdit
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont, QFontDatabase
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from pathlib import Path
from ...config import CONFIG


# Provider colors for UI badges
PROVIDER_COLORS = {
    'LOCAL': '#27ae60',
    'OPENAI': '#10a37f',
    'ANTHROPIC': '#d4a27f',
    'HUGGINGFACE': '#ffcc00',
}


class CodeGenerationWorker(QThread):
    """Background worker for code generation."""
    finished = pyqtSignal(str)  # Generated code
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, prompt, language, provider, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.language = language
        self.provider = provider
    
    def run(self):
        try:
            self.progress.emit(10)
            
            # Try to use module manager if available
            try:
                from ...modules.manager import ModuleManager
                manager = ModuleManager()
                
                if self.provider == 'LOCAL':
                    if manager.is_loaded('code_gen_local'):
                        module = manager.get_module('code_gen_local')
                        self.progress.emit(50)
                        result = module.generate(
                            self.prompt,
                            language=self.language
                        )
                        self.progress.emit(100)
                        self.finished.emit(result.get('code', ''))
                        return
                    elif manager.is_loaded('inference'):
                        # Use main Enigma model for code
                        inference = manager.get_module('inference')
                        self.progress.emit(50)
                        code_prompt = f"Write {self.language} code: {self.prompt}"
                        result = inference.generate(code_prompt)
                        self.progress.emit(100)
                        self.finished.emit(result)
                        return
                else:
                    if manager.is_loaded('code_gen_api'):
                        module = manager.get_module('code_gen_api')
                        self.progress.emit(50)
                        result = module.generate(
                            self.prompt,
                            language=self.language
                        )
                        self.progress.emit(100)
                        self.finished.emit(result.get('code', ''))
                        return
            except ImportError:
                pass
            
            # Mock generation for demo
            self.progress.emit(50)
            import time
            time.sleep(0.5)
            
            mock_code = f"# Generated {self.language} code for: {self.prompt}\n"
            mock_code += f"# (Load code_gen_local or inference module for real generation)\n\n"
            mock_code += f"def example():\n    pass\n"
            
            self.progress.emit(100)
            self.finished.emit(mock_code)
            
        except Exception as e:
            self.error.emit(str(e))


class ProviderCard(QFrame):
    """Card displaying a single code provider."""
    
    def __init__(self, name: str, info: dict, parent=None):
        super().__init__(parent)
        self.provider_name = name
        self.provider_info = info
        self.setup_ui()
    
    def setup_ui(self):
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(1)
        self.setMaximumHeight(100)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        
        # Header
        header = QHBoxLayout()
        
        name_label = QLabel(self.provider_info.get('name', self.provider_name))
        name_label.setFont(QFont('Arial', 10, QFont.Bold))
        header.addWidget(name_label)
        
        header.addStretch()
        
        # Provider badge
        provider = self.provider_info.get('provider', 'UNKNOWN')
        color = PROVIDER_COLORS.get(provider, '#666')
        provider_label = QLabel(provider)
        provider_label.setStyleSheet(
            f"background-color: {color}; color: white; "
            f"padding: 2px 6px; border-radius: 3px; font-size: 9px;"
        )
        header.addWidget(provider_label)
        
        layout.addLayout(header)
        
        # Description
        desc = self.provider_info.get('description', 'No description')
        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #888; font-size: 9px;")
        layout.addWidget(desc_label)


class CodeTab(QWidget):
    """Tab for code generation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.worker = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        
        # Left: Provider list
        left = QWidget()
        left.setMaximumWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        type_label = QLabel("Code Providers")
        type_label.setFont(QFont('Arial', 12, QFont.Bold))
        type_label.setStyleSheet("color: #3498db;")
        left_layout.addWidget(type_label)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        cards_widget = QWidget()
        cards_layout = QVBoxLayout(cards_widget)
        cards_layout.setSpacing(8)
        
        # Available providers
        providers = {
            'enigma_code': {
                'name': 'Enigma Code',
                'description': 'Use your trained Enigma model for code generation.',
                'requirements': [],
                'provider': 'LOCAL',
            },
            'openai_code': {
                'name': 'GPT-4 Code',
                'description': 'OpenAI GPT-4 for complex code tasks.',
                'requirements': ['openai'],
                'provider': 'OPENAI',
                'needs_api_key': True,
            },
            'anthropic_code': {
                'name': 'Claude Code',
                'description': 'Anthropic Claude for code generation.',
                'requirements': ['anthropic'],
                'provider': 'ANTHROPIC',
                'needs_api_key': True,
            },
        }
        
        for name, info in providers.items():
            card = ProviderCard(name, info)
            cards_layout.addWidget(card)
        
        cards_layout.addStretch()
        scroll.setWidget(cards_widget)
        left_layout.addWidget(scroll)
        
        layout.addWidget(left)
        
        # Right: Generation panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        # Provider selection
        provider_row = QHBoxLayout()
        provider_row.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(['Enigma (Local)', 'OpenAI (GPT-4)', 'Anthropic (Claude)'])
        provider_row.addWidget(self.provider_combo)
        
        provider_row.addWidget(QLabel("Language:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems([
            'python', 'javascript', 'typescript', 'rust', 'go',
            'java', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin'
        ])
        provider_row.addWidget(self.lang_combo)
        
        provider_row.addStretch()
        right_layout.addLayout(provider_row)
        
        # Prompt input
        prompt_label = QLabel("Describe what code you need:")
        right_layout.addWidget(prompt_label)
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setMaximumHeight(100)
        self.prompt_input.setPlaceholderText(
            "Example: Write a function that calculates fibonacci numbers recursively with memoization..."
        )
        right_layout.addWidget(self.prompt_input)
        
        # Generate button
        btn_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Code")
        self.generate_btn.setStyleSheet("background-color: #3498db; font-weight: bold;")
        self.generate_btn.clicked.connect(self._generate_code)
        btn_layout.addWidget(self.generate_btn)
        
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self._copy_code)
        btn_layout.addWidget(self.copy_btn)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_code)
        btn_layout.addWidget(self.save_btn)
        
        right_layout.addLayout(btn_layout)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        right_layout.addWidget(self.progress)
        
        # Result area - code output
        result_label = QLabel("Generated Code:")
        right_layout.addWidget(result_label)
        
        self.code_output = QPlainTextEdit()
        self.code_output.setReadOnly(True)
        self.code_output.setPlaceholderText("Generated code will appear here...")
        # Use monospace font for code
        font = QFont("Monospace")
        font.setStyleHint(QFont.TypeWriter)
        font.setPointSize(10)
        self.code_output.setFont(font)
        self.code_output.setStyleSheet(
            "background-color: #1e1e1e; color: #d4d4d4; "
            "border-radius: 4px; padding: 8px;"
        )
        right_layout.addWidget(self.code_output, stretch=1)
        
        layout.addWidget(right, stretch=1)
    
    def _generate_code(self):
        """Generate code."""
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "No Prompt", "Please describe what code you need")
            return
        
        # Determine provider
        provider_text = self.provider_combo.currentText()
        if 'Local' in provider_text or 'Enigma' in provider_text:
            provider = 'LOCAL'
        else:
            provider = 'API'
        
        self.generate_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        self.worker = CodeGenerationWorker(
            prompt,
            self.lang_combo.currentText(),
            provider
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_generation_complete)
        self.worker.error.connect(self._on_generation_error)
        self.worker.start()
    
    def _on_generation_complete(self, code: str):
        """Handle generation completion."""
        self.generate_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        self.code_output.setPlainText(code)
        self.copy_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
    
    def _on_generation_error(self, error: str):
        """Handle generation error."""
        self.generate_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.code_output.setPlainText(f"Error: {error}")
        QMessageBox.warning(self, "Generation Failed", f"Error: {error}")
    
    def _copy_code(self):
        """Copy generated code to clipboard."""
        from PyQt5.QtWidgets import QApplication
        code = self.code_output.toPlainText()
        if code:
            QApplication.clipboard().setText(code)
            QMessageBox.information(self, "Copied", "Code copied to clipboard!")
    
    def _save_code(self):
        """Save the generated code to a file."""
        code = self.code_output.toPlainText()
        if not code:
            return
        
        # Get file extension based on language
        lang = self.lang_combo.currentText()
        extensions = {
            'python': '.py', 'javascript': '.js', 'typescript': '.ts',
            'rust': '.rs', 'go': '.go', 'java': '.java',
            'c++': '.cpp', 'c#': '.cs', 'ruby': '.rb',
            'php': '.php', 'swift': '.swift', 'kotlin': '.kt'
        }
        ext = extensions.get(lang, '.txt')
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Code",
            str(Path.home() / f"generated_code{ext}"),
            f"Code Files (*{ext});;All Files (*)"
        )
        if path:
            with open(path, 'w') as f:
                f.write(code)
            QMessageBox.information(self, "Saved", f"Code saved to:\n{path}")


def create_code_tab(parent) -> QWidget:
    """Factory function for creating the code tab."""
    return CodeTab(parent)


if not HAS_PYQT:
    class CodeTab:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyQt5 is required for the Code Tab")
    
    def create_code_tab(parent):
        raise ImportError("PyQt5 is required for the Code Tab")
