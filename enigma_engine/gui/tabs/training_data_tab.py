"""
Training Data Generator Tab - Generate training data using AI models.

Supports:
- API providers (Claude, OpenAI) for highest quality
- HuggingFace models for local/free generation
- Specialized trainer data generation for bootstrapping
"""

import json
import logging
import os
from pathlib import Path
from typing import List

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...config import CONFIG

logger = logging.getLogger(__name__)

# Available local models (sorted by speed/resource usage)
AVAILABLE_LOCAL_MODELS = {
    "TinyLlama 1.1B (Fastest)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-2 2.7B (Fast)": "microsoft/phi-2",
    "Phi-3-mini 3.8B (Recommended)": "microsoft/Phi-3-mini-4k-instruct",
    "Mistral 7B (High Quality)": "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen2 7B (Good Alternative)": "Qwen/Qwen2-7B-Instruct",
}

# API providers
API_PROVIDERS = {
    "Claude (Anthropic)": "claude",
    "GPT-4 (OpenAI)": "openai",
}

# Router positions for trainer data generation
ROUTER_POSITIONS = [
    "router",
    "vision", 
    "code",
    "avatar",
    "chat",
    "math",
    "search",
    "audio",
    "video",
]


class GeneratorWorker(QThread):
    """Background worker for generating training data with local models."""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(
        self,
        model_id: str,
        topics: List[str],
        data_type: str,
        count_per_topic: int,
        style: str,
        use_4bit: bool,
    ):
        super().__init__()
        self.model_id = model_id
        self.topics = topics
        self.data_type = data_type
        self.count_per_topic = count_per_topic
        self.style = style
        self.use_4bit = use_4bit
        self._stop_requested = False
    
    def stop(self):
        """Request the worker to stop."""
        self._stop_requested = True
    
    def run(self):
        try:
            # Import here to avoid loading heavy deps on tab creation
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))
            
            from generate_training_data import TrainingDataGenerator
            
            self.progress.emit(f"Loading model: {self.model_id}")
            generator = TrainingDataGenerator(
                model_id=self.model_id,
                use_4bit=self.use_4bit,
            )
            
            if not generator.load():
                self.error.emit("Failed to load model. Check your internet connection and try again.")
                return
            
            all_data = []
            
            for i, topic in enumerate(self.topics):
                if self._stop_requested:
                    self.progress.emit("Generation stopped by user")
                    break
                
                self.progress.emit(f"Generating for topic {i+1}/{len(self.topics)}: {topic}")
                
                if self.data_type == "qa":
                    data = generator.generate_qa_pairs(
                        topic,
                        count=self.count_per_topic,
                        style=self.style,
                    )
                elif self.data_type == "conversations":
                    data = generator.generate_conversations(
                        topic,
                        count=self.count_per_topic,
                        turns=4,
                    )
                elif self.data_type == "instructions":
                    data = generator.generate_instructions(
                        topic,
                        count=self.count_per_topic,
                    )
                else:
                    data = []
                
                all_data.extend(data)
                self.progress.emit(f"Generated {len(data)} items for '{topic}'")
            
            self.finished.emit(all_data)
            
        except ImportError as e:
            self.error.emit(
                f"Missing dependencies. Please install:\n"
                f"pip install transformers accelerate bitsandbytes\n\n"
                f"Error: {e}"
            )
        except Exception as e:
            logger.exception("Training data generation failed")
            self.error.emit(str(e))


class APIGeneratorWorker(QThread):
    """Background worker for generating training data via API (Claude/OpenAI)."""
    finished = pyqtSignal(str)  # Returns raw formatted text
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(
        self,
        provider: str,  # "claude" or "openai"
        api_key: str,
        positions: List[str],  # Router positions to generate data for
        examples_per_position: int,
        generation_type: str,  # "trainer" or "direct"
    ):
        super().__init__()
        self.provider = provider
        self.api_key = api_key
        self.positions = positions
        self.examples_per_position = examples_per_position
        self.generation_type = generation_type
        self._stop_requested = False
    
    def stop(self):
        self._stop_requested = True
    
    def run(self):
        try:
            all_data = []
            
            for i, position in enumerate(self.positions):
                if self._stop_requested:
                    break
                
                self.progress.emit(f"Generating {position} data ({i+1}/{len(self.positions)})...")
                
                if self.generation_type == "trainer":
                    data = self._generate_trainer_data(position)
                else:
                    data = self._generate_direct_data(position)
                
                if data:
                    all_data.append(data)
                    self.progress.emit(f"Generated {position} examples")
            
            self.finished.emit("\n\n".join(all_data))
            
        except Exception as e:
            logger.exception("API generation failed")
            self.error.emit(str(e))
    
    def _generate_trainer_data(self, position: str) -> str:
        """Generate trainer training data (teaches model to generate training data)."""
        
        # Position-specific prompts
        position_prompts = {
            "router": """Generate {count} examples of training data that teaches an AI to classify user intents.
Format each example EXACTLY like this:
GENERATE_FOR: router
DATA:
INPUT: <user message> | INTENT: <category>

Categories: image, video, audio, code, chat, search, avatar, math

Generate diverse examples covering all categories.""",
            
            "vision": """Generate {count} examples of training data that teaches an AI to describe images.
Format each example EXACTLY like this:
GENERATE_FOR: vision
DATA:
IMAGE: <brief image description> | CAPTION: <detailed 2-3 sentence description>

Generate diverse examples: photos, artwork, screenshots, scenes, objects.""",
            
            "code": """Generate {count} examples of training data that teaches an AI to write code.
Format each example EXACTLY like this:
GENERATE_FOR: code
DATA:
TASK: <programming task description>
CODE:
<complete working code>

Generate diverse examples: Python, JavaScript, algorithms, classes, utilities.""",
            
            "avatar": """Generate {count} examples of training data that teaches an AI to control avatar bones.
Format each example EXACTLY like this:
GENERATE_FOR: avatar  
DATA:
COMMAND: <natural language command> | BONES: {{"bone_name": {{"rotation": [x,y,z], "speed": 0.5}}, "action": "<action_type>"}}

Include: head, arms, hands, spine movements. Actions: wave, nod, point, gesture, dance, bow.""",
            
            "chat": """Generate {count} examples of conversational training data.
Format each example EXACTLY like this:
GENERATE_FOR: chat
DATA:
Q: <user question or statement>
A: <helpful, natural response>

Generate diverse topics: greetings, questions, help requests, casual chat.""",
            
            "math": """Generate {count} examples of math problem solving training data.
Format each example EXACTLY like this:
GENERATE_FOR: math
DATA:
PROBLEM: <math problem>
SOLUTION:
<step by step solution>
Answer: <final answer>

Include: arithmetic, algebra, percentages, word problems.""",
        }
        
        prompt = position_prompts.get(position, position_prompts["chat"])
        prompt = prompt.format(count=self.examples_per_position)
        
        return self._call_api(prompt)
    
    def _generate_direct_data(self, position: str) -> str:
        """Generate direct training data for a position."""
        prompts = {
            "router": f"""Generate {self.examples_per_position} intent classification training examples.
Format: Q: <user message>\\nA: [E:tool]<intent>
Intents: image, video, audio, code, chat, search, avatar
Make them diverse and natural.""",
            
            "code": f"""Generate {self.examples_per_position} code generation training examples.
Format: Q: <task>\\nA: <complete code>
Include Python, JavaScript, algorithms. Make code correct and complete.""",
            
            "chat": f"""Generate {self.examples_per_position} Q&A training pairs.
Format: Q: <question>\\nA: <answer>
Diverse topics: general knowledge, help, conversation, explanations.""",
        }
        
        prompt = prompts.get(position, prompts["chat"])
        return self._call_api(prompt)
    
    def _call_api(self, prompt: str) -> str:
        """Call the appropriate API."""
        if self.provider == "claude":
            return self._call_claude(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _call_claude(self, prompt: str) -> str:
        """Call Claude API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        client = openai.OpenAI(api_key=self.api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
        )
        
        return response.choices[0].message.content


class TrainingDataTab(QWidget):
    """Tab for generating training data using AI models (API or local)."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._api_worker = None
        self._generated_data: List[dict] = []
        self._generated_text: str = ""
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Header
        header = QLabel("Training Data Generator")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        layout.addWidget(header)
        
        desc = QLabel(
            "Generate training data for Enigma models. Use API for highest quality (trainer bootstrap), "
            "or local models for free generation."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a6adc8;")
        layout.addWidget(desc)
        
        # Main content splitter
        splitter = QSplitter()
        
        # Left panel - Settings with tabs
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)
        
        # Mode tabs
        self._mode_tabs = QTabWidget()
        
        # ==================== API TAB ====================
        api_widget = QWidget()
        api_layout = QVBoxLayout(api_widget)
        
        # API info
        api_info = QLabel(
            "Use Claude and/or GPT-4 for highest quality data.\n"
            "API keys are loaded from Settings > API Keys."
        )
        api_info.setStyleSheet("color: #a6e3a1; font-style: italic;")
        api_layout.addWidget(api_info)
        
        # API Keys status - show which keys are configured
        keys_group = QGroupBox("API Keys (from Settings)")
        keys_layout = QVBoxLayout(keys_group)
        
        # Load keys from environment (set by Settings tab)
        claude_key = os.environ.get("ANTHROPIC_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        
        # Status display
        self._claude_status = QLabel(
            "Claude: Configured" if claude_key else "Claude: Not configured"
        )
        self._claude_status.setStyleSheet(
            "color: #a6e3a1;" if claude_key else "color: #f38ba8;"
        )
        keys_layout.addWidget(self._claude_status)
        
        self._openai_status = QLabel(
            "OpenAI: Configured" if openai_key else "OpenAI: Not configured"
        )
        self._openai_status.setStyleSheet(
            "color: #a6e3a1;" if openai_key else "color: #f38ba8;"
        )
        keys_layout.addWidget(self._openai_status)
        
        # Link to Settings
        settings_link = QPushButton("Configure API Keys in Settings")
        settings_link.setStyleSheet("color: #89b4fa; text-decoration: underline; border: none; text-align: left;")
        settings_link.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_link.clicked.connect(self._go_to_settings)
        keys_layout.addWidget(settings_link)
        
        # Provider preference
        pref_layout = QHBoxLayout()
        pref_layout.addWidget(QLabel("Prefer:"))
        self._provider_combo = QComboBox()
        self._provider_combo.addItem("Claude (Recommended)", "claude")
        self._provider_combo.addItem("OpenAI GPT-4", "openai")
        self._provider_combo.addItem("Both (alternate)", "both")
        pref_layout.addWidget(self._provider_combo)
        pref_layout.addStretch()
        keys_layout.addLayout(pref_layout)
        
        api_layout.addWidget(keys_group)
        
        # Generation type
        gen_type_group = QGroupBox("Generation Type")
        gen_type_layout = QVBoxLayout(gen_type_group)
        
        self._trainer_data_radio = QCheckBox("Trainer Data (teaches AI to generate training data)")
        self._trainer_data_radio.setChecked(True)
        self._trainer_data_radio.setToolTip(
            "Generate data that teaches the Trainer AI how to create training data.\n"
            "Use this to bootstrap the training system."
        )
        gen_type_layout.addWidget(self._trainer_data_radio)
        
        api_layout.addWidget(gen_type_group)
        
        # Position selection
        positions_group = QGroupBox("Router Positions")
        positions_layout = QVBoxLayout(positions_group)
        
        positions_layout.addWidget(QLabel("Generate data for:"))
        self._position_checkboxes = {}
        for pos in ROUTER_POSITIONS:
            cb = QCheckBox(pos.title())
            # Default: router and chat checked
            cb.setChecked(pos in ["router", "chat"])
            self._position_checkboxes[pos] = cb
            positions_layout.addWidget(cb)
        
        api_layout.addWidget(positions_group)
        
        # Count
        api_count_layout = QHBoxLayout()
        api_count_layout.addWidget(QLabel("Examples per position:"))
        self._api_count_spin = QSpinBox()
        self._api_count_spin.setRange(5, 100)
        self._api_count_spin.setValue(20)
        api_count_layout.addWidget(self._api_count_spin)
        api_count_layout.addStretch()
        api_layout.addLayout(api_count_layout)
        
        # API Generate button
        self._api_generate_btn = QPushButton("Generate with API")
        self._api_generate_btn.setStyleSheet(
            "background: #a6e3a1; color: #1e1e2e; font-weight: bold; padding: 10px;"
        )
        self._api_generate_btn.clicked.connect(self._on_api_generate)
        api_layout.addWidget(self._api_generate_btn)
        
        api_layout.addStretch()
        self._mode_tabs.addTab(api_widget, "API (Recommended)")
        
        # ==================== LOCAL TAB ====================
        local_widget = QWidget()
        local_layout = QVBoxLayout(local_widget)
        
        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)
        
        self._model_combo = QComboBox()
        for name, model_id in AVAILABLE_LOCAL_MODELS.items():
            self._model_combo.addItem(name, model_id)
        self._model_combo.setCurrentIndex(2)  # Default: Phi-3-mini
        model_layout.addWidget(self._model_combo)
        
        self._use_4bit = QCheckBox("Use 4-bit quantization (saves memory)")
        self._use_4bit.setChecked(True)
        model_layout.addWidget(self._use_4bit)
        
        # Custom model input
        custom_layout = QHBoxLayout()
        custom_layout.addWidget(QLabel("Or enter model ID:"))
        self._custom_model = QLineEdit()
        self._custom_model.setPlaceholderText("e.g., meta-llama/Llama-2-7b-chat-hf")
        custom_layout.addWidget(self._custom_model)
        model_layout.addLayout(custom_layout)
        
        local_layout.addWidget(model_group)
        
        # Topics input
        topics_group = QGroupBox("Topics")
        topics_layout = QVBoxLayout(topics_group)
        
        topics_layout.addWidget(QLabel("Enter topics (one per line):"))
        self._topics_input = QTextEdit()
        self._topics_input.setPlaceholderText(
            "Python programming\n"
            "Machine learning basics\n"
            "Data structures\n"
            "Web development"
        )
        self._topics_input.setMaximumHeight(120)
        topics_layout.addWidget(self._topics_input)
        
        local_layout.addWidget(topics_group)
        
        # Generation settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Data type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Data Type:"))
        self._type_combo = QComboBox()
        self._type_combo.addItem("Q&A Pairs", "qa")
        self._type_combo.addItem("Conversations", "conversations")
        self._type_combo.addItem("Instructions", "instructions")
        type_layout.addWidget(self._type_combo)
        type_layout.addStretch()
        settings_layout.addLayout(type_layout)
        
        # Style
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("Style:"))
        self._style_combo = QComboBox()
        self._style_combo.addItem("Informative", "informative")
        self._style_combo.addItem("Conversational", "conversational")
        self._style_combo.addItem("Technical", "technical")
        style_layout.addWidget(self._style_combo)
        style_layout.addStretch()
        settings_layout.addLayout(style_layout)
        
        # Count per topic
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Items per topic:"))
        self._count_spin = QSpinBox()
        self._count_spin.setRange(1, 100)
        self._count_spin.setValue(10)
        count_layout.addWidget(self._count_spin)
        count_layout.addStretch()
        settings_layout.addLayout(count_layout)
        
        local_layout.addWidget(settings_group)
        
        # Control buttons
        local_btn_layout = QHBoxLayout()
        
        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setStyleSheet(
            "background: #89b4fa; color: #1e1e2e; font-weight: bold; padding: 10px;"
        )
        self._generate_btn.clicked.connect(self._on_generate)
        local_btn_layout.addWidget(self._generate_btn)
        
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        local_btn_layout.addWidget(self._stop_btn)
        
        local_layout.addLayout(local_btn_layout)
        
        # Progress (shared between tabs)
        local_layout.addStretch()
        self._mode_tabs.addTab(local_widget, "Local (Free)")
        
        # Add tabs to left panel
        left_layout.addWidget(self._mode_tabs)
        
        # Shared progress bar and status
        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setMaximum(0)
        self._progress.setVisible(False)
        left_layout.addWidget(self._progress)
        
        self._status = QLabel("")
        self._status.setStyleSheet("color: #a6adc8;")
        left_layout.addWidget(self._status)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Preview & Save
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 0, 0)
        
        preview_group = QGroupBox("Preview Generated Data")
        preview_layout = QVBoxLayout(preview_group)
        
        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setPlaceholderText("Generated data will appear here...")
        preview_layout.addWidget(self._preview)
        
        # Stats
        self._stats_label = QLabel("No data generated yet")
        self._stats_label.setStyleSheet("color: #a6adc8;")
        preview_layout.addWidget(self._stats_label)
        
        right_layout.addWidget(preview_group)
        
        # Save options
        save_group = QGroupBox("Save")
        save_layout = QVBoxLayout(save_group)
        
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self._format_combo = QComboBox()
        self._format_combo.addItem("Enigma Training Format (.txt)", "enigma")
        self._format_combo.addItem("JSON Lines (.jsonl)", "jsonl")
        self._format_combo.addItem("JSON (.json)", "json")
        format_layout.addWidget(self._format_combo)
        save_layout.addLayout(format_layout)
        
        self._save_btn = QPushButton("Save to File")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        save_layout.addWidget(self._save_btn)
        
        self._append_btn = QPushButton("Append to Existing")
        self._append_btn.setEnabled(False)
        self._append_btn.clicked.connect(self._on_append)
        save_layout.addWidget(self._append_btn)
        
        right_layout.addWidget(save_group)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
    
    def _go_to_settings(self):
        """Navigate to Settings tab to configure API keys."""
        try:
            main_window = self.window()
            if hasattr(main_window, '_nav_buttons'):
                # Find and click Settings nav button
                for btn in main_window._nav_buttons:
                    if btn.text() == "Settings":
                        btn.click()
                        return
            # Fallback: try content_stack
            if hasattr(main_window, '_content_stack'):
                for i in range(main_window._content_stack.count()):
                    widget = main_window._content_stack.widget(i)
                    if widget and widget.__class__.__name__ == 'SettingsTab':
                        main_window._content_stack.setCurrentWidget(widget)
                        return
        except Exception as e:
            logger.debug(f"Could not navigate to Settings: {e}")
            QMessageBox.information(
                self, "API Keys",
                "Go to Settings tab and scroll down to 'API Keys' section\n"
                "to configure your Claude and OpenAI API keys."
            )
    
    def _refresh_api_status(self):
        """Refresh the API key status display."""
        claude_key = os.environ.get("ANTHROPIC_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        
        self._claude_status.setText(
            "Claude: Configured" if claude_key else "Claude: Not configured"
        )
        self._claude_status.setStyleSheet(
            "color: #a6e3a1;" if claude_key else "color: #f38ba8;"
        )
        
        self._openai_status.setText(
            "OpenAI: Configured" if openai_key else "OpenAI: Not configured"
        )
        self._openai_status.setStyleSheet(
            "color: #a6e3a1;" if openai_key else "color: #f38ba8;"
        )
    
    def _on_api_generate(self):
        """Start generating training data using API (Claude and/or GPT-4)."""
        # Refresh status to pick up any newly saved keys
        self._refresh_api_status()
        
        # Get API keys from environment (set by Settings tab)
        claude_key = os.environ.get("ANTHROPIC_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        
        provider_pref = self._provider_combo.currentData()
        
        # Determine which provider(s) to use
        if provider_pref == "both":
            if not claude_key and not openai_key:
                QMessageBox.warning(
                    self, "API Key Required",
                    "No API keys configured.\n\n"
                    "Go to Settings > API Keys to add your Claude and/or OpenAI keys."
                )
                return
            # Use whichever keys are available
            providers_to_use = []
            if claude_key:
                providers_to_use.append(("claude", claude_key))
            if openai_key:
                providers_to_use.append(("openai", openai_key))
        elif provider_pref == "claude":
            if not claude_key:
                QMessageBox.warning(
                    self, "API Key Required",
                    "Claude API key not configured.\n\n"
                    "Go to Settings > API Keys to add your Claude key."
                )
                return
            providers_to_use = [("claude", claude_key)]
        else:  # openai
            if not openai_key:
                QMessageBox.warning(
                    self, "API Key Required",
                    "OpenAI API key not configured.\n\n"
                    "Go to Settings > API Keys to add your OpenAI key."
                )
                return
            providers_to_use = [("openai", openai_key)]
        
        # Get selected positions
        positions = [pos for pos, cb in self._position_checkboxes.items() if cb.isChecked()]
        if not positions:
            QMessageBox.warning(self, "No Positions", "Please select at least one router position.")
            return
        
        count = self._api_count_spin.value()
        generation_type = "trainer" if self._trainer_data_radio.isChecked() else "direct"
        
        # Use first provider for now (can extend to alternate later)
        provider, api_key = providers_to_use[0]
        
        # Store all providers for potential alternating use
        self._providers_to_use = providers_to_use
        
        # Start worker
        self._api_worker = APIGeneratorWorker(
            provider=provider,
            api_key=api_key,
            positions=positions,
            examples_per_position=count,
            generation_type=generation_type,
        )
        self._api_worker.progress.connect(self._on_progress)
        self._api_worker.finished.connect(self._on_api_finished)
        self._api_worker.error.connect(self._on_error)
        self._api_worker.start()
        
        # Update UI
        self._api_generate_btn.setEnabled(False)
        self._progress.setVisible(True)
        provider_name = "Claude" if provider == "claude" else "OpenAI"
        self._status.setText(f"Calling {provider_name} API...")
    
    def _on_api_finished(self, data: str):
        """Handle API generation completion."""
        self._generated_text = data
        total_lines = len([l for l in data.split('\n') if l.strip()])
        
        # Preview first 50 lines
        preview_lines = data.split('\n')[:50]
        preview_text = '\n'.join(preview_lines)
        if total_lines > 50:
            preview_text += f"\n\n... and {total_lines - 50} more lines"
        
        self._preview.setText(preview_text)
        self._stats_label.setText(f"Generated {total_lines} lines")
        
        # Convert to data format for saving
        self._generated_data = []
        for line in data.split('\n'):
            if line.strip():
                self._generated_data.append({"raw": line})
        
        # Update UI
        self._api_generate_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._save_btn.setEnabled(len(self._generated_text) > 0)
        self._append_btn.setEnabled(len(self._generated_text) > 0)
        self._status.setText(f"Done! Generated {total_lines} lines.")

    def _on_generate(self):
        """Start generating training data."""
        # Get topics
        topics_text = self._topics_input.toPlainText().strip()
        if not topics_text:
            QMessageBox.warning(self, "No Topics", "Please enter at least one topic.")
            return
        
        topics = [t.strip() for t in topics_text.split('\n') if t.strip()]
        if not topics:
            QMessageBox.warning(self, "No Topics", "Please enter at least one topic.")
            return
        
        # Get model
        if self._custom_model.text().strip():
            model_id = self._custom_model.text().strip()
        else:
            model_id = self._model_combo.currentData()
        
        # Get settings
        data_type = self._type_combo.currentData()
        style = self._style_combo.currentData()
        count = self._count_spin.value()
        use_4bit = self._use_4bit.isChecked()
        
        # Start worker
        self._worker = GeneratorWorker(
            model_id=model_id,
            topics=topics,
            data_type=data_type,
            count_per_topic=count,
            style=style,
            use_4bit=use_4bit,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        
        # Update UI
        self._generate_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._status.setText("Starting...")
    
    def _on_stop(self):
        """Stop the generation."""
        if self._worker:
            self._worker.stop()
        self._status.setText("Stopping...")
    
    def _on_progress(self, message: str):
        """Handle progress updates."""
        self._status.setText(message)
    
    def _on_finished(self, data: List[dict]):
        """Handle generation completion."""
        self._generated_data = data
        
        # Update preview
        preview_text = ""
        for i, item in enumerate(data[:20]):  # Show first 20
            if "question" in item:
                preview_text += f"Q: {item['question']}\nA: {item['answer']}\n\n"
            elif "instruction" in item:
                preview_text += f"Instruction: {item['instruction']}\nResponse: {item['response']}\n\n"
            elif isinstance(item, list):
                for turn in item:
                    preview_text += f"{turn['role'].title()}: {turn['content']}\n"
                preview_text += "\n"
        
        if len(data) > 20:
            preview_text += f"... and {len(data) - 20} more items"
        
        self._preview.setText(preview_text)
        self._stats_label.setText(f"Generated {len(data)} items")
        
        # Update UI
        self._generate_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setVisible(False)
        self._save_btn.setEnabled(len(data) > 0)
        self._append_btn.setEnabled(len(data) > 0)
        self._status.setText(f"Done! Generated {len(data)} items.")
    
    def _on_error(self, error: str):
        """Handle generation error."""
        self._generate_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setVisible(False)
        self._status.setText(f"Error: {error}")
        
        QMessageBox.critical(self, "Generation Failed", error)
    
    def _on_save(self):
        """Save generated data to file."""
        if not self._generated_data:
            return
        
        format_type = self._format_combo.currentData()
        extensions = {"enigma": "txt", "jsonl": "jsonl", "json": "json"}
        ext = extensions[format_type]
        
        data_dir = Path(CONFIG.get("data_dir", "data"))
        default_path = str(data_dir / f"generated_training.{ext}")
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Training Data",
            default_path,
            f"{ext.upper()} Files (*.{ext});;All Files (*)",
        )
        
        if not file_path:
            return
        
        data_dir = Path(CONFIG.get("data_dir", "data"))
        self._save_data(file_path, format_type)
    
    def _on_append(self):
        """Append to existing training file."""
        if not self._generated_data:
            return
        
        format_type = self._format_combo.currentData()
        extensions = {"enigma": "txt", "jsonl": "jsonl", "json": "json"}
        ext = extensions[format_type]
        
        data_dir = Path(CONFIG.get("data_dir", "data"))
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File to Append To",
            str(data_dir),
            f"{ext.upper()} Files (*.{ext});;All Files (*)",
        )
        
        if not file_path:
            return
        
        self._save_data(file_path, format_type, append=True)
    
    def _save_data(self, file_path: str, format_type: str, append: bool = False):
        """Save or append data to file."""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            mode = 'a' if append else 'w'
            
            if format_type == "enigma":
                with open(path, mode, encoding='utf-8') as f:
                    if append:
                        f.write("\n")  # Separator
                    for item in self._generated_data:
                        # Handle raw API-generated data
                        if "raw" in item:
                            f.write(f"{item['raw']}\n")
                        elif "question" in item:
                            f.write(f"Question: {item['question']}\n")
                            f.write(f"Answer: {item['answer']}\n\n")
                        elif "instruction" in item:
                            f.write(f"Question: {item['instruction']}\n")
                            f.write(f"Answer: {item['response']}\n\n")
                        elif isinstance(item, list):
                            for turn in item:
                                role = "Question" if turn["role"] == "user" else "Answer"
                                f.write(f"{role}: {turn['content']}\n")
                            f.write("\n")
                            
            elif format_type == "jsonl":
                with open(path, mode, encoding='utf-8') as f:
                    for item in self._generated_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        
            elif format_type == "json":
                if append and path.exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                    existing.extend(self._generated_data)
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(existing, f, indent=2, ensure_ascii=False)
                else:
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(self._generated_data, f, indent=2, ensure_ascii=False)
            
            action = "Appended" if append else "Saved"
            QMessageBox.information(
                self,
                "Success",
                f"{action} {len(self._generated_data)} items to:\n{file_path}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Failed to save: {e}")


def create_training_data_tab(parent=None) -> TrainingDataTab:
    """Factory function to create the Training Data Generator tab."""
    return TrainingDataTab(parent)
