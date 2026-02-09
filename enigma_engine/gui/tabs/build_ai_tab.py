"""
Build Your AI Wizard Tab - Step-by-step AI creation workflow

This tab provides a unified wizard experience for creating and customizing
your AI from start to finish:
0. Setup - Choose model type and creation mode (Manual vs AI Trainer)
1. Basic Info - Name and personality (or AI Trainer questions)
2. Tools & Capabilities - Select what the AI can do
3. System Prompt - Define how your AI behaves (with personality mode)
4. Training Data - Select/generate knowledge data
5. Training - Actually train the model
6. Testing - Test your newly trained AI

Supports two modes:
- **Manual Mode**: Step-by-step tabs for full control
- **AI Trainer Mode**: Automated questions + data generation

Usage:
    from enigma_engine.gui.tabs.build_ai_tab import create_build_ai_tab
    
    build_widget = create_build_ai_tab(parent_window)
    tabs.addTab(build_widget, "Build AI")
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...config import CONFIG

logger = logging.getLogger(__name__)

# =============================================================================
# TOOL CAPABILITIES DEFINITION
# =============================================================================
TOOL_CAPABILITIES = {
    "chat": {"name": "Chat / Conversation", "description": "General conversation ability"},
    "image": {"name": "Image Generation", "description": "Create images from text prompts"},
    "code": {"name": "Code Generation", "description": "Write and explain code"},
    "search": {"name": "Web Search", "description": "Search the internet for information"},
    "file": {"name": "File Access", "description": "Read and write files"},
    "avatar": {"name": "Avatar Control", "description": "Control animated avatars"},
    "audio": {"name": "Voice/Audio", "description": "Text-to-speech and audio generation"},
    "video": {"name": "Video Generation", "description": "Create videos and animations"},
    "3d": {"name": "3D Model Generation", "description": "Generate 3D models"},
    "math": {"name": "Math/Calculations", "description": "Solve math problems"},
}

# =============================================================================
# ROUTER POSITIONS - Specialized Models to Train
# =============================================================================
ROUTER_POSITIONS = {
    "router": {
        "name": "Router (Intent Classification)",
        "description": "Classifies user intent and routes to correct tool",
        "recommended_for": ["image", "video", "audio", "3d", "search", "file"],
    },
    "chat": {
        "name": "Chat (Conversation)",
        "description": "General conversation responses",
        "recommended_for": ["chat"],
    },
    "code": {
        "name": "Code (Programming)",
        "description": "Code generation and explanation",
        "recommended_for": ["code"],
    },
    "vision": {
        "name": "Vision (Image Analysis)",
        "description": "Describes and analyzes images",
        "recommended_for": ["image"],
    },
    "math": {
        "name": "Math (Calculations)",
        "description": "Solves math problems step-by-step",
        "recommended_for": ["math"],
    },
    "avatar": {
        "name": "Avatar (Bone Control)",
        "description": "Converts commands to bone movements",
        "recommended_for": ["avatar"],
    },
    "trainer": {
        "name": "Trainer (Data Generation)",
        "description": "Generates training data for other positions",
        "recommended_for": [],
    },
    "teacher": {
        "name": "Teacher (Meta-Learning)",
        "description": "Evaluates and improves training data quality",
        "recommended_for": [],
    },
}

# =============================================================================
# STYLE CONSTANTS
# =============================================================================
STYLE_STEP_BTN = """
    QPushButton {
        background-color: transparent;
        color: #6c7086;
        border: 2px solid #6c7086;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        padding: 8px;
        min-width: 40px;
        min-height: 40px;
        max-width: 40px;
        max-height: 40px;
    }
    QPushButton:hover {
        border-color: #89b4fa;
        color: #89b4fa;
    }
"""

STYLE_STEP_BTN_ACTIVE = """
    QPushButton {
        background-color: #89b4fa;
        color: #1e1e2e;
        border: 2px solid #89b4fa;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        padding: 8px;
        min-width: 40px;
        min-height: 40px;
        max-width: 40px;
        max-height: 40px;
    }
"""

STYLE_STEP_BTN_COMPLETE = """
    QPushButton {
        background-color: #a6e3a1;
        color: #1e1e2e;
        border: 2px solid #a6e3a1;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        padding: 8px;
        min-width: 40px;
        min-height: 40px;
        max-width: 40px;
        max-height: 40px;
    }
"""

STYLE_PRIMARY_BTN = """
    QPushButton {
        background-color: #a6e3a1;
        color: #1e1e2e;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 6px;
        border: none;
        font-size: 13px;
    }
    QPushButton:hover {
        background-color: #94e2d5;
    }
    QPushButton:pressed {
        background-color: #74c7ec;
    }
    QPushButton:disabled {
        background-color: #313244;
        color: #6c7086;
    }
"""

STYLE_SECONDARY_BTN = """
    QPushButton {
        background-color: #45475a;
        color: #cdd6f4;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 6px;
        border: none;
        font-size: 13px;
    }
    QPushButton:hover {
        background-color: #585b70;
    }
    QPushButton:pressed {
        background-color: #6c7086;
    }
"""

STYLE_RADIO_CARD = """
    QRadioButton {
        background-color: #313244;
        border: 2px solid #45475a;
        border-radius: 8px;
        padding: 12px;
        color: #cdd6f4;
        font-size: 12px;
    }
    QRadioButton:checked {
        border-color: #89b4fa;
        background-color: #1e1e2e;
    }
    QRadioButton:hover {
        border-color: #6c7086;
    }
    QRadioButton::indicator {
        width: 16px;
        height: 16px;
    }
"""


class TrainingWorker(QThread):
    """Background worker for model training."""
    progress = pyqtSignal(int, str)  # percent, message
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, model, training_file, epochs, batch_size, learning_rate, nsfw_enabled=False):
        super().__init__()
        self.model = model
        self.training_file = training_file
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.nsfw_enabled = nsfw_enabled
        self._stop_requested = False
    
    def request_stop(self):
        """Request graceful stop."""
        self._stop_requested = True
    
    def run(self):
        """Run the training process."""
        try:
            from ...core.training import Trainer, TrainingConfig
            
            config = TrainingConfig(
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
            )
            
            trainer = Trainer(self.model, config)
            
            # Load training data
            self.progress.emit(5, "Loading training data...")
            with open(self.training_file, 'r', encoding='utf-8') as f:
                data = f.read()
            
            # Filter content based on NSFW setting
            if not self.nsfw_enabled:
                self.progress.emit(8, "Filtering training data...")
                try:
                    from ...core.content_rating import get_content_filter
                    filter_instance = get_content_filter()
                    lines = data.split('\n')
                    filtered_lines, stats = filter_instance.filter_training_data(
                        lines, include_nsfw=False
                    )
                    data = '\n'.join(filtered_lines)
                    logger.info(f"Filtered training data: {stats}")
                except ImportError:
                    logger.debug("Content filter not available")
            
            if self._stop_requested:
                self.finished.emit(False, "Training cancelled")
                return
            
            # Training loop with progress updates
            total_steps = self.epochs
            for epoch in range(self.epochs):
                if self._stop_requested:
                    self.finished.emit(False, "Training cancelled")
                    return
                
                progress_pct = int(10 + (epoch / total_steps) * 85)
                self.progress.emit(progress_pct, f"Training epoch {epoch + 1}/{self.epochs}...")
                
                # Note: Actual training happens in trainer.train()
                # This is a simplified progress indication
            
            # Actually run training
            self.progress.emit(50, "Training model...")
            trainer.train(data)
            
            self.progress.emit(95, "Saving model...")
            # Model auto-saves during training
            
            self.progress.emit(100, "Training complete!")
            self.finished.emit(True, "Training completed successfully!")
            
        except Exception as e:
            logger.exception("Training failed")
            self.finished.emit(False, f"Training failed: {str(e)}")


class DataGenWorker(QThread):
    """Background worker for AI Trainer data generation."""
    progress = pyqtSignal(int, str)  # percent, message
    finished = pyqtSignal(bool, str, str)  # success, message, data_path
    
    def __init__(self, ai_config: Dict):
        super().__init__()
        self.ai_config = ai_config
        self._stop_requested = False
    
    def request_stop(self):
        """Request graceful stop."""
        self._stop_requested = True
    
    def run(self):
        """Generate training data based on AI configuration."""
        try:
            from ...core.trainer_ai import get_trainer_ai
            
            trainer = get_trainer_ai()
            
            self.progress.emit(10, "Analyzing requirements...")
            
            # Use explicitly selected positions if provided, else derive from tools
            positions_needed = set(self.ai_config.get("positions", []))
            
            if not positions_needed:
                # Fallback: Map tools to positions
                tool_to_position = {
                    "chat": "chat",
                    "code": "code",
                    "math": "math",
                    "avatar": "avatar",
                    "image": "router",
                    "video": "router",
                    "audio": "router",
                    "3d": "router",
                    "search": "router",
                    "file": "router",
                }
                
                for tool in self.ai_config.get("tools", ["chat"]):
                    pos = tool_to_position.get(tool, "router")
                    positions_needed.add(pos)
            
            if self._stop_requested:
                self.finished.emit(False, "Cancelled", "")
                return
            
            self.progress.emit(20, "Generating training data...")
            
            # Generate data for each position
            all_data = []
            pos_list = list(positions_needed)
            
            for i, position in enumerate(pos_list):
                if self._stop_requested:
                    self.finished.emit(False, "Cancelled", "")
                    return
                
                pct = 20 + int((i / len(pos_list)) * 60)
                self.progress.emit(pct, f"Generating {position} data...")
                
                data = trainer.generate_training_data(
                    position=position,
                    count=self.ai_config.get("data_count", 200)
                )
                all_data.append(data)
            
            self.progress.emit(85, "Saving generated data...")
            
            # Save combined data
            ai_name = self.ai_config.get("name", "custom_ai").lower().replace(" ", "_")
            data_dir = Path(CONFIG.get("data_dir", "data")) / "generated"
            data_dir.mkdir(parents=True, exist_ok=True)
            data_path = data_dir / f"{ai_name}_training.txt"
            
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(f"# Generated Training Data for: {self.ai_config.get('name', 'Custom AI')}\n")
                f.write(f"# Tools: {', '.join(self.ai_config.get('tools', []))}\n")
                f.write(f"# Positions: {', '.join(self.ai_config.get('positions', list(positions_needed)))}\n")
                f.write(f"# Personality Mode: {self.ai_config.get('personality_mode', 'system_prompt')}\n\n")
                f.write("\n\n".join(all_data))
            
            self.progress.emit(100, "Complete!")
            self.finished.emit(True, "Training data generated successfully!", str(data_path))
            
        except Exception as e:
            logger.exception("Data generation failed")
            self.finished.emit(False, f"Data generation failed: {str(e)}", "")


class APIDataGenWorker(QThread):
    """Background worker for API-powered training data generation."""
    progress = pyqtSignal(int, str)  # percent, message
    finished = pyqtSignal(bool, str, str)  # success, message, data_path
    
    def __init__(self, provider: str, api_key: str, tasks: List[str], 
                 examples_per_task: int, store_securely: bool, ai_name: str):
        super().__init__()
        self.provider = provider
        self.api_key = api_key
        self.tasks = tasks
        self.examples_per_task = examples_per_task
        self.store_securely = store_securely
        self.ai_name = ai_name
        self._stop_requested = False
    
    def request_stop(self):
        """Request graceful stop."""
        self._stop_requested = True
    
    def run(self):
        """Generate training data using external API."""
        try:
            from ...core.trainer_ai import get_trainer_ai, get_api_training_provider
            
            self.progress.emit(5, "Configuring API...")
            
            trainer = get_trainer_ai()
            
            # Configure the API
            try:
                trainer.configure_api(
                    provider=self.provider,
                    api_key=self.api_key,
                    store_securely=self.store_securely
                )
            except Exception as e:
                self.finished.emit(False, f"API configuration failed: {str(e)}", "")
                return
            
            if self._stop_requested:
                self.finished.emit(False, "Cancelled", "")
                return
            
            self.progress.emit(10, "Generating training data from API...")
            
            # Use the API training provider directly for progress
            api = get_api_training_provider()
            
            # Generate data with progress callback
            total_tasks = len(self.tasks)
            all_data = []
            
            for i, task in enumerate(self.tasks):
                if self._stop_requested:
                    self.finished.emit(False, "Cancelled", "")
                    return
                
                pct = 10 + int((i / total_tasks) * 80)
                self.progress.emit(pct, f"Generating {task} data ({i+1}/{total_tasks})...")
                
                try:
                    data = api.generate_training_data(
                        tasks=[task],
                        examples_per_task=self.examples_per_task
                    )
                    all_data.extend(data.get(task, []))
                except Exception as e:
                    logger.warning(f"Failed to generate {task} data: {e}")
            
            if not all_data:
                self.finished.emit(False, "No training data generated. Check API key and connection.", "")
                return
            
            self.progress.emit(92, "Saving generated data...")
            
            # Save the data
            ai_name_safe = self.ai_name.lower().replace(" ", "_")
            data_dir = Path(CONFIG.get("data_dir", "data")) / "api_generated"
            data_dir.mkdir(parents=True, exist_ok=True)
            data_path = data_dir / f"{ai_name_safe}_api_training.txt"
            
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(f"# API-Generated Training Data for: {self.ai_name}\n")
                f.write(f"# Provider: {self.provider}\n")
                f.write(f"# Tasks: {', '.join(self.tasks)}\n")
                f.write(f"# Examples per task: {self.examples_per_task}\n\n")
                f.write("\n\n".join(all_data))
            
            self.progress.emit(100, "Complete!")
            total_examples = len(all_data)
            self.finished.emit(
                True, 
                f"Generated {total_examples} training examples from API!", 
                str(data_path)
            )
            
        except Exception as e:
            logger.exception("API data generation failed")
            self.finished.emit(False, f"API data generation failed: {str(e)}", "")


def create_build_ai_tab(parent):
    """Create the Build Your AI wizard tab with AI Trainer mode."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setSpacing(12)
    layout.setContentsMargins(16, 16, 16, 16)
    
    # Title
    title = QLabel("Build Your AI")
    title.setStyleSheet("font-size: 16px; font-weight: bold; color: #cdd6f4;")
    layout.addWidget(title)
    
    subtitle = QLabel("Create and customize your AI step by step")
    subtitle.setStyleSheet("font-size: 12px; color: #6c7086; margin-bottom: 8px;")
    layout.addWidget(subtitle)
    
    # Step indicator bar
    step_bar = QWidget()
    step_bar_layout = QHBoxLayout(step_bar)
    step_bar_layout.setContentsMargins(0, 8, 0, 8)
    
    # Store step buttons for updating - now 7 steps
    parent._build_step_buttons = []
    step_names = ["Setup", "Basic Info", "Tools", "Personality", "Training Data", "Train", "Test"]
    
    for i, name in enumerate(step_names):
        step_container = QVBoxLayout()
        step_container.setSpacing(4)
        
        # Step number button
        btn = QPushButton(str(i + 1))
        btn.setStyleSheet(STYLE_STEP_BTN if i > 0 else STYLE_STEP_BTN_ACTIVE)
        btn.clicked.connect(lambda checked, idx=i: _go_to_step(parent, idx))
        parent._build_step_buttons.append(btn)
        step_container.addWidget(btn, alignment=Qt.AlignHCenter)
        
        # Step name label
        label = QLabel(name)
        label.setStyleSheet("font-size: 9px; color: #6c7086;")
        label.setAlignment(Qt.AlignHCenter)
        step_container.addWidget(label)
        
        step_bar_layout.addLayout(step_container)
        
        # Connector line between steps
        if i < len(step_names) - 1:
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setStyleSheet("background-color: #6c7086; min-height: 2px; max-height: 2px;")
            step_bar_layout.addWidget(line, stretch=1)
    
    layout.addWidget(step_bar)
    
    # Stacked widget for step content
    parent._build_stack = QStackedWidget()
    
    # Initialize wizard state
    parent._build_wizard_state = {
        "mode": "manual",  # "manual" or "ai_trainer"
        "model_type": "new",  # "new" or "existing"
        "tools": ["chat"],
        "personality_mode": "system_prompt",  # "baked", "system_prompt", "hybrid"
        "generated_data_path": None,
    }
    
    # Step 0: Setup (Model + Mode Selection)
    step0 = _create_step_setup(parent)
    parent._build_stack.addWidget(step0)
    
    # Step 1: Basic Info
    step1 = _create_step_basic_info(parent)
    parent._build_stack.addWidget(step1)
    
    # Step 2: Tools & Capabilities
    step2 = _create_step_tools(parent)
    parent._build_stack.addWidget(step2)
    
    # Step 3: Personality Mode (System Prompt + Mode)
    step3 = _create_step_personality(parent)
    parent._build_stack.addWidget(step3)
    
    # Step 4: Training Data
    step4 = _create_step_training_data(parent)
    parent._build_stack.addWidget(step4)
    
    # Step 5: Training
    step5 = _create_step_training(parent)
    parent._build_stack.addWidget(step5)
    
    # Step 6: Testing
    step6 = _create_step_testing(parent)
    parent._build_stack.addWidget(step6)
    
    layout.addWidget(parent._build_stack, stretch=1)
    
    # Navigation buttons
    nav_layout = QHBoxLayout()
    nav_layout.addStretch()
    
    parent._build_prev_btn = QPushButton("Previous")
    parent._build_prev_btn.setStyleSheet(STYLE_SECONDARY_BTN)
    parent._build_prev_btn.clicked.connect(lambda: _prev_step(parent))
    parent._build_prev_btn.setEnabled(False)
    nav_layout.addWidget(parent._build_prev_btn)
    
    parent._build_next_btn = QPushButton("Next")
    parent._build_next_btn.setStyleSheet(STYLE_PRIMARY_BTN)
    parent._build_next_btn.clicked.connect(lambda: _next_step(parent))
    nav_layout.addWidget(parent._build_next_btn)
    
    layout.addLayout(nav_layout)
    
    # Track current step
    parent._build_current_step = 0
    parent._build_completed_steps = set()
    
    return widget


# =============================================================================
# STEP CREATION FUNCTIONS
# =============================================================================

def _create_step_setup(parent):
    """Create Step 0: Setup - Model and Mode Selection."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setSpacing(16)
    
    header = QLabel("Step 1: Setup")
    header.setStyleSheet("font-size: 14px; font-weight: bold; color: #89b4fa;")
    layout.addWidget(header)
    
    desc = QLabel("Choose whether to create a new AI or extend an existing one, and how you want to build it.")
    desc.setStyleSheet("color: #a6adc8; margin-bottom: 8px;")
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Model Type Selection
    model_group = QGroupBox("Base Model")
    model_layout = QVBoxLayout(model_group)
    
    parent._build_model_type_group = QButtonGroup(parent)
    
    # New Model option
    new_model_radio = QRadioButton("Create New AI")
    new_model_radio.setChecked(True)
    new_model_radio.setToolTip("Start fresh with no existing weights")
    parent._build_model_type_group.addButton(new_model_radio, 0)
    model_layout.addWidget(new_model_radio)
    
    new_model_desc = QLabel("    Start from scratch - train a completely new AI")
    new_model_desc.setStyleSheet("color: #6c7086; font-size: 11px; margin-left: 20px;")
    model_layout.addWidget(new_model_desc)
    
    # Existing Model option
    existing_radio = QRadioButton("Extend Existing AI")
    existing_radio.setToolTip("Add new capabilities to an existing trained model")
    parent._build_model_type_group.addButton(existing_radio, 1)
    model_layout.addWidget(existing_radio)
    
    existing_desc = QLabel("    Add new capabilities to an AI you've already trained")
    existing_desc.setStyleSheet("color: #6c7086; font-size: 11px; margin-left: 20px;")
    model_layout.addWidget(existing_desc)
    
    # Model selector (for existing)
    parent._build_existing_model_row = QWidget()
    existing_row_layout = QHBoxLayout(parent._build_existing_model_row)
    existing_row_layout.setContentsMargins(20, 4, 0, 0)
    existing_row_layout.addWidget(QLabel("Select model:"))
    parent._build_existing_model = QComboBox()
    parent._build_existing_model.addItem("(Select a model)")
    # Populate with existing models
    try:
        from ...core.model_registry import get_model_registry
        registry = get_model_registry()
        for model_name in registry.list_models():
            parent._build_existing_model.addItem(model_name)
    except Exception:
        pass
    existing_row_layout.addWidget(parent._build_existing_model, stretch=1)
    parent._build_existing_model_row.setVisible(False)
    model_layout.addWidget(parent._build_existing_model_row)
    
    # Show/hide model selector based on selection
    existing_radio.toggled.connect(lambda checked: parent._build_existing_model_row.setVisible(checked))
    
    layout.addWidget(model_group)
    
    # Creation Mode Selection
    mode_group = QGroupBox("Creation Mode")
    mode_layout = QVBoxLayout(mode_group)
    
    parent._build_mode_group = QButtonGroup(parent)
    
    # Manual Mode
    manual_radio = QRadioButton("Manual Mode")
    manual_radio.setChecked(True)
    manual_radio.setToolTip("Step-by-step control over every aspect")
    parent._build_mode_group.addButton(manual_radio, 0)
    mode_layout.addWidget(manual_radio)
    
    manual_desc = QLabel("    Full control - configure each step yourself")
    manual_desc.setStyleSheet("color: #6c7086; font-size: 11px;")
    mode_layout.addWidget(manual_desc)
    
    manual_detail = QLabel("    Best for: Advanced users, custom configurations")
    manual_detail.setStyleSheet("color: #45475a; font-size: 10px; font-style: italic;")
    mode_layout.addWidget(manual_detail)
    
    mode_layout.addSpacing(8)
    
    # AI Trainer Mode
    ai_trainer_radio = QRadioButton("AI Trainer Mode (Recommended)")
    ai_trainer_radio.setToolTip("Answer questions and let AI generate the training data")
    parent._build_mode_group.addButton(ai_trainer_radio, 1)
    mode_layout.addWidget(ai_trainer_radio)
    
    ai_trainer_desc = QLabel("    Guided - answer questions, AI generates training data automatically")
    ai_trainer_desc.setStyleSheet("color: #6c7086; font-size: 11px;")
    mode_layout.addWidget(ai_trainer_desc)
    
    ai_trainer_detail = QLabel("    Best for: Beginners, quick setup, common use cases")
    ai_trainer_detail.setStyleSheet("color: #45475a; font-size: 10px; font-style: italic;")
    mode_layout.addWidget(ai_trainer_detail)
    
    layout.addWidget(mode_group)
    
    # Quick preset selector (for AI Trainer mode)
    parent._build_preset_group = QGroupBox("Quick Start Preset (AI Trainer)")
    preset_layout = QVBoxLayout(parent._build_preset_group)
    
    parent._build_preset = QComboBox()
    parent._build_preset.addItems([
        "Custom (answer questions)",
        "Character Only - Conversational AI with personality",
        "Coder AI - Programming assistant",
        "Creative AI - Image/video/audio generation focus",
        "Full Assistant - All capabilities"
    ])
    parent._build_preset.setToolTip("Choose a preset to skip some questions")
    preset_layout.addWidget(parent._build_preset)
    
    preset_hint = QLabel("Presets auto-fill common configurations. You can still customize.")
    preset_hint.setStyleSheet("color: #6c7086; font-size: 11px;")
    preset_layout.addWidget(preset_hint)
    
    parent._build_preset_group.setVisible(False)
    layout.addWidget(parent._build_preset_group)
    
    # Show/hide preset based on mode
    ai_trainer_radio.toggled.connect(lambda checked: parent._build_preset_group.setVisible(checked))
    
    layout.addStretch()
    return widget


def _create_step_tools(parent):
    """Create Step 2: Tools & Capabilities Selection."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setSpacing(16)
    
    header = QLabel("Step 3: Tools & Capabilities")
    header.setStyleSheet("font-size: 14px; font-weight: bold; color: #89b4fa;")
    layout.addWidget(header)
    
    desc = QLabel("Select what capabilities your AI should have. Each tool requires specific training.")
    desc.setStyleSheet("color: #a6adc8; margin-bottom: 8px;")
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Scrollable area for both tools and positions
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    
    content_widget = QWidget()
    content_layout = QVBoxLayout(content_widget)
    content_layout.setSpacing(12)
    
    # --- TOOLS SECTION ---
    tools_group = QGroupBox("Runtime Tools (What the AI can do)")
    tools_group_layout = QVBoxLayout(tools_group)
    tools_group_layout.setSpacing(6)
    
    parent._build_tool_checkboxes = {}
    
    for tool_id, tool_info in TOOL_CAPABILITIES.items():
        row = QHBoxLayout()
        
        checkbox = QCheckBox(tool_info["name"])
        checkbox.setChecked(tool_id == "chat")  # Chat checked by default
        checkbox.setToolTip(tool_info["description"])
        parent._build_tool_checkboxes[tool_id] = checkbox
        row.addWidget(checkbox)
        
        desc_label = QLabel(f"- {tool_info['description']}")
        desc_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        row.addWidget(desc_label, stretch=1)
        
        tools_group_layout.addLayout(row)
    
    content_layout.addWidget(tools_group)
    
    # Tools quick select buttons
    tools_quick_row = QHBoxLayout()
    
    btn_minimal = QPushButton("Minimal")
    btn_minimal.setToolTip("Chat only")
    btn_minimal.clicked.connect(lambda: _select_tools_preset(parent, ["chat"]))
    tools_quick_row.addWidget(btn_minimal)
    
    btn_standard = QPushButton("Standard")
    btn_standard.setToolTip("Chat, Code, Search, File")
    btn_standard.clicked.connect(lambda: _select_tools_preset(parent, ["chat", "code", "search", "file"]))
    tools_quick_row.addWidget(btn_standard)
    
    btn_creative = QPushButton("Creative")
    btn_creative.setToolTip("Chat, Image, Video, Audio, 3D")
    btn_creative.clicked.connect(lambda: _select_tools_preset(parent, ["chat", "image", "video", "audio", "3d"]))
    tools_quick_row.addWidget(btn_creative)
    
    btn_all_tools = QPushButton("All Tools")
    btn_all_tools.clicked.connect(lambda: _select_tools_preset(parent, list(TOOL_CAPABILITIES.keys())))
    tools_quick_row.addWidget(btn_all_tools)
    
    tools_quick_row.addStretch()
    content_layout.addLayout(tools_quick_row)
    
    # --- ROUTER POSITIONS SECTION ---
    positions_group = QGroupBox("Training Positions (Specialized models to train)")
    positions_group_layout = QVBoxLayout(positions_group)
    positions_group_layout.setSpacing(6)
    
    positions_desc = QLabel("Select which specialized models to train. Auto-selects based on tools, but you can customize.")
    positions_desc.setStyleSheet("color: #6c7086; font-size: 11px; margin-bottom: 4px;")
    positions_desc.setWordWrap(True)
    positions_group_layout.addWidget(positions_desc)
    
    parent._build_position_checkboxes = {}
    
    for pos_id, pos_info in ROUTER_POSITIONS.items():
        row = QHBoxLayout()
        
        checkbox = QCheckBox(pos_info["name"])
        # Default: only chat position checked
        checkbox.setChecked(pos_id == "chat")
        checkbox.setToolTip(pos_info["description"])
        parent._build_position_checkboxes[pos_id] = checkbox
        row.addWidget(checkbox)
        
        desc_label = QLabel(f"- {pos_info['description']}")
        desc_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        row.addWidget(desc_label, stretch=1)
        
        positions_group_layout.addLayout(row)
    
    content_layout.addWidget(positions_group)
    
    # Positions quick select buttons
    pos_quick_row = QHBoxLayout()
    
    btn_auto = QPushButton("Auto (from tools)")
    btn_auto.setToolTip("Auto-select positions based on selected tools")
    btn_auto.clicked.connect(lambda: _auto_select_positions(parent))
    pos_quick_row.addWidget(btn_auto)
    
    btn_basic = QPushButton("Basic")
    btn_basic.setToolTip("Router + Chat")
    btn_basic.clicked.connect(lambda: _select_positions_preset(parent, ["router", "chat"]))
    pos_quick_row.addWidget(btn_basic)
    
    btn_coder = QPushButton("Coder")
    btn_coder.setToolTip("Router, Chat, Code")
    btn_coder.clicked.connect(lambda: _select_positions_preset(parent, ["router", "chat", "code"]))
    pos_quick_row.addWidget(btn_coder)
    
    btn_full = QPushButton("Full Router")
    btn_full.setToolTip("Router, Vision, Code, Math, Avatar")
    btn_full.clicked.connect(lambda: _select_positions_preset(parent, ["router", "vision", "code", "math", "avatar"]))
    pos_quick_row.addWidget(btn_full)
    
    btn_all_pos = QPushButton("All Positions")
    btn_all_pos.clicked.connect(lambda: _select_positions_preset(parent, list(ROUTER_POSITIONS.keys())))
    pos_quick_row.addWidget(btn_all_pos)
    
    pos_quick_row.addStretch()
    content_layout.addLayout(pos_quick_row)
    
    content_layout.addStretch()
    scroll.setWidget(content_widget)
    layout.addWidget(scroll, stretch=1)
    
    # Connect tool checkboxes to auto-update positions (optional)
    for checkbox in parent._build_tool_checkboxes.values():
        checkbox.stateChanged.connect(lambda state, p=parent: _auto_select_positions(p))
    
    return widget


def _select_positions_preset(parent, positions: list):
    """Select a preset of router positions."""
    if not hasattr(parent, '_build_position_checkboxes'):
        return
    for pos_id, checkbox in parent._build_position_checkboxes.items():
        checkbox.setChecked(pos_id in positions)


def _auto_select_positions(parent):
    """Auto-select positions based on selected tools."""
    if not hasattr(parent, '_build_tool_checkboxes') or not hasattr(parent, '_build_position_checkboxes'):
        return
    
    selected_tools = [
        tool_id for tool_id, checkbox in parent._build_tool_checkboxes.items()
        if checkbox.isChecked()
    ]
    
    # Build set of recommended positions
    positions_to_select = set()
    for tool_id in selected_tools:
        # Find positions that recommend this tool
        for pos_id, pos_info in ROUTER_POSITIONS.items():
            if tool_id in pos_info.get("recommended_for", []):
                positions_to_select.add(pos_id)
    
    # Always include router if any routing tools are selected
    routing_tools = {"image", "video", "audio", "3d", "search", "file"}
    if positions_to_select or any(t in selected_tools for t in routing_tools):
        positions_to_select.add("router")
    
    # If only chat selected and nothing else, just use chat
    if selected_tools == ["chat"]:
        positions_to_select = {"chat"}
    
    # Update checkboxes
    for pos_id, checkbox in parent._build_position_checkboxes.items():
        checkbox.setChecked(pos_id in positions_to_select)


def _create_step_personality(parent):
    """Create Step 3: Personality Mode - System Prompt + Personality Handling."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setSpacing(16)
    
    header = QLabel("Step 4: Personality & System Prompt")
    header.setStyleSheet("font-size: 14px; font-weight: bold; color: #89b4fa;")
    layout.addWidget(header)
    
    desc = QLabel("Define your AI's personality and how it should behave.")
    desc.setStyleSheet("color: #a6adc8; margin-bottom: 8px;")
    layout.addWidget(desc)
    
    # Personality Mode Selection
    mode_group = QGroupBox("Personality Application Mode")
    mode_layout = QVBoxLayout(mode_group)
    
    parent._build_personality_mode_group = QButtonGroup(parent)
    
    # System Prompt mode
    system_radio = QRadioButton("System Prompt (Easy to Change)")
    system_radio.setChecked(True)
    system_radio.setToolTip("Personality injected as instructions - easy to edit later")
    parent._build_personality_mode_group.addButton(system_radio, 0)
    mode_layout.addWidget(system_radio)
    
    system_desc = QLabel("    Personality defined in prompt box - edit anytime without retraining")
    system_desc.setStyleSheet("color: #6c7086; font-size: 11px;")
    mode_layout.addWidget(system_desc)
    
    # Baked mode
    baked_radio = QRadioButton("Baked Into Training (Permanent)")
    baked_radio.setToolTip("Personality woven into training data - becomes permanent")
    parent._build_personality_mode_group.addButton(baked_radio, 1)
    mode_layout.addWidget(baked_radio)
    
    baked_desc = QLabel("    Personality embedded in model weights - consistent but requires retraining to change")
    baked_desc.setStyleSheet("color: #6c7086; font-size: 11px;")
    mode_layout.addWidget(baked_desc)
    
    # Hybrid mode
    hybrid_radio = QRadioButton("Hybrid (Recommended)")
    hybrid_radio.setToolTip("Speech patterns baked in, personality traits in system prompt")
    parent._build_personality_mode_group.addButton(hybrid_radio, 2)
    mode_layout.addWidget(hybrid_radio)
    
    hybrid_desc = QLabel("    Speech patterns & vocabulary baked in, personality traits via system prompt")
    hybrid_desc.setStyleSheet("color: #6c7086; font-size: 11px;")
    mode_layout.addWidget(hybrid_desc)
    
    layout.addWidget(mode_group)
    
    # System Prompt Editor
    prompt_group = QGroupBox("System Prompt")
    prompt_layout = QVBoxLayout(prompt_group)
    
    # Template selector
    template_row = QHBoxLayout()
    template_row.addWidget(QLabel("Template:"))
    
    parent._build_template = QComboBox()
    parent._build_template.addItems([
        "Blank",
        "General Assistant",
        "Code Helper",
        "Creative Writer",
        "Technical Expert",
        "Friendly Tutor"
    ])
    parent._build_template.currentIndexChanged.connect(lambda: _apply_template(parent))
    template_row.addWidget(parent._build_template, stretch=1)
    prompt_layout.addLayout(template_row)
    
    # System prompt text
    parent._build_system_prompt = QTextEdit()
    parent._build_system_prompt.setPlaceholderText(
        "Enter instructions for your AI. For example:\n\n"
        "You are a helpful assistant named [Name]. You are friendly, helpful, and always try to provide accurate information. "
        "When you don't know something, you admit it rather than making things up."
    )
    parent._build_system_prompt.setMinimumHeight(150)
    parent._build_system_prompt.setStyleSheet("font-size: 12px;")
    prompt_layout.addWidget(parent._build_system_prompt)
    
    # Character count
    parent._build_prompt_char_count = QLabel("0 characters")
    parent._build_prompt_char_count.setStyleSheet("color: #6c7086; font-size: 11px;")
    parent._build_system_prompt.textChanged.connect(
        lambda: parent._build_prompt_char_count.setText(
            f"{len(parent._build_system_prompt.toPlainText())} characters"
        )
    )
    prompt_layout.addWidget(parent._build_prompt_char_count)
    
    layout.addWidget(prompt_group, stretch=1)
    
    return widget


def _select_tools_preset(parent, tools: List[str]):
    """Select a preset of tools."""
    for tool_id, checkbox in parent._build_tool_checkboxes.items():
        checkbox.setChecked(tool_id in tools)


def _create_step_basic_info(parent):
    """Create Step 2: Basic Info."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setSpacing(16)
    
    header = QLabel("Step 2: Basic Information")
    header.setStyleSheet("font-size: 14px; font-weight: bold; color: #89b4fa;")
    layout.addWidget(header)
    
    desc = QLabel("Give your AI a name and choose its base personality type.")
    desc.setStyleSheet("color: #a6adc8; margin-bottom: 8px;")
    layout.addWidget(desc)
    
    # AI Name
    name_group = QGroupBox("AI Name")
    name_layout = QVBoxLayout(name_group)
    
    parent._build_ai_name = QLineEdit()
    parent._build_ai_name.setPlaceholderText("Enter a name for your AI (e.g., Assistant, Helper, Luna)")
    parent._build_ai_name.setStyleSheet("padding: 8px; font-size: 13px;")
    name_layout.addWidget(parent._build_ai_name)
    
    name_hint = QLabel("This name will be used when the AI introduces itself.")
    name_hint.setStyleSheet("color: #6c7086; font-size: 11px;")
    name_layout.addWidget(name_hint)
    
    layout.addWidget(name_group)
    
    # Personality Type
    personality_group = QGroupBox("Base Personality")
    personality_layout = QVBoxLayout(personality_group)
    
    parent._build_personality = QComboBox()
    parent._build_personality.addItems([
        "Helpful Assistant - Friendly and informative",
        "Technical Expert - Precise and detailed",
        "Creative Writer - Imaginative and expressive",
        "Casual Friend - Relaxed and conversational",
        "Professional - Formal and business-like",
        "Teacher - Patient and educational",
        "Custom - Define your own personality"
    ])
    parent._build_personality.setStyleSheet("padding: 8px;")
    personality_layout.addWidget(parent._build_personality)
    
    personality_hint = QLabel("This sets the default tone for your AI's responses.")
    personality_hint.setStyleSheet("color: #6c7086; font-size: 11px;")
    personality_layout.addWidget(personality_hint)
    
    layout.addWidget(personality_group)
    
    # Model Size Selection
    model_group = QGroupBox("Model Size (optional)")
    model_layout = QVBoxLayout(model_group)
    
    parent._build_model_size = QComboBox()
    parent._build_model_size.addItems([
        "Use Current Model",
        "nano (~1M params) - Embedded/testing",
        "micro (~2M params) - Raspberry Pi",
        "tiny (~5M params) - Light devices",
        "small (~27M params) - Desktop default",
        "medium (~85M params) - Good balance",
        "large (~300M params) - Quality focus"
    ])
    parent._build_model_size.setStyleSheet("padding: 8px;")
    model_layout.addWidget(parent._build_model_size)
    
    model_hint = QLabel("Larger models are smarter but require more resources.")
    model_hint.setStyleSheet("color: #6c7086; font-size: 11px;")
    model_layout.addWidget(model_hint)
    
    layout.addWidget(model_group)
    
    layout.addStretch()
    return widget


def _create_step_system_prompt(parent):
    """Create Step 2: System Prompt."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setSpacing(16)
    
    header = QLabel("Step 2: System Prompt")
    header.setStyleSheet("font-size: 14px; font-weight: bold; color: #89b4fa;")
    layout.addWidget(header)
    
    desc = QLabel("Define how your AI should behave. This is the instruction given to the AI before every conversation.")
    desc.setStyleSheet("color: #a6adc8; margin-bottom: 8px;")
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Template selector
    template_row = QHBoxLayout()
    template_row.addWidget(QLabel("Start from template:"))
    
    parent._build_template = QComboBox()
    parent._build_template.addItems([
        "Blank - Start from scratch",
        "General Assistant",
        "Code Helper",
        "Creative Writer",
        "Technical Expert",
        "Friendly Tutor"
    ])
    parent._build_template.currentIndexChanged.connect(lambda: _apply_template(parent))
    template_row.addWidget(parent._build_template, stretch=1)
    
    layout.addLayout(template_row)
    
    # System prompt editor
    prompt_group = QGroupBox("System Prompt")
    prompt_layout = QVBoxLayout(prompt_group)
    
    parent._build_system_prompt = QTextEdit()
    parent._build_system_prompt.setPlaceholderText(
        "Enter instructions for your AI. For example:\n\n"
        "You are a helpful assistant named [Name]. You are friendly, helpful, and always try to provide accurate information. "
        "When you don't know something, you admit it rather than making things up."
    )
    parent._build_system_prompt.setMinimumHeight(200)
    parent._build_system_prompt.setStyleSheet("font-size: 13px;")
    prompt_layout.addWidget(parent._build_system_prompt)
    
    # Character count
    parent._build_prompt_char_count = QLabel("0 characters")
    parent._build_prompt_char_count.setStyleSheet("color: #6c7086; font-size: 11px;")
    parent._build_system_prompt.textChanged.connect(
        lambda: parent._build_prompt_char_count.setText(
            f"{len(parent._build_system_prompt.toPlainText())} characters"
        )
    )
    prompt_layout.addWidget(parent._build_prompt_char_count)
    
    layout.addWidget(prompt_group, stretch=1)
    
    # Tips
    tips_label = QLabel(
        "Tips:\n"
        "- Be specific about the AI's personality and capabilities\n"
        "- Include any constraints (e.g., 'Never give harmful advice')\n"
        "- Use [Name] as a placeholder for the AI's name"
    )
    tips_label.setStyleSheet("color: #6c7086; font-size: 11px; background: rgba(69, 71, 90, 0.5); padding: 8px; border-radius: 4px;")
    tips_label.setWordWrap(True)
    layout.addWidget(tips_label)
    
    return widget


def _create_step_training_data(parent):
    """Create Step 5: Training Data Selection."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setSpacing(16)
    
    # Use scroll area for long content
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
    
    scroll_content = QWidget()
    scroll_layout = QVBoxLayout(scroll_content)
    scroll_layout.setSpacing(12)
    
    header = QLabel("Step 5: Training Data")
    header.setStyleSheet("font-size: 14px; font-weight: bold; color: #89b4fa;")
    scroll_layout.addWidget(header)
    
    desc = QLabel("Select training data to give your AI knowledge. You can use existing datasets, add your own, or generate via API.")
    desc.setStyleSheet("color: #a6adc8; margin-bottom: 8px;")
    desc.setWordWrap(True)
    scroll_layout.addWidget(desc)
    
    # ==========================================================================
    # API Training Section (NEW)
    # ==========================================================================
    api_group = QGroupBox("API-Powered Training (GPT-4, Claude, etc.)")
    api_layout = QVBoxLayout(api_group)
    api_layout.setSpacing(8)
    
    api_desc = QLabel("Use external AI APIs to generate high-quality training data. Your API key is stored securely.")
    api_desc.setStyleSheet("color: #6c7086; font-size: 11px;")
    api_desc.setWordWrap(True)
    api_layout.addWidget(api_desc)
    
    # --------------------------------------------------------------------------
    # Stored Keys Management Section
    # --------------------------------------------------------------------------
    stored_keys_label = QLabel("Stored API Keys:")
    stored_keys_label.setStyleSheet("font-weight: bold; margin-top: 4px;")
    api_layout.addWidget(stored_keys_label)
    
    # List widget for stored keys
    parent._api_stored_keys_list = QListWidget()
    parent._api_stored_keys_list.setMaximumHeight(80)
    parent._api_stored_keys_list.setStyleSheet("""
        QListWidget {
            background: #1e1e2e;
            border: 1px solid #45475a;
            border-radius: 4px;
        }
        QListWidget::item {
            padding: 4px;
        }
        QListWidget::item:selected {
            background: #313244;
        }
    """)
    parent._api_stored_keys_list.itemClicked.connect(lambda item: _on_stored_key_selected(parent, item))
    api_layout.addWidget(parent._api_stored_keys_list)
    
    # Buttons for stored keys
    stored_keys_btn_row = QHBoxLayout()
    parent._api_use_key_btn = QPushButton("Use Selected")
    parent._api_use_key_btn.setMaximumWidth(100)
    parent._api_use_key_btn.setToolTip("Load the selected key for use")
    parent._api_use_key_btn.setEnabled(False)  # Disabled until key selected
    parent._api_use_key_btn.clicked.connect(lambda: _use_selected_key(parent))
    stored_keys_btn_row.addWidget(parent._api_use_key_btn)
    
    parent._api_delete_key_btn = QPushButton("Delete")
    parent._api_delete_key_btn.setMaximumWidth(70)
    parent._api_delete_key_btn.setStyleSheet("QPushButton { color: #f38ba8; } QPushButton:hover { background: #45475a; }")
    parent._api_delete_key_btn.setToolTip("Delete the selected key")
    parent._api_delete_key_btn.setEnabled(False)  # Disabled until key selected
    parent._api_delete_key_btn.clicked.connect(lambda: _delete_selected_key(parent))
    stored_keys_btn_row.addWidget(parent._api_delete_key_btn)
    
    parent._api_refresh_keys_btn = QPushButton("Refresh")
    parent._api_refresh_keys_btn.setMaximumWidth(70)
    parent._api_refresh_keys_btn.setToolTip("Refresh the list of stored keys")
    parent._api_refresh_keys_btn.clicked.connect(lambda: _refresh_stored_keys(parent))
    stored_keys_btn_row.addWidget(parent._api_refresh_keys_btn)
    
    stored_keys_btn_row.addStretch()
    api_layout.addLayout(stored_keys_btn_row)
    
    # Separator
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet("background: #45475a;")
    api_layout.addWidget(sep)
    
    # --------------------------------------------------------------------------
    # Add New Key Section
    # --------------------------------------------------------------------------
    new_key_label = QLabel("Add New Key:")
    new_key_label.setStyleSheet("font-weight: bold; margin-top: 4px;")
    api_layout.addWidget(new_key_label)
    
    # Provider selection row
    provider_row = QHBoxLayout()
    provider_row.addWidget(QLabel("Provider:"))
    parent._api_provider_combo = QComboBox()
    parent._api_provider_combo.addItems(["openai", "anthropic", "custom"])
    parent._api_provider_combo.setToolTip("Select API provider")
    parent._api_provider_combo.setMinimumWidth(120)
    provider_row.addWidget(parent._api_provider_combo)
    provider_row.addStretch()
    api_layout.addLayout(provider_row)
    
    # Key name row (for labeling the key)
    name_row = QHBoxLayout()
    name_row.addWidget(QLabel("Key Name:"))
    parent._api_key_name_input = QLineEdit()
    parent._api_key_name_input.setPlaceholderText("e.g., 'Main Key', 'Test Key', 'Work Account'")
    parent._api_key_name_input.setMinimumWidth(200)
    parent._api_key_name_input.setToolTip("A friendly name to identify this key")
    name_row.addWidget(parent._api_key_name_input)
    name_row.addStretch()
    api_layout.addLayout(name_row)
    
    # API Key row
    key_row = QHBoxLayout()
    key_row.addWidget(QLabel("API Key:"))
    parent._api_key_input = QLineEdit()
    parent._api_key_input.setEchoMode(QLineEdit.Password)
    parent._api_key_input.setPlaceholderText("sk-...")
    parent._api_key_input.setMinimumWidth(250)
    key_row.addWidget(parent._api_key_input)
    
    parent._api_show_key_btn = QPushButton("Show")
    parent._api_show_key_btn.setCheckable(True)
    parent._api_show_key_btn.setMaximumWidth(50)
    parent._api_show_key_btn.clicked.connect(lambda checked: _toggle_api_key_visibility(parent, checked))
    key_row.addWidget(parent._api_show_key_btn)
    key_row.addStretch()
    api_layout.addLayout(key_row)
    
    # Save key button
    save_key_row = QHBoxLayout()
    parent._api_save_key_btn = QPushButton("Save Key")
    parent._api_save_key_btn.setStyleSheet(STYLE_SECONDARY_BTN)
    parent._api_save_key_btn.setMaximumWidth(100)
    parent._api_save_key_btn.setToolTip("Save this key to encrypted storage")
    parent._api_save_key_btn.clicked.connect(lambda: _save_new_api_key(parent))
    save_key_row.addWidget(parent._api_save_key_btn)
    
    parent._api_key_status = QLabel("")
    parent._api_key_status.setStyleSheet("color: #a6e3a1; font-size: 11px;")
    save_key_row.addWidget(parent._api_key_status)
    save_key_row.addStretch()
    api_layout.addLayout(save_key_row)
    
    # Load stored keys on init
    _refresh_stored_keys(parent)
    
    # Separator
    sep2 = QFrame()
    sep2.setFrameShape(QFrame.HLine)
    sep2.setStyleSheet("background: #45475a;")
    api_layout.addWidget(sep2)
    
    # Task selection
    task_label = QLabel("Tasks to generate data for:")
    task_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
    api_layout.addWidget(task_label)
    
    # Task checkboxes in a grid-like layout
    parent._api_task_checks = {}
    task_grid = QHBoxLayout()
    task_col1 = QVBoxLayout()
    task_col2 = QVBoxLayout()
    task_col3 = QVBoxLayout()
    
    tasks = [
        ("chat", "Chat"), ("code", "Code"), ("vision", "Vision"), ("avatar", "Avatar"),
        ("image_gen", "Image Gen"), ("audio_gen", "Audio Gen"), ("video_gen", "Video Gen"), ("3d_gen", "3D Gen"),
        ("game", "Game"), ("robot", "Robot"), ("math", "Math"), ("router", "Router")
    ]
    
    for i, (task_key, task_name) in enumerate(tasks):
        cb = QCheckBox(task_name)
        cb.setChecked(task_key in ["chat", "code"])  # Default selections
        parent._api_task_checks[task_key] = cb
        if i < 4:
            task_col1.addWidget(cb)
        elif i < 8:
            task_col2.addWidget(cb)
        else:
            task_col3.addWidget(cb)
    
    task_grid.addLayout(task_col1)
    task_grid.addLayout(task_col2)
    task_grid.addLayout(task_col3)
    task_grid.addStretch()
    api_layout.addLayout(task_grid)
    
    # Quick select buttons
    quick_row = QHBoxLayout()
    btn_select_all = QPushButton("All")
    btn_select_all.setMaximumWidth(60)
    btn_select_all.clicked.connect(lambda: _api_select_tasks(parent, "all"))
    quick_row.addWidget(btn_select_all)
    
    btn_select_basic = QPushButton("Basic")
    btn_select_basic.setMaximumWidth(60)
    btn_select_basic.setToolTip("Chat + Code")
    btn_select_basic.clicked.connect(lambda: _api_select_tasks(parent, "basic"))
    quick_row.addWidget(btn_select_basic)
    
    btn_select_creative = QPushButton("Creative")
    btn_select_creative.setMaximumWidth(70)
    btn_select_creative.setToolTip("Chat + Image + Audio + Video")
    btn_select_creative.clicked.connect(lambda: _api_select_tasks(parent, "creative"))
    quick_row.addWidget(btn_select_creative)
    
    btn_select_none = QPushButton("None")
    btn_select_none.setMaximumWidth(60)
    btn_select_none.clicked.connect(lambda: _api_select_tasks(parent, "none"))
    quick_row.addWidget(btn_select_none)
    quick_row.addStretch()
    api_layout.addLayout(quick_row)
    
    # Examples per task
    examples_row = QHBoxLayout()
    examples_row.addWidget(QLabel("Examples per task:"))
    parent._api_examples_spin = QSpinBox()
    parent._api_examples_spin.setRange(10, 500)
    parent._api_examples_spin.setValue(50)
    parent._api_examples_spin.setSingleStep(10)
    parent._api_examples_spin.setToolTip("More examples = better quality but higher API cost")
    examples_row.addWidget(parent._api_examples_spin)
    examples_row.addStretch()
    api_layout.addLayout(examples_row)
    
    # Generate button and progress
    gen_api_row = QHBoxLayout()
    parent._api_gen_btn = QPushButton("Generate from API")
    parent._api_gen_btn.setStyleSheet(STYLE_PRIMARY_BTN)
    parent._api_gen_btn.clicked.connect(lambda: _generate_api_training_data(parent))
    gen_api_row.addWidget(parent._api_gen_btn)
    
    parent._api_cancel_btn = QPushButton("Cancel")
    parent._api_cancel_btn.setStyleSheet(STYLE_SECONDARY_BTN)
    parent._api_cancel_btn.setVisible(False)
    parent._api_cancel_btn.clicked.connect(lambda: _cancel_api_generation(parent))
    gen_api_row.addWidget(parent._api_cancel_btn)
    gen_api_row.addStretch()
    api_layout.addLayout(gen_api_row)
    
    parent._api_gen_progress = QProgressBar()
    parent._api_gen_progress.setVisible(False)
    parent._api_gen_progress.setTextVisible(True)
    api_layout.addWidget(parent._api_gen_progress)
    
    parent._api_gen_status = QLabel("")
    parent._api_gen_status.setStyleSheet("color: #6c7086; font-style: italic;")
    api_layout.addWidget(parent._api_gen_status)
    
    scroll_layout.addWidget(api_group)
    
    # ==========================================================================
    # Local AI Trainer section
    # ==========================================================================
    gen_group = QGroupBox("Local AI Trainer - Auto Generate Data")
    gen_layout = QVBoxLayout(gen_group)
    
    gen_desc = QLabel("Generate training data locally based on your tool selections (no API needed).")
    gen_desc.setStyleSheet("color: #6c7086; font-size: 11px;")
    gen_desc.setWordWrap(True)
    gen_layout.addWidget(gen_desc)
    
    gen_row = QHBoxLayout()
    parent._build_data_gen_btn = QPushButton("Generate Locally")
    parent._build_data_gen_btn.setStyleSheet(STYLE_SECONDARY_BTN)
    parent._build_data_gen_btn.clicked.connect(lambda: _generate_training_data(parent))
    gen_row.addWidget(parent._build_data_gen_btn)
    gen_row.addStretch()
    gen_layout.addLayout(gen_row)
    
    parent._build_data_gen_progress = QProgressBar()
    parent._build_data_gen_progress.setVisible(False)
    parent._build_data_gen_progress.setTextVisible(True)
    gen_layout.addWidget(parent._build_data_gen_progress)
    
    scroll_layout.addWidget(gen_group)
    
    # Data source options
    source_group = QGroupBox("Existing Data Sources")
    source_layout = QVBoxLayout(source_group)
    
    # Quick options
    parent._build_use_base = QCheckBox("Include Base Knowledge (recommended)")
    parent._build_use_base.setChecked(True)
    parent._build_use_base.setToolTip("Essential Q&A pairs for basic conversation skills")
    source_layout.addWidget(parent._build_use_base)
    
    parent._build_use_domain = QCheckBox("Include Domain-Specific Data")
    parent._build_use_domain.setToolTip("Add specialized knowledge in a particular field")
    source_layout.addWidget(parent._build_use_domain)
    
    scroll_layout.addWidget(source_group)
    
    # Custom training files
    files_group = QGroupBox("Training Files")
    files_layout = QVBoxLayout(files_group)
    
    # File list
    parent._build_training_files = QListWidget()
    parent._build_training_files.setMinimumHeight(100)
    parent._build_training_files.setToolTip("Training data files to use")
    files_layout.addWidget(parent._build_training_files)
    
    # File action buttons
    file_btn_row = QHBoxLayout()
    
    btn_add = QPushButton("Add File")
    btn_add.clicked.connect(lambda: _add_training_file(parent))
    file_btn_row.addWidget(btn_add)
    
    btn_remove = QPushButton("Remove")
    btn_remove.clicked.connect(lambda: _remove_training_file(parent))
    file_btn_row.addWidget(btn_remove)
    
    btn_browse_folder = QPushButton("Add Folder")
    btn_browse_folder.clicked.connect(lambda: _add_training_folder(parent))
    file_btn_row.addWidget(btn_browse_folder)
    
    file_btn_row.addStretch()
    files_layout.addLayout(file_btn_row)
    
    scroll_layout.addWidget(files_group)
    
    # Data summary
    parent._build_data_summary = QLabel("No training data selected")
    parent._build_data_summary.setStyleSheet("color: #6c7086; font-style: italic; padding: 8px;")
    scroll_layout.addWidget(parent._build_data_summary)
    
    # Finish scroll area setup
    scroll_layout.addStretch()
    scroll.setWidget(scroll_content)
    layout.addWidget(scroll)
    
    # Refresh summary on changes
    parent._build_training_files.itemChanged.connect(lambda: _update_data_summary(parent))
    parent._build_use_base.toggled.connect(lambda: _update_data_summary(parent))
    parent._build_use_domain.toggled.connect(lambda: _update_data_summary(parent))
    
    _update_data_summary(parent)
    
    return widget


def _create_step_training(parent):
    """Create Step 6: Training."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setSpacing(16)
    
    header = QLabel("Step 6: Train Your AI")
    header.setStyleSheet("font-size: 14px; font-weight: bold; color: #89b4fa;")
    layout.addWidget(header)
    
    desc = QLabel("Configure training parameters and start the training process.")
    desc.setStyleSheet("color: #a6adc8; margin-bottom: 8px;")
    layout.addWidget(desc)
    
    # Training parameters
    params_group = QGroupBox("Training Parameters")
    params_layout = QVBoxLayout(params_group)
    
    # Epochs
    epochs_row = QHBoxLayout()
    epochs_row.addWidget(QLabel("Epochs (training passes):"))
    parent._build_epochs = QSpinBox()
    parent._build_epochs.setRange(1, 100)
    parent._build_epochs.setValue(10)
    parent._build_epochs.setToolTip("Number of times to go through the training data")
    epochs_row.addWidget(parent._build_epochs)
    epochs_row.addStretch()
    params_layout.addLayout(epochs_row)
    
    # Batch size
    batch_row = QHBoxLayout()
    batch_row.addWidget(QLabel("Batch size:"))
    parent._build_batch = QSpinBox()
    parent._build_batch.setRange(1, 64)
    parent._build_batch.setValue(4)
    parent._build_batch.setToolTip("Samples processed together (lower = less memory)")
    batch_row.addWidget(parent._build_batch)
    batch_row.addStretch()
    params_layout.addLayout(batch_row)
    
    # Learning rate
    lr_row = QHBoxLayout()
    lr_row.addWidget(QLabel("Learning rate:"))
    parent._build_lr = QComboBox()
    parent._build_lr.addItems([
        "0.0001 (safe, recommended)",
        "0.0003 (moderate)",
        "0.0005 (aggressive)",
        "0.00005 (very conservative)"
    ])
    parent._build_lr.setToolTip("How fast the model learns (higher = faster but riskier)")
    lr_row.addWidget(parent._build_lr)
    lr_row.addStretch()
    params_layout.addLayout(lr_row)
    
    # Content Rating (NSFW training)
    nsfw_row = QHBoxLayout()
    parent._build_nsfw_check = QCheckBox("Include NSFW capability")
    parent._build_nsfw_check.setToolTip(
        "Train the model with adult content capability.\n"
        "If enabled, the model can generate NSFW content when toggled on in Settings.\n"
        "If disabled, NSFW mode will not be available for this model."
    )
    parent._build_nsfw_check.setChecked(False)
    nsfw_row.addWidget(parent._build_nsfw_check)
    
    nsfw_warning = QLabel("(Affects what content the model can produce)")
    nsfw_warning.setStyleSheet("color: #f59e0b; font-style: italic;")
    nsfw_row.addWidget(nsfw_warning)
    nsfw_row.addStretch()
    params_layout.addLayout(nsfw_row)
    
    layout.addWidget(params_group)
    
    # Training progress
    progress_group = QGroupBox("Training Progress")
    progress_layout = QVBoxLayout(progress_group)
    
    parent._build_progress = QProgressBar()
    parent._build_progress.setValue(0)
    parent._build_progress.setTextVisible(True)
    parent._build_progress.setFormat("%p% - Ready to train")
    progress_layout.addWidget(parent._build_progress)
    
    parent._build_status = QLabel("Ready to start training")
    parent._build_status.setStyleSheet("color: #6c7086;")
    progress_layout.addWidget(parent._build_status)
    
    layout.addWidget(progress_group)
    
    # Training controls
    control_row = QHBoxLayout()
    
    parent._build_train_btn = QPushButton("Start Training")
    parent._build_train_btn.setStyleSheet(STYLE_PRIMARY_BTN)
    parent._build_train_btn.clicked.connect(lambda: _start_training(parent))
    control_row.addWidget(parent._build_train_btn)
    
    parent._build_stop_btn = QPushButton("Stop")
    parent._build_stop_btn.setStyleSheet(STYLE_SECONDARY_BTN)
    parent._build_stop_btn.setEnabled(False)
    parent._build_stop_btn.clicked.connect(lambda: _stop_training(parent))
    control_row.addWidget(parent._build_stop_btn)
    
    control_row.addStretch()
    layout.addLayout(control_row)
    
    layout.addStretch()
    return widget


def _create_step_testing(parent):
    """Create Step 7: Testing."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setSpacing(16)
    
    header = QLabel("Step 7: Test Your AI")
    header.setStyleSheet("font-size: 14px; font-weight: bold; color: #89b4fa;")
    layout.addWidget(header)
    
    desc = QLabel("Test your newly trained AI! Try some conversations to see how it responds.")
    desc.setStyleSheet("color: #a6adc8; margin-bottom: 8px;")
    layout.addWidget(desc)
    
    # Test conversation area
    test_group = QGroupBox("Test Conversation")
    test_layout = QVBoxLayout(test_group)
    
    # Chat display
    parent._build_test_chat = QTextEdit()
    parent._build_test_chat.setReadOnly(True)
    parent._build_test_chat.setMinimumHeight(200)
    parent._build_test_chat.setStyleSheet("""
        QTextEdit {
            background-color: #1e1e2e;
            border: 1px solid #45475a;
            border-radius: 6px;
            padding: 8px;
        }
    """)
    test_layout.addWidget(parent._build_test_chat, stretch=1)
    
    # Input row
    input_row = QHBoxLayout()
    
    parent._build_test_input = QLineEdit()
    parent._build_test_input.setPlaceholderText("Type a message to test your AI...")
    parent._build_test_input.setStyleSheet("padding: 8px;")
    parent._build_test_input.returnPressed.connect(lambda: _send_test_message(parent))
    input_row.addWidget(parent._build_test_input, stretch=1)
    
    btn_send = QPushButton("Send")
    btn_send.setStyleSheet(STYLE_PRIMARY_BTN)
    btn_send.clicked.connect(lambda: _send_test_message(parent))
    input_row.addWidget(btn_send)
    
    test_layout.addLayout(input_row)
    
    layout.addWidget(test_group, stretch=1)
    
    # Suggested test prompts
    prompts_label = QLabel("Suggested test prompts:")
    prompts_label.setStyleSheet("color: #6c7086; font-size: 11px; margin-top: 8px;")
    layout.addWidget(prompts_label)
    
    prompts_row = QHBoxLayout()
    test_prompts = ["Hello!", "What can you help me with?", "Tell me about yourself"]
    
    for prompt in test_prompts:
        btn = QPushButton(prompt)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                padding: 6px 12px;
                border-radius: 4px;
                border: 1px solid #45475a;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
        """)
        btn.clicked.connect(lambda checked, p=prompt: _use_test_prompt(parent, p))
        prompts_row.addWidget(btn)
    
    prompts_row.addStretch()
    layout.addLayout(prompts_row)
    
    # Completion actions
    complete_row = QHBoxLayout()
    complete_row.addStretch()
    
    btn_save_persona = QPushButton("Save as Persona")
    btn_save_persona.setStyleSheet(STYLE_PRIMARY_BTN)
    btn_save_persona.clicked.connect(lambda: _save_as_persona(parent))
    complete_row.addWidget(btn_save_persona)
    
    btn_go_chat = QPushButton("Go to Chat")
    btn_go_chat.setStyleSheet(STYLE_SECONDARY_BTN)
    btn_go_chat.clicked.connect(lambda: _go_to_chat(parent))
    complete_row.addWidget(btn_go_chat)
    
    layout.addLayout(complete_row)
    
    return widget


# =============================================================================
# NAVIGATION FUNCTIONS
# =============================================================================

def _go_to_step(parent, step_index):
    """Navigate to a specific step."""
    parent._build_current_step = step_index
    parent._build_stack.setCurrentIndex(step_index)
    
    # Update step button styles
    for i, btn in enumerate(parent._build_step_buttons):
        if i < step_index:
            btn.setStyleSheet(STYLE_STEP_BTN_COMPLETE if i in parent._build_completed_steps else STYLE_STEP_BTN)
        elif i == step_index:
            btn.setStyleSheet(STYLE_STEP_BTN_ACTIVE)
        else:
            btn.setStyleSheet(STYLE_STEP_BTN)
    
    # Update navigation buttons
    parent._build_prev_btn.setEnabled(step_index > 0)
    
    if step_index == 6:  # Last step (Test)
        parent._build_next_btn.setText("Finish")
    else:
        parent._build_next_btn.setText("Next")


def _prev_step(parent):
    """Go to previous step."""
    if parent._build_current_step > 0:
        _go_to_step(parent, parent._build_current_step - 1)


def _next_step(parent):
    """Go to next step, validating current step first."""
    current = parent._build_current_step
    
    # Save wizard state from current step
    _save_step_state(parent, current)
    
    # Validate current step
    if current == 0:  # Setup
        # Store mode selection
        mode_id = parent._build_mode_group.checkedId()
        parent._build_wizard_state["mode"] = "ai_trainer" if mode_id == 1 else "manual"
        model_type_id = parent._build_model_type_group.checkedId()
        parent._build_wizard_state["model_type"] = "existing" if model_type_id == 1 else "new"
        
    elif current == 1:  # Basic Info
        if not parent._build_ai_name.text().strip():
            QMessageBox.warning(parent, "Missing Information", "Please enter a name for your AI.")
            return
            
    elif current == 2:  # Tools
        # Collect selected tools
        selected_tools = [
            tool_id for tool_id, cb in parent._build_tool_checkboxes.items()
            if cb.isChecked()
        ]
        if not selected_tools:
            QMessageBox.warning(parent, "No Tools Selected", "Please select at least one capability for your AI.")
            return
        parent._build_wizard_state["tools"] = selected_tools
        
        # Collect selected positions
        selected_positions = [
            pos_id for pos_id, cb in parent._build_position_checkboxes.items()
            if cb.isChecked()
        ]
        if not selected_positions:
            QMessageBox.warning(parent, "No Positions Selected", "Please select at least one training position.")
            return
        parent._build_wizard_state["positions"] = selected_positions
        
    elif current == 3:  # Personality
        # Store personality mode
        mode_id = parent._build_personality_mode_group.checkedId()
        modes = {0: "system_prompt", 1: "baked", 2: "hybrid"}
        parent._build_wizard_state["personality_mode"] = modes.get(mode_id, "system_prompt")
        
        # System prompt is optional but recommended
        if not parent._build_system_prompt.toPlainText().strip():
            reply = QMessageBox.question(
                parent, "No System Prompt",
                "You haven't set a system prompt. Your AI may not have clear personality instructions.\n\nContinue anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
                
    elif current == 4:  # Training Data
        # Check if any data is selected
        has_data = parent._build_use_base.isChecked() or parent._build_training_files.count() > 0
        has_generated = parent._build_wizard_state.get("generated_data_path") is not None
        
        if not has_data and not has_generated:
            # If in AI Trainer mode, offer to generate data
            if parent._build_wizard_state.get("mode") == "ai_trainer":
                reply = QMessageBox.question(
                    parent, "Generate Training Data?",
                    "No training data selected. Would you like to automatically generate training data based on your configuration?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    _generate_training_data(parent)
                    return  # Wait for generation to complete
            else:
                QMessageBox.warning(parent, "No Training Data", "Please select at least one training data source.")
                return
                
    elif current == 5:  # Training
        # Training step - check if training completed
        if 5 not in parent._build_completed_steps:
            reply = QMessageBox.question(
                parent, "Training Not Complete",
                "Training hasn't been completed. Continue to testing anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
    
    # Mark current step complete
    parent._build_completed_steps.add(current)
    
    # Go to next step or finish
    if current < 6:
        _go_to_step(parent, current + 1)
    else:
        # Finish - go to chat
        _go_to_chat(parent)


def _save_step_state(parent, step_index):
    """Save state from the current step to wizard state."""
    state = parent._build_wizard_state
    
    if step_index == 1:  # Basic Info
        state["name"] = parent._build_ai_name.text().strip()
        state["base_personality"] = parent._build_personality.currentText()
        state["model_size"] = parent._build_model_size.currentText()
    elif step_index == 3:  # Personality
        state["system_prompt"] = parent._build_system_prompt.toPlainText().strip()


def _generate_training_data(parent):
    """Generate training data using AI Trainer."""
    # Build config from wizard state
    config = {
        "name": parent._build_wizard_state.get("name", "custom_ai"),
        "tools": parent._build_wizard_state.get("tools", ["chat"]),
        "positions": parent._build_wizard_state.get("positions", []),
        "personality_mode": parent._build_wizard_state.get("personality_mode", "system_prompt"),
        "data_count": 200,
    }
    
    # Show progress
    parent._build_data_gen_progress.setVisible(True)
    parent._build_data_gen_progress.setValue(0)
    parent._build_data_gen_btn.setEnabled(False)
    
    # Create and start worker
    parent._build_data_gen_worker = DataGenWorker(config)
    parent._build_data_gen_worker.progress.connect(
        lambda pct, msg: _update_datagen_progress(parent, pct, msg)
    )
    parent._build_data_gen_worker.finished.connect(
        lambda success, msg, path: _datagen_finished(parent, success, msg, path)
    )
    parent._build_data_gen_worker.start()


def _update_datagen_progress(parent, percent, message):
    """Update data generation progress."""
    parent._build_data_gen_progress.setValue(percent)
    parent._build_data_gen_progress.setFormat(f"%p% - {message}")


def _datagen_finished(parent, success, message, data_path):
    """Handle data generation completion."""
    parent._build_data_gen_btn.setEnabled(True)
    
    if success:
        parent._build_wizard_state["generated_data_path"] = data_path
        parent._build_data_gen_progress.setValue(100)
        parent._build_data_gen_progress.setFormat("100% - Complete!")
        
        # Add generated file to training files list
        from PyQt5.QtCore import Qt
        item = QListWidgetItem(data_path)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        parent._build_training_files.addItem(item)
        
        _update_data_summary(parent)
        
        QMessageBox.information(parent, "Data Generated", f"Training data generated successfully!\n\nSaved to: {data_path}")
        
        # Auto-advance to next step
        parent._build_completed_steps.add(4)
        _go_to_step(parent, 5)
    else:
        parent._build_data_gen_progress.setFormat("Failed")
        QMessageBox.warning(parent, "Generation Failed", message)


# =============================================================================
# API TRAINING HELPER FUNCTIONS
# =============================================================================

def _refresh_stored_keys(parent):
    """Refresh the list of stored API keys."""
    parent._api_stored_keys_list.clear()
    
    try:
        from ...utils.api_key_encryption import SecureKeyStorage
        storage = SecureKeyStorage()
        keys = storage.list_keys()
        
        if not keys:
            item = QListWidgetItem("(No stored keys)")
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            item.setData(Qt.UserRole, None)
            parent._api_stored_keys_list.addItem(item)
        else:
            for service, meta in keys.items():
                # Format: "name (provider) - sk-...xxxx"
                display = f"{meta.description or service} ({service.split('_')[0]}) - {meta.masked_value}"
                item = QListWidgetItem(display)
                item.setData(Qt.UserRole, service)  # Store the actual key name
                item.setToolTip(f"Created: {meta.created_at[:10] if meta.created_at else 'Unknown'}\nLast used: {meta.last_used[:10] if meta.last_used else 'Never'}")
                parent._api_stored_keys_list.addItem(item)
                
    except Exception as e:
        logger.error(f"Failed to load stored keys: {e}")
        item = QListWidgetItem(f"(Error loading keys: {str(e)[:30]})")
        item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
        item.setData(Qt.UserRole, None)
        parent._api_stored_keys_list.addItem(item)


def _on_stored_key_selected(parent, item):
    """Handle stored key selection."""
    key_name = item.data(Qt.UserRole)
    if key_name:
        parent._api_use_key_btn.setEnabled(True)
        parent._api_delete_key_btn.setEnabled(True)
    else:
        parent._api_use_key_btn.setEnabled(False)
        parent._api_delete_key_btn.setEnabled(False)


def _use_selected_key(parent):
    """Load the selected key for use."""
    current_item = parent._api_stored_keys_list.currentItem()
    if not current_item:
        return
    
    key_name = current_item.data(Qt.UserRole)
    if not key_name:
        return
    
    try:
        from ...utils.api_key_encryption import SecureKeyStorage
        storage = SecureKeyStorage()
        key_value = storage.get_key(key_name)
        
        if key_value:
            parent._api_key_input.setText(key_value)
            # Set provider based on key name
            provider = key_name.split('_')[0] if '_' in key_name else key_name
            if provider in ['openai', 'anthropic', 'custom']:
                idx = parent._api_provider_combo.findText(provider)
                if idx >= 0:
                    parent._api_provider_combo.setCurrentIndex(idx)
            parent._api_key_status.setText(f"Loaded key: {key_name}")
            parent._api_key_status.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        else:
            QMessageBox.warning(parent, "Key Not Found", f"Could not load key: {key_name}")
            
    except Exception as e:
        logger.error(f"Failed to use key: {e}")
        QMessageBox.warning(parent, "Error", f"Failed to load key: {e}")


def _delete_selected_key(parent):
    """Delete the selected API key."""
    current_item = parent._api_stored_keys_list.currentItem()
    if not current_item:
        return
    
    key_name = current_item.data(Qt.UserRole)
    if not key_name:
        return
    
    # Confirm deletion
    reply = QMessageBox.question(
        parent, "Delete API Key",
        f"Are you sure you want to delete the key '{key_name}'?\n\nThis cannot be undone.",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No
    )
    
    if reply != QMessageBox.Yes:
        return
    
    try:
        from ...utils.api_key_encryption import SecureKeyStorage
        storage = SecureKeyStorage()
        
        if storage.delete_key(key_name):
            parent._api_key_status.setText(f"Deleted key: {key_name}")
            parent._api_key_status.setStyleSheet("color: #f38ba8; font-size: 11px;")
            _refresh_stored_keys(parent)
        else:
            QMessageBox.warning(parent, "Delete Failed", "Key not found or could not be deleted.")
            
    except Exception as e:
        logger.error(f"Failed to delete key: {e}")
        QMessageBox.warning(parent, "Error", f"Failed to delete key: {e}")


def _save_new_api_key(parent):
    """Save a new API key to secure storage."""
    api_key = parent._api_key_input.text().strip()
    key_name = parent._api_key_name_input.text().strip()
    provider = parent._api_provider_combo.currentText()
    
    if not api_key:
        QMessageBox.warning(parent, "No Key", "Please enter an API key to save.")
        return
    
    # Generate a unique key name if not provided
    if not key_name:
        key_name = provider
    
    # Create the storage key (provider_name format for multiple keys)
    storage_key = f"{provider}_{key_name.replace(' ', '_').lower()}"
    
    try:
        from ...utils.api_key_encryption import SecureKeyStorage
        storage = SecureKeyStorage()
        
        # Check if key already exists
        if storage.has_key(storage_key):
            reply = QMessageBox.question(
                parent, "Key Exists",
                f"A key named '{storage_key}' already exists.\n\nOverwrite it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        if storage.store_key(storage_key, api_key, key_name):
            parent._api_key_status.setText(f"Saved: {storage_key}")
            parent._api_key_status.setStyleSheet("color: #a6e3a1; font-size: 11px;")
            parent._api_key_name_input.clear()
            _refresh_stored_keys(parent)
        else:
            QMessageBox.warning(parent, "Save Failed", "Could not save the API key.")
            
    except Exception as e:
        logger.error(f"Failed to save key: {e}")
        QMessageBox.warning(parent, "Error", f"Failed to save key: {e}")


def _toggle_api_key_visibility(parent, show: bool):
    """Toggle API key visibility."""
    if show:
        parent._api_key_input.setEchoMode(QLineEdit.Normal)
        parent._api_show_key_btn.setText("Hide")
    else:
        parent._api_key_input.setEchoMode(QLineEdit.Password)
        parent._api_show_key_btn.setText("Show")


def _api_select_tasks(parent, preset: str):
    """Quick-select task presets."""
    presets = {
        "all": ["chat", "code", "vision", "avatar", "image_gen", "audio_gen", 
                "video_gen", "3d_gen", "game", "robot", "math", "router"],
        "basic": ["chat", "code"],
        "creative": ["chat", "image_gen", "audio_gen", "video_gen"],
        "none": [],
    }
    
    selected = presets.get(preset, [])
    for task_key, checkbox in parent._api_task_checks.items():
        checkbox.setChecked(task_key in selected)


def _generate_api_training_data(parent):
    """Generate training data using external API."""
    # Get selected tasks
    selected_tasks = [
        task_key for task_key, cb in parent._api_task_checks.items() 
        if cb.isChecked()
    ]
    
    if not selected_tasks:
        QMessageBox.warning(parent, "No Tasks Selected", 
            "Please select at least one task to generate data for.")
        return
    
    # Get API key from input field
    api_key = parent._api_key_input.text().strip()
    provider = parent._api_provider_combo.currentText()
    examples = parent._api_examples_spin.value()
    
    # If no key in input field, warn user
    if not api_key:
        QMessageBox.warning(parent, "No API Key",
            "Please enter an API key or select a stored key using 'Use Selected'.")
        return
    
    # Get AI name
    ai_name = parent._build_wizard_state.get("name", "custom_ai")
    
    # Show progress, hide generate button
    parent._api_gen_progress.setVisible(True)
    parent._api_gen_progress.setValue(0)
    parent._api_gen_btn.setEnabled(False)
    parent._api_cancel_btn.setVisible(True)
    parent._api_gen_status.setText("Starting API data generation...")
    
    # Create and start worker
    parent._api_gen_worker = APIDataGenWorker(
        provider=provider,
        api_key=api_key,
        tasks=selected_tasks,
        examples_per_task=examples,
        store_securely=False,  # Key management is separate now
        ai_name=ai_name
    )
    parent._api_gen_worker.progress.connect(
        lambda pct, msg: _update_api_gen_progress(parent, pct, msg)
    )
    parent._api_gen_worker.finished.connect(
        lambda success, msg, path: _api_gen_finished(parent, success, msg, path)
    )
    parent._api_gen_worker.start()


def _cancel_api_generation(parent):
    """Cancel API data generation."""
    if hasattr(parent, '_api_gen_worker') and parent._api_gen_worker.isRunning():
        parent._api_gen_worker.request_stop()
        parent._api_gen_status.setText("Cancelling...")


def _update_api_gen_progress(parent, percent, message):
    """Update API generation progress."""
    parent._api_gen_progress.setValue(percent)
    parent._api_gen_progress.setFormat(f"%p% - {message}")
    parent._api_gen_status.setText(message)


def _api_gen_finished(parent, success, message, data_path):
    """Handle API data generation completion."""
    parent._api_gen_btn.setEnabled(True)
    parent._api_cancel_btn.setVisible(False)
    
    if success:
        parent._build_wizard_state["generated_data_path"] = data_path
        parent._api_gen_progress.setValue(100)
        parent._api_gen_progress.setFormat("100% - Complete!")
        parent._api_gen_status.setText(message)
        parent._api_gen_status.setStyleSheet("color: #a6e3a1; font-style: italic;")
        
        # Add generated file to training files list
        item = QListWidgetItem(data_path)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        parent._build_training_files.addItem(item)
        
        _update_data_summary(parent)
        
        QMessageBox.information(parent, "API Data Generated", 
            f"{message}\n\nSaved to: {data_path}")
        
        # Mark step complete
        parent._build_completed_steps.add(4)
    else:
        parent._api_gen_progress.setFormat("Failed")
        parent._api_gen_status.setText(message)
        parent._api_gen_status.setStyleSheet("color: #f38ba8; font-style: italic;")
        QMessageBox.warning(parent, "API Generation Failed", message)


# =============================================================================
# STEP FUNCTIONS
# =============================================================================

def _apply_template(parent):
    """Apply a system prompt template."""
    templates = {
        0: "",  # Blank
        1: "You are a helpful AI assistant. You provide accurate, helpful information to users. You are friendly and professional. When you don't know something, you admit it honestly rather than guessing.",
        2: "You are an expert coding assistant. You help users write, debug, and understand code. You explain technical concepts clearly and provide well-commented code examples. You know multiple programming languages including Python, JavaScript, and C++.",
        3: "You are a creative writing assistant with a vivid imagination. You help users craft stories, poems, and other creative content. You use descriptive language and can adapt your writing style to match what the user needs.",
        4: "You are a technical expert. You provide detailed, accurate technical information. You cite sources when possible and explain complex topics in a clear, structured way. You use technical terminology appropriately.",
        5: "You are a patient and supportive tutor. You explain concepts step by step, adapting to the user's level of understanding. You encourage questions and provide examples to help learning. You celebrate progress.",
    }
    
    idx = parent._build_template.currentIndex()
    if idx in templates and templates[idx]:
        # Replace [Name] with actual name if set
        name = parent._build_ai_name.text().strip() or "Assistant"
        prompt = templates[idx]
        parent._build_system_prompt.setPlainText(prompt)


def _add_training_file(parent):
    """Add a training file to the list."""
    files, _ = QFileDialog.getOpenFileNames(
        parent,
        "Select Training Files",
        str(Path(CONFIG.get("data_dir", "data")) / "training"),
        "Text Files (*.txt *.json *.md);;All Files (*.*)"
    )
    
    for f in files:
        item = QListWidgetItem(f)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        parent._build_training_files.addItem(item)
    
    _update_data_summary(parent)


def _remove_training_file(parent):
    """Remove selected training file."""
    current = parent._build_training_files.currentRow()
    if current >= 0:
        parent._build_training_files.takeItem(current)
        _update_data_summary(parent)


def _add_training_folder(parent):
    """Add all training files from a folder."""
    folder = QFileDialog.getExistingDirectory(
        parent,
        "Select Training Data Folder",
        str(Path(CONFIG.get("data_dir", "data")) / "training")
    )
    
    if folder:
        folder_path = Path(folder)
        for f in folder_path.glob("*.txt"):
            item = QListWidgetItem(str(f))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            parent._build_training_files.addItem(item)
        
        _update_data_summary(parent)


def _update_data_summary(parent):
    """Update the training data summary."""
    total_files = 0
    total_lines = 0
    
    if parent._build_use_base.isChecked():
        base_file = Path(CONFIG.get("data_dir", "data")) / "training" / "base_knowledge.txt"
        if base_file.exists():
            total_files += 1
            try:
                total_lines += sum(1 for _ in open(base_file, encoding='utf-8'))
            except Exception:
                pass
    
    for i in range(parent._build_training_files.count()):
        item = parent._build_training_files.item(i)
        if item.checkState() == Qt.Checked:
            total_files += 1
            try:
                total_lines += sum(1 for _ in open(item.text(), encoding='utf-8'))
            except Exception:
                pass
    
    if total_files == 0:
        parent._build_data_summary.setText("No training data selected")
        parent._build_data_summary.setStyleSheet("color: #f38ba8; font-style: italic; padding: 8px;")
    else:
        parent._build_data_summary.setText(f"{total_files} file(s) selected, approximately {total_lines:,} lines")
        parent._build_data_summary.setStyleSheet("color: #a6e3a1; font-style: italic; padding: 8px;")


def _start_training(parent):
    """Start the training process."""
    # Collect training files
    training_files = []
    
    if parent._build_use_base.isChecked():
        base_file = Path(CONFIG.get("data_dir", "data")) / "training" / "base_knowledge.txt"
        if base_file.exists():
            training_files.append(base_file)
    
    for i in range(parent._build_training_files.count()):
        item = parent._build_training_files.item(i)
        if item.checkState() == Qt.Checked:
            training_files.append(Path(item.text()))
    
    if not training_files:
        QMessageBox.warning(parent, "No Training Data", "Please select at least one training file.")
        return
    
    # Get training parameters
    epochs = parent._build_epochs.value()
    batch_size = parent._build_batch.value()
    
    lr_text = parent._build_lr.currentText()
    learning_rate = float(lr_text.split()[0])
    
    # Get NSFW capability setting
    nsfw_enabled = parent._build_nsfw_check.isChecked() if hasattr(parent, '_build_nsfw_check') else False
    parent._build_nsfw_enabled = nsfw_enabled  # Store for metadata saving
    
    # Check if model is loaded
    if not hasattr(parent, 'model') or parent.model is None:
        QMessageBox.warning(parent, "No Model", "Please load a model first in the Chat tab.")
        return
    
    # Combine training files into a temp file
    combined_path = Path(CONFIG.get("data_dir", "data")) / "training" / "_build_wizard_combined.txt"
    try:
        with open(combined_path, 'w', encoding='utf-8') as out:
            for f in training_files:
                with open(f, 'r', encoding='utf-8') as inp:
                    out.write(inp.read())
                    out.write("\n\n")
    except Exception as e:
        QMessageBox.critical(parent, "Error", f"Failed to prepare training data:\n{e}")
        return
    
    # Update UI
    parent._build_train_btn.setEnabled(False)
    parent._build_stop_btn.setEnabled(True)
    parent._build_progress.setValue(0)
    parent._build_progress.setFormat("%p% - Starting...")
    parent._build_status.setText("Initializing training...")
    
    # Create and start worker with NSFW setting
    parent._build_training_worker = TrainingWorker(
        parent.model, combined_path, epochs, batch_size, learning_rate, nsfw_enabled
    )
    parent._build_training_worker.progress.connect(
        lambda pct, msg: _update_training_progress(parent, pct, msg)
    )
    parent._build_training_worker.finished.connect(
        lambda success, msg: _training_finished(parent, success, msg)
    )
    parent._build_training_worker.start()


def _stop_training(parent):
    """Stop the training process."""
    if hasattr(parent, '_build_training_worker') and parent._build_training_worker.isRunning():
        parent._build_training_worker.request_stop()
        parent._build_status.setText("Stopping training...")
        parent._build_stop_btn.setEnabled(False)


def _update_training_progress(parent, percent, message):
    """Update training progress display."""
    parent._build_progress.setValue(percent)
    parent._build_progress.setFormat(f"%p% - {message}")
    parent._build_status.setText(message)


def _training_finished(parent, success, message):
    """Handle training completion."""
    parent._build_train_btn.setEnabled(True)
    parent._build_stop_btn.setEnabled(False)
    
    if success:
        parent._build_progress.setValue(100)
        parent._build_progress.setFormat("100% - Complete!")
        parent._build_status.setText("Training completed successfully!")
        parent._build_status.setStyleSheet("color: #a6e3a1;")
        parent._build_completed_steps.add(3)
        
        # Save model metadata including NSFW capability
        _save_model_metadata(parent)
        
        # Update step indicator
        parent._build_step_buttons[3].setStyleSheet(STYLE_STEP_BTN_COMPLETE)
        
        QMessageBox.information(parent, "Training Complete", message)
    else:
        parent._build_progress.setFormat("Stopped")
        parent._build_status.setText(message)
        parent._build_status.setStyleSheet("color: #f38ba8;")
        
        QMessageBox.warning(parent, "Training Stopped", message)


def _save_model_metadata(parent):
    """Save model metadata after training."""
    import json
    from datetime import datetime
    
    try:
        models_dir = Path(CONFIG.get("models_dir", "models"))
        metadata_file = models_dir / "model_metadata.json"
        
        nsfw_enabled = getattr(parent, '_build_nsfw_enabled', False)
        
        metadata = {
            "supports_nsfw": nsfw_enabled,
            "content_rating": "nsfw" if nsfw_enabled else "sfw",
            "trained_date": datetime.now().isoformat(),
            "training_tasks": [],
            "trained_with_gui": True
        }
        
        # Save metadata
        models_dir.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved model metadata to {metadata_file}")
        
        # Update content filter with new capability
        try:
            from ...core.content_rating import get_content_filter
            content_filter = get_content_filter()
            content_filter.set_model_nsfw_capability(nsfw_enabled)
            content_filter.save_config()
        except ImportError:
            pass
            
    except Exception as e:
        logger.warning(f"Could not save model metadata: {e}")


def _send_test_message(parent):
    """Send a test message to the AI."""
    message = parent._build_test_input.text().strip()
    if not message:
        return
    
    parent._build_test_input.clear()
    
    # Add user message to display
    parent._build_test_chat.append(f"<b style='color: #89b4fa;'>You:</b> {message}")
    
    # Check if model is loaded
    if not hasattr(parent, 'model') or parent.model is None:
        parent._build_test_chat.append(
            "<b style='color: #f38ba8;'>Error:</b> No model loaded. Please load a model in the Chat tab."
        )
        return
    
    try:
        # Get system prompt
        system_prompt = parent._build_system_prompt.toPlainText().strip()
        ai_name = parent._build_ai_name.text().strip() or "Assistant"
        system_prompt = system_prompt.replace("[Name]", ai_name)
        
        # Generate response
        if hasattr(parent, 'engine') and parent.engine:
            # Build prompt with system
            full_prompt = f"{system_prompt}\n\nUser: {message}\nAssistant:"
            
            # Generate
            response = parent.engine.generate(
                full_prompt, max_new_tokens=150, temperature=0.7
            )
            
            # Clean up response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            parent._build_test_chat.append(f"<b style='color: #a6e3a1;'>{ai_name}:</b> {response}")
        else:
            parent._build_test_chat.append(
                "<b style='color: #f38ba8;'>Error:</b> Inference engine not available."
            )
    except Exception as e:
        logger.exception("Test message failed")
        parent._build_test_chat.append(f"<b style='color: #f38ba8;'>Error:</b> {str(e)}")


def _use_test_prompt(parent, prompt):
    """Use a suggested test prompt."""
    parent._build_test_input.setText(prompt)
    _send_test_message(parent)


def _save_as_persona(parent):
    """Save the current build as a persona."""
    try:
        from ...core.persona import AIPersona, get_persona_manager
        
        manager = get_persona_manager()
        
        name = parent._build_ai_name.text().strip()
        if not name:
            QMessageBox.warning(parent, "Missing Name", "Please enter a name for the persona.")
            return
        
        system_prompt = parent._build_system_prompt.toPlainText().strip()
        system_prompt = system_prompt.replace("[Name]", name)
        
        # Create persona
        persona = AIPersona(
            name=name,
            system_prompt=system_prompt,
            description=f"Created with Build Your AI wizard"
        )
        
        manager.save_persona(persona)
        
        QMessageBox.information(
            parent, "Persona Saved",
            f"Persona '{name}' has been saved! You can switch to it in the Persona tab."
        )
        
    except Exception as e:
        logger.exception("Failed to save persona")
        QMessageBox.critical(parent, "Error", f"Failed to save persona:\n{e}")


def _go_to_chat(parent):
    """Navigate to the Chat tab."""
    if hasattr(parent, 'sidebar') and hasattr(parent, '_nav_map'):
        if 'chat' in parent._nav_map:
            # Find and select the chat item in sidebar
            for i in range(parent.sidebar.count()):
                item = parent.sidebar.item(i)
                if item and item.data(Qt.UserRole) == 'chat':
                    parent.sidebar.setCurrentRow(i)
                    break
