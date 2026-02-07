"""
Build Your AI Wizard Tab - Step-by-step AI creation workflow

This tab provides a unified wizard experience for creating and customizing
your AI from start to finish:
1. Basic Info - Name and personality
2. System Prompt - Define how your AI behaves
3. Training Data - Select what knowledge to give your AI
4. Training - Actually train the model
5. Testing - Test your newly trained AI

Usage:
    from enigma_engine.gui.tabs.build_ai_tab import create_build_ai_tab
    
    build_widget = create_build_ai_tab(parent_window)
    tabs.addTab(build_widget, "Build AI")
"""

import logging
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
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


class TrainingWorker(QThread):
    """Background worker for model training."""
    progress = pyqtSignal(int, str)  # percent, message
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, model, training_file, epochs, batch_size, learning_rate):
        super().__init__()
        self.model = model
        self.training_file = training_file
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
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


def create_build_ai_tab(parent):
    """Create the Build Your AI wizard tab."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setSpacing(12)
    layout.setContentsMargins(16, 16, 16, 16)
    
    # Title
    title = QLabel("Build Your AI")
    title.setStyleSheet("font-size: 16px; font-weight: bold; color: #cdd6f4;")
    layout.addWidget(title)
    
    subtitle = QLabel("Create and customize your AI in a few simple steps")
    subtitle.setStyleSheet("font-size: 12px; color: #6c7086; margin-bottom: 8px;")
    layout.addWidget(subtitle)
    
    # Step indicator bar
    step_bar = QWidget()
    step_bar_layout = QHBoxLayout(step_bar)
    step_bar_layout.setContentsMargins(0, 8, 0, 8)
    
    # Store step buttons for updating
    parent._build_step_buttons = []
    step_names = ["Basic Info", "System Prompt", "Training Data", "Train", "Test"]
    
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
        label.setStyleSheet("font-size: 10px; color: #6c7086;")
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
    
    # Step 1: Basic Info
    step1 = _create_step_basic_info(parent)
    parent._build_stack.addWidget(step1)
    
    # Step 2: System Prompt
    step2 = _create_step_system_prompt(parent)
    parent._build_stack.addWidget(step2)
    
    # Step 3: Training Data
    step3 = _create_step_training_data(parent)
    parent._build_stack.addWidget(step3)
    
    # Step 4: Training
    step4 = _create_step_training(parent)
    parent._build_stack.addWidget(step4)
    
    # Step 5: Testing
    step5 = _create_step_testing(parent)
    parent._build_stack.addWidget(step5)
    
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


def _create_step_basic_info(parent):
    """Create Step 1: Basic Info."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setSpacing(16)
    
    header = QLabel("Step 1: Basic Information")
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
    """Create Step 3: Training Data Selection."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setSpacing(16)
    
    header = QLabel("Step 3: Training Data")
    header.setStyleSheet("font-size: 14px; font-weight: bold; color: #89b4fa;")
    layout.addWidget(header)
    
    desc = QLabel("Select training data to give your AI knowledge. You can use existing datasets or create your own.")
    desc.setStyleSheet("color: #a6adc8; margin-bottom: 8px;")
    desc.setWordWrap(True)
    layout.addWidget(desc)
    
    # Data source options
    source_group = QGroupBox("Data Sources")
    source_layout = QVBoxLayout(source_group)
    
    # Quick options
    parent._build_use_base = QCheckBox("Include Base Knowledge (recommended)")
    parent._build_use_base.setChecked(True)
    parent._build_use_base.setToolTip("Essential Q&A pairs for basic conversation skills")
    source_layout.addWidget(parent._build_use_base)
    
    parent._build_use_domain = QCheckBox("Include Domain-Specific Data")
    parent._build_use_domain.setToolTip("Add specialized knowledge in a particular field")
    source_layout.addWidget(parent._build_use_domain)
    
    layout.addWidget(source_group)
    
    # Custom training files
    files_group = QGroupBox("Training Files")
    files_layout = QVBoxLayout(files_group)
    
    # File list
    parent._build_training_files = QListWidget()
    parent._build_training_files.setMinimumHeight(120)
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
    
    layout.addWidget(files_group, stretch=1)
    
    # Data summary
    parent._build_data_summary = QLabel("No training data selected")
    parent._build_data_summary.setStyleSheet("color: #6c7086; font-style: italic; padding: 8px;")
    layout.addWidget(parent._build_data_summary)
    
    # Refresh summary on changes
    parent._build_training_files.itemChanged.connect(lambda: _update_data_summary(parent))
    parent._build_use_base.toggled.connect(lambda: _update_data_summary(parent))
    parent._build_use_domain.toggled.connect(lambda: _update_data_summary(parent))
    
    _update_data_summary(parent)
    
    return widget


def _create_step_training(parent):
    """Create Step 4: Training."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setSpacing(16)
    
    header = QLabel("Step 4: Train Your AI")
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
    """Create Step 5: Testing."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setSpacing(16)
    
    header = QLabel("Step 5: Test Your AI")
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
    
    if step_index == 4:  # Last step
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
    
    # Validate current step
    if current == 0:  # Basic Info
        if not parent._build_ai_name.text().strip():
            QMessageBox.warning(parent, "Missing Information", "Please enter a name for your AI.")
            return
    elif current == 1:  # System Prompt
        # System prompt is optional but recommended
        if not parent._build_system_prompt.toPlainText().strip():
            reply = QMessageBox.question(
                parent, "No System Prompt",
                "You haven't set a system prompt. Your AI may not have clear personality instructions.\n\nContinue anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
    elif current == 2:  # Training Data
        # Check if any data is selected
        has_data = parent._build_use_base.isChecked() or parent._build_training_files.count() > 0
        if not has_data:
            QMessageBox.warning(parent, "No Training Data", "Please select at least one training data source.")
            return
    elif current == 3:  # Training
        # Training step - check if training completed
        if 3 not in parent._build_completed_steps:
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
    if current < 4:
        _go_to_step(parent, current + 1)
    else:
        # Finish - go to chat
        _go_to_chat(parent)


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
    
    # Create and start worker
    parent._build_training_worker = TrainingWorker(
        parent.model, combined_path, epochs, batch_size, learning_rate
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
        
        # Update step indicator
        parent._build_step_buttons[3].setStyleSheet(STYLE_STEP_BTN_COMPLETE)
        
        QMessageBox.information(parent, "Training Complete", message)
    else:
        parent._build_progress.setFormat("Stopped")
        parent._build_status.setText(message)
        parent._build_status.setStyleSheet("color: #f38ba8;")
        
        QMessageBox.warning(parent, "Training Stopped", message)


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
