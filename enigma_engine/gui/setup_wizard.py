"""
First Run Setup Wizard

Guides new users through initial Enigma AI Engine configuration.
Shows on first launch to help users get started quickly.

Usage:
    from enigma_engine.gui.setup_wizard import SetupWizard, should_show_wizard
    
    if should_show_wizard():
        wizard = SetupWizard(parent)
        if wizard.exec_() == QDialog.Accepted:
            config = wizard.get_config()
            apply_config(config)
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

logger = logging.getLogger(__name__)

# Config file path
WIZARD_CONFIG_PATH = Path.home() / ".enigma_engine" / "setup_complete.json"


def should_show_wizard() -> bool:
    """Check if the setup wizard should be shown."""
    return not WIZARD_CONFIG_PATH.exists()


def mark_wizard_complete(config: dict[str, Any]):
    """Mark setup as complete and save config."""
    WIZARD_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WIZARD_CONFIG_PATH, 'w') as f:
        json.dump({
            'completed': True,
            'config': config
        }, f, indent=2)


# Styles
WIZARD_STYLE = """
QWizard {
    background: #1e1e2e;
}
QWizardPage {
    background: #1e1e2e;
    color: #cdd6f4;
}
QLabel {
    color: #cdd6f4;
}
QLabel#title {
    color: #89b4fa;
    font-size: 18px;
    font-weight: bold;
}
QLabel#subtitle {
    color: #a6adc8;
    font-size: 12px;
}
QRadioButton, QCheckBox {
    color: #cdd6f4;
    spacing: 8px;
}
QRadioButton::indicator, QCheckBox::indicator {
    width: 16px;
    height: 16px;
}
QGroupBox {
    color: #89b4fa;
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QPushButton {
    background: #45475a;
    color: #cdd6f4;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
}
QPushButton:hover {
    background: #585b70;
}
QLineEdit, QComboBox {
    background: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 6px;
}
"""


class WelcomePage(QWizardPage):
    """Welcome page introducing Enigma AI Engine."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("")
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Welcome to Enigma AI Engine")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        font = title.font()
        font.setPointSize(24)
        title.setFont(font)
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Your Local AI Assistant")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        layout.addSpacing(20)
        
        # Description
        desc = QLabel(
            "This wizard will help you set up Enigma AI Engine for the first time.\n\n"
            "We'll configure:\n"
            "  - Model size (based on your hardware)\n"
            "  - Privacy settings\n"
            "  - Interface preferences\n\n"
            "You can change these settings later in the Settings tab."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)
        
        layout.addStretch()


class AINamePage(QWizardPage):
    """Page for naming the AI assistant."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Name Your AI")
        self.setSubTitle("Give your AI assistant a name")
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Explanation
        desc = QLabel(
            "Choose a name for your AI assistant. This name will be used "
            "in conversations and will help personalize your experience.\n\n"
            "You can change this later in the Settings tab."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        layout.addSpacing(10)
        
        # Name input
        name_group = QGroupBox("AI Name")
        name_layout = QVBoxLayout(name_group)
        
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("Enter AI name (e.g., Enigma, Nova, Atlas)")
        self._name_input.setText("Enigma")  # Default name
        self._name_input.setMaxLength(50)
        name_layout.addWidget(self._name_input)
        
        # Suggestions
        suggestions_label = QLabel("Suggestions: Enigma, Nova, Atlas, Echo, Sage, Aria")
        suggestions_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        name_layout.addWidget(suggestions_label)
        
        layout.addWidget(name_group)
        
        # Quick select buttons
        quick_group = QGroupBox("Quick Select")
        quick_layout = QHBoxLayout(quick_group)
        
        quick_names = ["Enigma", "Nova", "Atlas", "Echo", "Sage", "Aria"]
        for name in quick_names:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, n=name: self._name_input.setText(n))
            quick_layout.addWidget(btn)
        
        layout.addWidget(quick_group)
        
        layout.addStretch()
        
        # Register field
        self.registerField("ai_name", self._name_input)
    
    def validatePage(self):
        """Ensure a name is entered."""
        name = self._name_input.text().strip()
        if not name:
            self._name_input.setText("Enigma")  # Default if empty
        return True


class HardwareDetectionPage(QWizardPage):
    """Page that detects and displays hardware capabilities."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Hardware Detection")
        self.setSubTitle("Detecting your system capabilities...")
        
        layout = QVBoxLayout(self)
        
        # Hardware info display
        self._hw_group = QGroupBox("Detected Hardware")
        hw_layout = QVBoxLayout(self._hw_group)
        
        self._cpu_label = QLabel("CPU: Detecting...")
        self._ram_label = QLabel("RAM: Detecting...")
        self._gpu_label = QLabel("GPU: Detecting...")
        self._recommended_label = QLabel("Recommended Model: Detecting...")
        self._recommended_label.setStyleSheet("font-weight: bold; color: #a6e3a1;")
        
        hw_layout.addWidget(self._cpu_label)
        hw_layout.addWidget(self._ram_label)
        hw_layout.addWidget(self._gpu_label)
        hw_layout.addWidget(self._recommended_label)
        
        layout.addWidget(self._hw_group)
        
        # Model size selection
        self._model_group = QGroupBox("Model Size")
        model_layout = QVBoxLayout(self._model_group)
        
        self._model_buttons = QButtonGroup(self)
        
        sizes = [
            ("nano", "Nano", "Fastest, least capable (~1M params)"),
            ("tiny", "Tiny", "Fast, basic capabilities (~5M params)"),
            ("small", "Small", "Good balance (~27M params)"),
            ("medium", "Medium", "More capable (~85M params)"),
            ("large", "Large", "High quality (~300M params)"),
        ]
        
        for i, (key, name, desc) in enumerate(sizes):
            radio = QRadioButton(f"{name} - {desc}")
            radio.setProperty("model_size", key)
            self._model_buttons.addButton(radio, i)
            model_layout.addWidget(radio)
        
        layout.addWidget(self._model_group)
        
        # Register field
        self.registerField("model_size*", self._model_buttons.buttons()[2])  # Default small
    
    def initializePage(self):
        """Detect hardware when page is shown."""
        super().initializePage()
        self._detect_hardware()
    
    def _detect_hardware(self):
        """Detect and display hardware info."""
        import os
        import platform
        import torch

        # CPU - with fallback chain
        cpu_name = None
        cpu_count = os.cpu_count() or 1
        
        try:
            import psutil
            cpu_count = psutil.cpu_count() or cpu_count
        except ImportError:
            pass
        
        # Try to get CPU name
        try:
            cpu_name = platform.processor()
            if not cpu_name or cpu_name == "":
                # Windows fallback
                import subprocess
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"], 
                    capture_output=True, text=True, timeout=5
                )
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    cpu_name = lines[1].strip()
        except Exception:
            pass
        
        if cpu_name and cpu_name != "":
            self._cpu_label.setText(f"CPU: {cpu_name} ({cpu_count} cores)")
        else:
            self._cpu_label.setText(f"CPU: {cpu_count} cores")
        
        # RAM
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
            self._ram_label.setText(f"RAM: {ram_gb:.1f} GB")
        except ImportError:
            ram_gb = 4
            self._ram_label.setText("RAM: Unknown")
        
        # GPU
        gpu_vram = 0
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self._gpu_label.setText(f"GPU: {gpu_name} ({gpu_vram:.1f} GB)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._gpu_label.setText("GPU: Apple MPS (Metal)")
            gpu_vram = 8  # Assume shared memory
        else:
            self._gpu_label.setText("GPU: None detected (CPU only)")
        
        # Recommend model size
        if gpu_vram >= 12:
            recommended = "large"
            idx = 4
        elif gpu_vram >= 6:
            recommended = "medium"
            idx = 3
        elif gpu_vram >= 4 or ram_gb >= 16:
            recommended = "small"
            idx = 2
        elif ram_gb >= 8:
            recommended = "tiny"
            idx = 1
        else:
            recommended = "nano"
            idx = 0
        
        self._recommended_label.setText(f"Recommended Model: {recommended.title()}")
        self._model_buttons.buttons()[idx].setChecked(True)


class PrivacyPage(QWizardPage):
    """Page for privacy settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Privacy Settings")
        self.setSubTitle("Choose how Enigma AI Engine handles your data")
        
        layout = QVBoxLayout(self)
        
        # Privacy mode
        privacy_group = QGroupBox("Data Privacy")
        privacy_layout = QVBoxLayout(privacy_group)
        
        self._local_only = QCheckBox("Local Only Mode (Recommended)")
        self._local_only.setChecked(True)
        self._local_only.setToolTip(
            "All AI processing happens on your device.\n"
            "No data is sent to external servers."
        )
        privacy_layout.addWidget(self._local_only)
        
        local_desc = QLabel(
            "When enabled, Enigma AI Engine runs entirely on your device.\n"
            "Cloud-based features (like DALL-E) will be unavailable."
        )
        local_desc.setStyleSheet("color: #6c7086; font-size: 11px;")
        local_desc.setWordWrap(True)
        privacy_layout.addWidget(local_desc)
        
        layout.addWidget(privacy_group)
        
        # Telemetry
        telemetry_group = QGroupBox("Usage Data")
        tele_layout = QVBoxLayout(telemetry_group)
        
        self._telemetry = QCheckBox("Send anonymous usage statistics")
        self._telemetry.setChecked(False)
        self._telemetry.setToolTip(
            "Help improve Enigma AI Engine by sharing anonymous usage data.\n"
            "No personal data or conversations are ever sent."
        )
        tele_layout.addWidget(self._telemetry)
        
        tele_desc = QLabel(
            "Currently, Enigma AI Engine does not collect any telemetry.\n"
            "This option is reserved for future updates."
        )
        tele_desc.setStyleSheet("color: #6c7086; font-size: 11px;")
        tele_desc.setWordWrap(True)
        tele_layout.addWidget(tele_desc)
        
        layout.addWidget(telemetry_group)
        
        layout.addStretch()
        
        # Register fields
        self.registerField("local_only", self._local_only)
        self.registerField("telemetry", self._telemetry)


class InterfacePage(QWizardPage):
    """Page for interface preferences."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Interface Preferences")
        self.setSubTitle("Customize how Enigma AI Engine looks and behaves")
        
        layout = QVBoxLayout(self)
        
        # GUI Mode
        mode_group = QGroupBox("Interface Complexity")
        mode_layout = QVBoxLayout(mode_group)
        
        self._mode_buttons = QButtonGroup(self)
        
        modes = [
            ("simple", "Simple", "Essential features only - great for beginners"),
            ("standard", "Standard", "Balanced feature set - recommended"),
            ("advanced", "Advanced", "All features visible - for power users"),
        ]
        
        for i, (key, name, desc) in enumerate(modes):
            radio = QRadioButton(f"{name} - {desc}")
            radio.setProperty("gui_mode", key)
            self._mode_buttons.addButton(radio, i)
            mode_layout.addWidget(radio)
        
        self._mode_buttons.buttons()[1].setChecked(True)  # Default standard
        
        layout.addWidget(mode_group)
        
        # Voice options
        voice_group = QGroupBox("Voice Features")
        voice_layout = QVBoxLayout(voice_group)
        
        self._voice_input = QCheckBox("Enable voice input (microphone)")
        self._voice_output = QCheckBox("Enable voice output (text-to-speech)")
        
        voice_layout.addWidget(self._voice_input)
        voice_layout.addWidget(self._voice_output)
        
        layout.addWidget(voice_group)
        
        layout.addStretch()
        
        # Register fields
        self.registerField("voice_input", self._voice_input)
        self.registerField("voice_output", self._voice_output)


class FinishPage(QWizardPage):
    """Final page with summary and completion."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Setup Complete!")
        self.setSubTitle("You're ready to start using Enigma AI Engine")
        
        layout = QVBoxLayout(self)
        
        # Summary
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)
        
        layout.addSpacing(20)
        
        # Quick tips
        tips_group = QGroupBox("Quick Tips")
        tips_layout = QVBoxLayout(tips_group)
        
        tips = [
            "Press F1 anytime to see keyboard shortcuts",
            "Right-click anywhere for context menus",
            "Visit the Modules tab to enable more features",
            "Check Settings tab to customize further",
        ]
        
        for tip in tips:
            tip_label = QLabel(f"  {tip}")
            tip_label.setStyleSheet("color: #a6adc8;")
            tips_layout.addWidget(tip_label)
        
        layout.addWidget(tips_group)
        
        layout.addStretch()
        
        # "Don't show again" option
        self._dont_show = QCheckBox("Don't show this wizard on next launch")
        self._dont_show.setChecked(True)
        layout.addWidget(self._dont_show)
    
    def initializePage(self):
        """Update summary with selected options."""
        super().initializePage()
        
        wizard = self.wizard()
        
        # Get AI name
        ai_name = wizard.field("ai_name") or "Enigma"
        
        # Get selected model size
        hw_page = wizard.page(2)  # After Welcome (0) and AIName (1)
        model_size = "small"
        for btn in hw_page._model_buttons.buttons():
            if btn.isChecked():
                model_size = btn.property("model_size")
                break
        
        # Get other settings
        local_only = wizard.field("local_only")
        voice_input = wizard.field("voice_input")
        voice_output = wizard.field("voice_output")
        
        summary = f"""
Your configuration:

<b>AI Name:</b> {ai_name}
<b>Model Size:</b> {model_size.title()}
<b>Privacy Mode:</b> {'Local Only' if local_only else 'Cloud Enabled'}
<b>Voice Input:</b> {'Enabled' if voice_input else 'Disabled'}
<b>Voice Output:</b> {'Enabled' if voice_output else 'Disabled'}

Click Finish to save these settings and start using Enigma AI Engine.
"""
        self._summary.setText(summary)


class SetupWizard(QWizard):
    """First-run setup wizard for Enigma AI Engine."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Enigma AI Engine Setup")
        self.setWizardStyle(QWizard.ModernStyle)
        self.setMinimumSize(600, 500)
        self.setStyleSheet(WIZARD_STYLE)
        
        # Add pages
        self.addPage(WelcomePage(self))       # 0
        self.addPage(AINamePage(self))        # 1
        self.addPage(HardwareDetectionPage(self))  # 2
        self.addPage(PrivacyPage(self))       # 3
        self.addPage(InterfacePage(self))     # 4
        self.addPage(FinishPage(self))        # 5
        
        # Connect finish signal
        self.finished.connect(self._on_finished)
    
    def _on_finished(self, result):
        """Handle wizard completion."""
        if result == QWizard.Accepted:
            config = self.get_config()
            mark_wizard_complete(config)
            logger.info(f"Setup wizard completed with config: {config}")
    
    def get_config(self) -> dict[str, Any]:
        """Get the configuration from wizard selections."""
        # Get AI name
        ai_name = self.field("ai_name")
        if not ai_name or not ai_name.strip():
            ai_name = "Enigma"
        
        # Get model size
        hw_page = self.page(2)  # After Welcome (0) and AIName (1)
        model_size = "small"
        for btn in hw_page._model_buttons.buttons():
            if btn.isChecked():
                model_size = btn.property("model_size")
                break
        
        # Get GUI mode
        iface_page = self.page(4)  # After Welcome (0), AIName (1), Hardware (2), Privacy (3)
        gui_mode = "standard"
        for btn in iface_page._mode_buttons.buttons():
            if btn.isChecked():
                gui_mode = btn.property("gui_mode")
                break
        
        return {
            'ai_name': ai_name.strip(),
            'model_size': model_size,
            'local_only': self.field("local_only"),
            'telemetry': self.field("telemetry"),
            'gui_mode': gui_mode,
            'voice_input': self.field("voice_input"),
            'voice_output': self.field("voice_output"),
        }


def run_setup_wizard(parent=None) -> Optional[dict[str, Any]]:
    """
    Show the setup wizard if needed.
    
    Returns:
        Configuration dict if completed, None if cancelled or not needed
    """
    if not should_show_wizard():
        return None
    
    wizard = SetupWizard(parent)
    if wizard.exec_() == QWizard.Accepted:
        return wizard.get_config()
    return None
