"""First Run Wizard for ForgeAI."""

try:
    from PyQt5.QtWidgets import QWizard, QWizardPage, QVBoxLayout, QLabel, QLineEdit, QRadioButton, QButtonGroup, QFormLayout
    from PyQt5.QtCore import pyqtSignal
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


class FirstRunWizard(QWizard):
    """First-run setup wizard."""
    
    setup_complete = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to ForgeAI")
        self.resize(700, 550)
        
        self.user_settings = {'gui_mode': 'standard', 'ai_name': None}
        self.addPage(self._create_welcome_page())
        self.addPage(self._create_mode_page())
        self.addPage(self._create_done_page())
    
    def _create_welcome_page(self):
        page = QWizardPage()
        page.setTitle("Welcome")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Welcome to ForgeAI!"))
        page.setLayout(layout)
        return page
    
    def _create_mode_page(self):
        page = QWizardPage()
        page.setTitle("Choose Mode")
        layout = QVBoxLayout()
        self.mode_group = QButtonGroup(page)
        for i, (mode_id, name) in enumerate([('simple', 'Simple'), ('standard', 'Standard'), ('advanced', 'Advanced'), ('gaming', 'Gaming')]):
            radio = QRadioButton(name)
            radio.setProperty('mode_id', mode_id)
            if mode_id == 'standard':
                radio.setChecked(True)
            self.mode_group.addButton(radio, i)
            layout.addWidget(radio)
        self.mode_group.buttonClicked.connect(lambda b: self.user_settings.update({'gui_mode': b.property('mode_id')}))
        page.setLayout(layout)
        return page
    
    def _create_done_page(self):
        page = QWizardPage()
        page.setTitle("Done")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Setup complete!"))
        page.setLayout(layout)
        return page
    
    def accept(self):
        self.setup_complete.emit(self.user_settings)
        super().accept()


def show_first_run_wizard(parent=None):
    """Show first-run wizard."""
    wizard = FirstRunWizard(parent)
    settings = None
    wizard.setup_complete.connect(lambda s: setattr(wizard, '_settings', s))
    if wizard.exec_():
        return getattr(wizard, '_settings', None)
    return None
