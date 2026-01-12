"""
AI Tester GUI Package
==================

PyQt5-based graphical user interface for AI Tester.

Features:
  - Setup wizard for first-run configuration
  - Chat interface with AI
  - Training data editor and model training
  - Avatar control and vision preview
  - Module management
  - Theme customization (Dark/Light/Shadow/Midnight)
  - Resource monitoring
  - Multi-AI support

Usage:
    # Launch the GUI
    python run.py --gui
    
    # Or programmatically:
    from ai_tester.gui import launch_gui, EnigmaWindow
    launch_gui()

Components:
    - enhanced_window.py: Main window with all tabs
    - theme_system.py: Theme management and customization
    - resource_monitor.py: Real-time resource display
    - tabs/: Individual tab modules (chat, training, etc.)
"""

# Main window
try:
    from .enhanced_window import EnigmaWindow
except ImportError:
    EnigmaWindow = None

# Theme system
try:
    from .theme_system import (
        ThemeColors,
        Theme,
        ThemeManager,
    )
except ImportError:
    ThemeColors = None
    Theme = None
    ThemeManager = None

# Resource monitor
try:
    from .resource_monitor import ResourceMonitor
except ImportError:
    ResourceMonitor = None

# Tab modules
try:
    from .tabs import (
        create_chat_tab,
        create_training_tab,
        create_avatar_tab,
        create_vision_tab,
        create_sessions_tab,
        create_instructions_tab,
        create_terminal_tab,
        log_to_terminal,
        ModulesTab,
        ScalingTab,
        create_scaling_tab,
        ExamplesTab,
        create_examples_tab,
    )
except ImportError:
    # Tabs may not be available in all installations
    create_chat_tab = None
    create_training_tab = None
    create_avatar_tab = None
    create_vision_tab = None
    create_sessions_tab = None
    create_instructions_tab = None
    create_terminal_tab = None
    log_to_terminal = None
    ModulesTab = None
    ScalingTab = None
    create_scaling_tab = None
    ExamplesTab = None
    create_examples_tab = None


def launch_gui():
    """
    Launch the AI Tester GUI application.
    
    Returns:
        int: Exit code from the application
    """
    try:
        from PyQt5.QtWidgets import QApplication
        import sys
        
        app = QApplication(sys.argv)
        window = EnigmaWindow()
        window.show()
        return app.exec_()
    except ImportError as e:
        print(f"Error: PyQt5 not installed. Install with: pip install PyQt5")
        print(f"Details: {e}")
        return 1
    except Exception as e:
        print(f"Error launching GUI: {e}")
        return 1


__all__ = [
    # Main window
    'EnigmaWindow',
    'launch_gui',
    # Theme system
    'ThemeColors',
    'Theme',
    'ThemeManager',
    # Resource monitor
    'ResourceMonitor',
    # Tab creators
    'create_chat_tab',
    'create_training_tab',
    'create_avatar_tab',
    'create_vision_tab',
    'create_sessions_tab',
    'create_instructions_tab',
    'create_terminal_tab',
    'log_to_terminal',
    'create_examples_tab',
    # Tab classes
    'ModulesTab',
    'ScalingTab',
    'create_scaling_tab',
    'ExamplesTab',
]
