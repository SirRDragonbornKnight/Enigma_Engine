"""
ForgeAI GUI Package
==================

PyQt5-based graphical user interface for ForgeAI.

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
    from forge_ai.gui import launch_gui, ForgeWindow
    launch_gui()

Components:
    - enhanced_window.py: Main window with all tabs
    - theme_system.py: Theme management and customization
    - resource_monitor.py: Real-time resource display
    - tabs/: Individual tab modules (chat, training, etc.)
"""

from typing import TYPE_CHECKING

# Only import for type checking, not at runtime
if TYPE_CHECKING:
    from .enhanced_window import ForgeWindow
    from .resource_monitor import ResourceMonitor
    from .tabs import (
        ExamplesTab,
        ModulesTab,
        ScalingTab,
        create_avatar_tab,
        create_chat_tab,
        create_examples_tab,
        create_instructions_tab,
        create_scaling_tab,
        create_sessions_tab,
        create_terminal_tab,
        create_training_tab,
        create_vision_tab,
        log_to_terminal,
    )
    from .theme_system import Theme, ThemeColors, ThemeManager

# Lazy loading cache
_cache = {}

def __getattr__(name: str):
    """Lazy load GUI components only when accessed."""
    if name in _cache:
        return _cache[name]
    
    _imports = {
        # Main window
        'ForgeWindow': ('.enhanced_window', 'EnhancedMainWindow'),
        'EnhancedMainWindow': ('.enhanced_window', 'EnhancedMainWindow'),
        # Theme system
        'ThemeColors': ('.theme_system', 'ThemeColors'),
        'Theme': ('.theme_system', 'Theme'),
        'ThemeManager': ('.theme_system', 'ThemeManager'),
        # Resource monitor
        'ResourceMonitor': ('.resource_monitor', 'ResourceMonitor'),
        # Tab creators and classes (forward to tabs module)
        'create_chat_tab': ('.tabs', 'create_chat_tab'),
        'create_training_tab': ('.tabs', 'create_training_tab'),
        'create_avatar_tab': ('.tabs', 'create_avatar_tab'),
        'create_vision_tab': ('.tabs', 'create_vision_tab'),
        'create_sessions_tab': ('.tabs', 'create_sessions_tab'),
        'create_instructions_tab': ('.tabs', 'create_instructions_tab'),
        'create_terminal_tab': ('.tabs', 'create_terminal_tab'),
        'log_to_terminal': ('.tabs', 'log_to_terminal'),
        'ModulesTab': ('.tabs', 'ModulesTab'),
        'ScalingTab': ('.tabs', 'ScalingTab'),
        'create_scaling_tab': ('.tabs', 'create_scaling_tab'),
        'ExamplesTab': ('.tabs', 'ExamplesTab'),
        'create_examples_tab': ('.tabs', 'create_examples_tab'),
        # Safety guards
        'SafetyGuards': ('.safety_guards', 'SafetyGuards'),
        'confirm_action': ('.safety_guards', 'confirm_action'),
        'confirm_destructive': ('.safety_guards', 'confirm_destructive'),
        'validate_input': ('.safety_guards', 'validate_input'),
        'rate_limit': ('.safety_guards', 'rate_limit'),
        'ActionHistory': ('.safety_guards', 'ActionHistory'),
        'InputValidator': ('.safety_guards', 'InputValidator'),
    }
    
    if name not in _imports:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    
    module_path, attr_name = _imports[name]
    
    try:
        import importlib
        module = importlib.import_module(module_path, __package__)
        value = getattr(module, attr_name)
        _cache[name] = value
        return value
    except ImportError:
        return None


def launch_gui():
    """
    Launch the ForgeAI GUI application.
    
    Returns:
        int: Exit code from the application
    """
    try:
        import sys

        from PyQt5.QtCore import QCoreApplication, Qt
        from PyQt5.QtWidgets import QApplication

        # Enable high-DPI scaling for sharp rendering on 4K/Retina displays
        # Must be set before QApplication is created
        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        from .enhanced_window import EnhancedMainWindow
        
        app = QApplication(sys.argv)
        window = EnhancedMainWindow()
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
    'ForgeWindow',
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
    # Safety guards
    'SafetyGuards',
    'confirm_action',
    'confirm_destructive',
    'validate_input',
    'rate_limit',
    'ActionHistory',
    'InputValidator',
]
