"""
Forge Background Service - Run AI without the full GUI.

This starts Forge in the system tray with quick command access.
The full GUI can be opened from the tray menu when needed.

Usage:
    python -m forge_ai.background
    
Or from run.py:
    python run.py --background
"""

import sys

def main():
    """Run Forge in background mode (system tray only)."""
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
    except ImportError:
        print("Error: PyQt5 is required. Install with: pip install PyQt5")
        sys.exit(1)
    
    # Create minimal app
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setQuitOnLastWindowClosed(False)
    
    # Create system tray
    try:
        from .gui.system_tray import create_system_tray
        tray = create_system_tray(app, main_window=None)
        
        if tray is None:
            print("Error: System tray not available on this system.")
            print("Please run with --gui instead.")
            sys.exit(1)
        
        # Override show_gui to create window on demand
        def show_gui_on_demand():
            from .gui.enhanced_window import EnhancedMainWindow
            window = EnhancedMainWindow()
            tray.main_window = window
            window.show()
            
            # Reconnect close to minimize
            original_close = window.closeEvent
            def close_to_tray(event):
                event.ignore()
                window.hide()
            window.closeEvent = close_to_tray
        
        tray.show_gui_requested.connect(show_gui_on_demand)
        
        # Show startup notification
        tray.show_notification(
            "Forge Background Service",
            "Forge is running in the background.\n"
            "Click tray icon or press Ctrl+Space for commands."
        )
        
        print("Forge running in background. Press Ctrl+C to exit.")
        
    except Exception as e:
        print(f"Error starting background service: {e}")
        sys.exit(1)
    
    sys.exit(app.exec_())


# Alias for imports
BackgroundService = main


if __name__ == "__main__":
    main()
