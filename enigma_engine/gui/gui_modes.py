"""
GUI Mode System - Different interface complexity levels for different users.

This module defines the GUI mode system that controls what features and tabs
are visible to users based on their preference and use case.
"""
from enum import Enum


class GUIMode(Enum):
    """GUI display modes with different feature sets."""
    SIMPLE = "simple"      # Essential features only - chat and basics
    STANDARD = "standard"  # Balanced feature set - most common features
    ADVANCED = "advanced"  # All features exposed - power users
    GAMING = "gaming"      # Optimized for gaming sessions - minimal UI


class GUIModeManager:
    """Manages GUI mode and determines what features should be visible."""
    
    def __init__(self, mode: GUIMode = GUIMode.STANDARD):
        self.mode = mode
        
        # Define which tabs are visible in each mode
        self._mode_tabs = {
            GUIMode.SIMPLE: {
                'chat', 'settings'
            },
            GUIMode.STANDARD: {
                'chat', 'workspace', 'history',
                'image', 'code', 'video', 'audio',
                'avatar', 'settings'
            },
            GUIMode.ADVANCED: {
                # All tabs visible in advanced mode
                'chat', 'workspace', 'history',
                'scale', 'modules', 'tools', 'router',
                'image', 'code', 'video', 'audio', 'voice', '3d', 'gif',
                'search', 'avatar', 'game', 'robot', 'vision', 'camera',
                'terminal', 'files', 'logs', 'network', 'analytics', 'examples',
                'settings'
            },
            GUIMode.GAMING: {
                'chat', 'settings'  # Minimal for gaming
            }
        }
        
        # Define consolidated tabs for Standard mode
        # These are grouped tabs that combine multiple features
        self._consolidated_tabs = {
            GUIMode.STANDARD: {
                'create': ['image', 'code', 'video', 'audio'],  # Creation tab
                'ai': ['avatar', 'modules', 'scale'],            # AI configuration
            }
        }
    
    def get_visible_tabs(self) -> set[str]:
        """Get the set of tabs that should be visible in current mode."""
        return self._mode_tabs.get(self.mode, self._mode_tabs[GUIMode.STANDARD])
    
    def is_tab_visible(self, tab_key: str) -> bool:
        """Check if a specific tab should be visible in current mode."""
        return tab_key in self.get_visible_tabs()
    
    def get_consolidated_tabs(self) -> dict:
        """Get consolidated tab definitions for current mode."""
        return self._consolidated_tabs.get(self.mode, {})
    
    def set_mode(self, mode: GUIMode):
        """Change the GUI mode."""
        self.mode = mode
    
    def get_mode_name(self) -> str:
        """Get human-readable name of current mode."""
        names = {
            GUIMode.SIMPLE: "Simple Mode",
            GUIMode.STANDARD: "Standard Mode",
            GUIMode.ADVANCED: "Advanced Mode",
            GUIMode.GAMING: "Gaming Mode"
        }
        return names.get(self.mode, "Standard Mode")
    
    def get_mode_description(self) -> str:
        """Get description of current mode."""
        descriptions = {
            GUIMode.SIMPLE: "Essential features - perfect for beginners",
            GUIMode.STANDARD: "Balanced features - recommended for most users",
            GUIMode.ADVANCED: "All features - for power users",
            GUIMode.GAMING: "Minimal interface - optimized for gaming"
        }
        return descriptions.get(self.mode, "")


# Keyboard shortcuts for all modes
KEYBOARD_SHORTCUTS = {
    # Window
    "Alt+F4": "Close Enigma AI Engine",
    "Ctrl+Shift+F4": "Emergency quit",
    "Escape": "Close popup/overlay",
    
    # Navigation
    "Ctrl+1": "Switch to Chat tab",
    "Ctrl+2": "Switch to Create/Image tab", 
    "Ctrl+3": "Switch to AI/Avatar tab",
    "Ctrl+,": "Open Settings",
    
    # Chat
    "Ctrl+Enter": "Send message",
    "Ctrl+N": "New conversation",
    "Ctrl+Shift+V": "Paste and send",
    "Ctrl+L": "Clear chat",
    
    # Features
    "Ctrl+Shift+Space": "Toggle game mode",
    "Ctrl+M": "Toggle microphone",
    "Ctrl+Shift+S": "Toggle auto-speak",
    
    # General
    "F1": "Quick help",
    "Tab": "Navigate UI elements",
    "Ctrl+Plus": "Increase font size",
    "Ctrl+Minus": "Decrease font size",
}
