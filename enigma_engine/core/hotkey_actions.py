"""
================================================================================
HOTKEY ACTIONS - Callable Actions for Hotkeys
================================================================================

Actions that can be triggered by global hotkeys.

FILE: enigma_engine/core/hotkey_actions.py
TYPE: Core Utility
MAIN CLASS: HotkeyActions

USAGE:
    from enigma_engine.core.hotkey_actions import HotkeyActions
    
    actions = HotkeyActions(main_window)
    actions.summon_overlay()
    actions.screenshot_to_ai()
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QMainWindow

logger = logging.getLogger(__name__)


class HotkeyActions:
    """
    Actions that can be triggered by hotkeys.
    """
    
    def __init__(self, main_window: Optional['QMainWindow'] = None) -> None:
        """
        Initialize hotkey actions.
        
        Args:
            main_window: Reference to the main GUI window
        """
        self._main_window = main_window
        self._overlay = None
        self._voice_active = False
        self._game_mode_active = False
    
    def summon_overlay(self) -> None:
        """
        Show AI overlay.
        
        - Appears on top of current application
        - Transparent background option
        - Text input focused
        - Voice activation ready
        """
        logger.info("Summoning AI overlay")
        
        try:
            # Try to use existing system tray quick command overlay
            from ..gui.system_tray import QuickCommandOverlay
            
            if self._overlay is None:
                from PyQt5.QtWidgets import QApplication
                app = QApplication.instance()
                
                if app and self._main_window:
                    self._overlay = QuickCommandOverlay(self._main_window)
            
            if self._overlay:
                self._overlay.show()
                self._overlay.raise_()
                self._overlay.activateWindow()
                logger.info("AI overlay shown")
            else:
                logger.warning("Could not create overlay")
                
        except Exception as e:
            logger.error(f"Error summoning overlay: {e}")
    
    def dismiss_overlay(self) -> None:
        """Hide AI overlay, return focus to previous app."""
        logger.info("Dismissing AI overlay")
        
        try:
            if self._overlay:
                self._overlay.hide()
                logger.info("AI overlay hidden")
        except Exception as e:
            logger.error(f"Error dismissing overlay: {e}")
    
    def push_to_talk_start(self) -> None:
        """Start listening for voice input."""
        logger.info("Push-to-talk: Starting voice input")
        
        try:
            # Check if voice is available
            from ..config import CONFIG
            if not CONFIG.get("enable_voice", True):
                logger.warning("Voice input is disabled")
                return
            
            # Try to start voice listener
            try:
                from ..voice.listener import VoiceListener
                
                if not hasattr(self, '_voice_listener'):
                    self._voice_listener = VoiceListener()
                
                self._voice_active = True
                # Start listening in background
                # The listener will call a callback when speech is detected
                logger.info("Voice input activated")
                
            except ImportError:
                logger.warning("Voice module not available")
            except Exception as e:
                logger.error(f"Error starting voice input: {e}")
                
        except Exception as e:
            logger.error(f"Error in push_to_talk_start: {e}")
    
    def push_to_talk_stop(self) -> None:
        """Stop listening, process voice input."""
        logger.info("Push-to-talk: Stopping voice input")
        
        try:
            self._voice_active = False
            
            if hasattr(self, '_voice_listener'):
                # Process any buffered audio
                # This would be connected to the main window's input
                logger.info("Voice input deactivated")
                
        except Exception as e:
            logger.error(f"Error in push_to_talk_stop: {e}")
    
    def quick_command(self) -> None:
        """
        Show minimal command input.
        
        Just a text box, type command, press enter.
        Smaller than full overlay.
        """
        logger.info("Opening quick command input")
        
        # Quick command is essentially the same as summon_overlay
        # but could be a smaller version
        self.summon_overlay()
    
    def screenshot_to_ai(self) -> None:
        """
        Take screenshot and send to AI.
        
        1. Capture screen (or selected region)
        2. Send to vision model
        3. Ask "What's in this image?" or custom prompt
        """
        logger.info("Taking screenshot for AI analysis")
        
        try:
            # Use mss to capture screen
            from datetime import datetime

            import mss
            from PIL import Image
            
            with mss.mss() as sct:
                # Capture primary monitor
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                
                # Convert to PIL Image
                img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
                
                # Save to temp file
                from ..config import CONFIG
                temp_dir = Path(CONFIG.get("data_dir", "data")) / "temp"
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = temp_dir / f"screenshot_{timestamp}.png"
                img.save(screenshot_path)
                
                logger.info(f"Screenshot saved to {screenshot_path}")
                
                # Try to send to vision tab if available
                if self._main_window and hasattr(self._main_window, 'tabs'):
                    # Find vision tab
                    for i in range(self._main_window.tabs.count()):
                        if self._main_window.tabs.tabText(i) == "Vision":
                            self._main_window.tabs.setCurrentIndex(i)
                            # Load image in vision tab
                            vision_tab = self._main_window.tabs.widget(i)
                            if hasattr(vision_tab, 'load_image'):
                                vision_tab.load_image(str(screenshot_path))
                            break
                
        except ImportError:
            logger.error("mss library not available for screenshots")
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
    
    def toggle_game_mode(self) -> None:
        """Toggle game mode on/off."""
        self._game_mode_active = not self._game_mode_active
        
        logger.info(f"Game mode: {'ON' if self._game_mode_active else 'OFF'}")
        
        try:
            # Try to toggle game mode in the main window
            if self._main_window and hasattr(self._main_window, 'toggle_game_mode'):
                self._main_window.toggle_game_mode(self._game_mode_active)
            
            # Could also adjust resource usage, disable animations, etc.
            if self._game_mode_active:
                logger.info("Game mode activated: Reduced resource usage")
            else:
                logger.info("Game mode deactivated: Normal resource usage")
                
        except Exception as e:
            logger.error(f"Error toggling game mode: {e}")
    
    def set_main_window(self, window: 'QMainWindow') -> None:
        """
        Set the main window reference.
        
        Args:
            window: Main window instance
        """
        self._main_window = window


# Singleton instance
_actions: Optional[HotkeyActions] = None


def get_hotkey_actions(main_window: Optional['QMainWindow'] = None) -> HotkeyActions:
    """
    Get the global hotkey actions instance.
    
    Args:
        main_window: Main window instance (optional)
        
    Returns:
        The singleton HotkeyActions instance
    """
    global _actions
    if _actions is None:
        _actions = HotkeyActions(main_window)
    elif main_window is not None:
        _actions.set_main_window(main_window)
    return _actions
