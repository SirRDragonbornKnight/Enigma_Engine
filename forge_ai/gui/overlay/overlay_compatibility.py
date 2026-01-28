"""
Overlay Compatibility - Game and application compatibility detection.

Helps the overlay adjust its behavior based on the target application
and its display mode.
"""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class OverlayCompatibility:
    """
    Ensure overlay works with games and fullscreen applications.
    """
    
    def __init__(self):
        self._cached_mode: Optional[str] = None
        
    def detect_game_mode(self) -> str:
        """
        Detect game display mode.
        
        Returns:
            "fullscreen", "borderless", "windowed", or "unknown"
        """
        # This is a basic implementation - full detection would require
        # platform-specific APIs (e.g., Windows GetWindowPlacement)
        
        try:
            if sys.platform == 'win32':
                return self._detect_windows_mode()
            elif sys.platform == 'darwin':
                return self._detect_macos_mode()
            else:  # Linux
                return self._detect_linux_mode()
        except Exception as e:
            logger.warning(f"Could not detect display mode: {e}")
            return "unknown"
            
    def _detect_windows_mode(self) -> str:
        """Detect display mode on Windows."""
        # Would use win32gui/ctypes to check window styles
        # For now, return windowed as safe default
        return "windowed"
        
    def _detect_macos_mode(self) -> str:
        """Detect display mode on macOS."""
        # Would use Cocoa/AppKit to check window level
        return "windowed"
        
    def _detect_linux_mode(self) -> str:
        """Detect display mode on Linux."""
        # Would check X11/Wayland window properties
        return "windowed"
        
    def adjust_for_game(self, game_mode: str):
        """
        Adjust overlay behavior for game mode.
        
        Args:
            game_mode: Display mode ("fullscreen", "borderless", "windowed")
            
        Returns:
            Dict with recommended settings
        """
        adjustments = {
            "window_level": "normal",
            "use_transparency": True,
            "click_through_recommended": False,
        }
        
        if game_mode == "fullscreen":
            # Fullscreen games may need special handling
            adjustments["window_level"] = "above_all"
            adjustments["click_through_recommended"] = True
            logger.info("Fullscreen mode detected - enabling click-through recommendation")
            
        elif game_mode == "borderless":
            # Borderless windowed usually works well with standard overlay
            adjustments["window_level"] = "normal"
            adjustments["use_transparency"] = True
            
        elif game_mode == "windowed":
            # Standard windowed mode
            adjustments["window_level"] = "normal"
            
        return adjustments
        
    def check_game_overlay_support(self, game_exe: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if game allows overlays.
        
        Some games with anti-cheat software may block overlays.
        
        Args:
            game_exe: Path to game executable (if known)
            
        Returns:
            Dict with support info:
                - supported: bool
                - notes: str
                - anti_cheat: bool
        """
        result = {
            "supported": True,
            "notes": "No issues detected",
            "anti_cheat": False,
        }
        
        if not game_exe:
            return result
            
        # Known problematic games/anti-cheat systems
        # This is a basic check - would need a more comprehensive database
        problematic_patterns = [
            "battleye",
            "easyanticheat",
            "vanguard",
            "gameguard",
        ]
        
        game_path = Path(game_exe).stem.lower()
        
        for pattern in problematic_patterns:
            if pattern in game_path:
                result["supported"] = False
                result["anti_cheat"] = True
                result["notes"] = f"May be blocked by anti-cheat: {pattern}"
                logger.warning(f"Potential overlay conflict: {pattern}")
                break
                
        return result
        
    def get_recommended_settings(self) -> Dict[str, Any]:
        """
        Get recommended overlay settings for current environment.
        
        Returns:
            Dict with recommended settings
        """
        game_mode = self.detect_game_mode()
        adjustments = self.adjust_for_game(game_mode)
        
        return {
            "mode": game_mode,
            "adjustments": adjustments,
            "platform": sys.platform,
        }
