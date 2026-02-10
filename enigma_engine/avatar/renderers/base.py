"""
Base Avatar Renderer

Abstract base class for all avatar renderers.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..avatar_identity import AvatarAppearance


class BaseRenderer(ABC):
    """
    Base class for avatar renderers.
    
    All renderers must implement these methods to provide
    a consistent interface for the avatar controller.
    """
    
    def __init__(self, controller):
        """
        Initialize renderer.
        
        Args:
            controller: AvatarController instance
        """
        self.controller = controller
        self._visible = False
        self._current_appearance: Optional[AvatarAppearance] = None
        self._current_expression: str = "neutral"
    
    @abstractmethod
    def show(self) -> None:
        """Show avatar window/display."""
    
    @abstractmethod
    def hide(self) -> None:
        """Hide avatar window/display."""
    
    @abstractmethod
    def set_position(self, x: int, y: int) -> None:
        """
        Update avatar position on screen.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
    
    @abstractmethod
    def render_frame(self, animation_data: Optional[dict] = None) -> None:
        """
        Render a single frame.
        
        Args:
            animation_data: Optional animation state data
        """
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a 3D model or sprite sheet.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if loaded successfully
        """
        # Optional - not all renderers support models
        return False
    
    def set_scale(self, scale: float) -> None:
        """
        Set avatar scale/size.
        
        Args:
            scale: Scale factor (1.0 = normal)
        """
    
    def set_opacity(self, opacity: float) -> None:
        """
        Set avatar transparency.
        
        Args:
            opacity: 0.0 (transparent) to 1.0 (opaque)
        """
    
    def set_color(self, color: str) -> None:
        """
        Set avatar color/tint.
        
        Args:
            color: Hex color code (e.g., "#FF0000")
        """
    
    def set_appearance(self, appearance: AvatarAppearance) -> None:
        """
        Set complete avatar appearance.
        
        Args:
            appearance: AvatarAppearance instance
        """
        self._current_appearance = appearance
        
        # Apply colors
        self.set_color(appearance.primary_color)
        
        # Apply size
        size_map = {"small": 0.7, "medium": 1.0, "large": 1.5}
        self.set_scale(size_map.get(appearance.size, 1.0))
    
    def set_expression(self, expression: str) -> None:
        """
        Set avatar facial expression.
        
        Args:
            expression: Expression name (happy, sad, neutral, etc.)
        """
        self._current_expression = expression
    
    def play_animation(self, animation: str, duration: float = 1.0) -> None:
        """
        Play an animation.
        
        Args:
            animation: Animation name
            duration: Animation duration in seconds
        """
        # Default implementation - can be overridden
    
    @property
    def is_visible(self) -> bool:
        """Check if avatar is currently visible."""
        return self._visible
