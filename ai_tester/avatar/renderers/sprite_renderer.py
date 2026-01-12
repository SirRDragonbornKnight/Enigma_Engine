"""
Sprite-based Avatar Renderer

Simple 2D sprite renderer that works out of the box.
Uses built-in SVG sprites with customizable colors.
"""

import time
from typing import Optional, Dict
from pathlib import Path

from .base import BaseRenderer
from .default_sprites import generate_sprite, SPRITE_TEMPLATES, save_sprite
from ..avatar_identity import AvatarAppearance


class SpriteRenderer(BaseRenderer):
    """
    Simple 2D sprite renderer using built-in sprites.
    
    This is the default renderer - lightweight and works everywhere.
    Renders to console or can save sprites to files.
    """
    
    def __init__(self, controller):
        """
        Initialize sprite renderer.
        
        Args:
            controller: AvatarController instance
        """
        super().__init__(controller)
        self._sprite_cache: Dict[str, str] = {}
        self._current_sprite_name = "idle"
        self._animation_frame = 0
        self._last_render_time = 0
    
    def show(self) -> None:
        """Show avatar (console output or save to file)."""
        self._visible = True
        pos = self.controller.position
        print(f"[SpriteRenderer] Avatar shown at ({pos.x}, {pos.y})")
        print(f"[SpriteRenderer] Current sprite: {self._current_sprite_name}")
        
        # Generate current sprite if needed
        if self._current_appearance:
            self._update_sprite()
    
    def hide(self) -> None:
        """Hide avatar."""
        self._visible = False
        print("[SpriteRenderer] Avatar hidden")
    
    def set_position(self, x: int, y: int) -> None:
        """
        Update avatar position.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        if self._visible:
            print(f"[SpriteRenderer] Moved to ({x}, {y})")
    
    def render_frame(self, animation_data: Optional[Dict] = None) -> None:
        """
        Render a single frame.
        
        Args:
            animation_data: Optional animation state
        """
        if not self._visible:
            return
        
        current_time = time.time()
        
        # Throttle rendering (avoid spam)
        if current_time - self._last_render_time < 0.1:  # 10 FPS
            return
        
        self._last_render_time = current_time
        
        # Handle animation
        if animation_data:
            anim_type = animation_data.get("type", "idle")
            
            # Update sprite based on animation
            if anim_type == "speak":
                # Alternate between speaking frames
                self._animation_frame = (self._animation_frame + 1) % 2
                self._current_sprite_name = f"speaking_{self._animation_frame + 1}"
            elif anim_type == "think":
                self._current_sprite_name = "thinking"
            elif anim_type == "expression":
                expression = animation_data.get("expression", "neutral")
                self._current_sprite_name = expression
            else:
                self._current_sprite_name = "idle"
        
        # Generate/cache sprite
        self._update_sprite()
    
    def _update_sprite(self):
        """Update current sprite with appearance colors."""
        if not self._current_appearance:
            return
        
        cache_key = f"{self._current_sprite_name}_{self._current_appearance.primary_color}"
        
        if cache_key not in self._sprite_cache:
            # Generate sprite with current appearance colors
            svg = generate_sprite(
                self._current_sprite_name,
                self._current_appearance.primary_color,
                self._current_appearance.secondary_color,
                self._current_appearance.accent_color
            )
            self._sprite_cache[cache_key] = svg
    
    def set_appearance(self, appearance: AvatarAppearance) -> None:
        """
        Set complete avatar appearance.
        
        Args:
            appearance: AvatarAppearance instance
        """
        super().set_appearance(appearance)
        
        # Clear cache to force regeneration with new colors
        self._sprite_cache.clear()
        
        if self._visible:
            print(f"[SpriteRenderer] Appearance updated:")
            print(f"  Style: {appearance.style}")
            print(f"  Colors: {appearance.primary_color}, {appearance.secondary_color}, {appearance.accent_color}")
            print(f"  Shape: {appearance.shape}, Size: {appearance.size}")
            self._update_sprite()
    
    def set_expression(self, expression: str) -> None:
        """
        Set avatar facial expression.
        
        Args:
            expression: Expression name
        """
        super().set_expression(expression)
        
        # Map expression to sprite
        if expression in SPRITE_TEMPLATES:
            self._current_sprite_name = expression
        else:
            self._current_sprite_name = "idle"
        
        if self._visible:
            print(f"[SpriteRenderer] Expression: {expression}")
            self._update_sprite()
    
    def play_animation(self, animation: str, duration: float = 1.0) -> None:
        """
        Play an animation.
        
        Args:
            animation: Animation name
            duration: Duration in seconds
        """
        if self._visible:
            print(f"[SpriteRenderer] Animation: {animation} ({duration}s)")
        
        # Map animation to appropriate sprite
        anim_sprite_map = {
            "idle": "idle",
            "speak": "speaking_1",
            "think": "thinking",
            "happy": "happy",
            "sad": "sad",
            "surprised": "surprised",
            "confused": "confused",
            "excited": "excited",
        }
        
        sprite = anim_sprite_map.get(animation, "idle")
        self._current_sprite_name = sprite
        self._update_sprite()
    
    def export_sprites(self, output_dir: str) -> bool:
        """
        Export all sprites to files.
        
        Args:
            output_dir: Directory to export to
            
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if not self._current_appearance:
                print("[SpriteRenderer] No appearance set, using defaults")
                appearance = AvatarAppearance()
            else:
                appearance = self._current_appearance
            
            # Export all sprite templates
            for sprite_name in SPRITE_TEMPLATES.keys():
                save_sprite(
                    sprite_name,
                    str(output_path / f"{sprite_name}.svg"),
                    appearance.primary_color,
                    appearance.secondary_color,
                    appearance.accent_color
                )
            
            print(f"[SpriteRenderer] Exported {len(SPRITE_TEMPLATES)} sprites to {output_dir}")
            return True
            
        except Exception as e:
            print(f"[SpriteRenderer] Export failed: {e}")
            return False
    
    def get_current_sprite_svg(self) -> Optional[str]:
        """
        Get current sprite as SVG string.
        
        Returns:
            SVG string or None
        """
        if not self._current_appearance:
            return None
        
        return generate_sprite(
            self._current_sprite_name,
            self._current_appearance.primary_color,
            self._current_appearance.secondary_color,
            self._current_appearance.accent_color
        )
