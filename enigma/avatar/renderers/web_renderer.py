"""
Web Avatar Renderer

Renders avatar in web browser via WebSocket for web dashboard integration.
"""

from typing import Optional, Dict
import json

from .base import BaseRenderer
from .default_sprites import get_sprite_data_url
from ..avatar_identity import AvatarAppearance


class WebAvatarRenderer(BaseRenderer):
    """
    Renders avatar in web browser.
    
    Sends avatar state to web clients via WebSocket (Flask-SocketIO).
    Used by web dashboard for browser-based avatar display.
    """
    
    def __init__(self, controller, socketio=None):
        """
        Initialize web renderer.
        
        Args:
            controller: AvatarController instance
            socketio: Flask-SocketIO instance (optional)
        """
        super().__init__(controller)
        self.socketio = socketio
        self._current_sprite_name = "idle"
    
    def show(self) -> None:
        """Show avatar in web clients."""
        self._visible = True
        self._emit_state({
            "visible": True,
            "position": {
                "x": self.controller.position.x,
                "y": self.controller.position.y,
            }
        })
        print("[WebAvatarRenderer] Avatar shown to web clients")
    
    def hide(self) -> None:
        """Hide avatar in web clients."""
        self._visible = False
        self._emit_state({"visible": False})
        print("[WebAvatarRenderer] Avatar hidden from web clients")
    
    def set_position(self, x: int, y: int) -> None:
        """
        Update avatar position.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        if self._visible:
            self._emit_state({
                "position": {"x": x, "y": y}
            })
    
    def render_frame(self, animation_data: Optional[Dict] = None) -> None:
        """
        Render a single frame.
        
        Args:
            animation_data: Optional animation state
        """
        if not self._visible:
            return
        
        # Update sprite based on animation
        if animation_data:
            anim_type = animation_data.get("type", "idle")
            
            if anim_type == "speak":
                # Send speaking animation
                self._emit_animation({
                    "type": "speak",
                    "text": animation_data.get("text", "")
                })
            elif anim_type == "expression":
                expression = animation_data.get("expression", "neutral")
                self.set_expression(expression)
            elif anim_type == "think":
                self.set_expression("thinking")
    
    def set_appearance(self, appearance: AvatarAppearance) -> None:
        """
        Set complete avatar appearance.
        
        Args:
            appearance: AvatarAppearance instance
        """
        super().set_appearance(appearance)
        
        if self._visible:
            self._emit_state({
                "appearance": appearance.to_dict()
            })
            
            # Send all sprites with new colors
            self._emit_sprites()
    
    def set_expression(self, expression: str) -> None:
        """
        Set avatar facial expression.
        
        Args:
            expression: Expression name
        """
        super().set_expression(expression)
        self._current_sprite_name = expression if expression else "idle"
        
        if self._visible:
            self._emit_expression(expression)
    
    def set_scale(self, scale: float) -> None:
        """
        Set avatar scale.
        
        Args:
            scale: Scale factor (1.0 = normal)
        """
        if self._visible:
            self._emit_state({"scale": scale})
    
    def set_opacity(self, opacity: float) -> None:
        """
        Set avatar opacity.
        
        Args:
            opacity: 0.0 (transparent) to 1.0 (opaque)
        """
        if self._visible:
            self._emit_state({"opacity": opacity})
    
    def play_animation(self, animation: str, duration: float = 1.0) -> None:
        """
        Play an animation.
        
        Args:
            animation: Animation name
            duration: Duration in seconds
        """
        if self._visible:
            self._emit_animation({
                "type": animation,
                "duration": duration
            })
    
    def _emit_state(self, state: dict):
        """
        Send avatar state to web clients.
        
        Args:
            state: State dictionary
        """
        if not self.socketio:
            return
        
        try:
            self.socketio.emit('avatar_state', state, namespace='/avatar')
        except Exception as e:
            print(f"[WebAvatarRenderer] Failed to emit state: {e}")
    
    def _emit_expression(self, expression: str):
        """
        Send expression update to web clients.
        
        Args:
            expression: Expression name
        """
        if not self.socketio:
            return
        
        try:
            # Get sprite data URL for the expression
            if self._current_appearance:
                sprite_url = get_sprite_data_url(
                    expression,
                    self._current_appearance.primary_color,
                    self._current_appearance.secondary_color,
                    self._current_appearance.accent_color
                )
            else:
                sprite_url = get_sprite_data_url(expression)
            
            self.socketio.emit('avatar_expression', {
                "expression": expression,
                "sprite": sprite_url
            }, namespace='/avatar')
        except Exception as e:
            print(f"[WebAvatarRenderer] Failed to emit expression: {e}")
    
    def _emit_animation(self, animation: dict):
        """
        Send animation to web clients.
        
        Args:
            animation: Animation data
        """
        if not self.socketio:
            return
        
        try:
            self.socketio.emit('avatar_animation', animation, namespace='/avatar')
        except Exception as e:
            print(f"[WebAvatarRenderer] Failed to emit animation: {e}")
    
    def _emit_sprites(self):
        """Send all sprite data URLs to web clients."""
        if not self.socketio or not self._current_appearance:
            return
        
        try:
            # Common sprite names
            sprite_names = [
                "idle", "happy", "sad", "thinking", "surprised",
                "confused", "excited", "speaking_1", "speaking_2"
            ]
            
            sprites = {}
            for name in sprite_names:
                sprites[name] = get_sprite_data_url(
                    name,
                    self._current_appearance.primary_color,
                    self._current_appearance.secondary_color,
                    self._current_appearance.accent_color
                )
            
            self.socketio.emit('avatar_sprites', sprites, namespace='/avatar')
        except Exception as e:
            print(f"[WebAvatarRenderer] Failed to emit sprites: {e}")
