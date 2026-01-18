"""
Avatar Tools - Control and customize the desktop avatar.

Tools:
  - control_avatar: Move, show, hide, jump, pin the avatar
  - customize_avatar: Change colors, lighting, rotation
  - avatar_gesture: Make the avatar perform gestures/animations
"""

import json
import time
from pathlib import Path
from typing import Dict, Any
from .tool_registry import Tool


# Path to AI command file - avatar reads this
AVATAR_COMMAND_FILE = Path(__file__).parent.parent.parent / "data" / "avatar" / "ai_command.json"


def _send_avatar_command(action: str, value: str = "") -> Dict[str, Any]:
    """Write a command to the avatar command file for the avatar to pick up."""
    try:
        AVATAR_COMMAND_FILE.parent.mkdir(parents=True, exist_ok=True)
        command = {
            "action": action,
            "value": value,
            "timestamp": time.time()
        }
        AVATAR_COMMAND_FILE.write_text(json.dumps(command, indent=2))
        return {"success": True, "action": action, "value": value}
    except Exception as e:
        return {"success": False, "error": str(e)}


class AvatarControlTool(Tool):
    """
    Control the desktop avatar - move, show, hide, jump, pin.
    """
    
    name = "control_avatar"
    description = "Control the desktop avatar - move it, make it jump, pin it in place, or show/hide it"
    parameters = {
        "action": "Action: show, hide, jump, pin, unpin, move, resize, orientation",
        "value": "Value for action (coords for move, size for resize, direction for orientation)",
    }
    
    def execute(self, action: str, value: str = "", **kwargs) -> Dict[str, Any]:
        action = action.lower().strip()
        
        valid_actions = ["show", "hide", "jump", "pin", "unpin", "move", "resize", "orientation"]
        if action not in valid_actions:
            return {"success": False, "error": f"Invalid action '{action}'. Valid: {valid_actions}"}
        
        # Format value for specific actions
        if action == "orientation" and value:
            presets = {"front": "front", "back": "back", "left": "left", "right": "right"}
            if value.lower() in presets:
                value = presets[value.lower()]
        
        return _send_avatar_command(action, value)


class AvatarCustomizeTool(Tool):
    """
    Customize the avatar's visual appearance.
    """
    
    name = "customize_avatar"
    description = "Customize the avatar's colors, lighting, rotation speed, and visual effects"
    parameters = {
        "setting": "Setting to change: primary_color, secondary_color, accent_color, light_intensity, ambient_strength, wireframe, show_grid, rotate_speed, auto_rotate, reset",
        "value": "Value for setting (hex color, number 0-100, or true/false)",
    }
    
    VALID_SETTINGS = [
        "primary_color", "secondary_color", "accent_color",
        "light_intensity", "ambient_strength",
        "wireframe", "show_grid", "auto_rotate",
        "rotate_speed", "reset"
    ]
    
    def execute(self, setting: str, value: str, **kwargs) -> Dict[str, Any]:
        setting = setting.lower().strip()
        
        if setting not in self.VALID_SETTINGS:
            return {"success": False, "error": f"Invalid setting '{setting}'. Valid: {self.VALID_SETTINGS}"}
        
        # The avatar doesn't directly support customize commands yet,
        # but we can use existing animation/visual commands
        # For now, pass through as-is
        return _send_avatar_command(f"customize_{setting}", str(value))


class AvatarGestureTool(Tool):
    """
    Make the avatar perform gestures/animations (Atlas/Portal 2 style).
    """
    
    name = "avatar_gesture"
    description = "Make the avatar perform a gesture or animation"
    parameters = {
        "gesture": "Gesture to perform: wave, nod, shake (head), blink, speak",
        "intensity": "Optional intensity 0.5-2.0 (default 1.0)",
    }
    
    VALID_GESTURES = ["wave", "nod", "shake", "blink", "speak", "look_at"]
    
    def execute(self, gesture: str, intensity: float = 1.0, **kwargs) -> Dict[str, Any]:
        gesture = gesture.lower().strip()
        
        if gesture not in self.VALID_GESTURES:
            return {"success": False, "error": f"Invalid gesture '{gesture}'. Valid: {self.VALID_GESTURES}"}
        
        try:
            intensity = max(0.5, min(2.0, float(intensity)))
        except (ValueError, TypeError):
            intensity = 1.0
        
        return _send_avatar_command(gesture, str(intensity))


class AvatarEmotionTool(Tool):
    """
    Set the avatar's emotional expression.
    """
    
    name = "avatar_emotion"
    description = "Set the avatar's emotional expression/mood"
    parameters = {
        "emotion": "Emotion to express: happy, sad, angry, surprised, neutral, excited, thinking, confused, love, scared",
    }
    
    VALID_EMOTIONS = [
        "happy", "sad", "angry", "surprised", "neutral",
        "excited", "thinking", "confused", "love", "scared",
        "bored", "curious", "proud", "embarrassed"
    ]
    
    def execute(self, emotion: str, **kwargs) -> Dict[str, Any]:
        emotion = emotion.lower().strip()
        
        if emotion not in self.VALID_EMOTIONS:
            return {"success": False, "error": f"Invalid emotion '{emotion}'. Valid: {self.VALID_EMOTIONS}"}
        
        return _send_avatar_command("emotion", emotion)
