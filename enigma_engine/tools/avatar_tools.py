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
from typing import Any

from .tool_registry import RichParameter, Tool

# Path to AI command file - avatar reads this
AVATAR_COMMAND_FILE = Path(__file__).parent.parent.parent / "data" / "avatar" / "ai_command.json"


def _send_avatar_command(action: str, value: str = "") -> dict[str, Any]:
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
    """Control the desktop avatar - move, show, hide, jump, pin."""
    
    name = "control_avatar"
    description = "Control the desktop avatar - move it, make it jump, pin it in place, or show/hide it. The avatar is a 3D character that appears on your desktop."
    category = "avatar"
    
    rich_parameters = [
        RichParameter(
            name="action",
            type="string",
            description="Action to perform on the avatar",
            required=True,
            enum=["show", "hide", "jump", "pin", "unpin", "move", "resize", "orientation"],
        ),
        RichParameter(
            name="value",
            type="string",
            description="Value for the action. For 'move': 'x,y' coordinates. For 'resize': pixel size like '250'. For 'orientation': 'front', 'back', 'left', 'right'",
            required=False,
            default="",
        ),
    ]
    
    examples = [
        "Show the avatar on desktop",
        "Make the avatar jump",
        "Pin the avatar in place",
        "Move the avatar to position 100,200",
        "Hide the avatar",
    ]
    
    # Legacy simple parameters for backwards compatibility
    parameters = {
        "action": "Action: show, hide, jump, pin, unpin, move, resize, orientation",
        "value": "Value for action (coords for move, size for resize, direction for orientation)",
    }
    
    def execute(self, action: str, value: str = "", **kwargs) -> dict[str, Any]:
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
    """Customize the avatar's visual appearance."""
    
    name = "customize_avatar"
    description = "Customize the avatar's colors, lighting, rotation speed, and visual effects"
    category = "avatar"
    
    rich_parameters = [
        RichParameter(
            name="setting",
            type="string",
            description="Setting to change",
            required=True,
            enum=["primary_color", "secondary_color", "accent_color", "light_intensity", 
                  "ambient_strength", "wireframe", "show_grid", "rotate_speed", "auto_rotate", "reset"],
        ),
        RichParameter(
            name="value",
            type="string",
            description="Value for setting. Colors: hex like '#ff0000'. Numbers: 0-100. Booleans: 'true'/'false'",
            required=True,
        ),
    ]
    
    examples = [
        "Change avatar primary color to red",
        "Turn on wireframe mode",
        "Set lighting to 80%",
        "Enable auto-rotation",
    ]
    
    # Legacy
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
    
    def execute(self, setting: str, value: str, **kwargs) -> dict[str, Any]:
        setting = setting.lower().strip()
        
        if setting not in self.VALID_SETTINGS:
            return {"success": False, "error": f"Invalid setting '{setting}'. Valid: {self.VALID_SETTINGS}"}
        
        return _send_avatar_command(f"customize_{setting}", str(value))


class AvatarGestureTool(Tool):
    """Make the avatar perform gestures/animations."""
    
    name = "avatar_gesture"
    description = "Make the avatar perform a gesture or animation like waving, nodding, or blinking"
    category = "avatar"
    
    rich_parameters = [
        RichParameter(
            name="gesture",
            type="string",
            description="Gesture to perform",
            required=True,
            enum=["wave", "nod", "shake", "blink", "speak", "look_at"],
        ),
        RichParameter(
            name="intensity",
            type="float",
            description="Gesture intensity multiplier",
            required=False,
            default=1.0,
            min_value=0.5,
            max_value=2.0,
        ),
    ]
    
    examples = [
        "Make the avatar wave",
        "Avatar nod gesture",
        "Make it blink",
    ]
    
    # Legacy
    parameters = {
        "gesture": "Gesture to perform: wave, nod, shake (head), blink, speak",
        "intensity": "Optional intensity 0.5-2.0 (default 1.0)",
    }
    
    VALID_GESTURES = ["wave", "nod", "shake", "blink", "speak", "look_at"]
    
    def execute(self, gesture: str, intensity: float = 1.0, **kwargs) -> dict[str, Any]:
        gesture = gesture.lower().strip()
        
        if gesture not in self.VALID_GESTURES:
            return {"success": False, "error": f"Invalid gesture '{gesture}'. Valid: {self.VALID_GESTURES}"}
        
        try:
            intensity = max(0.5, min(2.0, float(intensity)))
        except (ValueError, TypeError):
            intensity = 1.0
        
        return _send_avatar_command(gesture, str(intensity))


class AvatarEmotionTool(Tool):
    """Set the avatar's emotional expression."""
    
    name = "avatar_emotion"
    description = "Set the avatar's emotional expression/mood. Changes facial expression and body language."
    category = "avatar"
    
    rich_parameters = [
        RichParameter(
            name="emotion",
            type="string",
            description="Emotion to express",
            required=True,
            enum=["happy", "sad", "angry", "surprised", "neutral", "excited", 
                  "thinking", "confused", "love", "scared", "bored", "curious", 
                  "proud", "embarrassed"],
        ),
    ]
    
    examples = [
        "Set avatar emotion to happy",
        "Make the avatar look confused",
        "Show excitement",
    ]
    
    # Legacy
    parameters = {
        "emotion": "Emotion to express: happy, sad, angry, surprised, neutral, excited, thinking, confused, love, scared",
    }
    
    VALID_EMOTIONS = [
        "happy", "sad", "angry", "surprised", "neutral",
        "excited", "thinking", "confused", "love", "scared",
        "bored", "curious", "proud", "embarrassed"
    ]
    
    def execute(self, emotion: str, **kwargs) -> dict[str, Any]:
        emotion = emotion.lower().strip()
        
        if emotion not in self.VALID_EMOTIONS:
            return {"success": False, "error": f"Invalid emotion '{emotion}'. Valid: {self.VALID_EMOTIONS}"}
        
        return _send_avatar_command("emotion", emotion)


class AdjustIdleAnimationTool(Tool):
    """
    Adjust the avatar's idle animations - breathing, blinking, sway, and micro-expressions.
    AI can use this to show emotion through subtle animation changes.
    """
    
    name = "adjust_idle_animation"
    description = "Adjust avatar idle animations - breathing rate, sway, blinking. Use to show mood through subtle movement (calm = slow breathing, nervous = fast breathing)"
    category = "avatar"
    
    rich_parameters = [
        RichParameter(
            name="breath_rate",
            type="float",
            description="Breaths per second. 0.1=very calm, 0.2=normal, 0.4=excited/nervous",
            required=False,
            default=None,
            min_value=0.05,
            max_value=0.5,
        ),
        RichParameter(
            name="sway_enabled",
            type="bool",
            description="Enable subtle idle swaying motion",
            required=False,
            default=None,
        ),
        RichParameter(
            name="sway_amount",
            type="float",
            description="How much to sway. 0.001=subtle, 0.005=normal, 0.01=noticeable",
            required=False,
            default=None,
            min_value=0.001,
            max_value=0.02,
        ),
        RichParameter(
            name="blink_rate",
            type="float",
            description="Blinks per second. 0.03=slow/relaxed, 0.05=normal, 0.1=nervous",
            required=False,
            default=None,
            min_value=0.01,
            max_value=0.2,
        ),
        RichParameter(
            name="look_enabled",
            type="bool",
            description="Enable random look-around behavior",
            required=False,
            default=None,
        ),
        RichParameter(
            name="micro_expressions",
            type="bool",
            description="Enable subtle micro-expressions",
            required=False,
            default=None,
        ),
    ]
    
    examples = [
        "Set breathing to calm (0.1)",
        "Make avatar blink nervously (0.1)",
        "Enable idle swaying",
        "Show nervousness with fast breathing and blinking",
    ]
    
    # Legacy
    parameters = {
        "breath_rate": "Breaths per second (0.1=calm, 0.2=normal, 0.4=excited). Optional.",
        "sway_enabled": "Enable subtle idle sway (true/false). Optional.",
        "sway_amount": "How much to sway (0.001=subtle, 0.005=normal, 0.01=noticeable). Optional.",
        "blink_rate": "Blinks per second (0.03=slow, 0.05=normal, 0.1=nervous). Optional.",
        "look_enabled": "Enable random look-around (true/false). Optional.",
        "micro_expressions": "Enable subtle micro-expressions (true/false). Optional.",
    }
    
    def execute(
        self, 
        breath_rate: float = None,
        sway_enabled: bool = None,
        sway_amount: float = None,
        blink_rate: float = None,
        look_enabled: bool = None,
        micro_expressions: bool = None,
        **kwargs
    ) -> dict[str, Any]:
        try:
            from ..avatar.procedural_animation import get_procedural_animator
            
            animator = get_procedural_animator()
            config = animator.config
            changes = []
            
            # Apply changes
            if breath_rate is not None:
                breath_rate = max(0.05, min(0.5, float(breath_rate)))
                config.breath_rate = breath_rate
                changes.append(f"breath_rate={breath_rate}")
            
            if sway_enabled is not None:
                config.sway_enabled = bool(sway_enabled)
                changes.append(f"sway_enabled={sway_enabled}")
            
            if sway_amount is not None:
                sway_amount = max(0.001, min(0.02, float(sway_amount)))
                config.sway_amount = sway_amount
                changes.append(f"sway_amount={sway_amount}")
            
            if blink_rate is not None:
                blink_rate = max(0.01, min(0.2, float(blink_rate)))
                config.blink_rate = blink_rate
                changes.append(f"blink_rate={blink_rate}")
            
            if look_enabled is not None:
                config.look_enabled = bool(look_enabled)
                changes.append(f"look_enabled={look_enabled}")
            
            if micro_expressions is not None:
                config.micro_expression_rate = 0.02 if micro_expressions else 0.0
                changes.append(f"micro_expressions={micro_expressions}")
            
            # Re-apply config to animator (triggers controller recreation)
            animator.config = config
            
            if changes:
                return {
                    "success": True, 
                    "message": f"Adjusted idle animation: {', '.join(changes)}",
                    "current_settings": {
                        "breath_rate": config.breath_rate,
                        "sway_enabled": config.sway_enabled,
                        "sway_amount": config.sway_amount,
                        "blink_rate": config.blink_rate,
                        "look_enabled": config.look_enabled,
                        "micro_expression_rate": config.micro_expression_rate,
                    }
                }
            else:
                return {
                    "success": True,
                    "message": "No changes specified. Current settings returned.",
                    "current_settings": {
                        "breath_rate": config.breath_rate,
                        "sway_enabled": config.sway_enabled,
                        "sway_amount": config.sway_amount,
                        "blink_rate": config.blink_rate,
                        "look_enabled": config.look_enabled,
                        "micro_expression_rate": config.micro_expression_rate,
                    }
                }
                
        except ImportError:
            return {"success": False, "error": "Procedural animation system not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
