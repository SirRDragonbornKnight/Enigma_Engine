"""
AI Avatar Control System

Allows Enigma AI Engine models to control the avatar through natural language.
Parses bone control commands and executes them with proper priority.

Usage:
    from enigma_engine.avatar.ai_control import AIAvatarControl, parse_bone_commands
    
    # Initialize
    ai_control = AIAvatarControl()
    
    # AI generates response with bone commands
    response = "I'll nod <bone_control>head|pitch=15,yaw=0,roll=0</bone_control>"
    
    # Parse and execute
    ai_control.process_response(response)
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class BoneCommand:
    """Represents a single bone control command."""
    
    def __init__(self, bone_name: str, pitch: float = None, yaw: float = None, roll: float = None):
        self.bone_name = bone_name
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
    
    def __repr__(self):
        return f"BoneCommand({self.bone_name}, pitch={self.pitch}, yaw={self.yaw}, roll={self.roll})"


def parse_bone_commands(text: str) -> tuple[str, list[BoneCommand]]:
    """Parse bone control commands from AI response.
    
    Format: <bone_control>bone_name|pitch=value,yaw=value,roll=value</bone_control>
    
    Args:
        text: AI response text with bone commands
    
    Returns:
        Tuple of (clean_text, list_of_bone_commands)
    """
    commands = []
    clean_text = text
    
    # Find all bone control tags
    pattern = r'<bone_control>(.*?)</bone_control>'
    matches = re.finditer(pattern, text)
    
    for match in matches:
        command_str = match.group(1).strip()
        
        try:
            # Parse: bone_name|pitch=15,yaw=0,roll=0
            if '|' not in command_str:
                logger.warning(f"Invalid bone command format: {command_str}")
                continue
            
            bone_name, params_str = command_str.split('|', 1)
            bone_name = bone_name.strip()
            
            # Parse parameters
            pitch, yaw, roll = None, None, None
            for param in params_str.split(','):
                param = param.strip()
                if '=' not in param:
                    continue
                
                key, value = param.split('=', 1)
                key = key.strip().lower()
                
                try:
                    value = float(value.strip())
                    if key == 'pitch':
                        pitch = value
                    elif key == 'yaw':
                        yaw = value
                    elif key == 'roll':
                        roll = value
                except ValueError:
                    logger.warning(f"Invalid numeric value in bone command: {param}")
                    continue
            
            commands.append(BoneCommand(bone_name, pitch, yaw, roll))
        
        except Exception as e:
            logger.error(f"Error parsing bone command '{command_str}': {e}")
    
    # Remove bone control tags from text
    clean_text = re.sub(pattern, '', clean_text).strip()
    
    return clean_text, commands


class AIAvatarControl:
    """AI control interface for avatar bone animation."""
    
    def __init__(self):
        self._bone_controller = None
        self._avatar_controller = None
        self._enabled = False
        
        # Initialize controllers
        self._init_controllers()
    
    def _init_controllers(self):
        """Initialize bone and avatar controllers."""
        try:
            from . import get_avatar
            from .bone_control import get_bone_controller
            
            self._avatar_controller = get_avatar()
            self._bone_controller = get_bone_controller(avatar_controller=self._avatar_controller)
            self._enabled = True
            
            logger.info("AI avatar control initialized")
        except Exception as e:
            logger.warning(f"Could not initialize AI avatar control: {e}")
            self._enabled = False
    
    def process_response(self, text: str) -> str:
        """Process AI response and execute bone commands.
        
        Args:
            text: AI response with potential bone control tags
        
        Returns:
            Clean text with bone control tags removed
        """
        if not self._enabled:
            return text
        
        clean_text, commands = parse_bone_commands(text)
        
        if commands:
            logger.info(f"Executing {len(commands)} bone commands from AI")
            self.execute_commands(commands)
        
        return clean_text
    
    def execute_commands(self, commands: list[BoneCommand], delay: float = 0.1):
        """Execute a sequence of bone commands.
        
        Args:
            commands: List of BoneCommand objects
            delay: Delay between commands (seconds)
        """
        if not self._enabled or not self._bone_controller:
            logger.warning("Bone controller not available")
            return
        
        import time
        
        for cmd in commands:
            try:
                result = self._bone_controller.move_bone(
                    cmd.bone_name,
                    pitch=cmd.pitch,
                    yaw=cmd.yaw,
                    roll=cmd.roll,
                    smooth=True
                )
                logger.debug(f"Moved {cmd.bone_name} to pitch={result[0]:.1f}, yaw={result[1]:.1f}, roll={result[2]:.1f}")
                
                if delay > 0:
                    time.sleep(delay)
            
            except Exception as e:
                logger.error(f"Error executing bone command {cmd}: {e}")
    
    def execute_gesture(self, gesture_name: str) -> bool:
        """Execute a predefined gesture.
        
        Args:
            gesture_name: Name of gesture (wave, nod, shrug, etc.)
        
        Returns:
            True if gesture executed, False otherwise
        """
        gestures = {
            'nod': [
                BoneCommand('head', pitch=15, yaw=0, roll=0),
                BoneCommand('head', pitch=0, yaw=0, roll=0),
            ],
            'shake': [
                BoneCommand('head', pitch=0, yaw=-20, roll=0),
                BoneCommand('head', pitch=0, yaw=20, roll=0),
                BoneCommand('head', pitch=0, yaw=0, roll=0),
            ],
            'wave': [
                BoneCommand('right_upper_arm', pitch=90, yaw=0, roll=-45),
                BoneCommand('right_forearm', pitch=90, yaw=0, roll=0),
            ],
            'shrug': [
                BoneCommand('left_shoulder', pitch=20, yaw=0, roll=0),
                BoneCommand('right_shoulder', pitch=20, yaw=0, roll=0),
            ],
            'point': [
                BoneCommand('right_upper_arm', pitch=90, yaw=0, roll=0),
                BoneCommand('right_forearm', pitch=0, yaw=0, roll=0),
            ],
            'thinking': [
                BoneCommand('head', pitch=-10, yaw=15, roll=5),
                BoneCommand('right_upper_arm', pitch=90, yaw=30, roll=0),
                BoneCommand('right_forearm', pitch=120, yaw=0, roll=0),
            ],
        }
        
        if gesture_name.lower() not in gestures:
            logger.warning(f"Unknown gesture: {gesture_name}")
            return False
        
        commands = gestures[gesture_name.lower()]
        self.execute_commands(commands, delay=0.2)
        return True
    
    def reset_pose(self):
        """Reset avatar to neutral pose."""
        commands = [
            BoneCommand('head', pitch=0, yaw=0, roll=0),
            BoneCommand('neck', pitch=0, yaw=0, roll=0),
            BoneCommand('spine', pitch=0, yaw=0, roll=0),
            BoneCommand('chest', pitch=0, yaw=0, roll=0),
            BoneCommand('left_upper_arm', pitch=0, yaw=0, roll=0),
            BoneCommand('right_upper_arm', pitch=0, yaw=0, roll=0),
            BoneCommand('left_forearm', pitch=10, yaw=0, roll=0),
            BoneCommand('right_forearm', pitch=10, yaw=0, roll=0),
        ]
        self.execute_commands(commands, delay=0.05)


# Global instance
_ai_avatar_control: Optional[AIAvatarControl] = None


def get_ai_avatar_control() -> AIAvatarControl:
    """Get the global AI avatar control instance."""
    global _ai_avatar_control
    if _ai_avatar_control is None:
        _ai_avatar_control = AIAvatarControl()
    return _ai_avatar_control


def process_ai_response(text: str) -> str:
    """Convenience function to process AI response.
    
    Args:
        text: AI response text
    
    Returns:
        Clean text with bone commands executed
    """
    return get_ai_avatar_control().process_response(text)
