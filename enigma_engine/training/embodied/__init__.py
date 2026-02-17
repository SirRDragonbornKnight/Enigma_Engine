"""
Embodied AI Training Data Generation

Generates training data for AI to directly control various systems
by outputting continuous values instead of discrete tool calls.

Systems:
- Avatar: Bone angles, poses, expressions
- Game: Controller inputs, keyboard/mouse
- Robot: Motor values, sensor responses
- Camera: PTZ controls, attention

Each generator produces text pairs in the format:
    Input: [context + user request]
    Output: [AI response with embedded control values]
"""

from .avatar_training import AvatarTrainingGenerator
from .game_training import GameTrainingGenerator
from .robot_training import RobotTrainingGenerator
from .camera_training import CameraTrainingGenerator

__all__ = [
    "AvatarTrainingGenerator",
    "GameTrainingGenerator", 
    "RobotTrainingGenerator",
    "CameraTrainingGenerator",
]
