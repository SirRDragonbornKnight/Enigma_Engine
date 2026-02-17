"""
Training Data Generation

Tools for generating training data for Enigma AI models.

Modules:
- embodied: Training for direct control systems (avatar, game, robot, camera)
"""

from .embodied import (
    AvatarTrainingGenerator,
    GameTrainingGenerator,
    RobotTrainingGenerator,
    CameraTrainingGenerator,
)

__all__ = [
    "AvatarTrainingGenerator",
    "GameTrainingGenerator",
    "RobotTrainingGenerator",
    "CameraTrainingGenerator",
]
