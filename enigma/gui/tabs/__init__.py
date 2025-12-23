"""
GUI Tabs for Enigma Engine
Each tab is in its own module for better organization.
"""

from .chat_tab import create_chat_tab
from .training_tab import create_training_tab
from .avatar_tab import create_avatar_tab
from .vision_tab import create_vision_tab
from .sessions_tab import create_sessions_tab
from .instructions_tab import create_instructions_tab

__all__ = [
    'create_chat_tab',
    'create_training_tab', 
    'create_avatar_tab',
    'create_vision_tab',
    'create_sessions_tab',
    'create_instructions_tab',
]
