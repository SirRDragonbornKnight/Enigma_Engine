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
from .terminal_tab import create_terminal_tab, log_to_terminal
from .modules_tab import ModulesTab
from .scaling_tab import ScalingTab, create_scaling_tab
from .examples_tab import ExamplesTab, create_examples_tab
from .image_tab import ImageTab, create_image_tab
from .code_tab import CodeTab, create_code_tab
from .video_tab import VideoTab, create_video_tab
from .audio_tab import AudioTab, create_audio_tab
from .embeddings_tab import EmbeddingsTab, create_embeddings_tab

__all__ = [
    'create_chat_tab',
    'create_training_tab', 
    'create_avatar_tab',
    'create_vision_tab',
    'create_sessions_tab',
    'create_instructions_tab',
    'create_terminal_tab',
    'log_to_terminal',
    'ModulesTab',
    'ScalingTab',
    'create_scaling_tab',
    'ExamplesTab',
    'create_examples_tab',
    'ImageTab',
    'create_image_tab',
    'CodeTab',
    'create_code_tab',
    'VideoTab',
    'create_video_tab',
    'AudioTab',
    'create_audio_tab',
    'EmbeddingsTab',
    'create_embeddings_tab',
]
