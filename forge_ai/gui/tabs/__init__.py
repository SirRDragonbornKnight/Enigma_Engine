"""
GUI Tabs for ForgeAI
Each tab is in its own module for better organization.
"""

from .chat_tab import create_chat_tab
from .training_tab import create_training_tab
from .avatar_tab import create_avatar_tab
from .avatar.avatar_display import create_avatar_subtab
from .game.game_connection import create_game_subtab
from .robot.robot_control import create_robot_subtab
from .vision_tab import create_vision_tab
from .camera_tab import CameraTab, create_camera_tab
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
from .threed_tab import ThreeDTab, create_threed_tab
from .tool_manager_tab import ToolManagerTab
from .logs_tab import LogsTab, create_logs_tab
from .notes_tab import NotesTab, create_notes_tab
from .network_tab import NetworkTab, create_network_tab
from .analytics_tab import AnalyticsTab, create_analytics_tab
from .scheduler_tab import SchedulerTab, create_scheduler_tab
from .gif_tab import GIFTab, create_gif_tab
from .settings_tab import create_settings_tab
from .model_router_tab import ModelRouterTab

# Shared UI components for use across all tabs
from .shared_components import (
    STYLE_PRESETS,
    COLOR_PRESETS,
    PresetSelector,
    ColorCustomizer,
    ModuleStateChecker,
    SettingsPersistence,
    create_settings_group,
    create_action_button,
    DirectoryWatcher,
)

__all__ = [
    'create_chat_tab',
    'create_training_tab', 
    'create_avatar_tab',
    'create_avatar_subtab',
    'create_game_subtab',
    'create_robot_subtab',
    'create_vision_tab',
    'CameraTab',
    'create_camera_tab',
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
    'ThreeDTab',
    'create_threed_tab',
    'ToolManagerTab',
    'LogsTab',
    'create_logs_tab',
    'NotesTab',
    'create_notes_tab',
    'NetworkTab',
    'create_network_tab',
    'AnalyticsTab',
    'create_analytics_tab',
    'SchedulerTab',
    'create_scheduler_tab',
    'GIFTab',
    'create_gif_tab',
    'create_settings_tab',
    'ModelRouterTab',
    # Shared components
    'STYLE_PRESETS',
    'COLOR_PRESETS',
    'PresetSelector',
    'ColorCustomizer',
    'ModuleStateChecker',
    'SettingsPersistence',
    'create_settings_group',
    'create_action_button',
    'DirectoryWatcher',
]
