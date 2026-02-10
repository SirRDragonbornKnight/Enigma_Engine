"""
GUI Tabs for Enigma AI Engine
Each tab is in its own module for better organization.

Uses lazy loading to improve startup time - tabs are only imported when accessed.
"""

from typing import TYPE_CHECKING

# Only import for type checking (not at runtime)
if TYPE_CHECKING:
    from .analytics_tab import AnalyticsTab, create_analytics_tab
    from .audio_tab import AudioTab, create_audio_tab
    from .avatar.avatar_display import create_avatar_subtab
    from .avatar_tab import create_avatar_tab
    from .camera_tab import CameraTab, create_camera_tab
    from .chat_tab import create_chat_tab
    from .code_tab import CodeTab, create_code_tab
    from .embeddings_tab import EmbeddingsTab, create_embeddings_tab
    from .examples_tab import ExamplesTab, create_examples_tab
    from .game.game_connection import create_game_subtab
    from .gif_tab import GIFTab, create_gif_tab
    from .image_tab import ImageTab, create_image_tab
    from .instructions_tab import create_instructions_tab
    from .logs_tab import LogsTab, create_logs_tab
    from .model_router_tab import ModelRouterTab
    from .modules_tab import ModulesTab
    from .network_tab import NetworkTab, create_network_tab
    from .notes_tab import NotesTab, create_notes_tab
    from .robot.robot_control import create_robot_subtab
    from .scaling_tab import ScalingTab, create_scaling_tab
    from .scheduler_tab import SchedulerTab, create_scheduler_tab
    from .sessions_tab import create_sessions_tab
    from .settings_tab import create_settings_tab
    from .shared_components import (
        COLOR_PRESETS,
        STYLE_PRESETS,
        ColorCustomizer,
        DirectoryWatcher,
        ModuleStateChecker,
        PresetSelector,
        SettingsPersistence,
        create_action_button,
        create_settings_group,
    )
    from .terminal_tab import create_terminal_tab, log_to_terminal
    from .threed_tab import ThreeDTab, create_threed_tab
    from .tool_manager_tab import ToolManagerTab
    from .training_tab import create_training_tab
    from .video_tab import VideoTab, create_video_tab
    from .vision_tab import create_vision_tab
    from .voice_clone_tab import VoiceCloneTab
    from .workspace_tab import create_workspace_tab

# Lazy loading cache
_cache = {}

def __getattr__(name: str):
    """Lazy load tabs only when accessed."""
    if name in _cache:
        return _cache[name]
    
    # Map names to their modules
    _imports = {
        # Chat and core tabs
        'create_chat_tab': ('.chat_tab', 'create_chat_tab'),
        'create_training_tab': ('.training_tab', 'create_training_tab'),
        'create_workspace_tab': ('.workspace_tab', 'create_workspace_tab'),
        'create_dashboard_tab': ('.dashboard_tab', 'create_dashboard_tab'),
        'create_avatar_tab': ('.avatar_tab', 'create_avatar_tab'),
        'create_avatar_subtab': ('.avatar.avatar_display', 'create_avatar_subtab'),
        'create_game_subtab': ('.game.game_connection', 'create_game_subtab'),
        'create_robot_subtab': ('.robot.robot_control', 'create_robot_subtab'),
        'create_vision_tab': ('.vision_tab', 'create_vision_tab'),
        'CameraTab': ('.camera_tab', 'CameraTab'),
        'create_camera_tab': ('.camera_tab', 'create_camera_tab'),
        'create_sessions_tab': ('.sessions_tab', 'create_sessions_tab'),
        'create_instructions_tab': ('.instructions_tab', 'create_instructions_tab'),
        'create_terminal_tab': ('.terminal_tab', 'create_terminal_tab'),
        'log_to_terminal': ('.terminal_tab', 'log_to_terminal'),
        'create_settings_tab': ('.settings_tab', 'create_settings_tab'),
        
        # Module/system tabs
        'ModulesTab': ('.modules_tab', 'ModulesTab'),
        'ScalingTab': ('.scaling_tab', 'ScalingTab'),
        'create_scaling_tab': ('.scaling_tab', 'create_scaling_tab'),
        'ExamplesTab': ('.examples_tab', 'ExamplesTab'),
        'create_examples_tab': ('.examples_tab', 'create_examples_tab'),
        'ToolManagerTab': ('.tool_manager_tab', 'ToolManagerTab'),
        'ModelRouterTab': ('.model_router_tab', 'ModelRouterTab'),
        'LearningTab': ('.learning_tab', 'LearningTab'),
        'create_learning_tab': ('.learning_tab', 'create_learning_tab'),
        'BuildAITab': ('.build_ai_tab', 'BuildAITab'),
        'create_build_ai_tab': ('.build_ai_tab', 'create_build_ai_tab'),
        
        # Generation tabs
        'ImageTab': ('.image_tab', 'ImageTab'),
        'create_image_tab': ('.image_tab', 'create_image_tab'),
        'CodeTab': ('.code_tab', 'CodeTab'),
        'create_code_tab': ('.code_tab', 'create_code_tab'),
        'VideoTab': ('.video_tab', 'VideoTab'),
        'create_video_tab': ('.video_tab', 'create_video_tab'),
        'AudioTab': ('.audio_tab', 'AudioTab'),
        'create_audio_tab': ('.audio_tab', 'create_audio_tab'),
        'EmbeddingsTab': ('.embeddings_tab', 'EmbeddingsTab'),
        'create_embeddings_tab': ('.embeddings_tab', 'create_embeddings_tab'),
        'ThreeDTab': ('.threed_tab', 'ThreeDTab'),
        'create_threed_tab': ('.threed_tab', 'create_threed_tab'),
        'GIFTab': ('.gif_tab', 'GIFTab'),
        'create_gif_tab': ('.gif_tab', 'create_gif_tab'),
        'VoiceCloneTab': ('.voice_clone_tab', 'VoiceCloneTab'),
        
        # Utility tabs
        'LogsTab': ('.logs_tab', 'LogsTab'),
        'create_logs_tab': ('.logs_tab', 'create_logs_tab'),
        'NotesTab': ('.notes_tab', 'NotesTab'),
        'create_notes_tab': ('.notes_tab', 'create_notes_tab'),
        'NetworkTab': ('.network_tab', 'NetworkTab'),
        'create_network_tab': ('.network_tab', 'create_network_tab'),
        'FederationTab': ('.federation_tab', 'FederationTab'),
        'create_federation_tab': ('.federation_tab', 'create_federation_tab'),
        'AnalyticsTab': ('.analytics_tab', 'AnalyticsTab'),
        'create_analytics_tab': ('.analytics_tab', 'create_analytics_tab'),
        'SchedulerTab': ('.scheduler_tab', 'SchedulerTab'),
        'create_scheduler_tab': ('.scheduler_tab', 'create_scheduler_tab'),
        'DevicesTab': ('.devices_tab', 'DevicesTab'),
        'create_devices_tab': ('.devices_tab', 'create_devices_tab'),
        'create_model_comparison_tab': ('.model_comparison_tab', 'create_model_comparison_tab'),
        'TrainingDataTab': ('.training_data_tab', 'TrainingDataTab'),
        'create_training_data_tab': ('.training_data_tab', 'create_training_data_tab'),
        'BundleManagerTab': ('.bundle_manager_tab', 'BundleManagerTab'),
        'create_bundle_manager_tab': ('.bundle_manager_tab', 'create_bundle_manager_tab'),
        
        # Shared components
        'STYLE_PRESETS': ('.shared_components', 'STYLE_PRESETS'),
        'COLOR_PRESETS': ('.shared_components', 'COLOR_PRESETS'),
        'PresetSelector': ('.shared_components', 'PresetSelector'),
        'ColorCustomizer': ('.shared_components', 'ColorCustomizer'),
        'ModuleStateChecker': ('.shared_components', 'ModuleStateChecker'),
        'SettingsPersistence': ('.shared_components', 'SettingsPersistence'),
        'create_settings_group': ('.shared_components', 'create_settings_group'),
        'create_action_button': ('.shared_components', 'create_action_button'),
        'DirectoryWatcher': ('.shared_components', 'DirectoryWatcher'),
        'NoScrollComboBox': ('.shared_components', 'NoScrollComboBox'),
        'disable_scroll_on_combos': ('.shared_components', 'disable_scroll_on_combos'),
        
        # Base generation tab
        'BaseGenerationTab': ('.base_generation_tab', 'BaseGenerationTab'),
        'BaseGenerationWorker': ('.base_generation_tab', 'BaseGenerationWorker'),
        'BUTTON_STYLE_PRIMARY': ('.base_generation_tab', 'BUTTON_STYLE_PRIMARY'),
        'BUTTON_STYLE_SECONDARY': ('.base_generation_tab', 'BUTTON_STYLE_SECONDARY'),
        'BUTTON_STYLE_SUCCESS': ('.base_generation_tab', 'BUTTON_STYLE_SUCCESS'),
        'BUTTON_STYLE_DANGER': ('.base_generation_tab', 'BUTTON_STYLE_DANGER'),
        'create_group_box': ('.base_generation_tab', 'create_group_box'),
        
        # Provider base
        'GenerationProvider': ('.provider_base', 'GenerationProvider'),
        'ProviderRegistry': ('.provider_base', 'ProviderRegistry'),
        'get_provider_registry': ('.provider_base', 'get_provider_registry'),
        
        # Unified patterns
        'StyleConfig': ('.unified_patterns', 'StyleConfig'),
        'get_style_config': ('.unified_patterns', 'get_style_config'),
        'Colors': ('.unified_patterns', 'Colors'),
        'DeviceUIClass': ('.unified_patterns', 'DeviceUIClass'),
        'get_button_style': ('.unified_patterns', 'get_button_style'),
        'get_header_style': ('.unified_patterns', 'get_header_style'),
        'get_group_style': ('.unified_patterns', 'get_group_style'),
        'UnifiedWorker': ('.unified_patterns', 'UnifiedWorker'),
        'create_styled_button': ('.unified_patterns', 'create_styled_button'),
        'create_styled_group': ('.unified_patterns', 'create_styled_group'),
        
        # Consolidated tabs (for Standard mode)
        'create_create_tab': ('.create_tab', 'create_create_tab'),
        'create_ai_tab': ('.ai_tab', 'create_ai_tab'),
    }
    
    if name not in _imports:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    
    module_path, attr_name = _imports[name]
    
    # Import the module
    import importlib
    module = importlib.import_module(module_path, __package__)
    value = getattr(module, attr_name)
    
    # Cache it
    _cache[name] = value
    return value


__all__ = [
    'create_chat_tab',
    'create_training_tab',
    'create_workspace_tab',
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
    'FederationTab',
    'create_federation_tab',
    'AnalyticsTab',
    'create_analytics_tab',
    'SchedulerTab',
    'create_scheduler_tab',
    'create_model_comparison_tab',
    'TrainingDataTab',
    'create_training_data_tab',
    'BundleManagerTab',
    'create_bundle_manager_tab',
    'GIFTab',
    'create_gif_tab',
    'create_settings_tab',
    'ModelRouterTab',
    'VoiceCloneTab',
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
    'NoScrollComboBox',
    'disable_scroll_on_combos',
    # Base generation tab
    'BaseGenerationTab',
    'BaseGenerationWorker',
    'BUTTON_STYLE_PRIMARY',
    'BUTTON_STYLE_SECONDARY',
    'BUTTON_STYLE_SUCCESS',
    'BUTTON_STYLE_DANGER',
    'create_group_box',
    # Provider base
    'GenerationProvider',
    'ProviderRegistry',
    'get_provider_registry',
    # Unified patterns
    'StyleConfig',
    'get_style_config',
    'Colors',
    'DeviceUIClass',
    'get_button_style',
    'get_header_style',
    'get_group_style',
    'UnifiedWorker',
    'create_styled_button',
    'create_styled_group',
    # Consolidated tabs
    'create_create_tab',
    'create_ai_tab',
]
