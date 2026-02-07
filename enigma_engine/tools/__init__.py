# tools package
"""
Forge Tools - Capabilities for the AI

Available tools:
  - web_search: Search the internet
  - fetch_webpage: Get content from a URL
  - read_file: Read a file
  - write_file: Write to a file
  - list_directory: List files in a directory
  - move_file: Move/rename a file
  - delete_file: Delete a file
  - read_document: Read books, PDFs, etc.
  - extract_text: Extract text from any file
  - run_command: Execute shell commands
  - screenshot: Take a screenshot
  - get_system_info: Get system information

AI Tool Use System:
  - tool_definitions: Define what tools are available to the AI
  - tool_executor: Execute tool calls from AI output
  - Enables AI to use image generation, vision, avatar, voice, etc.

USAGE:
    from enigma_engine.tools import ToolRegistry, execute_tool
    
    # Quick execute
    result = execute_tool("web_search", query="python tutorials")
    
    # Or use registry
    tools = ToolRegistry()
    result = tools.execute("read_file", path="README.md")
    
    # AI tool use
    from enigma_engine.tools import ToolExecutor, get_available_tools_for_prompt
    
    # Get tool descriptions for AI
    tools_desc = get_available_tools_for_prompt()
    
    # Execute tool calls from AI output
    executor = ToolExecutor(module_manager=manager)
    result = executor.execute_tool("generate_image", {"prompt": "sunset"})
    
    # Async execution
    from enigma_engine.tools import AsyncToolExecutor
    async_executor = AsyncToolExecutor(tool_executor=executor)
    result = await async_executor.execute_tool("web_search", {"query": "AI"})
    
    # With caching
    from enigma_engine.tools import ToolCache
    cache = ToolCache()
    cached_result = cache.get("web_search", {"query": "AI"})
    
    # With rate limiting
    from enigma_engine.tools import RateLimiter
    limiter = RateLimiter()
    if limiter.is_allowed("web_search"):
        limiter.record_request("web_search")
"""

# New tool modules - available for direct import
from . import (
    automation_tools,
    browser_tools,
    communication_tools,
    data_tools,
    gaming_tools,
    iot_tools,
    knowledge_tools,
    media_tools,
    productivity_tools,
)

# Low-priority features
from .analytics import ToolAnalytics, UsageRecord

# High-priority features
from .async_executor import AsyncToolExecutor
from .cache import CACHEABLE_TOOLS, ToolCache
from .dependencies import (
    INSTALL_INSTRUCTIONS,
    TOOL_COMMANDS,
    TOOL_DEPENDENCIES,
    ToolDependencyChecker,
)
from .game_router import GameAIRouter, GameConfig, get_game_router

# Code style analysis
from .code_style_analyzer import CodeStyleAnalyzer, ProjectContextExtractor, get_style_analyzer

# Medium-priority features
from .history import ToolExecution, ToolExecutionHistory
from .parallel import ParallelToolExecutor
from .permissions import (
    CONFIRMATION_REQUIRED_TOOLS,
    DEFAULT_TOOL_PERMISSIONS,
    PermissionLevel,
    ToolPermissionManager,
    default_confirmation_callback,
)
from .plugins import ToolPluginLoader
from .rate_limiter import DEFAULT_RATE_LIMITS, RateLimiter

# Robot and Game control systems
from .robot_modes import RobotMode, RobotModeController, get_mode_controller
from .streaming import StreamingToolExecutor, StreamingToolResult, StreamState
from .tool_definitions import (
    ToolDefinition,
    ToolParameter,
    get_all_tools,
    get_available_tools_for_prompt,
    get_tool_definition,
    get_tools_by_category,
)
from .tool_executor import ToolExecutor, execute_tool_from_text

# Tool Manager for enabling/disabling tools
from .tool_manager import PRESETS, TOOL_CATEGORIES, ToolManager, get_tool_manager
from .tool_registry import Tool, ToolRegistry, execute_tool, get_registry
from .validation import ToolSchemaValidator
from .vision import (
    FindOnScreenTool,
    ScreenCapture,
    ScreenVision,
    ScreenVisionTool,
    get_screen_vision,
)

__all__ = [
    # Core
    "Tool",
    "ToolRegistry", 
    "get_registry",
    "execute_tool",
    # Tool Manager
    "ToolManager",
    "get_tool_manager",
    "PRESETS",
    "TOOL_CATEGORIES",
    # Vision
    "ScreenCapture",
    "ScreenVision",
    "get_screen_vision",
    "ScreenVisionTool",
    "FindOnScreenTool",
    # AI Tool Use
    "ToolDefinition",
    "ToolParameter",
    "ToolExecutor",
    "get_tool_definition",
    "get_all_tools",
    "get_tools_by_category",
    "get_available_tools_for_prompt",
    "execute_tool_from_text",
    # High-priority features
    "AsyncToolExecutor",
    "ToolCache",
    "CACHEABLE_TOOLS",
    "RateLimiter",
    "DEFAULT_RATE_LIMITS",
    # Medium-priority features
    "ToolExecution",
    "ToolExecutionHistory",
    "PermissionLevel",
    "ToolPermissionManager",
    "DEFAULT_TOOL_PERMISSIONS",
    "CONFIRMATION_REQUIRED_TOOLS",
    "default_confirmation_callback",
    "ToolDependencyChecker",
    "TOOL_DEPENDENCIES",
    "TOOL_COMMANDS",
    "INSTALL_INSTRUCTIONS",
    "ParallelToolExecutor",
    # Low-priority features
    "ToolAnalytics",
    "UsageRecord",
    "ToolSchemaValidator",
    "ToolPluginLoader",
    "StreamingToolResult",
    "StreamingToolExecutor",
    "StreamState",
    # Robot and Game control
    "RobotModeController",
    "RobotMode",
    "get_mode_controller",
    "GameAIRouter",
    "GameConfig",
    "get_game_router",
    # Code style
    "CodeStyleAnalyzer",
    "ProjectContextExtractor",
    "get_style_analyzer",
    # New tool modules
    "automation_tools",
    "knowledge_tools",
    "communication_tools",
    "media_tools",
    "productivity_tools",
    "iot_tools",
    "data_tools",
    "gaming_tools",
    "browser_tools",
]
