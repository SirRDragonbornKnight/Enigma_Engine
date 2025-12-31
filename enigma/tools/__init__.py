# tools package
"""
Enigma Tools - Capabilities for the AI

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
    from enigma.tools import ToolRegistry, execute_tool
    
    # Quick execute
    result = execute_tool("web_search", query="python tutorials")
    
    # Or use registry
    tools = ToolRegistry()
    result = tools.execute("read_file", path="README.md")
    
    # AI tool use
    from enigma.tools import ToolExecutor, get_available_tools_for_prompt
    
    # Get tool descriptions for AI
    tools_desc = get_available_tools_for_prompt()
    
    # Execute tool calls from AI output
    executor = ToolExecutor(module_manager=manager)
    result = executor.execute_tool("generate_image", {"prompt": "sunset"})
"""

from .tool_registry import Tool, ToolRegistry, get_registry, execute_tool
from .vision import ScreenCapture, ScreenVision, get_screen_vision, ScreenVisionTool, FindOnScreenTool
from .tool_definitions import (
    ToolDefinition, 
    ToolParameter,
    get_tool_definition,
    get_all_tools,
    get_tools_by_category,
    get_available_tools_for_prompt,
)
from .tool_executor import ToolExecutor, execute_tool_from_text

__all__ = [
    "Tool",
    "ToolRegistry", 
    "get_registry",
    "execute_tool",
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
]
