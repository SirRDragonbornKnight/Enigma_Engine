"""
AI Tool Integration Interface
============================

Allows the AI model to invoke tools like image generation, avatar control,
vision, voice, web search, and file operations.

This bridges the gap between "tools exist" and "AI can use them".

Features:
  - Parse AI output to detect tool invocations
  - Execute tools via module manager and addon system
  - Format results for AI understanding
  - Support for all major tool types

Usage:
    from enigma.core.tool_interface import ToolInterface
    from enigma.modules import ModuleManager
    
    manager = ModuleManager()
    tool_interface = ToolInterface(manager)
    
    # Parse AI output
    ai_output = '<|tool_call|>generate_image("a sunset")<|tool_end|>'
    tool_call = tool_interface.parse_tool_call(ai_output)
    
    # Execute tool
    if tool_call:
        result = tool_interface.execute_tool(tool_call)
        formatted = tool_interface.format_tool_result(result)
"""

import re
import json
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a parsed tool invocation from AI output."""
    tool_name: str
    arguments: Dict[str, Any]
    raw_text: str
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class ToolResult:
    """Represents the result of tool execution."""
    success: bool
    tool_name: str
    data: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    message: Optional[str] = None


class ToolInterface:
    """
    Interface for AI to invoke tools.
    
    Handles parsing tool calls from AI output, executing them,
    and formatting results for the AI to understand.
    """
    
    def __init__(self, module_manager=None):
        """
        Initialize tool interface.
        
        Args:
            module_manager: ModuleManager instance for accessing loaded modules
        """
        self.manager = module_manager
        self.available_tools: Dict[str, Callable] = {}
        self.tool_descriptions: Dict[str, str] = {}
        self._register_tools()
    
    def _register_tools(self):
        """Register all available tools the AI can use."""
        # Image generation
        self.available_tools['generate_image'] = self._generate_image
        self.tool_descriptions['generate_image'] = (
            "Generate an image from a text description. "
            "Args: prompt (str), width (int, optional), height (int, optional)"
        )
        
        # Avatar control
        self.available_tools['avatar_action'] = self._avatar_action
        self.tool_descriptions['avatar_action'] = (
            "Control the avatar. "
            "Args: action (str: 'speak', 'set_expression', 'animate'), params (dict)"
        )
        
        # Vision/screen capture
        self.available_tools['capture_screen'] = self._capture_screen
        self.tool_descriptions['capture_screen'] = (
            "Capture and analyze what's on the screen. "
            "Args: None"
        )
        
        # Voice output
        self.available_tools['speak'] = self._speak
        self.tool_descriptions['speak'] = (
            "Speak text out loud using text-to-speech. "
            "Args: text (str)"
        )
        
        # Web search
        self.available_tools['search_web'] = self._search_web
        self.tool_descriptions['search_web'] = (
            "Search the web for information. "
            "Args: query (str)"
        )
        
        # File operations
        self.available_tools['read_file'] = self._read_file
        self.tool_descriptions['read_file'] = (
            "Read contents of a file. "
            "Args: path (str)"
        )
        
        self.available_tools['write_file'] = self._write_file
        self.tool_descriptions['write_file'] = (
            "Write content to a file. "
            "Args: path (str), content (str)"
        )
        
        self.available_tools['list_directory'] = self._list_directory
        self.tool_descriptions['list_directory'] = (
            "List files in a directory. "
            "Args: path (str)"
        )
    
    def parse_tool_call(self, ai_output: str) -> Optional[ToolCall]:
        """
        Parse AI output to detect tool invocations.
        
        Looks for patterns like:
          <|tool_call|>generate_image("a sunset")<|tool_end|>
          <|tool_call|>avatar_action("set_expression", {"expression": "happy"})<|tool_end|>
        
        Args:
            ai_output: Raw AI output text
            
        Returns:
            ToolCall object if found, None otherwise
        """
        # Pattern to match tool calls
        pattern = r'<\|tool_call\|>(.*?)<\|tool_end\|>'
        match = re.search(pattern, ai_output)
        
        if not match:
            return None
        
        tool_text = match.group(1).strip()
        start_pos = match.start()
        end_pos = match.end()
        
        # Parse function-style call: function_name(args)
        func_pattern = r'(\w+)\((.*?)\)$'
        func_match = re.match(func_pattern, tool_text, re.DOTALL)
        
        if not func_match:
            logger.warning(f"Could not parse tool call: {tool_text}")
            return None
        
        tool_name = func_match.group(1)
        args_str = func_match.group(2).strip()
        
        # Parse arguments
        arguments = self._parse_arguments(args_str)
        
        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            raw_text=tool_text,
            start_pos=start_pos,
            end_pos=end_pos
        )
    
    def _parse_arguments(self, args_str: str) -> Dict[str, Any]:
        """
        Parse function arguments from string.
        
        Handles:
          - Simple strings: "hello"
          - Numbers: 512, 3.14
          - JSON objects: {"key": "value"}
          - Multiple args: "text", 512, {"key": "value"}
        """
        if not args_str:
            return {}
        
        arguments = {}
        
        # Try to parse as JSON first
        try:
            # Wrap in array if multiple args
            if args_str.startswith('{'):
                # Single dict argument
                parsed = json.loads(args_str)
                if isinstance(parsed, dict):
                    return parsed
            else:
                # Try as array of arguments
                parsed = json.loads(f"[{args_str}]")
                if isinstance(parsed, list):
                    # Map to generic arg names
                    for i, val in enumerate(parsed):
                        arguments[f'arg{i}'] = val
                    # Also set first arg as common names
                    if len(parsed) > 0:
                        arguments['prompt'] = parsed[0]
                        arguments['text'] = parsed[0]
                        arguments['query'] = parsed[0]
                        arguments['path'] = parsed[0]
                    if len(parsed) > 1:
                        arguments['action'] = parsed[1]
                        arguments['params'] = parsed[1]
                    return arguments
        except json.JSONDecodeError:
            pass
        
        # Fallback: parse as simple comma-separated values
        parts = [p.strip() for p in args_str.split(',')]
        for i, part in enumerate(parts):
            # Remove quotes if present
            part = part.strip('"').strip("'")
            arguments[f'arg{i}'] = part
        
        # Set common argument names
        if len(parts) > 0:
            arguments['prompt'] = parts[0].strip('"').strip("'")
            arguments['text'] = parts[0].strip('"').strip("'")
            arguments['query'] = parts[0].strip('"').strip("'")
            arguments['path'] = parts[0].strip('"').strip("'")
        
        return arguments
    
    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool and return the result.
        
        Args:
            tool_call: Parsed tool call
            
        Returns:
            ToolResult with execution outcome
        """
        tool_func = self.available_tools.get(tool_call.tool_name)
        
        if not tool_func:
            return ToolResult(
                success=False,
                tool_name=tool_call.tool_name,
                error=f"Unknown tool: {tool_call.tool_name}"
            )
        
        try:
            import time
            start = time.time()
            result = tool_func(**tool_call.arguments)
            duration = time.time() - start
            
            # Normalize result format
            if isinstance(result, dict):
                return ToolResult(
                    success=result.get('success', True),
                    tool_name=tool_call.tool_name,
                    data=result.get('result') or result.get('data'),
                    error=result.get('error'),
                    message=result.get('message'),
                    duration=duration
                )
            else:
                return ToolResult(
                    success=True,
                    tool_name=tool_call.tool_name,
                    data=result,
                    duration=duration
                )
        except Exception as e:
            logger.error(f"Error executing {tool_call.tool_name}: {e}", exc_info=True)
            return ToolResult(
                success=False,
                tool_name=tool_call.tool_name,
                error=str(e)
            )
    
    def format_tool_result(self, result: ToolResult) -> str:
        """
        Format tool result for the AI to understand.
        
        Args:
            result: Tool execution result
            
        Returns:
            Formatted string with special tokens
        """
        if result.success:
            msg = result.message or f"{result.tool_name} completed successfully"
            if result.data:
                if isinstance(result.data, (str, int, float, bool)):
                    msg += f": {result.data}"
                elif isinstance(result.data, dict):
                    msg += f": {json.dumps(result.data, indent=2)}"
                elif isinstance(result.data, list) and len(result.data) < 10:
                    msg += f": {result.data}"
            return f"<|tool_result|>{msg}<|tool_result_end|>"
        else:
            error_msg = result.error or "Unknown error"
            return f"<|tool_result|>Error: {error_msg}<|tool_result_end|>"
    
    # =========================================================================
    # Tool Implementations
    # =========================================================================
    
    def _generate_image(self, prompt: str = None, width: int = 512, height: int = 512, **kwargs) -> Dict[str, Any]:
        """Generate an image using available image generation addon."""
        if prompt is None:
            prompt = kwargs.get('arg0', kwargs.get('text', ''))
        
        if not prompt:
            return {"success": False, "error": "No prompt provided"}
        
        # Try to use loaded image generation module
        if self.manager:
            try:
                # Check for local image gen
                img_gen = self.manager.get_module('image_gen_local')
                if img_gen and hasattr(img_gen, 'generate'):
                    result = img_gen.generate(prompt, width=width, height=height)
                    return {
                        "success": True,
                        "result": f"Image generated: {prompt[:50]}...",
                        "message": f"Generated image from prompt: {prompt}"
                    }
            except Exception as e:
                logger.error(f"Error using image_gen_local: {e}")
        
        # Fallback: return simulated result
        return {
            "success": True,
            "result": f"[Image: {prompt}]",
            "message": f"Image generation requested for: {prompt}"
        }
    
    def _avatar_action(self, action: str = None, params: Any = None, **kwargs) -> Dict[str, Any]:
        """Control the avatar."""
        if action is None:
            action = kwargs.get('arg0', 'speak')
        if params is None:
            params = kwargs.get('arg1', {})
        
        if self.manager:
            try:
                # Try to access avatar controller
                from ..avatar import controller
                if hasattr(controller, 'execute_action'):
                    controller.execute_action(action, params)
                    return {
                        "success": True,
                        "message": f"Avatar action '{action}' executed"
                    }
            except Exception as e:
                logger.debug(f"Avatar controller not available: {e}")
        
        return {
            "success": True,
            "message": f"Avatar action '{action}' requested"
        }
    
    def _capture_screen(self, **kwargs) -> Dict[str, Any]:
        """Capture and analyze screen."""
        try:
            from ..tools.system_tools import ScreenshotTool
            tool = ScreenshotTool()
            result = tool.execute()
            if result.get('success'):
                return {
                    "success": True,
                    "message": "Screen captured",
                    "result": result.get('result', 'Screenshot taken')
                }
        except Exception as e:
            logger.debug(f"Screenshot tool not available: {e}")
        
        return {
            "success": True,
            "message": "Screen capture requested"
        }
    
    def _speak(self, text: str = None, **kwargs) -> Dict[str, Any]:
        """Speak text using TTS."""
        if text is None:
            text = kwargs.get('arg0', kwargs.get('prompt', ''))
        
        if not text:
            return {"success": False, "error": "No text provided"}
        
        try:
            from ..voice.tts_simple import speak
            speak(text)
            return {
                "success": True,
                "message": f"Speaking: {text[:50]}..."
            }
        except Exception as e:
            logger.debug(f"TTS not available: {e}")
            return {
                "success": True,
                "message": f"Speech requested: {text[:50]}..."
            }
    
    def _search_web(self, query: str = None, **kwargs) -> Dict[str, Any]:
        """Search the web."""
        if query is None:
            query = kwargs.get('arg0', kwargs.get('prompt', kwargs.get('text', '')))
        
        if not query:
            return {"success": False, "error": "No query provided"}
        
        try:
            from ..tools.web_tools import WebSearchTool
            tool = WebSearchTool()
            result = tool.execute(query=query)
            return result
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _read_file(self, path: str = None, **kwargs) -> Dict[str, Any]:
        """Read file contents."""
        if path is None:
            path = kwargs.get('arg0', '')
        
        if not path:
            return {"success": False, "error": "No path provided"}
        
        try:
            from ..tools.file_tools import ReadFileTool
            tool = ReadFileTool()
            result = tool.execute(path=path)
            return result
        except Exception as e:
            logger.error(f"Read file error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _write_file(self, path: str = None, content: str = None, **kwargs) -> Dict[str, Any]:
        """Write to file."""
        if path is None:
            path = kwargs.get('arg0', kwargs.get('path', ''))
        if content is None:
            content = kwargs.get('arg1', kwargs.get('text', kwargs.get('content', '')))
        
        if not path or not content:
            return {"success": False, "error": "Path and content required"}
        
        try:
            from ..tools.file_tools import WriteFileTool
            tool = WriteFileTool()
            result = tool.execute(path=path, content=content)
            return result
        except Exception as e:
            logger.error(f"Write file error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _list_directory(self, path: str = None, **kwargs) -> Dict[str, Any]:
        """List directory contents."""
        if path is None:
            path = kwargs.get('arg0', kwargs.get('path', '.'))
        
        try:
            from ..tools.file_tools import ListDirectoryTool
            tool = ListDirectoryTool()
            result = tool.execute(path=path)
            return result
        except Exception as e:
            logger.error(f"List directory error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_tools_list(self) -> List[Dict[str, str]]:
        """Get list of available tools with descriptions."""
        return [
            {"name": name, "description": desc}
            for name, desc in self.tool_descriptions.items()
        ]
    
    def get_tool_names(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.available_tools.keys())


# =============================================================================
# Convenience Functions
# =============================================================================

def create_tool_interface(module_manager=None) -> ToolInterface:
    """
    Create a tool interface instance.
    
    Args:
        module_manager: Optional ModuleManager instance
        
    Returns:
        ToolInterface instance
    """
    return ToolInterface(module_manager)


def parse_and_execute_tool(ai_output: str, module_manager=None) -> Optional[str]:
    """
    Convenience function to parse and execute a tool call in one step.
    
    Args:
        ai_output: AI output containing tool call
        module_manager: Optional ModuleManager instance
        
    Returns:
        Formatted tool result string, or None if no tool call found
    """
    interface = ToolInterface(module_manager)
    tool_call = interface.parse_tool_call(ai_output)
    
    if not tool_call:
        return None
    
    result = interface.execute_tool(tool_call)
    return interface.format_tool_result(result)


__all__ = [
    'ToolCall',
    'ToolResult',
    'ToolInterface',
    'create_tool_interface',
    'parse_and_execute_tool',
]
