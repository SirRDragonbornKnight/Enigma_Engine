"""
Tool Executor for Enigma AI
============================

Executes tool calls from the AI by:
1. Parsing tool call JSON from AI output
2. Validating parameters
3. Executing the appropriate tool/module
4. Returning structured results
5. Handling errors gracefully
"""

import json
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from .tool_definitions import get_tool_definition, TOOLS_BY_NAME

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Executes tool calls from the AI.
    
    Parses tool call format:
        <tool_call>
        {"tool": "tool_name", "params": {"param1": "value1"}}
        </tool_call>
    
    Returns results in format:
        <tool_result>
        {"tool": "tool_name", "success": true/false, "result": "..."}
        </tool_result>
    """
    
    def __init__(self, module_manager=None):
        """
        Initialize tool executor.
        
        Args:
            module_manager: Optional ModuleManager instance for executing module-based tools
        """
        self.module_manager = module_manager
        self._tool_registry = None
    
    def _get_tool_registry(self):
        """Lazy load tool registry to avoid circular imports."""
        if self._tool_registry is None:
            try:
                from .tool_registry import get_registry
                self._tool_registry = get_registry()
            except ImportError as e:
                logger.warning(f"Could not load tool registry: {e}")
        return self._tool_registry
    
    def parse_tool_calls(self, text: str) -> List[Tuple[str, Dict[str, Any], int, int]]:
        """
        Parse tool calls from AI output.
        
        Args:
            text: AI output text containing tool calls
            
        Returns:
            List of (tool_name, params, start_pos, end_pos) tuples
        """
        tool_calls = []
        
        # Pattern to match <tool_call>...</tool_call>
        pattern = r'<tool_call>(.*?)</tool_call>'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            json_str = match.group(1).strip()
            start_pos = match.start()
            end_pos = match.end()
            
            try:
                # Parse JSON
                data = json.loads(json_str)
                
                if not isinstance(data, dict):
                    logger.warning(f"Tool call is not a dict: {json_str}")
                    continue
                
                tool_name = data.get("tool")
                params = data.get("params", {})
                
                if not tool_name:
                    logger.warning(f"Tool call missing 'tool' field: {json_str}")
                    continue
                
                if not isinstance(params, dict):
                    logger.warning(f"Tool params is not a dict: {params}")
                    params = {}
                
                tool_calls.append((tool_name, params, start_pos, end_pos))
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON: {json_str} - {e}")
                continue
        
        return tool_calls
    
    def validate_params(
        self, 
        tool_name: str, 
        params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validate tool parameters against schema.
        
        Args:
            tool_name: Name of the tool
            params: Parameters provided
            
        Returns:
            (is_valid, error_message, validated_params)
        """
        tool_def = get_tool_definition(tool_name)
        
        if not tool_def:
            return False, f"Unknown tool: {tool_name}", params
        
        validated = {}
        
        # Check required parameters
        for param_def in tool_def.parameters:
            param_name = param_def.name
            
            if param_name in params:
                value = params[param_name]
                
                # Type validation (basic)
                expected_type = param_def.type
                if expected_type == "int" and not isinstance(value, int):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        return False, f"Parameter '{param_name}' must be an integer", params
                
                elif expected_type == "float" and not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        return False, f"Parameter '{param_name}' must be a number", params
                
                elif expected_type == "bool" and not isinstance(value, bool):
                    # Try to convert
                    if isinstance(value, str):
                        value = value.lower() in ('true', 'yes', '1', 'on')
                    else:
                        value = bool(value)
                
                elif expected_type == "string" and not isinstance(value, str):
                    value = str(value)
                
                # Enum validation
                if param_def.enum and value not in param_def.enum:
                    return False, f"Parameter '{param_name}' must be one of {param_def.enum}", params
                
                validated[param_name] = value
            
            elif param_def.required:
                return False, f"Missing required parameter: {param_name}", params
            
            elif param_def.default is not None:
                validated[param_name] = param_def.default
        
        return True, None, validated
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            
        Returns:
            Result dictionary with 'success', 'result', and optionally 'error'
        """
        # Validate parameters
        is_valid, error_msg, validated_params = self.validate_params(tool_name, params)
        
        if not is_valid:
            return {
                "success": False,
                "error": error_msg,
                "tool": tool_name,
            }
        
        tool_def = get_tool_definition(tool_name)
        
        if not tool_def:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "tool": tool_name,
            }
        
        try:
            # Check if tool requires a module
            if tool_def.module:
                return self._execute_module_tool(tool_name, tool_def.module, validated_params)
            else:
                return self._execute_builtin_tool(tool_name, validated_params)
        
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
            }
    
    def _execute_module_tool(
        self, 
        tool_name: str, 
        module_name: str, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool that requires a module."""
        if not self.module_manager:
            return {
                "success": False,
                "error": f"Module manager not available. Cannot execute {tool_name}",
                "tool": tool_name,
            }
        
        # Check if module is loaded
        if not self.module_manager.is_loaded(module_name):
            return {
                "success": False,
                "error": f"Module '{module_name}' is not loaded. Load it first to use {tool_name}",
                "tool": tool_name,
            }
        
        # Get module instance
        module = self.module_manager.get_module(module_name)
        
        if not module:
            return {
                "success": False,
                "error": f"Could not get module instance for '{module_name}'",
                "tool": tool_name,
            }
        
        # Execute based on tool type
        if tool_name == "generate_image":
            return self._execute_generate_image(module, params)
        elif tool_name == "analyze_image":
            return self._execute_analyze_image(module, params)
        elif tool_name == "control_avatar":
            return self._execute_control_avatar(module, params)
        elif tool_name == "speak":
            return self._execute_speak(module, params)
        elif tool_name == "generate_code":
            return self._execute_generate_code(module, params)
        elif tool_name == "generate_video":
            return self._execute_generate_video(module, params)
        elif tool_name == "generate_audio":
            return self._execute_generate_audio(module, params)
        else:
            return {
                "success": False,
                "error": f"Tool execution not implemented for {tool_name}",
                "tool": tool_name,
            }
    
    def _execute_builtin_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a built-in tool (doesn't require module)."""
        registry = self._get_tool_registry()
        
        if not registry:
            return {
                "success": False,
                "error": "Tool registry not available",
                "tool": tool_name,
            }
        
        # Execute through tool registry
        try:
            result = registry.execute(tool_name, **params)
            
            # Standardize result format
            if isinstance(result, dict):
                if "success" not in result:
                    result["success"] = True
                result["tool"] = tool_name
                return result
            else:
                return {
                    "success": True,
                    "result": result,
                    "tool": tool_name,
                }
        
        except Exception as e:
            logger.exception(f"Error executing builtin tool {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
            }
    
    # Tool-specific execution methods
    
    def _execute_generate_image(self, module, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute image generation."""
        try:
            prompt = params.get("prompt", "")
            width = params.get("width", 512)
            height = params.get("height", 512)
            steps = params.get("steps", 20)
            
            # Call module's generate method
            if hasattr(module, "generate"):
                result = module.generate(prompt=prompt, width=width, height=height, steps=steps)
            elif hasattr(module, "_instance") and hasattr(module._instance, "generate"):
                result = module._instance.generate(prompt=prompt, width=width, height=height, steps=steps)
            else:
                return {
                    "success": False,
                    "error": "Image generation module does not have generate() method",
                    "tool": "generate_image",
                }
            
            return {
                "success": True,
                "result": f"Image generated successfully: {result}",
                "tool": "generate_image",
                "output_path": str(result) if result else None,
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": "generate_image",
            }
    
    def _execute_analyze_image(self, module, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute image analysis."""
        try:
            image_path = params.get("image_path", "")
            detail_level = params.get("detail_level", "normal")
            
            # Call module's analyze method
            if hasattr(module, "analyze"):
                result = module.analyze(image_path=image_path, detail=detail_level)
            elif hasattr(module, "_instance") and hasattr(module._instance, "analyze"):
                result = module._instance.analyze(image_path=image_path, detail=detail_level)
            else:
                return {
                    "success": False,
                    "error": "Vision module does not have analyze() method",
                    "tool": "analyze_image",
                }
            
            return {
                "success": True,
                "result": str(result),
                "tool": "analyze_image",
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": "analyze_image",
            }
    
    def _execute_control_avatar(self, module, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute avatar control."""
        try:
            action = params.get("action", "")
            value = params.get("value", "")
            
            # Call module's control method
            if hasattr(module, "control"):
                result = module.control(action=action, value=value)
            elif hasattr(module, "_instance") and hasattr(module._instance, "control"):
                result = module._instance.control(action=action, value=value)
            else:
                return {
                    "success": False,
                    "error": "Avatar module does not have control() method",
                    "tool": "control_avatar",
                }
            
            return {
                "success": True,
                "result": f"Avatar {action} set to {value}",
                "tool": "control_avatar",
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": "control_avatar",
            }
    
    def _execute_speak(self, module, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text-to-speech."""
        try:
            text = params.get("text", "")
            voice = params.get("voice", "default")
            
            # Call module's speak method
            if hasattr(module, "speak"):
                result = module.speak(text=text, voice=voice)
            elif hasattr(module, "_instance") and hasattr(module._instance, "speak"):
                result = module._instance.speak(text=text, voice=voice)
            else:
                return {
                    "success": False,
                    "error": "Voice module does not have speak() method",
                    "tool": "speak",
                }
            
            return {
                "success": True,
                "result": f"Spoke: {text[:50]}...",
                "tool": "speak",
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": "speak",
            }
    
    def _execute_generate_code(self, module, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code generation."""
        try:
            description = params.get("description", "")
            language = params.get("language", "python")
            
            # Call module's generate method
            if hasattr(module, "generate"):
                result = module.generate(description=description, language=language)
            elif hasattr(module, "_instance") and hasattr(module._instance, "generate"):
                result = module._instance.generate(description=description, language=language)
            else:
                return {
                    "success": False,
                    "error": "Code generation module does not have generate() method",
                    "tool": "generate_code",
                }
            
            return {
                "success": True,
                "result": str(result),
                "tool": "generate_code",
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": "generate_code",
            }
    
    def _execute_generate_video(self, module, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute video generation."""
        try:
            prompt = params.get("prompt", "")
            duration = params.get("duration", 3.0)
            fps = params.get("fps", 24)
            
            # Call module's generate method
            if hasattr(module, "generate"):
                result = module.generate(prompt=prompt, duration=duration, fps=fps)
            elif hasattr(module, "_instance") and hasattr(module._instance, "generate"):
                result = module._instance.generate(prompt=prompt, duration=duration, fps=fps)
            else:
                return {
                    "success": False,
                    "error": "Video generation module does not have generate() method",
                    "tool": "generate_video",
                }
            
            return {
                "success": True,
                "result": f"Video generated successfully: {result}",
                "tool": "generate_video",
                "output_path": str(result) if result else None,
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": "generate_video",
            }
    
    def _execute_generate_audio(self, module, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute audio generation."""
        try:
            prompt = params.get("prompt", "")
            duration = params.get("duration", 5.0)
            
            # Call module's generate method
            if hasattr(module, "generate"):
                result = module.generate(prompt=prompt, duration=duration)
            elif hasattr(module, "_instance") and hasattr(module._instance, "generate"):
                result = module._instance.generate(prompt=prompt, duration=duration)
            else:
                return {
                    "success": False,
                    "error": "Audio generation module does not have generate() method",
                    "tool": "generate_audio",
                }
            
            return {
                "success": True,
                "result": f"Audio generated successfully: {result}",
                "tool": "generate_audio",
                "output_path": str(result) if result else None,
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": "generate_audio",
            }
    
    def format_tool_result(self, result: Dict[str, Any]) -> str:
        """
        Format tool result as text for injection back into AI context.
        
        Args:
            result: Result dictionary from tool execution
            
        Returns:
            Formatted string with <tool_result> tags
        """
        # Create clean result dict
        clean_result = {
            "tool": result.get("tool", "unknown"),
            "success": result.get("success", False),
        }
        
        if result.get("success"):
            clean_result["result"] = result.get("result", "")
            if "output_path" in result:
                clean_result["output_path"] = result["output_path"]
        else:
            clean_result["error"] = result.get("error", "Unknown error")
        
        # Format as JSON inside tags
        json_str = json.dumps(clean_result, indent=2)
        return f"<tool_result>\n{json_str}\n</tool_result>"


def execute_tool_from_text(
    text: str,
    module_manager=None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse and execute all tool calls from text, returning modified text with results.
    
    Args:
        text: Text containing tool calls
        module_manager: Optional ModuleManager for module-based tools
        
    Returns:
        (modified_text, list_of_results)
    """
    executor = ToolExecutor(module_manager=module_manager)
    
    # Parse tool calls
    tool_calls = executor.parse_tool_calls(text)
    
    if not tool_calls:
        return text, []
    
    results = []
    modified_text = text
    
    # Execute tools (in reverse order to preserve positions)
    for tool_name, params, start_pos, end_pos in reversed(tool_calls):
        # Execute
        result = executor.execute_tool(tool_name, params)
        results.insert(0, result)
        
        # Format result
        result_str = executor.format_tool_result(result)
        
        # Replace tool call with result
        modified_text = (
            modified_text[:start_pos] + 
            result_str + 
            modified_text[end_pos:]
        )
    
    return modified_text, results


__all__ = [
    "ToolExecutor",
    "execute_tool_from_text",
]
